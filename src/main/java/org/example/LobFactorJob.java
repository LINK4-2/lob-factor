package org.example;

import org.apache.hadoop.conf.Configuration;
import org.apache.hadoop.fs.Path;
import org.apache.hadoop.io.LongWritable;
import org.apache.hadoop.io.Text;
import org.apache.hadoop.mapreduce.Job;
import org.apache.hadoop.mapreduce.Mapper;
import org.apache.hadoop.mapreduce.Reducer;
import org.apache.hadoop.mapreduce.TaskCounter;
import org.apache.hadoop.mapreduce.lib.input.FileInputFormat;
import org.apache.hadoop.mapreduce.lib.output.FileOutputFormat;

import org.apache.hadoop.mapreduce.lib.input.CombineTextInputFormat;

import java.io.IOException;
import java.util.HashMap;
import java.util.Map;

/**
 * 单 Job 完成：
 * 1. 预处理：从原始 LOB 行情快照中只读取需要的列；
 * 2. 按“股票 × 时间”计算 20 个因子；
 * 3. 在 300 只股票上对 20 个因子做横截面平均。
 *
 * 输入：
 *   - 一个目录或文件，里面是 300 只股票的一天快照文件（CSV），每个文件第一行为表头：
 *     tradingDay,tradeTime,recvTime,MIC,code,cumCnt,cumVol,turnover,last,open,high,low,
 *     tBidVol,tAskVol,wBidPrc,wAskPrc,openInterest,
 *     bp1,bv1,ap1,av1,...,bp10,bv10,ap10,av10
 *
 * 输出：
 *   - text 文件，第一行：tradeTime,alpha_1,...,alpha_20
 *   - 之后每行：某个时间点的 20 个平均因子值。
 */
public class LobFactorJob {

    private static final int N = 5;           // 只用前 N 档
    private static final double EPS = 1e-7;   // 分母为 0 时加 epsilon

    /**
     * 安全除法：防止分母为 0 导致 NaN/Infinity。
     */
    private static double safeDiv(double num, double den) {
        return num / (den + EPS);
    }

    /**
     * 保存“上一时刻”的关键信息，用于 17/18/19 因子
     */
    private static class PrevState {
        double ap1;
        double bp1;
        double mid;
        double depthRatio;

        PrevState(double ap1, double bp1, double mid, double depthRatio) {
            this.ap1 = ap1;
            this.bp1 = bp1;
            this.mid = mid;
            this.depthRatio = depthRatio;
        }
    }

    // ======================  Mapper  ======================
    // ✅ 优化目标：减少 Map -> Reduce 的中间输出数量（MAP_OUTPUT_RECORDS）
    // 原始版本：每读取一行快照就 context.write 一条  (≈ 144 万条)
    // 优化版本：在 Mapper 内部先按 tradeTime 聚合(sum+count)，最后在 cleanup() 统一输出
    //          => MAP_OUTPUT_RECORDS 从百万级降到“时间点数量”（你实测 4802）
    //
    // ✅ 为什么有效：
    // 1) 横截面平均只需要每个 tradeTime 的“总和 + 个数”
    // 2) 不需要把每只股票每一行的 20 因子都发到 reducer
    public static class FactorMapper extends Mapper<LongWritable, Text, Text, Text> {

        // 记录每支股票上一时刻状态（用于 17/18/19）
        private final Map<String, PrevState> prevMap = new HashMap<>();

        // ✅ 核心优化：tradeTime -> 聚合器 Agg(sum[20], count)
        // 原来是：每行直接输出 20 个因子字符串
        // 现在是：同一个 tradeTime 的多行先累加到 Agg 里，避免大量 context.write
        private final Map<String, Agg> aggMap = new HashMap<>(8192);

        // Text 对象复用，避免频繁 new Text 造成 GC 压力
        private final Text outKey = new Text();
        private final Text outVal = new Text();

        // factor 数组复用：每行都写入同一个 double[20]，避免每行 new double[20]
        // 注意：因为我们在 Agg.add() 中“把数值加到 sum 里”，不会引用 factor 数组本身，
        //       所以复用是安全的。
        private final double[] factor = new double[20];

        // ✅ 聚合器：保存同一 tradeTime 下的 20 因子总和 + 样本数
        private static class Agg {
            final double[] sum = new double[20];
            long count = 0;

            //  每读一行快照，只把 20 个因子“加到 sum”，并 count++
            void add(double[] f) {
                for (int i = 0; i < 20; i++) sum[i] += f[i];
                count++;
            }

            // ✅ 输出给 reducer 的 value：sum0,sum1,...,sum19,count
            // Reducer 再把不同 mapper 的 sum/count 合并即可，不再需要每行的因子明细
            String toCsvSumAndCount() {
                // 输出：sum0,sum1,...,sum19,count
                StringBuilder sb = new StringBuilder(20 * 12);
                for (int i = 0; i < 20; i++) {
                    if (i > 0) sb.append(',');
                    sb.append(sum[i]);
                }
                sb.append(',').append(count);
                return sb.toString();
            }
        }

        @Override
        protected void map(LongWritable key, Text value, Context context)
                throws IOException, InterruptedException {

            String line = value.toString();
            if (line == null) return;
            line = line.trim();
            if (line.isEmpty()) return;

            // 跳过表头
            if (line.startsWith("tradingDay")) return;

            // 仍用 split（优化重点是“减少输出条数”；如果你还想更快，下一步再换手写解析）
            String[] f = line.split(",");
            if (f.length < 57) return;

            String tradeTimeStr = f[1];
            String code = f[4];

            double tBidVol = Double.parseDouble(f[12]);
            double tAskVol = Double.parseDouble(f[13]);

            int idx = 17;
            double[] bp = new double[N];
            double[] bv = new double[N];
            double[] ap = new double[N];
            double[] av = new double[N];
            for (int i = 0; i < N; i++) {
                bp[i] = Double.parseDouble(f[idx++]);
                bv[i] = Double.parseDouble(f[idx++]);
                ap[i] = Double.parseDouble(f[idx++]);
                av[i] = Double.parseDouble(f[idx++]);
            }

            // tradeTime 处理（假设已是 hhmmss；若你的数据带后缀，需要你自己恢复砍位逻辑）
            int hms = (int) Long.parseLong(tradeTimeStr);

            // 仅输出 09:30~15:00；早盘用于初始化 prev
            boolean inOutputWindow = (hms >= 93000 && hms <= 150000);

            double ap1 = ap[0];
            double bp1 = bp[0];
            double bv1 = bv[0];
            double av1 = av[0];

            double sumBv = 0, sumAv = 0;
            double sumBpBv = 0, sumApAv = 0;
            double sumBvOverI = 0, sumAvOverI = 0;

            for (int i = 0; i < N; i++) {
                int level = i + 1;
                double bv_i = bv[i], av_i = av[i];
                double bp_i = bp[i], ap_i = ap[i];

                sumBv += bv_i;
                sumAv += av_i;
                sumBpBv += bp_i * bv_i;
                sumApAv += ap_i * av_i;
                sumBvOverI += bv_i / level;
                sumAvOverI += av_i / level;
            }

            double depthSum = sumBv + sumAv;
            double depthDiff = sumBv - sumAv;
            double depthRatio = safeDiv(sumBv, sumAv);
            double spread = ap1 - bp1;
            double mid = (ap1 + bp1) / 2.0;

            // ---- 1~16（当前时刻）----
            factor[0]  = spread;
            factor[1]  = safeDiv(spread, mid);
            factor[2]  = mid;
            factor[3]  = safeDiv(bv1 - av1, bv1 + av1);
            factor[4]  = safeDiv(sumBv - sumAv, depthSum);
            factor[5]  = sumBv;
            factor[6]  = sumAv;
            factor[7]  = depthDiff;
            factor[8]  = depthRatio;
            factor[9]  = safeDiv(tBidVol - tAskVol, tBidVol + tAskVol);
            factor[10] = safeDiv(sumBpBv, sumBv);
            factor[11] = safeDiv(sumApAv, sumAv);
            factor[12] = safeDiv(sumBpBv + sumApAv, depthSum);
            factor[13] = factor[11] - factor[10];
            factor[14] = (sumBv / N) - (sumAv / N);
            factor[15] = safeDiv(sumBvOverI - sumAvOverI, sumBvOverI + sumAvOverI);

            // ---- 17~19（上一时刻）----
            PrevState ps = prevMap.get(code);
            double prevAp1, prevMid, prevDepthRatio;
            if (ps == null) {
                prevAp1 = ap1;
                prevMid = mid;
                prevDepthRatio = depthRatio;
            } else {
                prevAp1 = ps.ap1;
                prevMid = ps.mid;
                prevDepthRatio = ps.depthRatio;
            }

            factor[16] = ap1 - prevAp1;
            factor[17] = mid - prevMid;
            factor[18] = depthRatio - prevDepthRatio;

            // 20
            factor[19] = safeDiv(spread, depthSum);

            // 更新 prev 状态
            prevMap.put(code, new PrevState(ap1, bp1, mid, depthRatio));

            // 09:30~15:00 之外不输出
            if (!inOutputWindow) return;

            // 生成固定 6 位 tradeTime key（避免 String.format 的开销）
            String timeKey = pad6(hms);

            // ✅ 核心优化：in-mapper aggregation
            // 不再 context.write(timeKey, factorCsv)
            // 而是把因子加到 aggMap 里
            Agg agg = aggMap.get(timeKey);
            if (agg == null) {
                agg = new Agg();
                aggMap.put(timeKey, agg);
            }
            agg.add(factor);

            // ✅（调试用）统计窗口内处理了多少行
            // 性能测试时建议关掉（每行 increment 会有开销）
            context.getCounter("DBG", "IN_WINDOW_LINES").increment(1);

        }

        @Override
        protected void cleanup(Context context) throws IOException, InterruptedException {
            // ✅ cleanup() 在一个 Mapper 结束时调用一次
            // 这里一次性把“每个 tradeTime 的 sum/count”输出，输出条数≈时间点数（几千）
            // 这就是 MAP_OUTPUT_RECORDS 大幅下降的原因

            // ✅（调试用）输出的 tradeTime key 数量（也就是 aggMap.size()）
            context.getCounter("DBG", "EMITTED_TIME_KEYS").increment(aggMap.size());
            for (Map.Entry<String, Agg> e : aggMap.entrySet()) {
                outKey.set(e.getKey());
                outVal.set(e.getValue().toCsvSumAndCount());
                context.write(outKey, outVal);
            }
            // 释放引用，帮助 GC（local 模式下也能略减内存压力）
            aggMap.clear();
            prevMap.clear();
        }

        private static String pad6(int hms) {
            // hms 形如 93000 -> "093000"
            String s = Integer.toString(hms);
            int n = s.length();
            if (n >= 6) return s;
            // 手动补 0，比 String.format 快
            StringBuilder sb = new StringBuilder(6);
            for (int i = 0; i < 6 - n; i++) sb.append('0');
            sb.append(s);
            return sb.toString();
        }
    }


    // ======================  Reducer  ======================
    // ✅ Reducer 输入从“每行 20 因子”变成 “每个 mapper 在该时间点的 sum+count”
    // 原始版本：values 里每个元素都是 20 个因子（字符串） -> reducer 需要 parse 144 万次
    // 现在版本：values 里每个元素都是 sum[20] + count -> reducer 只需要处理约 4802 行 key
    //
    public static class AverageReducer extends Reducer<Text, Text, Text, Text> {

        private boolean headerWritten = false;
        private final Text outVal = new Text();

        @Override
        protected void reduce(Text key, Iterable<Text> values, Context context)
                throws IOException, InterruptedException {

            if (!headerWritten) {
                StringBuilder header = new StringBuilder();
                header.append("alpha_1");
                for (int i = 2; i <= 20; i++) header.append(',').append("alpha_").append(i);
                context.write(new Text("tradeTime"), new Text(header.toString()));
                headerWritten = true;
            }

            double[] totalSum = new double[20];
            long totalCount = 0;

            // 每个 value 是：sum0,sum1,...,sum19,count
            for (Text v : values) {
                String[] parts = v.toString().split(",");
                if (parts.length < 21) continue;

                for (int i = 0; i < 20; i++) {
                    totalSum[i] += Double.parseDouble(parts[i]);
                }
                totalCount += Long.parseLong(parts[20]);
            }

            if (totalCount == 0) return;

            //  最后除以总样本数得到横截面均值
            for (int i = 0; i < 20; i++) totalSum[i] /= totalCount;

            outVal.set(factorsToCsv(totalSum));
            context.write(key, outVal);
        }

        private static String factorsToCsv(double[] f) {
            StringBuilder sb = new StringBuilder();
            for (int i = 0; i < f.length; i++) {
                if (i > 0) sb.append(',');
                sb.append(f[i]);
            }
            return sb.toString();
        }
    }


    // ======================  Driver  ======================
    //
    // ✅ 优化点 1：CombineTextInputFormat 合并小文件
    // 你的输入是 300 个股票文件（小文件很多），默认 TextInputFormat 往往是“一文件一 map”
    // 那样每个 mapper 里同一 tradeTime 很难重复出现，in-mapper aggregation 聚合不起来
    //
    // 使用 CombineTextInputFormat 后：一个 map task 可以读取多个文件
    // => 同一 mapper 内会见到很多股票的同一 tradeTime
    // => aggMap 的 key 数量≈时间点数（几千），而不是行数（百万）
    //
    // ✅ 优化点 2：把 split size 设大，让本地模式下 map task 更少
    // local 模式下 map task 太多会有调度/启动开销，合并成更大的 split 往往更快

    public static void main(String[] args) throws Exception {
        if (args.length != 2) {
            System.err.println("Usage: LobFactorSingleJob <oneDayInputDir> <outputDir>");
            System.exit(1);
        }

        Configuration conf = new Configuration();

        // 让 key,value 用逗号分隔，确保输出是 CSV（逗号分隔）
        conf.set("mapreduce.output.textoutputformat.separator", ",");

        Job job = Job.getInstance(conf, "lob-factor-single-job");

        // ✅ 小文件合并输入格式：让一个 mapper 读多个股票文件
        job.setInputFormatClass(CombineTextInputFormat.class);
        // ✅ 控制合并后的 split 最大大小（local 模式可设置大一些减少 map 个数）
        CombineTextInputFormat.setMaxInputSplitSize(job,  1024L * 1024 * 1024);

        job.setJarByClass(LobFactorJob.class);
        job.setMapperClass(FactorMapper.class);
        job.setReducerClass(AverageReducer.class);

        // 单 Job，单 Reducer（这样可以方便地写一行表头），符合“不能 jobchain”的要求
        // ⚠️ 性能上会让 reducer 成为瓶颈（但题目规模可能还可接受）
        job.setNumReduceTasks(1);

        job.setMapOutputKeyClass(Text.class);
        job.setMapOutputValueClass(Text.class);
        job.setOutputKeyClass(Text.class);
        job.setOutputValueClass(Text.class);

        org.apache.hadoop.mapreduce.lib.input.FileInputFormat.setInputDirRecursive(job, true);

        // args[0] 传入的是某一天（比如 /data/0102），里面再去递归找 300 个 snapshot.csv
        FileInputFormat.addInputPath(job, new Path(args[0]));
        FileOutputFormat.setOutputPath(job, new Path(args[1]));

        // ===================== 计时开始（Wall-clock）=====================
        final long t0 = System.nanoTime();

        boolean ok = job.waitForCompletion(true);

        org.apache.hadoop.mapreduce.Counters c = job.getCounters();
        System.out.println("MAP_INPUT_RECORDS=" + c.findCounter(org.apache.hadoop.mapreduce.TaskCounter.MAP_INPUT_RECORDS).getValue());
        System.out.println("MAP_OUTPUT_RECORDS=" + c.findCounter(org.apache.hadoop.mapreduce.TaskCounter.MAP_OUTPUT_RECORDS).getValue());
        System.out.println("REDUCE_INPUT_RECORDS=" + c.findCounter(org.apache.hadoop.mapreduce.TaskCounter.REDUCE_INPUT_RECORDS).getValue());
        System.out.println("SPILLED_RECORDS=" + c.findCounter(org.apache.hadoop.mapreduce.TaskCounter.SPILLED_RECORDS).getValue());

        System.out.println("DBG_IN_WINDOW_LINES=" + c.findCounter("DBG","IN_WINDOW_LINES").getValue());
        System.out.println("DBG_EMITTED_TIME_KEYS=" + c.findCounter("DBG","EMITTED_TIME_KEYS").getValue());


        final long t1 = System.nanoTime();
        double elapsedSec = (t1 - t0) / 1_000_000_000.0;

        // 把计时结果打印出来（你可以在终端重定向保存）
        System.out.printf(
                "JOB_ELAPSED_SECONDS=%.3f | input=%s | output=%s | jobId=%s%n",
                elapsedSec, args[0], args[1], job.getJobID()
        );

        // =====================（可选）额外统计：CPU/GC（聚合 counters）=====================
        try {
            long cpuMs = job.getCounters().findCounter(TaskCounter.CPU_MILLISECONDS).getValue();
            long gcMs  = job.getCounters().findCounter(TaskCounter.GC_TIME_MILLIS).getValue();
            System.out.printf("TASK_CPU_MILLISECONDS=%d | TASK_GC_TIME_MILLIS=%d%n", cpuMs, gcMs);
        } catch (Exception ignored) {
            // 某些环境/权限下 counters 可能不可用，忽略不影响主流程
        }

        System.exit(ok ? 0 : 1);
    }
}
