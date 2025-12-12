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

        // ✅ 性能测试时建议关闭（每行 Counter.increment 会拖慢）。把它
        private static final boolean DEBUG = true;

        // ✅ code 用 int，避免 substring 得到新 String（同时也更省内存）
        private final Map<Integer, PrevState> prevMap = new HashMap<>(512);

        // ✅ tradeTime -> 聚合(sum[20], count)
        private final Map<String, Agg> aggMap = new HashMap<>(8192);

        private final Text outKey = new Text();
        private final Text outVal = new Text();

        // ✅ 复用 factor 数组：每行填充一次，Agg.add 时把数值加到 sum 里即可
        private final double[] factor = new double[20];

        private static class Agg {
            final double[] sum = new double[20];
            long count = 0;

            void add(double[] f) {
                // ✅ 只做 20 次加法，不产生新对象
                for (int i = 0; i < 20; i++) sum[i] += f[i];
                count++;
            }

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
            if (line == null || line.isEmpty()) return;

            // ✅ 比 startsWith("tradingDay") 更快：表头第一列一定是 't'
            if (line.charAt(0) == 't') return;

            // ===========================
            // ✅ 核心优化 1：不用 split(",")
            //  - String.split 是正则，会创建大量 String 对象，CPU + GC 都贵
            //  - 我们只需要有限的列，所以用扫描逗号的方式只解析需要的字段
            // ===========================

            int hms = 0;          // col 1
            int code = 0;         // col 4
            double tBidVol = 0;   // col 12
            double tAskVol = 0;   // col 13

            // 下面这些是从 bp1..av5 直接累计出来的（不再 new 数组）
            double bp1 = 0, bv1 = 0, ap1 = 0, av1 = 0;
            double sumBv = 0, sumAv = 0;
            double sumBpBv = 0, sumApAv = 0;
            double sumBvOverI = 0, sumAvOverI = 0;

            // 当前档位的临时 bp / ap（因为列顺序是 bp,bv,ap,av）
            double curBp = 0, curAp = 0;

            int col = 0;
            int start = 0;
            final int len = line.length();

            // 扫描到 bp1..av5 就够了：列 17~36
            for (int i = 0; i <= len; i++) {
                if (i == len || line.charAt(i) == ',') {
                    int end = i;

                    if (col == 1) {
                        // tradeTime（6 位 hhmmss）
                        hms = parseIntFast(line, start, end);
                    } else if (col == 4) {
                        // code（你数据里是数字：1 / 2 / ...）
                        code = parseIntFast(line, start, end);
                    } else if (col == 12) {
                        tBidVol = parseDoubleFast(line, start, end);
                    } else if (col == 13) {
                        tAskVol = parseDoubleFast(line, start, end);
                    } else if (col >= 17 && col <= 36) {
                        // ✅ 只解析前 5 档（bp1,bv1,ap1,av1,...,bp5,bv5,ap5,av5）
                        int k = col - 17;       // 0..19
                        int level = k / 4 + 1;  // 1..5
                        int pos = k & 3;        // 0 bp, 1 bv, 2 ap, 3 av

                        double v = parseDoubleFast(line, start, end);

                        if (pos == 0) { // bp
                            curBp = v;
                            if (level == 1) bp1 = v;
                        } else if (pos == 1) { // bv
                            double bv = v;
                            if (level == 1) bv1 = bv;
                            sumBv += bv;
                            sumBpBv += curBp * bv;
                            sumBvOverI += bv / level;
                        } else if (pos == 2) { // ap
                            curAp = v;
                            if (level == 1) ap1 = v;
                        } else { // av
                            double av = v;
                            if (level == 1) av1 = av;
                            sumAv += av;
                            sumApAv += curAp * av;
                            sumAvOverI += av / level;
                        }
                    }

                    col++;
                    start = i + 1;

                    // ✅ 解析到 av5 就停止（col==37 表示已经处理完 0..36）
                    if (col > 36) break;
                }
            }

            // 如果列不够，直接丢弃
            if (col <= 36) return;

            // 输出窗口：9:30:00 ~ 15:00:00（09:25 用于初始化 prev）
            boolean inOutputWindow = (hms >= 93000 && hms <= 150000);

            double depthSum = sumBv + sumAv;
            double depthRatio = safeDiv(sumBv, sumAv);
            double spread = ap1 - bp1;
            double mid = (ap1 + bp1) / 2.0;

            // ========== 1~16（当前时刻） ==========
            factor[0]  = spread;
            factor[1]  = safeDiv(spread, mid);
            factor[2]  = mid;
            factor[3]  = safeDiv(bv1 - av1, bv1 + av1);
            factor[4]  = safeDiv(sumBv - sumAv, depthSum);
            factor[5]  = sumBv;
            factor[6]  = sumAv;
            factor[7]  = (sumBv - sumAv);
            factor[8]  = depthRatio;
            factor[9]  = safeDiv(tBidVol - tAskVol, tBidVol + tAskVol);
            factor[10] = safeDiv(sumBpBv, sumBv);
            factor[11] = safeDiv(sumApAv, sumAv);
            factor[12] = safeDiv(sumBpBv + sumApAv, depthSum);
            factor[13] = factor[11] - factor[10];
            factor[14] = (sumBv / N) - (sumAv / N);
            factor[15] = safeDiv(sumBvOverI - sumAvOverI, sumBvOverI + sumAvOverI);

            // ========== 17~19（上一时刻） ==========
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

            // 更新 prev
            prevMap.put(code, new PrevState(ap1, bp1, mid, depthRatio));

            // 09:30~15:00 之外不输出（只更新 prev）
            if (!inOutputWindow) return;

            String timeKey = pad6(hms);

            // ✅ 核心优化 2：in-mapper aggregation（你已经验证能把 MAP_OUTPUT_RECORDS 降到 4802）
            Agg agg = aggMap.get(timeKey);
            if (agg == null) {
                agg = new Agg();
                aggMap.put(timeKey, agg);
            }
            agg.add(factor);

            if (DEBUG) context.getCounter("DBG", "IN_WINDOW_LINES").increment(1);
        }

        @Override
        protected void cleanup(Context context) throws IOException, InterruptedException {
            if (DEBUG) context.getCounter("DBG", "EMITTED_TIME_KEYS").increment(aggMap.size());

            for (Map.Entry<String, Agg> e : aggMap.entrySet()) {
                outKey.set(e.getKey());
                outVal.set(e.getValue().toCsvSumAndCount());
                context.write(outKey, outVal);
            }

            aggMap.clear();
            prevMap.clear();
        }

        private static String pad6(int hms) {
            String s = Integer.toString(hms);
            int n = s.length();
            if (n >= 6) return s;
            StringBuilder sb = new StringBuilder(6);
            for (int i = 0; i < 6 - n; i++) sb.append('0');
            sb.append(s);
            return sb.toString();
        }

        // ===========================
        // ✅ 解析工具：避免 substring + Double.parseDouble 的大量对象创建
        // 数据里基本是整数（价格/量），这里实现一个“切片解析数字”
        // 如遇到小数点，也支持解析（但更慢一点）
        // ===========================

        private static int parseIntFast(String s, int start, int end) {
            int i = start;
            boolean neg = false;
            if (i < end && s.charAt(i) == '-') { neg = true; i++; }

            int v = 0;
            for (; i < end; i++) {
                char c = s.charAt(i);
                v = v * 10 + (c - '0');
            }
            return neg ? -v : v;
        }

        private static double parseDoubleFast(String s, int start, int end) {
            int i = start;
            boolean neg = false;
            if (i < end && s.charAt(i) == '-') { neg = true; i++; }

            long intPart = 0;
            long fracPart = 0;
            long fracDiv = 1;
            boolean frac = false;

            for (; i < end; i++) {
                char c = s.charAt(i);
                if (c == '.') {
                    frac = true;
                    continue;
                }
                int d = c - '0';
                if (!frac) {
                    intPart = intPart * 10 + d;
                } else {
                    fracPart = fracPart * 10 + d;
                    fracDiv *= 10;
                }
            }

            double v = (double) intPart + (frac ? (fracPart / (double) fracDiv) : 0.0);
            return neg ? -v : v;
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
