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
    /**
     * Mapper 输入：
     *   Key: LongWritable —— 当前行在文件中的偏移量
     *   Value: Text —— 一行 CSV 文本
     *
     * Mapper 输出：
     *   Key: Text —— tradeTime（格式化为 6 位：093000）
     *   Value: Text —— 20 个因子用逗号拼起来的字符串（CSV）
     *
     * Reducer 会按 Key（tradeTime）把所有股票同一时刻的因子聚在一起做平均。
     */
    public static class FactorMapper extends Mapper<LongWritable, Text, Text, Text> {

        // code -> 上一时刻状态（300 个文件，每个 code 只在一个文件里）
        private final Map<String, PrevState> prevMap = new HashMap<>();

        // 复用 Text 对象，减少 GC (Garbage Collection) 压力（MR 里是常用优化）
        private final Text outKey = new Text();
        private final Text outVal = new Text();

        @Override
        protected void map(LongWritable key, Text value, Context context)
                throws IOException, InterruptedException {

            /* ========== 1) 读一行、做基本过滤 ========== */

            String line = value.toString().trim();
            if (line.isEmpty()) return;

            // 每个文件第一行都是表头，从 "tradingDay" 开头，直接跳过
            if (line.startsWith("tradingDay")) {
                return;
            }

            // 数据是 CSV，用逗号分隔（和 snapshot.csv / 0102.csv 一样）
            String[] f = line.split(",");

            // 安全起见，长度太短的行丢弃（真实数据会比 57 列多）
            if (f.length < 57) {
                return;
            }

            /* ========== 2) 预处理：只取需要列（过滤无用列） ========== */

            String tradeTimeStr = f[1]; // tradeTime
            String code         = f[4]; // code

            // 全市场买/卖总量
            double tBidVol = Double.parseDouble(f[12]); // tBidVol
            double tAskVol = Double.parseDouble(f[13]); // tAskVol

            // 从 17 开始是 bp1,bv1,ap1,av1,...,bp10,bv10,ap10,av10
            int idx = 17;
            double[] bp = new double[N];
            double[] bv = new double[N];
            double[] ap = new double[N];
            double[] av = new double[N];
            for (int i = 0; i < N; i++) {      // N = 5，只用前五档
                bp[i] = Double.parseDouble(f[idx++]); // bp1,bp2,...
                bv[i] = Double.parseDouble(f[idx++]); // bv1,bv2,...
                ap[i] = Double.parseDouble(f[idx++]); // ap1,ap2,...
                av[i] = Double.parseDouble(f[idx++]); // av1,av2,...
            }
            // 后面的 bp6..bp10 等列我们完全忽略，相当于“过滤掉无用列”

            /* ========== 3) 处理 tradeTime 格式：hmmssffffffff -> hhmmss. 输出按 hhmmss 聚合 ========== */

            long tradeTimeRaw = Long.parseLong(tradeTimeStr);
            int hms;
//            if (tradeTimeStr.length() > 6) {
//                // hmmssffffffff，砍掉后 8 位
//                hms = (int) (tradeTimeRaw / 100000000L);
//            } else {
            hms = (int) tradeTimeRaw;
//            }

            // 输出窗口：9:30:00 ~ 15:00:00（11:30~13:00 中间本来就没数据）
            boolean inOutputWindow = (hms >= 93000 && hms <= 150000);

            /* ========== 4) 计算当前时刻的各种聚合量（用于多个因子复用） ========== */

            // 最优一档（买一/卖一）及对应量
            double ap1 = ap[0];
            double bp1 = bp[0];
            double bv1 = bv[0];
            double av1 = av[0];

            // 多档合计：深度、加权价格、衰减权重等
            double sumBv = 0, sumAv = 0;            // 买/卖深度（数量合计）
            double sumBpBv = 0, sumApAv = 0;        // VWAP 分子：Σ(price * volume)
            double sumBvOverI = 0, sumAvOverI = 0;  // 16 因子：按档位衰减 Σ(volume / level)

            for (int i = 0; i < N; i++) {
                int level = i + 1;
                double bv_i = bv[i];
                double av_i = av[i];
                double bp_i = bp[i];
                double ap_i = ap[i];

                sumBv += bv_i;
                sumAv += av_i;
                sumBpBv += bp_i * bv_i;
                sumApAv += ap_i * av_i;
                sumBvOverI += bv_i / level;
                sumAvOverI += av_i / level;
            }

            // 深度相关派生量
            double depthSum   = sumBv + sumAv;          //总深度
            double depthDiff  = sumBv - sumAv;          //深度差
            double depthRatio = safeDiv(sumBv, sumAv);  //深度比（用safeDiv）
            double spread     = ap1 - bp1;              //最优价差
            double mid        = (ap1 + bp1) / 2.0;      //中间价

            // 20 个因子数组（factor[0] 对应 alpha_1）
            double[] factor = new double[20];

            /* ========== 5) 因子 1~16：只依赖当前时刻 ========== */

            // 1 最优价差
            factor[0] = spread;
            // 2 相对价差 = spread / mid
            factor[1] = safeDiv(spread, mid);
            // 3 中间价
            factor[2] = mid;
            // 4 买一不平衡 (bv1 - av1) / (bv1 + av1)
            factor[3] = safeDiv(bv1 - av1, bv1 + av1);
            // 5 多档不平衡 (sumBv - sumAv) / (sumBv + sumAv)
            factor[4] = safeDiv(sumBv - sumAv, depthSum);
            // 6 买方深度
            factor[5] = sumBv;
            // 7 卖方深度
            factor[6] = sumAv;
            // 8 深度差
            factor[7] = depthDiff;
            // 9 深度比
            factor[8] = depthRatio;
            // 10 全市场买卖总量平衡指标 (tBidVol - tAskVol)/(tBidVol + tAskVol)
            factor[9] = safeDiv(tBidVol - tAskVol, tBidVol + tAskVol);
            // 11 VWAPBid = Σ(bp_i*bv_i) / Σ(bv_i)
            factor[10] = safeDiv(sumBpBv, sumBv);
            // 12 VWAPAsk = Σ(ap_i*av_i) / Σ(av_i)
            factor[11] = safeDiv(sumApAv, sumAv);
            // 13 加权中间价 = (Σ(bp_i*bv_i) + Σ(ap_i*av_i)) / (sumBv + sumAv)
            factor[12] = safeDiv(sumBpBv + sumApAv, depthSum);
            // 14 加权价差 = VWAPAsk - VWAPBid
            factor[13] = factor[11] - factor[10];
            // 15 买卖深度的密度差：平均每档深度差 (sumBv/N - sumAv/N)
            factor[14] = (sumBv / N) - (sumAv / N);
            // 16 按档位衰减的不对称度： (Σ(bv_i/i) - Σ(av_i/i)) / (Σ(bv_i/i) + Σ(av_i/i))
            double num16 = sumBvOverI - sumAvOverI;
            double den16 = sumBvOverI + sumAvOverI;
            factor[15] = safeDiv(num16, den16);

            /* ========== 6) 因子 17~19：需要上一时刻 ========== */

            PrevState ps = prevMap.get(code);
            double prevAp1, prevBp1, prevMid, prevDepthRatio;
            if (ps == null) {
                // 第一条就用自己当“上一条”，变化量为 0
                prevAp1 = ap1;
                prevBp1 = bp1;
                prevMid = mid;
                prevDepthRatio = depthRatio;
            } else {
                prevAp1 = ps.ap1;
                prevBp1 = ps.bp1;
                prevMid = ps.mid;
                prevDepthRatio = ps.depthRatio;
            }

            // 17 最优价变动：ap1 - prevAp1
            factor[16] = ap1 - prevAp1;
            // 18 中间价变动：mid - prevMid
            factor[17] = mid - prevMid;
            // 19 深度比变动：depthRatio - prevDepthRatio
            factor[18] = depthRatio - prevDepthRatio;
            // 20 价压指标：spread / (sumBv + sumAv)
            factor[19] = safeDiv(spread, depthSum);

            // 更新该股票的上一时刻状态
            prevMap.put(code, new PrevState(ap1, bp1, mid, depthRatio));

            // 只对 09:30~15:00 输出，09:25 的数据只用来初始化 prev 状态
            if (!inOutputWindow) return;

            // tradeTime 输出格式：固定 6 位，例如 093000
            String timeKey = String.format("%06d", hms);

            outKey.set(timeKey);
            outVal.set(factorsToCsv(factor)); // value 是 20 个因子 CSV
            context.write(outKey, outVal);
        }

        /**
         * 把 double[20] 转成 "f1,f2,...,f20"
         * Reducer 会再 split 回来求和求平均
         */
        private static String factorsToCsv(double[] f) {
            StringBuilder sb = new StringBuilder();
            for (int i = 0; i < f.length; i++) {
                if (i > 0) sb.append(',');
                sb.append(f[i]);
            }
            return sb.toString();
        }
    }

    // ======================  Reducer  ======================
    /**
     * Reducer 输入：
     *   Key: tradeTime（093000）
     *   Values: 该时刻所有股票（最多 300 个）的 20 因子字符串
     *
     * Reducer 输出：
     *   Key: tradeTime
     *   Value: 平均后的 20 因子字符串
     */

    public static class AverageReducer extends Reducer<Text, Text, Text, Text> {

        private boolean headerWritten = false;
        private final Text outVal = new Text();

        @Override
        protected void reduce(Text key, Iterable<Text> values, Context context)
                throws IOException, InterruptedException {

            // 第一次写表头：tradeTime,alpha_1,...,alpha_20
            if (!headerWritten) {
                StringBuilder header = new StringBuilder();
                header.append("alpha_1");
                for (int i = 2; i <= 20; i++) {
                    header.append(',').append("alpha_").append(i);
                }
                context.write(new Text("tradeTime"), new Text(header.toString()));
                headerWritten = true;
            }

            // sum[i]：累加所有股票在该时刻的第 i 个因子
            double[] sum = new double[20];
            int count = 0;

            // values：形如 "f1,f2,...,f20"
            for (Text v : values) {
                String[] parts = v.toString().split(",");
                if (parts.length < 20) continue;
                for (int i = 0; i < 20; i++) {
                    sum[i] += Double.parseDouble(parts[i]);
                }
                count++;
            }

            if (count == 0) return;

            // 横截面平均：sum / 股票数
            for (int i = 0; i < 20; i++) {
                sum[i] /= count;
            }

            outVal.set(factorsToCsv(sum));
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
    /**
     * Driver：负责配置并提交 Hadoop Job
     *
     * args[0]：输入目录（一天的数据目录，里面含 300 个股票文件/子目录）
     * args[1]：输出目录（必须不存在，否则 Hadoop 报错）
     */

    public static void main(String[] args) throws Exception {
        if (args.length != 2) {
            System.err.println("Usage: LobFactorSingleJob <oneDayInputDir> <outputDir>");
            System.exit(1);
        }

        Configuration conf = new Configuration();

        // 让 key,value 用逗号分隔，确保输出是 CSV（逗号分隔）
        conf.set("mapreduce.output.textoutputformat.separator", ",");

        Job job = Job.getInstance(conf, "lob-factor-single-job");

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
