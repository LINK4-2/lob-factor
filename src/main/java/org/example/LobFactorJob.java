package org.example;

import org.apache.hadoop.conf.Configuration;
import org.apache.hadoop.fs.Path;
import org.apache.hadoop.io.LongWritable;
import org.apache.hadoop.io.Text;
import org.apache.hadoop.mapreduce.Job;
import org.apache.hadoop.mapreduce.Mapper;
import org.apache.hadoop.mapreduce.Reducer;
import org.apache.hadoop.mapreduce.lib.input.FileInputFormat;
import org.apache.hadoop.mapreduce.lib.output.FileOutputFormat;

import java.io.IOException;
import java.util.ArrayList;
import java.util.Collections;
import java.util.Comparator;
import java.util.List;

public class LobFactorJob {

    // ==== 一些通用工具 ====

    private static final int N = 5;
    private static final double EPS = 1e-7;

    private static double safeDiv(double num, double den) {
        return num / (den + EPS);
    }

    // 一个简单的 LOB 记录结构（只保留需要的字段）
    private static class LobRecord {
        String code;
        long tradeTimeRaw; // 原 tradeTime 字段（可能有小数）
        int hms;           // 转成 hhmmss，例如 93000 表示 09:30:00

        long tBidVol;
        long tAskVol;
        long[] bp = new long[N];
        long[] bv = new long[N];
        long[] ap = new long[N];
        long[] av = new long[N];

        static LobRecord fromFields(String[] f) {
            // 假设你预处理后的字段顺序是：
            //  0 tradeTime 1 code 2 tBidVol 3 tAskVol
            // 然后 5.. 5+4*N-1 是 bp1,bv1,ap1,av1,...,bp5,bv5,ap5,av5
            LobRecord r = new LobRecord();
            r.tradeTimeRaw = Long.parseLong(f[1]);
            String tradeTimeStr = f[0];
            if (tradeTimeStr.length() > 6) {
                // hmmssffffffff → 取高位 hmmss
                long v = Long.parseLong(tradeTimeStr);
                r.hms = (int) (v / 1_000_000_000L);
            } else {
                r.hms = Integer.parseInt(tradeTimeStr);
            }

            r.code = f[1];
            r.tBidVol = Long.parseLong(f[2]);
            r.tAskVol = Long.parseLong(f[3]);

            int idx = 4;
            for (int i = 0; i < N; i++) {
                r.bp[i] = Long.parseLong(f[idx++]);
                r.bv[i] = Long.parseLong(f[idx++]);
                r.ap[i] = Long.parseLong(f[idx++]);
                r.av[i] = Long.parseLong(f[idx++]);
            }
            return r;
        }
    }

    // 计算 20 个因子的工具
    private static double[] computeFactors(LobRecord cur, LobRecord prev) {
        double[] f = new double[20];

        double bp1 = cur.bp[0];
        double ap1 = cur.ap[0];
        double bv1 = cur.bv[0];
        double av1 = cur.av[0];

        // 预先求各种和
        double sumBv = 0, sumAv = 0;
        double sumBpBv = 0, sumApAv = 0;
        double sumBvOverI = 0, sumAvOverI = 0;
        for (int i = 0; i < N; i++) {
            int level = i + 1;
            double bv = cur.bv[i];
            double av = cur.av[i];
            double bp = cur.bp[i];
            double ap = cur.ap[i];

            sumBv += bv;
            sumAv += av;
            sumBpBv += bp * bv;
            sumApAv += ap * av;
            sumBvOverI += bv / (double) level;
            sumAvOverI += av / (double) level;
        }

        double depthDiff = sumBv - sumAv;
        double depthSum = sumBv + sumAv;
        double depthRatio = safeDiv(sumBv, sumAv);

        // 1 最优价差 ap1 - bp1
        f[0] = ap1 - bp1;

        // 2 相对价差
        double mid = (ap1 + bp1) / 2.0;
        f[1] = safeDiv(ap1 - bp1, mid);

        // 3 中间价
        f[2] = mid;

        // 4 买一不平衡
        f[3] = safeDiv(bv1 - av1, bv1 + av1);

        // 5 多档不平衡
        f[4] = safeDiv(sumBv - sumAv, depthSum);

        // 6 买方深度
        f[5] = sumBv;

        // 7 卖方深度
        f[6] = sumAv;

        // 8 深度差
        f[7] = depthDiff;

        // 9 深度比
        f[8] = depthRatio;

        // 10 买卖量平衡指数
        f[9] = safeDiv(cur.tBidVol - cur.tAskVol, cur.tBidVol + cur.tAskVol);

        // 11 买方加权价格 VWAPBid
        f[10] = safeDiv(sumBpBv, sumBv);

        // 12 卖方加权价格 VWAPAsk
        f[11] = safeDiv(sumApAv, sumAv);

        // 13 加权中间价
        f[12] = safeDiv(sumBpBv + sumApAv, depthSum);

        // 14 加权价差
        f[13] = f[11] - f[10];

        // 15 买卖密度差
        f[14] = (sumBv / N) - (sumAv / N);

        // 16 买卖不对称度（按档位衰减）
        double num16 = sumBvOverI - sumAvOverI;
        double den16 = sumBvOverI + sumAvOverI;
        f[15] = safeDiv(num16, den16);

        // 需要前一时刻数据的因子：17,18,19
        double prevAp1 = ap1;
        double prevBp1 = bp1;
        double prevDepthRatio = depthRatio;

        if (prev != null) {
            double pAp1 = prev.ap[0];
            double pBp1 = prev.bp[0];
            double pSumBv = 0, pSumAv = 0;
            for (int i = 0; i < N; i++) {
                pSumBv += prev.bv[i];
                pSumAv += prev.av[i];
            }
            prevAp1 = pAp1;
            prevBp1 = pBp1;
            prevDepthRatio = safeDiv(pSumBv, pSumAv);
        }

        // 17 最优价变动
        f[16] = ap1 - prevAp1;

        // 18 中间价变动
        double prevMid = (prevAp1 + prevBp1) / 2.0;
        f[17] = mid - prevMid;

        // 19 深度比变动
        f[18] = depthRatio - prevDepthRatio;

        // 20 价压指标
        f[19] = safeDiv(ap1 - bp1, depthSum);

        return f;
    }

    private static String factorsToString(double[] f) {
        StringBuilder sb = new StringBuilder();
        for (int i = 0; i < f.length; i++) {
            if (i > 0) sb.append('\t');
            sb.append(f[i]);
        }
        return sb.toString();
    }

    // ==== Job1: 按股票输出每个时刻的因子 ====

    public static class StockKeyMapper extends Mapper<LongWritable, Text, Text, Text> {
        private boolean isHeader = true;
        private Text outKey = new Text();
        private Text outVal = new Text();

        @Override
        protected void map(LongWritable key, Text value, Context context)
                throws IOException, InterruptedException {
            String line = value.toString().trim();
            if (line.isEmpty()) return;

            // 跳过表头（第一行以 tradingDay 开头）
            if (isHeader && line.startsWith("tradingDay")) {
                isHeader = false;
                return;
            }
            isHeader = false;

            String[] fields = line.split("\t");
            // 这里假设 0 是 tradingDay，2 是 code
            String code = fields[2];

            outKey.set(code);
            outVal.set(line);
            context.write(outKey, outVal);
        }
    }

    public static class FactorByStockReducer extends Reducer<Text, Text, Text, Text> {
        private Text outKey = new Text();
        private Text outVal = new Text();

        @Override
        protected void reduce(Text key, Iterable<Text> values, Context context)
                throws IOException, InterruptedException {

            // 收集并解析成 LobRecord 列表
            List<LobRecord> records = new ArrayList<>();
            for (Text v : values) {
                String line = v.toString();
                String[] fields = line.split("\t");
                records.add(LobRecord.fromFields(fields));
            }

            // 按 tradeTime 排序
            Collections.sort(records, Comparator.comparingLong(r -> r.tradeTimeRaw));

            LobRecord prev = null;
            for (LobRecord cur : records) {
                int hms = cur.hms;  // hhmmss

                // 只保留 09:30:00 ~ 15:00:00
                if (hms < 93000 || hms > 150000) {
                    prev = cur;
                    continue;
                }

                double[] factors = computeFactors(cur, prev);
                String timeStr = String.format("%06d", hms); // 093000 形式

                outKey.set(timeStr);  // Job2 用 tradeTime 做 key
                outVal.set(factorsToString(factors));
                context.write(outKey, outVal);

                prev = cur;
            }
        }
    }

    // ==== Job2: 按时间求 300 只股票的平均 ====

    public static class TimeKeyMapper extends Mapper<LongWritable, Text, Text, Text> {
        private Text outKey = new Text();
        private Text outVal = new Text();

        @Override
        protected void map(LongWritable key, Text value, Context context)
                throws IOException, InterruptedException {
            String line = value.toString().trim();
            if (line.isEmpty()) return;

            String[] parts = line.split("\t", 2);
            if (parts.length < 2) return;

            outKey.set(parts[0]);     // tradeTime
            outVal.set(parts[1]);     // 20 个因子
            context.write(outKey, outVal);
        }
    }

    public static class AverageReducer extends Reducer<Text, Text, Text, Text> {
        private Text outVal = new Text();

        @Override
        protected void reduce(Text key, Iterable<Text> values, Context context)
                throws IOException, InterruptedException {

            double[] sum = new double[20];
            int count = 0;

            for (Text v : values) {
                String[] fs = v.toString().split("\t");
                if (fs.length < 20) continue;
                for (int i = 0; i < 20; i++) {
                    sum[i] += Double.parseDouble(fs[i]);
                }
                count++;
            }

            if (count == 0) return;

            for (int i = 0; i < 20; i++) {
                sum[i] /= count;
            }

            outVal.set(factorsToString(sum));
            context.write(key, outVal);
        }
    }

    // ==== Driver：顺序跑两个 Job ====

    public static void main(String[] args) throws Exception {
        if (args.length != 3) {
            System.err.println("Usage: LobFactorJob <input> <tmpOutput> <finalOutput>");
            System.exit(1);
        }

        Configuration conf = new Configuration();

        // Job1
        Job job1 = Job.getInstance(conf, "lob-factor-job1");
        job1.setJarByClass(LobFactorJob.class);
        job1.setMapperClass(StockKeyMapper.class);
        job1.setReducerClass(FactorByStockReducer.class);

        job1.setMapOutputKeyClass(Text.class);
        job1.setMapOutputValueClass(Text.class);
        job1.setOutputKeyClass(Text.class);
        job1.setOutputValueClass(Text.class);

        FileInputFormat.addInputPath(job1, new Path(args[0]));
        FileOutputFormat.setOutputPath(job1, new Path(args[1]));

        if (!job1.waitForCompletion(true)) {
            System.exit(1);
        }

        // Job2
        Job job2 = Job.getInstance(conf, "lob-factor-job2");
        job2.setJarByClass(LobFactorJob.class);
        job2.setMapperClass(TimeKeyMapper.class);
        job2.setReducerClass(AverageReducer.class);

        job2.setMapOutputKeyClass(Text.class);
        job2.setMapOutputValueClass(Text.class);
        job2.setOutputKeyClass(Text.class);
        job2.setOutputValueClass(Text.class);

        FileInputFormat.addInputPath(job2, new Path(args[1]));
        FileOutputFormat.setOutputPath(job2, new Path(args[2]));

        System.exit(job2.waitForCompletion(true) ? 0 : 1);
    }
}

