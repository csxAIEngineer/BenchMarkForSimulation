import {
  BarChart,
  Callout,
  Divider,
  Grid,
  H1,
  H2,
  H3,
  Stack,
  Stat,
  Table,
  Text,
} from "cursor/canvas";

const SEEDS = ["11111", "7", "11", "23", "42"];

export default function EducationN25PairedEffect() {
  return (
    <Stack gap={20}>
      <Stack gap={6}>
        <H1>温度 0.8 加 recent：五个 seed 的配对差</H1>
        <Text tone="secondary">
          教育分层，每组 25 人，14 道人口学题一次请求。差是处理减对照。负值表示处理更接近 Twin-2K。seed 101 和校准臂没有跑。
        </Text>
      </Stack>

      <Callout tone="warning" title="预定的 6 个 seed 没有收齐">
        已完成 11111、7、11、23、42 的对照和处理。seed 101 的两臂，以及 seed 11111 的再答一遍，都没有开跑。五个差值即使全部同号，双侧符号翻转也是 2/32 = 0.0625，不低于 0.05。
      </Callout>

      <Grid columns={3} gap={16}>
        <Stat value="+0.0450" label="gpt-5.5 平均 TVD 差" tone="danger" />
        <Stat value="−0.0008" label="glm-5.3 平均 TVD 差" />
        <Stat value="+0.0018" label="deepseek-v4-pro 平均 TVD 差" />
      </Grid>

      <Stack gap={8}>
        <H2>平均 TVD 差</H2>
        <BarChart
          categories={SEEDS}
          beginAtZero={false}
          height={240}
          referenceLines={[{ value: 0, label: "0" }]}
          series={[
            { name: "gpt-5.5", data: [0.0221, 0.032, 0.0548, 0.023, 0.0931], tone: "danger" },
            { name: "glm-5.3", data: [-0.0093, -0.0055, -0.0094, 0.0064, 0.0137], tone: "info" },
            { name: "deepseek-v4-pro", data: [0.0114, -0.0003, 0.0076, -0.0097, -0.0003], tone: "neutral" },
          ]}
        />
        <Text size="small" tone="tertiary">
          横轴是 seed，纵轴是 14 题平均总变异距离的处理减对照。来源：五个已完成的 n=25 batch 臂，2026-10-14。gpt-5.5 的处理同时关掉了推理。
        </Text>
      </Stack>

      <Stack gap={8}>
        <H2>平均份额 MSE 差</H2>
        <BarChart
          categories={SEEDS}
          beginAtZero={false}
          height={240}
          referenceLines={[{ value: 0, label: "0" }]}
          series={[
            { name: "gpt-5.5", data: [0.0082, 0.0072, 0.013, 0.0033, 0.0244], tone: "danger" },
            { name: "glm-5.3", data: [-0.0007, 0.0001, -0.0013, 0.0021, 0.002], tone: "info" },
            { name: "deepseek-v4-pro", data: [0.0024, 0, 0.0011, -0.004, -0.003], tone: "neutral" },
          ]}
        />
        <Text size="small" tone="tertiary">
          横轴是 seed，纵轴是 14 题平均选项份额均方误差的处理减对照。同一批人和同一份份额，指标分开算。
        </Text>
      </Stack>

      <Stack gap={8}>
        <H3>五个 seed 的符号翻转</H3>
        <Table
          headers={["模型", "指标", "五个差", "均值", "同号", "符号翻转 p"]}
          columnAlign={["left", "left", "left", "right", "right", "right"]}
          rows={[
            ["gpt-5.5", "平均 TVD", "全部为正", "+0.0450", "5/5", "0.0625"],
            ["gpt-5.5", "平均份额 MSE", "全部为正", "+0.0112", "5/5", "0.0625"],
            ["glm-5.3", "平均 TVD", "3 负 2 正", "−0.0008", "3/5", "0.875"],
            ["glm-5.3", "平均份额 MSE", "2 负 3 正", "+0.0004", "3/5", "0.500"],
            ["deepseek-v4-pro", "平均 TVD", "3 负 2 正", "+0.0018", "3/5", "0.750"],
            ["deepseek-v4-pro", "平均份额 MSE", "3 负 2 正", "−0.0007", "3/5", "0.5625"],
          ]}
          rowTone={["danger", "danger", undefined, undefined, undefined, undefined]}
        />
        <Text size="small" tone="tertiary">
          人留在各自的 seed 里，没有合成 125 人。seed 内置换是 9999 次蒙特卡洛，问的是这两份答案能否互换。
        </Text>
      </Stack>

      <Divider />

      <Stack gap={8}>
        <H2>gpt-5.5 在卡片题上没有改答</H2>
        <Text>
          性别、年龄、学历、种族在五个 seed、三个模型上的改答都是 0/25。主终点里 gpt-5.5 的位移不是在改写卡片上已经写明的答案。
        </Text>
        <Table
          headers={["seed", "家庭收入", "政治倾向", "党派", "宗教活动", "seed 内 TVD p", "seed 内 MSE p"]}
          columnAlign={["left", "right", "right", "right", "right", "right", "right"]}
          rows={[
            ["11111", "+0.240", "+0.080", "+0.172", "0.000", "0.187", "0.027"],
            ["7", "+0.040", "+0.120", "+0.240", "+0.080", "0.081", "0.063"],
            ["11", "+0.200", "+0.160", "+0.320", "+0.160", "0.0005", "0.0002"],
            ["23", "+0.080", "0.000", "+0.012", "+0.080", "0.040", "0.069"],
            ["42", "+0.280", "+0.240", "+0.400", "+0.207", "0.0003", "0.0004"],
          ]}
          striped
        />
        <Text size="small" tone="tertiary">
          前四列是 gpt-5.5 这四题的 TVD 差。正值表示处理更远离 Twin-2K。党派和家庭收入在多数 seed 上把平均差往上拉。
        </Text>
      </Stack>

      <Stack gap={8}>
        <H3>每个 seed 的 recent 配比</H3>
        <Table
          headers={["seed", "中彩票", "日常琐事", "核爆炸"]}
          columnAlign={["left", "right", "right", "right"]}
          rows={[
            ["11111", "12", "5", "8"],
            ["7", "7", "11", "7"],
            ["11", "9", "8", "8"],
            ["23", "9", "7", "9"],
            ["42", "6", "8", "11"],
          ]}
        />
        <Text size="small" tone="tertiary">
          事件仍按人物 id 抽取。换 seed 换了一批人，三种事件的人数跟着变。每组都是 25 人。
        </Text>
      </Stack>
    </Stack>
  );
}
