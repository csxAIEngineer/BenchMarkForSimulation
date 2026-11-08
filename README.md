# popbench

三块本地流程：制作人物卡、经 CR API 做定价模拟、评价模拟结果。仓库只提交代码和上游源文件锁。人物卡、抽样、outcomes、模拟回复和评价报告都写在本机，不入库。

## 安装

需要 Python 3.11 或更新版本。

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -e ".[dev]"
```

模拟需要 `CR_API_KEY`。复制 `.env.example` 为 `.env` 后填写。`.env` 不入库。评价和人物卡构建不调用模型。

## 1. 人物卡预处理

把 Twin-2K-500 与加拿大 BDM 生菜实验的原文基本资料抄成人物卡，并按种子 42 抽样。不调用语言模型，不补写生活史，不把空单元格写成倾向。Twin-2K 给模型看的正文只有 14 行 `标签: 题干: 答案`，标签是短英文，不含题号、中文和 Identity。题号只留在 `basic_info`。BDM 与这 14 项不是同一问卷，差异见 `experiments/prepro/README.md`。算法、列约束和源文件锁也在那里。锁文件是 `experiments/prepro/source-lock.json`。

下面是 `subject_id` 为 1 的模型侧 `text`。这不是审计用的 `basic_info`；QID 和 evidence 不在这段正文里。

```text
Region: Which part of the United States do you currently live in?: Midwest (ND, SD, NE, KS, MN, IA, MO, WI, IL, MI, IN, OH)
Sex: What is the sex that you were assigned at birth?: Female
Age: How old are you?: 30-49
Educ: What is the highest level of schooling or degree that you have completed?: Some college, no degree
Race: What is your race or origin?: Black
Citizen: Are you a citizen of the United States?: Yes
Marital: Which of these best describes you?: Married
Religion: What is your present religion, if any?: Nothing in particular
Attend: Aside from weddings and funerals, how often do you attend religious services?: A few times a year
Party: In politics today, do you consider yourself a: Independent
Income: Last year, what was your total family income from all sources, before taxes?: $50,000-$75,000
Ideology: In general, would you describe your political views as: Moderate
HHSize: Including yourself, how many people currently live in your household?: 3
Employ: What is your current employment status?: Part-time employment
```

```bash
python3 scripts/build_prepro.py
python3 scripts/build_prepro.py --verify
```

本地产物：

| 路径 | 内容 |
| --- | --- |
| `data/prepro/raw/` | 按锁下载的上游文件 |
| `data/prepro/outcomes/` | 留出的定价答案，不进人物卡 |
| `outputs/prepro/` | `cards.jsonl`、`sample.json`、`distribution.json`、`exclusions.json`、`field_map.json`、`wave_audit.json`、`manifest.json` |

这些目录已在 `.gitignore` 里。`--verify` 只核对本地产物的哈希，不把它们提交进 git。`--init-source-lock` 才会改写锁，复现时不要加。

## 2. 经 CR API 模拟

`cr_api.CRClient` 是唯一的模型入口，地址为 `https://api.creative-reasoning.com`。模型 id 必须在 `cr_api.TEXT_MODELS` 里。默认 `deepseek-v4-pro`。每个人物卡一次请求，问题来自留出 outcomes 的题干或 BDM 出价刺激，人类答案不放进请求。

先完成本地人物卡构建，再：

```bash
python3 scripts/simulate.py
python3 scripts/simulate.py --model deepseek-v4-pro --limit-subjects 1
```

配置在 `config/simulate.json`。回复写到 `outputs/simulate/<model>/<dataset>/responses.jsonl`。

## 3. 评价

不调用模型。把模拟回复和留出的人类答案按 `dataset + subject_id + question_id` 对齐，不按行序。报告仍包含分类题的 TVD 与群体 Top-1，以及 BDM 出价的原有指标。Twin2K 定价题另外给出个人层指标。短答（`Yes` / `No` / `Would purchase`）按确定规则对齐到留出的整句选项后再编码，规则见 `popbench.evaluate.options`。对不上的答复保持未映射，算作个人准确率的错误，也不进入购买率。

群体 `categorical.mean_top1_acc` 只看每一题上全体答案的众数是否重合，不是个人准确率。

| 字段 | 方向 | 含义 |
| --- | --- | --- |
| `individual_accuracy_micro` | ↑ | 可评分的（人，题）对里，编码后与该人留出答案一致的比例 |
| `individual_accuracy_macro` | ↑ | 每个至少有一题可评分的人先算准确率，再对人取不加权平均 |
| `cross_person_pearson` | ↑ | 每人在同一组定价题上的购买率（Yes=1）的 Pearson 相关 |
| `cross_person_spearman` | ↑ | 同上购买率的 Spearman 相关，并列取平均秩 |
| `glass_delta` | →0 | `(模型购买率均值 - 人类购买率均值) / 人类购买率样本标准差`，`ddof=1`。正数表示模型比人类更倾向购买 |
| `sd_ratio` | →1 | 模型与人类购买率的样本标准差之比，`ddof=1`。小于 1 表示人与人之间的差异被压扁 |

购买率是这组共同题目上的 Yes 比例，不是出价，也不从二元选择反推 WTP。只有在共同题目上每题都能编成 0/1 的人进入相关、Glass Δ 和 SD 比；覆盖人数和未映射、解析错误、重复冲突都写在 `coverage` 里。样本标准差为 0 或完整人数少于 2 时，对应指标是 JSON `null` 并带 `reason`，不写成 0、NaN 或 Infinity。BDM 出价仍用原来的总体标准差 Glass Δ，不受这组个人购买率指标影响。

```bash
python3 scripts/evaluate.py
```

报告写到 `outputs/evaluate/report.json`。Twin2K 的个人层结果在 `datasets.twin2k_500.pricing_person`。

## 测试

```bash
python3 -m pytest -q
```

测试使用夹具和假的 CR 客户端，不下载数据，也不读取密钥。
