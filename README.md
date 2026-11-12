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

配置在 `config/simulate.json`。不写 `reasoning_effort` 时，请求不带这个字段，回复仍写到 `outputs/simulate/<model>/<dataset>/responses.jsonl`，缓存键与以前相同。这不是已经证明的「推理关闭」，只是 API 默认。

显式设置 `reasoning_effort`（例如 `none`、`low`、`medium`、`high`）后，字段会传给 `CRClient.chat`，并进入缓存键。文件改写到 `outputs/simulate/<model>__reasoning-<effort>/` 和 `outputs/simulate/cache/<model>__reasoning-<effort>/`，不会读取或覆盖默认目录。清单记录 `model`、`temperature` 和 `reasoning_effort`。某个取值被 API 拒绝时，换一个已记录的允许值，不要把温度偷偷改掉。

```bash
python3 scripts/simulate.py /tmp/simulate-twin-only.json
```

`/tmp` 配置里可以加 `"reasoning_effort": "none"` 或 `"medium"`。温度仍由该配置的 `temperature` 决定。2026-11-10 对这 7 个模型各发了一条短消息、温度 0：`none` 在 claude-opus-5-5、deepseek-v4-flash、MiniMax-M3.1-Flash 上返回 400。`low` 对 gemini-3.8-flash 不产生 `reasoning_content`，`medium` 对除 claude-opus-5-5 以外的模型都会产生。因此显式开启统一用 `medium`，显式关闭只用 API 接受的 `none`。省略字段的旧 93 人结果不标成关闭。

One question per request is a separate protocol. Set `"questions_per_request": 1` with `"reasoning_effort": "none"` and `"temperature": 0`. Each call sends the persona card and that question only. Files go to `outputs/simulate/<model>__reasoning-none__single-q/` and the matching cache directory. The cache key includes the subject and question, so a resumed run skips finished items. This does not read or write `outputs/simulate/<model>/` or `outputs/simulate/<model>__reasoning-none/`. The config is `config/simulate_single_q_none.json`.

```bash
python3 scripts/simulate.py config/simulate_single_q_none.json --model gemini-3.8-flash
```

The three-option arm is also batched: one persona card and all 40 pricing questions, with reasoning left unset and temperature copied from the default run. Each question keeps the wave-4 stem and replaces the two purchase lines with three letters whose order is a hash of person and product. Files go to `outputs/simulate/<model>__three-option/` and the matching cache. This does not read or write the default, `__reasoning-*`, or `__single-q` directories. The config is `config/simulate_three_option.json`.

```bash
python3 scripts/simulate.py config/simulate_three_option.json --model gemini-3.8-flash
```

The need-inject arm is batched the same way as the default two-option run. Reasoning stays unset and temperature stays 0. Each stem keeps the official Yes/No options. One need sentence, chosen by a hash of person and product, is inserted between the price and `Would you`. Files go to `outputs/simulate/<model>__need-inject/` and the matching cache. The locked triplets in `outputs/evaluate/triplets.parquet` are not resampled. The config is `config/simulate_need_inject.json`.

```bash
python3 scripts/simulate.py config/simulate_need_inject.json --model gemini-3.8-flash
```

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

## 4. 免费价格端点

不调用模型，也不读 BDM。锁定样本里的定价刺激没有单独的 `price_shown` 列。价格是 wave 4 `QuestionText` 里固定句子的 `$X.XX`，已经抄在 `outcomes.jsonl` 的 `question_text`。句子对不上就不解析，不当成免费。`$0.00` 是该商品自身最高正价的 0% 档，后面还有 10% 到 100% 共十档，四舍五入到分。不同商品的美元价格不混在一条曲线里。等权曲线先在商品内算购买率，再对有观测的商品取平均。

非递增指相邻已观测档的购买率不上升。持平可以。上升才算一次向上跳。没有观测的档是缺失，不是需求为零。这是被试被分配到不同价格格子后的描述，不是同一个人的反事实需求，也不是对原论文样本或模型的复现。

```bash
python3 scripts/free_price_demand.py
```

配置是 `config/free_price.json`。汇总写到 `outputs/evaluate/free_price/`。免费档的主购买率只用双方都映射成功的报价。未映射、解析失败和冲突报价留在未知里，并同时给出把未知全部当成拒绝或全部当成购买的上下界。

## 测试

```bash
python3 -m pytest -q
```

测试使用夹具和假的 CR 客户端，不下载数据，也不读取密钥。
