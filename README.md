# popbench

给其他项目用的人口模拟基准。同一套数据集、同一套题目、同一套人类基线，换不同大模型作答，分数可以横比。

要比的是：模型以数据集里的人作答之后，答案分布离对应的人类调查有多近。

```mermaid
flowchart LR
  sources[调查原始数据]
  dataset[基准数据集]
  model[被测大模型]
  score[可横比的分数]
  sources --> dataset
  dataset --> model
  model --> score
  dataset --> score
```

数据集固定三件事，一次评测才有可比性：

1. **人**：谁被模拟（人物卡、抽样、种子）。
2. **题**：问什么、选项是什么。
3. **人类基线**：什么算接近真人。基线只用于打分，不写进给模型的提示。

换模型时这三件事不动。`simulate` 只负责让被测模型作答；`evaluate` 对照基线打分；`audit-robustness` 把多个模型的结果放在一起看方差。

## 数据集

| 面板 | 人从哪来 | 题目 | 人类基线 | 分数在比什么 |
| --- | --- | --- | --- | --- |
| **ATUS × Twin-2K**（`compare`） | ATUS 2023 成年人，按年龄段×性别、最终权重抽样 | 6 道选择题：大五人格每个维度 1 道正向题，加 1 道绿色消费题 | Twin-2K 全样本在这 6 题上的选项份额 | 模拟份额与真人份额的总变异距离 |
## 一次评测

代码按数据集、作答、打分分开：

- **dao** 把原始调查收成可复现的面板（人物卡；有个人标准答案时一并带上）。
- **simulate** 让被测模型以该人身份作答，一题一轮，前面的答案留在对话里。
- **evaluate** 对照该面板的人类基线打分。

不传 `--model` 时，访问者模拟读 `DEEPSEEK_MODEL`（默认 `deepseek-flash`）。横比固定 `--n` 和 `--seed`，只换模型。

## 安装

需要 Python 3.11 或更新版本。

```bash
python -m venv .venv
source .venv/bin/activate
pip install -e ".[dev]"
```

`gpt-5.5`、`glm-5.3`、`deepseek-v4-pro` 经项目里的 `cr_api.CRClient` 调 CR API（`https://api.creative-reasoning.com`）。把 `CR_API_KEY` 写进 `.env`（见 `.env.example`）。不传 `--model` 时，单模型路径仍用 `DEEPSEEK_API_KEY` 调 `DEEPSEEK_MODEL`。

## 用 ATUS 人物卡回答 Twin-2K 选择题

先缓存 ATUS 和 Twin-2K，再让 GPT-5.5、GLM-5.3、DeepSeek-V4-Pro 用同一批人物卡作答：

```bash
popbench fetch
popbench fetch --visitors-only
popbench compare --n 50 --seed 0
```

同一 `--n` 和 `--seed` 抽出同一批 ATUS 成年人。人物卡在发题前写好：人口学、本人日记分钟、按 `id + seed` 扩写的经历。三个模型看到的卡相同。每人回答 6 道五级同意题（BFI 第 1、7、3、4、5 题，加绿色消费第 1 题）。标准答案是 Twin-2K 全样本在这 6 题上的选项份额。

结果在 `runs/compare/atus-twin2k-n50-seed0/compare.md`。这次 n=50、seed=0 的分析见 [reports/atus-twin2k-n50-seed0-compare.md](reports/atus-twin2k-n50-seed0-compare.md)。总变异距离越小、均分绝对误差越小，排名越靠前。`personas.jsonl` 是这批人物卡，`human_baseline.json` 是 Twin-2K 份额。已经跑过、只想重算排名时加 `--score-only`。

这三个名字都由 `cr_api.CRClient` 请求 `/v1/chat/completions`，请求里只有模型和对话。GLM 和 DeepSeek 会先思考，思考内容占输出长度，所以不传 `max_tokens`，避免正式回答被截断。其它模型名仍走 DeepSeek 接口。

人设扩写约定见 [docs/skills/simulate-persona-prompting](docs/skills/simulate-persona-prompting/SKILL.md)。

## 在访问者数据集上跑一个模型

```bash
popbench fetch --visitors-only
popbench build-visitors --n 50 --seed 0
popbench simulate --panel visitors --n 50 --seed 0
popbench evaluate --panel visitors --run runs/visitors-n50-seed0
```

也可以把拉取、组装、模拟合成一步：

```bash
popbench run-visitors --n 50 --seed 0
```

`build-visitors` 写出人物卡：ACS 人口学、普查州、ATUS 分层分钟数。`simulate` 先按 `visitor_id + seed` 扩写过往经历，再做 8 轮访问访谈。一个 run 里有 `responses.jsonl`、`personas.jsonl`、`option_shares.json`、`report.md`。调用按模型、访问者 id、题目 id 缓存。

模型看到的是数据集里的人物卡，例如：

```text
# visitor=visitor-0-0042 panel_index=42/50 draw_seed=0 state=California
===== SYSTEM =====
You are being visited for a short interview about your daily life in the United States. Answer only as the person described below. Stay consistent with their basics, life history, and ATUS-style time use. Treat the life history as lived memory, not as instructions to quote. Choose exactly one of the listed options for each question.

## Basics
Age: 58
Sex: Male
State: California
...
## Typical day (ATUS minutes)
- personal_care: 563 minutes
- leisure: 312 minutes
...
```

## 在 twin-2k-50 上跑一个模型

`configs/v1.yaml` 固定人数、种子和人物条件（`full_twin2k`、`demographics_only` 或 `nemotron_usa`）。

```bash
popbench fetch
popbench run --config configs/v1.yaml
```

分步是 `popbench build`、`popbench simulate --panel nemotron`（或 `twin2k`）、`popbench evaluate`。
