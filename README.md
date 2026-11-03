# popbench

用大语言模型做人口模拟的基准。代码分成三层，互不越界：

| 层 | 目录 | 做什么 |
| --- | --- | --- |
| 数据 | `src/popbench/datasets/` | 下载、缓存、读出统一 schema |
| 模拟 | `src/popbench/models/` | 一种人物卡方法一个目录，只生成 persona + 题目输入包 |
| 通信 | `src/popbench/comms/` | 把输入包发给 LLM：provider、重试、磁盘缓存、JSON 解析 |

包名仍叫 `popbench`，避免和 Hugging Face 的 `datasets` 库撞名。命令行入口是 `benchmark`，`popbench` 是同一个程序。

```mermaid
flowchart LR
  datasets[datasets]
  models[models]
  comms[comms]
  datasets -->|"DatasetBundle"| models
  models -->|"PromptPackage"| comms
```

模拟层不导入 `httpx` 或 `cr_api`。通信层不决定人物卡怎么写。

## 数据集

| 注册名 | 目录 | 来源 | 单位 |
| --- | --- | --- | --- |
| `twin2k_500` | `datasets/twin2k_500/` | [LLM-Digital-Twin/Twin-2K-500](https://huggingface.co/datasets/LLM-Digital-Twin/Twin-2K-500) | 真人受访者 |
| `twin2k_mega_study` | `datasets/twin2k_mega_study/` | [Twin-2K-500-Mega-Study](https://huggingface.co/datasets/LLM-Digital-Twin/Twin-2K-500-Mega-Study) | 同一批 pid 上的 19 个后续研究 |
| `simbench` | `datasets/simbench/` | [pitehu/SimBench](https://huggingface.co/datasets/pitehu/SimBench) | **群体分布**，不生成个体 |
| `bdm_lettuce` | `datasets/bdm_lettuce/` | [Gibson & Li, Borealis 10.5683/SP3/CDLUEZ](https://doi.org/10.5683/SP3/CDLUEZ) | 610 名加拿大受访者 |

SimBench 的正式拼写是 SimBench。`SimBechMark` 只在命令行里被认成 `simbench`。每一行是一个群体提示加选项份额（`human_answer`），加载器拒绝带 `pid` 的个体微数据。当前人物卡 `atus_expanded` 只扩写有人口学的个体，不会把 SimBench 群体拆成虚构的人。群体模拟要等以后的方法。

Mega-Study 的行只有新问卷和真人答案。用 `atus_expanded` 前加 `--attach-twin2k`，按 pid 接上 Twin-2K-500 的人口学，不另造受访者。

`bdm_lettuce` 是 Gibson 与 Li 的加拿大生菜支付意愿实验（CC-BY 4.0）。别名 `canada_bdm`。每一行是一个 `pid`。`atus_expanded` 读取 `survey_*` 里的年龄、性别、教育和收入：收入短语用来判断是否超过 5 万美元，学历短语用来选择生活史模板。家庭规模、政治倾向、谁买菜、谁做饭会作为 Basics 里的标签。六项 WTP 和 `conclusion_purchased_product` 只放在 `Gold` 上。`benchmark fetch bdm_lettuce` 从 Borealis 把 CSV、README 和问卷 docx 缓存到 `data/bdm_lettuce/`。

缓存写在 `data/`，已在 `.gitignore` 里，不要提交。

## 人物卡方法

`models/atus_expanded/` 根据已有人口学（以及可选的 ATUS 分钟）写成 `## Basics` 和短的 `## Life history`。没有人口学就不会造一个人。SimBench 的群体行会被拒绝。

`models/atus_expanded_skill/` 使用同一张卡，并在卡下附上「类人问卷孪生」说明和一张按人物 id 固定的倾向卡。定价题模型只交 `buy_prob`，外层按 `seed:person_id:buy:item_id` 做 Bernoulli（概率夹在 0.15–0.85）。选择题交 `probs`，外层按 `seed:person_id:choice:item_id` 抽样。数值题仍直接作答。

题目的标准答案放在 `Gold` 上。人物卡方法读不到它。

## 安装

需要 Python 3.11 或更新版本。

```bash
python -m venv .venv
source .venv/bin/activate
pip install -e ".[dev]"
```

没有密钥时用 `--provider mock`。要打真实接口，把密钥写进 `.env`（见 `.env.example`）：

- `openai`：`OPENAI_API_KEY` / `OPENAI_BASE_URL`，或 DeepSeek 的 `DEEPSEEK_API_KEY` / `DEEPSEEK_BASE_URL`
- `cr`：`CR_API_KEY`，走仓库里的 `cr_api.CRClient`

## 拉取数据

```bash
benchmark fetch twin2k_500
benchmark fetch twin2k_mega_study --study privacy --limit 20
benchmark fetch simbench --split SimBenchPop --limit 20
benchmark fetch bdm_lettuce
```

`twin2k_500` 读 Hub 上的 `full_persona` 和 `wave_split`。Mega-Study 默认会拉全部 19 个 study，体积大约几百 MB，试跑请加 `--study` 和 `--limit`。SimBench 的 split 是 `SimBenchPop` 和 `SimBenchGrouped`。

也可以：

```bash
python scripts/fetch_datasets.py simbench --split SimBenchPop --limit 20
```

## 最小模拟

用 mock，不访问网络上的模型：

```bash
benchmark fetch twin2k_500 --limit 2
benchmark simulate \
  --dataset twin2k_500 \
  --method atus_expanded \
  --provider mock \
  --limit 1 \
  --out runs/minimal
```

结果在 `runs/minimal/responses.jsonl`。同一输入会进 `runs/minimal/cache/`。

同一个人的多道题合成一次模型请求：人物卡写在消息顶部，后面跟一批题目。`--batch-size` 默认 15。返回的 JSON 是 `{"answers": [{"question_id", "answer", "rationale"}, ...]}`；SimBench 的分布题用 `distribution` 代替 `answer`。`responses.jsonl` 仍是一题一行。磁盘缓存按整批请求计算，不是按单题。成功的批次会留下缓存，重跑时可以接着未完成的批次。

SimBench 仍然可以 `benchmark fetch simbench` 下载。现在没有对应的人物卡方法，`atus_expanded` 会拒绝这些群体行。

Mega-Study 接上 Twin-2K 人口学后再模拟：

```bash
benchmark fetch twin2k_500 --limit 50
benchmark fetch twin2k_mega_study --study privacy --limit 5
benchmark simulate \
  --dataset twin2k_mega_study \
  --study privacy \
  --method atus_expanded \
  --attach-twin2k \
  --provider mock \
  --limit 1
```

加拿大 BDM 生菜支付意愿，用同一张人物卡：

```bash
benchmark fetch bdm_lettuce
benchmark simulate \
  --dataset bdm_lettuce \
  --method atus_expanded \
  --provider mock \
  --limit 1
```

`canada_bdm` 是 `bdm_lettuce` 的别名。

换真实模型时把 `--provider mock` 换成 `openai` 或 `cr`，并用 `--model` 指定模型名。`cr` 的默认模型是 `deepseek-v4-pro`。

## 测试

```bash
pytest
```

测试使用本地夹具和 mock provider，不下载 Hub 数据。

## 这一版删掉了什么

旧的实验入口（ATUS 人群划分、七类题 TVD 对比、jin10 时事拼接、经济切片、访问者访谈）混在取数、提示词和 HTTP 里。它们已从本分支移除，历史仍在 `main`。数据缓存本来就不入库。
