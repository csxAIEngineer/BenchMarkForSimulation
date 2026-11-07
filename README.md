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

把 Twin-2K-500 与加拿大 BDM 生菜实验的原文基本资料抄成人物卡，并按种子 42 抽样。不调用语言模型，不补写生活史，不把空单元格写成倾向。Twin-2K 给模型看的 Basics 行是中文标签、目录原文题干和答案，不含题号。题号只留在 `basic_info`。BDM 与这 14 项不是同一问卷，差异见 `experiments/prepro/README.md`。算法、列约束和源文件锁也在那里。锁文件是 `experiments/prepro/source-lock.json`。

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

不调用模型。把模拟回复和留出的人类答案对齐，报告分类题的 TVD 与 Top-1，以及 BDM 出价的个体准确率、MAE、Fisher z 相关、Glass Δ 和 SD 比。

```bash
python3 scripts/evaluate.py
```

报告写到 `outputs/evaluate/report.json`。

## 测试

```bash
python3 -m pytest -q
```

测试使用夹具和假的 CR 客户端，不下载数据，也不读取密钥。
