# 定价实验第一步：原文 basic info 人物卡与可复现抽样

这一步只做两件事：把 Twin-2K-500 定价任务和加拿大 BDM 生菜实验的原文基本资料抄成人物卡，并给出一个固定种子的抽样。人物卡不调用语言模型，不补写生活史，也不把空单元格写成倾向。

`atus_expanded` 仍会根据学历、收入和婚姻状况生成生活史。那不是这一步的人物卡。本目录的卡片只有 `## Identity` 和 `## Basics`。

## 数据来源

| 数据集 | 总体 | 许可 | 版本 |
| --- | --- | --- | --- |
| Twin-2K-500 | 已发布的 2,058 名美国受访者，全部纳入 | CC BY 4.0 | Hugging Face `LLM-Digital-Twin/Twin-2K-500`，revision `f883165a3026fde855dfd448e0cd16443ab257b6` |
| BDM lettuce | 已发布的 610 名加拿大受访者，全部纳入 | CC BY 4.0 | Borealis [10.5683/SP3/CDLUEZ](https://doi.org/10.5683/SP3/CDLUEZ)，V1 |

Twin-2K 的 basic info 是人口学块 QID11–QID24，取自 `question_catalog_and_human_response_csv/wave1_3_response_label.csv`。题干从 `question_catalog.json` 原样抄入 JSON 字段 `source_question_text`。目录里的题干如果句子不完整（例如 QID20 停在 “consider yourself a”），卡片保持原样，不补全。

定价结果是 wave 4 的 `wave4_Q_wave4_A` 里 QID9_1–QID9_40。`QID9_n` 是该受访者看到的第 n 道定价题，不是固定商品。商品名和价格写在每个人的 `question_text` 里。标签 CSV 只有 “会买 / 不会买”，单独使用会丢掉价格，所以结果文件来自 wave JSON，并与 `wave4_response_label.csv` 逐格核对。Wave 1–3 的 QID9 是更早一次作答：82,320 个单元格里 69,059 个与 wave 4 相同、13,261 个不同。早一次的答案不进入人物卡，也不是这一步的留出结果。

BDM 的 basic info 是 CSV 里全部 `survey_*` 列。`100A_README.txt` 写明这些列的题干在 `DemographicQuestionnaire.docx`。该 docx 只有图片、没有可抽取的正文，因此卡片用列名和单元格原文，不使用 `fields.py` 里的英文转述。六项 WTP 和 `conclusion_purchased_product` 是结果。处理、呈现顺序、练习题和计时不是基本资料。

空白单元格写成 `MISSING`，状态为 `missing`。源数据里的 “Prefer not to answer” 原样保留，状态为 `present`。本次构建没有因缺人口学而删除任何人：Twin-2K 纳入 2,058，BDM 纳入 610，排除人数都是 0。BDM 全卡共有 383 个 `MISSING` 单元格，其中 `survey_q13_lack_of_resources` 空白 270 人。

## 抽样

全量人物卡都会写出。抽样是给后续模型实验用的子集，样本量可以改。

未指定样本量时，默认用比例的 Cochran 规划公式，并做有限总体校正：

- z = 1.96，p = 0.5，允许误差 e = 0.10
- n0 = ceil(z² p (1-p) / e²) = 97
- n = ceil(n0 N / (N + n0 - 1))

p = 0.5 是方差最大的规划值，不是从数据里估出来的购买率。e = 0.10 而不是 0.05，是因为 0.05 在 Twin-2K 上大约是 324 人；按 `experiments/cr-pricing-models-n50/README.md` 里每批约 40.8 秒、每人 40 道定价题、每批 15 题来算，324 人大约是 n = 50 那次运行的 6 倍以上。0.10 对应 Twin-2K 93 人、BDM 84 人，单模型的 Twin-2K 定价题大约三个多小时量级。这不是对某个价格效应的功效计算。要换误差或直接指定人数，改 `config/pricing_basic_info.json` 的 `margin_of_error` 或把 `sample_size` 设成整数。

种子是 42，与 `scripts/pricing_models_cr.py` 相同，但分配方法不同：按家庭收入比例分配，最大余数法，余数相同则按阶层名字升序。阶层内部用 `SHA-256(seed, dataset, stratum, pid)` 排序，不使用 `random.shuffle`。

分层变量：

- Twin-2K：`QID21`（家庭收入）
- BDM：`survey_q15_income`（空白单独成为 `MISSING` 层）

种子 42 的结果：

| 数据集 | 总体 | 样本 | 收入层总变差距离 | 收入层最大绝对比例差 |
| --- | --- | --- | --- | --- |
| twin2k_500 | 2058 | 93 | 0.00916957 | 0.00616007 |
| bdm_lettuce | 610 | 84 | 0.01740827 | 0.00585480 |

这只说明收入层的份额接近总体。其他字段没有被分层控制。Twin-2K 家庭规模 QID23 的总变差距离是 0.14279444，其中 “4” 的样本比例比总体低 0.08501834。BDM 年龄是原始岁数，水平很多，总变差距离是 0.32825917，但单个岁数的最大绝对比例差是 0.03329430。BDM 收入空白的 2 人、性别空白的 3 人、Intersex 的 2 人在这 84 人里名额为 0。分布文件只记录计数、比例和绝对偏差，没有显著性检验。

## 文件

提交进仓库的是基本资料卡、字段映射、样本、分布和哈希。定价结果留在 gitignore 的 `data/`，因为其中有每人不同的商品、价格和出价；`manifest.json` 记录了它们的 SHA-256。

```text
experiments/pricing-basic-info/
  field_map.json
  manifest.json
  source-lock.json
  twin2k_500/cards.jsonl
  twin2k_500/sample.json
  twin2k_500/distribution.json
  twin2k_500/exclusions.json
  twin2k_500/wave_audit.json
  bdm_lettuce/cards.jsonl
  bdm_lettuce/sample.json
  bdm_lettuce/distribution.json
  bdm_lettuce/exclusions.json
data/pricing_basic_info/          # 不提交
  raw/
  outcomes/twin2k_500/outcomes.jsonl
  outcomes/bdm_lettuce/outcomes.jsonl
```

## 复现

在仓库根目录，先安装项目（`datasets` 会带上读取 parquet 的依赖）：

```bash
pip install -e ".[dev]"
python scripts/build_pricing_basic_info.py
```

配置是 `config/pricing_basic_info.json`。相同的源文件、种子和配置会把人物卡和 `sample.json` 写成同样的字节。脚本在写盘前会把装配函数跑两遍并比较。源文件哈希与 `source-lock.json` 不一致时会停下来，不会换一份样本继续。

检查：

```bash
pytest tests/test_pricing_basic_info.py
```
