# 定价实验第一步：原文 basic info 人物卡与可复现抽样

这一步把 Twin-2K-500 定价任务和加拿大 BDM 生菜实验的原文基本资料抄成人物卡，并固定一个种子为 42 的样本。人物卡不调用语言模型，不补写生活史，也不把空单元格写成倾向。

`atus_expanded` 仍会生成生活史。那不是这一步的人物卡。本目录的卡片只有 `## Identity` 和 `## Basics`。

## 数据来源

| 数据集 | 总体 | 许可 | 版本 |
| --- | --- | --- | --- |
| Twin-2K-500 | 已发布的 2,058 人，全部纳入 | CC BY 4.0 | Hugging Face `LLM-Digital-Twin/Twin-2K-500`，revision `f883165a3026fde855dfd448e0cd16443ab257b6` |
| BDM lettuce | 已发布的 610 人，全部纳入 | CC BY 4.0 | Borealis [10.5683/SP3/CDLUEZ](https://doi.org/10.5683/SP3/CDLUEZ)，V1 |

Twin-2K 的 basic info 是 QID11–QID24，取自 `wave1_3_response_label.csv`。题干和选项从 `question_catalog.json` 原样写入 `source_question_text` 和 `source_options`。目录里句子不完整的题干保持原样。回答必须落在选项列表里。

定价结果是 wave 4 的 QID9_1–QID9_40。`QID9_n` 是该人看到的第 n 题，不是固定商品。商品名和价格在 `question_text` 里。Wave 1–3 的同名列是更早一次作答，不进入人物卡。2,058 人全部纳入。

BDM 的 basic info 是全部 `survey_*` 列。问卷 docx 只有图片，卡片用列名和单元格，不用 `fields.py` 的转述。`last_page`、计时、练习题、处理分配和呈现顺序被明确标成行政或设计字段，不进卡片。文件里多出来的未分类列会失败，不会被静默丢掉，也不会变成 `MISSING`。缺了已映射的列同样失败。空白单元格才写成 `MISSING`。源单元格如果本来就是 `MISSING` 这个词，构建失败，避免和空白哨兵混在一起。

## 抽样算法

全量卡片都会写出。样本人数仍是 Cochran 规划值：z=1.96，p=0.5，e=0.10，n0=97，再做有限总体校正。Twin-2K **93**，BDM **84**。种子 **42**。这不是对价格效应的功效计算，也不是说样本在统计上匹配总体。

精确步骤：

1. 对每一个分层变量单独做 Hamilton 最大余数分配。名额是 `floor(n * N_level / N)`，余数 `n * N_level mod N` 大的先加 1。余数相同则按水平名字升序。因此每个水平满足 `|样本人数 - n * N_level / N| < 1`。名额为 0 的水平不会出现在样本里。
2. Twin-2K 有两个分层变量：`QID21`（家庭收入）和 `QID23`（家庭规模）。用确定性增广路径填一张同时满足两条边际、且不超过每个交叉格人数的整数表，再用保持边际不变的单位换位把交叉格拉向 `n * N_cell / N`。BDM 只有 `survey_q15_income`，退化为第 1 步。
3. 格子内部按 `SHA-256(seed, dataset, cell, pid)` 取人。样本顺序按 `SHA-256(seed, dataset, "selected", pid)`。

种子 42 下，QID23 的 “4” 是 352/2058 对 16/93，配额 15.90670554，绝对差距 0.09329446。QID23 的总变差距离是 0.00525095。收入边际同样落在这个 1 人舍入界限内。

比例分配不保证覆盖稀有类别。BDM 收入空白 2 人的 Hamilton 名额是 0，性别里的 Intersex 2 人也是 0。这是分配规则的结果，不是漏抽后的补救。

BDM 的单岁年龄不是分层变量，单岁总变差距离 0.32825917 不作为失败。按 `floor(年龄/10)*10` 到该十年的描述性分组（空白仍是 `MISSING`），总变差距离是 0.11518345。这个分组写在 `reported_groupings` 里，角色是描述性分箱，不是设计分层。

### 未分层字段

下面的总变差距离只是计数差，不是匹配结论。

Twin-2K（分层变量是 QID21、QID23）：

| 字段 | 总变差距离 | 最大绝对比例差 |
| --- | --- | --- |
| QID11 | 0.06401455 | 0.03965642 |
| QID12 | 0.03034578 | 0.03034578 |
| QID13 | 0.07472021 | 0.06953196 |
| QID14 | 0.10771498 | 0.10522273 |
| QID15 | 0.07158532 | 0.07158532 |
| QID16 | 0.00880905 | 0.00880905 |
| QID17 | 0.07852911 | 0.06171040 |
| QID18 | 0.11009750 | 0.05426502 |
| QID19 | 0.06749428 | 0.04442146 |
| QID20 | 0.02067463 | 0.01507884 |
| QID22 | 0.09765196 | 0.05928086 |
| QID24 | 0.07570770 | 0.04688235 |

BDM（分层变量只有 `survey_q15_income`）：

| 字段 | 总变差距离 | 最大绝对比例差 |
| --- | --- | --- |
| survey_q01_age | 0.32825917 | 0.03329430 |
| survey_q02_sex | 0.07170180 | 0.06428571 |
| survey_q03_brand_name | 0.09047619 | 0.07267760 |
| survey_q03_locally_grown | 0.11561280 | 0.10327869 |
| survey_q03_pesticide_free | 0.08946136 | 0.08836846 |
| survey_q03_price | 0.09129586 | 0.04937549 |
| survey_q04_heard_of_vertical_farming | 0.01705699 | 0.01705699 |
| survey_q05_heard_of_greenhouse_growing | 0.04340359 | 0.04340359 |
| survey_q06_primary_grocery | 0.01100703 | 0.01100703 |
| survey_q07_primary_cook | 0.02447307 | 0.02447307 |
| survey_q08_leafy_green_meals | 0.08161593 | 0.07462920 |
| survey_q09_not_prepared_meals | 0.03602654 | 0.02985948 |
| survey_q10_try_novel_technology | 0.10499610 | 0.06440281 |
| survey_q11_grow_own_food | 0.06549571 | 0.06549571 |
| survey_q12_variety_concern | 0.05647931 | 0.04921936 |
| survey_q13_lack_of_resources | 0.21615925 | 0.03579235 |
| survey_q14_political | 0.06451991 | 0.03204528 |
| survey_q16_education | 0.11108509 | 0.04824356 |
| survey_q17_household_size | 0.11463700 | 0.04246682 |

## 文件、锁和校验范围

```text
experiments/pricing-basic-info/
  field_map.json
  manifest.json
  source-lock.json
  twin2k_500/cards.jsonl sample.json distribution.json exclusions.json wave_audit.json
  bdm_lettuce/cards.jsonl sample.json distribution.json exclusions.json
data/pricing_basic_info/          # 不提交
  raw/
  outcomes/
```

`manifest.json` 记录逻辑文件名和 SHA-256。它不含 `output_dir`、机器路径或时间戳。`manifest_sha256` 是去掉该字段之后正文的 SHA-256。哈希一致只说明这些字节没变，不说明未分层字段与总体一致。

`source-lock.json` 是提交进仓库的源文件钉。普通构建不会创建它，也不会在缓存哈希不符时改写它。非空缓存每次都重新计算哈希；对不上就失败，不换文件、不改锁。

`--init-source-lock` 是唯一会写出这把锁的命令，用来钉住一份新的上游快照。复现不要加这个参数。

首次克隆时锁已经在仓库里。普通命令会把缺的源文件下载到 `data/pricing_basic_info/raw/`，再按锁校验。

## 命令

```bash
pip install -e ".[dev]"
python scripts/build_pricing_basic_info.py
python scripts/build_pricing_basic_info.py --verify
pytest
```

`--verify` 只重算已提交产物的哈希，不下载、不重写卡片。改动任一字节都会失败。

直接依赖包括 `pyarrow`。测试收集 `run_pricing_models_cr.py` 需要 `python-dotenv`，它在 `dev` 额外依赖里。
