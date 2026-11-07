# 人物卡预处理

这一步把 Twin-2K-500 定价任务和加拿大 BDM 生菜实验的原文基本资料抄成人物卡，并固定种子 42 的样本。人物卡不调用语言模型，不补写生活史，也不把空单元格写成倾向。卡片只有 `## Identity` 和 `## Basics`。

本目录只提交 `source-lock.json`。卡片、样本、分布、排除名单、字段图、wave 审计和 manifest 由构建写到 `outputs/prepro/`，原始文件和留出答案写到 `data/prepro/`。这些产物不入库。

## 数据来源

| 数据集 | 总体 | 许可 | 版本 |
| --- | --- | --- | --- |
| Twin-2K-500 | 已发布的 2,058 人，全部纳入 | CC BY 4.0 | Hugging Face `LLM-Digital-Twin/Twin-2K-500`，revision `f883165a3026fde855dfd448e0cd16443ab257b6` |
| BDM lettuce | 已发布的 610 人，全部纳入 | CC BY 4.0 | Borealis [10.5683/SP3/CDLUEZ](https://doi.org/10.5683/SP3/CDLUEZ)，V1 |

Twin-2K 的 basic info 是 QID11–QID24，取自 `wave1_3_response_label.csv`。题干和选项从 `question_catalog.json` 原样写入 `source_question_text` 和 `source_options`。目录里句子不完整的题干保持原样。回答必须落在选项列表里。

这 14 项是：地区、性别、年龄段、学历、种族、公民、婚姻、宗教、礼拜频率、党派、家庭收入、政治倾向、家庭人数、就业，依次对应 QID11–QID24。`basic_info` 里每一项有中文 `label`、`qid`、`source_question_text`、`value`、`status`，以及 `evidence`（指向该 QID 和源列）。给模型看的 `text` 不写这些题号，也不写 evidence。Twin-2K 的一行是中文标签、目录原文题干、单元格。截断的题干保持原样。例如题干停在 `In politics today, do you consider yourself a` 时，行是 `党派: In politics today, do you consider yourself a: Independent`。

定价结果是 wave 4 的 QID9_1–QID9_40。`QID9_n` 是该人看到的第 n 题，不是固定商品。商品名和价格在 `question_text` 里。Wave 1–3 的同名列是更早一次作答，不进入人物卡。

BDM 不是这 14 项的同一份问卷，不把列改写成 Twin 的题号。问卷 docx 只有图片，没有可抄的题干。性别、政治倾向、家庭收入、学历、家庭人数和 Twin 是同一类事实、不同工具，`text` 只用中文标签和单元格，`evidence` 仍指向 `survey_*` 列，`qid` 为空。`survey_q01_age` 是实足年龄，不是年龄段，不标成年龄段。其余购物习惯列对不上这 14 项，`text` 保留列名，例如 `survey_q03_price: ...`，并在 `basic_info` 标明无法对应。不编造问句。`last_page`、计时、练习题、处理分配和呈现顺序是行政或设计字段，不进卡片。文件里多出来的未分类列会失败，不会被静默丢掉，也不会变成 `MISSING`。缺了已映射的列同样失败。空白单元格才写成 `MISSING`。源单元格如果本来就是 `MISSING` 这个词，构建失败。

## 抽样

全量卡片都会写到本地输出目录。样本人数是 Cochran 规划值：z=1.96，p=0.5，e=0.10，n0=97，再做有限总体校正。Twin-2K 为 93，BDM 为 84。种子是 42。这不是对价格效应的功效计算。

1. 对每一个分层变量单独做 Hamilton 最大余数分配。名额是 `floor(n * N_level / N)`，余数大的先加 1。余数相同则按水平名字升序。名额为 0 的水平不会出现在样本里。
2. Twin-2K 同时锁定 `QID21`（家庭收入）和 `QID23`（家庭规模）。先得到一张满足两条 Hamilton 边际、且不超过交叉格人数的整数表。然后每次只做一次 2×2 换人：正向和反向都检查，只接受让交叉格到目标配额的 L1 距离严格下降的可行换位。四格互不相同，两条边际在每一步前后都不变。BDM 只有 `survey_q15_income`。
3. 格子内部按 `SHA-256(seed, dataset, cell, pid)` 取人。样本顺序按 `SHA-256(seed, dataset, "selected", pid)`。

联合换位不是精确配额匹配，也不调整未分层字段。比例分配不保证覆盖稀有类别，名额为 0 是分配规则的结果。BDM 单岁年龄不是分层变量。按十年分组只写在本地 `distribution.json` 的 `reported_groupings` 里，是描述性分箱。

计数差写在本地 `distribution.json`，不作为仓库里的报告。

## 锁

`source-lock.json` 钉住上游文件的字节和 SHA-256。普通构建不会创建它，也不会在缓存哈希不符时改写它。非空缓存每次都重新计算哈希；对不上就失败。

`--init-source-lock` 是唯一会写出这把锁的命令。复现不要加这个参数。

首次克隆时锁已经在仓库里。普通命令把缺的源文件下载到 `data/prepro/raw/`，再按锁校验。

`outputs/prepro/manifest.json` 记录本地逻辑文件名和 SHA-256。它不含 `output_dir`、机器路径或时间戳。`--verify` 重算这些哈希。交付目录里未登记的文件，或已存在但被改过的 outcomes，都会失败。outcomes 尚未生成时，校验通过并标明跳过。

## 命令

```bash
pip install -e ".[dev]"
python3 scripts/build_prepro.py
python3 scripts/build_prepro.py --verify
```

直接依赖包括 `pyarrow`、`huggingface_hub` 和 `httpx`。
