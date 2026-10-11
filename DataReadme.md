# Twin-2K 按论文类别分类

主路径第 1 步使用这里的七类题，总览见 [README.md](README.md)。ATUS 人群划分在 [HumanReadme.md](HumanReadme.md)。

分类依据是 Twin-2K-500 论文和题库说明里已经写明的测量类别。类别由题库条目的 `BlockName` 决定，对应 [question_catalog_README](https://huggingface.co/datasets/LLM-Digital-Twin/Twin-2K-500/blob/main/question_catalog_and_human_response_csv/question_catalog_README.md)。题库 revision 为 `f883165`。同一个 `QuestionID` 可以出现在不同区块，类别按该条目的 `BlockName` 定。

`QuestionID` 是 Qualtrics 题块编号，题库共 256 条。展开后有两套计数：

- **闭卷作答题**共 654。矩阵行、单选、数字和滑杆各算一条可计分回答。说明屏（`DB`）和自由文本留在 CSV 里，不计作答题。
- **CSV 列**是前三波响应表的数据列，共 760，不含 `pid`。自由文本占列，多选的每个选项也各占一列。

题库说明里的「约 500 题」按另一套口径：人格和认知计到条目，经济偏好与实验把整张矩阵算成 1 题，被试间实验每人只计一个条件。

| 论文类别 | 文件夹 | QuestionID | 闭卷作答题 | 前三波列 | 第四波列 |
| --- | --- | ---: | ---: | ---: | ---: |
| 人口学 | `demographics` | 14 | 14 | 14 |  |
| 人格 | `personality` | 49 | 275 | 338 |  |
| 认知 | `cognitive` | 70 | 63 | 88 |  |
| 经济偏好 | `economic` | 38 | 176 | 194 |  |
| 非实验启发式 | `heuristics` | 5 | 20 | 20 | 20 |
| 行为经济学实验 | `experiments` | 39 | 66 | 66 | 66 |
| 定价 | `pricing` | 41 | 40 | 40 | 40 |
| 合计 |  | 256 | 654 | 760 | 126 |

认知 70 = `Cognitive tests` 69 + `Forward Flow` 1。经济偏好 38 = `Economic preferences` 36（含 2 段说明）+ `Economic preferences - intro` 的 2 段引言。题库说明写的是 34（+2 段引言），块内那 2 段说明（QID243、QID249）要另计，七类才凑满 256。定价 41 含 1 段说明（QID8）。

说明性 `DB` 题标成「说明」。人格、认知、经济偏好和行为经济学实验再按题干归到论文 Table 1 的量表或实验。对不上表的实验保留题干摘要。人口学、非实验启发式和定价按题目列出，不再另起量表名。

非实验启发式的 5 个题块展开成 20 道闭卷题：两张产品评价矩阵、一道流感题、一道弹珠盘题，以及 10 个政策滑杆。

## 人格

49 个 QuestionID，275 道闭卷作答题，338 列。自我问卷 3 道是自由文本。抑郁 21 道闭卷题里，贝克量表的多选每题拆成 4 个选项列，所以是 81 列。

| 量表 | QuestionID | 闭卷作答题 | CSV 列 |
| --- | ---: | ---: | ---: |
| 说明 | 5 | 0 | 0 |
| Big Five | 1 | 44 | 44 |
| Need for cognition | 1 | 18 | 18 |
| Agentic vs. communal values | 1 | 24 | 24 |
| Consumer minimalism | 1 | 12 | 12 |
| Empathy | 1 | 20 | 20 |
| Green values | 1 | 6 | 6 |
| Social desirability | 1 | 13 | 13 |
| Conscientiousness | 1 | 8 | 8 |
| Anxiety | 1 | 21 | 21 |
| Individualism vs. collectivism | 1 | 16 | 16 |
| Selves questionnaire | 3 | 0 | 3 |
| Regulatory focus | 1 | 10 | 10 |
| Tightwads vs. spendthrift | 4 | 4 | 4 |
| Depression | 21 | 21 | 81 |
| Need for uniqueness | 1 | 12 | 12 |
| Self-monitoring | 1 | 13 | 13 |
| Self-concept clarity | 1 | 12 | 12 |
| Need for closure | 1 | 15 | 15 |
| Maximization | 1 | 6 | 6 |

## 认知

70 个 QuestionID，63 道闭卷作答题，88 列。Forward Flow（QID10）的 20 个词槽是自由文本。CRT 的 QID52、金融素养的信用卡题 QID43 也是开放文本。闭卷 63 道加上这 2 道和 20 个词槽，对应题库说明里的 85 题（65 + 20）。Wason selection 是一道多选，占 4 列。

| 量表或任务 | QuestionID | 闭卷作答题 | CSV 列 |
| --- | ---: | ---: | ---: |
| 说明 | 4 | 0 | 0 |
| CRT | 4 | 3 | 4 |
| Fluid intelligence | 6 | 6 | 6 |
| Crystallized intelligence | 20 | 20 | 20 |
| Syllogisms | 12 | 12 | 12 |
| Overconfidence | 1 | 1 | 1 |
| Overplacement | 1 | 1 | 1 |
| Financial literacy | 8 | 7 | 8 |
| Numeracy | 8 | 8 | 8 |
| Deductive certainty | 4 | 4 | 4 |
| Forward Flow | 1 | 0 | 20 |
| Wason selection | 1 | 1 | 4 |

## 经济偏好

38 个 QuestionID，176 道闭卷作答题，194 列。折扣、现时偏好、风险规避和损失规避的作答题数是矩阵行数。三道想法列举是自由文本，各 6 列。去掉 4 段说明后，剩下 34 个题块，与题库说明里「经济偏好 34 题」一致；说明把每张矩阵算成 1 题。

| 任务 | QuestionID | 闭卷作答题 | CSV 列 |
| --- | ---: | ---: | ---: |
| 说明 | 4 | 0 | 0 |
| Ultimatum game (sender) | 1 | 1 | 1 |
| Ultimatum game (receiver) | 6 | 6 | 6 |
| Mental accounting | 4 | 4 | 4 |
| Discount | 3 | 30 | 30 |
| Present bias | 3 | 30 | 30 |
| Risk aversion | 3 | 42 | 42 |
| Loss aversion | 4 | 56 | 56 |
| Trust game (sender) | 1 | 1 | 1 |
| Trust game (receiver) | 5 | 5 | 5 |
| Trust game (sender) thought listing | 1 | 0 | 6 |
| Trust game (receiver) thought listing | 1 | 0 | 6 |
| Dictator game | 1 | 1 | 1 |
| Dictator game thought listing | 1 | 0 | 6 |

## 行为经济学实验

39 个 QuestionID，66 道闭卷作答题，66 列。每个被试间条件是单独的 `BlockName`，响应表里各条件的列都保留。虚假共识、Linda 和概率匹配的作答题数是矩阵行数。Proportion dominance 在 Table 1 里没有单独的任务名，1A–1C 与 2A–2C 各自按题干归成一组。

| 实验 | QuestionID | 闭卷作答题 | CSV 列 |
| --- | ---: | ---: | ---: |
| False consensus | 1 | 10 | 10 |
| Base rate | 2 | 2 | 2 |
| Framing | 2 | 2 | 2 |
| Conjunction (Linda) | 2 | 6 | 6 |
| Outcome bias | 2 | 2 | 2 |
| Anchoring | 8 | 8 | 8 |
| Less is more | 3 | 3 | 3 |
| Proportion dominance 1 | 3 | 3 | 3 |
| Proportion dominance 2 | 3 | 3 | 3 |
| Sunk cost | 2 | 2 | 2 |
| Absolute vs. relative savings | 2 | 2 | 2 |
| WTA/WTP | 3 | 3 | 3 |
| Allais problem | 2 | 2 | 2 |
| Myside bias | 2 | 2 | 2 |
| Probability matching | 2 | 16 | 16 |

## 按类别切开的数据

`popbench split-categories` 把题库和回答表按上表七类写到 `data/twin2k/by_category/`。每一类一个文件夹，里面是该类的 `question_catalog.json`、`wave1_3_response.csv` 和 `wave1_3_response_label.csv`。第四波只重复了非实验启发式、行为经济学实验和定价，所以只有这三个文件夹有 `wave4_response.csv` 和 `wave4_response_label.csv`。每个回答表都保留 `pid`。760 列和 126 列分别覆盖前三波与第四波的全部数据列，列与列之间不重叠。
