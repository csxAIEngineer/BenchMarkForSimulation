---
name: recent-experience-loop
description: >-
  Append one recent event that has already happened to an existing persona
  card and re-answer BFI-44, iterating at most three rounds against the frozen
  10.12 farthest and closest items. The event pool includes a lottery payout
  and a distant nuclear explosion. Use when running the recent-experience
  pilot, editing recent.py round text, or deciding whether a round is accepted.
---

# Recent 经历多轮

在已有人物卡末尾加一节已经发生的大事，再送去答大五人格 44 题。不改 `data/visitors/by_population/adults/personas.jsonl`，不改 BFI 题面，不改 [simulate-persona-prompting](../simulate-persona-prompting/SKILL.md) 的过往经历槽。默认 `answer_survey` 不拼接；只有本循环的试跑才拼接。

基线是 10.12 的 BFI-44（`deepseek-flash`，每年龄段 50 人）。最远 / 最近各 8 题以 `reports/10.12-report/raw/significance.md` 为准。整段循环不更换这 16 个题号。对照是没有 recent 的第 0 轮，不是上一轮。

## 冻结的 16 题

均分是原始同意程度 1–5（1 很不同意，5 很同意），不是反向计分后的量表分。括号里是基线模拟均分 → Twin-2K 全样本均分。

### 最远 8 题

| 题号 | 短名 | 这一轮 |
| --- | --- | --- |
| `QID25_39` | 容易紧张 | 往下拉（3.73→2.92） |
| `QID25_21` | 偏安静 | 往下拉（3.86→3.53） |
| `QID25_19` | 常担心 | 往下拉（3.77→3.19） |
| `QID25_27` | 冷淡疏离 | 往上抬（2.21→2.61） |
| `QID25_35` | 喜欢常规 | 往下拉（3.84→3.40） |
| `QID25_6` | 矜持 | 往下拉（3.87→3.62） |
| `QID25_41` | 艺术兴趣少 | 轻轻下拉（2.56→2.35） |
| `QID25_18` | 杂乱 | 保护。均分已经对齐（2.36≈2.39），不准把它推开。它仍算在最远 8 题的平均 TVD 里 |

### 最近 8 题（全部保护）

彻底 `QID25_3`、可靠 `QID25_13`、有效率 `QID25_33`、少争吵 `QID25_12`、体贴 `QID25_32`、坚持 `QID25_28`、热情 `QID25_16`、精力 `QID25_11`。

这些题的模拟均分已经在人类附近，或比人类更极端。recent 里不写把它们再推极端的事。大事的后文不准写成“因此更认真、更可靠、更体贴、更有精力”。

## 句子怎么写

recent 是一件已经发生、会改掉这个人接下来怎么过的大事。中彩票、核爆炸这类小概率事件留在池子里，不因为罕见就删掉。不写新的学历、职业、婚姻，不写「我是一个紧张/外向的人」这类特质标签。

大事轮次的标题是 `## Recent event`，避免模型把它当成又一周杂事。第 1–3 轮已经用 `## Recent weeks` 答完，那些轮次不回头改。

年龄段只改事发时的所在，不改事件有多大。

| 年龄段 | 所在 |
| --- | --- |
| 18-29 | 班次 |
| 30-49 | 家里 |
| 50-64 | 单位 |
| 65+ | 住处 |

同一年龄段不再共用一段话。每人一件事，由 `person id + 轮次` 确定性抽到，重跑不变。

第 1–3 轮是已拒绝的小事，原文留在 `src/popbench/simulate/recent.py` 的 `RECENT_ROUNDS`。新轮次从第 4 轮起，写在同一文件的按人抽取里，不回头改旧轮。

第 4 轮池子，每个年龄段 8 人里的人数：

- 5 人：中彩票或同等突然到账。过去两周里钱已经到手，原来的日程停了，亲戚或陌生人打来，此人只接了其中一次并说了几句，另外买了一件平时不会买的非日用品（票、课、短途）。不写投资、预算、帮助别人。
- 2 人：被迫公开说一段话（被点名、本地新闻找上门、台上要此人开口）。说完就离开，不写因此变得热情或精力旺盛。
- 1 人：远处核爆炸或同等已经结束的爆炸。此人在爆心之外，危险已经过去。用它把一部分人推向紧张的那一端，同时给疏离一个由头（少停下来交谈）。不写成仍在逃难，也不写成因此把生活重新安排得很有条理。

核爆炸会把紧张和担心往上推，而这两题的模拟均分已经高于 Twin-2K。所以它只分给少数人，用来撑开尾部，不作为全年龄段的同一件事。中彩票一类的横财分给多数人：常规被打断，有人来找所以得开口，钱上的担心卸掉。后文不写理财、捐赠或帮忙。艺术兴趣少只轻轻动。

第 5、6 轮只改这个配比和后文：紧张均分被爆炸抬离人类，就减少爆炸人数；安静和矜持没降，就增加被迫开口；保护题的均分被推离人类，就删掉对应后文。最多再跑 3 轮。

## 人和请求

- 人：从 `runs/simulation/personas.jsonl` 里每个年龄段按原抽样顺序取前 8 人，共 32 人。
- 第 0 轮直接用已有 `runs/simulation/responses.jsonl`，不再请求。
- `system_message` 仍吃人物卡正文。试跑前把 recent 接在正文末尾。
- 第 1–3 轮同一年龄段同一段话。第 4 轮起按该人的 id 与轮次抽取。
- 每轮用新目录 `runs/simulation/recent-pilot/round-N`。cache 键不含 system 文本，不能复用基线 cache。
- 人类份额仍用 Twin-2K 全样本（`runs/simulation/human_baseline.json`）。
- 作答仍是整份 BFI-44。评分只看冻结的 16 题。

## 接受

相对第 0 轮，三个条件都要成立：

1. 最远 8 题平均 TVD 至少降 0.02。
2. 最近 8 题平均 TVD 上升不超过 0.02。
3. 最近 8 题均分绝对差上升不超过 0.05。

停在接受的那一轮。某一轮没有句子，或这一串 3 轮都拒绝，就停。

## 结果

写到 `reports/10.12-report/recent-pilot/`：每轮 recent 原文、16 题 TVD、是否接受。

试点接受之后，用该轮的抽取规则重答原来的 200 人，对照这 200 人的第 0 轮答案，仍只评这 16 题。结果放在同一目录。没有接受的轮次不去答这 200 人。
