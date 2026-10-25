---
name: recent-experience-loop
description: >-
  Append one recent Jin10 flash that has already been stored to an existing
  persona card and re-answer BFI-44, iterating at most three rounds against
  the frozen 10.12 farthest and closest items. The paragraph is the flash
  plain text. Use when running the recent-experience pilot, editing recent.py
  round assignment, or deciding whether a round is accepted.
---

# Recent 经历多轮

在已有人物卡末尾加一节金十快讯原文，再送去答大五人格 44 题。不改 `data/visitors/by_population/adults/personas.jsonl`，不改 BFI 题面，不改 [simulate-persona-prompting](../simulate-persona-prompting/SKILL.md) 的过往经历槽。默认 `answer_survey` 不拼接；只有本循环的试跑才拼接。快讯来自 `data/recent/jin10.sqlite` 里免费、有主题的正文。

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

recent 是库里一条已经抓到的金十快讯。卡片上的段落就是这条快讯去掉标记后的正文。不另写中彩票、被迫开口、核爆炸，也不把快讯包进「此人当时在班次/家里」这类句子。

大事轮次的标题是 `## Recent event`。第 1–3 轮的标题仍是 `## Recent weeks`，那些轮次不回头改标题。第 1–3 轮所有人用最新一条免费快讯。第 4 轮起每人一条，由 `person id + 轮次` 对免费快讯条数取余，重跑不变。年龄段不改这段文字。

第 1–3 轮里旧的小事原文留在 `src/popbench/simulate/recent.py` 的 `RECENT_ROUNDS`，不拼进卡片。

保护题的后文不再手写。快讯原文是什么，卡片上就是什么。

## 人和请求

- 人：从 `runs/simulation/personas.jsonl` 里每个年龄段按原抽样顺序取前 8 人，共 32 人。
- 第 0 轮直接用已有 `runs/simulation/responses.jsonl`，不再请求。
- `system_message` 仍吃人物卡正文。试跑前把 recent 接在正文末尾。
- 第 1–3 轮所有人同一条最新快讯。第 4 轮起按该人的 id 与轮次抽取。
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
