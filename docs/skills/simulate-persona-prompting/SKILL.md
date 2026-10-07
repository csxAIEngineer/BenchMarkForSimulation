---
name: simulate-persona-prompting
description: >-
  Persona prompt engineering for popbench simulate: expand ATUS or ACS base
  demographics into a seeded life history, then send the full persona card
  to the LLM. Use when running or changing simulate, compare, visitors, 人设,
  persona prompt, background story, or interview questions.
---

# Simulate 人设提示工程

模拟时**先扩人设、再问答**。不要把裸人口学直接丢给大模型。

## 硬规则

1. 输入只信基本人物条件：年龄、性别、州、学历、婚姻、种族、收入档、工时、ATUS 作息分钟。
2. 用 `visitor_id + run seed` 做确定性随机，生成**过往经历**（童年/求学/工作/家庭/迁徙/转折），禁止每次重跑换故事。
3. 经历必须与基本条件自洽（学历、年龄、婚姻、州、收入）；禁止编造与条件冲突的身份。
4. 把「基本条件 + 过往经历 + 作息」合成 system persona，再送访问题目。
5. 问答阶段只扮演该人；不解释提示工程，不跳出角色。

## 工作流

```
基本条件 (ATUS 受访者，或 ACS 访问者)
    → 人设扩写 (seeded random background)
    → system prompt
    → 多轮选择题
    → responses.jsonl
```

`popbench compare` 走 ATUS 人物卡，题目是 6 道 Twin-2K 选择题。访问者命令仍走 8 道时间利用题。

代码入口：

- 扩写：`popbench.simulate.persona_background.expand_persona`
- ATUS 人物卡 + Twin-2K 题：`popbench.dao.atus_twin2k.build_atus_twin2k`，作答 `popbench.simulate.interview.run_records`
- CLI：`popbench compare --n 50 --seed 0`
- 访问模拟：`popbench.simulate.visitors.run_visitors`

改人设逻辑时优先改 `persona_background.py`，不要在 CLI 里临时拼 prompt。

## 过往经历槽位

每次扩写填满这些槽（可短句，2–4 句/槽）：

| 槽 | 约束 |
| --- | --- |
| childhood | 与年龄、种族、可能的迁徙自洽 |
| schooling | 必须匹配 `education` 档 |
| work_path | 匹配工时、收入档；失业/兼职要说得通 |
| family | 匹配 `marital_status`；有子女才写育儿负担 |
| moves | 当前 `state` 为现居地；可写如何到此 |
| turning_point | 1 个与作息相关的具体转折（加班、照护、失业、搬家） |

槽位文案与示例见 [background-slots.md](background-slots.md)。

## 检查清单

改 simulate / 人设相关代码或跑访问前：

- [ ] 背景由 `expand_persona`（或同等 seeded 逻辑）生成，不是模型当场自由发挥人口学
- [ ] 同一 `seed + visitor_id` 重跑背景不变
- [ ] system 里同时有基本条件、过往经历、ATUS 作息
- [ ] 响应里保存了 `background` 便于审计
- [ ] 人设扩写不改题目文本。`compare` 用固定的 6 道 Twin-2K 选择题；访问访谈仍用 `VISIT_TURNS`

## 反模式

- 只贴 Age/Sex/State 就开问
- 用温度随机让模型自己编完整人生且不落盘
- 给高中学历写博士经历，或未婚却写配偶日常
- 把访问题改成开放闲聊却仍按五选一解析
