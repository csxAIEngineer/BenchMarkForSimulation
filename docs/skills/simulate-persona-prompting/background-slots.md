# 过往经历槽位参考

`expand_persona` 按人物条件从池子里抽句，再拼成英文背景（问答模型用英文 system）。智能体手写扩写时也按同一约束。

## schooling ↔ education

| education | 允许写 | 禁止写 |
| --- | --- | --- |
| less_than_high_school | 辍学、GED、晚补课 | 大学学位 |
| high_school | 高中毕业、职校、入伍 | 学士及以上 |
| some_college | 社区大学、肄业、在读证书 | 已获学士（除非写「差一点」） |
| bachelor_or_higher | 学士/硕博、专业执照 | 从未上过高中 |

## work_path ↔ income / hours

- `income_over_50k=true` 且工时 ≥35：稳定全职、升职、技术/管理岗更合理
- `income_over_50k=false` 且工时低：兼职、零工、照护占白天更合理
- `work_hours` 缺失：写「工时不固定」或「阶段待业」，不要编精确时薪

## family ↔ marital_status

- Married → 可写配偶、共同开销；子女可选
- Never married → 不写现配偶；可写室友/伴侣同居需谨慎，默认不写
- Divorced / Separated / Widowed → 与状态一致的家庭结构

## turning_point ↔ ATUS 作息

转折应能解释人物卡上偏高的分钟数，例如：

- `care_household` 高 → 近期接手照护父母/孩子
- `work` 高 → 换岗加班或第二份工
- `leisure` 高 → 远程/失业后时间变松
- `traveling` 高 → 通勤变长或跨城接送

## System 拼装顺序

```
1. 角色指令（访问访谈、只选给定选项）
2. ## Basics（人口学一行表）
3. ## Life history（各槽短段落）
4. ## Typical day (ATUS minutes)（分钟列表）
```

不要把 life history 写进访问题；只进 system。
