# CR pricing-only multi-model benchmark (n=50)

Base-model comparison on pricing items. This is not a skill-vs-base run.
Persona method is `atus_expanded` (the CLI default and the base arm of
`scripts/run_skill_compare_cr.py`). `atus_expanded_skill` is not used.

`n` is the number of **subjects**, income-stratified with the same allocator
as `scripts/run_skill_compare_cr.py`. Seed is **42**.

## Model ids

Resolved from `cr_api.client.TEXT_MODELS` (also the installed `cr_api` package).
No log in this repo had called GLM, so the non-flash id listed first is used.

| Asked for | CR model id | Why this slug |
| --- | --- | --- |
| deepseek v4 pro | `deepseek-v4-pro` | First DeepSeek id. `benchmark` default for `--provider cr`. |
| gpt5.5 (GPT-5.5) | `gpt-5.5` | First GPT id. |
| glm | `glm-5.3` | First GLM id. Not `glm-5.3-flash` or `glm-5.2`. |

## Data

Read from `data/pricing_only/` (gitignored; do not commit the jsonl):

- `twin2k_500_pricing_subjects.jsonl` — `QID9_*` buy / not-buy (40 items per person)
- `bdm_lettuce_pricing_subjects.jsonl` — 6 WTP bids + `conclusion_purchased_product`

`twin2k_mega_study` is skipped. Its loader has no structured pricing outcome.

Sampled cases: 50 × 40 twin items and 50 × 7 bdm items, once, shared by all models.
Output layout:

```text
runs/cr-pricing-models-n50/
  progress.json
  manifest.json
  twin2k_sample.json
  bdm_sample.json
  <model-id>/twin2k_500/atus_expanded/
  <model-id>/bdm_lettuce/atus_expanded/
logs/cr-pricing-models-n50.log
```

`runs/` and `data/` stay gitignored. The log directory is local.

## Launch and resume

From the repo root, with the existing `.venv` and `CR_API_KEY` in `.env`:

```bash
mkdir -p logs
PYTHONUNBUFFERED=1 .venv/bin/python scripts/run_pricing_models_cr.py \
  > logs/cr-pricing-models-n50.log 2>&1
```

A finished job (non-empty `responses.jsonl` plus `manifest.json`) is skipped.
Each successful batch is cached under that job's `cache/`. A CR timeout is
retried 3 times inside `LLMClient` (provider timeout for this run is 600s).
If the batch still fails, the job is left incomplete and the next launch
retries it from the cache. Do not delete `cache/` if you want that resume.

Monitor:

```bash
tail -f logs/cr-pricing-models-n50.log
.venv/bin/python -c 'import json; print(json.dumps(json.load(open("runs/cr-pricing-models-n50/progress.json")), indent=2))'
```

`progress.json` includes `pid`, `status`, batch counts, and
`eta_seconds_if_remaining_uncached` after the first live batch.
`status: all_done` means every model finished both datasets.
`INCOMPLETE` in the log means re-run the same command.

## Score

Only after `ALL_DONE`. The script refuses to write `compare_models.json`
while any job is missing.

```bash
.venv/bin/python scripts/score_pricing_models_cr.py
```

Table columns: TVD (`mean_tvd`), Top-1 Accuracy (`mean_top1_acc`),
个体准确率（1−MAD/range） (`indiv_acc`), 跨人相关（Fisher z 平均）
(`fisher_z_corr`), Glass Δ (`glass_delta`), SD 比 (`sd_ratio`).

Exact-match `accuracy`, `n_scored`, and `parse_errors` are not in that table.
Per-job `manifest.json` still stores them under `operational` so a failed
parse can be spotted without treating them as the comparison.

## Timing note

Prior `atus_expanded` + `deepseek-v4-flash` income-50 run
(`runs/cr-skill-compare-income50`): about 40.8s per twin batch
(300 batches, 12225s) and about 10.7s per bdm batch (50 batches, 533.5s).
This experiment is 150 twin batches + 50 bdm batches per model (600 calls).
At that flash rate the floor is about 5.5 hours for three models.
`deepseek-v4-pro`, `gpt-5.5`, and `glm-5.3` do not have a local timing here;
expect this run to take longer than that floor. The log's ETA replaces the
floor once live batches return.
