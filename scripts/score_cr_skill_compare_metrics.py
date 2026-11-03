#!/usr/bin/env python3
"""Re-score cr-skill-compare runs with personalization + distribution metrics.

Reads existing responses.jsonl only. Does not call any model API.

Reported metrics (no classic exact-match accuracy / n_scored / parse_errors in the
compare table):

1) individual accuracy = 1 - MAD/range   (popbench ordinal_accuracy / individual_accuracy)
2) cross-person correlation = Fisher-z mean of per-item Pearson r
3) Glass Δ = (mean_twin - mean_human) / sd_human   (per-item, then unweighted mean)
4) SD ratio = sd_twin / sd_human                   (per-item, then unweighted mean)
5) mean_tvd = mean over items of 0.5 Σ|p_i-q_i|

Categorical encoding (project convention / Twin-2K CSV):
- Closed-ended options: 1-based index in option/column order (span = n_options - 1).
- Numeric / slider: float value; span from catalog Range when present, else
  max(observed human∪model) - min(...) for that item (skipped if span <= 0).
- BDM WTP: CAD dollars; purchase TRUE/FALSE -> 1/0 (span 1 or 10 for WTP).
- Mega: prefer SelectedByPosition; Values as float; SelectedText mapped via Options.
  Matrix lists expand to study:Qk:i when both sides match in length.
"""

from __future__ import annotations

import json
import math
import re
import sys
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))

from popbench.evaluate.metrics import (  # noqa: E402
    fisher_z_mean_correlation,
    glass_delta,
    individual_accuracy,
    pearson,
    population_std,
    sd_ratio,
    total_variation_distance,
)
import score_cr_skill_compare_tvd as tvdmod  # noqa: E402

RUN = ROOT / "runs" / "cr-skill-compare-income50"
METHODS = ("atus_expanded", "atus_expanded_skill")
DATASETS = ("twin2k_500", "bdm_lettuce", "twin2k_mega_study")
CATALOG_PATH = (
    ROOT
    / "data"
    / "twin2k"
    / "question_catalog_and_human_response_csv"
    / "question_catalog.json"
)
SUBJECTS_PATH = ROOT / "data" / "twin2k_500" / "subjects.jsonl"


def _casefold(text: str) -> str:
    return text.strip().casefold()


def load_twin2k_scales() -> dict[str, dict[str, Any]]:
    """Map question_id -> {options: [...], span: float|None, kind: str}."""
    scales: dict[str, dict[str, Any]] = {}
    if SUBJECTS_PATH.is_file():
        with SUBJECTS_PATH.open(encoding="utf-8") as handle:
            for line in handle:
                row = json.loads(line)
                for question in row.get("questions") or []:
                    qid = str(question.get("id") or "")
                    if not qid or qid in scales:
                        continue
                    options = [str(item) for item in (question.get("options") or [])]
                    kind = str(question.get("kind") or "")
                    span = float(len(options) - 1) if len(options) >= 2 else None
                    scales[qid] = {"options": options, "span": span, "kind": kind}
    catalog: list[dict[str, Any]] = []
    if CATALOG_PATH.is_file():
        catalog = json.loads(CATALOG_PATH.read_text(encoding="utf-8"))
    by_id = {str(item.get("QuestionID")): item for item in catalog}
    # Fill catalog Options/Columns/Range for base QIDs and matrix rows QID:n
    for qid, item in list(by_id.items()):
        options = item.get("Options") or item.get("Columns") or []
        if not isinstance(options, list):
            options = []
        options = [str(x) for x in options if str(x).strip()]
        range_info = item.get("Range") if isinstance(item.get("Range"), dict) else None
        span = None
        if len(options) >= 2:
            span = float(len(options) - 1)
        elif range_info and range_info.get("Max") is not None and range_info.get("Min") is not None:
            span = float(range_info["Max"]) - float(range_info["Min"])
        if qid not in scales:
            scales[qid] = {
                "options": options,
                "span": span,
                "kind": str(item.get("QuestionType") or ""),
                "range": range_info,
            }
        else:
            if not scales[qid]["options"] and options:
                scales[qid]["options"] = options
                scales[qid]["span"] = span
            if scales[qid].get("span") is None and span is not None:
                scales[qid]["span"] = span
            if range_info:
                scales[qid]["range"] = range_info
        # Matrix rows stored as QID:index in this repo
        rows = item.get("Rows") or []
        if isinstance(rows, list) and rows and options:
            for index in range(1, len(rows) + 1):
                child = f"{qid}:{index}"
                if child not in scales:
                    scales[child] = {
                        "options": options,
                        "span": float(len(options) - 1),
                        "kind": "likert",
                    }
    return scales


TWIN2K_SCALES = load_twin2k_scales()


def parse_number(value: Any) -> float | None:
    if value is None:
        return None
    if isinstance(value, (int, float)) and not isinstance(value, bool):
        if math.isnan(float(value)) or math.isinf(float(value)):
            return None
        return float(value)
    text = str(value).strip()
    if not text:
        return None
    folded = text.casefold()
    if folded in {"true", "yes"}:
        return 1.0
    if folded in {"false", "no"}:
        return 0.0
    text = text.replace(",", "")
    if text.endswith("%"):
        text = text[:-1].strip()
    try:
        number = float(text)
    except ValueError:
        return None
    if math.isnan(number) or math.isinf(number):
        return None
    return number


def encode_with_options(label: Any, options: list[str]) -> float | None:
    if label is None:
        return None
    text = str(label).strip()
    if not text or not options:
        return None
    lookup = {_casefold(opt): float(index) for index, opt in enumerate(options, start=1)}
    hit = lookup.get(_casefold(text))
    if hit is not None:
        return hit
    # tolerate model paraphrases that still start/end with an option
    folded = _casefold(text)
    for opt, code in lookup.items():
        if folded == opt or folded.startswith(opt) or opt.startswith(folded):
            return code
    return None


def shares_from_labels(labels: list[str], vocabulary: list[str]) -> list[float]:
    counts = Counter(labels)
    n = len(labels)
    return [counts[label] / n for label in vocabulary]


def mean_tvd_by_item(pairs_by_item: dict[str, list[tuple[str, str]]]) -> dict[str, Any]:
    tvds: list[float] = []
    for item_id in sorted(pairs_by_item):
        pairs = pairs_by_item[item_id]
        human = [h for h, _ in pairs]
        model = [m for _, m in pairs]
        vocab = sorted(set(human) | set(model))
        if not vocab:
            continue
        tvds.append(
            total_variation_distance(
                shares_from_labels(model, vocab),
                shares_from_labels(human, vocab),
            )
        )
    return {
        "mean_tvd": (sum(tvds) / len(tvds)) if tvds else None,
        "n_items_tvd": len(tvds),
    }


def aggregate_coded(
    codes_by_item: dict[str, list[tuple[float, float]]],
    spans_by_item: dict[str, float],
) -> dict[str, Any]:
    """codes_by_item: item -> list of (human_code, model_code)."""
    indiv_scores: list[float] = []
    item_corrs: list[float] = []
    item_glass: list[float] = []
    item_sd: list[float] = []
    item_rows: list[dict[str, Any]] = []
    skipped_span = 0
    skipped_corr = 0

    for item_id in sorted(codes_by_item):
        pairs = codes_by_item[item_id]
        if not pairs:
            continue
        human = [h for h, _ in pairs]
        model = [m for _, m in pairs]
        span = spans_by_item.get(item_id)
        if span is None or span <= 0:
            # fall back to observed range on this item
            lo = min(human + model)
            hi = max(human + model)
            span = hi - lo
        if span <= 0:
            skipped_span += 1
            continue
        spans_by_item[item_id] = float(span)
        acc = individual_accuracy(model, human, span=float(span))
        indiv_scores.append(acc)

        corr = None
        if len(pairs) >= 2:
            corr = pearson(human, model)
            if corr is None:
                skipped_corr += 1
            else:
                item_corrs.append(corr)

        g = glass_delta(model, human)
        if g is not None:
            item_glass.append(g)
        s = sd_ratio(model, human)
        if s is not None:
            item_sd.append(s)

        item_rows.append(
            {
                "item_id": item_id,
                "n": len(pairs),
                "span": float(span),
                "indiv_acc": acc,
                "pearson": corr,
                "glass_delta": g,
                "sd_ratio": s,
                "human_mean": sum(human) / len(human),
                "model_mean": sum(model) / len(model),
                "human_sd": population_std(human),
                "model_sd": population_std(model),
            }
        )

    return {
        "indiv_acc": (sum(indiv_scores) / len(indiv_scores)) if indiv_scores else None,
        "fisher_z_corr": fisher_z_mean_correlation(item_corrs),
        "glass_delta": (sum(item_glass) / len(item_glass)) if item_glass else None,
        "sd_ratio": (sum(item_sd) / len(item_sd)) if item_sd else None,
        "n_items_coded": len(indiv_scores),
        "n_items_corr": len(item_corrs),
        "n_items_glass": len(item_glass),
        "n_items_sd": len(item_sd),
        "n_pairs_coded": sum(len(v) for v in codes_by_item.values()),
        "skipped_zero_span": skipped_span,
        "skipped_zero_var_corr": skipped_corr,
        "items": item_rows,
    }


def encode_twin2k_pair(question_id: str, human_raw: Any, model_raw: Any) -> list[tuple[str, float, float, float]]:
    """Return list of (item_id, human_code, model_code, span)."""
    scale = TWIN2K_SCALES.get(question_id) or TWIN2K_SCALES.get(question_id.split(":")[0]) or {}
    options = list(scale.get("options") or [])
    span = scale.get("span")

    # Multi-slider / comma lists (e.g. QID290)
    if (
        isinstance(human_raw, str)
        and "," in human_raw
        and isinstance(model_raw, str)
        and "," in str(model_raw)
    ):
        h_parts = [p.strip() for p in human_raw.split(",")]
        m_parts = [p.strip() for p in str(model_raw).split(",")]
        if len(h_parts) == len(m_parts) and len(h_parts) > 1:
            out = []
            child_span = span
            if child_span is None:
                rng = scale.get("range") or {}
                if rng.get("Max") is not None and rng.get("Min") is not None:
                    child_span = float(rng["Max"]) - float(rng["Min"])
            if child_span is None:
                child_span = 100.0
            for index, (h, m) in enumerate(zip(h_parts, m_parts), start=1):
                hc = parse_number(h)
                mc = parse_number(m)
                if hc is None or mc is None:
                    continue
                out.append((f"{question_id}:{index}", hc, mc, float(child_span)))
            return out

    if options:
        hc = encode_with_options(human_raw, options)
        mc = encode_with_options(model_raw, options)
        if hc is not None and mc is not None:
            if span is None:
                span = float(len(options) - 1)
            return [(question_id, hc, mc, float(span))]

    hc = parse_number(human_raw)
    mc = parse_number(model_raw)
    if hc is None or mc is None:
        return []
    if span is None:
        rng = scale.get("range") or {}
        if rng.get("Max") is not None and rng.get("Min") is not None:
            span = float(rng["Max"]) - float(rng["Min"])
    # BDM-like open numeric without catalog range: span filled later from observed
    return [(question_id, hc, mc, float(span) if span is not None else -1.0)]


def score_twin2k(rows: list[dict[str, Any]]) -> dict[str, Any]:
    label_pairs: dict[str, list[tuple[str, str]]] = defaultdict(list)
    codes_by_item: dict[str, list[tuple[float, float]]] = defaultdict(list)
    spans_by_item: dict[str, float] = {}
    encode_fail = 0
    for row in rows:
        gold = row.get("gold") or {}
        human_raw = gold.get("answer")
        model_raw = (row.get("parsed") or {}).get("answer")
        h_lab = tvdmod.norm_label(human_raw)
        m_lab = tvdmod.norm_label(model_raw)
        qid = str(row.get("question_id") or "")
        if h_lab is not None and m_lab is not None:
            label_pairs[qid].append((h_lab, m_lab))
        encoded = encode_twin2k_pair(qid, human_raw, model_raw)
        if not encoded:
            if h_lab is not None and m_lab is not None:
                encode_fail += 1
            continue
        for item_id, hc, mc, span in encoded:
            codes_by_item[item_id].append((hc, mc))
            if span is not None and span > 0:
                spans_by_item[item_id] = float(span)
    # Fill missing spans from observed codes
    for item_id, pairs in codes_by_item.items():
        if item_id in spans_by_item and spans_by_item[item_id] > 0:
            continue
        vals = [h for h, _ in pairs] + [m for _, m in pairs]
        lo, hi = min(vals), max(vals)
        if hi > lo:
            spans_by_item[item_id] = hi - lo
    coded = aggregate_coded(codes_by_item, spans_by_item)
    tvd = mean_tvd_by_item(label_pairs)
    return {
        **{k: v for k, v in coded.items() if k != "items"},
        **tvd,
        "encode_failures_among_labeled": encode_fail,
        "encoding": (
            "likert/MC: 1-based option index from Twin-2K subjects.jsonl / catalog; "
            "slider/TE numeric: float; QID290 comma lists expanded to :1..n; "
            "span = n_options-1 or catalog Range or observed max-min"
        ),
        "items": coded["items"],
    }


def score_bdm(rows: list[dict[str, Any]]) -> dict[str, Any]:
    label_pairs: dict[str, list[tuple[str, str]]] = defaultdict(list)
    codes_by_item: dict[str, list[tuple[float, float]]] = defaultdict(list)
    spans_by_item: dict[str, float] = {}
    encode_fail = 0
    for row in rows:
        gold = row.get("gold") or {}
        human_raw = gold.get("answer")
        model_raw = (row.get("parsed") or {}).get("answer")
        qid = str(row.get("question_id") or "")
        h_lab = tvdmod.norm_label(human_raw)
        m_lab = tvdmod.norm_label(model_raw)
        if h_lab is not None and m_lab is not None:
            label_pairs[qid].append((h_lab, m_lab))
        hc = parse_number(human_raw)
        mc = parse_number(model_raw)
        if hc is None or mc is None:
            if h_lab is not None and m_lab is not None:
                encode_fail += 1
            continue
        codes_by_item[qid].append((hc, mc))
        if qid == "conclusion_purchased_product":
            spans_by_item[qid] = 1.0
        else:
            spans_by_item[qid] = 10.0  # BDM WTP prompt: 0 to 10 CAD
    coded = aggregate_coded(codes_by_item, spans_by_item)
    tvd = mean_tvd_by_item(label_pairs)
    return {
        **{k: v for k, v in coded.items() if k != "items"},
        **tvd,
        "encode_failures_among_labeled": encode_fail,
        "encoding": "WTP: float CAD with span=10; purchase TRUE/FALSE -> 1/0 with span=1",
        "items": coded["items"],
    }


def _options_from_question(question: dict[str, Any]) -> list[str]:
    for key in ("Options", "Columns", "Choices"):
        raw = question.get(key)
        if isinstance(raw, list) and raw:
            return [str(x).strip() for x in raw if str(x).strip()]
    return []


def _human_numeric_cells(payload_text: str) -> list[tuple[str, float | None, list[float] | None, list[str], float | None]]:
    """Return (type, scalar_code, list_codes, options, span_hint)."""
    payload = json.loads(payload_text)
    questions: list[dict[str, Any]] = []

    def walk(obj: Any) -> None:
        if isinstance(obj, dict):
            answers = obj.get("Answers")
            if isinstance(answers, dict) and any(
                key in answers for key in ("SelectedByPosition", "SelectedText", "Values")
            ):
                questions.append(obj)
            for value in obj.values():
                walk(value)
        elif isinstance(obj, list):
            for value in obj:
                walk(value)

    walk(payload)
    out = []
    for question in questions:
        qtype = tvdmod._human_type(question)
        answers = question.get("Answers") or {}
        options = _options_from_question(question)
        span = float(len(options) - 1) if len(options) >= 2 else None
        if "Values" in answers and answers["Values"] is not None:
            vals = answers["Values"]
            if isinstance(vals, list):
                codes = [parse_number(v) for v in vals]
                if all(c is not None for c in codes):
                    out.append((qtype, None, codes, options, span))
                    continue
            number = parse_number(vals)
            out.append((qtype, number, None, options, span))
            continue
        pos = answers.get("SelectedByPosition")
        if isinstance(pos, list):
            codes = []
            for item in pos:
                number = parse_number(item)
                codes.append(number)
            if all(c is not None for c in codes):
                out.append((qtype, None, codes, options, span))
                continue
        if pos is not None and not isinstance(pos, list):
            out.append((qtype, parse_number(pos), None, options, span))
            continue
        text = answers.get("SelectedText")
        if isinstance(text, list):
            codes = [encode_with_options(t, options) if options else parse_number(t) for t in text]
            if all(c is not None for c in codes):
                out.append((qtype, None, codes, options, span))
                continue
        if text is not None:
            code = encode_with_options(text, options) if options else parse_number(text)
            out.append((qtype, code, None, options, span))
            continue
        out.append((qtype, None, None, options, span))
    return out


def _model_numeric_cells(parsed_answer: Any) -> list[tuple[str, str, float | None, list[float] | None, list[str], float | None]]:
    if isinstance(parsed_answer, str):
        try:
            parsed_answer = json.loads(parsed_answer)
        except json.JSONDecodeError:
            return []
    if not isinstance(parsed_answer, dict):
        return []
    keys = sorted(
        (key for key in parsed_answer if re.fullmatch(r"Q\d+", str(key))),
        key=lambda key: int(str(key)[1:]),
    )
    out = []
    for key in keys:
        cell = parsed_answer[key]
        if not isinstance(cell, dict):
            continue
        answers = cell.get("Answers") if isinstance(cell.get("Answers"), dict) else cell
        if not isinstance(answers, dict):
            continue
        options = _options_from_question(cell)
        if not options:
            # sometimes options sit beside Answers
            options = _options_from_question(answers)
        span = float(len(options) - 1) if len(options) >= 2 else None
        qtype = tvdmod._model_type(cell)
        if answers.get("Values") is not None:
            vals = answers["Values"]
            if isinstance(vals, list):
                codes = [parse_number(v) for v in vals]
                if all(c is not None for c in codes):
                    out.append((str(key), qtype, None, codes, options, span))
                    continue
            out.append((str(key), qtype, parse_number(vals), None, options, span))
            continue
        pos = answers.get("SelectedByPosition")
        if isinstance(pos, list):
            codes = [parse_number(item) for item in pos]
            if all(c is not None for c in codes):
                out.append((str(key), qtype, None, codes, options, span))
                continue
        if pos is not None and not isinstance(pos, list):
            out.append((str(key), qtype, parse_number(pos), None, options, span))
            continue
        text = answers.get("SelectedText")
        if isinstance(text, list):
            codes = [encode_with_options(t, options) if options else parse_number(t) for t in text]
            if all(c is not None for c in codes):
                out.append((str(key), qtype, None, codes, options, span))
                continue
        if text is not None:
            code = encode_with_options(text, options) if options else parse_number(text)
            out.append((str(key), qtype, code, None, options, span))
            continue
    return out


def _align_numeric(
    human: list[tuple[str, float | None, list[float] | None, list[str], float | None]],
    model: list[tuple[str, str, float | None, list[float] | None, list[str], float | None]],
) -> list[tuple[str, float, float, float]]:
    """Align like TVD (index or LCS on types); expand list codes to :i children."""
    if not human or not model:
        return []
    human_types = [h[0] for h in human]
    model_types = [m[1] for m in model]
    pairs_idx: list[tuple[int, int]] = []
    if len(human) == len(model):
        pairs_idx = list(enumerate(range(len(model))))
    else:
        n_h, n_m = len(human), len(model)
        dp = [[0] * (n_m + 1) for _ in range(n_h + 1)]
        for i in range(n_h - 1, -1, -1):
            for j in range(n_m - 1, -1, -1):
                if human_types[i] == model_types[j]:
                    dp[i][j] = 1 + dp[i + 1][j + 1]
                else:
                    dp[i][j] = max(dp[i + 1][j], dp[i][j + 1])
        i = j = 0
        while i < n_h and j < n_m:
            if human_types[i] == model_types[j]:
                pairs_idx.append((i, j))
                i += 1
                j += 1
            elif dp[i + 1][j] >= dp[i][j + 1]:
                i += 1
            else:
                j += 1

    out: list[tuple[str, float, float, float]] = []
    for hi, mi in pairs_idx:
        h_type, h_scalar, h_list, h_opts, h_span = human[hi]
        qkey, m_type, m_scalar, m_list, m_opts, m_span = model[mi]
        options = h_opts or m_opts
        span = h_span if h_span is not None else m_span
        if span is None and len(options) >= 2:
            span = float(len(options) - 1)
        if h_list is not None and m_list is not None and len(h_list) == len(m_list):
            child_span = span
            if child_span is None:
                vals = list(h_list) + list(m_list)
                child_span = max(vals) - min(vals) if vals else None
            if child_span is None or child_span <= 0:
                child_span = 1.0
            for index, (hc, mc) in enumerate(zip(h_list, m_list), start=1):
                out.append((f"{qkey}:{index}", float(hc), float(mc), float(child_span)))
            continue
        if h_scalar is None or m_scalar is None:
            continue
        if span is None or span <= 0:
            span = abs(float(h_scalar) - float(m_scalar)) or 1.0
        out.append((qkey, float(h_scalar), float(m_scalar), float(span)))
    return out


def score_mega(rows: list[dict[str, Any]]) -> dict[str, Any]:
    label_pairs: dict[str, list[tuple[str, str]]] = defaultdict(list)
    codes_by_item: dict[str, list[tuple[float, float]]] = defaultdict(list)
    spans_by_item: dict[str, float] = {}
    aligned_cases = 0
    encode_fail = 0
    for row in rows:
        gold = row.get("gold") or {}
        payload = gold.get("payload")
        parsed_answer = (row.get("parsed") or {}).get("answer")
        if not payload or parsed_answer is None:
            continue
        try:
            human_labs = tvdmod._extract_human_answers(str(payload))
        except Exception:
            continue
        model_labs = tvdmod._extract_model_answers(parsed_answer)
        aligned_labs = tvdmod._align(human_labs, model_labs)
        study = str(row.get("question_id") or "").split(":", 1)[0]
        for qkey, h_lab, m_lab in aligned_labs:
            h = tvdmod.norm_label(h_lab)
            m = tvdmod.norm_label(m_lab)
            if h is None or m is None:
                continue
            label_pairs[f"{study}:{qkey}"].append((h, m))
        try:
            human_num = _human_numeric_cells(str(payload))
        except Exception:
            encode_fail += 1
            continue
        model_num = _model_numeric_cells(parsed_answer)
        aligned_num = _align_numeric(human_num, model_num)
        if not aligned_num:
            encode_fail += 1
            continue
        aligned_cases += 1
        for qkey, hc, mc, span in aligned_num:
            item = f"{study}:{qkey}"
            codes_by_item[item].append((hc, mc))
            if span is not None and span > 0:
                # keep max span seen (option scales stable)
                spans_by_item[item] = max(float(span), spans_by_item.get(item, 0.0))
    for item_id, pairs in codes_by_item.items():
        if item_id in spans_by_item and spans_by_item[item_id] > 0:
            continue
        vals = [h for h, _ in pairs] + [m for _, m in pairs]
        lo, hi = min(vals), max(vals)
        if hi > lo:
            spans_by_item[item_id] = hi - lo
    coded = aggregate_coded(codes_by_item, spans_by_item)
    tvd = mean_tvd_by_item(label_pairs)
    return {
        **{k: v for k, v in coded.items() if k != "items"},
        **tvd,
        "aligned_cases_coded": aligned_cases,
        "encode_failures_among_labeled": encode_fail,
        "encoding": (
            "Mega: SelectedByPosition as 1-based codes; Values as float; "
            "SelectedText via Options order; matrix/list answers expanded to :i; "
            "alignment index zip or LCS on question types (same as TVD script)"
        ),
        "gold_source": "gold.payload Answers",
        "items": coded["items"],
    }


def score_run(dataset: str, method: str) -> dict[str, Any]:
    path = RUN / dataset / method / "responses.jsonl"
    rows = tvdmod.read_jsonl(path)
    if dataset == "twin2k_500":
        metrics = score_twin2k(rows)
    elif dataset == "bdm_lettuce":
        metrics = score_bdm(rows)
    else:
        metrics = score_mega(rows)
    return {
        "dataset": dataset,
        "method": method,
        "path": str(path),
        "n_rows": len(rows),
        **metrics,
    }


def main() -> None:
    details: list[dict[str, Any]] = []
    compare: list[dict[str, Any]] = []
    formulas = {
        "indiv_acc": {
            "name": "个体准确率",
            "formula": "1 - MAD/range = mean_i (1 - |twin_i - human_i| / span)",
            "source": "popbench.evaluate.metrics.individual_accuracy / ordinal_accuracy",
            "note": "paired twin vs same human; must report when paired gold exists",
        },
        "fisher_z_corr": {
            "name": "跨人相关",
            "formula": "per-item Pearson r across subjects; average via Fisher z; report tanh(mean z)",
            "source": "popbench.evaluate.metrics.fisher_z_mean_correlation + pearson",
            "note": "primary personalization metric (rank who is high/low)",
        },
        "glass_delta": {
            "name": "Glass Δ (均值差)",
            "formula": "(mean_twin - mean_human) / sd_human per item, then unweighted mean",
            "source": "popbench.evaluate.metrics.glass_delta (population_std)",
            "note": "population mean drift vs humans; 0 is ideal",
        },
        "sd_ratio": {
            "name": "SD 比 (孪生/真人)",
            "formula": "sd_twin / sd_human per item, then unweighted mean",
            "source": "popbench.evaluate.metrics.sd_ratio (population_std)",
            "note": "under-dispersion monitor; 1 is ideal, <1 compressed",
        },
        "mean_tvd": {
            "name": "mean TVD",
            "formula": "TVD(p,q)=0.5*Σ|p_i-q_i| on per-item answer shares; unweighted mean over items",
            "source": "popbench.evaluate.metrics.total_variation_distance",
            "note": "lower is closer to human answer distribution",
        },
    }

    def delta(a: Any, b: Any) -> float | None:
        if a is None or b is None:
            return None
        return round(float(b) - float(a), 6)

    for dataset in DATASETS:
        by_method: dict[str, dict[str, Any]] = {}
        for method in METHODS:
            metrics = score_run(dataset, method)
            slim = {k: v for k, v in metrics.items() if k != "items"}
            details.append(metrics)
            by_method[method] = slim
            summary_path = RUN / dataset / method / "run_summary_metrics.json"
            summary_path.write_text(json.dumps(slim, ensure_ascii=False, indent=2) + "\n")
        base = by_method["atus_expanded"]
        skill = by_method["atus_expanded_skill"]
        compare.append(
            {
                "dataset": dataset,
                "atus_expanded": {
                    "indiv_acc": base.get("indiv_acc"),
                    "fisher_z_corr": base.get("fisher_z_corr"),
                    "glass_delta": base.get("glass_delta"),
                    "sd_ratio": base.get("sd_ratio"),
                    "mean_tvd": base.get("mean_tvd"),
                },
                "atus_expanded_skill": {
                    "indiv_acc": skill.get("indiv_acc"),
                    "fisher_z_corr": skill.get("fisher_z_corr"),
                    "glass_delta": skill.get("glass_delta"),
                    "sd_ratio": skill.get("sd_ratio"),
                    "mean_tvd": skill.get("mean_tvd"),
                },
                "delta_skill_minus_base": {
                    "indiv_acc": delta(base.get("indiv_acc"), skill.get("indiv_acc")),
                    "fisher_z_corr": delta(base.get("fisher_z_corr"), skill.get("fisher_z_corr")),
                    "glass_delta": delta(base.get("glass_delta"), skill.get("glass_delta")),
                    "sd_ratio": delta(base.get("sd_ratio"), skill.get("sd_ratio")),
                    "mean_tvd": delta(base.get("mean_tvd"), skill.get("mean_tvd")),
                },
                "encoding_base": base.get("encoding"),
                "n_items_coded_base": base.get("n_items_coded"),
                "n_items_coded_skill": skill.get("n_items_coded"),
            }
        )

    out = {
        "formulas": formulas,
        "compare": compare,
        "methods": list(METHODS),
        "run_dir": str(RUN),
        "no_api_calls": True,
        "excluded_from_table": ["accuracy", "n_scored", "parse_errors"],
    }
    (RUN / "compare_vs_human_metrics.json").write_text(
        json.dumps(out, ensure_ascii=False, indent=2) + "\n"
    )
    detail_out = []
    for row in details:
        detail_out.append({k: v for k, v in row.items() if k != "items"})
    (RUN / "metrics_score_details.json").write_text(
        json.dumps({"formulas": formulas, "runs": detail_out}, ensure_ascii=False, indent=2) + "\n"
    )

    # Optionally extend compare_vs_human.json with new fields without deleting old ones.
    legacy_path = RUN / "compare_vs_human.json"
    legacy = json.loads(legacy_path.read_text()) if legacy_path.is_file() else []
    if isinstance(legacy, dict):
        legacy_rows = legacy.get("compare") or []
    else:
        legacy_rows = legacy
    legacy_by_dataset = {row["dataset"]: dict(row) for row in legacy_rows}
    extended = []
    for row in compare:
        old = legacy_by_dataset.get(row["dataset"], {"dataset": row["dataset"]})
        old["indiv_acc_base"] = row["atus_expanded"]["indiv_acc"]
        old["indiv_acc_skill"] = row["atus_expanded_skill"]["indiv_acc"]
        old["fisher_z_corr_base"] = row["atus_expanded"]["fisher_z_corr"]
        old["fisher_z_corr_skill"] = row["atus_expanded_skill"]["fisher_z_corr"]
        old["glass_delta_base"] = row["atus_expanded"]["glass_delta"]
        old["glass_delta_skill"] = row["atus_expanded_skill"]["glass_delta"]
        old["sd_ratio_base"] = row["atus_expanded"]["sd_ratio"]
        old["sd_ratio_skill"] = row["atus_expanded_skill"]["sd_ratio"]
        old["mean_tvd_base"] = row["atus_expanded"]["mean_tvd"]
        old["mean_tvd_skill"] = row["atus_expanded_skill"]["mean_tvd"]
        old["delta_metrics_skill_minus_base"] = row["delta_skill_minus_base"]
        extended.append(old)
    legacy_path.write_text(json.dumps(extended, ensure_ascii=False, indent=2) + "\n")

    # Compact stdout table
    print(json.dumps(out, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
