"""Build the closed-ended Twin-2K item list and the Nemotron panel that answers it.

The paper's "~500 questions" counts some multi-row tasks as one question and
includes open-ended items. This module expands each scorable closed-ended
answer (matrix row, single choice, number, slider) and skips free text.
Each catalog entry also carries a paper category and, inside the four large
blocks, the Table 1 scale or experiment it belongs to.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pandas as pd

from popbench.dao.interview import InterviewError
from popbench.dao.io import read_jsonl, write_jsonl
from popbench.dao.personas import sample_usa_adults


def survey_dir(data_dir: Path) -> Path:
    return Path(data_dir) / "interview" / "full"


# Seven paper categories, keyed by the stripped BlockName values in the
# Twin-2K catalog (revision f883165). Behavioral economics experiments are
# not one block: each condition has its own BlockName. Forward Flow is its
# own block and belongs with the cognitive tests.
_BLOCK_FAMILY = {
    "Demographics": "demographics",
    "Personality": "personality",
    "Cognitive tests": "cognitive",
    "Forward Flow": "cognitive",
    "Economic preferences": "economic",
    "Economic preferences - intro": "economic",
    "Non-experimental heuristics and biases": "heuristics",
    "Product Preferences - Pricing": "pricing",
    "Absolute vs. relative - calculator": "experiments",
    "Absolute vs. relative - jacket": "experiments",
    "Allais Form 1": "experiments",
    "Allais Form 2": "experiments",
    "Anchoring - African countries high": "experiments",
    "Anchoring - African countries low": "experiments",
    "Anchoring - redwood high": "experiments",
    "Anchoring - redwood low": "experiments",
    "Base-rate 30 engineers": "experiments",
    "Base-rate 70 engineers": "experiments",
    "Disease - gain": "experiments",
    "Disease-loss": "experiments",
    "False consensus": "experiments",
    "Less is More Gamble A": "experiments",
    "Less is More Gamble B": "experiments",
    "Less is More Gamble C": "experiments",
    "Linda -no conjunction": "experiments",
    "Linda-conjunction": "experiments",
    "Myside Ford": "experiments",
    "Myside German": "experiments",
    "Outcome bias - failure": "experiments",
    "Outcome bias - success": "experiments",
    "Probability matching vs. maximizing - Problem 1": "experiments",
    "Probability matching vs. maximizing - Problem 2": "experiments",
    "Proportion dominance 1A": "experiments",
    "Proportion dominance 1B": "experiments",
    "Proportion dominance 1C": "experiments",
    "Proportion dominance 2A": "experiments",
    "Proportion dominance 2B": "experiments",
    "Proportion dominance 2C": "experiments",
    "Sunk cost - no": "experiments",
    "Sunk cost - yes": "experiments",
    "WTA/WTP Thaler - WTP noncertainty": "experiments",
    "WTA/WTP Thaler problem - WTA certainty": "experiments",
    "WTA/WTP Thaler problem - WTP certainty": "experiments",
}

PAPER_FAMILIES = (
    "demographics",
    "personality",
    "cognitive",
    "economic",
    "heuristics",
    "experiments",
    "pricing",
)

# Table 1 tasks are recovered inside these blocks. Demographics, the
# non-experimental heuristics block, and pricing stay unlabeled.
INSTRUMENT_FAMILIES = (
    "personality",
    "cognitive",
    "economic",
    "experiments",
)

# A descriptive screen, not an answerable task.
INSTRUCTION = "instruction"


def family_for(block: str) -> str:
    """Map one catalog BlockName onto a paper category.

    Matching ignores surrounding whitespace. A block that is not in the
    published catalog is an error, rather than a behavioral experiment.
    """
    name = block.strip()
    family = _BLOCK_FAMILY.get(name)
    if family is None:
        raise InterviewError(f"unknown Twin-2K block {name!r}")
    return family


def families_by_question(catalog: list[dict[str, Any]]) -> list[dict[str, str]]:
    """Classify every catalog entry.

    QuestionID strings repeat across blocks. ``instrument`` is the Table 1
    task inside the four large blocks, ``instruction`` for a descriptive
    screen, and empty for the other blocks.
    """
    classified: list[dict[str, str]] = []
    for entry in catalog:
        question_id = entry.get("QuestionID")
        if not isinstance(question_id, str) or not question_id:
            raise InterviewError(f"catalog entry is missing QuestionID: {question_id!r}")
        block = (entry.get("BlockName") or "").strip()
        classified.append(
            {
                "question_id": question_id,
                "block": block,
                "family": family_for(block),
                "instrument": instrument_for(entry),
            }
        )
    return classified


def instrument_for(entry: dict[str, Any]) -> str:
    """Table 1 task for one catalog entry.

    Personality, cognitive, economic-preference, and experiment questions
    take the paper's task name. The first statement separates scales that
    share a boilerplate opening. A question the table does not list keeps
    the first sentence of its stem. Descriptive screens are instructions.
    Other blocks are blank.
    """
    if entry.get("is_descriptive") or entry.get("QuestionType") == "DB":
        return INSTRUCTION
    block = (entry.get("BlockName") or "").strip()
    family = family_for(block)
    if family == "personality":
        return _personality_instrument(entry)
    if family == "cognitive":
        return _cognitive_instrument(entry)
    if family == "economic":
        return _economic_instrument(entry)
    if family == "experiments":
        return _experiment_instrument(entry)
    return ""


# Phrases that identify a Table 1 personality scale. Question text is tried
# first; scales that share an opening are then split on the first statement.
_PERSONALITY_NEEDLES = (
    ("I see myself as someone who", "Big Five"),
    ("complex to simple problems", "Need for cognition"),
    ("24 different values", "Agentic vs. communal values"),
    ("avoid accumulating lots of stuff", "Consumer minimalism"),
    ("friends' emotions", "Empathy"),
    ("do not harm the environment", "Green values"),
    ("pertains to your personally", "Social desirability"),
    ("accurately or inaccurately", "Conscientiousness"),
    ("bothered by each of the following symptoms", "Anxiety"),
    ("depend on myself than others", "Individualism vs. collectivism"),
    ("type of person you", "Selves questionnaire"),
    ("true or untrue", "Regulatory focus"),
    ("spending habits", "Tightwads vs. spendthrift"),
    ("trouble limiting your spending", "Tightwads vs. spendthrift"),
    ("trouble spending money", "Tightwads vs. spendthrift"),
    ("two shoppers", "Tightwads vs. spendthrift"),
    ("purposely trying to lose weight", "Depression"),
    ("cannot be duplicated", "Need for uniqueness"),
    ("alter my behavior", "Self-monitoring"),
    ("beliefs about myself often conflict", "Self-concept clarity"),
    ("situations that are uncertain", "Need for closure"),
    ("check other stations", "Maximization"),
)

# The four verbal riddles are the CRT. The bat-and-ball, widget, and lily-pad
# items sit in the eight-item numeracy battery, not in that CRT.
_COGNITIVE_NEEDLES = (
    ("April and May", "CRT"),
    ("cubic feet of dirt", "CRT"),
    ("person in second place", "CRT"),
    ("all but 8 died", "CRT"),
    ("rotation of the cube", "Fluid intelligence"),
    ("puzzle below", "Fluid intelligence"),
    ("figure below", "Fluid intelligence"),
    ("you answered correctly", "Overconfidence"),
    ("representative sample", "Overplacement"),
    ("capital letter on one side", "Wason selection"),
    ("15-year mortgage", "Financial literacy"),
    ("interest rate on your saving", "Financial literacy"),
    ("highest fluctuations", "Financial literacy"),
    ("stock mutual fund", "Financial literacy"),
    ("spreads their money", "Financial literacy"),
    ("gives the highest return", "Financial literacy"),
    ("401(k)", "Financial literacy"),
    ("credit card", "Financial literacy"),
    ("six-sided die", "Numeracy"),
    ("BIG BUCKS", "Numeracy"),
    ("20 out of 100", "Numeracy"),
    ("ACME PUBLISHING", "Numeracy"),
    ("chance of getting a disease is 10%", "Numeracy"),
    ("5 machines 5 minutes", "Numeracy"),
    ("bat and ball", "Numeracy"),
    ("lily pads", "Numeracy"),
    ("series of four problems", "Deductive certainty"),
    ("Premise 1: If ", "Deductive certainty"),
    ("series of eight problems", "Syllogisms"),
    ("XAR", "Syllogisms"),
    ("YOF", "Syllogisms"),
    ("Premise 1: All ", "Syllogisms"),
    ("Premise I:", "Syllogisms"),
)

# Several between-subject conditions are one Table 1 experiment. Proportion
# dominance is not in the table (the table's denominator-neglect task is the
# marble tray in the heuristics block), so those stems stay as written.
_EXPERIMENT_PREFIXES = (
    ("Anchoring -", "Anchoring"),
    ("Base-rate", "Base rate"),
    ("Disease", "Framing"),
    ("Outcome bias", "Outcome bias"),
    ("Sunk cost", "Sunk cost"),
    ("Allais", "Allais problem"),
    ("Linda", "Conjunction (Linda)"),
    ("Absolute vs. relative", "Absolute vs. relative savings"),
    ("Myside", "Myside bias"),
    ("Less is More", "Less is more"),
    ("WTA/WTP", "WTA/WTP"),
    ("False consensus", "False consensus"),
    ("Probability matching", "Probability matching"),
)


def _personality_instrument(entry: dict[str, Any]) -> str:
    text = _flat(entry.get("QuestionText"))
    # Beck items store the choices in Options and leave QuestionText empty.
    if not text:
        return "Depression"
    matched = _first_needle(text, _PERSONALITY_NEEDLES)
    if matched:
        return matched
    rows = entry.get("Rows") or []
    if rows:
        matched = _first_needle(_flat(rows[0]), _PERSONALITY_NEEDLES)
        if matched:
            return matched
    return _stem_summary(text)


def _cognitive_instrument(entry: dict[str, Any]) -> str:
    if (entry.get("BlockName") or "").strip() == "Forward Flow":
        return "Forward Flow"
    text = _flat(entry.get("QuestionText"))
    if text.casefold().startswith(("synonym:", "antonym:")):
        return "Crystallized intelligence"
    matched = _first_needle(text, _COGNITIVE_NEEDLES)
    if matched:
        return matched
    return _stem_summary(text)


def _economic_instrument(entry: dict[str, Any]) -> str:
    text = _flat(entry.get("QuestionText"))
    folded = text.casefold()
    rows = _flat(" ".join(entry.get("Rows") or []))
    if "whether or not to accept your offer" in folded:
        return "Ultimatum game (sender)"
    if "offers to keep" in folded:
        return "Ultimatum game (receiver)"
    if "who was happier" in folded or "who was more upset" in folded:
        return "Mental accounting"
    if "tripled" in folded:
        return "Trust game (sender)"
    if "if the person sends me" in folded:
        return "Trust game (receiver)"
    if folded.startswith("second, please list the thoughts"):
        return "Trust game (receiver) thought listing"
    if "send back" in folded and "thinking about" in folded:
        return "Trust game (sender) thought listing"
    if "you and you only would decide" in folded:
        return "Dictator game"
    if folded.startswith("we are now interested in what you were thinking"):
        return "Dictator game thought listing"
    # Loss lists also say "lottery", and both risk and time lists say
    # "please choose between". The distinctive clause has to win.
    if "losing" in folded:
        return "Loss aversion"
    if "the lottery is" in folded:
        return "Risk aversion"
    if folded.startswith("please choose between the following options"):
        if _has_word(rows, "now"):
            return "Present bias"
        return "Discount"
    return _stem_summary(text)


def _experiment_instrument(entry: dict[str, Any]) -> str:
    block = (entry.get("BlockName") or "").strip()
    for prefix, name in _EXPERIMENT_PREFIXES:
        if block.startswith(prefix):
            return name
    return _stem_summary(entry.get("QuestionText") or "")


def _first_needle(text: str, needles: tuple[tuple[str, str], ...]) -> str | None:
    folded = text.casefold()
    for needle, name in needles:
        if needle.casefold() in folded:
            return name
    return None


def _has_word(text: str, word: str) -> bool:
    tokens = [raw.strip(".,:;") for raw in text.casefold().split()]
    return word.casefold() in tokens


def _flat(text: str | None) -> str:
    cleaned = (
        (text or "")
        .replace("\u2019", "'")
        .replace("\u2018", "'")
        .replace("\u201c", '"')
        .replace("\u201d", '"')
        .replace("&ldquo;", '"')
        .replace("&rdquo;", '"')
        .replace("&#39;", "'")
        .replace("&amp;", "&")
    )
    return " ".join(cleaned.split())


def _stem_summary(text: str) -> str:
    """First sentence of a question the paper's table does not name."""
    flat = _flat(text)
    if not flat:
        raise InterviewError("Twin-2K question has no stem to summarize")
    end = len(flat)
    for index, char in enumerate(flat):
        if char not in ".!?":
            continue
        if char == "." and index + 1 < len(flat) and flat[index + 1].isdigit():
            continue
        if index + 1 == len(flat) or flat[index + 1] == " ":
            end = index + 1
            break
    summary = flat[:end].strip()
    if len(summary) > 180:
        summary = summary[:180].rsplit(" ", 1)[0].rstrip(".,;:")
    return summary


def _task_labels(entry: dict[str, Any]) -> dict[str, str]:
    block = (entry.get("BlockName") or "").strip()
    return {
        "family": family_for(block),
        "block": block,
        "instrument": instrument_for(entry),
    }


def is_open_entry(entry: dict[str, Any]) -> bool:
    if entry.get("is_descriptive") or entry.get("QuestionType") == "DB":
        return True
    settings = entry.get("Settings") or {}
    selector = settings.get("Selector")
    content = settings.get("ContentType")
    if entry.get("QuestionType") == "TE" and selector in {"ML", "FORM"}:
        return True
    if entry.get("QuestionType") == "TE" and selector == "SL" and content != "ValidNumber":
        return True
    return False


ASK_N = 500


def build_closed_items(catalog: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """One scorable answer per row, in catalog order."""
    # Open and descriptive questions are skipped below, so classify the whole
    # catalog first. An unknown block then fails before those rows disappear.
    classified = families_by_question(catalog)
    if len(classified) != len(catalog):
        raise InterviewError("catalog entries were not all classified")
    items: list[dict[str, Any]] = []
    for entry in catalog:
        if is_open_entry(entry):
            continue
        items.extend(_expand(entry))
    if len(items) < 400:
        raise InterviewError(f"expected several hundred closed items, found {len(items)}")
    for turn, item in enumerate(items, start=1):
        item["turn"] = turn
    return items


def items_by_family(items: list[dict[str, Any]]) -> dict[str, list[dict[str, Any]]]:
    """Copy closed items into the seven paper categories and restart turn numbers.

    Each category is its own interview, so the first question of a category is turn 1.
    """
    grouped: dict[str, list[dict[str, Any]]] = {family: [] for family in PAPER_FAMILIES}
    for item in items:
        family = item.get("family")
        if family not in grouped:
            raise InterviewError(f"closed item {item.get('item_id')!r} has unknown family {family!r}")
        grouped[family].append(dict(item))
    for rows in grouped.values():
        for turn, row in enumerate(rows, start=1):
            row["turn"] = turn
    return grouped


def cap_items(items: list[dict[str, Any]], n: int = ASK_N) -> list[dict[str, Any]]:
    """Keep the first n items, and finish the matrix question that contains item n."""
    if len(items) <= n:
        return items
    question = _question_key(items[n - 1]["item_id"])
    end = n
    while end < len(items) and _question_key(items[end]["item_id"]) == question:
        end += 1
    kept = items[:end]
    for turn, item in enumerate(kept, start=1):
        item["turn"] = turn
    return kept


def _question_key(item_id: str) -> str:
    head, sep, tail = item_id.rpartition("_")
    if sep and tail.isdigit():
        return head
    return item_id


def _expand(entry: dict[str, Any]) -> list[dict[str, Any]]:
    question_type = entry.get("QuestionType")
    selector = (entry.get("Settings") or {}).get("Selector")
    if question_type == "Matrix":
        return _matrix_items(entry)
    if question_type == "Slider":
        return _slider_items(entry)
    if selector in {"MAVR", "MAHR"}:
        return [_multi_as_choice(entry)]
    if question_type == "TE":
        return [_numeric_item(entry, entry["csv_columns"][0], _stem(entry))]
    return [_choice_item(entry, entry["csv_columns"][0], _stem(entry), list(entry.get("Options") or []))]


def _matrix_items(entry: dict[str, Any]) -> list[dict[str, Any]]:
    rows = entry.get("Rows") or []
    columns = entry.get("csv_columns") or []
    if len(rows) != len(columns):
        raise InterviewError(f"{entry['QuestionID']} rows and csv columns differ")
    selector = (entry.get("Settings") or {}).get("Selector")
    scale = list(entry.get("Columns") or [])
    items = []
    stem = _stem(entry)
    for statement, column in zip(rows, columns):
        if selector == "Bipolar":
            left, right = statement.split(":", 1)
            options = [left.strip(), right.strip()]
            prompt = f"{stem}\n\nChoose one.\nLeft: {options[0]}\nRight: {options[1]}"
        else:
            options = scale
            prompt = f"{stem}\n\n{statement}"
        items.append(_choice_item(entry, column, prompt, options))
    return items


def _slider_items(entry: dict[str, Any]) -> list[dict[str, Any]]:
    columns = entry.get("csv_columns") or []
    statements = entry.get("Statements") or []
    if statements and len(statements) != len(columns):
        raise InterviewError(f"{entry['QuestionID']} slider statements and columns differ")
    bounds = entry.get("Range") or {}
    low = bounds.get("Min", 0)
    high = bounds.get("Max", 100)
    stem = _stem(entry)
    items = []
    for index, column in enumerate(columns):
        statement = statements[index].strip() if statements else ""
        extra = f"\n\n{statement}" if statement else ""
        prompt = f"{stem}{extra}\n\nReply with a number from {low} to {high}."
        items.append(_numeric_item(entry, column, prompt))
    return items


def _multi_as_choice(entry: dict[str, Any]) -> dict[str, Any]:
    options = list(entry.get("Options") or [])
    columns = list(entry.get("csv_columns") or [])
    if len(options) != len(columns):
        raise InterviewError(f"{entry['QuestionID']} options and columns differ")
    stem = _stem(entry) or "Which statement fits you best?"
    item = _choice_item(entry, columns[0], stem, options)
    item["item_id"] = entry["QuestionID"]
    item["encoding"] = "first_selected"
    item["columns"] = columns
    return item


def _choice_item(
    entry: dict[str, Any],
    column: str,
    prompt: str,
    options: list[str],
) -> dict[str, Any]:
    if len(options) < 2:
        raise InterviewError(f"{column} needs at least two options")
    return {
        "item_id": column,
        "column": column,
        "columns": [column],
        "encoding": "single",
        "kind": "choice",
        **_task_labels(entry),
        "prompt": prompt.strip(),
        "options": options,
    }


def _numeric_item(entry: dict[str, Any], column: str, prompt: str) -> dict[str, Any]:
    return {
        "item_id": column,
        "column": column,
        "columns": [column],
        "encoding": "number",
        "kind": "numeric",
        **_task_labels(entry),
        "prompt": prompt.strip(),
        "options": [],
    }


def _stem(entry: dict[str, Any]) -> str:
    text = (entry.get("QuestionText") or "").strip()
    if not text:
        return ""
    head = text.split("\n\n")[0].strip()
    if len(head) > 900:
        head = head[:900].rsplit(" ", 1)[0]
    return head


def human_baseline(numeric: pd.DataFrame, items: list[dict[str, Any]]) -> dict[str, Any]:
    rows = []
    for item in items:
        if item["kind"] == "choice":
            codes = _choice_codes(numeric, item)
            observed = codes.dropna().astype(int)
            n_options = len(item["options"])
            shares = []
            for code in range(1, n_options + 1):
                shares.append(float((observed == code).mean()) if len(observed) else 0.0)
            rows.append(
                {
                    "item_id": item["item_id"],
                    "kind": "choice",
                    "family": item["family"],
                    "n": int(len(observed)),
                    "shares": shares,
                    "mean_code": float(observed.mean()) if len(observed) else None,
                }
            )
        else:
            values = pd.to_numeric(numeric[item["column"]], errors="coerce").dropna()
            rows.append(
                {
                    "item_id": item["item_id"],
                    "kind": "numeric",
                    "family": item["family"],
                    "n": int(len(values)),
                    "mean": float(values.mean()) if len(values) else None,
                }
            )
    return {"n_respondents": int(len(numeric)), "items": rows}


def _choice_codes(frame: pd.DataFrame, item: dict[str, Any]) -> pd.Series:
    if item["encoding"] == "single":
        return pd.to_numeric(frame[item["column"]], errors="coerce")
    block = frame[item["columns"]].apply(pd.to_numeric, errors="coerce")
    selected = block.fillna(0).gt(0)
    any_selected = selected.any(axis=1)
    index = selected.to_numpy().argmax(axis=1) + 1
    codes = pd.Series(index, index=frame.index, dtype="float64")
    return codes.where(any_selected)


def sample_panel(data_dir: Path, items: list[dict[str, Any]], n: int, seed: int) -> list[dict[str, Any]]:
    """Nemotron adults, with one turn per closed-ended item and no gold answer."""
    from popbench.dao.personas import PersonaError

    try:
        adults = sample_usa_adults(data_dir, n=n, seed=seed)
    except PersonaError as exc:
        raise InterviewError(str(exc)) from exc
    turns = [
        {
            "turn": item["turn"],
            "item_id": item["item_id"],
            "kind": item["kind"],
            "family": item["family"],
            "block": item["block"],
            "instrument": item["instrument"],
            "prompt": item["prompt"],
            "options": list(item["options"]),
        }
        for item in items
    ]
    records = []
    for person in adults:
        records.append(
            {
                "id": f"nemotron:{person.id}",
                "persona": {
                    "id": person.id,
                    "age_band": person.age_band,
                    "sex": person.sex,
                    "text": person.text,
                },
                "turns": turns,
            }
        )
    return records


def prepare_survey(
    data_dir: Path,
    n_people: int = 50,
    seed: int = 0,
    ask_n: int = ASK_N,
) -> dict[str, Any]:
    """Load or build the closed-item panel and the human baseline."""
    from popbench.dao.twin2k import load_twin2k

    data = load_twin2k(data_dir, download=False)
    closed = build_closed_items(data.catalog)
    items = cap_items(closed, ask_n)
    out = survey_dir(data_dir)
    out.mkdir(parents=True, exist_ok=True)
    baseline_path = out / "human_baseline.json"
    if baseline_path.is_file():
        baseline = json.loads(baseline_path.read_text())
    else:
        baseline = human_baseline(data.wave1_3, items)
        baseline_path.write_text(json.dumps(baseline) + "\n")
    panel_path = out / "nemotron_panel.jsonl"
    if panel_path.is_file():
        records = read_jsonl(panel_path)
        kept = {item["item_id"] for item in items}
        for record in records:
            selected = [turn for turn in record["turns"] if turn["item_id"] in kept]
            for turn_no, turn in enumerate(selected, start=1):
                turn["turn"] = turn_no
            record["turns"] = selected
    else:
        records = sample_panel(data_dir, items, n=n_people, seed=seed)
        write_jsonl(panel_path, records)
    manifest = {"n_closed": len(closed), "n_asked": len(items), "n_people": len(records)}
    (out / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    return {
        "items": items,
        "baseline": baseline,
        "records": records,
        "n_closed": len(closed),
        "panel_path": panel_path,
    }


# Folder name under data/twin2k/. Labels match DataReadme.md.
CATEGORY_SPLIT_DIR = "by_category"
CATEGORY_LABELS = {
    "demographics": "人口学",
    "personality": "人格",
    "cognitive": "认知",
    "economic": "经济偏好",
    "heuristics": "非实验启发式",
    "experiments": "行为经济学实验",
    "pricing": "定价",
}


def category_split_dir(data_dir: Path) -> Path:
    from popbench.dao.twin2k import twin2k_dir

    return twin2k_dir(data_dir) / CATEGORY_SPLIT_DIR


def write_category_splits(data: Any, dest: Path) -> dict[str, Any]:
    """Write one folder per paper category.

    Each folder holds that category's catalog entries and the response columns
    those entries own. Wave 4 is written only for categories that were repeated.
    Column order follows the source response table.
    """
    catalog = data.catalog
    _same_response_columns(data.wave1_3, data.wave1_3_labels, "wave1_3 labels")
    _same_response_columns(data.wave4, data.wave4_labels, "wave4 labels")
    classified = families_by_question(catalog)
    owner = _column_owner(catalog, classified)
    wave1_columns = _data_columns(data.wave1_3)
    _partition(owner, wave1_columns, "wave1_3")
    wave4_columns = _data_columns(data.wave4)
    extra_wave4 = [column for column in wave4_columns if column not in owner]
    if extra_wave4:
        raise InterviewError(
            "wave4 columns are not in the catalog: " + ", ".join(extra_wave4[:8])
        )
    dest = Path(dest)
    dest.mkdir(parents=True, exist_ok=True)
    categories = []
    for family in PAPER_FAMILIES:
        entries = [entry for entry, row in zip(catalog, classified) if row["family"] == family]
        if not entries:
            continue
        columns = [column for column in wave1_columns if owner[column] == family]
        repeated = [column for column in wave4_columns if owner[column] == family]
        folder = dest / family
        folder.mkdir(parents=True, exist_ok=True)
        (folder / "question_catalog.json").write_text(
            json.dumps(entries, ensure_ascii=False, indent=2) + "\n"
        )
        _write_slice(data.wave1_3, columns, folder / "wave1_3_response.csv")
        _write_slice(data.wave1_3_labels, columns, folder / "wave1_3_response_label.csv")
        if repeated:
            _write_slice(data.wave4, repeated, folder / "wave4_response.csv")
            _write_slice(data.wave4_labels, repeated, folder / "wave4_response_label.csv")
        categories.append(
            {
                "family": family,
                "label": CATEGORY_LABELS[family],
                "n_questions": len(entries),
                "wave1_3_columns": len(columns),
                "wave4_columns": len(repeated),
                "dir": family,
            }
        )
    summary = {
        "revision": data.revision,
        "n_respondents": int(len(data.wave1_3)),
        "categories": categories,
    }
    (dest / "manifest.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2) + "\n")
    return summary


def _column_owner(
    catalog: list[dict[str, Any]],
    classified: list[dict[str, str]],
) -> dict[str, str]:
    owner: dict[str, str] = {}
    for entry, row in zip(catalog, classified, strict=True):
        for column in entry.get("csv_columns") or []:
            if not isinstance(column, str) or not column:
                raise InterviewError(f"{row['question_id']} has an empty csv column")
            if column in owner:
                raise InterviewError(f"{column} belongs to more than one catalog entry")
            owner[column] = row["family"]
    return owner


def _data_columns(frame: pd.DataFrame) -> list[str]:
    if frame.columns[0] != "pid":
        raise InterviewError("response table must start with pid")
    return [str(column) for column in frame.columns[1:]]


def _partition(owner: dict[str, str], columns: list[str], table: str) -> None:
    owned = set(owner)
    present = set(columns)
    missing = sorted(owned - present)
    extra = sorted(present - owned)
    if missing or extra:
        detail = []
        if missing:
            detail.append("missing " + ", ".join(missing[:8]))
        if extra:
            detail.append("extra " + ", ".join(extra[:8]))
        raise InterviewError(f"{table} columns do not match the catalog: " + "; ".join(detail))


def _same_response_columns(numeric: pd.DataFrame, labels: pd.DataFrame, name: str) -> None:
    if list(numeric.columns) != list(labels.columns):
        raise InterviewError(f"{name} columns differ from the numeric table")


def _write_slice(frame: pd.DataFrame, columns: list[str], path: Path) -> None:
    frame.loc[:, ["pid", *columns]].to_csv(path, index=False)


