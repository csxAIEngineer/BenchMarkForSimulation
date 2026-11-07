"""Source columns that are allowed onto a basic-info card.

Outcome columns and experimental assignments are listed so callers can keep
them off the card. Question wording is not invented here.
"""

from __future__ import annotations

CARD_VERSION = "basic-info-v2"
MISSING_TOKEN = "MISSING"

# The fourteen Twin-2K demographic items, in QID order. Labels are field names.
# They are not rewrites of the catalog stem. Stems are copied separately.
TWIN_DEMOGRAPHICS: tuple[tuple[str, str], ...] = (
    ("QID11", "Region"),
    ("QID12", "Sex"),
    ("QID13", "Age"),
    ("QID14", "Educ"),
    ("QID15", "Race"),
    ("QID16", "Citizen"),
    ("QID17", "Marital"),
    ("QID18", "Religion"),
    ("QID19", "Attend"),
    ("QID20", "Party"),
    ("QID21", "Income"),
    ("QID22", "Ideology"),
    ("QID23", "HHSize"),
    ("QID24", "Employ"),
)
TWIN_LABEL_BY_QID: dict[str, str] = dict(TWIN_DEMOGRAPHICS)
TWIN_BASIC_COLUMNS: tuple[str, ...] = tuple(qid for qid, _label in TWIN_DEMOGRAPHICS)

# BDM columns that name the same concept as one Twin item, on a different instrument.
# The cell is not a Twin answer, so these rows do not receive a Twin qid.
BDM_SAME_CONCEPT: dict[str, str] = {
    "survey_q02_sex": "Sex",
    "survey_q14_political": "Ideology",
    "survey_q15_income": "Income",
    "survey_q16_education": "Educ",
    "survey_q17_household_size": "HHSize",
}
# survey_q01_age is age in years, not the Twin age band, so it is not Age.
TWIN_STRATIFY_COLUMNS: tuple[str, ...] = ("QID21", "QID23")
TWIN_PRICING_IDS: tuple[str, ...] = tuple(f"QID9_{index}" for index in range(1, 41))
TWIN_PUBLISHED_N = 2058
TWIN_DATASET = "twin2k_500"
TWIN_DEMOGRAPHIC_FILE = "question_catalog_and_human_response_csv/wave1_3_response_label.csv"
TWIN_WAVE4_FILE = "question_catalog_and_human_response_csv/wave4_response_label.csv"
TWIN_CATALOG_FILE = "question_catalog_and_human_response_csv/question_catalog.json"
TWIN_REPO_ID = "LLM-Digital-Twin/Twin-2K-500"
TWIN_REVISION = "f883165a3026fde855dfd448e0cd16443ab257b6"
TWIN_WAVE_CHUNKS: tuple[str, ...] = tuple(
    f"wave_split/chunks/wave_persona_chunk_{index:03d}.parquet" for index in range(1, 8)
)

# CSV columns from Gibson & Li. The codebook points survey_* items at
# DemographicQuestionnaire.docx, which stores the questionnaire as images, so
# these cards use the column name and the cell text only.
BDM_BASIC_COLUMNS: tuple[str, ...] = (
    "survey_q01_age",
    "survey_q02_sex",
    "survey_q03_price",
    "survey_q03_locally_grown",
    "survey_q03_pesticide_free",
    "survey_q03_brand_name",
    "survey_q04_heard_of_vertical_farming",
    "survey_q05_heard_of_greenhouse_growing",
    "survey_q06_primary_grocery",
    "survey_q07_primary_cook",
    "survey_q08_leafy_green_meals",
    "survey_q09_not_prepared_meals",
    "survey_q10_try_novel_technology",
    "survey_q11_grow_own_food",
    "survey_q12_variety_concern",
    "survey_q13_lack_of_resources",
    "survey_q14_political",
    "survey_q15_income",
    "survey_q16_education",
    "survey_q17_household_size",
)
BDM_STRATIFY_COLUMNS: tuple[str, ...] = ("survey_q15_income",)
BDM_DATASET = "bdm_lettuce"
BDM_PUBLISHED_N = 610
BDM_SOURCE_FILE = "20240723_VFExperiment_01.csv"
BDM_DOI = "10.5683/SP3/CDLUEZ"
BDM_VERSION = "V1"

# Held out of the card. Pricing bids, the realized purchase, and the drawn
# BDM price are outcomes. Assignment and stimulus order are design variables.
BDM_OUTCOME_COLUMNS: tuple[str, ...] = (
    "local_field",
    "local_greenhouse",
    "local_vertical",
    "nonlocal_field",
    "nonlocal_greenhouse",
    "nonlocal_vertical",
    "conclusion_purchased_product",
)
BDM_ADMIN_COLUMNS: tuple[str, ...] = (
    "last_page",
    "overall_start_time",
    "bidding_start_time",
    "bidding_end_time",
    "overall_end_time",
)
BDM_DESIGN_COLUMNS: tuple[str, ...] = (
    "treatment",
    "treatment3_order",
    "cluster_order",
    "local_order",
    "nonlocal_order",
    "conclusion_selected_product",
    "conclusion_generated_price",
    "P1",
    "P2",
    "P3",
    "P4",
    "P5",
)

ID_COLUMN = "pid"

CARD_KEYS = frozenset(
    {
        "basic_info",
        "card_version",
        "dataset",
        "source_file",
        "source_id_column",
        "subject_id",
        "text",
    }
)
FIELD_KEYS = frozenset(
    {
        "demographic_match",
        "evidence",
        "label",
        "qid",
        "source_column",
        "source_options",
        "source_question_id",
        "source_question_text",
        "source_question_text_status",
        "status",
        "unmapped_reason",
        "value",
    }
)
EVIDENCE_KEYS = frozenset({"qid", "source_column", "source_file"})

MATCH_TWIN_ITEM = "twin_item"
MATCH_SAME_CONCEPT = "same_concept_different_instrument"
MATCH_NONE = "no_counterpart"


def demographic_meta(column: str) -> dict[str, object]:
    """Audit metadata for one basic-info column. This is not model-facing text."""
    if column in TWIN_LABEL_BY_QID:
        return {
            "demographic_match": MATCH_TWIN_ITEM,
            "label": TWIN_LABEL_BY_QID[column],
            "qid": column,
            "unmapped_reason": None,
        }
    concept = BDM_SAME_CONCEPT.get(column)
    if concept is not None:
        return {
            "demographic_match": MATCH_SAME_CONCEPT,
            "label": concept,
            "qid": None,
            "unmapped_reason": None,
        }
    if column == "survey_q01_age":
        reason = "exact age in years, not an age band"
    else:
        reason = "not one of the 14 demographic items"
    return {
        "demographic_match": MATCH_NONE,
        "label": None,
        "qid": None,
        "unmapped_reason": reason,
    }


QUESTION_TEXT_COPIED = "copied_from_question_catalog"
QUESTION_TEXT_UNAVAILABLE = "unavailable_machine_readable_wording"
STATUS_PRESENT = "present"
STATUS_MISSING = "missing"
