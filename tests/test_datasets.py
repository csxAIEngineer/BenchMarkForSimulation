"""Dataset loaders: unified schema, and SimBench stays a group distribution."""

import json

import pytest

from popbench.datasets.io import DatasetError, read_jsonl
from popbench.datasets.simbench.loader import bundle_from_records as simbench_bundle
from popbench.datasets.simbench.loader import download as download_simbench
from popbench.datasets.twin2k_500.loader import bundle_from_records as twin_bundle
from popbench.datasets.twin2k_500.loader import download as download_twin
from popbench.datasets.twin2k_500.render import demographic_fields, held_out_questions
from popbench.datasets.twin2k_mega_study.loader import attach_twin2k
from popbench.datasets.twin2k_mega_study.loader import bundle_from_records as mega_bundle
from popbench.datasets.twin2k_mega_study.loader import download as download_mega
from popbench.datasets import resolve_dataset
from popbench.datasets.twin2k_500.loader import load as load_twin


PERSONA = {
    "Elements": [
        {
            "ElementType": "Block",
            "BlockName": "Demographics",
            "Questions": [
                {
                    "QuestionID": "age",
                    "QuestionText": "What is your age?",
                    "QuestionType": "TE",
                    "Answers": {"Text": "34"},
                }
            ],
        },
        {
            "ElementType": "Block",
            "BlockName": "Personality",
            "Questions": [
                {
                    "QuestionID": "bfi",
                    "QuestionText": "I am outgoing",
                    "QuestionType": "MC",
                    "Options": ["Yes", "No"],
                    "Answers": {"SelectedText": "SECRET_PERSONA"},
                }
            ],
        },
    ]
}

WAVE4 = {
    "Elements": [
        {
            "ElementType": "Block",
            "BlockName": "Wave 4",
            "Questions": [
                {
                    "QuestionID": "Q9",
                    "QuestionText": "Pick one",
                    "QuestionType": "MC",
                    "Options": ["red", "blue"],
                    "Answers": {"SelectedText": "SECRET_GOLD"},
                }
            ],
        }
    ]
}


def test_unknown_dataset_and_missing_cache_say_what_to_run(tmp_path):
    with pytest.raises(KeyError, match="unknown dataset"):
        resolve_dataset("nope")
    with pytest.raises(DatasetError, match="benchmark fetch twin2k_500"):
        load_twin(tmp_path)
    broken = tmp_path / "broken.jsonl"
    broken.write_text("{not json}\n", encoding="utf-8")
    with pytest.raises(DatasetError, match="not JSON"):
        read_jsonl(broken)


def test_aliases_include_the_misspelled_simbench_name():
    assert resolve_dataset("SimBench") == "simbench"
    assert resolve_dataset("SimBechMark") == "simbench"
    assert resolve_dataset("twin-2k-500-mega-study") == "twin2k_mega_study"


def test_demographic_variant_keeps_only_the_first_block():
    text, fields = demographic_fields(PERSONA)
    assert "What is your age?" in text
    assert "34" in text
    assert "SECRET_PERSONA" not in text
    assert "SECRET_PERSONA" not in fields.values()


def test_wave4_prompt_does_not_contain_the_held_out_answer():
    questions = held_out_questions(WAVE4)
    assert questions[0]["gold"] == "SECRET_GOLD"
    assert "Answer: SECRET_GOLD" not in questions[0]["prompt"]
    assert "red" in questions[0]["prompt"]


def test_matrix_rows_are_separate_questions_without_the_selected_cell():
    payload = {
        "Elements": [
            {
                "ElementType": "Block",
                "BlockName": "Scale",
                "Questions": [
                    {
                        "QuestionID": "M1",
                        "QuestionText": "Rate these",
                        "QuestionType": "Matrix",
                        "Columns": ["Low", "High"],
                        "Rows": ["Item one", "Item two"],
                        "Answers": {"SelectedText": ["Low", "SECRET_CELL"]},
                    }
                ],
            }
        ]
    }
    questions = held_out_questions(payload)
    assert [item["id"] for item in questions] == ["M1:1", "M1:2"]
    assert questions[1]["gold"] == "SECRET_CELL"
    assert "Answer: SECRET_CELL" not in questions[1]["prompt"]
    assert "High" in questions[1]["prompt"]


def test_twin2k_bundle_is_individual_and_hides_gold():
    bundle = twin_bundle(
        [
            {
                "id": "7",
                "demographics": {"age": "34"},
                "demographic_text": "Age: 34",
                "persona_text": "Full card",
                "persona_summary": "Short summary",
                "questions": [
                    {
                        "id": "Q9",
                        "prompt": "Pick one\nOptions:\n- red\n- blue",
                        "options": ["red", "blue"],
                        "kind": "multiple_choice",
                        "gold": "SECRET_GOLD",
                    }
                ],
            }
        ]
    )
    assert bundle.unit == "individual"
    assert bundle.cases[0].gold.answer == "SECRET_GOLD"
    assert "SECRET_GOLD" not in bundle.cases[0].question.prompt


def test_mega_study_keeps_human_json_off_the_question():
    bundle = mega_bundle(
        [
            {
                "id": "pid_7",
                "study": "accuracy_nudges",
                "survey_text": "Please answer the new items.",
                "gold_payload": "{\"answer\": \"SECRET_GOLD\"}",
            }
        ]
    )
    case = bundle.cases[0]
    assert case.subject.id == "7"
    assert case.subject.unit == "individual"
    assert case.subject.persona_text is None
    assert "SECRET_GOLD" not in case.question.prompt
    assert case.gold.payload == "{\"answer\": \"SECRET_GOLD\"}"


def test_attach_twin2k_copies_persona_fields_by_pid():
    mega = mega_bundle(
        [
            {
                "id": "7",
                "study": "privacy",
                "survey_text": "New study question",
                "gold_payload": "{\"kept\": true}",
            }
        ]
    )
    twin = twin_bundle(
        [
            {
                "id": "7",
                "demographics": {"age": "34"},
                "demographic_text": "Age: 34",
                "persona_text": "Full card",
                "persona_summary": "Short summary",
                "questions": [],
            }
        ]
    )
    joined = attach_twin2k(mega, twin)
    subject = joined.cases[0].subject
    assert subject.persona_text == "Full card"
    assert subject.persona_summary == "Short summary"
    assert subject.demographic_text == "Age: 34"
    assert joined.cases[0].gold.payload == "{\"kept\": true}"


def test_simbench_stays_a_group_and_refuses_a_person_row():
    bundle = simbench_bundle(
        [
            {
                "split": "SimBenchPop",
                "dataset_name": "OpinionQA",
                "group_key": "OpinionQA:population",
                "group_prompt": "You are a respondent in the OpinionQA sample.",
                "variables": {},
                "question_id": "q1",
                "prompt": "Which way?",
                "options": ["A", "B"],
                "distribution": {"A": 0.25, "B": 0.75},
                "group_size": 100,
            },
            {
                "split": "SimBenchPop",
                "dataset_name": "OpinionQA",
                "group_key": "OpinionQA:population",
                "group_prompt": "You are a respondent in the OpinionQA sample.",
                "variables": {},
                "question_id": "q2",
                "prompt": "Again?",
                "options": ["A", "B"],
                "distribution": {"A": 0.5, "B": 0.5},
                "group_size": 100,
            },
        ]
    )
    assert bundle.unit == "group"
    assert len(bundle.subjects) == 1
    assert bundle.subjects[0].unit == "group"
    assert bundle.cases[0].gold.distribution_dict() == {"A": 0.25, "B": 0.75}
    assert "pid_" not in bundle.subjects[0].id
    with pytest.raises(DatasetError, match="individual"):
        simbench_bundle([{"pid": "1", "human_answer": "A"}])


def test_simbench_formats_the_published_group_template(tmp_path, monkeypatch):
    rows = [
        {
            "dataset_name": "ESS",
            "group_prompt_template": "You live in {country} and are {age_group}.",
            "group_prompt_variable_map": {"country": "France", "age_group": "30-49"},
            "input_template": "Trust most people?",
            "human_answer": {"A": 0.4, "B": 0.6},
            "group_size": 50,
            "auxiliary": {"task_id": "trust"},
        }
    ]
    _patch_hub(monkeypatch, {"SimBenchGrouped": rows})
    download_simbench(tmp_path, split="SimBenchGrouped", limit=1)
    from popbench.datasets.simbench.loader import load

    bundle = load(tmp_path, split="SimBenchGrouped")
    subject = bundle.subjects[0]
    assert subject.unit == "group"
    assert subject.group_prompt == "You live in France and are 30-49."
    assert subject.id == "ESS:age_group=30-49|country=France"
    assert bundle.cases[0].question.id == "trust"
    assert bundle.cases[0].gold.group_size == 50


def test_twin_and_mega_downloads_normalize_hub_rows(tmp_path, monkeypatch):
    full = [
        {
            "pid": "pid_7",
            "persona_summary": "Short summary",
            "persona_text": "Full card",
            "persona_json": json.dumps(PERSONA),
        }
    ]
    wave = [
        {
            "pid": "7",
            "wave1_3_persona_json": json.dumps(PERSONA),
            "wave4_Q_wave4_A": json.dumps(WAVE4),
        }
    ]

    def load_dataset(repo, config=None, split=None):
        assert repo == "LLM-Digital-Twin/Twin-2K-500"
        if config == "full_persona":
            return full
        if config == "wave_split":
            return wave
        raise AssertionError(config)

    _install_loader(monkeypatch, load_dataset)
    download_twin(tmp_path, limit=1)
    from popbench.datasets.twin2k_500.loader import load

    bundle = load(tmp_path)
    assert bundle.subjects[0].id == "7"
    assert bundle.subjects[0].persona_summary == "Short summary"
    assert "SECRET_PERSONA" not in (bundle.subjects[0].demographic_text or "")
    assert bundle.cases[0].gold.answer == "SECRET_GOLD"
    assert "Answer: SECRET_GOLD" not in bundle.cases[0].question.prompt

    mega_rows = {
        "privacy": [
            {
                "PID": "pid_7",
                "survey_text": "Study question",
                "survey_json_with_human_response": json.dumps({"answer": "SECRET_GOLD"}),
            }
        ]
    }
    _patch_hub(monkeypatch, mega_rows)
    download_mega(tmp_path, study="privacy", limit=1)
    from popbench.datasets.twin2k_mega_study.loader import load as load_mega

    mega = load_mega(tmp_path, study="privacy")
    assert mega.cases[0].subject.id == "7"
    assert "SECRET_GOLD" not in mega.cases[0].question.prompt


def _patch_hub(monkeypatch, splits):
    def load_dataset(repo, config=None, split=None):
        return splits

    _install_loader(monkeypatch, load_dataset)


def _install_loader(monkeypatch, load_dataset):
    import datasets as hf_datasets

    monkeypatch.setattr(hf_datasets, "load_dataset", load_dataset, raising=False)
