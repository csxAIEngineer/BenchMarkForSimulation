"""Load and validate `configs/v1.yaml`."""

from dataclasses import dataclass
from pathlib import Path

import yaml

from popbench.schema import TWIN2K_N

SUITES = ("decision", "sentiment", "both")
PERSONA_CONDITIONS = ("full_twin2k", "demographics_only", "nemotron_usa")
_FIELDS = (
    "suite",
    "persona_condition",
    "n_people",
    "full_n",
    "item_cap",
    "seed",
    "data_dir",
)


class ConfigError(ValueError):
    """The benchmark config is missing a field or has a value this runner rejects."""


@dataclass(frozen=True)
class BenchConfig:
    suite: str
    persona_condition: str
    n_people: int
    full_n: bool
    item_cap: int | None
    seed: int
    data_dir: Path

    @property
    def effective_n_people(self) -> int:
        """Sample size for this run. `full_n` selects every Twin-2K respondent."""
        if self.full_n:
            return TWIN2K_N
        return self.n_people


def load_config(path: Path) -> BenchConfig:
    path = Path(path)
    if not path.is_file():
        raise ConfigError(f"config not found: {path}")
    raw = yaml.safe_load(path.read_text())
    if raw is None:
        raw = {}
    if not isinstance(raw, dict):
        raise ConfigError(f"config must be a mapping: {path}")
    unknown = sorted(set(raw) - set(_FIELDS))
    if unknown:
        raise ConfigError(f"unknown config fields: {', '.join(unknown)}")

    suite = raw.get("suite", "both")
    persona_condition = raw.get("persona_condition", "full_twin2k")
    n_people = raw.get("n_people", 50)
    full_n = raw.get("full_n", False)
    item_cap = raw.get("item_cap", None)
    seed = raw.get("seed", 0)
    data_dir = raw.get("data_dir", "data")

    if suite not in SUITES:
        raise ConfigError(f"suite must be one of {', '.join(SUITES)}")
    if persona_condition not in PERSONA_CONDITIONS:
        raise ConfigError(
            "persona_condition must be one of " + ", ".join(PERSONA_CONDITIONS)
        )
    if isinstance(full_n, str) or not isinstance(full_n, bool):
        raise ConfigError("full_n must be true or false")
    if isinstance(n_people, bool) or not isinstance(n_people, int) or n_people < 1:
        raise ConfigError("n_people must be a positive integer")
    if (
        not full_n
        and persona_condition != "nemotron_usa"
        and n_people > TWIN2K_N
    ):
        raise ConfigError(
            f"n_people cannot exceed {TWIN2K_N} when the persona condition "
            "is a Twin-2K respondent. Set full_n: true for the full sample, "
            "or lower n_people."
        )
    if item_cap is not None and (
        isinstance(item_cap, bool) or not isinstance(item_cap, int) or item_cap < 1
    ):
        raise ConfigError("item_cap must be a positive integer or null")
    if isinstance(seed, bool) or not isinstance(seed, int) or seed < 0:
        raise ConfigError("seed must be a non-negative integer")
    if not isinstance(data_dir, str) or not data_dir:
        raise ConfigError("data_dir must be a path")

    data_path = Path(data_dir)
    if not data_path.is_absolute():
        data_path = Path.cwd() / data_path

    return BenchConfig(
        suite=suite,
        persona_condition=persona_condition,
        n_people=n_people,
        full_n=full_n,
        item_cap=item_cap,
        seed=seed,
        data_dir=data_path,
    )
