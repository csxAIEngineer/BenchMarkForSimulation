"""Dataset layer: download, cache, and load a unified bundle.

Each dataset has its own package. Loaders return ``DatasetBundle`` and do not
call an LLM.
"""

from __future__ import annotations

from popbench.datasets.bdm_lettuce.loader import BDM_LETTUCE as BDM_LETTUCE
from popbench.datasets.simbench.loader import SIMBENCH as SIMBENCH
from popbench.datasets.twin2k_500.loader import TWIN2K_500 as TWIN2K_500
from popbench.datasets.twin2k_mega_study.loader import TWIN2K_MEGA_STUDY as TWIN2K_MEGA_STUDY

DATASETS = {
    TWIN2K_500.name: TWIN2K_500,
    TWIN2K_MEGA_STUDY.name: TWIN2K_MEGA_STUDY,
    SIMBENCH.name: SIMBENCH,
    BDM_LETTUCE.name: BDM_LETTUCE,
}

# Accept the spelling used in papers and the user's earlier notes.
ALIASES = {
    "twin-2k-500": TWIN2K_500.name,
    "twin2k": TWIN2K_500.name,
    "mega": TWIN2K_MEGA_STUDY.name,
    "mega-study": TWIN2K_MEGA_STUDY.name,
    "twin-2k-500-mega-study": TWIN2K_MEGA_STUDY.name,
    "simbench": SIMBENCH.name,
    "SimBench": SIMBENCH.name,
    "simbechmark": SIMBENCH.name,
    "SimBechMark": SIMBENCH.name,
    "canada_bdm": BDM_LETTUCE.name,
    "canadian_bdm": BDM_LETTUCE.name,
}


def resolve_dataset(name: str) -> str:
    """Map a user-facing dataset label onto a registry key."""
    if name in DATASETS:
        return name
    key = ALIASES.get(name) or ALIASES.get(name.strip())
    if key is None:
        known = ", ".join(sorted(DATASETS))
        raise KeyError(f"unknown dataset {name!r}. Known datasets: {known}")
    return key


__all__ = [
    "ALIASES",
    "DATASETS",
    "BDM_LETTUCE",
    "SIMBENCH",
    "TWIN2K_500",
    "TWIN2K_MEGA_STUDY",
    "resolve_dataset",
]
