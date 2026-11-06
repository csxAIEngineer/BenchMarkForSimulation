"""Exact basic-info cards for the Twin-2K pricing task and BDM lettuce.

This package copies source cells onto a persona card. It does not call a
language model and it does not fill in missing attributes.
"""

from popbench.pricing_basic_info.assemble import assemble_dataset
from popbench.pricing_basic_info.cards import build_card
from popbench.pricing_basic_info.sample import planned_sample_size, stratified_sample

__all__ = [
    "assemble_dataset",
    "build_card",
    "planned_sample_size",
    "stratified_sample",
]
