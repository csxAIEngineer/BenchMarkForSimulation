"""Exact basic-info cards for the Twin-2K pricing task and BDM lettuce.

This package copies source cells onto a persona card. It does not call a
language model and it does not fill in missing attributes.
"""

from popbench.prepro.assemble import assemble_dataset
from popbench.prepro.cards import build_card
from popbench.prepro.sample import planned_sample_size, stratified_sample

__all__ = [
    "assemble_dataset",
    "build_card",
    "planned_sample_size",
    "stratified_sample",
]
