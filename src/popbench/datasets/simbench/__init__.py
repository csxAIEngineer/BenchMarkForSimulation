"""SimBench group-level response distributions. The directory name is SimBench's package form.

The published benchmark is SimBench (not SimBechMark). Rows stay groups.
"""

from popbench.datasets.simbench.loader import SIMBENCH, download, load

__all__ = ["SIMBENCH", "download", "load"]
