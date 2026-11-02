"""Population simulation benchmark, split into three layers.

- ``popbench.datasets`` loads Twin-2K-500, Twin-2K-500-Mega-Study, and SimBench.
- ``popbench.models`` turns a case into a persona prompt. One directory per method.
- ``popbench.comms`` sends that prompt to an LLM provider.

The package name stays ``popbench`` so it does not shadow the Hugging Face
``datasets`` library.
"""

__version__ = "0.2.0"

__all__ = ["__version__"]
