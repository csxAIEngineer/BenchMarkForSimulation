"""Ask the CR API to answer pricing questions as a basic-info persona.

The persona text comes from the local prepro outputs. Held-out human answers
are not placed in the request. ``cr_api.CRClient`` is the only model client.
"""

from popbench.simulate.prompt import build_messages
from popbench.simulate.run import run_simulation

__all__ = ["build_messages", "run_simulation"]
