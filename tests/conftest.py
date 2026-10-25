"""Keep paragraph tests on a temporary Jin10 store."""

import pytest

from recent_db import seed_flashes


@pytest.fixture(autouse=True)
def jin10_recent_db(tmp_path_factory, monkeypatch):
    # Keep this off the test's tmp_path. Some tests assert that directory's names.
    path = tmp_path_factory.mktemp("jin10-recent") / "jin10.sqlite"
    monkeypatch.setenv("POPBENCH_RECENT_DB", str(path))
    seed_flashes(path)
    return path
