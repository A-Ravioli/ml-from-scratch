"""pytest plugin (auto-registered when `mlfs` is installed): the `nb` fixture.

Every topic test takes one fixture, `nb`: the notebook module under test.
`test_<name>.py` tests `<name>.py` in the same directory (your code), or
`solutions/<name>.py` when MLFS_SOLUTIONS=1.

Other `mlfs` modules a notebook imports resolve to the reference solutions during tests
(MLFS_REFERENCE defaults to `*`), so a bug in an earlier chapter can't fail a later one.
Set MLFS_REFERENCE=none to test against your own earlier code instead.
"""
import os

import pytest

from mlfs import notebooks


def pytest_configure(config):
    os.environ.setdefault("MLFS_REFERENCE", "*")
    config.addinivalue_line("markers", "slow: takes more than a few seconds; skipped by live notebook checks")


@pytest.fixture
def nb(request):
    test_file = request.path
    name = test_file.stem.removeprefix("test_")
    return notebooks.load(notebooks.resolve(test_file.parent, name))
