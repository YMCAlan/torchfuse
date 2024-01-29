import pytest


def pytest_addoption(parser):
    parser.addoption("--slow-first",
                     action="store_true",
                     default=False,
                     help="run slow tests first")

    parser.addoption("--slow-last",
                     action="store_true",
                     default=False,
                     help="run slow tests last")


def pytest_collection_modifyitems(config, items):
    if config.getoption("--slow-first"):
        run_first = pytest.mark.order("first")
        for item in items:
            if item.get_closest_marker("slow"):
                item.add_marker(run_first)

    elif config.getoption("--slow-last"):
        run_last = pytest.mark.order("last")
        for item in items:
            if item.get_closest_marker("slow"):
                item.add_marker(run_last)
