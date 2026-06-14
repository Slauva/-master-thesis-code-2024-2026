from importlib.util import find_spec

import pytest


@pytest.mark.parametrize("package", ["mne", "numpy", "pydantic"])
def test_required_dataset_package_is_available(package: str) -> None:
    assert find_spec(package) is not None
