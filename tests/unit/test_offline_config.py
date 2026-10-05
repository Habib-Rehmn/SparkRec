import pytest
from pydantic import ValidationError

from sparkrec.config import OfflineSettings


@pytest.mark.parametrize(
    "values",
    [
        {"validation_start": 20},
        {"validation_start": 30, "test_start": 20},
        {"ranks": [0]},
        {"reg_params": [float("nan")]},
        {"reg_params": [-1]},
        {"initial_candidates": 10, "max_candidates": 5},
        {"ranks": [1, 2, 3], "reg_params": [0.1, 0.2, 0.3]},
    ],
)
def test_invalid_offline_settings(values):
    with pytest.raises(ValidationError):
        OfflineSettings(**values)
