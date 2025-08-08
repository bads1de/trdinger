from datetime import datetime
import numpy as np
import pandas as pd
from database.repositories.backtest_result_repository import BacktestResultRepository


class DummySession:
    def add(self, *args, **kwargs):
        pass

    def commit(self):
        pass

    def rollback(self):
        pass

    def refresh(self, *args, **kwargs):
        pass


def test_to_json_safe_various_types():
    repo = BacktestResultRepository(DummySession())

    data = {
        "a": datetime(2020, 1, 1),
        "b": np.int64(5),
        "c": np.array([1, 2, 3]),
        "d": pd.Timestamp("2021-05-05T00:00:00Z"),
        "e": [datetime(2020, 1, 2), np.float64(1.23)],
        "f": {"x": np.bool_(True)},
    }

    safe = repo._to_json_safe(data)

    assert isinstance(safe["a"], str)
    assert safe["b"] == 5
    assert safe["c"] == [1, 2, 3]
    assert isinstance(safe["d"], str)
    assert isinstance(safe["e"][0], str)
    assert isinstance(safe["e"][1], float)
    assert safe["f"]["x"] is True

