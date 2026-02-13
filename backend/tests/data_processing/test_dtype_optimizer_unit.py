import pytest
import pandas as pd
import numpy as np
from app.utils.data_processing.transformers.dtype_optimizer import DtypeOptimizer


class TestDtypeOptimizerUnit:
    def test_optimization_logic(self):
        optimizer = DtypeOptimizer()

        df = pd.DataFrame(
            {
                "float_col": [1.0, 2.0, 3.0],  # float64
                "int8_col": [1, 50, 100],  # int64, but fits in int8
                "int16_col": [1, 500, 1000],  # int64, but fits in int16
                "int32_col": [1, 100000, 2000000],  # fits in int32
                "string_col": ["A", "B", "C"],
            }
        )

        # 明示的に型を指定
        df["float_col"] = df["float_col"].astype("float64")
        df["int8_col"] = df["int8_col"].astype("int64")

        optimizer.fit(df)
        result = optimizer.transform(df)

        assert result["float_col"].dtype == "float32"
        assert result["int8_col"].dtype == "int8"
        assert result["int16_col"].dtype == "int16"
        assert result["int32_col"].dtype == "int32"
        assert result["string_col"].dtype == "object" or isinstance(
            result["string_col"].dtype, pd.StringDtype
        )  # No change

    def test_non_dataframe_input(self):
        optimizer = DtypeOptimizer()
        data = [[1.0, 2.0], [3.0, 4.0]]
        optimizer.fit(data)
        result = optimizer.transform(data)
        assert isinstance(result, pd.DataFrame)
        assert result.iloc[:, 0].dtype == "float32"
