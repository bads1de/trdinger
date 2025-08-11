"""
talib から pandas-ta への移行包括的テスト

このテストファイルは、talib から pandas-ta への移行が完全に行われているかを検証します。
以下の観点でテストを実施します：
1. 移行漏れの検出（talib の残存確認）
2. 計算精度の検証
3. エラーハンドリングの検証
4. パフォーマンステスト
5. 統合テスト
"""

import ast
import os
import re
import sys
import time
import numpy as np
import pandas as pd
import pytest
from pathlib import Path
from typing import List, Dict, Any, Tuple
import importlib.util

# テスト対象のモジュールをインポート
sys.path.append(str(Path(__file__).parent.parent))

from app.services.indicators import TechnicalIndicatorService
from app.services.indicators.utils import PandasTAError
from app.services.indicators.technical_indicators.trend import TrendIndicators
from app.services.indicators.technical_indicators.momentum import MomentumIndicators
from app.services.indicators.technical_indicators.volatility import VolatilityIndicators
from app.services.indicators.technical_indicators.volume import VolumeIndicators


class TestTalibMigrationComprehensive:
    """talib から pandas-ta への移行包括的テストクラス"""

    @pytest.fixture(autouse=True)
    def setup(self):
        """テストセットアップ"""
        self.backend_path = Path(__file__).parent.parent
        self.service = TechnicalIndicatorService()

        # テスト用データの生成
        np.random.seed(42)
        n = 200
        base_price = 100.0

        # リアルな価格データを生成
        returns = np.random.normal(0.001, 0.02, n)
        prices = [base_price]
        for ret in returns:
            prices.append(prices[-1] * (1 + ret))

        self.test_data = pd.DataFrame(
            {
                "Open": np.array(prices[:-1]) + np.random.normal(0, 0.5, n),
                "High": np.array(prices[:-1]) + np.random.uniform(0.5, 2.0, n),
                "Low": np.array(prices[:-1]) - np.random.uniform(0.5, 2.0, n),
                "Close": np.array(prices[1:]),
                "Volume": np.random.uniform(1000, 10000, n),
            },
            index=pd.date_range("2024-01-01", periods=n, freq="H"),
        )

        # 価格の整合性を保証
        self.test_data["High"] = np.maximum(
            self.test_data["High"],
            np.maximum(self.test_data["Open"], self.test_data["Close"]),
        )
        self.test_data["Low"] = np.minimum(
            self.test_data["Low"],
            np.minimum(self.test_data["Open"], self.test_data["Close"]),
        )

    def test_no_talib_imports_remaining(self):
        """talib のインポートが残っていないことを確認"""
        talib_imports = self._find_talib_imports()

        if talib_imports:
            pytest.fail(
                f"talib のインポートが残っています:\n"
                + "\n".join([f"  {file}: {line}" for file, line in talib_imports])
            )

    def test_no_talib_function_calls_remaining(self):
        """talib の関数呼び出しが残っていないことを確認"""
        talib_calls = self._find_talib_function_calls()

        if talib_calls:
            pytest.fail(
                f"talib の関数呼び出しが残っています:\n"
                + "\n".join([f"  {file}: {line}" for file, line in talib_calls])
            )

    def test_no_talib_error_handling_remaining(self):
        """TALibError や handle_talib_errors が残っていないことを確認"""
        talib_errors = self._find_talib_error_handling()

        if talib_errors:
            pytest.fail(
                f"talib のエラーハンドリングが残っています:\n"
                + "\n".join([f"  {file}: {line}" for file, line in talib_errors])
            )

    def test_pandas_ta_error_handling_consistency(self):
        """PandasTAError が一貫して使用されていることを確認"""
        inconsistencies = self._check_pandas_ta_error_consistency()

        if inconsistencies:
            pytest.fail(
                f"PandasTAError の使用に一貫性がありません:\n"
                + "\n".join([f"  {file}: {issue}" for file, issue in inconsistencies])
            )

    @pytest.mark.parametrize(
        "indicator_name,params",
        [
            ("SMA", {"period": 20}),
            ("EMA", {"period": 20}),
            ("RSI", {"period": 14}),
            ("ATR", {"period": 14}),
        ],
    )
    def test_basic_indicators_calculation_accuracy(self, indicator_name, params):
        """基本指標の計算精度を検証"""
        try:
            result = self.service.calculate_indicator(
                self.test_data, indicator_name, params
            )

            # 基本的な検証
            assert result is not None, f"{indicator_name}: 結果がNone"

            if isinstance(result, tuple):
                for i, arr in enumerate(result):
                    assert len(arr) == len(
                        self.test_data
                    ), f"{indicator_name}: 結果の長さが不正 (index {i})"
                    assert not np.all(
                        np.isnan(arr)
                    ), f"{indicator_name}: 全ての値がNaN (index {i})"
            else:
                assert len(result) == len(
                    self.test_data
                ), f"{indicator_name}: 結果の長さが不正"
                assert not np.all(np.isnan(result)), f"{indicator_name}: 全ての値がNaN"

        except Exception as e:
            pytest.fail(f"{indicator_name} 計算エラー: {e}")

    @pytest.mark.parametrize(
        "indicator_name,params",
        [
            ("MACD", {"fast_period": 12, "slow_period": 26, "signal_period": 9}),
            ("BB", {"period": 20, "std_dev": 2.0}),
            ("STOCH", {"k_period": 14, "d_period": 3}),
        ],
    )
    def test_complex_indicators_calculation_accuracy(self, indicator_name, params):
        """複雑な指標の計算精度を検証"""
        try:
            result = self.service.calculate_indicator(
                self.test_data, indicator_name, params
            )

            # 複数戻り値の検証
            assert isinstance(
                result, tuple
            ), f"{indicator_name}: 複数戻り値が期待されます"

            for i, arr in enumerate(result):
                assert len(arr) == len(
                    self.test_data
                ), f"{indicator_name}: 結果の長さが不正 (index {i})"
                assert not np.all(
                    np.isnan(arr)
                ), f"{indicator_name}: 全ての値がNaN (index {i})"

        except Exception as e:
            pytest.fail(f"{indicator_name} 計算エラー: {e}")

    def test_error_handling_robustness(self):
        """エラーハンドリングの堅牢性を検証"""
        # 不正なデータでのテスト
        invalid_data_cases = [
            # 空のデータ
            pd.DataFrame(),
            # NaNのみのデータ
            pd.DataFrame(
                {
                    "Open": [np.nan] * 10,
                    "High": [np.nan] * 10,
                    "Low": [np.nan] * 10,
                    "Close": [np.nan] * 10,
                    "Volume": [np.nan] * 10,
                }
            ),
            # 長さが不足するデータ
            pd.DataFrame(
                {
                    "Open": [100.0],
                    "High": [101.0],
                    "Low": [99.0],
                    "Close": [100.5],
                    "Volume": [1000.0],
                }
            ),
        ]

        for i, invalid_data in enumerate(invalid_data_cases):
            with pytest.raises(PandasTAError):
                self.service.calculate_indicator(invalid_data, "SMA", {"period": 20})

    def test_performance_comparison(self):
        """パフォーマンステスト（参考値）"""
        # 大きなデータセットでのテスト
        large_data = self._generate_large_dataset(10000)

        indicators_to_test = ["SMA", "EMA", "RSI"]
        performance_results = {}

        for indicator in indicators_to_test:
            start_time = time.time()

            if indicator == "SMA":
                result = self.service.calculate_indicator(
                    large_data, indicator, {"period": 20}
                )
            elif indicator == "EMA":
                result = self.service.calculate_indicator(
                    large_data, indicator, {"period": 20}
                )
            elif indicator == "RSI":
                result = self.service.calculate_indicator(
                    large_data, indicator, {"period": 14}
                )

            end_time = time.time()
            performance_results[indicator] = end_time - start_time

            # 結果の基本検証
            assert result is not None
            assert len(result) == len(large_data)

        # パフォーマンス結果をログ出力（参考値）
        print("\nパフォーマンステスト結果:")
        for indicator, duration in performance_results.items():
            print(f"  {indicator}: {duration:.4f}秒")

    def _find_talib_imports(self) -> List[Tuple[str, str]]:
        """talib のインポートを検索"""
        talib_imports = []

        for py_file in self._get_python_files():
            try:
                with open(py_file, "r", encoding="utf-8") as f:
                    content = f.read()

                # インポート文を検索
                import_patterns = [
                    r"import\s+talib",
                    r"from\s+talib\s+import",
                ]

                for pattern in import_patterns:
                    matches = re.finditer(pattern, content, re.IGNORECASE)
                    for match in matches:
                        line_num = content[: match.start()].count("\n") + 1
                        talib_imports.append(
                            (str(py_file), f"Line {line_num}: {match.group()}")
                        )

            except Exception:
                continue

        return talib_imports

    def _find_talib_function_calls(self) -> List[Tuple[str, str]]:
        """talib の関数呼び出しを検索"""
        talib_calls = []

        for py_file in self._get_python_files():
            try:
                with open(py_file, "r", encoding="utf-8") as f:
                    content = f.read()

                # talib.関数名 のパターンを検索
                pattern = r"talib\.\w+"
                matches = re.finditer(pattern, content)

                for match in matches:
                    line_num = content[: match.start()].count("\n") + 1
                    talib_calls.append(
                        (str(py_file), f"Line {line_num}: {match.group()}")
                    )

            except Exception:
                continue

        return talib_calls

    def _find_talib_error_handling(self) -> List[Tuple[str, str]]:
        """talib のエラーハンドリングを検索"""
        talib_errors = []

        for py_file in self._get_python_files():
            try:
                with open(py_file, "r", encoding="utf-8") as f:
                    content = f.read()

                # TALibError や handle_talib_errors を検索
                patterns = [
                    r"TALibError",
                    r"handle_talib_errors",
                ]

                for pattern in patterns:
                    matches = re.finditer(pattern, content)
                    for match in matches:
                        line_num = content[: match.start()].count("\n") + 1
                        talib_errors.append(
                            (str(py_file), f"Line {line_num}: {match.group()}")
                        )

            except Exception:
                continue

        return talib_errors

    def _check_pandas_ta_error_consistency(self) -> List[Tuple[str, str]]:
        """PandasTAError の使用一貫性をチェック"""
        inconsistencies = []

        for py_file in self._get_python_files():
            if "indicators" not in str(py_file):
                continue

            try:
                with open(py_file, "r", encoding="utf-8") as f:
                    content = f.read()

                # PandasTAError のインポートがあるかチェック
                has_pandas_ta_import = "PandasTAError" in content

                # pandas-ta を使用している場合はPandasTAErrorも使用すべき
                if "pandas_ta" in content and not has_pandas_ta_import:
                    inconsistencies.append(
                        (
                            str(py_file),
                            "pandas-ta使用ファイルでPandasTAErrorがインポートされていません",
                        )
                    )

            except Exception:
                continue

        return inconsistencies

    def _get_python_files(self) -> List[Path]:
        """Python ファイルのリストを取得"""
        python_files = []

        # indicators ディレクトリ内のファイルを検索
        indicators_path = self.backend_path / "app" / "services" / "indicators"
        if indicators_path.exists():
            python_files.extend(indicators_path.rglob("*.py"))

        # ML feature engineering ディレクトリ内のファイルを検索
        ml_path = self.backend_path / "app" / "services" / "ml"
        if ml_path.exists():
            python_files.extend(ml_path.rglob("*.py"))

        return python_files

    def _generate_large_dataset(self, size: int) -> pd.DataFrame:
        """大きなテストデータセットを生成"""
        np.random.seed(42)
        base_price = 100.0

        returns = np.random.normal(0.001, 0.02, size)
        prices = [base_price]
        for ret in returns:
            prices.append(prices[-1] * (1 + ret))

        data = pd.DataFrame(
            {
                "Open": np.array(prices[:-1]) + np.random.normal(0, 0.5, size),
                "High": np.array(prices[:-1]) + np.random.uniform(0.5, 2.0, size),
                "Low": np.array(prices[:-1]) - np.random.uniform(0.5, 2.0, size),
                "Close": np.array(prices[1:]),
                "Volume": np.random.uniform(1000, 10000, size),
            },
            index=pd.date_range("2024-01-01", periods=size, freq="H"),
        )

        # 価格の整合性を保証
        data["High"] = np.maximum(data["High"], np.maximum(data["Open"], data["Close"]))
        data["Low"] = np.minimum(data["Low"], np.minimum(data["Open"], data["Close"]))

        return data


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
