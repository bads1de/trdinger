"""
talib から pandas-ta への移行エッジケーステスト

このテストファイルは、移行後のシステムのエッジケースを検証します。
以下の観点でテストを実施します：
1. 異常データでのテスト
2. 境界値テスト
3. エラー条件テスト
4. パフォーマンス限界テスト
"""

import sys
import numpy as np
import pandas as pd
import pytest
from pathlib import Path
from typing import Dict, Any, List
import warnings

# テスト対象のモジュールをインポート
sys.path.append(str(Path(__file__).parent.parent))

from app.services.indicators import TechnicalIndicatorService
from app.services.indicators.utils import PandasTAError


class TestMigrationEdgeCases:
    """移行エッジケーステストクラス"""

    @pytest.fixture(autouse=True)
    def setup(self):
        """テストセットアップ"""
        self.service = TechnicalIndicatorService()

    def test_empty_dataframe(self):
        """空のDataFrameでのテスト"""
        empty_df = pd.DataFrame()

        with pytest.raises(PandasTAError):
            self.service.calculate_indicator(empty_df, "SMA", {"period": 20})

    def test_single_row_dataframe(self):
        """1行のDataFrameでのテスト"""
        single_row_df = pd.DataFrame(
            {
                "Open": [100.0],
                "High": [101.0],
                "Low": [99.0],
                "Close": [100.5],
                "Volume": [1000.0],
            }
        )

        with pytest.raises(PandasTAError):
            self.service.calculate_indicator(single_row_df, "SMA", {"period": 20})

    def test_insufficient_data_length(self):
        """データ長が不足している場合のテスト"""
        short_df = pd.DataFrame(
            {
                "Open": [100.0, 101.0, 102.0],
                "High": [101.0, 102.0, 103.0],
                "Low": [99.0, 100.0, 101.0],
                "Close": [100.5, 101.5, 102.5],
                "Volume": [1000.0, 1100.0, 1200.0],
            }
        )

        # 期間がデータ長より長い場合
        with pytest.raises(PandasTAError):
            self.service.calculate_indicator(short_df, "SMA", {"period": 20})

    def test_all_nan_data(self):
        """全てNaNのデータでのテスト"""
        nan_df = pd.DataFrame(
            {
                "Open": [np.nan] * 100,
                "High": [np.nan] * 100,
                "Low": [np.nan] * 100,
                "Close": [np.nan] * 100,
                "Volume": [np.nan] * 100,
            }
        )

        with pytest.raises(PandasTAError):
            self.service.calculate_indicator(nan_df, "SMA", {"period": 20})

    def test_partial_nan_data(self):
        """部分的にNaNを含むデータでのテスト"""
        # 正常なデータを作成
        n = 100
        data = pd.DataFrame(
            {
                "Open": np.random.uniform(90, 110, n),
                "High": np.random.uniform(100, 120, n),
                "Low": np.random.uniform(80, 100, n),
                "Close": np.random.uniform(95, 105, n),
                "Volume": np.random.uniform(1000, 5000, n),
            }
        )

        # 一部にNaNを挿入
        data.iloc[10:20, :] = np.nan

        # 計算は成功するが、結果にNaNが含まれることを確認
        result = self.service.calculate_indicator(data, "SMA", {"period": 20})
        assert isinstance(result, np.ndarray)
        assert len(result) == len(data)
        # NaNが含まれていることを確認
        assert np.any(np.isnan(result))

    def test_infinite_values(self):
        """無限大値を含むデータでのテスト"""
        n = 100
        data = pd.DataFrame(
            {
                "Open": np.random.uniform(90, 110, n),
                "High": np.random.uniform(100, 120, n),
                "Low": np.random.uniform(80, 100, n),
                "Close": np.random.uniform(95, 105, n),
                "Volume": np.random.uniform(1000, 5000, n),
            }
        )

        # 無限大値を挿入
        data.iloc[50, 0] = np.inf
        data.iloc[51, 1] = -np.inf

        # エラーが発生するか、適切に処理されることを確認
        try:
            result = self.service.calculate_indicator(data, "SMA", {"period": 20})
            # 結果に無限大値が含まれていないことを確認
            assert not np.any(np.isinf(result))
        except PandasTAError:
            # エラーが発生することも許容される
            pass

    def test_zero_values(self):
        """ゼロ値を含むデータでのテスト"""
        n = 100
        data = pd.DataFrame(
            {
                "Open": np.random.uniform(90, 110, n),
                "High": np.random.uniform(100, 120, n),
                "Low": np.random.uniform(80, 100, n),
                "Close": np.random.uniform(95, 105, n),
                "Volume": np.random.uniform(1000, 5000, n),
            }
        )

        # ゼロ値を挿入
        data.iloc[50:55, :] = 0.0

        # 計算が成功することを確認
        result = self.service.calculate_indicator(data, "SMA", {"period": 20})
        assert isinstance(result, np.ndarray)
        assert len(result) == len(data)

    def test_negative_values(self):
        """負の値を含むデータでのテスト"""
        n = 100
        data = pd.DataFrame(
            {
                "Open": np.random.uniform(-10, 10, n),
                "High": np.random.uniform(0, 20, n),
                "Low": np.random.uniform(-20, 0, n),
                "Close": np.random.uniform(-5, 15, n),
                "Volume": np.random.uniform(1000, 5000, n),  # Volumeは正の値
            }
        )

        # 価格の整合性を保証
        data["High"] = np.maximum(data["High"], np.maximum(data["Open"], data["Close"]))
        data["Low"] = np.minimum(data["Low"], np.minimum(data["Open"], data["Close"]))

        # 計算が成功することを確認
        result = self.service.calculate_indicator(data, "SMA", {"period": 20})
        assert isinstance(result, np.ndarray)
        assert len(result) == len(data)

    def test_very_large_values(self):
        """非常に大きな値でのテスト"""
        n = 100
        large_value = 1e10

        data = pd.DataFrame(
            {
                "Open": np.random.uniform(large_value * 0.9, large_value * 1.1, n),
                "High": np.random.uniform(large_value, large_value * 1.2, n),
                "Low": np.random.uniform(large_value * 0.8, large_value, n),
                "Close": np.random.uniform(large_value * 0.95, large_value * 1.05, n),
                "Volume": np.random.uniform(1000, 5000, n),
            }
        )

        # 価格の整合性を保証
        data["High"] = np.maximum(data["High"], np.maximum(data["Open"], data["Close"]))
        data["Low"] = np.minimum(data["Low"], np.minimum(data["Open"], data["Close"]))

        # 計算が成功することを確認
        result = self.service.calculate_indicator(data, "SMA", {"period": 20})
        assert isinstance(result, np.ndarray)
        assert len(result) == len(data)
        assert not np.any(np.isnan(result[20:]))  # 初期のNaNを除く

    def test_very_small_values(self):
        """非常に小さな値でのテスト"""
        n = 100
        small_value = 1e-10

        data = pd.DataFrame(
            {
                "Open": np.random.uniform(small_value * 0.9, small_value * 1.1, n),
                "High": np.random.uniform(small_value, small_value * 1.2, n),
                "Low": np.random.uniform(small_value * 0.8, small_value, n),
                "Close": np.random.uniform(small_value * 0.95, small_value * 1.05, n),
                "Volume": np.random.uniform(1000, 5000, n),
            }
        )

        # 価格の整合性を保証
        data["High"] = np.maximum(data["High"], np.maximum(data["Open"], data["Close"]))
        data["Low"] = np.minimum(data["Low"], np.minimum(data["Open"], data["Close"]))

        # 計算が成功することを確認
        result = self.service.calculate_indicator(data, "SMA", {"period": 20})
        assert isinstance(result, np.ndarray)
        assert len(result) == len(data)

    def test_invalid_parameters(self):
        """無効なパラメータでのテスト"""
        data = self._create_normal_data(100)

        # 負の期間
        with pytest.raises(PandasTAError):
            self.service.calculate_indicator(data, "SMA", {"period": -5})

        # ゼロの期間
        with pytest.raises(PandasTAError):
            self.service.calculate_indicator(data, "SMA", {"period": 0})

        # 非整数の期間
        with pytest.raises((PandasTAError, TypeError, ValueError)):
            self.service.calculate_indicator(data, "SMA", {"period": 20.5})

    def test_missing_required_columns(self):
        """必要なカラムが不足している場合のテスト"""
        # Closeカラムのみのデータ
        incomplete_data = pd.DataFrame({"Close": np.random.uniform(95, 105, 100)})

        # SMAは動作するはず（Closeのみ必要）
        result = self.service.calculate_indicator(
            incomplete_data, "SMA", {"period": 20}
        )
        assert isinstance(result, np.ndarray)

        # ATRは失敗するはず（High, Low, Closeが必要）
        with pytest.raises(PandasTAError):
            self.service.calculate_indicator(incomplete_data, "ATR", {"period": 14})

    def test_wrong_column_names(self):
        """間違ったカラム名でのテスト"""
        data = pd.DataFrame(
            {
                "open": np.random.uniform(90, 110, 100),  # 小文字
                "high": np.random.uniform(100, 120, 100),
                "low": np.random.uniform(80, 100, 100),
                "close": np.random.uniform(95, 105, 100),
                "volume": np.random.uniform(1000, 5000, 100),
            }
        )

        # 大文字のカラム名が期待される場合、エラーが発生するはず
        with pytest.raises(PandasTAError):
            self.service.calculate_indicator(data, "SMA", {"period": 20})

    def test_extreme_parameter_values(self):
        """極端なパラメータ値でのテスト"""
        data = self._create_normal_data(1000)

        # 非常に大きな期間
        with pytest.raises(PandasTAError):
            self.service.calculate_indicator(data, "SMA", {"period": 2000})

        # データ長と同じ期間
        with pytest.raises(PandasTAError):
            self.service.calculate_indicator(data, "SMA", {"period": 1000})

    def test_data_type_consistency(self):
        """データ型の一貫性テスト"""
        # 整数型のデータ
        int_data = pd.DataFrame(
            {
                "Open": np.random.randint(90, 110, 100),
                "High": np.random.randint(100, 120, 100),
                "Low": np.random.randint(80, 100, 100),
                "Close": np.random.randint(95, 105, 100),
                "Volume": np.random.randint(1000, 5000, 100),
            }
        )

        # 価格の整合性を保証
        int_data["High"] = np.maximum(
            int_data["High"], np.maximum(int_data["Open"], int_data["Close"])
        )
        int_data["Low"] = np.minimum(
            int_data["Low"], np.minimum(int_data["Open"], int_data["Close"])
        )

        # 計算が成功し、結果がfloat型であることを確認
        result = self.service.calculate_indicator(int_data, "SMA", {"period": 20})
        assert isinstance(result, np.ndarray)
        assert result.dtype in [np.float64, np.float32]

    def test_unicode_and_special_characters(self):
        """Unicode文字や特殊文字を含むインデックスでのテスト"""
        data = self._create_normal_data(100)

        # 特殊文字を含むインデックス
        special_index = [f"時刻_{i}_🕐" for i in range(100)]
        data.index = special_index

        # 計算が成功することを確認
        result = self.service.calculate_indicator(data, "SMA", {"period": 20})
        assert isinstance(result, np.ndarray)
        assert len(result) == len(data)

    def test_memory_stress(self):
        """メモリストレステスト"""
        # 大きなデータセットでのテスト
        large_data = self._create_normal_data(100000)

        # メモリ使用量を監視しながら計算
        import psutil

        process = psutil.Process()
        initial_memory = process.memory_info().rss

        result = self.service.calculate_indicator(large_data, "SMA", {"period": 20})

        final_memory = process.memory_info().rss
        memory_increase = (final_memory - initial_memory) / 1024 / 1024  # MB

        # 結果の検証
        assert isinstance(result, np.ndarray)
        assert len(result) == len(large_data)

        # メモリ使用量が合理的な範囲内であることを確認
        assert (
            memory_increase < 1000
        ), f"メモリ使用量が過大です: {memory_increase:.2f}MB"

    def _create_normal_data(self, n: int) -> pd.DataFrame:
        """正常なテストデータを作成"""
        np.random.seed(42)

        data = pd.DataFrame(
            {
                "Open": np.random.uniform(90, 110, n),
                "High": np.random.uniform(100, 120, n),
                "Low": np.random.uniform(80, 100, n),
                "Close": np.random.uniform(95, 105, n),
                "Volume": np.random.uniform(1000, 5000, n),
            }
        )

        # 価格の整合性を保証
        data["High"] = np.maximum(data["High"], np.maximum(data["Open"], data["Close"]))
        data["Low"] = np.minimum(data["Low"], np.minimum(data["Open"], data["Close"]))

        return data


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
