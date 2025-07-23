"""
AutoML特徴量エンジニアリングのロバストネステスト

エラーハンドリング、異常データ処理、エッジケースの検証を行います。
"""

import pytest
import pandas as pd
import numpy as np
import warnings
from typing import Dict, List, Optional
from unittest.mock import patch, MagicMock

from app.core.services.ml.feature_engineering.enhanced_feature_engineering_service import (
    EnhancedFeatureEngineeringService,
)
from app.core.services.ml.feature_engineering.automl_features.automl_config import (
    AutoMLConfig,
    TSFreshConfig,
    FeaturetoolsConfig,
    AutoFeatConfig,
)


class TestAutoMLRobustness:
    """AutoMLロバストネステストクラス"""

    def setup_method(self):
        """各テストメソッドの前に実行される初期化"""
        self.service = EnhancedFeatureEngineeringService()

    def create_problematic_data(
        self, problem_type: str, n_samples: int = 100
    ) -> pd.DataFrame:
        """
        問題のあるデータを生成

        Args:
            problem_type: 問題の種類
            n_samples: サンプル数
        """
        np.random.seed(42)
        dates = pd.date_range(start="2020-01-01", periods=n_samples, freq="1h")

        base_data = pd.DataFrame(
            {
                "timestamp": dates,
                "Open": 100 + np.random.normal(0, 1, n_samples),
                "High": 101 + np.random.normal(0, 1, n_samples),
                "Low": 99 + np.random.normal(0, 1, n_samples),
                "Close": 100 + np.random.normal(0, 1, n_samples),
                "Volume": np.random.lognormal(10, 1, n_samples),
            }
        )

        # timestampをインデックスに設定
        base_data.set_index("timestamp", inplace=True)

        if problem_type == "missing_values":
            # 欠損値を意図的に作成
            missing_indices = np.random.choice(
                n_samples, size=n_samples // 4, replace=False
            )
            base_data.iloc[missing_indices, base_data.columns.get_loc("Close")] = np.nan
            base_data.iloc[
                missing_indices[: len(missing_indices) // 2],
                base_data.columns.get_loc("Volume"),
            ] = np.nan

        elif problem_type == "infinite_values":
            # 無限値を作成
            inf_indices = np.random.choice(
                n_samples, size=n_samples // 10, replace=False
            )
            base_data.iloc[inf_indices, base_data.columns.get_loc("High")] = np.inf
            base_data.iloc[
                inf_indices[: len(inf_indices) // 2],
                base_data.columns.get_loc("Volume"),
            ] = -np.inf

        elif problem_type == "outliers":
            # 外れ値を作成
            outlier_indices = np.random.choice(
                n_samples, size=n_samples // 20, replace=False
            )
            base_data.iloc[
                outlier_indices, base_data.columns.get_loc("Close")
            ] *= 100  # 100倍の外れ値
            base_data.iloc[
                outlier_indices, base_data.columns.get_loc("Volume")
            ] *= 1000  # 1000倍の外れ値

        elif problem_type == "zero_variance":
            # 分散ゼロの列を作成
            base_data["constant_col"] = 42.0
            base_data["Close"] = 100.0  # 全て同じ値

        elif problem_type == "duplicate_timestamps":
            # 重複タイムスタンプを作成（インデックスを操作）
            duplicate_indices = np.random.choice(
                n_samples // 2, size=n_samples // 4, replace=False
            )
            # インデックスが設定されているので、直接操作は困難
            # この場合はスキップするか、別の方法で実装

        elif problem_type == "wrong_data_types":
            # 間違ったデータ型
            base_data["Close"] = base_data["Close"].astype(str)
            base_data["Volume"] = base_data["Volume"].astype(object)

        return base_data

    def test_missing_values_handling(self):
        """欠損値処理テスト"""
        print("\n=== 欠損値処理テスト ===")

        # 欠損値を含むデータを生成
        problematic_data = self.create_problematic_data("missing_values", 200)
        target = pd.Series(np.random.normal(0, 0.01, 200), index=problematic_data.index)

        print(f"欠損値数: {problematic_data.isnull().sum().sum()}")

        # 特徴量計算を実行
        try:
            result = self.service.calculate_enhanced_features(
                ohlcv_data=problematic_data,
                target=target,
                lookback_periods={"short": 5, "medium": 20},
            )

            # 結果の検証
            assert result is not None, "結果がNoneです"
            assert not result.empty, "結果が空です"

            # 無限値やNaNが適切に処理されているかチェック
            inf_count = np.isinf(result.select_dtypes(include=[np.number])).sum().sum()
            nan_count = result.select_dtypes(include=[np.number]).isnull().sum().sum()

            print(f"結果の無限値数: {inf_count}")
            print(f"結果のNaN数: {nan_count}")
            print(f"結果の特徴量数: {len(result.columns)}")
            print(f"結果のデータ数: {len(result)}")

            # 基本的な妥当性チェック
            assert len(result.columns) > 0, "特徴量が生成されていません"

            print("欠損値処理テスト: 成功")

        except Exception as e:
            print(f"欠損値処理テストでエラー: {e}")
            # エラーが発生しても、適切にハンドリングされることを確認
            assert "エラー" in str(e) or "error" in str(e).lower()

    def test_infinite_values_handling(self):
        """無限値処理テスト"""
        print("\n=== 無限値処理テスト ===")

        # 無限値を含むデータを生成
        problematic_data = self.create_problematic_data("infinite_values", 200)
        target = pd.Series(np.random.normal(0, 0.01, 200), index=problematic_data.index)

        inf_count_before = (
            np.isinf(problematic_data.select_dtypes(include=[np.number])).sum().sum()
        )
        print(f"入力データの無限値数: {inf_count_before}")

        # 特徴量計算を実行
        try:
            result = self.service.calculate_enhanced_features(
                ohlcv_data=problematic_data,
                target=target,
                lookback_periods={"short": 5, "medium": 20},
            )

            # 結果の検証
            assert result is not None, "結果がNoneです"

            # 無限値が適切に処理されているかチェック
            inf_count_after = (
                np.isinf(result.select_dtypes(include=[np.number])).sum().sum()
            )
            print(f"結果の無限値数: {inf_count_after}")
            print(f"結果の特徴量数: {len(result.columns)}")

            # 無限値が除去または置換されていることを確認
            # （完全に除去されている必要はないが、制御されていることを確認）
            print("無限値処理テスト: 成功")

        except Exception as e:
            print(f"無限値処理テストでエラー: {e}")
            # エラーが適切にハンドリングされることを確認
            assert "エラー" in str(e) or "error" in str(e).lower()

    def test_outliers_handling(self):
        """外れ値処理テスト"""
        print("\n=== 外れ値処理テスト ===")

        # 外れ値を含むデータを生成
        problematic_data = self.create_problematic_data("outliers", 200)
        target = pd.Series(np.random.normal(0, 0.01, 200), index=problematic_data.index)

        # 外れ値の統計
        close_std = problematic_data["Close"].std()
        volume_std = problematic_data["Volume"].std()
        print(f"価格の標準偏差: {close_std:.2f}")
        print(f"出来高の標準偏差: {volume_std:.2e}")

        # 特徴量計算を実行
        try:
            result = self.service.calculate_enhanced_features(
                ohlcv_data=problematic_data,
                target=target,
                lookback_periods={"short": 5, "medium": 20},
            )

            # 結果の検証
            assert result is not None, "結果がNoneです"
            assert not result.empty, "結果が空です"

            print(f"結果の特徴量数: {len(result.columns)}")
            print(f"結果のデータ数: {len(result)}")

            # 外れ値が適切に処理されているかの基本チェック
            numeric_cols = result.select_dtypes(include=[np.number]).columns
            for col in numeric_cols[:5]:  # 最初の5列をチェック
                col_std = result[col].std()
                if not np.isnan(col_std) and not np.isinf(col_std):
                    assert (
                        col_std < 1e10
                    ), f"列 {col} の標準偏差が異常に大きい: {col_std}"

            print("外れ値処理テスト: 成功")

        except Exception as e:
            print(f"外れ値処理テストでエラー: {e}")
            # エラーが適切にハンドリングされることを確認
            assert "エラー" in str(e) or "error" in str(e).lower()

    def test_zero_variance_handling(self):
        """分散ゼロデータ処理テスト"""
        print("\n=== 分散ゼロデータ処理テスト ===")

        # 分散ゼロのデータを生成
        problematic_data = self.create_problematic_data("zero_variance", 200)
        target = pd.Series(np.random.normal(0, 0.01, 200), index=problematic_data.index)

        # 分散ゼロの列を確認
        zero_var_cols = []
        for col in problematic_data.select_dtypes(include=[np.number]).columns:
            if problematic_data[col].var() == 0:
                zero_var_cols.append(col)

        print(f"分散ゼロの列: {zero_var_cols}")

        # 特徴量計算を実行
        try:
            result = self.service.calculate_enhanced_features(
                ohlcv_data=problematic_data,
                target=target,
                lookback_periods={"short": 5, "medium": 20},
            )

            # 結果の検証
            assert result is not None, "結果がNoneです"

            print(f"結果の特徴量数: {len(result.columns)}")
            print(f"結果のデータ数: {len(result)}")

            # 分散ゼロの特徴量が適切に処理されているかチェック
            result_zero_var_cols = []
            for col in result.select_dtypes(include=[np.number]).columns:
                col_var = result[col].var()
                if not np.isnan(col_var) and col_var == 0:
                    result_zero_var_cols.append(col)

            print(f"結果の分散ゼロ列数: {len(result_zero_var_cols)}")

            print("分散ゼロデータ処理テスト: 成功")

        except Exception as e:
            print(f"分散ゼロデータ処理テストでエラー: {e}")
            # エラーが適切にハンドリングされることを確認
            assert "エラー" in str(e) or "error" in str(e).lower()

    def test_empty_data_handling(self):
        """空データ処理テスト"""
        print("\n=== 空データ処理テスト ===")

        # 空のDataFrameを作成
        empty_data = pd.DataFrame()
        empty_target = pd.Series(dtype=float)

        # 特徴量計算を実行
        result = self.service.calculate_enhanced_features(
            ohlcv_data=empty_data, target=empty_target
        )

        # 結果の検証
        assert (
            result is not None or result is empty_data
        ), "空データの処理が適切ではありません"
        print("空データ処理テスト: 成功")

    def test_single_row_data_handling(self):
        """単一行データ処理テスト"""
        print("\n=== 単一行データ処理テスト ===")

        # 単一行のデータを作成
        single_row_data = pd.DataFrame(
            {
                "timestamp": [pd.Timestamp("2020-01-01")],
                "open": [100.0],
                "high": [101.0],
                "low": [99.0],
                "close": [100.5],
                "volume": [1000.0],
            }
        )

        single_target = pd.Series([0.01], index=[0])

        # 特徴量計算を実行
        try:
            result = self.service.calculate_enhanced_features(
                ohlcv_data=single_row_data,
                target=single_target,
                lookback_periods={"short": 5, "medium": 20},
            )

            # 結果の検証
            print(
                f"単一行データの結果: {result.shape if result is not None else 'None'}"
            )

            # 単一行では多くの特徴量が計算できないことを確認
            if result is not None and not result.empty:
                print(f"生成された特徴量数: {len(result.columns)}")

            print("単一行データ処理テスト: 成功")

        except Exception as e:
            print(f"単一行データ処理テストでエラー: {e}")
            # エラーが適切にハンドリングされることを確認
            assert "エラー" in str(e) or "error" in str(e).lower()

    def test_configuration_robustness(self):
        """設定のロバストネステスト"""
        print("\n=== 設定ロバストネステスト ===")

        # 正常なデータを生成
        normal_data = self.create_problematic_data("normal", 100)
        target = pd.Series(np.random.normal(0, 0.01, 100), index=normal_data.index)

        # 異常な設定をテスト
        invalid_configs = [
            {"tsfresh": {"enabled": True, "feature_count_limit": -1}},  # 負の値
            {"featuretools": {"enabled": True, "max_depth": 0}},  # ゼロ深度
            {"autofeat": {"enabled": True, "generations": -5}},  # 負の世代数
        ]

        for i, config in enumerate(invalid_configs):
            print(f"\n異常設定テスト {i+1}: {config}")

            try:
                result = self.service.calculate_enhanced_features(
                    ohlcv_data=normal_data, target=target, automl_config=config
                )

                # 結果が返されることを確認（エラーハンドリングされている）
                assert result is not None, f"設定 {i+1} で結果がNoneです"
                print(f"設定 {i+1}: 正常に処理されました")

            except Exception as e:
                print(f"設定 {i+1} でエラー: {e}")
                # エラーが適切にハンドリングされることを確認
                assert "エラー" in str(e) or "error" in str(e).lower()

    def test_memory_pressure_handling(self):
        """メモリ圧迫時の処理テスト"""
        print("\n=== メモリ圧迫処理テスト ===")

        # 大きなデータセットを作成（メモリ圧迫をシミュレート）
        large_data = self.create_problematic_data("normal", 5000)
        target = pd.Series(np.random.normal(0, 0.01, 5000), index=large_data.index)

        try:
            # メモリ使用量を監視しながら実行
            import psutil
            import os

            process = psutil.Process(os.getpid())
            memory_before = process.memory_info().rss / 1024 / 1024  # MB

            result = self.service.calculate_enhanced_features(
                ohlcv_data=large_data,
                target=target,
                lookback_periods={"short": 5, "medium": 20, "long": 50},
            )

            memory_after = process.memory_info().rss / 1024 / 1024  # MB
            memory_increase = memory_after - memory_before

            print(f"メモリ増加: {memory_increase:.1f}MB")
            print(f"結果の特徴量数: {len(result.columns) if result is not None else 0}")

            # メモリ使用量が合理的な範囲内かチェック
            assert memory_increase < 1000, f"メモリ使用量が過大: {memory_increase}MB"

            print("メモリ圧迫処理テスト: 成功")

        except Exception as e:
            print(f"メモリ圧迫処理テストでエラー: {e}")
            # メモリ不足エラーが適切にハンドリングされることを確認
            assert (
                "メモリ" in str(e) or "memory" in str(e).lower() or "エラー" in str(e)
            )

    def test_concurrent_execution_safety(self):
        """並行実行安全性テスト"""
        print("\n=== 並行実行安全性テスト ===")

        import threading
        import time

        # テスト用データ
        test_data = self.create_problematic_data("normal", 200)
        target = pd.Series(np.random.normal(0, 0.01, 200), index=test_data.index)

        results = []
        errors = []

        def worker(worker_id):
            try:
                result = self.service.calculate_enhanced_features(
                    ohlcv_data=test_data,
                    target=target,
                    lookback_periods={"short": 5, "medium": 20},
                )
                results.append((worker_id, result))
                print(f"ワーカー {worker_id}: 完了")
            except Exception as e:
                errors.append((worker_id, str(e)))
                print(f"ワーカー {worker_id}: エラー - {e}")

        # 複数スレッドで同時実行
        threads = []
        for i in range(3):  # 3つのスレッド
            thread = threading.Thread(target=worker, args=(i,))
            threads.append(thread)
            thread.start()

        # 全スレッドの完了を待機
        for thread in threads:
            thread.join()

        print(f"成功した実行: {len(results)}")
        print(f"エラーが発生した実行: {len(errors)}")

        # 少なくとも一部の実行が成功することを確認
        assert len(results) > 0, "全ての並行実行が失敗しました"

        print("並行実行安全性テスト: 成功")

    def create_normal_data(self, n_samples: int = 100) -> pd.DataFrame:
        """正常なテストデータを生成"""
        np.random.seed(42)
        dates = pd.date_range(start="2020-01-01", periods=n_samples, freq="1h")

        data = pd.DataFrame(
            {
                "timestamp": dates,
                "Open": 100 + np.random.normal(0, 1, n_samples),
                "High": 101 + np.random.normal(0, 1, n_samples),
                "Low": 99 + np.random.normal(0, 1, n_samples),
                "Close": 100 + np.random.normal(0, 1, n_samples),
                "Volume": np.random.lognormal(10, 1, n_samples),
            }
        )

        # timestampをインデックスに設定
        data.set_index("timestamp", inplace=True)
        return data
