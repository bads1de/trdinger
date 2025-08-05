"""
エッジケーステストスイート

境界値や異常なデータでの動作を検証し、潜在的な問題を発見します。
"""

import logging
import os
import sys
import time
import warnings
from dataclasses import dataclass
from typing import Any, Dict, List, Optional
from unittest.mock import Mock, patch

import numpy as np
import pandas as pd

# プロジェクトルートをPythonパスに追加
backend_path = os.path.dirname(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
)
if backend_path not in sys.path:
    sys.path.insert(0, backend_path)

# 警告を抑制
warnings.filterwarnings("ignore")

# ログ設定
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


@dataclass
class EdgeCaseTestResult:
    """エッジケーステスト結果"""

    test_name: str
    component_name: str
    success: bool
    execution_time: float
    edge_case_type: str
    error_message: Optional[str] = None
    data_size: int = 0
    memory_usage_mb: float = 0.0
    robustness_score: float = 0.0


class EdgeCaseTestSuite:
    """エッジケーステストスイート"""

    def __init__(self):
        self.results: List[EdgeCaseTestResult] = []

    def create_empty_data(self) -> pd.DataFrame:
        """空のデータセットを作成"""
        return pd.DataFrame()

    def create_single_row_data(self) -> pd.DataFrame:
        """単一行のデータセットを作成"""
        return pd.DataFrame(
            {
                "timestamp": [pd.Timestamp("2024-01-01")],
                "Open": [100.0],
                "High": [105.0],
                "Low": [95.0],
                "Close": [102.0],
                "Volume": [1000.0],
            }
        )

    def create_extreme_values_data(self) -> pd.DataFrame:
        """極端な値を含むデータセットを作成"""
        return pd.DataFrame(
            {
                "timestamp": pd.date_range("2024-01-01", periods=10, freq="1H"),
                "Open": [1e10, -1e10, 0, np.inf, -np.inf, np.nan, 1e-10, -1e-10, 1, -1],
                "High": [1e10, -1e10, 0, np.inf, -np.inf, np.nan, 1e-10, -1e-10, 1, -1],
                "Low": [1e10, -1e10, 0, np.inf, -np.inf, np.nan, 1e-10, -1e-10, 1, -1],
                "Close": [
                    1e10,
                    -1e10,
                    0,
                    np.inf,
                    -np.inf,
                    np.nan,
                    1e-10,
                    -1e-10,
                    1,
                    -1,
                ],
                "Volume": [1e10, 0, np.inf, np.nan, 1e-10, 1, 0, 0, 0, 0],
            }
        )

    def create_invalid_ohlc_data(self) -> pd.DataFrame:
        """論理的に無効なOHLCデータを作成"""
        return pd.DataFrame(
            {
                "timestamp": pd.date_range("2024-01-01", periods=5, freq="1H"),
                "Open": [100, 100, 100, 100, 100],
                "High": [90, 100, 110, 100, 100],  # High < Open
                "Low": [110, 100, 90, 100, 100],  # Low > Open
                "Close": [105, 100, 95, 100, 100],
                "Volume": [-100, 0, 1000, np.nan, np.inf],  # 負のボリューム
            }
        )

    def create_duplicate_timestamps_data(self) -> pd.DataFrame:
        """重複したタイムスタンプを含むデータを作成"""
        timestamps = ["2024-01-01 00:00:00"] * 5
        return pd.DataFrame(
            {
                "timestamp": pd.to_datetime(timestamps),
                "Open": [100, 101, 102, 103, 104],
                "High": [105, 106, 107, 108, 109],
                "Low": [95, 96, 97, 98, 99],
                "Close": [102, 103, 104, 105, 106],
                "Volume": [1000, 1100, 1200, 1300, 1400],
            }
        )

    def create_all_nan_columns_data(self) -> pd.DataFrame:
        """すべてNaNのカラムを含むデータを作成"""
        return pd.DataFrame(
            {
                "timestamp": pd.date_range("2024-01-01", periods=10, freq="1H"),
                "Open": [100] * 10,
                "High": [105] * 10,
                "Low": [95] * 10,
                "Close": [102] * 10,
                "Volume": [1000] * 10,
                "all_nan_column": [np.nan] * 10,
                "another_nan_column": [np.nan] * 10,
            }
        )

    def test_empty_data_handling(self):
        """空データの処理テスト"""
        logger.info("🔍 空データ処理テスト開始")

        start_time = time.time()
        robustness_score = 0.0

        try:
            empty_data = self.create_empty_data()

            # 特徴量エンジニアリングサービスのテスト
            try:
                from app.services.ml.feature_engineering.feature_engineering_service import (
                    FeatureEngineeringService,
                )

                fe_service = FeatureEngineeringService()
                result = fe_service.calculate_advanced_features(empty_data)
                robustness_score += 25.0
                logger.info("✅ 特徴量エンジニアリング: 空データを適切に処理")
            except Exception as e:
                logger.warning(f"❌ 特徴量エンジニアリング: 空データ処理失敗 - {e}")

            # データプロセッサのテスト
            try:
                from app.utils.data_processing import DataProcessor

                processor = DataProcessor()
                processed = processor.preprocess_features(empty_data)
                robustness_score += 25.0
                logger.info("✅ データプロセッサ: 空データを適切に処理")
            except Exception as e:
                logger.warning(f"❌ データプロセッサ: 空データ処理失敗 - {e}")

            # ラベル生成器のテスト
            try:
                from app.utils.label_generation import LabelGenerator

                label_gen = LabelGenerator()
                labels = label_gen.generate_labels(empty_data)
                robustness_score += 25.0
                logger.info("✅ ラベル生成器: 空データを適切に処理")
            except Exception as e:
                logger.warning(f"❌ ラベル生成器: 空データ処理失敗 - {e}")

            # バリデーターのテスト
            try:
                from app.utils.data_validation import DataValidator

                validator = DataValidator()
                validation_result = validator.validate_ohlcv_data(empty_data)
                robustness_score += 25.0
                logger.info("✅ データバリデーター: 空データを適切に処理")
            except Exception as e:
                logger.warning(f"❌ データバリデーター: 空データ処理失敗 - {e}")

            execution_time = time.time() - start_time

            self.results.append(
                EdgeCaseTestResult(
                    test_name="空データ処理",
                    component_name="全コンポーネント",
                    success=robustness_score > 50.0,
                    execution_time=execution_time,
                    edge_case_type="empty_data",
                    data_size=0,
                    robustness_score=robustness_score,
                )
            )

            logger.info(f"✅ 空データ処理テスト完了: 堅牢性スコア {robustness_score}%")

        except Exception as e:
            execution_time = time.time() - start_time

            self.results.append(
                EdgeCaseTestResult(
                    test_name="空データ処理",
                    component_name="全コンポーネント",
                    success=False,
                    execution_time=execution_time,
                    edge_case_type="empty_data",
                    error_message=str(e),
                    data_size=0,
                    robustness_score=0.0,
                )
            )

            logger.error(f"❌ 空データ処理テスト失敗: {e}")

    def test_extreme_values_handling(self):
        """極端な値の処理テスト"""
        logger.info("🔍 極端な値処理テスト開始")

        start_time = time.time()
        robustness_score = 0.0

        try:
            extreme_data = self.create_extreme_values_data()

            # 特徴量エンジニアリングのテスト
            try:
                from app.services.ml.feature_engineering.feature_engineering_service import (
                    FeatureEngineeringService,
                )

                fe_service = FeatureEngineeringService()
                result = fe_service.calculate_advanced_features(extreme_data)

                # 結果の検証
                if not result.isin([np.inf, -np.inf]).any().any():
                    robustness_score += 30.0
                    logger.info("✅ 特徴量エンジニアリング: 無限大値を適切に処理")
                else:
                    logger.warning("⚠️ 特徴量エンジニアリング: 無限大値が残存")

                if not result.isna().all().any():
                    robustness_score += 20.0
                    logger.info("✅ 特徴量エンジニアリング: NaN値を適切に処理")
                else:
                    logger.warning("⚠️ 特徴量エンジニアリング: 全NaNカラムが存在")

            except Exception as e:
                logger.warning(f"❌ 特徴量エンジニアリング: 極端値処理失敗 - {e}")

            # データプロセッサのテスト
            try:
                from app.utils.data_processing import DataProcessor

                processor = DataProcessor()
                processed = processor.preprocess_features(extreme_data)

                # 外れ値除去の確認
                if processed.shape[0] < extreme_data.shape[0]:
                    robustness_score += 25.0
                    logger.info("✅ データプロセッサ: 外れ値を適切に除去")

                # スケーリングの確認
                numeric_cols = processed.select_dtypes(include=[np.number]).columns
                if len(numeric_cols) > 0:
                    if processed[numeric_cols].std().max() < 10:
                        robustness_score += 25.0
                        logger.info("✅ データプロセッサ: スケーリングが適切")

            except Exception as e:
                logger.warning(f"❌ データプロセッサ: 極端値処理失敗 - {e}")

            execution_time = time.time() - start_time

            self.results.append(
                EdgeCaseTestResult(
                    test_name="極端値処理",
                    component_name="全コンポーネント",
                    success=robustness_score > 50.0,
                    execution_time=execution_time,
                    edge_case_type="extreme_values",
                    data_size=len(extreme_data),
                    robustness_score=robustness_score,
                )
            )

            logger.info(f"✅ 極端値処理テスト完了: 堅牢性スコア {robustness_score}%")

        except Exception as e:
            execution_time = time.time() - start_time

            self.results.append(
                EdgeCaseTestResult(
                    test_name="極端値処理",
                    component_name="全コンポーネント",
                    success=False,
                    execution_time=execution_time,
                    edge_case_type="extreme_values",
                    error_message=str(e),
                    data_size=10,
                    robustness_score=0.0,
                )
            )

            logger.error(f"❌ 極端値処理テスト失敗: {e}")

    def test_invalid_ohlc_logic(self):
        """論理的に無効なOHLCデータの処理テスト"""
        logger.info("🔍 無効OHLC論理テスト開始")

        start_time = time.time()
        robustness_score = 0.0

        try:
            invalid_data = self.create_invalid_ohlc_data()

            # データバリデーターのテスト
            try:
                from app.utils.data_validation import DataValidator

                validator = DataValidator()
                validation_result = validator.validate_ohlcv_data(invalid_data)

                if not validation_result.get("is_valid", True):
                    robustness_score += 50.0
                    logger.info("✅ データバリデーター: 無効なOHLCを適切に検出")
                else:
                    logger.warning("⚠️ データバリデーター: 無効なOHLCを検出できず")

            except Exception as e:
                logger.warning(f"❌ データバリデーター: 無効OHLC検証失敗 - {e}")

            # 特徴量エンジニアリングのテスト
            try:
                from app.services.ml.feature_engineering.feature_engineering_service import (
                    FeatureEngineeringService,
                )

                fe_service = FeatureEngineeringService()
                result = fe_service.calculate_advanced_features(invalid_data)

                # 結果が生成されたかチェック
                if len(result) > 0:
                    robustness_score += 30.0
                    logger.info("✅ 特徴量エンジニアリング: 無効データでも処理継続")

                # 異常値が適切に処理されているかチェック
                if not result.isin([np.inf, -np.inf]).any().any():
                    robustness_score += 20.0
                    logger.info("✅ 特徴量エンジニアリング: 異常値を適切に処理")

            except Exception as e:
                logger.warning(f"❌ 特徴量エンジニアリング: 無効OHLC処理失敗 - {e}")

            execution_time = time.time() - start_time

            self.results.append(
                EdgeCaseTestResult(
                    test_name="無効OHLC論理",
                    component_name="データバリデーター",
                    success=robustness_score > 50.0,
                    execution_time=execution_time,
                    edge_case_type="invalid_ohlc",
                    data_size=len(invalid_data),
                    robustness_score=robustness_score,
                )
            )

            logger.info(f"✅ 無効OHLC論理テスト完了: 堅牢性スコア {robustness_score}%")

        except Exception as e:
            execution_time = time.time() - start_time

            self.results.append(
                EdgeCaseTestResult(
                    test_name="無効OHLC論理",
                    component_name="データバリデーター",
                    success=False,
                    execution_time=execution_time,
                    edge_case_type="invalid_ohlc",
                    error_message=str(e),
                    data_size=5,
                    robustness_score=0.0,
                )
            )

            logger.error(f"❌ 無効OHLC論理テスト失敗: {e}")

    def test_duplicate_timestamps(self):
        """重複タイムスタンプの処理テスト"""
        logger.info("🔍 重複タイムスタンプテスト開始")

        start_time = time.time()
        robustness_score = 0.0

        try:
            duplicate_data = self.create_duplicate_timestamps_data()

            # データ頻度マネージャーのテスト
            try:
                from app.services.ml.feature_engineering.data_frequency_manager import (
                    DataFrequencyManager,
                )

                freq_manager = DataFrequencyManager()
                validation_result = freq_manager.validate_data_consistency(
                    duplicate_data, None, None, "1h"
                )

                if not validation_result.get("is_valid", True):
                    robustness_score += 40.0
                    logger.info("✅ データ頻度マネージャー: 重複タイムスタンプを検出")
                else:
                    logger.warning(
                        "⚠️ データ頻度マネージャー: 重複タイムスタンプを検出できず"
                    )

            except Exception as e:
                logger.warning(f"❌ データ頻度マネージャー: 重複検証失敗 - {e}")

            # データプロセッサのテスト
            try:
                from app.utils.data_processing import DataProcessor

                processor = DataProcessor()
                processed = processor.preprocess_features(duplicate_data)

                # 重複が除去されたかチェック
                if len(processed) < len(duplicate_data):
                    robustness_score += 30.0
                    logger.info("✅ データプロセッサ: 重複データを適切に除去")

                # データの整合性チェック
                if not processed.duplicated().any():
                    robustness_score += 30.0
                    logger.info("✅ データプロセッサ: 重複除去後のデータが整合")

            except Exception as e:
                logger.warning(f"❌ データプロセッサ: 重複処理失敗 - {e}")

            execution_time = time.time() - start_time

            self.results.append(
                EdgeCaseTestResult(
                    test_name="重複タイムスタンプ",
                    component_name="データ頻度マネージャー",
                    success=robustness_score > 50.0,
                    execution_time=execution_time,
                    edge_case_type="duplicate_timestamps",
                    data_size=len(duplicate_data),
                    robustness_score=robustness_score,
                )
            )

            logger.info(
                f"✅ 重複タイムスタンプテスト完了: 堅牢性スコア {robustness_score}%"
            )

        except Exception as e:
            execution_time = time.time() - start_time

            self.results.append(
                EdgeCaseTestResult(
                    test_name="重複タイムスタンプ",
                    component_name="データ頻度マネージャー",
                    success=False,
                    execution_time=execution_time,
                    edge_case_type="duplicate_timestamps",
                    error_message=str(e),
                    data_size=5,
                    robustness_score=0.0,
                )
            )

            logger.error(f"❌ 重複タイムスタンプテスト失敗: {e}")

    def test_all_nan_columns(self):
        """全NaNカラムの処理テスト"""
        logger.info("🔍 全NaNカラムテスト開始")

        start_time = time.time()
        robustness_score = 0.0

        try:
            nan_data = self.create_all_nan_columns_data()

            # データプロセッサのテスト
            try:
                from app.utils.data_processing import DataProcessor

                processor = DataProcessor()
                processed = processor.preprocess_features(nan_data)

                # 全NaNカラムが除去されたかチェック
                nan_columns_before = nan_data.isna().all().sum()
                nan_columns_after = processed.isna().all().sum()

                if nan_columns_after < nan_columns_before:
                    robustness_score += 40.0
                    logger.info("✅ データプロセッサ: 全NaNカラムを適切に除去")
                else:
                    logger.warning("⚠️ データプロセッサ: 全NaNカラムが残存")

                # 欠損値補完の確認
                if processed.isna().sum().sum() < nan_data.isna().sum().sum():
                    robustness_score += 30.0
                    logger.info("✅ データプロセッサ: 欠損値を適切に補完")

            except Exception as e:
                logger.warning(f"❌ データプロセッサ: 全NaN処理失敗 - {e}")

            # 特徴量エンジニアリングのテスト
            try:
                from app.services.ml.feature_engineering.feature_engineering_service import (
                    FeatureEngineeringService,
                )

                fe_service = FeatureEngineeringService()
                result = fe_service.calculate_advanced_features(nan_data)

                # 結果にNaNが含まれていないかチェック
                if not result.isna().any().any():
                    robustness_score += 30.0
                    logger.info("✅ 特徴量エンジニアリング: NaN値を適切に処理")

            except Exception as e:
                logger.warning(f"❌ 特徴量エンジニアリング: 全NaN処理失敗 - {e}")

            execution_time = time.time() - start_time

            self.results.append(
                EdgeCaseTestResult(
                    test_name="全NaNカラム",
                    component_name="データプロセッサ",
                    success=robustness_score > 50.0,
                    execution_time=execution_time,
                    edge_case_type="all_nan_columns",
                    data_size=len(nan_data),
                    robustness_score=robustness_score,
                )
            )

            logger.info(f"✅ 全NaNカラムテスト完了: 堅牢性スコア {robustness_score}%")

        except Exception as e:
            execution_time = time.time() - start_time

            self.results.append(
                EdgeCaseTestResult(
                    test_name="全NaNカラム",
                    component_name="データプロセッサ",
                    success=False,
                    execution_time=execution_time,
                    edge_case_type="all_nan_columns",
                    error_message=str(e),
                    data_size=10,
                    robustness_score=0.0,
                )
            )

            logger.error(f"❌ 全NaNカラムテスト失敗: {e}")

    def run_all_tests(self):
        """すべてのエッジケーステストを実行"""
        logger.info("🚀 エッジケーステストスイート開始")

        self.test_empty_data_handling()
        self.test_extreme_values_handling()
        self.test_invalid_ohlc_logic()
        self.test_duplicate_timestamps()
        self.test_all_nan_columns()

        # 結果の集計
        total_tests = len(self.results)
        successful_tests = sum(1 for result in self.results if result.success)
        total_execution_time = sum(result.execution_time for result in self.results)
        average_robustness = (
            sum(result.robustness_score for result in self.results) / total_tests
            if total_tests > 0
            else 0
        )

        logger.info("=" * 80)
        logger.info("🔍 エッジケーステスト結果")
        logger.info("=" * 80)
        logger.info(f"📊 総テスト数: {total_tests}")
        logger.info(f"✅ 成功: {successful_tests}")
        logger.info(f"❌ 失敗: {total_tests - successful_tests}")
        logger.info(f"📈 成功率: {successful_tests / total_tests * 100:.1f}%")
        logger.info(f"🛡️ 平均堅牢性スコア: {average_robustness:.1f}%")
        logger.info(f"⏱️ 総実行時間: {total_execution_time:.2f}秒")

        logger.info("\n🔍 エッジケーステスト詳細:")
        for result in self.results:
            status = "✅" if result.success else "❌"
            logger.info(f"{status} {result.test_name}")
            logger.info(f"   コンポーネント: {result.component_name}")
            logger.info(f"   実行時間: {result.execution_time:.2f}秒")
            logger.info(f"   エッジケース種別: {result.edge_case_type}")
            logger.info(f"   データサイズ: {result.data_size}")
            logger.info(f"   堅牢性スコア: {result.robustness_score:.1f}%")
            if result.error_message:
                logger.info(f"   エラー: {result.error_message[:100]}...")

        logger.info("=" * 80)
        logger.info("🎯 エッジケーステストスイート完了")

        return self.results


if __name__ == "__main__":
    suite = EdgeCaseTestSuite()
    results = suite.run_all_tests()
