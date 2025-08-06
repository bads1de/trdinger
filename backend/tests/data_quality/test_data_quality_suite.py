"""
データ品質テストスイート

データの整合性、統計的特性、時系列の連続性などを検証し、
機械学習の精度に影響する潜在的な問題を発見します。
"""

import logging
import os
import sys
import time
import warnings
from dataclasses import dataclass
from typing import Dict, List, Optional

import numpy as np
import pandas as pd
from scipy import stats

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
class DataQualityTestResult:
    """データ品質テスト結果"""

    test_name: str
    component_name: str
    success: bool
    execution_time: float
    quality_score: float
    data_size: int
    issues_found: List[str]
    error_message: Optional[str] = None
    statistical_metrics: Optional[Dict[str, float]] = None


class DataQualityTestSuite:
    """データ品質テストスイート"""

    def __init__(self):
        self.results: List[DataQualityTestResult] = []

    def create_quality_test_data(self, size: int = 1000) -> pd.DataFrame:
        """品質テスト用のデータを作成"""
        np.random.seed(42)
        dates = pd.date_range("2024-01-01", periods=size, freq="1H")

        # 基本的な価格データ
        base_price = 100
        price_changes = np.random.normal(0, 0.02, size)
        prices = [base_price]

        for change in price_changes[1:]:
            prices.append(prices[-1] * (1 + change))

        prices = np.array(prices)

        return pd.DataFrame(
            {
                "timestamp": dates,
                "Open": prices,
                "High": prices * (1 + np.abs(np.random.normal(0, 0.01, size))),
                "Low": prices * (1 - np.abs(np.random.normal(0, 0.01, size))),
                "Close": prices * (1 + np.random.normal(0, 0.005, size)),
                "Volume": np.random.lognormal(10, 1, size),
            }
        )

    def create_corrupted_data(self, size: int = 1000) -> pd.DataFrame:
        """品質問題を含むデータを作成"""
        base_data = self.create_quality_test_data(size)

        # データ品質問題を意図的に導入
        corrupted_data = base_data.copy()

        # 1. 価格の論理的不整合
        corrupted_data.loc[10:20, "High"] = corrupted_data.loc[10:20, "Low"] * 0.9
        corrupted_data.loc[30:40, "Low"] = corrupted_data.loc[30:40, "High"] * 1.1

        # 2. 異常な価格ジャンプ
        corrupted_data.loc[50:55, "Close"] = corrupted_data.loc[50:55, "Close"] * 10

        # 3. 負のボリューム
        corrupted_data.loc[70:80, "Volume"] = -corrupted_data.loc[70:80, "Volume"]

        # 4. 欠損値のクラスター
        corrupted_data.loc[100:120, ["Open", "High", "Low", "Close"]] = np.nan

        # 5. 重複したタイムスタンプ
        corrupted_data.loc[200:205, "timestamp"] = corrupted_data.loc[200, "timestamp"]

        return corrupted_data

    def test_data_consistency(self):
        """データ整合性テスト"""
        logger.info("🔍 データ整合性テスト開始")

        start_time = time.time()
        quality_score = 100.0
        issues_found = []

        try:
            # 正常データと破損データの両方をテスト
            normal_data = self.create_quality_test_data(500)
            corrupted_data = self.create_corrupted_data(500)

            # データバリデーターのテスト
            try:
                from app.utils.data_validation import DataValidator

                validator = DataValidator()

                # 正常データの検証
                normal_result = validator.validate_ohlcv_data(normal_data)
                if normal_result.get("is_valid", False):
                    logger.info("✅ 正常データを正しく検証")
                else:
                    quality_score -= 20.0
                    issues_found.append("正常データが無効と判定された")

                # 破損データの検証
                corrupted_result = validator.validate_ohlcv_data(corrupted_data)
                if not corrupted_result.get("is_valid", True):
                    logger.info("✅ 破損データを正しく検出")
                else:
                    quality_score -= 30.0
                    issues_found.append("破損データが有効と判定された")

                # エラー詳細の確認
                errors = corrupted_result.get("errors", [])
                if len(errors) > 0:
                    logger.info(f"✅ {len(errors)}個のデータ品質問題を検出")
                else:
                    quality_score -= 25.0
                    issues_found.append("データ品質問題が検出されなかった")

            except Exception as e:
                quality_score -= 50.0
                issues_found.append(f"データバリデーター実行エラー: {e}")

            # OHLC論理整合性チェック
            try:
                # High >= max(Open, Close) のチェック
                ohlc_issues = 0
                for data, name in [(normal_data, "正常"), (corrupted_data, "破損")]:
                    high_violations = (
                        data["High"] < np.maximum(data["Open"], data["Close"])
                    ).sum()
                    low_violations = (
                        data["Low"] > np.minimum(data["Open"], data["Close"])
                    ).sum()

                    if name == "正常" and (high_violations > 0 or low_violations > 0):
                        quality_score -= 15.0
                        issues_found.append(
                            f"正常データにOHLC論理違反: H={high_violations}, L={low_violations}"
                        )

                    if name == "破損":
                        ohlc_issues = high_violations + low_violations

                if ohlc_issues > 0:
                    logger.info(f"✅ {ohlc_issues}個のOHLC論理違反を検出")
                else:
                    quality_score -= 10.0
                    issues_found.append("OHLC論理違反が検出されなかった")

            except Exception as e:
                quality_score -= 25.0
                issues_found.append(f"OHLC整合性チェックエラー: {e}")

            execution_time = time.time() - start_time

            self.results.append(
                DataQualityTestResult(
                    test_name="データ整合性",
                    component_name="DataValidator",
                    success=quality_score > 70.0,
                    execution_time=execution_time,
                    quality_score=quality_score,
                    data_size=len(normal_data) + len(corrupted_data),
                    issues_found=issues_found,
                )
            )

            logger.info(f"✅ データ整合性テスト完了: 品質スコア {quality_score:.1f}%")

        except Exception as e:
            execution_time = time.time() - start_time

            self.results.append(
                DataQualityTestResult(
                    test_name="データ整合性",
                    component_name="DataValidator",
                    success=False,
                    execution_time=execution_time,
                    quality_score=0.0,
                    data_size=0,
                    issues_found=["テスト実行エラー"],
                    error_message=str(e),
                )
            )

            logger.error(f"❌ データ整合性テスト失敗: {e}")

    def test_statistical_anomalies(self):
        """統計的異常値検出テスト"""
        logger.info("🔍 統計的異常値検出テスト開始")

        start_time = time.time()
        quality_score = 100.0
        issues_found = []
        statistical_metrics = {}

        try:
            test_data = self.create_corrupted_data(1000)

            # 統計的異常値の検出
            try:
                from app.utils.data_processing import DataProcessor

                processor = DataProcessor()

                # 価格データの統計分析
                price_columns = ["Open", "High", "Low", "Close"]
                for col in price_columns:
                    if col in test_data.columns:
                        data_series = test_data[col].dropna()
                        if len(data_series) > 0:
                            # Z-score による異常値検出
                            z_scores = np.abs(stats.zscore(data_series))
                            outliers = (z_scores > 3).sum()
                            statistical_metrics[f"{col}_outliers"] = outliers

                            # 変動係数の計算
                            cv = data_series.std() / data_series.mean()
                            statistical_metrics[f"{col}_cv"] = cv

                            if outliers > len(data_series) * 0.1:  # 10%以上が異常値
                                quality_score -= 15.0
                                issues_found.append(
                                    f"{col}に過度の異常値: {outliers}個"
                                )

                # ボリュームの異常値検出
                if "Volume" in test_data.columns:
                    volume_data = test_data["Volume"].dropna()
                    negative_volume = (volume_data < 0).sum()
                    if negative_volume > 0:
                        quality_score -= 20.0
                        issues_found.append(f"負のボリューム: {negative_volume}個")
                        statistical_metrics["negative_volume_count"] = negative_volume

                    # ボリュームの対数正規性テスト
                    if len(volume_data[volume_data > 0]) > 10:
                        log_volume = np.log(volume_data[volume_data > 0])
                        _, p_value = stats.normaltest(log_volume)
                        statistical_metrics["volume_lognormal_pvalue"] = p_value

                        if p_value < 0.01:
                            quality_score -= 10.0
                            issues_found.append("ボリュームが対数正規分布に従わない")

                # データプロセッサによる前処理テスト
                processed_data = processor.preprocess_features(test_data)

                # 前処理後の品質チェック
                if processed_data.isna().any().any():
                    remaining_nan = processed_data.isna().sum().sum()
                    if remaining_nan > 0:
                        quality_score -= 15.0
                        issues_found.append(f"前処理後にNaN残存: {remaining_nan}個")

                # 無限大値のチェック
                if (
                    np.isinf(processed_data.select_dtypes(include=[np.number]))
                    .any()
                    .any()
                ):
                    quality_score -= 20.0
                    issues_found.append("前処理後に無限大値が残存")

                logger.info("✅ 統計的異常値検出完了")

            except Exception as e:
                quality_score -= 50.0
                issues_found.append(f"統計的異常値検出エラー: {e}")

            execution_time = time.time() - start_time

            self.results.append(
                DataQualityTestResult(
                    test_name="統計的異常値検出",
                    component_name="DataProcessor",
                    success=quality_score > 70.0,
                    execution_time=execution_time,
                    quality_score=quality_score,
                    data_size=len(test_data),
                    issues_found=issues_found,
                    statistical_metrics=statistical_metrics,
                )
            )

            logger.info(
                f"✅ 統計的異常値検出テスト完了: 品質スコア {quality_score:.1f}%"
            )

        except Exception as e:
            execution_time = time.time() - start_time

            self.results.append(
                DataQualityTestResult(
                    test_name="統計的異常値検出",
                    component_name="DataProcessor",
                    success=False,
                    execution_time=execution_time,
                    quality_score=0.0,
                    data_size=0,
                    issues_found=["テスト実行エラー"],
                    error_message=str(e),
                )
            )

            logger.error(f"❌ 統計的異常値検出テスト失敗: {e}")

    def test_time_series_continuity(self):
        """時系列データの連続性テスト"""
        logger.info("🔍 時系列連続性テスト開始")

        start_time = time.time()
        quality_score = 100.0
        issues_found = []
        statistical_metrics = {}

        try:
            # 連続性問題を含むデータを作成
            base_data = self.create_quality_test_data(1000)

            # 時系列の問題を導入
            discontinuous_data = base_data.copy()

            # 1. タイムスタンプのギャップ
            discontinuous_data = discontinuous_data.drop(index=range(100, 150))

            # 2. 重複タイムスタンプ
            duplicate_rows = discontinuous_data.iloc[200:205].copy()
            discontinuous_data = pd.concat(
                [discontinuous_data, duplicate_rows], ignore_index=True
            )

            # 3. 逆順のタイムスタンプ
            discontinuous_data.loc[300:310, "timestamp"] = discontinuous_data.loc[
                300:310, "timestamp"
            ] - pd.Timedelta(days=1)

            # データ頻度マネージャーのテスト
            try:
                from app.services.ml.feature_engineering.data_frequency_manager import (
                    DataFrequencyManager,
                )

                freq_manager = DataFrequencyManager()

                # 正常データの検証
                normal_result = freq_manager.validate_data_consistency(
                    base_data, None, None, "1h"
                )

                if normal_result.get("is_valid", False):
                    logger.info("✅ 正常な時系列データを正しく検証")
                else:
                    quality_score -= 20.0
                    issues_found.append("正常な時系列データが無効と判定")

                # 不連続データの検証
                discontinuous_result = freq_manager.validate_data_consistency(
                    discontinuous_data, None, None, "1h"
                )

                if not discontinuous_result.get("is_valid", True):
                    logger.info("✅ 不連続な時系列データを正しく検出")
                else:
                    quality_score -= 30.0
                    issues_found.append("不連続な時系列データが有効と判定")

            except Exception as e:
                quality_score -= 40.0
                issues_found.append(f"データ頻度マネージャーエラー: {e}")

            # 時系列の統計的特性チェック
            try:
                # タイムスタンプの間隔チェック
                time_diffs = base_data["timestamp"].diff().dropna()
                expected_interval = pd.Timedelta(hours=1)

                # 間隔の一貫性
                irregular_intervals = (time_diffs != expected_interval).sum()
                statistical_metrics["irregular_intervals"] = irregular_intervals

                if irregular_intervals > 0:
                    quality_score -= 15.0
                    issues_found.append(f"不規則な時間間隔: {irregular_intervals}個")

                # 重複タイムスタンプのチェック
                duplicates = discontinuous_data["timestamp"].duplicated().sum()
                statistical_metrics["duplicate_timestamps"] = duplicates

                if duplicates > 0:
                    logger.info(f"✅ 重複タイムスタンプを検出: {duplicates}個")
                else:
                    quality_score -= 10.0
                    issues_found.append("重複タイムスタンプが検出されなかった")

                # 時系列の順序チェック
                unsorted_count = (
                    discontinuous_data["timestamp"].diff() < pd.Timedelta(0)
                ).sum()
                statistical_metrics["unsorted_timestamps"] = unsorted_count

                if unsorted_count > 0:
                    logger.info(f"✅ 逆順タイムスタンプを検出: {unsorted_count}個")

            except Exception as e:
                quality_score -= 30.0
                issues_found.append(f"時系列統計チェックエラー: {e}")

            execution_time = time.time() - start_time

            self.results.append(
                DataQualityTestResult(
                    test_name="時系列連続性",
                    component_name="DataFrequencyManager",
                    success=quality_score > 70.0,
                    execution_time=execution_time,
                    quality_score=quality_score,
                    data_size=len(discontinuous_data),
                    issues_found=issues_found,
                    statistical_metrics=statistical_metrics,
                )
            )

            logger.info(f"✅ 時系列連続性テスト完了: 品質スコア {quality_score:.1f}%")

        except Exception as e:
            execution_time = time.time() - start_time

            self.results.append(
                DataQualityTestResult(
                    test_name="時系列連続性",
                    component_name="DataFrequencyManager",
                    success=False,
                    execution_time=execution_time,
                    quality_score=0.0,
                    data_size=0,
                    issues_found=["テスト実行エラー"],
                    error_message=str(e),
                )
            )

            logger.error(f"❌ 時系列連続性テスト失敗: {e}")

    def test_feature_distribution_stability(self):
        """特徴量分布の安定性テスト"""
        logger.info("🔍 特徴量分布安定性テスト開始")

        start_time = time.time()
        quality_score = 100.0
        issues_found = []
        statistical_metrics = {}

        try:
            # 2つの期間のデータを作成（分布ドリフトをシミュレート）
            period1_data = self.create_quality_test_data(500)
            period2_data = self.create_quality_test_data(500)

            # 期間2のデータに分布ドリフトを導入
            period2_data["Close"] = period2_data["Close"] * 1.5  # 価格レベルの変化
            period2_data["Volume"] = period2_data["Volume"] * 0.7  # ボリュームの変化

            # 特徴量エンジニアリングの実行
            try:
                from app.services.ml.feature_engineering.feature_engineering_service import (
                    FeatureEngineeringService,
                )

                fe_service = FeatureEngineeringService()

                features1 = fe_service.calculate_advanced_features(period1_data)
                features2 = fe_service.calculate_advanced_features(period2_data)

                # 共通の特徴量カラムを取得
                common_features = set(features1.columns) & set(features2.columns)
                numeric_features = [
                    col
                    for col in common_features
                    if features1[col].dtype in [np.float64, np.int64]
                ]

                # 分布の比較
                distribution_shifts = 0
                for feature in numeric_features[:10]:  # 最初の10個の特徴量をテスト
                    try:
                        data1 = features1[feature].dropna()
                        data2 = features2[feature].dropna()

                        if len(data1) > 10 and len(data2) > 10:
                            # Kolmogorov-Smirnov テスト
                            ks_stat, p_value = stats.ks_2samp(data1, data2)
                            statistical_metrics[f"{feature}_ks_pvalue"] = p_value

                            if p_value < 0.01:  # 有意な分布の違い
                                distribution_shifts += 1

                            # 平均と分散の変化
                            mean_change = abs(data2.mean() - data1.mean()) / (
                                data1.std() + 1e-8
                            )
                            var_change = abs(data2.var() - data1.var()) / (
                                data1.var() + 1e-8
                            )

                            statistical_metrics[f"{feature}_mean_change"] = mean_change
                            statistical_metrics[f"{feature}_var_change"] = var_change

                    except Exception as e:
                        logger.warning(f"特徴量 {feature} の分布比較でエラー: {e}")

                statistical_metrics["distribution_shifts"] = distribution_shifts

                if (
                    distribution_shifts > len(numeric_features) * 0.3
                ):  # 30%以上で分布シフト
                    quality_score -= 30.0
                    issues_found.append(
                        f"過度の分布シフト: {distribution_shifts}個の特徴量"
                    )
                elif distribution_shifts > 0:
                    logger.info(f"✅ 分布シフトを検出: {distribution_shifts}個の特徴量")

                # 特徴量の相関関係の安定性
                if len(numeric_features) > 5:
                    corr1 = features1[numeric_features[:5]].corr()
                    corr2 = features2[numeric_features[:5]].corr()

                    # 相関行列の差
                    corr_diff = np.abs(corr1 - corr2).mean().mean()
                    statistical_metrics["correlation_stability"] = 1 - corr_diff

                    if corr_diff > 0.2:
                        quality_score -= 20.0
                        issues_found.append(f"相関関係の不安定性: 差={corr_diff:.3f}")

            except Exception as e:
                quality_score -= 50.0
                issues_found.append(f"特徴量エンジニアリングエラー: {e}")

            execution_time = time.time() - start_time

            self.results.append(
                DataQualityTestResult(
                    test_name="特徴量分布安定性",
                    component_name="FeatureEngineeringService",
                    success=quality_score > 70.0,
                    execution_time=execution_time,
                    quality_score=quality_score,
                    data_size=len(period1_data) + len(period2_data),
                    issues_found=issues_found,
                    statistical_metrics=statistical_metrics,
                )
            )

            logger.info(
                f"✅ 特徴量分布安定性テスト完了: 品質スコア {quality_score:.1f}%"
            )

        except Exception as e:
            execution_time = time.time() - start_time

            self.results.append(
                DataQualityTestResult(
                    test_name="特徴量分布安定性",
                    component_name="FeatureEngineeringService",
                    success=False,
                    execution_time=execution_time,
                    quality_score=0.0,
                    data_size=0,
                    issues_found=["テスト実行エラー"],
                    error_message=str(e),
                )
            )

            logger.error(f"❌ 特徴量分布安定性テスト失敗: {e}")

    def run_all_tests(self):
        """すべてのデータ品質テストを実行"""
        logger.info("🚀 データ品質テストスイート開始")

        self.test_data_consistency()
        self.test_statistical_anomalies()
        self.test_time_series_continuity()
        self.test_feature_distribution_stability()

        # 結果の集計
        total_tests = len(self.results)
        successful_tests = sum(1 for result in self.results if result.success)
        total_execution_time = sum(result.execution_time for result in self.results)
        average_quality = (
            sum(result.quality_score for result in self.results) / total_tests
            if total_tests > 0
            else 0
        )
        total_issues = sum(len(result.issues_found) for result in self.results)

        logger.info("=" * 80)
        logger.info("📊 データ品質テスト結果")
        logger.info("=" * 80)
        logger.info(f"📊 総テスト数: {total_tests}")
        logger.info(f"✅ 成功: {successful_tests}")
        logger.info(f"❌ 失敗: {total_tests - successful_tests}")
        logger.info(f"📈 成功率: {successful_tests / total_tests * 100:.1f}%")
        logger.info(f"🎯 平均品質スコア: {average_quality:.1f}%")
        logger.info(f"⚠️ 発見された問題: {total_issues}個")
        logger.info(f"⏱️ 総実行時間: {total_execution_time:.2f}秒")

        logger.info("\n📊 データ品質テスト詳細:")
        for result in self.results:
            status = "✅" if result.success else "❌"
            logger.info(f"{status} {result.test_name}")
            logger.info(f"   コンポーネント: {result.component_name}")
            logger.info(f"   実行時間: {result.execution_time:.2f}秒")
            logger.info(f"   品質スコア: {result.quality_score:.1f}%")
            logger.info(f"   データサイズ: {result.data_size}")
            logger.info(f"   発見された問題: {len(result.issues_found)}個")

            if result.issues_found:
                for issue in result.issues_found[:3]:  # 最初の3個の問題を表示
                    logger.info(f"     - {issue}")
                if len(result.issues_found) > 3:
                    logger.info(f"     - ... 他{len(result.issues_found) - 3}個")

            if result.error_message:
                logger.info(f"   エラー: {result.error_message[:100]}...")

        logger.info("=" * 80)
        logger.info("🎯 データ品質テストスイート完了")

        return self.results


if __name__ == "__main__":
    suite = DataQualityTestSuite()
    results = suite.run_all_tests()
