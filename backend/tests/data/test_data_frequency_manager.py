"""
DataFrequencyManagerのテスト

分析報告書で特定された最優先問題「データ頻度の深刻な不一致問題」の
解決策であるDataFrequencyManagerクラスの動作を検証します。
"""

import pandas as pd
import numpy as np
import logging

import sys
import os

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from app.services.ml.feature_engineering.data_frequency_manager import (
    DataFrequencyManager,
)

logger = logging.getLogger(__name__)


class TestDataFrequencyManager:
    """DataFrequencyManagerのテストクラス"""

    def setup_method(self):
        """各テストメソッドの前に実行される初期化"""
        self.manager = DataFrequencyManager()

    def create_sample_ohlcv_data(self, timeframe="1h", periods=100):
        """サンプルOHLCVデータを作成"""
        # timeframeに応じた時間間隔を設定
        freq_map = {
            "1m": "1min",
            "5m": "5min",
            "15m": "15min",
            "30m": "30min",
            "1h": "1h",
            "4h": "4h",
            "1d": "1d",
        }

        freq = freq_map.get(timeframe, "1h")
        dates = pd.date_range(start="2023-01-01", periods=periods, freq=freq)

        # 価格データを生成
        np.random.seed(42)
        base_price = 50000
        price_changes = np.random.normal(0, 0.02, periods)
        prices = base_price * np.cumprod(1 + price_changes)

        return pd.DataFrame(
            {
                "timestamp": dates,
                "Open": prices * (1 + np.random.normal(0, 0.001, periods)),
                "High": prices * (1 + np.abs(np.random.normal(0, 0.005, periods))),
                "Low": prices * (1 - np.abs(np.random.normal(0, 0.005, periods))),
                "Close": prices,
                "Volume": np.random.uniform(1000, 10000, periods),
            }
        )

    def create_sample_funding_rate_data(self, periods=50):
        """サンプルファンディングレートデータを作成（8時間間隔）"""
        dates = pd.date_range(start="2023-01-01", periods=periods, freq="8h")

        return pd.DataFrame(
            {
                "timestamp": dates,
                "funding_rate": np.random.normal(0.0001, 0.0005, periods),
            }
        )

    def create_sample_open_interest_data(self, periods=200):
        """サンプル建玉残高データを作成（1時間間隔）"""
        dates = pd.date_range(start="2023-01-01", periods=periods, freq="1h")

        return pd.DataFrame(
            {
                "timestamp": dates,
                "open_interest": np.random.uniform(1000000, 5000000, periods),
            }
        )

    def test_timeframe_detection(self):
        """timeframe自動検出のテスト"""
        logger.info("=== timeframe自動検出テスト ===")

        # 異なるtimeframeのデータでテスト
        test_cases = ["1m", "5m", "15m", "30m", "1h", "4h", "1d"]

        for expected_timeframe in test_cases:
            ohlcv_data = self.create_sample_ohlcv_data(expected_timeframe, 50)
            detected_timeframe = self.manager.detect_ohlcv_timeframe(ohlcv_data)

            logger.info(f"期待: {expected_timeframe}, 検出: {detected_timeframe}")

            # 完全一致または近似一致を確認
            assert (
                detected_timeframe in test_cases
            ), f"無効なtimeframe検出: {detected_timeframe}"

    def test_data_alignment_basic(self):
        """基本的なデータ頻度統一のテスト"""
        logger.info("=== 基本的なデータ頻度統一テスト ===")

        # 1時間間隔のOHLCVデータ
        ohlcv_data = self.create_sample_ohlcv_data("1h", 100)

        # 8時間間隔のファンディングレートデータ
        fr_data = self.create_sample_funding_rate_data(25)

        # 1時間間隔の建玉残高データ
        oi_data = self.create_sample_open_interest_data(100)

        # データ頻度統一を実行
        aligned_fr, aligned_oi = self.manager.align_data_frequencies(
            ohlcv_data, fr_data, oi_data, "1h"
        )

        # 結果の検証
        assert aligned_fr is not None, "ファンディングレートデータの統一に失敗"
        assert aligned_oi is not None, "建玉残高データの統一に失敗"

        logger.info(f"元のFRデータ: {len(fr_data)}行 → 統一後: {len(aligned_fr)}行")
        logger.info(f"元のOIデータ: {len(oi_data)}行 → 統一後: {len(aligned_oi)}行")

        # データ長の妥当性チェック
        assert len(aligned_fr) >= len(fr_data), "FRデータが予期せず減少"
        assert len(aligned_oi) <= len(oi_data) * 2, "OIデータが異常に増加"

    def test_data_alignment_different_timeframes(self):
        """異なるtimeframeでのデータ頻度統一テスト"""
        logger.info("=== 異なるtimeframeでのデータ頻度統一テスト ===")

        test_timeframes = ["15m", "30m", "1h", "4h"]

        for timeframe in test_timeframes:
            logger.info(f"--- {timeframe} timeframeのテスト ---")

            # OHLCVデータ
            ohlcv_data = self.create_sample_ohlcv_data(timeframe, 50)

            # ファンディングレートデータ（8時間間隔）
            fr_data = self.create_sample_funding_rate_data(15)

            # 建玉残高データ（1時間間隔）
            oi_data = self.create_sample_open_interest_data(50)

            # データ頻度統一を実行
            aligned_fr, aligned_oi = self.manager.align_data_frequencies(
                ohlcv_data, fr_data, oi_data, timeframe
            )

            # 結果の検証
            assert aligned_fr is not None, f"{timeframe}: FRデータの統一に失敗"
            assert aligned_oi is not None, f"{timeframe}: OIデータの統一に失敗"

            logger.info(
                f"{timeframe}: FR {len(fr_data)} → {len(aligned_fr)}, OI {len(oi_data)} → {len(aligned_oi)}"
            )

    def test_data_validation(self):
        """データ整合性検証のテスト"""
        logger.info("=== データ整合性検証テスト ===")

        # 正常なデータ
        ohlcv_data = self.create_sample_ohlcv_data("1h", 100)
        fr_data = self.create_sample_funding_rate_data(25)
        oi_data = self.create_sample_open_interest_data(100)

        # 検証実行
        validation_result = self.manager.validate_data_alignment(
            ohlcv_data, fr_data, oi_data
        )

        # 結果の確認
        assert "is_valid" in validation_result
        assert "statistics" in validation_result
        assert "ohlcv_timeframe" in validation_result["statistics"]

        logger.info(f"検証結果: {validation_result['is_valid']}")
        logger.info(f"統計情報: {validation_result['statistics']}")

        # 空のOHLCVデータでのテスト
        empty_ohlcv = pd.DataFrame()
        validation_result_empty = self.manager.validate_data_alignment(
            empty_ohlcv, fr_data, oi_data
        )

        assert not validation_result_empty[
            "is_valid"
        ], "空データの検証が正しく失敗していない"
        assert (
            len(validation_result_empty["errors"]) > 0
        ), "エラーメッセージが設定されていない"

    def test_frequency_mappings(self):
        """頻度マッピングのテスト"""
        logger.info("=== 頻度マッピングテスト ===")

        # 各timeframeに対する推奨頻度を確認
        test_cases = [
            ("1m", "ohlcv", "1m"),
            ("1h", "fr", "8h"),
            ("4h", "oi", "1h"),
            ("1d", "ohlcv", "1d"),
        ]

        for ohlcv_timeframe, data_type, expected_freq in test_cases:
            actual_freq = self.manager.get_target_frequency(data_type, ohlcv_timeframe)
            assert (
                actual_freq == expected_freq
            ), f"{ohlcv_timeframe}の{data_type}頻度: 期待={expected_freq}, 実際={actual_freq}"

            logger.info(f"{ohlcv_timeframe} {data_type}: {actual_freq}")

    def test_resample_funding_rate(self):
        """ファンディングレート再サンプリングのテスト"""
        logger.info("=== ファンディングレート再サンプリングテスト ===")

        # 8時間間隔のデータを作成
        fr_data = self.create_sample_funding_rate_data(20)

        # 異なるtimeframeへの再サンプリングをテスト
        test_timeframes = ["1h", "4h", "1d"]

        for timeframe in test_timeframes:
            resampled = self.manager._resample_funding_rate(fr_data, timeframe)

            assert not resampled.empty, f"{timeframe}への再サンプリングが失敗"
            assert "timestamp" in resampled.columns, "timestampカラムが失われている"
            assert (
                "funding_rate" in resampled.columns
            ), "funding_rateカラムが失われている"

            logger.info(
                f"{timeframe}再サンプリング: {len(fr_data)} → {len(resampled)}行"
            )

    def test_resample_open_interest(self):
        """建玉残高再サンプリングのテスト"""
        logger.info("=== 建玉残高再サンプリングテスト ===")

        # 1時間間隔のデータを作成
        oi_data = self.create_sample_open_interest_data(100)

        # 異なるtimeframeへの再サンプリングをテスト
        test_timeframes = ["15m", "30m", "4h", "1d"]

        for timeframe in test_timeframes:
            resampled = self.manager._resample_open_interest(oi_data, timeframe)

            assert not resampled.empty, f"{timeframe}への再サンプリングが失敗"
            assert "timestamp" in resampled.columns, "timestampカラムが失われている"
            assert (
                "open_interest" in resampled.columns
            ), "open_interestカラムが失われている"

            logger.info(
                f"{timeframe}再サンプリング: {len(oi_data)} → {len(resampled)}行"
            )

    def test_error_handling(self):
        """エラーハンドリングのテスト"""
        logger.info("=== エラーハンドリングテスト ===")

        # 空のデータでのテスト
        empty_df = pd.DataFrame()

        # 空のOHLCVデータ
        timeframe = self.manager.detect_ohlcv_timeframe(empty_df)
        assert timeframe == "1h", "空データのデフォルトtimeframeが正しくない"

        # 空のFRデータの再サンプリング
        resampled_fr = self.manager._resample_funding_rate(empty_df, "1h")
        assert resampled_fr.empty, "空データの再サンプリング結果が正しくない"

        # 空のOIデータの再サンプリング
        resampled_oi = self.manager._resample_open_interest(empty_df, "1h")
        assert resampled_oi.empty, "空データの再サンプリング結果が正しくない"

        logger.info("エラーハンドリングテスト完了")


if __name__ == "__main__":
    # ログ設定
    logging.basicConfig(level=logging.INFO)

    # テスト実行
    test_manager = TestDataFrequencyManager()
    test_manager.setup_method()

    try:
        test_manager.test_timeframe_detection()
        test_manager.test_data_alignment_basic()
        test_manager.test_data_alignment_different_timeframes()
        test_manager.test_data_validation()
        test_manager.test_frequency_mappings()
        test_manager.test_resample_funding_rate()
        test_manager.test_resample_open_interest()
        test_manager.test_error_handling()

        logger.info("=== 全テスト完了 ===")

    except Exception as e:
        logger.error(f"テスト実行エラー: {e}")
        raise
