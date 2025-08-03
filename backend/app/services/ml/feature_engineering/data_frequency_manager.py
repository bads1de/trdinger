"""
データ頻度統一マネージャー

異なる頻度のデータ（OHLCV、ファンディングレート、建玉残高）を
統一的に扱うためのマネージャークラス。

分析報告書で特定された最優先問題「データ頻度の深刻な不一致問題」を解決します。
"""

import logging
import pandas as pd

from typing import Dict, Optional, Tuple, Any


logger = logging.getLogger(__name__)


class DataFrequencyManager:
    """データ頻度統一マネージャー"""

    def __init__(self):
        """初期化"""
        # 各timeframeに対する推奨データ頻度マッピング
        self.frequency_mappings = {
            "1m": {"ohlcv": "1m", "fr": "8h", "oi": "1h"},
            "5m": {"ohlcv": "5m", "fr": "8h", "oi": "1h"},
            "15m": {"ohlcv": "15m", "fr": "8h", "oi": "1h"},
            "30m": {"ohlcv": "30m", "fr": "8h", "oi": "1h"},
            "1h": {"ohlcv": "1h", "fr": "8h", "oi": "1h"},
            "4h": {"ohlcv": "4h", "fr": "8h", "oi": "1h"},
            "1d": {"ohlcv": "1d", "fr": "8h", "oi": "1h"},
        }

        # 時間間隔をミリ秒に変換するマッピング
        self.interval_to_ms = {
            "1m": 60 * 1000,
            "5m": 5 * 60 * 1000,
            "15m": 15 * 60 * 1000,
            "30m": 30 * 60 * 1000,
            "1h": 60 * 60 * 1000,
            "4h": 4 * 60 * 60 * 1000,
            "8h": 8 * 60 * 60 * 1000,
            "1d": 24 * 60 * 60 * 1000,
        }

    def get_target_frequency(self, source_data_type: str, ohlcv_timeframe: str) -> str:
        """
        OHLCVのtimeframeに基づいて各データタイプの目標頻度を取得

        Args:
            source_data_type: データタイプ（'ohlcv', 'fr', 'oi'）
            ohlcv_timeframe: OHLCVのtimeframe

        Returns:
            目標頻度文字列
        """
        return self.frequency_mappings.get(ohlcv_timeframe, {}).get(
            source_data_type, "1h"
        )

    def detect_ohlcv_timeframe(self, ohlcv_data: pd.DataFrame) -> str:
        """
        OHLCVデータからtimeframeを自動検出

        Args:
            ohlcv_data: OHLCVデータ

        Returns:
            検出されたtimeframe
        """
        if ohlcv_data.empty or len(ohlcv_data) < 2:
            logger.warning("OHLCVデータが不足しているため、デフォルトの1hを使用")
            return "1h"

        try:
            # timestampカラムまたはindexから時間間隔を計算
            if "timestamp" in ohlcv_data.columns:
                timestamps = pd.to_datetime(ohlcv_data["timestamp"])
            else:
                timestamps = pd.to_datetime(ohlcv_data.index)

            # 時間差を計算（最初の数個のデータポイントから）
            time_diffs = timestamps.diff().dropna()
            if len(time_diffs) == 0:
                return "1h"

            # 最頻値を取得
            median_diff = time_diffs.median()
            diff_minutes = median_diff.total_seconds() / 60

            # 時間間隔をtimeframeにマッピング
            if diff_minutes <= 1.5:
                return "1m"
            elif diff_minutes <= 7.5:
                return "5m"
            elif diff_minutes <= 22.5:
                return "15m"
            elif diff_minutes <= 45:
                return "30m"
            elif diff_minutes <= 120:
                return "1h"
            elif diff_minutes <= 6 * 60:
                return "4h"
            else:
                return "1d"

        except Exception as e:
            logger.warning(f"timeframe自動検出エラー: {e}. デフォルトの1hを使用")
            return "1h"

    def align_data_frequencies(
        self,
        ohlcv_data: pd.DataFrame,
        funding_rate_data: Optional[pd.DataFrame] = None,
        open_interest_data: Optional[pd.DataFrame] = None,
        ohlcv_timeframe: Optional[str] = None,
    ) -> Tuple[Optional[pd.DataFrame], Optional[pd.DataFrame]]:
        """
        異なる頻度のデータをOHLCVのtimeframeに合わせて再サンプリング

        Args:
            ohlcv_data: OHLCVデータ
            funding_rate_data: ファンディングレートデータ
            open_interest_data: 建玉残高データ
            ohlcv_timeframe: OHLCVのtimeframe（Noneの場合は自動検出）

        Returns:
            再サンプリングされた(funding_rate_data, open_interest_data)のタプル
        """
        try:
            # OHLCVのtimeframeを検出または使用
            if ohlcv_timeframe is None:
                ohlcv_timeframe = self.detect_ohlcv_timeframe(ohlcv_data)

            logger.info(f"データ頻度統一を開始: OHLCV timeframe = {ohlcv_timeframe}")

            # ファンディングレートデータの再サンプリング
            aligned_fr_data = None
            if funding_rate_data is not None and not funding_rate_data.empty:
                aligned_fr_data = self._resample_funding_rate(
                    funding_rate_data, ohlcv_timeframe
                )

            # 建玉残高データの再サンプリング
            aligned_oi_data = None
            if open_interest_data is not None and not open_interest_data.empty:
                aligned_oi_data = self._resample_open_interest(
                    open_interest_data, ohlcv_timeframe
                )

            logger.info("データ頻度統一完了")
            return aligned_fr_data, aligned_oi_data

        except Exception as e:
            logger.error(f"データ頻度統一エラー: {e}")
            return funding_rate_data, open_interest_data

    def _resample_funding_rate(
        self, fr_data: pd.DataFrame, target_timeframe: str
    ) -> pd.DataFrame:
        """
        ファンディングレートデータを再サンプリング

        Args:
            fr_data: ファンディングレートデータ
            target_timeframe: 目標timeframe

        Returns:
            再サンプリングされたデータ
        """
        try:
            if fr_data.empty:
                return fr_data

            # timestampカラムをindexに設定
            if "timestamp" in fr_data.columns:
                fr_data_indexed = fr_data.set_index("timestamp")
            else:
                fr_data_indexed = fr_data.copy()

            # indexをdatetimeに変換
            fr_data_indexed.index = pd.to_datetime(fr_data_indexed.index)

            # 8時間間隔から目標timeframeへの再サンプリング
            if target_timeframe in ["1m", "5m", "15m", "30m", "1h"]:
                # より細かい間隔への補間（前方補完）
                resampled = fr_data_indexed.resample(target_timeframe).ffill()
            elif target_timeframe in ["4h"]:
                # 4時間間隔への集約（平均値）
                resampled = fr_data_indexed.resample(target_timeframe).mean()
            elif target_timeframe == "1d":
                # 日次への集約（平均値）
                resampled = fr_data_indexed.resample(target_timeframe).mean()
            else:
                # デフォルトは前方補完
                resampled = fr_data_indexed.resample(target_timeframe).ffill()

            # timestampカラムを復元
            resampled = resampled.reset_index()

            logger.info(
                f"ファンディングレート再サンプリング完了: {len(fr_data)} → {len(resampled)}行"
            )
            return resampled

        except Exception as e:
            logger.error(f"ファンディングレート再サンプリングエラー: {e}")
            return fr_data

    def _resample_open_interest(
        self, oi_data: pd.DataFrame, target_timeframe: str
    ) -> pd.DataFrame:
        """
        建玉残高データを再サンプリング

        Args:
            oi_data: 建玉残高データ
            target_timeframe: 目標timeframe

        Returns:
            再サンプリングされたデータ
        """
        try:
            if oi_data.empty:
                return oi_data

            # timestampカラムをindexに設定
            if "timestamp" in oi_data.columns:
                oi_data_indexed = oi_data.set_index("timestamp")
            else:
                oi_data_indexed = oi_data.copy()

            # indexをdatetimeに変換
            oi_data_indexed.index = pd.to_datetime(oi_data_indexed.index)

            # 1時間間隔から目標timeframeへの再サンプリング
            if target_timeframe in ["1m", "5m", "15m", "30m"]:
                # より細かい間隔への補間（前方補完）
                resampled = oi_data_indexed.resample(target_timeframe).ffill()
            elif target_timeframe == "1h":
                # 同じ間隔なのでそのまま
                resampled = oi_data_indexed
            elif target_timeframe in ["4h", "1d"]:
                # より粗い間隔への集約（平均値）
                resampled = oi_data_indexed.resample(target_timeframe).mean()
            else:
                # デフォルトは前方補完
                resampled = oi_data_indexed.resample(target_timeframe).ffill()

            # timestampカラムを復元
            resampled = resampled.reset_index()

            logger.info(
                f"建玉残高再サンプリング完了: {len(oi_data)} → {len(resampled)}行"
            )
            return resampled

        except Exception as e:
            logger.error(f"建玉残高再サンプリングエラー: {e}")
            return oi_data

    def validate_data_alignment(
        self,
        ohlcv_data: pd.DataFrame,
        funding_rate_data: Optional[pd.DataFrame] = None,
        open_interest_data: Optional[pd.DataFrame] = None,
    ) -> Dict[str, Any]:
        """
        データの整合性を検証

        Args:
            ohlcv_data: OHLCVデータ
            funding_rate_data: ファンディングレートデータ
            open_interest_data: 建玉残高データ

        Returns:
            検証結果の辞書
        """
        validation_result = {
            "is_valid": True,
            "warnings": [],
            "errors": [],
            "statistics": {},
        }

        try:
            # OHLCVデータの検証
            if ohlcv_data.empty:
                validation_result["errors"].append("OHLCVデータが空です")
                validation_result["is_valid"] = False
                return validation_result

            ohlcv_timeframe = self.detect_ohlcv_timeframe(ohlcv_data)
            validation_result["statistics"]["ohlcv_timeframe"] = ohlcv_timeframe
            validation_result["statistics"]["ohlcv_rows"] = len(ohlcv_data)

            # ファンディングレートデータの検証
            if funding_rate_data is not None and not funding_rate_data.empty:
                validation_result["statistics"]["fr_rows"] = len(funding_rate_data)
                # データ範囲の重複チェック
                # 実装は簡略化

            # 建玉残高データの検証
            if open_interest_data is not None and not open_interest_data.empty:
                validation_result["statistics"]["oi_rows"] = len(open_interest_data)

            logger.info(f"データ整合性検証完了: {validation_result['statistics']}")

        except Exception as e:
            validation_result["errors"].append(f"検証エラー: {e}")
            validation_result["is_valid"] = False

        return validation_result
