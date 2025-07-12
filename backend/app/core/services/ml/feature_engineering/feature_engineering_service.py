"""
特徴量エンジニアリングサービス

OHLCV、ファンディングレート（FR）、建玉残高（OI）データを受け取り、
市場の歪みや偏りを捉える高度な特徴量を計算します。

リファクタリング後：責任を分割し、各特徴量計算クラスを統合します。
"""

import logging
import pandas as pd
import numpy as np
from typing import Dict, Any, Optional, List
from datetime import datetime, timedelta

from .price_features import PriceFeatureCalculator
from .market_data_features import MarketDataFeatureCalculator
from .technical_features import TechnicalFeatureCalculator

logger = logging.getLogger(__name__)


class FeatureEngineeringService:
    """
    特徴量エンジニアリングサービス

    各特徴量計算クラスを統合し、高度な特徴量を生成します。
    単一責任原則に従い、各特徴量タイプの計算は専用クラスに委譲します。
    """

    def __init__(self):
        """初期化"""
        self.feature_cache = {}
        self.max_cache_size = 10  # 最大キャッシュサイズ
        self.cache_ttl = 3600  # キャッシュ有効期限（秒）

        # 特徴量計算クラスを初期化
        self.price_calculator = PriceFeatureCalculator()
        self.market_data_calculator = MarketDataFeatureCalculator()
        self.technical_calculator = TechnicalFeatureCalculator()

    def calculate_advanced_features(
        self,
        ohlcv_data: pd.DataFrame,
        funding_rate_data: Optional[pd.DataFrame] = None,
        open_interest_data: Optional[pd.DataFrame] = None,
        lookback_periods: Optional[Dict[str, int]] = None
    ) -> pd.DataFrame:
        """
        高度な特徴量を計算

        Args:
            ohlcv_data: OHLCV価格データ
            funding_rate_data: ファンディングレートデータ（オプション）
            open_interest_data: 建玉残高データ（オプション）
            lookback_periods: 各特徴量の計算期間設定

        Returns:
            特徴量が追加されたDataFrame
        """
        try:
            if ohlcv_data.empty:
                raise ValueError("OHLCVデータが空です")

            # メモリ使用量制限
            if len(ohlcv_data) > 50000:
                logger.warning(f"大量のデータ（{len(ohlcv_data)}行）、最新50,000行に制限")
                ohlcv_data = ohlcv_data.tail(50000)

            # キャッシュキーを生成
            cache_key = self._generate_cache_key(ohlcv_data, funding_rate_data, open_interest_data, lookback_periods)

            # キャッシュから結果を取得
            cached_result = self._get_from_cache(cache_key)
            if cached_result is not None:
                logger.debug("キャッシュから特徴量を取得")
                return cached_result

            # デフォルトの計算期間
            if lookback_periods is None:
                lookback_periods = {
                    "short_ma": 10,
                    "long_ma": 50,
                    "volatility": 20,
                    "momentum": 14,
                    "volume": 20
                }

            # 結果DataFrameを初期化（メモリ効率化）
            result_df = ohlcv_data.copy()

            # データ型を最適化
            result_df = self._optimize_dtypes(result_df)

            # 基本的な価格特徴量
            result_df = self.price_calculator.calculate_price_features(result_df, lookback_periods)

            # ボラティリティ特徴量
            result_df = self.price_calculator.calculate_volatility_features(result_df, lookback_periods)

            # 出来高特徴量
            result_df = self.price_calculator.calculate_volume_features(result_df, lookback_periods)

            # ファンディングレート特徴量（データがある場合）
            if funding_rate_data is not None and not funding_rate_data.empty:
                result_df = self.market_data_calculator.calculate_funding_rate_features(
                    result_df, funding_rate_data, lookback_periods
                )

            # 建玉残高特徴量（データがある場合）
            if open_interest_data is not None and not open_interest_data.empty:
                result_df = self.market_data_calculator.calculate_open_interest_features(
                    result_df, open_interest_data, lookback_periods
                )

            # 複合特徴量（FR + OI）
            if (funding_rate_data is not None and not funding_rate_data.empty and
                open_interest_data is not None and not open_interest_data.empty):
                result_df = self.market_data_calculator.calculate_composite_features(
                    result_df, funding_rate_data, open_interest_data, lookback_periods
                )

            # 市場レジーム特徴量
            result_df = self.technical_calculator.calculate_market_regime_features(result_df, lookback_periods)

            # モメンタム特徴量
            result_df = self.technical_calculator.calculate_momentum_features(result_df, lookback_periods)

            # パターン認識特徴量
            result_df = self.technical_calculator.calculate_pattern_features(result_df, lookback_periods)

            logger.info(f"特徴量計算完了: {len(result_df.columns)}個の特徴量を生成")

            # 結果をキャッシュに保存
            self._save_to_cache(cache_key, result_df)

            return result_df

        except Exception as e:
            logger.error(f"特徴量計算エラー: {e}")
            raise















    def get_feature_names(self) -> List[str]:
        """生成される特徴量名のリストを取得"""
        feature_names = []

        # 各計算クラスから特徴量名を取得
        feature_names.extend(self.price_calculator.get_feature_names())
        feature_names.extend(self.market_data_calculator.get_feature_names())
        feature_names.extend(self.technical_calculator.get_feature_names())

        return feature_names

    def _generate_cache_key(self, ohlcv_data: pd.DataFrame,
                           funding_rate_data: Optional[pd.DataFrame],
                           open_interest_data: Optional[pd.DataFrame],
                           lookback_periods: Optional[Dict[str, int]]) -> str:
        """キャッシュキーを生成"""
        import hashlib

        # データのハッシュを計算
        ohlcv_hash = hashlib.md5(str(ohlcv_data.shape).encode()).hexdigest()[:8]
        fr_hash = hashlib.md5(str(funding_rate_data.shape if funding_rate_data is not None else "None").encode()).hexdigest()[:8]
        oi_hash = hashlib.md5(str(open_interest_data.shape if open_interest_data is not None else "None").encode()).hexdigest()[:8]
        periods_hash = hashlib.md5(str(sorted(lookback_periods.items()) if lookback_periods is not None else "None").encode()).hexdigest()[:8]

        return f"features_{ohlcv_hash}_{fr_hash}_{oi_hash}_{periods_hash}"

    def _get_from_cache(self, cache_key: str) -> Optional[pd.DataFrame]:
        """キャッシュから結果を取得"""
        try:
            if cache_key in self.feature_cache:
                cached_data, timestamp = self.feature_cache[cache_key]

                # TTLチェック
                if datetime.now().timestamp() - timestamp < self.cache_ttl:
                    return cached_data.copy()
                else:
                    # 期限切れのキャッシュを削除
                    del self.feature_cache[cache_key]

            return None

        except Exception as e:
            logger.warning(f"キャッシュ取得エラー: {e}")
            return None

    def _save_to_cache(self, cache_key: str, data: pd.DataFrame):
        """結果をキャッシュに保存"""
        try:
            # キャッシュサイズ制限
            if len(self.feature_cache) >= self.max_cache_size:
                # 最も古いキャッシュを削除
                oldest_key = min(self.feature_cache.keys(),
                               key=lambda k: self.feature_cache[k][1])
                del self.feature_cache[oldest_key]

            # 新しいキャッシュを保存
            self.feature_cache[cache_key] = (data.copy(), datetime.now().timestamp())

        except Exception as e:
            logger.warning(f"キャッシュ保存エラー: {e}")

    def _optimize_dtypes(self, df: pd.DataFrame) -> pd.DataFrame:
        """データ型を最適化してメモリ使用量を削減"""
        try:
            optimized_df = df.copy()

            for col in optimized_df.columns:
                if col == 'timestamp':
                    continue

                if optimized_df[col].dtype == 'float64':
                    # float64をfloat32に変換（精度は十分）
                    optimized_df[col] = optimized_df[col].astype('float32')
                elif optimized_df[col].dtype == 'int64':
                    # int64をint32に変換（範囲が十分な場合）
                    if optimized_df[col].min() >= -2147483648 and optimized_df[col].max() <= 2147483647:
                        optimized_df[col] = optimized_df[col].astype('int32')

            return optimized_df

        except Exception as e:
            logger.warning(f"データ型最適化エラー: {e}")
            return df

    def clear_cache(self):
        """キャッシュをクリア"""
        self.feature_cache.clear()
        logger.info("特徴量キャッシュをクリアしました")

    def get_cache_info(self) -> Dict[str, Any]:
        """キャッシュ情報を取得"""
        return {
            "cache_size": len(self.feature_cache),
            "max_cache_size": self.max_cache_size,
            "cache_ttl": self.cache_ttl,
            "cache_keys": list(self.feature_cache.keys())
        }
