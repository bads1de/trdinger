"""
特徴量エンジニアリングサービス

OHLCV、ファンディングレート（FR）、建玉残高（OI）データを受け取り、
市場の歪みや偏りを捉える高度な特徴量を計算します。
"""

import logging
import pandas as pd
import numpy as np
from typing import Dict, Any, Optional, List
from datetime import datetime, timedelta

logger = logging.getLogger(__name__)


class FeatureEngineeringService:
    """
    特徴量エンジニアリングサービス
    
    従来のテクニカル指標だけでは捉えきれない市場の力学を
    モデルに組み込むための高度な特徴量を生成します。
    """

    def __init__(self):
        """初期化"""
        self.feature_cache = {}
        self.max_cache_size = 10  # 最大キャッシュサイズ
        self.cache_ttl = 3600  # キャッシュ有効期限（秒）

    def calculate_advanced_features(
        self,
        ohlcv_data: pd.DataFrame,
        funding_rate_data: Optional[pd.DataFrame] = None,
        open_interest_data: Optional[pd.DataFrame] = None,
        lookback_periods: Dict[str, int] = None
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
            result_df = self._calculate_price_features(result_df, lookback_periods)

            # ボラティリティ特徴量
            result_df = self._calculate_volatility_features(result_df, lookback_periods)

            # 出来高特徴量
            result_df = self._calculate_volume_features(result_df, lookback_periods)

            # ファンディングレート特徴量（データがある場合）
            if funding_rate_data is not None and not funding_rate_data.empty:
                result_df = self._calculate_funding_rate_features(
                    result_df, funding_rate_data, lookback_periods
                )

            # 建玉残高特徴量（データがある場合）
            if open_interest_data is not None and not open_interest_data.empty:
                result_df = self._calculate_open_interest_features(
                    result_df, open_interest_data, lookback_periods
                )

            # 複合特徴量（FR + OI）
            if (funding_rate_data is not None and not funding_rate_data.empty and
                open_interest_data is not None and not open_interest_data.empty):
                result_df = self._calculate_composite_features(
                    result_df, funding_rate_data, open_interest_data, lookback_periods
                )

            # 市場レジーム特徴量
            result_df = self.calculate_market_regime_features(result_df, lookback_periods)

            # モメンタム特徴量
            result_df = self.calculate_momentum_features(result_df, lookback_periods)

            # パターン認識特徴量
            result_df = self.calculate_pattern_features(result_df, lookback_periods)

            logger.info(f"特徴量計算完了: {len(result_df.columns)}個の特徴量を生成")

            # 結果をキャッシュに保存
            self._save_to_cache(cache_key, result_df)

            return result_df

        except Exception as e:
            logger.error(f"特徴量計算エラー: {e}")
            raise

    def _calculate_price_features(
        self, df: pd.DataFrame, periods: Dict[str, int]
    ) -> pd.DataFrame:
        """価格関連の特徴量を計算"""
        try:
            # 移動平均線
            df[f'SMA_{periods["short_ma"]}'] = df['close'].rolling(
                window=periods["short_ma"]
            ).mean()
            df[f'SMA_{periods["long_ma"]}'] = df['close'].rolling(
                window=periods["long_ma"]
            ).mean()

            # 価格と移動平均線の乖離率
            df['Price_MA_Ratio_Short'] = (
                df['close'] / df[f'SMA_{periods["short_ma"]}'] - 1
            )
            df['Price_MA_Ratio_Long'] = (
                df['close'] / df[f'SMA_{periods["long_ma"]}'] - 1
            )

            # 価格モメンタム
            df[f'Price_Momentum_{periods["momentum"]}'] = (
                df['close'] / df['close'].shift(periods["momentum"]) - 1
            )

            # 高値・安値からの位置
            df['High_Low_Position'] = (
                (df['close'] - df['low']) / (df['high'] - df['low'])
            ).fillna(0.5)

            return df

        except Exception as e:
            logger.error(f"価格特徴量計算エラー: {e}")
            return df

    def _calculate_volatility_features(
        self, df: pd.DataFrame, periods: Dict[str, int]
    ) -> pd.DataFrame:
        """ボラティリティ関連の特徴量を計算"""
        try:
            # リターンを計算
            df['returns'] = df['close'].pct_change()

            # 実現ボラティリティ
            df[f'Realized_Volatility_{periods["volatility"]}'] = (
                df['returns'].rolling(window=periods["volatility"]).std() * np.sqrt(252)
            )

            # ボラティリティスパイク（急激なボラティリティ上昇）
            volatility_ma = df[f'Realized_Volatility_{periods["volatility"]}'].rolling(
                window=periods["volatility"]
            ).mean()
            df['Volatility_Spike'] = (
                df[f'Realized_Volatility_{periods["volatility"]}'] / volatility_ma - 1
            ).fillna(0)

            # True Range
            df['True_Range'] = np.maximum(
                df['high'] - df['low'],
                np.maximum(
                    abs(df['high'] - df['close'].shift(1)),
                    abs(df['low'] - df['close'].shift(1))
                )
            )

            # Average True Range
            df[f'ATR_{periods["volatility"]}'] = df['True_Range'].rolling(
                window=periods["volatility"]
            ).mean()

            return df

        except Exception as e:
            logger.error(f"ボラティリティ特徴量計算エラー: {e}")
            return df

    def _calculate_volume_features(
        self, df: pd.DataFrame, periods: Dict[str, int]
    ) -> pd.DataFrame:
        """出来高関連の特徴量を計算"""
        try:
            # 出来高移動平均
            df[f'Volume_MA_{periods["volume"]}'] = df['volume'].rolling(
                window=periods["volume"]
            ).mean()

            # 出来高比率
            df['Volume_Ratio'] = (
                df['volume'] / df[f'Volume_MA_{periods["volume"]}']
            ).fillna(1.0)

            # 価格・出来高関係
            df['Price_Volume_Trend'] = (
                df['returns'] * df['Volume_Ratio']
            ).fillna(0)

            return df

        except Exception as e:
            logger.error(f"出来高特徴量計算エラー: {e}")
            return df

    def _calculate_funding_rate_features(
        self, df: pd.DataFrame, fr_data: pd.DataFrame, periods: Dict[str, int]
    ) -> pd.DataFrame:
        """ファンディングレート関連の特徴量を計算"""
        try:
            # ファンディングレートデータをマージ
            # タイムスタンプでマージ（最も近い時刻のデータを使用）
            merged_df = pd.merge_asof(
                df.sort_values('timestamp'),
                fr_data.sort_values('timestamp'),
                on='timestamp',
                direction='backward',
                suffixes=('', '_fr')
            )

            if 'funding_rate' in merged_df.columns:
                # ファンディングレート移動平均
                merged_df[f'FR_MA_{periods.get("funding", 24)}'] = (
                    merged_df['funding_rate'].rolling(
                        window=periods.get("funding", 24)
                    ).mean()
                )

                # ファンディングレートの変化
                merged_df['FR_Change'] = merged_df['funding_rate'].diff()

                # 価格とファンディングレートの逆行
                merged_df['Price_FR_Divergence'] = (
                    merged_df['returns'] * merged_df['FR_Change'] * -1
                ).fillna(0)

            return merged_df

        except Exception as e:
            logger.error(f"ファンディングレート特徴量計算エラー: {e}")
            return df

    def _calculate_open_interest_features(
        self, df: pd.DataFrame, oi_data: pd.DataFrame, periods: Dict[str, int]
    ) -> pd.DataFrame:
        """建玉残高関連の特徴量を計算"""
        try:
            # 建玉残高データをマージ
            merged_df = pd.merge_asof(
                df.sort_values('timestamp'),
                oi_data.sort_values('timestamp'),
                on='timestamp',
                direction='backward',
                suffixes=('', '_oi')
            )

            if 'open_interest_value' in merged_df.columns:
                # 建玉残高の変化率
                merged_df['OI_Change_Rate'] = (
                    merged_df['open_interest_value'].pct_change()
                ).fillna(0)

                # 建玉残高急増（サージ）
                oi_ma = merged_df['open_interest_value'].rolling(
                    window=periods.get("open_interest", 24)
                ).mean()
                merged_df['OI_Surge'] = (
                    merged_df['open_interest_value'] / oi_ma - 1
                ).fillna(0)

                # ボラティリティ調整済み建玉残高変化
                if f'Realized_Volatility_{periods["volatility"]}' in merged_df.columns:
                    merged_df['Volatility_Adjusted_OI'] = (
                        merged_df['OI_Change_Rate'] / 
                        (merged_df[f'Realized_Volatility_{periods["volatility"]}'] + 1e-8)
                    ).fillna(0)

            return merged_df

        except Exception as e:
            logger.error(f"建玉残高特徴量計算エラー: {e}")
            return df

    def _calculate_composite_features(
        self, df: pd.DataFrame, fr_data: pd.DataFrame, oi_data: pd.DataFrame, periods: Dict[str, int]
    ) -> pd.DataFrame:
        """複合特徴量を計算（FR + OI）"""
        try:
            if 'funding_rate' in df.columns and 'open_interest_value' in df.columns:
                # ファンディングレートと建玉残高の比率
                df['FR_OI_Ratio'] = (
                    df['funding_rate'] / (df['open_interest_value'] / 1e9 + 1e-8)
                ).fillna(0)

                # 市場過熱指標
                df['Market_Heat_Index'] = (
                    abs(df['funding_rate']) * df['OI_Surge']
                ).fillna(0)

            return df

        except Exception as e:
            logger.error(f"複合特徴量計算エラー: {e}")
            return df

    def calculate_market_regime_features(
        self, df: pd.DataFrame, periods: Dict[str, int]
    ) -> pd.DataFrame:
        """市場レジーム関連の特徴量を計算"""
        try:
            # トレンド強度
            df['Trend_Strength'] = abs(
                df[f'SMA_{periods["short_ma"]}'] - df[f'SMA_{periods["long_ma"]}']
            ) / df['close']

            # レンジ相場判定
            high_20 = df['high'].rolling(window=20).max()
            low_20 = df['low'].rolling(window=20).min()
            df['Range_Bound_Ratio'] = (high_20 - low_20) / df['close']

            # ブレイクアウト強度
            df['Breakout_Strength'] = np.where(
                df['close'] > high_20.shift(1),
                (df['close'] - high_20.shift(1)) / high_20.shift(1),
                np.where(
                    df['close'] < low_20.shift(1),
                    (low_20.shift(1) - df['close']) / low_20.shift(1),
                    0
                )
            )

            return df

        except Exception as e:
            logger.error(f"市場レジーム特徴量計算エラー: {e}")
            return df

    def calculate_momentum_features(
        self, df: pd.DataFrame, periods: Dict[str, int]
    ) -> pd.DataFrame:
        """モメンタム関連の特徴量を計算"""
        try:
            # RSI
            delta = df['close'].diff()
            gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
            loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
            rs = gain / loss
            df['RSI'] = 100 - (100 / (1 + rs))

            # MACD
            ema_12 = df['close'].ewm(span=12).mean()
            ema_26 = df['close'].ewm(span=26).mean()
            df['MACD'] = ema_12 - ema_26
            df['MACD_Signal'] = df['MACD'].ewm(span=9).mean()
            df['MACD_Histogram'] = df['MACD'] - df['MACD_Signal']

            # ストキャスティクス
            lowest_low = df['low'].rolling(window=14).min()
            highest_high = df['high'].rolling(window=14).max()
            df['Stochastic_K'] = 100 * (
                (df['close'] - lowest_low) / (highest_high - lowest_low)
            )
            df['Stochastic_D'] = df['Stochastic_K'].rolling(window=3).mean()

            return df

        except Exception as e:
            logger.error(f"モメンタム特徴量計算エラー: {e}")
            return df

    def calculate_pattern_features(
        self, df: pd.DataFrame, periods: Dict[str, int]
    ) -> pd.DataFrame:
        """パターン認識関連の特徴量を計算"""
        try:
            # ダイバージェンス検出
            price_peaks = df['close'].rolling(window=5, center=True).max() == df['close']
            rsi_peaks = df['RSI'].rolling(window=5, center=True).max() == df['RSI']

            # ベアダイバージェンス（価格上昇、RSI下降）
            df['Bear_Divergence'] = (
                price_peaks &
                (df['close'] > df['close'].shift(10)) &
                (df['RSI'] < df['RSI'].shift(10))
            ).astype(int)

            # ブルダイバージェンス（価格下降、RSI上昇）
            price_troughs = df['close'].rolling(window=5, center=True).min() == df['close']
            rsi_troughs = df['RSI'].rolling(window=5, center=True).min() == df['RSI']

            df['Bull_Divergence'] = (
                price_troughs &
                (df['close'] < df['close'].shift(10)) &
                (df['RSI'] > df['RSI'].shift(10))
            ).astype(int)

            # サポート・レジスタンス近接度
            support_level = df['low'].rolling(window=50).min()
            resistance_level = df['high'].rolling(window=50).max()

            df['Support_Distance'] = (df['close'] - support_level) / df['close']
            df['Resistance_Distance'] = (resistance_level - df['close']) / df['close']

            return df

        except Exception as e:
            logger.error(f"パターン特徴量計算エラー: {e}")
            return df

    def get_feature_names(self) -> List[str]:
        """生成される特徴量名のリストを取得"""
        return [
            # 価格特徴量
            'Price_MA_Ratio_Short', 'Price_MA_Ratio_Long',
            'Price_Momentum_14', 'High_Low_Position',

            # ボラティリティ特徴量
            'Realized_Volatility_20', 'Volatility_Spike', 'ATR_20',

            # 出来高特徴量
            'Volume_Ratio', 'Price_Volume_Trend',

            # ファンディングレート特徴量
            'FR_MA_24', 'FR_Change', 'Price_FR_Divergence',

            # 建玉残高特徴量
            'OI_Change_Rate', 'OI_Surge', 'Volatility_Adjusted_OI',

            # 複合特徴量
            'FR_OI_Ratio', 'Market_Heat_Index',

            # 市場レジーム特徴量
            'Trend_Strength', 'Range_Bound_Ratio', 'Breakout_Strength',

            # モメンタム特徴量
            'RSI', 'MACD', 'MACD_Signal', 'MACD_Histogram',
            'Stochastic_K', 'Stochastic_D',

            # パターン特徴量
            'Bear_Divergence', 'Bull_Divergence',
            'Support_Distance', 'Resistance_Distance'
        ]

    def _generate_cache_key(self, ohlcv_data: pd.DataFrame,
                           funding_rate_data: Optional[pd.DataFrame],
                           open_interest_data: Optional[pd.DataFrame],
                           lookback_periods: Dict[str, int]) -> str:
        """キャッシュキーを生成"""
        import hashlib

        # データのハッシュを計算
        ohlcv_hash = hashlib.md5(str(ohlcv_data.shape).encode()).hexdigest()[:8]
        fr_hash = hashlib.md5(str(funding_rate_data.shape if funding_rate_data is not None else "None").encode()).hexdigest()[:8]
        oi_hash = hashlib.md5(str(open_interest_data.shape if open_interest_data is not None else "None").encode()).hexdigest()[:8]
        periods_hash = hashlib.md5(str(sorted(lookback_periods.items())).encode()).hexdigest()[:8]

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
