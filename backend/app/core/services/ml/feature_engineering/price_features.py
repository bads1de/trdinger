"""
価格特徴量計算クラス

OHLCV価格データから基本的な価格関連特徴量を計算します。
単一責任原則に従い、価格特徴量の計算のみを担当します。
"""

import logging
import pandas as pd
import numpy as np
from typing import Dict, Any

from ....utils.ml_error_handler import safe_ml_operation
from ....utils.data_validation import DataValidator

logger = logging.getLogger(__name__)


class PriceFeatureCalculator:
    """
    価格特徴量計算クラス
    
    OHLCV価格データから基本的な価格関連特徴量を計算します。
    """
    
    def __init__(self):
        """初期化"""
        pass
    
    @safe_ml_operation(default_value=None, error_message="価格特徴量計算でエラーが発生しました")
    def calculate_price_features(
        self,
        df: pd.DataFrame,
        lookback_periods: Dict[str, int]
    ) -> pd.DataFrame:
        """
        価格特徴量を計算

        Args:
            df: OHLCV価格データ
            lookback_periods: 計算期間設定

        Returns:
            価格特徴量が追加されたDataFrame
        """
        if df is None or df.empty:
            logger.warning("空のデータが提供されました")
            return df

        result_df = df.copy()

        # 移動平均比率
        short_ma = lookback_periods.get("short_ma", 10)
        long_ma = lookback_periods.get("long_ma", 50)

        result_df[f'MA_{short_ma}'] = result_df['Close'].rolling(window=short_ma).mean()
        result_df[f'MA_{long_ma}'] = result_df['Close'].rolling(window=long_ma).mean()

        result_df['Price_MA_Ratio_Short'] = result_df['Close'] / result_df[f'MA_{short_ma}']
        result_df['Price_MA_Ratio_Long'] = result_df['Close'] / result_df[f'MA_{long_ma}']

        # 価格モメンタム
        momentum_period = lookback_periods.get("momentum", 14)
        result_df['Price_Momentum_14'] = result_df['Close'].pct_change(momentum_period)

        # 高値・安値ポジション
        result_df['High_Low_Position'] = (
            (result_df['Close'] - result_df['Low']) /
            (result_df['High'] - result_df['Low'] + 1e-8)
        )

        # 価格変化率
        result_df['Price_Change_1'] = result_df['Close'].pct_change(1)
        result_df['Price_Change_5'] = result_df['Close'].pct_change(5)
        result_df['Price_Change_20'] = result_df['Close'].pct_change(20)

        # 価格レンジ
        result_df['Price_Range'] = (result_df['High'] - result_df['Low']) / result_df['Close']

        # ボディサイズ（実体の大きさ）
        result_df['Body_Size'] = abs(result_df['Close'] - result_df['Open']) / result_df['Close']

        # 上ヒゲ・下ヒゲ
        result_df['Upper_Shadow'] = (result_df['High'] - np.maximum(result_df['Open'], result_df['Close'])) / result_df['Close']
        result_df['Lower_Shadow'] = (np.minimum(result_df['Open'], result_df['Close']) - result_df['Low']) / result_df['Close']

        # 価格位置（期間内での相対位置）
        period = lookback_periods.get("volatility", 20)
        close_min = result_df['Close'].rolling(window=period).min()
        close_max = result_df['Close'].rolling(window=period).max()
        result_df['Price_Position'] = DataValidator.safe_divide(
            result_df['Close'] - close_min,
            close_max - close_min,
            default_value=0.5
        )

        # ギャップ（前日終値との差）
        result_df['Gap'] = DataValidator.safe_divide(
            result_df['Open'] - result_df['Close'].shift(1),
            result_df['Close'].shift(1),
            default_value=0.0
        )

        logger.debug("価格特徴量計算完了")
        return result_df
    
    def calculate_volatility_features(
        self, 
        df: pd.DataFrame, 
        lookback_periods: Dict[str, int]
    ) -> pd.DataFrame:
        """
        ボラティリティ特徴量を計算
        
        Args:
            df: OHLCV価格データ
            lookback_periods: 計算期間設定
        
        Returns:
            ボラティリティ特徴量が追加されたDataFrame
        """
        try:
            result_df = df.copy()
            
            volatility_period = lookback_periods.get("volatility", 20)
            
            # リターンを計算
            result_df['Returns'] = result_df['Close'].pct_change()
            
            # 実現ボラティリティ
            result_df['Realized_Volatility_20'] = (
                result_df['Returns'].rolling(window=volatility_period).std() * np.sqrt(24)
            )
            
            # ボラティリティスパイク
            vol_ma = result_df['Realized_Volatility_20'].rolling(window=volatility_period).mean()
            result_df['Volatility_Spike'] = DataValidator.safe_divide(
                result_df['Realized_Volatility_20'],
                vol_ma,
                default_value=1.0
            )
            
            # ATR（Average True Range）
            result_df['TR'] = np.maximum(
                result_df['High'] - result_df['Low'],
                np.maximum(
                    abs(result_df['High'] - result_df['Close'].shift(1)),
                    abs(result_df['Low'] - result_df['Close'].shift(1))
                )
            )
            result_df['ATR_20'] = result_df['TR'].rolling(window=volatility_period).mean()
            
            # 正規化ATR
            result_df['ATR_Normalized'] = result_df['ATR_20'] / result_df['Close']
            
            # ボラティリティレジーム
            vol_quantile = result_df['Realized_Volatility_20'].rolling(window=volatility_period*2).quantile(0.8)
            result_df['High_Vol_Regime'] = (result_df['Realized_Volatility_20'] > vol_quantile).astype(int)
            
            # ボラティリティ変化率
            result_df['Vol_Change'] = result_df['Realized_Volatility_20'].pct_change()
            
            logger.debug("ボラティリティ特徴量計算完了")
            return result_df
            
        except Exception as e:
            logger.error(f"ボラティリティ特徴量計算エラー: {e}")
            return df
    
    def calculate_volume_features(
        self, 
        df: pd.DataFrame, 
        lookback_periods: Dict[str, int]
    ) -> pd.DataFrame:
        """
        出来高特徴量を計算
        
        Args:
            df: OHLCV価格データ
            lookback_periods: 計算期間設定
        
        Returns:
            出来高特徴量が追加されたDataFrame
        """
        try:
            result_df = df.copy()
            
            volume_period = lookback_periods.get("volume", 20)
            
            # 出来高移動平均
            result_df[f'Volume_MA_{volume_period}'] = result_df['Volume'].rolling(window=volume_period).mean()
            
            # 出来高比率
            result_df['Volume_Ratio'] = DataValidator.safe_divide(
                result_df['Volume'],
                result_df[f'Volume_MA_{volume_period}'],
                default_value=1.0
            )
            
            # 価格・出来高トレンド
            price_change = result_df['Close'].pct_change()
            volume_change = result_df['Volume'].pct_change()
            result_df['Price_Volume_Trend'] = price_change * volume_change
            
            # 出来高加重平均価格（VWAP）
            typical_price = (result_df['High'] + result_df['Low'] + result_df['Close']) / 3
            result_df['VWAP'] = DataValidator.safe_divide(
                (typical_price * result_df['Volume']).rolling(window=volume_period).sum(),
                result_df['Volume'].rolling(window=volume_period).sum(),
                default_value=typical_price
            )

            # VWAPからの乖離
            result_df['VWAP_Deviation'] = DataValidator.safe_divide(
                result_df['Close'] - result_df['VWAP'],
                result_df['VWAP'],
                default_value=0.0
            )
            
            # 出来高スパイク
            vol_threshold = result_df['Volume'].rolling(window=volume_period).quantile(0.9)
            result_df['Volume_Spike'] = (result_df['Volume'] > vol_threshold).astype(int)
            
            # 出来高トレンド
            result_df['Volume_Trend'] = DataValidator.safe_divide(
                result_df['Volume'].rolling(window=5).mean(),
                result_df['Volume'].rolling(window=volume_period).mean(),
                default_value=1.0
            )
            
            logger.debug("出来高特徴量計算完了")
            return result_df
            
        except Exception as e:
            logger.error(f"出来高特徴量計算エラー: {e}")
            return df
    
    def get_feature_names(self) -> list:
        """
        生成される価格特徴量名のリストを取得
        
        Returns:
            特徴量名のリスト
        """
        return [
            # 価格特徴量
            'Price_MA_Ratio_Short', 'Price_MA_Ratio_Long',
            'Price_Momentum_14', 'High_Low_Position',
            'Price_Change_1', 'Price_Change_5', 'Price_Change_20',
            'Price_Range', 'Body_Size', 'Upper_Shadow', 'Lower_Shadow',
            'Price_Position', 'Gap',
            
            # ボラティリティ特徴量
            'Realized_Volatility_20', 'Volatility_Spike', 'ATR_20',
            'ATR_Normalized', 'High_Vol_Regime', 'Vol_Change',
            
            # 出来高特徴量
            'Volume_Ratio', 'Price_Volume_Trend', 'VWAP_Deviation',
            'Volume_Spike', 'Volume_Trend'
        ]
