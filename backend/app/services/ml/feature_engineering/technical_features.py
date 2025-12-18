"""
テクニカル指標特徴量計算クラス

従来のテクニカル指標（RSI、MACD、ストキャスティクスなど）と
テクニカル特徴量を計算します。
ATRやVWAPなど、他のファイルで重複していた指標もここに集約されました。
"""

import logging
from typing import Any, Dict

import numpy as np
import pandas as pd

from ....utils.error_handler import safe_ml_operation
from ...indicators.technical_indicators.advanced_features import AdvancedFeatures
from ...indicators.technical_indicators.momentum import MomentumIndicators
from ...indicators.technical_indicators.trend import TrendIndicators
from ...indicators.technical_indicators.volatility import VolatilityIndicators
from ...indicators.technical_indicators.volume import VolumeIndicators
from .base_feature_calculator import BaseFeatureCalculator

logger = logging.getLogger(__name__)


class TechnicalFeatureCalculator(BaseFeatureCalculator):
    """
    テクニカル指標特徴量計算クラス

    従来のテクニカル指標特徴量を計算します。
    """

    def __init__(self):
        """初期化"""
        super().__init__()

    def calculate_features(
        self, df: pd.DataFrame, config: Dict[str, Any]
    ) -> pd.DataFrame:
        """
        テクニカル特徴量を計算
        """
        lookback_periods = config.get("lookback_periods", {})
        result_df = self.create_result_dataframe(df)

        # 各カテゴリの計算（result_dfを直接更新）
        result_df = self.calculate_volatility_features(result_df, lookback_periods)
        result_df = self.calculate_volume_features(result_df, lookback_periods)
        result_df = self.calculate_market_regime_features(result_df, lookback_periods)
        result_df = self.calculate_momentum_features(result_df, lookback_periods)
        result_df = self.calculate_trend_features(result_df, lookback_periods)

        # Fractional Differencing
        try:
            log_close = np.log(df["close"])
            result_df["FracDiff_04"] = AdvancedFeatures.frac_diff_ffd(log_close, d=0.4).fillna(0.0)
        except Exception as e:
            logger.warning(f"FracDiff計算失敗: {e}")
            result_df["FracDiff_04"] = 0.0

        return result_df

    def calculate_volatility_features(
        self, df: pd.DataFrame, lookback_periods: Dict[str, int]
    ) -> pd.DataFrame:
        """ボラティリティ特徴量を計算"""
        try:
            if not self.validate_input_data(df, ["close", "high", "low"]):
                return df

            vol_p = lookback_periods.get("volatility", 20)
            # NATR
            df["NATR"] = VolatilityIndicators.natr(
                high=df["high"], low=df["low"], close=df["close"], length=vol_p
            ).fillna(0.0)

            # BBW
            upper, middle, lower = VolatilityIndicators.bbands(df["close"], length=vol_p)
            df["BB_Width"] = ((upper - lower) / middle).replace([np.inf, -np.inf], np.nan).fillna(0.0)

            # Yang-Zhang, Parkinson, Garman-Klass
            if "open" in df.columns:
                df["Yang_Zhang_Vol_20"] = VolatilityIndicators.yang_zhang(
                    open_=df["open"], high=df["high"], low=df["low"], close=df["close"], length=vol_p
                ).fillna(0.0)
                df[f"Garman_Klass_Vol_{vol_p}"] = VolatilityIndicators.garman_klass(
                    open_=df["open"], high=df["high"], low=df["low"], close=df["close"], length=vol_p
                ).fillna(0.0)

            df[f"Parkinson_Vol_{vol_p}"] = VolatilityIndicators.parkinson(
                high=df["high"], low=df["low"], length=vol_p
            ).fillna(0.0)

            return df
        except Exception as e:
            return self.handle_calculation_error(e, "ボラティリティ計算", df)

    @safe_ml_operation(
        default_return=None, context="出来高特徴量計算でエラーが発生しました"
    )
    def calculate_volume_features(
        self, df: pd.DataFrame, lookback_periods: Dict[str, int]
    ) -> pd.DataFrame:
        """出来高特徴量を計算"""
        try:
            if not self.validate_input_data(df, ["volume", "close", "high", "low"]):
                return df

            vol_p = lookback_periods.get("volume", 20)

            # 出来高MA
            v_ma = TrendIndicators.sma(df["volume"], length=vol_p).fillna(df["volume"])
            v_max = df["volume"].quantile(0.99) * 10
            df[f"Volume_MA_{vol_p}"] = np.clip(v_ma, 0, v_max)

            # VWAP & Deviation
            vwap = VolumeIndicators.vwap(
                high=df["high"], low=df["low"], close=df["close"], volume=df["volume"], period=vol_p
            ).fillna(df["close"])
            df["VWAP_Deviation"] = ((df["close"] - vwap) / vwap).replace([np.inf, -np.inf], np.nan).fillna(0.0)

            # その他指標
            df["MFI"] = VolumeIndicators.mfi(
                high=df["high"], low=df["low"], close=df["close"], volume=df["volume"], length=14
            ).fillna(50.0)
            df["OBV"] = VolumeIndicators.obv(close=df["close"], volume=df["volume"]).fillna(0.0)
            df["AD"] = VolumeIndicators.ad(
                high=df["high"], low=df["low"], close=df["close"], volume=df["volume"]
            ).fillna(0.0)
            df["ADOSC"] = VolumeIndicators.adosc(
                high=df["high"], low=df["low"], close=df["close"], volume=df["volume"]
            ).fillna(0.0)
            df[f"VWAP_Z_Score_{vol_p}"] = VolumeIndicators.vwap_z_score(
                high=df["high"], low=df["low"], close=df["close"], volume=df["volume"], period=vol_p
            ).fillna(0.0)
            df[f"RVOL_{vol_p}"] = VolumeIndicators.rvol(df["volume"], window=vol_p).fillna(0.0)
            df[f"Absorption_Score_{vol_p}"] = VolumeIndicators.absorption_score(
                high=df["high"], low=df["low"], volume=df["volume"], window=vol_p
            ).fillna(0.0)

            return df
        except Exception as e:
            return self.handle_calculation_error(e, "出来高計算", df)

    def calculate_market_regime_features(
        self, df: pd.DataFrame, lookback_periods: Dict[str, int] = None
    ) -> pd.DataFrame:
        """市場レジーム特徴量を計算"""
        if lookback_periods is None:
            lookback_periods = {"volatility": 20}

        try:
            if not self.validate_input_data(df, ["close", "high", "low"]):
                return df

            vol_p = lookback_periods.get("volatility", 20)
            rets = df["close"].pct_change(fill_method=None).fillna(0)
            
            new_features = {
                "Market_Efficiency": rets.rolling(window=vol_p, min_periods=3).corr(
                    rets.shift(1).rolling(window=vol_p, min_periods=3).mean()
                ).fillna(0.0),
                "Choppiness_Index_14": TrendIndicators.chop(
                    high=df["high"], low=df["low"], close=df["close"], length=14
                ).fillna(50.0),
                "Amihud_Illiquidity": np.log(rets.abs() / (df["volume"] * df["close"] + 1e-9) + 1e-9).fillna(0.0),
                "Efficiency_Ratio": (df["close"].diff(10).abs() / (
                    df["close"].diff().abs().rolling(window=10).sum() + 1e-9
                )).fillna(0.0),
                "Market_Impact": np.log((df["high"] - df["low"]) / (df["volume"] + 1e-9) + 1e-9).fillna(0.0)
            }

            return pd.concat([df, pd.DataFrame(new_features, index=df.index)], axis=1)
        except Exception as e:
            return self.handle_calculation_error(e, "市場レジーム計算", df)

    def calculate_momentum_features(
        self, df: pd.DataFrame, lookback_periods: Dict[str, int] = None
    ) -> pd.DataFrame:
        """モメンタム特徴量を計算"""
        try:
            if not self.validate_input_data(df, ["close", "high", "low"]):
                return df

            _, _, hist = MomentumIndicators.macd(df["close"])
            new_features = {
                "RSI": MomentumIndicators.rsi(df["close"]).fillna(50.0),
                "MACD_Histogram": hist.fillna(0.0),
                "Williams_R": MomentumIndicators.willr(
                    high=df["high"], low=df["low"], close=df["close"]
                ).fillna(-50.0)
            }
            return pd.concat([df, pd.DataFrame(new_features, index=df.index)], axis=1)
        except Exception as e:
            return self.handle_calculation_error(e, "モメンタム計算", df)

    def calculate_trend_features(
        self, df: pd.DataFrame, lookback_periods: Dict[str, int] = None
    ) -> pd.DataFrame:
        """トレンド特徴量を計算"""
        lookback_periods = lookback_periods or {"long_ma": 50}
        try:
            if not self.validate_input_data(df, ["close", "high", "low"]):
                return df

            adx, _, _ = TrendIndicators.adx(high=df["high"], low=df["low"], close=df["close"])
            _, _, aroon_osc = TrendIndicators.aroon(high=df["high"], low=df["low"])
            ichimoku = MomentumIndicators.ichimoku(high=df["high"], low=df["low"], close=df["close"])
            psar = TrendIndicators.sar(high=df["high"], low=df["low"])
            
            new_features = {
                "ADX": adx.fillna(0.0),
                "AROONOSC": aroon_osc.fillna(0.0),
                "MA_Long": TrendIndicators.sma(df["close"], length=lookback_periods.get("long_ma", 50)).fillna(df["close"]),
                "Ichimoku_TK_Dist": ((ichimoku["tenkan_sen"] - ichimoku["kijun_sen"]) / df["close"]).fillna(0.0),
                "Ichimoku_Kijun_Dist": ((df["close"] - ichimoku["kijun_sen"]) / df["close"]).fillna(0.0),
                "PSAR_Trend": ((df["close"] - psar) / df["close"]).fillna(0.0),
                "SMA_Cross_50_200": ((TrendIndicators.sma(df["close"], 50) - TrendIndicators.sma(df["close"], 200)) / df["close"]).fillna(0.0)
            }
            return pd.concat([df, pd.DataFrame(new_features, index=df.index)], axis=1)
        except Exception as e:
            return self.handle_calculation_error(e, "トレンド計算", df)

    def get_feature_names(self) -> list:
        """
        生成されるテクニカル特徴量名のリストを取得

        Returns:
            特徴量名のリスト
        """
        return [
            # 市場レジーム特徴量
            "Market_Efficiency",
            "Choppiness_Index_14",
            "Amihud_Illiquidity",
            "Efficiency_Ratio",
            "Market_Impact",
            # モメンタム特徴量
            "RSI",
            "MACD_Histogram",
            "Williams_R",
            # トレンド特徴量
            "MA_Long",
            "Ichimoku_TK_Dist",
            "Ichimoku_Kijun_Dist",
            "PSAR_Trend",
            "SMA_Cross_50_200",
            # ボラティリティ特徴量
            "NATR",
            "BB_Width",
            "Yang_Zhang_Vol_20",
            # 出来高特徴量
            "Volume_MA_20",
            "VWAP_Deviation",
            "MFI",
            "OBV",
            "AD",
            "ADOSC",
            # トレンド特徴量
            "ADX",
            "AROONOSC",
        ]



