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

        # 特徴量を一時的に保持する辞書
        new_features = {}

        # 内部メソッド（Dictを返す）を呼び出して収集
        new_features.update(
            self._calculate_volatility_features_internal(df, lookback_periods)
        )
        new_features.update(
            self._calculate_volume_features_internal(df, lookback_periods)
        )
        new_features.update(
            self._calculate_market_regime_features_internal(df, lookback_periods)
        )
        new_features.update(
            self._calculate_momentum_features_internal(df, lookback_periods)
        )
        new_features.update(
            self._calculate_trend_features_internal(df, lookback_periods)
        )

        # Fractional Differencing
        try:
            log_close = np.log(df["close"])
            new_features["FracDiff_04"] = AdvancedFeatures.frac_diff_ffd(
                log_close, d=0.4
            ).fillna(0.0)
        except Exception as e:
            logger.warning(f"FracDiff計算失敗: {e}")
            new_features["FracDiff_04"] = pd.Series(0.0, index=df.index)

        # 一括結合（BaseFeatureCalculatorの高速化メソッドを利用）
        return self.create_result_dataframe_efficient(df, new_features)

    def calculate_volatility_features(
        self, df: pd.DataFrame, lookback_periods: Dict[str, int]
    ) -> pd.DataFrame:
        """ボラティリティ特徴量を計算 (公開API: DataFrameを返す)"""
        features = self._calculate_volatility_features_internal(df, lookback_periods)
        return self.create_result_dataframe_efficient(df, features)

    def _calculate_volatility_features_internal(
        self, df: pd.DataFrame, lookback_periods: Dict[str, int]
    ) -> Dict[str, pd.Series]:
        """ボラティリティ特徴量を計算 (内部用: Dictを返す)"""
        features = {}
        try:
            if not self.validate_input_data(df, ["close", "high", "low"]):
                return {}

            vol_p = lookback_periods.get("volatility", 20)
            # NATR
            features["NATR"] = VolatilityIndicators.natr(
                high=df["high"], low=df["low"], close=df["close"], length=vol_p
            ).fillna(0.0)

            # BBW
            upper, middle, lower = VolatilityIndicators.bbands(
                df["close"], length=vol_p
            )
            features["BB_Width"] = (
                ((upper - lower) / (middle.replace(0, np.nan)))
                .replace([np.inf, -np.inf], np.nan)
                .fillna(0.0)
            )

            # Yang-Zhang, Parkinson, Garman-Klass
            if "open" in df.columns:
                features["Yang_Zhang_Vol_20"] = VolatilityIndicators.yang_zhang(
                    open_=df["open"],
                    high=df["high"],
                    low=df["low"],
                    close=df["close"],
                    length=vol_p,
                ).fillna(0.0)
                features[f"Garman_Klass_Vol_{vol_p}"] = (
                    VolatilityIndicators.garman_klass(
                        open_=df["open"],
                        high=df["high"],
                        low=df["low"],
                        close=df["close"],
                        length=vol_p,
                    ).fillna(0.0)
                )

            features[f"Parkinson_Vol_{vol_p}"] = VolatilityIndicators.parkinson(
                high=df["high"], low=df["low"], length=vol_p
            ).fillna(0.0)

            return features
        except Exception as e:
            logger.error(f"ボラティリティ特徴量計算エラー: {e}")
            return {}

    def calculate_volume_features(
        self, df: pd.DataFrame, lookback_periods: Dict[str, int]
    ) -> pd.DataFrame:
        """出来高特徴量を計算 (公開API)"""
        features = self._calculate_volume_features_internal(df, lookback_periods)
        return self.create_result_dataframe_efficient(df, features)

    @safe_ml_operation(
        default_return={}, context="出来高特徴量計算でエラーが発生しました"
    )
    def _calculate_volume_features_internal(
        self, df: pd.DataFrame, lookback_periods: Dict[str, int]
    ) -> Dict[str, pd.Series]:
        """出来高特徴量を計算 (内部用)"""
        features = {}
        try:
            if not self.validate_input_data(df, ["volume", "close", "high", "low"]):
                return {}

            vol_p = lookback_periods.get("volume", 20)

            # 出来高MA
            v_ma = TrendIndicators.sma(df["volume"], length=vol_p).fillna(df["volume"])
            v_max = df["volume"].quantile(0.99) * 10
            features[f"Volume_MA_{vol_p}"] = np.clip(v_ma, 0, v_max)

            # VWAP & Deviation
            vwap = VolumeIndicators.vwap(
                high=df["high"],
                low=df["low"],
                close=df["close"],
                volume=df["volume"],
                period=vol_p,
            ).fillna(df["close"])
            features["VWAP_Deviation"] = (
                ((df["close"] - vwap) / (vwap.replace(0, np.nan)))
                .replace([np.inf, -np.inf], np.nan)
                .fillna(0.0)
            )

            # その他指標
            try:
                features["MFI"] = VolumeIndicators.mfi(
                    high=df["high"],
                    low=df["low"],
                    close=df["close"],
                    volume=df["volume"],
                    length=14,
                ).fillna(50.0)
            except Exception:
                features["MFI"] = pd.Series(50.0, index=df.index)

            features["OBV"] = VolumeIndicators.obv(
                close=df["close"], volume=df["volume"]
            ).fillna(0.0)
            features["AD"] = VolumeIndicators.ad(
                high=df["high"], low=df["low"], close=df["close"], volume=df["volume"]
            ).fillna(0.0)
            features["ADOSC"] = VolumeIndicators.adosc(
                high=df["high"], low=df["low"], close=df["close"], volume=df["volume"]
            ).fillna(0.0)
            features[f"VWAP_Z_Score_{vol_p}"] = VolumeIndicators.vwap_z_score(
                high=df["high"],
                low=df["low"],
                close=df["close"],
                volume=df["volume"],
                period=vol_p,
            ).fillna(0.0)
            features[f"RVOL_{vol_p}"] = VolumeIndicators.rvol(
                df["volume"], window=vol_p
            ).fillna(0.0)
            features[f"Absorption_Score_{vol_p}"] = VolumeIndicators.absorption_score(
                high=df["high"], low=df["low"], volume=df["volume"], window=vol_p
            ).fillna(0.0)

            return features
        except Exception as e:
            logger.error(f"出来高特徴量計算エラー: {e}")
            return {}

    def calculate_market_regime_features(
        self, df: pd.DataFrame, lookback_periods: Dict[str, int] = None
    ) -> pd.DataFrame:
        """市場レジーム特徴量を計算 (公開API)"""
        features = self._calculate_market_regime_features_internal(df, lookback_periods)
        return self.create_result_dataframe_efficient(df, features)

    def _calculate_market_regime_features_internal(
        self, df: pd.DataFrame, lookback_periods: Dict[str, int] = None
    ) -> Dict[str, pd.Series]:
        """市場レジーム特徴量を計算 (内部用)"""
        if lookback_periods is None:
            lookback_periods = {"volatility": 20}

        try:
            if not self.validate_input_data(df, ["close", "high", "low"]):
                return {}

            vol_p = lookback_periods.get("volatility", 20)
            rets = df["close"].pct_change(fill_method=None).fillna(0)

            new_features = {
                "Market_Efficiency": rets.rolling(window=vol_p, min_periods=3)
                .corr(rets.shift(1).rolling(window=vol_p, min_periods=3).mean())
                .fillna(0.0),
                "Choppiness_Index_14": TrendIndicators.chop(
                    high=df["high"], low=df["low"], close=df["close"], length=14
                ).fillna(50.0),
                "Amihud_Illiquidity": np.log(
                    rets.abs() / (df["volume"] * df["close"] + 1e-9) + 1e-9
                ).fillna(0.0),
                "Efficiency_Ratio": (
                    df["close"].diff(10).abs()
                    / (df["close"].diff().abs().rolling(window=10).sum() + 1e-9)
                ).fillna(0.0),
                "Market_Impact": np.log(
                    (df["high"] - df["low"]) / (df["volume"] + 1e-9) + 1e-9
                ).fillna(0.0),
            }

            return new_features
        except Exception as e:
            logger.error(f"市場レジーム特徴量計算エラー: {e}")
            return {}

    def calculate_momentum_features(
        self, df: pd.DataFrame, lookback_periods: Dict[str, int] = None
    ) -> pd.DataFrame:
        """モメンタム特徴量を計算 (公開API)"""
        features = self._calculate_momentum_features_internal(df, lookback_periods)
        return self.create_result_dataframe_efficient(df, features)

    def _calculate_momentum_features_internal(
        self, df: pd.DataFrame, lookback_periods: Dict[str, int] = None
    ) -> Dict[str, pd.Series]:
        """モメンタム特徴量を計算 (内部用)"""
        try:
            if not self.validate_input_data(df, ["close", "high", "low"]):
                return {}

            _, _, hist = MomentumIndicators.macd(df["close"])
            new_features = {
                "RSI": MomentumIndicators.rsi(df["close"]).fillna(50.0),
                "MACD_Histogram": hist.fillna(0.0),
                "Williams_R": MomentumIndicators.willr(
                    high=df["high"], low=df["low"], close=df["close"]
                ).fillna(-50.0),
            }
            return new_features
        except Exception as e:
            logger.error(f"モメンタム特徴量計算エラー: {e}")
            return {}

    def calculate_trend_features(
        self, df: pd.DataFrame, lookback_periods: Dict[str, int] = None
    ) -> pd.DataFrame:
        """トレンド特徴量を計算 (公開API)"""
        features = self._calculate_trend_features_internal(df, lookback_periods)
        return self.create_result_dataframe_efficient(df, features)

    def _calculate_trend_features_internal(
        self, df: pd.DataFrame, lookback_periods: Dict[str, int] = None
    ) -> Dict[str, pd.Series]:
        """トレンド特徴量を計算 (内部用)"""
        lookback_periods = lookback_periods or {"long_ma": 50}
        try:
            if not self.validate_input_data(df, ["close", "high", "low"]):
                return {}

            adx, _, _ = TrendIndicators.adx(
                high=df["high"], low=df["low"], close=df["close"]
            )
            _, _, aroon_osc = TrendIndicators.aroon(high=df["high"], low=df["low"])
            ichimoku = TrendIndicators.ichimoku(
                high=df["high"], low=df["low"], close=df["close"]
            )
            psar = TrendIndicators.sar(high=df["high"], low=df["low"])

            # Ichimoku安全策
            if (
                not ichimoku.empty
                and "tenkan_sen" in ichimoku.columns
                and "kijun_sen" in ichimoku.columns
            ):
                tenkan = ichimoku["tenkan_sen"]
                kijun = ichimoku["kijun_sen"]
            else:
                tenkan = df["close"]
                kijun = df["close"]

            new_features = {
                "ADX": adx.fillna(0.0),
                "AROONOSC": aroon_osc.fillna(0.0),
                "MA_Long": TrendIndicators.sma(
                    df["close"], length=lookback_periods.get("long_ma", 50)
                ).fillna(df["close"]),
                "Ichimoku_TK_Dist": (
                    (tenkan - kijun) / (df["close"].replace(0, np.nan))
                ).fillna(0.0),
                "Ichimoku_Kijun_Dist": (
                    (df["close"] - kijun) / (df["close"].replace(0, np.nan))
                ).fillna(0.0),
                "PSAR_Trend": (
                    (df["close"] - psar) / (df["close"].replace(0, np.nan))
                ).fillna(0.0),
                "SMA_Cross_50_200": (
                    (
                        TrendIndicators.sma(df["close"], 50)
                        - TrendIndicators.sma(df["close"], 200)
                    )
                    / (df["close"].replace(0, np.nan))
                ).fillna(0.0),
            }
            return new_features
        except Exception as e:
            logger.error(f"トレンド特徴量計算エラー: {e}")
            return {}

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
