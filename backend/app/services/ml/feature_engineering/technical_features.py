"""
テクニカル指標特徴量計算クラス

従来のテクニカル指標（RSI、MACD、ストキャスティクスなど）と
テクニカル特徴量を計算します。
ATRやVWAPなど、他のファイルで重複していた指標もここに集約されました。
"""

import logging
from typing import Any, Dict


import pandas as pd
import numpy as np

from ...indicators.technical_indicators.momentum import MomentumIndicators
from ...indicators.technical_indicators.trend import TrendIndicators
from ...indicators.technical_indicators.volatility import VolatilityIndicators
from ...indicators.technical_indicators.volume import VolumeIndicators
from ....utils.error_handler import safe_ml_operation
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
        テクニカル特徴量を計算（BaseFeatureCalculatorの抽象メソッド実装）

        Args:
            df: OHLCV価格データ
            config: 計算設定（lookback_periodsを含む）

        Returns:
            テクニカル特徴量が追加されたDataFrame
        """
        lookback_periods = config.get("lookback_periods", {})

        # 複数のテクニカル特徴量を順次計算（全パターン生成）
        result_df = self.create_result_dataframe(df)  # Ensure result_df starts clean
        result_df = self.calculate_volatility_features(result_df, lookback_periods)
        result_df = self.calculate_volume_features(result_df, lookback_periods)
        result_df = self.calculate_market_regime_features(result_df, lookback_periods)
        result_df = self.calculate_momentum_features(result_df, lookback_periods)
        result_df = self.calculate_trend_features(result_df, lookback_periods)
        # パターン特徴量と分数次差分特徴量は削除

        return result_df

    # calculate_fractional_difference_features は削除されました

    def calculate_advanced_technical_features(
        self, df: pd.DataFrame, lookback_periods: Dict[str, int]
    ) -> pd.DataFrame:
        """
        高度な技術指標を計算（互換性とテスト用）
        実際には各カテゴリのメソッドに分散されていますが、
        テストがこのメソッドを呼ぶため、ラッパーとして提供します。
        """
        # 既にcalculate_featuresで呼ばれているメソッド群を実行すれば網羅されるはずだが、
        # 個別にテストしたい場合のために実装
        result_df = df.copy()
        result_df = self.calculate_trend_features(result_df, lookback_periods)

        # 他のカテゴリの追加分もここで呼ぶか、あるいはテスト側を修正するか。
        # テストは `calculate_advanced_technical_features` を呼んで `MFI` 等を期待している。
        # したがって、ここでそれらを計算して返す必要がある。

        # Volume features (MFI, OBV, AD, ADOSC)
        result_df = self.calculate_volume_features(result_df, lookback_periods)

        # Momentum features (Ultimate Oscillator)
        result_df = self.calculate_momentum_features(result_df, lookback_periods)

        # Volatility features (BBW, NATR, TRANGE)
        result_df = self.calculate_volatility_features(result_df, lookback_periods)

        # 分数次差分はデフォルトで無効化されているためここには含めない、
        # もしくはテスト用に明示的に呼び出す必要があるが、互換性メソッドなので一旦含めない。

        return result_df

    def calculate_market_regime_features(
        self, df: pd.DataFrame, lookback_periods: Dict[str, int] = None
    ) -> pd.DataFrame:
        """
        市場レジーム特徴量を計算

        Args:
            df: OHLCV価格データ
            lookback_periods: 計算期間設定（オプション、旧API互換用）

        Returns:
            市場レジーム特徴量が追加されたDataFrame
        """
        # 旧API互換：lookback_periodsがNoneの場合はデフォルト値を設定
        if lookback_periods is None:
            lookback_periods = {"short_ma": 10, "long_ma": 50, "volatility": 20}

        try:
            if not self.validate_input_data(df, ["close", "high", "low"]):
                return df

            result_df = self.create_result_dataframe(df)

            # 新しい特徴量を辞書で収集（DataFrame断片化対策）
            new_features = {}

            # レンジ相場判定 - 削除
            # volatility_period = ...
            # new_features["Range_Bound_Ratio"] = ...

            # 市場効率性（価格のランダムウォーク度）
            volatility_period = lookback_periods.get("volatility", 20)
            returns = result_df["close"].pct_change(fill_method=None).fillna(0)
            returns_lag1 = returns.shift(1)

            # Rolling correlation計算
            new_features["Market_Efficiency"] = (
                returns.rolling(window=volatility_period, min_periods=3)
                .corr(
                    returns_lag1.rolling(window=volatility_period, min_periods=3).mean()
                )
                .fillna(0.0)
            )

            # 一括で結合（DataFrame断片化回避）
            result_df = pd.concat(
                [result_df, pd.DataFrame(new_features, index=result_df.index)], axis=1
            )

            # Choppiness Index (CHOP)
            chop_window = 14
            chop = TrendIndicators.chop(
                high=result_df["high"],
                low=result_df["low"],
                close=result_df["close"],
                length=chop_window,
            )
            result_df[f"Choppiness_Index_{chop_window}"] = chop.fillna(50.0)

            # === Group C: Market Structure ===

            # 1. Amihud Illiquidity (非流動性指標)
            # |Return| / (Volume * Close)
            # 値が大きいほど流動性が低く、価格変動コストが高い
            illiquidity = returns.abs() / (
                result_df["volume"] * result_df["close"] + 1e-9
            )
            # 対数変換して分布を整える
            new_features["Amihud_Illiquidity"] = np.log(illiquidity + 1e-9).fillna(0.0)

            # 2. Kaufman Efficiency Ratio (KER / 効率性レシオ)
            # トレンドのノイズの少なさ (1に近いほど一直線、0に近いほどランダム)
            ker_period = 10
            change = result_df["close"].diff(ker_period).abs()
            volatility_sum = (
                result_df["close"].diff().abs().rolling(window=ker_period).sum()
            )
            new_features["Efficiency_Ratio"] = (
                change / (volatility_sum + 1e-9)
            ).fillna(0.0)

            # 3. Market Impact (Kyle's Lambda like)
            # (High - Low) / Volume
            # 1単位のボリュームあたりの価格変動幅
            market_impact = (result_df["high"] - result_df["low"]) / (
                result_df["volume"] + 1e-9
            )
            new_features["Market_Impact"] = np.log(market_impact + 1e-9).fillna(0.0)

            # Fractal Dimension Index (FDI) - 削除 (Choppiness Indexで代用)

            self.log_feature_calculation_complete("市場レジーム")
            return result_df

        except Exception as e:
            return self.handle_calculation_error(e, "市場レジーム特徴量計算", df)

    @safe_ml_operation(
        default_return=None, context="ボラティリティ特徴量計算でエラーが発生しました"
    )
    def calculate_volatility_features(
        self, df: pd.DataFrame, lookback_periods: Dict[str, int]
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
            if not self.validate_input_data(df, ["close", "high", "low"]):
                return df

            result_df = self.create_result_dataframe(df)

            volatility_period = lookback_periods.get("volatility", 20)

            # ATR - 削除 (NATRを使用)
            # atr_result = ...
            # result_df["ATR_20"] = ...

            # NATR (Normalized Average True Range)
            natr = VolatilityIndicators.natr(
                high=result_df["high"],
                low=result_df["low"],
                close=result_df["close"],
                length=volatility_period,
            )
            result_df["NATR"] = natr.fillna(0.0)

            # TRANGE - 削除
            # trange = ...
            # result_df["TRANGE"] = ...

            # BBW (Bollinger Band Width)
            upper, middle, lower = VolatilityIndicators.bbands(
                result_df["close"], length=volatility_period, std=2.0
            )
            # BBW = (Upper - Lower) / Middle
            bbw = (upper - lower) / middle
            result_df["BBW"] = bbw.replace([np.inf, -np.inf], np.nan).fillna(0.0)

            # Yang-Zhang Volatility (Open, High, Low, Close)
            if "open" in result_df.columns:
                yz_vol = VolatilityIndicators.yang_zhang(
                    open_=result_df["open"],
                    high=result_df["high"],
                    low=result_df["low"],
                    close=result_df["close"],
                    length=volatility_period,
                )
                result_df["Yang_Zhang_Vol_20"] = yz_vol.fillna(0.0)

            self.log_feature_calculation_complete("ボラティリティ")
            return result_df

        except Exception as e:
            return self.handle_calculation_error(e, "ボラティリティ特徴量計算", df)

    @safe_ml_operation(
        default_return=None, context="出来高特徴量計算でエラーが発生しました"
    )
    def calculate_volume_features(
        self, df: pd.DataFrame, lookback_periods: Dict[str, int]
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
            if not self.validate_input_data(df, ["volume", "close", "high", "low"]):
                return df

            result_df = self.create_result_dataframe(df)

            volume_period = lookback_periods.get("volume", 20)

            # 出来高移動平均（TrendIndicators使用）
            volume_ma = TrendIndicators.sma(result_df["volume"], length=volume_period)
            volume_ma = volume_ma.fillna(result_df["volume"])

            # 異常に大きな値をクリップ（最大値を制限）
            volume_max = (
                result_df["volume"].quantile(0.99) * 10
            )  # 99%分位点の10倍を上限とする
            result_df[f"Volume_MA_{volume_period}"] = np.clip(volume_ma, 0, volume_max)

            # 出来高加重平均価格（VWAP）- Deviation計算用
            vwap = VolumeIndicators.vwap(
                high=result_df["high"],
                low=result_df["low"],
                close=result_df["close"],
                volume=result_df["volume"],
                period=volume_period,
            ).fillna(result_df["close"])
            # result_df["VWAP"] = vwap # リストにないので追加しない

            # VWAPからの乖離
            result_df["VWAP_Deviation"] = (
                ((result_df["close"] - vwap) / vwap)
                .replace([np.inf, -np.inf], np.nan)
                .fillna(0.0)
            )

            # 出来高トレンド - 削除
            # volume_trend = ...
            # MFI (Money Flow Index)
            mfi = VolumeIndicators.mfi(
                high=result_df["high"],
                low=result_df["low"],
                close=result_df["close"],
                volume=result_df["volume"],
                length=14,
            )
            result_df["MFI"] = mfi.fillna(50.0)

            # OBV (On-Balance Volume)
            obv = VolumeIndicators.obv(
                close=result_df["close"], volume=result_df["volume"]
            )
            result_df["OBV"] = obv.fillna(0.0)

            # AD (Chaikin A/D Line)
            ad = VolumeIndicators.ad(
                high=result_df["high"],
                low=result_df["low"],
                close=result_df["close"],
                volume=result_df["volume"],
            )
            result_df["AD"] = ad.fillna(0.0)

            # ADOSC (Chaikin A/D Oscillator)
            adosc = VolumeIndicators.adosc(
                high=result_df["high"],
                low=result_df["low"],
                close=result_df["close"],
                volume=result_df["volume"],
                fast=3,
                slow=10,
            )
            result_df["ADOSC"] = adosc.fillna(0.0)

            self.log_feature_calculation_complete("出来高")
            return result_df

        except Exception as e:
            return self.handle_calculation_error(e, "出来高特徴量計算", df)

    # calculate_pattern_features は削除されました

    def _detect_price_patterns(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        簡易的な価格パターン検出

        Args:
            df: 価格データが含まれるDataFrame

        Returns:
            パターン特徴量が追加されたDataFrame
        """
        try:
            # 新しい特徴量を辞書で収集（DataFrame断片化対策）
            new_features = {}

            window_size = 20
            support_level = df["close"].rolling(window=window_size, min_periods=1).min()
            resistance_level = (
                df["close"].rolling(window=window_size, min_periods=1).max()
            )

            # 価格がサポート/レジスタントに近いことを示す特徴量
            new_features["Near_Support"] = self.safe_ratio_calculation(
                df["close"] - support_level,
                resistance_level - support_level,
                fill_value=0.5,
            )
            new_features["Near_Resistance"] = self.safe_ratio_calculation(
                resistance_level - df["close"],
                resistance_level - support_level,
                fill_value=0.5,
            )

            # 一括で結合
            new_df = pd.concat([df, pd.DataFrame(new_features, index=df.index)], axis=1)

            return new_df

        except Exception as e:
            logger.error(f"価格パターン検出エラー: {e}")
            # エラー時は元のDataFrameを返す
            return df

    def calculate_momentum_features(
        self, df: pd.DataFrame, lookback_periods: Dict[str, int] = None
    ) -> pd.DataFrame:
        """
        モメンタム特徴量を計算

        Args:
            df: OHLCV価格データ
            lookback_periods: 計算期間設定（オプション、旧API互換用）

        Returns:
            モメンタム特徴量が追加されたDataFrame
        """
        # 旧API互換：lookback_periodsがNoneの場合はデフォルト値を設定
        if lookback_periods is None:
            lookback_periods = {"short_ma": 10, "long_ma": 50}

        try:
            if not self.validate_input_data(df, ["close", "high", "low"]):
                return df

            result_df = self.create_result_dataframe(df)

            # 新しい特徴量を辞書で収集（DataFrame断片化対策）
            new_features = {}

            # RSI
            rsi = MomentumIndicators.rsi(result_df["close"], period=14)
            new_features["RSI"] = rsi.fillna(50.0)

            # MACD（Histogramのみ保持）
            macd, signal, hist = MomentumIndicators.macd(
                result_df["close"], fast=12, slow=26, signal=9
            )
            # new_features["MACD"] = macd.fillna(0.0)
            # new_features["MACD_Signal"] = signal.fillna(0.0)
            new_features["MACD_Histogram"] = hist.fillna(0.0)

            # ウィリアムズ%R
            willr = MomentumIndicators.willr(
                high=result_df["high"],
                low=result_df["low"],
                close=result_df["close"],
                length=14,
            )
            new_features["Williams_R"] = willr.fillna(-50.0)

            # CCI - 削除
            # cci = ...
            # new_features["CCI"] = ...

            # Stochastic RSI - 削除
            # stoch_rsi_k, stoch_rsi_d = ...
            # new_features["Stoch_RSI_K"] = ...

            # ROC - 削除
            # new_features["ROC"] = ...

            # モメンタム - 削除
            # new_features["Momentum"] = ...

            # Ultimate Oscillator - 削除
            # new_features["Ultimate_Oscillator"] = ...

            # 一括で結合
            result_df = pd.concat(
                [result_df, pd.DataFrame(new_features, index=result_df.index)], axis=1
            )

            self.log_feature_calculation_complete("モメンタム")
            return result_df

        except Exception as e:
            return self.handle_calculation_error(e, "モメンタム特徴量計算", df)

    def calculate_trend_features(
        self, df: pd.DataFrame, lookback_periods: Dict[str, int] = None
    ) -> pd.DataFrame:
        """
        トレンド特徴量を計算
        """
        if lookback_periods is None:
            lookback_periods = {"short_ma": 10, "long_ma": 50}

        try:
            if not self.validate_input_data(df, ["close", "high", "low"]):
                return df

            result_df = self.create_result_dataframe(df)
            new_features = {}

            # ADX
            adx, di_plus, di_minus = TrendIndicators.adx(
                high=result_df["high"],
                low=result_df["low"],
                close=result_df["close"],
                length=14,
            )
            new_features["ADX"] = adx.fillna(0.0)
            # new_features["DI_Plus"] = di_plus.fillna(0.0)
            # DI_Minus - 削除 (Ichimoku等と入れ替え)
            # new_features["DI_Minus"] = di_minus.fillna(0.0)

            # Aroon (Oscillatorのみ保持)
            aroon_up, aroon_down, aroon_osc = TrendIndicators.aroon(
                high=result_df["high"], low=result_df["low"], length=25
            )
            new_features["AROONOSC"] = aroon_osc.fillna(0.0)

            # 移動平均（MA_Long）
            long_ma = lookback_periods.get("long_ma", 50)
            ma_long = TrendIndicators.sma(result_df["close"], length=long_ma)
            new_features["MA_Long"] = ma_long.fillna(result_df["close"])

            # === Ichimoku Cloud ===
            # 転換線:9, 基準線:26, 先行スパンB:52
            ichimoku = MomentumIndicators.ichimoku(
                high=result_df["high"],
                low=result_df["low"],
                close=result_df["close"],
                tenkan_period=9,
                kijun_period=26,
                senkou_span_b_period=52,
            )

            # Tenkan - Kijun 距離 (正規化)
            tk_dist = (ichimoku["tenkan_sen"] - ichimoku["kijun_sen"]) / result_df[
                "close"
            ]
            new_features["Ichimoku_TK_Dist"] = tk_dist.fillna(0.0)

            # Close - Kijun 距離 (正規化)
            kijun_dist = (result_df["close"] - ichimoku["kijun_sen"]) / result_df[
                "close"
            ]
            new_features["Ichimoku_Kijun_Dist"] = kijun_dist.fillna(0.0)

            # === Parabolic SAR ===
            psar = TrendIndicators.sar(
                high=result_df["high"], low=result_df["low"], af=0.02, max_af=0.2
            )
            # トレンド方向と強度 (正規化)
            psar_trend = (result_df["close"] - psar) / result_df["close"]
            new_features["PSAR_Trend"] = psar_trend.fillna(0.0)

            # === SMA Cross (50 vs 200) ===
            sma_50 = TrendIndicators.sma(result_df["close"], length=50)
            sma_200 = TrendIndicators.sma(result_df["close"], length=200)
            # ゴールデンクロス/デッドクロスの距離 (正規化)
            sma_cross = (sma_50 - sma_200) / result_df["close"]
            new_features["SMA_Cross_50_200"] = sma_cross.fillna(0.0)

            # 一括で結合
            result_df = pd.concat(
                [result_df, pd.DataFrame(new_features, index=result_df.index)], axis=1
            )

            self.log_feature_calculation_complete("トレンド")
            return result_df

        except Exception as e:
            return self.handle_calculation_error(e, "トレンド特徴量計算", df)

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
            "BBW",
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
