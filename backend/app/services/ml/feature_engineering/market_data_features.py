"""
市場データ特徴量計算クラス

ファンディングレート（FR）、建玉残高（OI）データから
市場の歪みや偏りを捉える特徴量を計算します。
"""

import logging
from typing import Any, Dict

import numpy as np
import pandas as pd

from .base_feature_calculator import BaseFeatureCalculator

logger = logging.getLogger(__name__)


class MarketDataFeatureCalculator(BaseFeatureCalculator):
    """
    市場データ特徴量計算クラス

    ファンディングレート、建玉残高データから特徴量を計算します。
    """

    def __init__(self):
        """初期化"""
        super().__init__()

    def calculate_features(
        self, df: pd.DataFrame, config: Dict[str, Any]
    ) -> pd.DataFrame:
        """
        市場データ特徴量を計算（BaseFeatureCalculatorの抽象メソッド実装）

        Args:
            df: OHLCV価格データ
            config: 計算設定（lookback_periods、funding_rate_data、open_interest_dataを含む）

        Returns:
            市場データ特徴量が追加されたDataFrame
        """
        lookback_periods = config.get("lookback_periods", {})
        funding_rate_data = config.get("funding_rate_data")
        open_interest_data = config.get("open_interest_data")

        result_df = df

        if funding_rate_data is not None:
            result_df = self.calculate_funding_rate_features(
                result_df, funding_rate_data, lookback_periods
            )

        if open_interest_data is not None:
            result_df = self.calculate_open_interest_features(
                result_df, open_interest_data, lookback_periods
            )

        if funding_rate_data is not None and open_interest_data is not None:
            result_df = self.calculate_composite_features(
                result_df, funding_rate_data, open_interest_data, lookback_periods
            )
            # 市場ダイナミクス特徴量は削除

        return result_df

    # calculate_market_dynamics_features は削除されました

    def _process_market_data(
        self,
        df: pd.DataFrame,
        data: pd.DataFrame,
        column_candidates: list[str],
        suffix: str,
    ) -> tuple[pd.DataFrame, str | None]:
        """
        市場データをマージし、カラムを特定して前処理を行う共通メソッド

        Args:
            df: ベースとなるDataFrame
            data: マージする市場データ
            column_candidates: カラム名の候補リスト
            suffix: マージ時のサフィックス

        Returns:
            (マージ済みDataFrame, 特定されたカラム名)
        """
        result_df = df.copy()

        # データをOHLCVデータにマージ
        if "timestamp" in data.columns:
            data = data.set_index("timestamp")

        # タイムゾーン調整: result_dfのインデックスに合わせてdataのインデックスを変換
        if isinstance(result_df.index, pd.DatetimeIndex) and isinstance(
            data.index, pd.DatetimeIndex
        ):
            try:
                if result_df.index.tz is not None and data.index.tz is None:
                    # result_dfがtz-awareでdataがtz-naiveなら、dataをlocalize
                    # ただし、単純なlocalizeだとズレる可能性があるので、UTCと仮定
                    data.index = data.index.tz_localize("UTC").tz_convert(
                        result_df.index.tz
                    )
                elif result_df.index.tz is None and data.index.tz is not None:
                    # result_dfがtz-naiveなら、dataもtz-naiveにする
                    data.index = data.index.tz_localize(None)
                elif (
                    result_df.index.tz is not None
                    and data.index.tz is not None
                    and result_df.index.tz != data.index.tz
                ):
                    # 両方tz-awareで異なる場合、合わせる
                    data.index = data.index.tz_convert(result_df.index.tz)
            except Exception as e:
                logger.warning(f"タイムゾーン調整エラー: {e}")

        # インデックスを合わせてマージ
        merged_df = result_df.join(data, how="left", rsuffix=suffix)

        # カラムを特定
        target_column = None
        for col in column_candidates:
            if col in merged_df.columns:
                target_column = col
                break

        if target_column is not None:
            # 欠損値を前方補完
            merged_df[target_column] = merged_df[target_column].ffill()

        return merged_df, target_column

    def calculate_funding_rate_features(
        self,
        df: pd.DataFrame,
        funding_rate_data: pd.DataFrame,
        lookback_periods: Dict[str, int],
    ) -> pd.DataFrame:
        """
        ファンディングレート特徴量を計算

        Args:
            df: OHLCV価格データ
            funding_rate_data: ファンディングレートデータ
            lookback_periods: 計算期間設定

        Returns:
            ファンディングレート特徴量が追加されたDataFrame
        """
        try:
            # 共通処理でデータマージとカラム特定
            merged_df, fr_column = self._process_market_data(
                df, funding_rate_data, ["funding_rate", "fundingRate", "rate"], "_fr"
            )

            if fr_column is None:
                logger.warning("ファンディングレートカラムが見つかりません")
                return df

            result_df = merged_df

            # === 新特徴量1: FR Extremity (Z-Score) ===
            # 極端に高い/低いFRを検出（過熱/過冷判定）
            # OIデータ不要なので、ここで計算
            fr_mean = result_df[fr_column].rolling(window=168, min_periods=1).mean()
            fr_std = (
                result_df[fr_column]
                .rolling(window=168, min_periods=1)
                .std()
                .replace(0, np.nan)
            )
            result_df["FR_Extremity_Zscore"] = (
                ((result_df[fr_column] - fr_mean) / fr_std)
                .replace([np.inf, -np.inf], np.nan)
                .fillna(0.0)
            )

            # === 新特徴量2: FR Momentum ===
            # FRの変化の勢いを測定（急激なFR変化を捉える）
            # OIデータ不要なので、ここで計算
            fr_change_rate = result_df[fr_column].pct_change(periods=8)
            fr_momentum = fr_change_rate - fr_change_rate.shift(8)
            result_df["FR_Momentum"] = fr_momentum.replace(
                [np.inf, -np.inf], np.nan
            ).fillna(0.0)

            # === 新特徴量3: FR Moving Average (24h) ===
            # FRの短期的な傾向（24時間移動平均）
            # 単発のスパイクを除去し、基調としての強気/弱気を判断
            result_df["FR_MA_24"] = (
                result_df[fr_column]
                .rolling(window=24, min_periods=1)
                .mean()
                .replace([np.inf, -np.inf], np.nan)
                .fillna(0.0)
            )

            # 削除: funding_rate (生データ) - 理由: 加工済み特徴量で代替（分析日: 2025-01-07）
            # 生のファンディングレートデータは使用せず、加工済み特徴量（複合特徴量）のみを使用
            # 実際に削除処理を実行
            if fr_column in result_df.columns:
                result_df = result_df.drop(columns=[fr_column])
                logger.info(f"Removed raw funding_rate column: {fr_column}")

            return result_df

        except Exception as e:
            logger.error(f"ファンディングレート特徴量計算エラー: {e}")
            return df

    def calculate_open_interest_features(
        self,
        df: pd.DataFrame,
        open_interest_data: pd.DataFrame,
        lookback_periods: Dict[str, int],
    ) -> pd.DataFrame:
        """
        建玉残高特徴量を計算

        Args:
            df: OHLCV価格データ
            open_interest_data: 建玉残高データ
            lookback_periods: 計算期間設定

        Returns:
            建玉残高特徴量が追加されたDataFrame
        """
        try:
            # 共通処理でデータマージとカラム特定
            merged_df, oi_column = self._process_market_data(
                df,
                open_interest_data,
                ["open_interest", "openInterest", "oi", "open_interest_value"],
                "_oi",
            )

            if oi_column is None:
                logger.warning("建玉残高カラムが見つかりません")
                return df

            result_df = merged_df

            # 削除: open_interest (生データ) - 理由: 加工済み特徴量で代替（分析日: 2025-01-07）
            # 生の建玉残高データは使用せず、加工済み特徴量（変化率、正規化値等）のみを使用
            # 実際に削除処理を実行
            if oi_column in result_df.columns:
                # 計算用にSeriesを保持してから削除
                oi_series = result_df[oi_column]
                result_df = result_df.drop(columns=[oi_column])
                logger.info(f"Removed raw open_interest column: {oi_column}")
            else:
                # 万が一カラムがない場合（通常ありえないが）
                logger.warning(f"OI column {oi_column} not found in merged dataframe")
                return df

            # 共通ロジックを使用してOI特徴量を計算
            result_df = self._calculate_oi_derived_features(result_df, oi_series)

            return result_df

        except Exception as e:
            logger.error(f"建玉残高特徴量計算エラー: {e}")
            return df

    def calculate_pseudo_open_interest_features(
        self, df: pd.DataFrame, lookback_periods: Dict[str, int]
    ) -> pd.DataFrame:
        """
        建玉残高疑似特徴量を生成

        Args:
            df: 価格データ
            lookback_periods: 計算期間設定

        Returns:
            疑似特徴量が追加されたDataFrame
        """
        try:
            result_df = df.copy()

            # ボリュームベースの疑似建玉残高
            # volumeは必須カラムと想定
            if "volume" not in result_df.columns:
                logger.warning("volumeカラムがないため、疑似OI特徴量を生成できません")
                return result_df

            pseudo_oi = result_df["volume"].rolling(24).mean() * 10
            # 明示的にpandas Seriesであることを保証
            pseudo_oi = pd.Series(pseudo_oi, index=result_df.index)

            # 共通ロジックを使用してOI特徴量を計算
            result_df = self._calculate_oi_derived_features(result_df, pseudo_oi)

            logger.info("建玉残高疑似特徴量を生成しました")
            return result_df

        except Exception as e:
            logger.error(f"建玉残高疑似特徴量生成エラー: {e}")
            return df

        return result_df

    def _calculate_oi_derived_features(
        self, df: pd.DataFrame, oi_series: pd.Series
    ) -> pd.DataFrame:
        """
        建玉残高（または疑似建玉残高）から派生特徴量を計算する共通メソッド

        Args:
            df: 特徴量を追加するDataFrame (価格データを含む)
            oi_series: 建玉残高のSeries

        Returns:
            特徴量が追加されたDataFrame
        """
        result_df = df.copy()

        # === 新特徴量1: OI RSI (Open Interest RSI) ===
        # 価格ではなくOIの過熱感を測定
        # OIが急激に増加しすぎている（買われすぎ/売られすぎ）状態を検知
        delta = oi_series.diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
        rs = gain / loss.replace(0, np.nan)
        result_df["OI_RSI"] = 100 - (100 / (1 + rs))
        result_df["OI_RSI"] = result_df["OI_RSI"].fillna(50.0)  # 欠損値は中立50で埋める

        # === 新特徴量2: Volume / OI Ratio ===
        # 出来高と建玉の比率（投機熱の指標）
        if "volume" in result_df.columns:
            vol_oi_ratio = np.log((result_df["volume"] + 1) / (oi_series + 1))
            result_df["Volume_OI_Ratio"] = vol_oi_ratio.replace(
                [np.inf, -np.inf], np.nan
            ).fillna(0.0)
        else:
            result_df["Volume_OI_Ratio"] = 0.0

        # === 新特徴量3: OI Change Rate ===
        # 建玉残高の変化率（トレンドの強さを測定）
        oi_change = (
            oi_series.pct_change(periods=1)
            .replace([np.inf, -np.inf], np.nan)
            .fillna(0.0)
        )
        result_df["OI_Change_Rate"] = oi_change

        # === 新特徴量4: Price-OI Divergence ===
        # 価格とOIのダイバージェンスを検知
        price_change = result_df["close"].pct_change(periods=1).fillna(0.0)
        result_df["Price_OI_Divergence"] = (
            (price_change * oi_change).rolling(window=8).mean().fillna(0.0)
        )

        return result_df

    def calculate_composite_features(
        self,
        df: pd.DataFrame,
        funding_rate_data: pd.DataFrame,
        open_interest_data: pd.DataFrame,
        lookback_periods: Dict[str, int],
    ) -> pd.DataFrame:
        """
        複合特徴量を計算（FR + OI）

        Args:
            df: OHLCV価格データ
            funding_rate_data: ファンディングレートデータ
            open_interest_data: 建玉残高データ
            lookback_periods: 計算期間設定

        Returns:
            複合特徴量が追加されたDataFrame
        """
        try:
            result_df = df.copy()

            # 両方のデータをマージ
            if "timestamp" in funding_rate_data.columns:
                funding_rate_data = funding_rate_data.set_index("timestamp")
            if "timestamp" in open_interest_data.columns:
                open_interest_data = open_interest_data.set_index("timestamp")

            merged_df = result_df.join(funding_rate_data, how="left", rsuffix="_fr")
            merged_df = merged_df.join(open_interest_data, how="left", rsuffix="_oi")

            # カラムを特定
            fr_column = None
            for col in ["funding_rate", "fundingRate", "rate"]:
                if col in merged_df.columns:
                    fr_column = col
                    break

            oi_column = None
            for col in ["open_interest", "openInterest", "oi", "open_interest_value"]:
                if col in merged_df.columns:
                    oi_column = col
                    break

            if fr_column is None or oi_column is None:
                logger.warning(
                    "ファンディングレートまたは建玉残高カラムが見つかりません"
                )
                return result_df

            # 欠損値を前方補完
            merged_df[fr_column] = merged_df[fr_column].ffill()
            merged_df[oi_column] = merged_df[oi_column].ffill()

            # === 新特徴量3: FR Cumulative Trend ===
            # FRの累積値（ポジション保有コストの蓄積）と価格トレンドの関係
            # 24時間のFR累積和
            fr_cum_24h = merged_df[fr_column].rolling(window=24).sum()

            # 正規化して扱いやすくする
            fr_cum_mean = fr_cum_24h.rolling(window=168, min_periods=1).mean()
            fr_cum_std = (
                fr_cum_24h.rolling(window=168, min_periods=1).std().replace(0, np.nan)
            )
            result_df["FR_Cumulative_Trend"] = (
                ((fr_cum_24h - fr_cum_mean) / fr_cum_std)
                .replace([np.inf, -np.inf], np.nan)
                .fillna(0.0)
            )

            # 市場ストレス指標: FR_Extremity_ZscoreとOI正規化の複合
            # FR_Extremity_Zscoreは既にcalculate_funding_rate_featuresで計算済み
            # ここではOI成分のみを追加して計算する
            oi_mean = merged_df[oi_column].rolling(window=168, min_periods=1).mean()
            oi_std = (
                merged_df[oi_column]
                .rolling(window=168, min_periods=1)
                .std()
                .replace(0, np.nan)
            )
            oi_normalized = (
                ((merged_df[oi_column] - oi_mean) / oi_std)
                .replace([np.inf, -np.inf], np.nan)
                .fillna(0.0)
            )

            # FR_Extremity_Zscoreが既に計算されている場合はそれを使用
            # そうでない場合は再計算
            if "FR_Extremity_Zscore" in result_df.columns:
                fr_normalized = result_df["FR_Extremity_Zscore"]
            else:
                fr_mean = merged_df[fr_column].rolling(window=168, min_periods=1).mean()
                fr_std = (
                    merged_df[fr_column]
                    .rolling(window=168, min_periods=1)
                    .std()
                    .replace(0, np.nan)
                )
                fr_normalized = (
                    ((merged_df[fr_column] - fr_mean) / fr_std)
                    .replace([np.inf, -np.inf], np.nan)
                    .fillna(0.0)
                )

            result_df["Market_Stress"] = np.sqrt(fr_normalized**2 + oi_normalized**2)

            # === 新特徴量5: FR-OI Sentiment ===
            # FRとOIの組み合わせによるセンチメント指標
            # FR > 0 (Long払い) かつ OI増加 => 強気過熱 (Long積み増し) => +1方向
            # FR < 0 (Short払い) かつ OI増加 => 弱気過熱 (Short積み増し) => -1方向
            # OI減少はポジション解消を示唆

            # OIの変化率（短期）
            oi_change_short = merged_df[oi_column].pct_change(periods=1).fillna(0.0)

            # FRの符号（+1 or -1）
            fr_sign = np.sign(merged_df[fr_column]).replace(0, 1)  # 0の場合は1とみなす

            # センチメントスコア: FRの符号 * OIの変化率
            # 例: FR正(+) * OI増(+) = + (強気圧力増加)
            # 例: FR負(-) * OI増(+) = - (弱気圧力増加)
            # 例: FR正(+) * OI減(-) = - (強気圧力減少＝利確/損切り)
            result_df["FR_OI_Sentiment"] = (
                (fr_sign * oi_change_short).rolling(window=8).mean().fillna(0.0)
            )

            # === 新特徴量6: Liquidation Risk (Pseudo) ===
            # 清算リスク（疑似）: 価格の急変動とOIの積み上がりの積
            # 価格が急激に動いているのにOIが減っていない＝清算の燃料が溜まっている
            price_volatility = (
                result_df["close"].pct_change(periods=1).abs().rolling(window=4).mean()
            )
            result_df["Liquidation_Risk"] = price_volatility * oi_normalized

            # === 新特徴量7: OI Weighted Price (VWAP like) ===
            # OI加重価格と現在価格の乖離
            # OIが高い価格帯は重要なサポート/レジスタンスになる可能性
            oi_weighted_price = (result_df["close"] * merged_df[oi_column]).rolling(
                window=24
            ).sum() / merged_df[oi_column].rolling(window=24).sum()
            result_df["OI_Weighted_Price_Dev"] = (
                result_df["close"] - oi_weighted_price
            ) / oi_weighted_price

            # === 新特徴量8: FR Volatility ===
            # FRのボラティリティ（市場の不安定さ）
            result_df["FR_Volatility"] = (
                merged_df[fr_column].rolling(window=24).std().fillna(0.0)
            )

            # === 新特徴量9: OI Trend Efficiency ===
            # 価格変化 / OI変化（効率性レシオ）
            # 小さなOI変化で価格が大きく動く＝低流動性・高インパクト
            # 大きなOI変化で価格が動かない＝吸収・蓄積
            price_abs_change = result_df["close"].pct_change().abs()
            oi_abs_change = (
                merged_df[oi_column].pct_change().abs().replace(0, np.nan)
            )  # 0除算回避
            result_df["OI_Trend_Efficiency"] = (
                (price_abs_change / oi_abs_change)
                .replace([np.inf, -np.inf], np.nan)
                .fillna(0.0)
            )

            # === 新特徴量10: Volume / OI Ratio (Speculation Index) ===
            # 出来高 / OI（投機熱）
            # 高い: 短期売買中心（過熱）
            # 低い: ポジション積み上げ（エネルギー充填）
            if "volume" in result_df.columns:
                result_df["Volume_OI_Ratio"] = (
                    (result_df["volume"] / merged_df[oi_column])
                    .replace([np.inf, -np.inf], np.nan)
                    .fillna(0.0)
                )
            else:
                result_df["Volume_OI_Ratio"] = 0.0

            return result_df

        except Exception as e:
            logger.error(f"複合特徴量計算エラー: {e}")
            return df

    def get_feature_names(self) -> list:
        """
        生成される市場データ特徴量名のリストを取得

        Returns:
            特徴量名のリスト
        """
        return [
            # 建玉残高特徴量
            "OI_RSI",  # 新特徴量
            # "Volume_OI_Ratio",  # 新特徴量 (複合特徴量に移動)
            "OI_Change_Rate",  # 新特徴量
            "Price_OI_Divergence",  # 新特徴量
            # 複合特徴量
            "FR_Cumulative_Trend",  # 新特徴量
            "FR_Extremity_Zscore",
            "FR_Momentum",
            "FR_MA_24",  # 新特徴量
            "Market_Stress",
            "FR_OI_Sentiment",  # 新特徴量
            "Liquidation_Risk",  # 新特徴量
            "OI_Weighted_Price_Dev",  # 新特徴量
            "FR_Volatility",  # 新特徴量
            "OI_Trend_Efficiency",  # 新特徴量
            "Volume_OI_Ratio",  # 新特徴量
        ]
