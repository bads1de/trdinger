"""
HybridFeatureAdapter

StrategyGene → ML特徴量DataFrame変換を担当
GA + ML ハイブリッドアプローチの特徴量エンジニアリング
"""

import hashlib
import logging
from typing import Any, Callable, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

from app.services.auto_strategy.genes.strategy import StrategyGene
from app.services.ml.exceptions import MLFeatureError

logger = logging.getLogger(__name__)


class HybridFeatureAdapter:
    """
    StrategyGene → ML特徴量変換アダプタ

    GAで生成された戦略遺伝子をMLモデルで評価可能な特徴量に変換します。
    """

    def __init__(
        self,
        wavelet_config: Optional[Dict[str, Any]] = None,
        use_derived_features: bool = True,
        preprocess_handler: Optional[
            Callable[[pd.DataFrame, pd.DataFrame], Tuple[pd.DataFrame, pd.DataFrame]]
        ] = None,
    ):
        """
        初期化

        Args:
            wavelet_config: ウェーブレット変換設定
            use_derived_features: 派生特徴量を生成するか
            preprocess_handler: 前処理ハンドラ
        """
        self._use_derived_features = use_derived_features
        self._preprocess_handler = preprocess_handler
        self._preprocess_trainer = None
        self._wavelet_transformer: Optional["WaveletFeatureTransformer"] = None

        if isinstance(wavelet_config, dict) and wavelet_config.get("enabled", False):
            try:
                self._wavelet_transformer = WaveletFeatureTransformer(wavelet_config)
            except Exception as exc:
                logger.warning("WaveletFeatureTransformer初期化に失敗しました: %s", exc)
                self._wavelet_transformer = None

    def gene_to_features(
        self,
        gene: StrategyGene,
        ohlcv_data: pd.DataFrame,
        apply_preprocessing: bool = False,
        label_data: Optional[pd.DataFrame] = None,
        sentiment_scores: Optional[pd.Series] = None,
    ) -> pd.DataFrame:
        """
        戦略遺伝子と市場データを統合して、MLモデル用の特徴量DataFrameを生成します。

        工程:
        1. 遺伝子の構造（指標数、条件数、TP/SL設定等）を数値化
        2. イベントラベル（HRHP等）を統合
        3. 市場データ（OI/FR）の統計量を計算
        4. ウェーブレット変換や派生統計量を付与
        5. スケーリングやフィリング等の前処理を適用

        Args:
            gene: 変換対象の戦略遺伝子
            ohlcv_data: 特徴量抽出のベースとなるOHLCVデータ
            apply_preprocessing: スケーリング等の前処理を適用するか
            label_data: 外部から提供される正解ラベルや市場環境データ
            sentiment_scores: SNS等のセンチメントスコア

        Returns:
            統合された特徴量DataFrame
        """
        try:
            if gene is None or ohlcv_data is None or ohlcv_data.empty:
                raise MLFeatureError("入力データが無効です")

            features_df = ohlcv_data.copy()

            # 1. Gene情報から特徴量を抽出・追加
            for key, value in self._extract_gene_features(gene).items():
                features_df[key] = value

            # 2. イベントラベル特徴量（整列して統合）
            label_cols = ["label_hrhp", "label_lrlp", "market_regime"]
            if label_data is not None and not label_data.empty:
                aligned = label_data.reindex(features_df.index).ffill().fillna(0)
                for col in label_cols:
                    features_df[f"{col}_signal" if "label" in col else col] = (
                        aligned.get(col, 0.0)
                    )
            else:
                for col in label_cols:
                    features_df[f"{col}_signal" if "label" in col else col] = 0.0

            # 3. マーケット特性（OI/FR/Sentiment）
            # OI変化率
            if "open_interest" in features_df.columns:
                features_df["oi_pct_change"] = (
                    features_df["open_interest"]
                    .replace(0, np.nan)
                    .pct_change()
                    .fillna(0)
                )
            else:
                features_df["oi_pct_change"] = 0.0

            # FR変化
            if "funding_rate" in features_df.columns:
                features_df["funding_rate_change"] = (
                    features_df["funding_rate"].diff().fillna(0)
                )
            else:
                features_df["funding_rate_change"] = 0.0

            # センチメント
            if sentiment_scores is not None and not sentiment_scores.empty:
                features_df["sentiment_smoothed"] = (
                    sentiment_scores.reindex(features_df.index)
                    .ffill()
                    .rolling(3, min_periods=1)
                    .mean()
                    .fillna(0)
                )
            else:
                features_df["sentiment_smoothed"] = 0.0

            # 4. 拡張特徴量（Wavelet / 派生）
            if self._wavelet_transformer:
                features_df = self._wavelet_transformer.append_features(features_df)

            if self._use_derived_features:
                features_df = self._augment_with_derived_features(features_df)

            # 5. 前処理
            features_df = (
                self._apply_preprocessing(features_df)
                if apply_preprocessing
                else self._fallback_preprocess(features_df)
            )

            return features_df.loc[:, ~features_df.columns.duplicated()]

        except Exception as e:
            logger.error(f"Gene→特徴量変換エラー: {e}")
            raise MLFeatureError(f"変換失敗: {e}")

    def _extract_gene_features(self, gene: StrategyGene) -> Dict[str, Any]:
        """
        戦略遺伝子の構成（指標、条件、パラメータ、メタデータ等）をベクトル化可能な辞書に変換します。

        Args:
            gene: 抽出元の戦略遺伝子

        Returns:
            特徴量キーと値の辞書
        """
        enabled_inds = [ind for ind in gene.indicators if ind.enabled]
        long_c = len(gene.long_entry_conditions or [])
        short_c = len(gene.short_entry_conditions or [])

        periods = [
            i.parameters.get("period") for i in enabled_inds if "period" in i.parameters
        ]

        features = {
            "indicator_count": len(enabled_inds),
            "condition_count": long_c + short_c,
            "long_condition_count": long_c,
            "short_condition_count": short_c,
            "has_long_short_separation": int(bool(gene.has_long_short_separation())),
            "avg_indicator_period": np.mean(periods) if periods else 0.0,
        }

        # 指標の有無
        for itype in ["SMA", "EMA", "RSI", "MACD", "BollingerBands"]:
            features[f"has_{itype.lower()}"] = int(
                any(itype in ind.type for ind in enabled_inds)
            )

        # TP/SL & ポジションサイジング
        tpsl = gene.tpsl_gene
        features.update(
            {
                "has_tpsl": int(bool(tpsl and tpsl.enabled)),
                "take_profit_ratio": (
                    getattr(tpsl, "take_profit_pct", 0.02) if tpsl else 0.02
                ),
                "stop_loss_ratio": (
                    getattr(tpsl, "stop_loss_pct", 0.01) if tpsl else 0.01
                ),
            }
        )

        ps = gene.position_sizing_gene
        features["has_position_sizing"] = int(bool(ps and ps.enabled))
        method_str = str(getattr(ps, "method", ""))
        features["position_sizing_method"] = (
            int(hashlib.md5(method_str.encode()).hexdigest()[:6], 16) if ps else 0
        )

        return features

    def _fallback_preprocess(self, features_df: pd.DataFrame) -> pd.DataFrame:
        """
        特徴量の前処理

        Args:
            features_df: 特徴量DataFrame

        Returns:
            前処理後のDataFrame
        """
        processed = features_df.replace([np.inf, -np.inf], np.nan)
        processed = processed.ffill().bfill().fillna(0)
        return processed

    def _apply_preprocessing(self, features_df: pd.DataFrame) -> pd.DataFrame:
        """BaseMLTrainerの前処理ロジックを活用してスケーリングを行う"""

        preprocess_callable = self._get_preprocess_callable()
        if preprocess_callable is None:
            return self._fallback_preprocess(features_df)

        try:
            processed = preprocess_callable(features_df, features_df.copy())
            if isinstance(processed, tuple):
                return processed[0]
            return processed
        except Exception as exc:
            logger.warning(f"前処理の適用に失敗したためフォールバックします: {exc}")
            return self._fallback_preprocess(features_df)

    def _get_preprocess_callable(
        self,
    ) -> Optional[
        Callable[[pd.DataFrame, pd.DataFrame], Tuple[pd.DataFrame, pd.DataFrame]]
    ]:
        """BaseMLTrainer由来の前処理関数を取得"""

        if self._preprocess_handler is not None:
            return self._preprocess_handler

        try:
            from app.services.ml.base_ml_trainer import BaseMLTrainer

            if self._preprocess_trainer is None:
                self._preprocess_trainer = BaseMLTrainer(
                    trainer_config={"type": "single", "model_type": "lightgbm"},
                )

            return self._preprocess_trainer._preprocess_data  # type: ignore[return-value]
        except Exception as exc:
            logger.warning(f"BaseMLTrainer初期化に失敗しました: {exc}")
            return None

    def _augment_with_derived_features(self, features_df: pd.DataFrame) -> pd.DataFrame:
        """派生特徴量（ローリング統計量、リターン等）を生成"""

        augmented = features_df.copy()
        numeric_cols = augmented.select_dtypes(include=[np.number]).columns

        if "close" in augmented.columns:
            close = augmented["close"].astype(float)
            augmented["close_return_1"] = close.pct_change().fillna(0)
            augmented["close_return_5"] = close.pct_change(periods=5).fillna(0)
            augmented["close_rolling_mean_5"] = close.rolling(
                window=5, min_periods=1
            ).mean()
            augmented["close_rolling_std_5"] = (
                close.rolling(window=5, min_periods=1).std().fillna(0)
            )
            augmented["close_rolling_min_5"] = close.rolling(
                window=5, min_periods=1
            ).min()
            augmented["close_rolling_max_5"] = close.rolling(
                window=5, min_periods=1
            ).max()

        if "volume" in augmented.columns:
            volume = augmented["volume"].astype(float)
            augmented["volume_rolling_mean_5"] = volume.rolling(
                window=5, min_periods=1
            ).mean()
            augmented["volume_rolling_std_5"] = (
                volume.rolling(window=5, min_periods=1).std().fillna(0)
            )

        if set(["high", "low"]).issubset(augmented.columns):
            spread = augmented["high"].astype(float) - augmented["low"].astype(float)
            augmented["hl_spread"] = spread
            augmented["hl_spread_ratio"] = np.where(
                augmented["low"].astype(float) == 0,
                0,
                spread / augmented["low"].astype(float),
            )

        if numeric_cols.size > 0:
            shifted = augmented[numeric_cols].shift(1).add_suffix("_lag1")
            augmented = pd.concat([augmented, shifted], axis=1)

        augmented = augmented.ffill().bfill().fillna(0)
        return augmented

    def genes_to_features_batch(
        self,
        genes: List[StrategyGene],
        ohlcv_data: pd.DataFrame,
    ) -> List[pd.DataFrame]:
        """
        複数Geneのバッチ変換

        Args:
            genes: 戦略遺伝子リスト
            ohlcv_data: OHLCVデータ

        Returns:
            特徴量DataFrameのリスト
        """
        features_list = []

        for gene in genes:
            try:
                features_df = self.gene_to_features(gene, ohlcv_data)
                features_list.append(features_df)
            except Exception as e:
                gene_id = getattr(gene, "id", "unknown")
                logger.error(f"Gene {gene_id} の変換エラー: {e}")
                # エラー時は空のDataFrameを追加
                features_list.append(pd.DataFrame())

        return features_list


class WaveletFeatureTransformer:
    """簡易ウェーブレット変換で特徴量を生成するヘルパー"""

    def __init__(self, config: Dict[str, Any]):
        self.base_wavelet = config.get("base_wavelet", "haar")
        self.scales = self._validate_scales(config.get("scales", [2, 4]))
        self.target_columns = config.get("target_columns") or ["close"]

        if self.base_wavelet != "haar":
            logger.warning(
                "未対応のウェーブレット '%s' が指定されたため 'haar' を使用します",
                self.base_wavelet,
            )
            self.base_wavelet = "haar"

    @staticmethod
    def _validate_scales(scales: Any) -> List[int]:
        if not isinstance(scales, (list, tuple, set)):
            return [2, 4]
        validated: List[int] = []
        for scale in scales:
            try:
                scale_int = int(scale)
            except (TypeError, ValueError):
                continue
            if scale_int >= 2:
                validated.append(scale_int)
        return sorted(set(validated)) or [2, 4]

    def append_features(self, features_df: pd.DataFrame) -> pd.DataFrame:
        """指定カラムにウェーブレットディテール成分を付与"""

        for column in self.target_columns:
            if column not in features_df.columns:
                continue
            series = features_df[column]
            if not np.issubdtype(series.dtype, np.number):
                continue

            numeric_series = series.astype(float)
            for scale in self.scales:
                detail = self._haar_detail(numeric_series, scale)
                new_column = f"wavelet_{column}_scale_{scale}"
                features_df[new_column] = detail

        return features_df

    @staticmethod
    def _haar_detail(series: pd.Series, scale: int) -> pd.Series:
        """
        指定されたスケールでHaar様（移動平均との差分）のディテール成分を計算します。

        Args:
            series: 入力データ
            scale: 窓幅

        Returns:
            高周波（ディテール）成分の時系列
        """
        window = max(2, scale)
        coarse = series.rolling(window=window, min_periods=1).mean()
        detail = series - coarse
        detail = detail.replace([np.inf, -np.inf], 0.0).fillna(0.0)
        return detail
