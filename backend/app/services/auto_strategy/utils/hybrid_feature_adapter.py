"""
HybridFeatureAdapter

StrategyGene → ML特徴量DataFrame変換を担当
GA + ML ハイブリッドアプローチの特徴量エンジニアリング
"""

import logging
import hashlib
from typing import Any, Callable, Dict, List, Optional, Tuple

import pandas as pd
import numpy as np

from app.services.auto_strategy.models.strategy_gene import StrategyGene
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
        StrategyGene → 特徴量DataFrame変換

        Args:
            gene: 戦略遺伝子
            ohlcv_data: OHLCVデータ
            apply_preprocessing: 前処理を適用するか
            label_data: イベントドリブンラベルデータ
            sentiment_scores: センチメントスコア系列

        Returns:
            特徴量DataFrame

        Raises:
            MLFeatureError: 変換に失敗した場合
        """
        try:
            # 入力検証
            if gene is None:
                raise MLFeatureError("Geneがnullです")

            if ohlcv_data is None or ohlcv_data.empty:
                raise MLFeatureError("OHLCVデータが空です")

            # 基本特徴量を作成
            features_df = ohlcv_data.copy()
            if not isinstance(features_df, pd.DataFrame):
                raise MLFeatureError("OHLCVデータはDataFrameである必要があります")

            # Gene情報から特徴量を抽出
            gene_features = self._extract_gene_features(gene)

            # Gene特徴量をDataFrameに追加
            for key, value in gene_features.items():
                features_df[key] = value

            # イベントラベル特徴量
            if label_data is not None and not label_data.empty:
                aligned_labels = label_data.reindex(features_df.index)
                aligned_labels = aligned_labels.ffill().fillna(0)
                if "label_hrhp" in aligned_labels.columns:
                    features_df["label_hrhp_signal"] = aligned_labels[
                        "label_hrhp"
                    ].astype(float)
                else:
                    features_df["label_hrhp_signal"] = 0.0
                if "label_lrlp" in aligned_labels.columns:
                    features_df["label_lrlp_signal"] = aligned_labels[
                        "label_lrlp"
                    ].astype(float)
                else:
                    features_df["label_lrlp_signal"] = 0.0
                if "market_regime" in aligned_labels.columns:
                    features_df["market_regime"] = aligned_labels[
                        "market_regime"
                    ].astype(float)
                elif "market_regime" not in features_df:
                    features_df["market_regime"] = 0.0
            else:
                features_df["label_hrhp_signal"] = 0.0
                features_df["label_lrlp_signal"] = 0.0
                features_df["market_regime"] = 0.0

            # OI変化率
            if "open_interest" in features_df.columns:
                oi_series = features_df["open_interest"].astype(float)
                features_df["oi_pct_change"] = (
                    oi_series.replace(0, np.nan)
                    .pct_change()
                    .replace([np.inf, -np.inf], 0)
                    .fillna(0)
                )
            else:
                features_df["oi_pct_change"] = 0.0

            # ファンディングレート変化
            if "funding_rate" in features_df.columns:
                fr_series = features_df["funding_rate"].astype(float).fillna(0)
                features_df["funding_rate_change"] = fr_series.diff().fillna(0)
            else:
                features_df["funding_rate_change"] = 0.0

            # センチメント特徴
            if sentiment_scores is not None and not sentiment_scores.empty:
                aligned_sentiment = sentiment_scores.reindex(features_df.index)
                aligned_sentiment = aligned_sentiment.ffill().fillna(0.0)
                features_df["sentiment_smoothed"] = (
                    aligned_sentiment.rolling(window=3, min_periods=1)
                    .mean()
                    .fillna(0.0)
                )
            else:
                features_df["sentiment_smoothed"] = 0.0

            # ウェーブレット特徴量の追加
            if self._wavelet_transformer is not None:
                try:
                    features_df = self._wavelet_transformer.append_features(features_df)
                except Exception as exc:
                    logger.warning(
                        "ウェーブレット特徴量の追加に失敗したためスキップします: %s",
                        exc,
                    )

            # 派生特徴量生成（有効な場合）
            if self._use_derived_features:
                try:
                    features_df = self._augment_with_derived_features(features_df)
                except Exception as e:
                    logger.warning(f"派生特徴量生成エラー: {e}")

            # 前処理（オプション）
            if apply_preprocessing:
                features_df = self._apply_preprocessing(features_df)
            else:
                features_df = self._fallback_preprocess(features_df)

            # 重複カラムを排除（派生特徴量追加時の安全策）
            features_df = features_df.loc[:, ~features_df.columns.duplicated()]

            return features_df

        except MLFeatureError:
            raise
        except Exception as e:
            logger.error(f"Gene→特徴量変換エラー: {e}")
            raise MLFeatureError(f"Gene→特徴量変換に失敗: {e}")

    def _extract_gene_features(self, gene: StrategyGene) -> Dict[str, Any]:
        """
        StrategyGeneから特徴量を抽出

        Args:
            gene: 戦略遺伝子

        Returns:
            特徴量辞書
        """
        features = {}

        # インジケータ数
        features["indicator_count"] = len(gene.indicators) if gene.indicators else 0

        # 条件数
        long_conditions = gene.get_effective_long_conditions()
        short_conditions = gene.get_effective_short_conditions()
        features["condition_count"] = len(long_conditions) + len(short_conditions)
        features["long_condition_count"] = len(long_conditions)
        features["short_condition_count"] = len(short_conditions)

        # ロング・ショート分離フラグ
        features["has_long_short_separation"] = int(gene.has_long_short_separation())

        # インジケータタイプの特徴
        if gene.indicators:
            indicator_types = [getattr(ind, "type", "") for ind in gene.indicators]
            # 主要インジケータの存在フラグ
            features["has_sma"] = int("SMA" in indicator_types)
            features["has_ema"] = int("EMA" in indicator_types)
            features["has_rsi"] = int("RSI" in indicator_types)
            features["has_macd"] = int("MACD" in indicator_types)
            features["has_bollinger"] = int("BollingerBands" in indicator_types)

            # インジケータパラメータの平均
            periods = []
            for ind in gene.indicators:
                parameters = getattr(ind, "parameters", {}) or {}
                if "period" in parameters:
                    periods.append(parameters["period"])
            if periods:
                features["avg_indicator_period"] = np.mean(periods)
            else:
                features["avg_indicator_period"] = 0
        else:
            features.update(
                {
                    "has_sma": 0,
                    "has_ema": 0,
                    "has_rsi": 0,
                    "has_macd": 0,
                    "has_bollinger": 0,
                    "avg_indicator_period": 0,
                }
            )

        # TPSL設定
        if gene.tpsl_gene:
            features["has_tpsl"] = 1
            tpsl_gene = gene.tpsl_gene
            tp_value = getattr(tpsl_gene, "take_profit_ratio", None)
            if tp_value is None:
                tp_value = getattr(tpsl_gene, "take_profit_pct", 0)
            sl_value = getattr(tpsl_gene, "stop_loss_ratio", None)
            if sl_value is None:
                sl_value = getattr(tpsl_gene, "stop_loss_pct", 0)
            features["take_profit_ratio"] = tp_value
            features["stop_loss_ratio"] = sl_value
        else:
            features["has_tpsl"] = 0
            features["take_profit_ratio"] = 0
            features["stop_loss_ratio"] = 0

        # ポジションサイジング設定
        if gene.position_sizing_gene:
            features["has_position_sizing"] = 1
            method_repr = str(getattr(gene.position_sizing_gene, "method", ""))
            digest = hashlib.sha256(method_repr.encode("utf-8")).hexdigest()
            features["position_sizing_method"] = int(digest[:6], 16)
        else:
            features["has_position_sizing"] = 0
            features["position_sizing_method"] = 0

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
        window = max(2, scale)
        coarse = series.rolling(window=window, min_periods=1).mean()
        detail = series - coarse
        detail = detail.replace([np.inf, -np.inf], 0.0).fillna(0.0)
        return detail
