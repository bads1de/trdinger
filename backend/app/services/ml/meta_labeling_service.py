import logging
import pandas as pd
from typing import Dict, Any, Optional, Tuple, List
from sklearn.ensemble import RandomForestClassifier

logger = logging.getLogger(__name__)


class MetaLabelingService:
    """
    メタラベリングサービス

    一次モデル（Primary Model）の予測結果を入力として、
    その予測が正解（収益につながる）かどうかを判定する二次モデル（Meta Model）を構築します。
    """

    def __init__(
        self,
        model_type: str = "random_forest",
        base_model_names: Optional[List[str]] = None,
    ):
        self.model_type = model_type
        self.model = None
        self.is_trained = False
        self.oof_preds_df = None
        self.base_model_names = base_model_names  # 追加

    def _add_base_model_statistics(
        self, X_meta: pd.DataFrame, base_probs_filtered: pd.DataFrame
    ) -> pd.DataFrame:
        """
        ベースモデルの予測確率から統計量を計算してメタ特徴量に追加

        Args:
            X_meta: メタ特徴量DataFrame
            base_probs_filtered: フィルタリングされたベースモデル予測確率

        Returns:
            統計量が追加されたメタ特徴量DataFrame
        """
        X_meta["base_prob_mean"] = base_probs_filtered.mean(axis=1)
        X_meta["base_prob_std"] = base_probs_filtered.std(axis=1).fillna(0)
        X_meta["base_prob_min"] = base_probs_filtered.min(axis=1)
        X_meta["base_prob_max"] = base_probs_filtered.max(axis=1)
        return X_meta

    def create_meta_labels(
        self, primary_preds_proba: pd.Series, y_true: pd.Series, threshold: float = 0.5
    ) -> Tuple[pd.DataFrame, pd.Series]:
        """
        メタラベルとフィルタリングされた特徴量を作成します。

        Args:
            primary_preds_proba: 一次モデルの予測確率 (Trendである確率)
            y_true: 正解ラベル (1=Trend, 0=Range)
            threshold: 一次モデルがTrendと判定する閾値

        Returns:
            X_meta: メタモデル学習用特徴量 (一次モデルがTrendと判定したサンプルのインデックス)
            y_meta: メタラベル (1=一次モデル正解, 0=一次モデル不正解)
        """
        # 一次モデルが「トレンド」と予測したサンプルのみを対象にする
        trend_mask = primary_preds_proba >= threshold

        # フィルタリングされたインデックス
        indices = primary_preds_proba.index[trend_mask]

        if len(indices) == 0:
            logger.warning("一次モデルがトレンドと予測したサンプルがありません。")
            return pd.DataFrame(), pd.Series()

        # メタラベルの生成
        # 一次モデルがTrend(1)と予測し、かつ正解もTrend(1)なら -> 1 (Execute)
        # 一次モデルがTrend(1)と予測したが、正解はRange(0)なら -> 0 (Pass)
        # ※ trend_maskでフィルタリング済みなので、単純に y_true と一致します。
        y_meta = y_true.loc[indices]

        # メタモデルの特徴量は、ここでは呼び出し元で結合することを想定し、
        # インデックス情報を保持するためにフィルタリングした予測確率を返す
        # 実際には、呼び出し元で元の特徴量Xと結合する必要があります。
        # ここでは便宜上、メタラベル作成ロジックのみを提供します。

        return trend_mask, y_meta

    def train(
        self,
        X_train: pd.DataFrame,
        y_train: pd.Series,
        primary_proba_train: pd.Series,
        base_model_probs_df: pd.DataFrame,  # 追加: 各ベースモデルの予測確率
        threshold: float = 0.5,
    ) -> Dict[str, Any]:
        """
        メタモデルを学習します。

        Args:
            X_train: 学習用特徴量（元の特徴量）
            y_train: 学習用正解ラベル（元のラベル）
            primary_proba_train: 学習用データの一次モデル予測確率
            base_model_probs_df: 各ベースモデルのOOF予測確率DataFrame
            threshold: 一次モデルの閾値

        Returns:
            学習結果メトリクス
        """
        self.oof_preds_df = base_model_probs_df
        self.base_model_names = (
            base_model_probs_df.columns.tolist()
        )  # base_model_namesを保存

        # メタラベルの作成
        trend_mask, y_meta = self.create_meta_labels(
            primary_proba_train, y_train, threshold
        )

        if len(y_meta) < 50:  # 学習データが少なすぎる場合はスキップ
            logger.warning(
                f"メタモデルの学習データが不足しています: {len(y_meta)}サンプル"
            )
            return {"status": "skipped", "reason": "insufficient_data"}

        # メタモデル用の特徴量を作成
        X_meta = X_train.loc[trend_mask].copy()
        X_meta["primary_proba"] = primary_proba_train.loc[trend_mask]

        # 各ベースモデルの予測確率を追加
        X_meta = pd.concat([X_meta, base_model_probs_df.loc[trend_mask]], axis=1)

        # モデル間の合意度・不一致度を示す統計量を追加（共通メソッド使用）
        base_probs_filtered = base_model_probs_df.loc[trend_mask]
        X_meta = self._add_base_model_statistics(X_meta, base_probs_filtered)

        # モデルの初期化と学習
        if self.model_type == "random_forest":
            self.model = RandomForestClassifier(
                n_estimators=100,
                max_depth=5,
                class_weight="balanced",
                random_state=42,
                n_jobs=-1,
            )
        else:
            raise ValueError(f"未サポートのモデルタイプ: {self.model_type}")

        logger.info(
            f"メタモデル学習開始: {len(X_meta)}サンプル, Positive Rate: {y_meta.mean():.2%}"
        )
        self.model.fit(X_meta, y_meta)
        self.is_trained = True

        return {"status": "success", "samples": len(X_meta)}

    def predict(
        self,
        X: pd.DataFrame,
        primary_proba: pd.Series,
        base_model_probs_df: pd.DataFrame,  # 追加: 各ベースモデルの予測確率
        threshold: float = 0.5,
    ) -> pd.Series:
        """
        メタモデルによるフィルタリング予測を行います。

        Args:
            X: 特徴量
            primary_proba: 一次モデルの予測確率
            base_model_probs_df: 各ベースモデルのOOF予測確率DataFrame
            threshold: 一次モデルの閾値

        Returns:
            最終的な予測フラグ (1=Execute, 0=Pass/Range)
            一次モデルがRangeと予測したものは0、
            Trendと予測したもののうちメタモデルがNGと出したものも0になります。
        """
        if not self.is_trained:
            raise RuntimeError("メタモデルが学習されていません")

        # デフォルトは全て0 (Pass)
        final_pred = pd.Series(0, index=X.index)

        # 一次モデルがトレンドと予測した箇所を特定
        trend_mask = primary_proba >= threshold

        if not trend_mask.any():
            return final_pred

        # メタモデル用特徴量
        X_meta = X.loc[trend_mask].copy()
        X_meta["primary_proba"] = primary_proba.loc[trend_mask]

        # 各ベースモデルの予測確率を追加
        X_meta = pd.concat([X_meta, base_model_probs_df.loc[trend_mask]], axis=1)

        # モデル間の合意度・不一致度を示す統計量を追加（共通メソッド使用）
        base_probs_filtered = base_model_probs_df[self.base_model_names].loc[trend_mask]
        X_meta = self._add_base_model_statistics(X_meta, base_probs_filtered)

        # メタモデル予測 (1=Execute, 0=Pass)
        meta_pred = self.model.predict(X_meta)

        # 結果を格納
        final_pred.loc[trend_mask] = meta_pred

        return final_pred

    def evaluate(
        self,
        X_test: pd.DataFrame,
        y_test: pd.Series,
        primary_proba_test: pd.Series,
        base_model_probs_df: pd.DataFrame,  # 追加
        threshold: float = 0.5,
    ) -> Dict[str, float]:
        """
        メタラベリング適用後のパフォーマンスを評価します。
        """
        from .common.evaluation_utils import evaluate_model_predictions

        # メタラベリング適用後の最終予測
        final_pred = self.predict(
            X_test, primary_proba_test, base_model_probs_df, threshold
        )

        # 統一された評価システムを使用してメタモデルの評価
        meta_metrics = evaluate_model_predictions(
            y_true=y_test,
            y_pred=(
                final_pred.values if isinstance(final_pred, pd.Series) else final_pred
            ),
            y_pred_proba=None,  # メタモデルは2値予測のみ
        )

        # 一次モデル（Primary Model）の評価
        primary_pred_bin = (primary_proba_test >= threshold).astype(int)
        primary_metrics = evaluate_model_predictions(
            y_true=y_test,
            y_pred=(
                primary_pred_bin.values
                if isinstance(primary_pred_bin, pd.Series)
                else primary_pred_bin
            ),
            y_pred_proba=None,
        )

        # 結果を整形して返す
        return {
            "meta_accuracy": meta_metrics.get("accuracy", 0.0),
            "meta_precision": meta_metrics.get("precision", 0.0),
            "meta_recall": meta_metrics.get("recall", 0.0),
            "meta_f1": meta_metrics.get("f1", 0.0),
            "primary_accuracy": primary_metrics.get("accuracy", 0.0),
            "primary_precision": primary_metrics.get("precision", 0.0),
            "primary_recall": primary_metrics.get("recall", 0.0),
            "primary_f1": primary_metrics.get("f1", 0.0),
            "improvement_precision": meta_metrics.get("precision", 0.0)
            - primary_metrics.get("precision", 0.0),
            "improvement_recall": meta_metrics.get("recall", 0.0)
            - primary_metrics.get("recall", 0.0),
            "improvement_f1": meta_metrics.get("f1", 0.0)
            - primary_metrics.get("f1", 0.0),
            "meta_classification_report": meta_metrics.get("classification_report", {}),
            "primary_classification_report": primary_metrics.get(
                "classification_report", {}
            ),
            "meta_balanced_accuracy": meta_metrics.get("balanced_accuracy", 0.0),
            "primary_balanced_accuracy": primary_metrics.get("balanced_accuracy", 0.0),
        }
