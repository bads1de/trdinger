"""
ML指標サービス（非推奨）

@deprecated: このクラスは非推奨です。MLOrchestratorを使用してください。

機械学習による予測確率を指標として提供するサービス。
MLOrchestratorへのプロキシとして動作し、既存のAPI互換性を保持します。
"""

import logging
import warnings
import pandas as pd
import numpy as np
import os
from typing import Dict, Any, Optional

from .ml_orchestrator import MLOrchestrator

logger = logging.getLogger(__name__)


class MLIndicatorService:
    """
    ML指標サービス（非推奨）

    @deprecated: このクラスは非推奨です。MLOrchestratorを使用してください。

    MLOrchestratorへのプロキシとして動作し、既存のAPI互換性を保持します。
    """

    def __init__(self, trainer=None):
        """
        初期化

        Args:
            trainer: 使用するMLトレーナー（非推奨、無視されます）
        """
        # 非推奨警告を表示
        warnings.warn(
            "MLIndicatorServiceは非推奨です。MLOrchestratorを使用してください。\n"
            "詳細: https://github.com/your-repo/docs/ml_migration_guide.md",
            DeprecationWarning,
            stacklevel=2,
        )

        # MLOrchestratorにプロキシ
        self._orchestrator = MLOrchestrator()

        # 互換性のためのプロパティ
        self.trainer = self._orchestrator.trainer
        self.is_model_loaded = self._orchestrator.is_model_loaded
        self._last_predictions = self._orchestrator._last_predictions

    def calculate_ml_indicators(
        self,
        df: pd.DataFrame,
        funding_rate_data: Optional[pd.DataFrame] = None,
        open_interest_data: Optional[pd.DataFrame] = None,
    ) -> Dict[str, np.ndarray]:
        """
        ML予測確率指標を計算（MLOrchestratorにプロキシ）

        Args:
            df: OHLCV価格データ
            funding_rate_data: ファンディングレートデータ（オプション）
            open_interest_data: 建玉残高データ（オプション）

        Returns:
            ML指標の辞書 {"ML_UP_PROB": array, "ML_DOWN_PROB": array, "ML_RANGE_PROB": array}
        """
        return self._orchestrator.calculate_ml_indicators(
            df, funding_rate_data, open_interest_data
        )

    def load_model(self, model_path: str) -> bool:
        """
        学習済みモデルを読み込み（MLOrchestratorにプロキシ）

        Args:
            model_path: モデルファイルパス

        Returns:
            読み込み成功フラグ
        """
        return self._orchestrator.load_model(model_path)

    def train_model(
        self,
        training_data: pd.DataFrame,
        funding_rate_data: Optional[pd.DataFrame] = None,
        open_interest_data: Optional[pd.DataFrame] = None,
        save_model: bool = True,
        train_test_split: float = 0.8,
        cross_validation_folds: int = 5,
        random_state: int = 42,
        early_stopping_rounds: int = 100,
        max_depth: int = 10,
        n_estimators: int = 100,
        learning_rate: float = 0.1,
    ) -> Dict[str, Any]:
        """
        MLモデルを学習（MLOrchestratorにプロキシ）

        Args:
            training_data: 学習用OHLCVデータ
            funding_rate_data: ファンディングレートデータ（オプション）
            open_interest_data: 建玉残高データ（オプション）
            save_model: モデルを保存するか
            train_test_split: トレーニング/テスト分割比率
            cross_validation_folds: クロスバリデーション分割数
            random_state: ランダムシード
            early_stopping_rounds: 早期停止ラウンド数
            max_depth: 最大深度
            n_estimators: 推定器数
            learning_rate: 学習率

        Returns:
            学習結果
        """
        return self._orchestrator.train_model(
            training_data=training_data,
            funding_rate_data=funding_rate_data,
            open_interest_data=open_interest_data,
            save_model=save_model,
            train_test_split=train_test_split,
            cross_validation_folds=cross_validation_folds,
            random_state=random_state,
            early_stopping_rounds=early_stopping_rounds,
            max_depth=max_depth,
            n_estimators=n_estimators,
            learning_rate=learning_rate,
        )

    def get_model_status(self) -> Dict[str, Any]:
        """
        モデルの状態を取得（MLOrchestratorにプロキシ）

        Returns:
            モデル状態の辞書
        """
        return self._orchestrator.get_model_status()

    def update_predictions(self, predictions: Dict[str, float]):
        """
        予測値を更新（MLOrchestratorにプロキシ）

        Args:
            predictions: 予測確率の辞書
        """
        self._orchestrator.update_predictions(predictions)

    def get_feature_importance(self, top_n: int = 10) -> Dict[str, float]:
        """
        特徴量重要度を取得（MLOrchestratorにプロキシ）

        Args:
            top_n: 上位N個の特徴量

        Returns:
            特徴量重要度の辞書
        """
        return self._orchestrator.get_feature_importance(top_n)

    def calculate_single_ml_indicator(
        self, indicator_type: str, df: pd.DataFrame
    ) -> np.ndarray:
        """
        単一のML指標を計算（MLOrchestratorにプロキシ）

        Args:
            indicator_type: 指標タイプ（ML_UP_PROB, ML_DOWN_PROB, ML_RANGE_PROB）
            df: OHLCVデータ

        Returns:
            指標値の配列
        """
        return self._orchestrator.calculate_single_ml_indicator(indicator_type, df)

    # 以下のメソッドは非推奨（MLOrchestratorで実装済み）
    def _get_default_indicators(self, data_length: int) -> Dict[str, np.ndarray]:
        """デフォルトのML指標を取得（非推奨）"""
        warnings.warn(
            "このメソッドは非推奨です。MLOrchestratorを使用してください。",
            DeprecationWarning,
        )
        return self._orchestrator.config.prediction.get_default_indicators(data_length)

    def _safe_ml_prediction(self, features_df: pd.DataFrame) -> Dict[str, float]:
        """安全なML予測実行（非推奨）"""
        warnings.warn(
            "このメソッドは非推奨です。MLOrchestratorを使用してください。",
            DeprecationWarning,
        )
        return self._orchestrator.predict_probabilities(features_df)

    def _validate_predictions(self, predictions: Dict[str, float]) -> bool:
        """予測値の妥当性を検証（非推奨）"""

        warnings.warn(
            "このメソッドは非推奨です。MLOrchestratorを使用してください。",
            DeprecationWarning,
        )
        return self._orchestrator.config.prediction.validate_predictions(predictions)

    def _expand_predictions_to_data_length(
        self, predictions: Dict[str, float], data_length: int
    ) -> Dict[str, np.ndarray]:
        """予測値をデータ長に拡張（非推奨）"""
        warnings.warn(
            "このメソッドは非推奨です。MLOrchestratorを使用してください。",
            DeprecationWarning,
        )
        # MLOrchestratorの同等機能を使用
        return {
            "ML_UP_PROB": np.full(data_length, predictions["up"]),
            "ML_DOWN_PROB": np.full(data_length, predictions["down"]),
            "ML_RANGE_PROB": np.full(data_length, predictions["range"]),
        }

    def _preprocess_training_data(
        self,
        ohlcv_data: pd.DataFrame,
        funding_rate_data: Optional[pd.DataFrame] = None,
        open_interest_data: Optional[pd.DataFrame] = None,
    ) -> pd.DataFrame:
        """
        トレーニングデータの前処理（非推奨）

        Args:
            ohlcv_data: OHLCVデータ
            funding_rate_data: ファンディングレートデータ
            open_interest_data: オープンインタレストデータ

        Returns:
            前処理済みデータ
        """
        warnings.warn(
            "このメソッドは非推奨です。MLOrchestratorを使用してください。",
            DeprecationWarning,
        )
        # 基本的な前処理のみ実装（詳細はMLOrchestratorで）
        processed = ohlcv_data.copy()
        processed.columns = [col.lower() for col in processed.columns]
        return processed

    def _create_features_and_targets(self, data: pd.DataFrame) -> tuple:
        """
        特徴量とターゲットを作成（非推奨）

        Args:
            data: 前処理済みデータ

        Returns:
            (特徴量DataFrame, ターゲットSeries)
        """
        warnings.warn(
            "このメソッドは非推奨です。MLOrchestratorのFeatureEngineeringServiceを使用してください。",
            DeprecationWarning,
        )

        # 簡略化された特徴量計算（詳細はFeatureEngineeringServiceで）
        features = pd.DataFrame(index=data.index)
        features["price_change"] = data["close"].pct_change()
        features["volume_change"] = data["volume"].pct_change()

        # 欠損値を除去
        features = features.dropna()

        # ダミーターゲット（実際の学習ではMLOrchestratorを使用）
        targets = pd.Series(0, index=features.index)

        return features, targets

    def _save_model(self) -> str:
        """
        モデルを保存（非推奨）

        Returns:
            保存されたモデルのパス
        """
        warnings.warn(
            "このメソッドは非推奨です。MLOrchestratorのsave_modelを使用してください。",
            DeprecationWarning,
        )
        return self._orchestrator.save_model("deprecated_ml_indicator_model")

    def _validate_ml_indicators(self, ml_indicators: Dict[str, np.ndarray]) -> bool:
        """ML指標の妥当性を検証（非推奨）"""
        warnings.warn(
            "このメソッドは非推奨です。MLOrchestratorを使用してください。",
            DeprecationWarning,
        )
        # 基本的な検証のみ実装
        try:
            required_indicators = ["ML_UP_PROB", "ML_DOWN_PROB", "ML_RANGE_PROB"]
            return all(indicator in ml_indicators for indicator in required_indicators)
        except Exception:
            return False

    def _try_load_latest_model(self) -> bool:
        """
        最新の学習済みモデルを自動読み込み（非推奨）

        Returns:
            読み込み成功フラグ
        """
        warnings.warn(
            "このメソッドは非推奨です。MLOrchestratorを使用してください。",
            DeprecationWarning,
        )
        # MLOrchestratorの自動読み込み機能を使用
        return self._orchestrator.is_model_loaded
