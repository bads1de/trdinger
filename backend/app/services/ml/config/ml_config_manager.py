"""
ML設定管理サービス

ML設定の永続化、更新、リセット機能を提供します。
設定はJSONファイルに保存され、アプリケーション起動時に読み込まれます。
"""

import json
import logging
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List

from app.services.ml.config.ml_config import MLConfig
from app.utils.unified_error_handler import UnifiedValidationError

logger = logging.getLogger(__name__)


class MLConfigManager:
    """
    ML設定管理クラス

    設定の永続化、更新、リセット、バリデーション機能を提供します。
    """

    def __init__(self, config_file_path: str = "config/ml_config.json"):
        """
        初期化

        Args:
            config_file_path: 設定ファイルのパス
        """
        self.config_file_path = Path(config_file_path)
        self._ml_config = MLConfig()

        # ディレクトリの自動作成は行わない（勝手にconfigディレクトリを作成しない方針）
        # 必要に応じて呼び出し側でディレクトリを用意すること

        # 設定ファイルが存在する場合は読み込み
        if self.config_file_path.exists():
            self.load_config()

    def get_config_dict(self) -> Dict[str, Any]:
        """
        設定を辞書形式で取得（Pydantic自動シリアライゼーション使用）

        手動マッピングを削除し、Pydanticの model_dump() を活用して
        保守性を向上させました。
        """
        return {
            "data_processing": self._ml_config.data_processing.model_dump(),
            "model": self._ml_config.model.model_dump(),
            "training": self._ml_config.training.model_dump(),
            "prediction": self._ml_config.prediction.model_dump(),
            "ensemble": self._ml_config.ensemble.model_dump(),
            "retraining": self._ml_config.retraining.model_dump(),
        }

    def save_config(self) -> bool:
        """
        現在の設定をファイルに保存

        Returns:
            保存成功フラグ
        """
        try:
            # 設定を辞書形式に変換
            config_dict = self.get_config_dict()

            # メタデータを追加
            config_dict["_metadata"] = {
                "saved_at": datetime.now().isoformat(),
                "version": "1.0.0",
                "description": "ML設定ファイル",
            }

            # ファイルに保存
            with open(self.config_file_path, "w", encoding="utf-8") as f:
                json.dump(config_dict, f, indent=2, ensure_ascii=False)

            logger.info(f"ML設定を保存しました: {self.config_file_path}")
            return True

        except Exception as e:
            logger.error(f"ML設定の保存に失敗しました: {e}")
            return False

    def load_config(self) -> bool:
        """
        ファイルから設定を読み込み

        Returns:
            読み込み成功フラグ
        """
        try:
            if not self.config_file_path.exists():
                logger.warning(f"設定ファイルが存在しません: {self.config_file_path}")
                return False

            with open(self.config_file_path, "r", encoding="utf-8") as f:
                config_dict = json.load(f)

            # メタデータを除去
            config_dict.pop("_metadata", None)

            # 設定を適用
            self._apply_config_dict(config_dict)

            logger.info(f"ML設定を読み込みました: {self.config_file_path}")
            return True

        except Exception as e:
            logger.error(f"ML設定の読み込みに失敗しました: {e}")
            return False

    def update_config(self, config_updates: Dict[str, Any]) -> bool:
        """
        設定を更新

        Args:
            config_updates: 更新する設定項目

        Returns:
            更新成功フラグ
        """
        try:
            # バリデーション
            validation_errors = self.validate_config_updates(config_updates)
            if validation_errors:
                raise UnifiedValidationError(
                    f"設定バリデーションエラー: {validation_errors}"
                )

            # 現在の設定を取得
            current_config = self._ml_config.to_dict()

            # 設定を更新
            updated_config = self._merge_config_updates(current_config, config_updates)

            # 更新された設定を適用
            self._ml_config = MLConfig.from_dict(updated_config)

            # ファイルに保存
            if self.save_config():
                logger.info(f"ML設定を更新しました: {list(config_updates.keys())}")
                return True
            else:
                raise Exception("設定の保存に失敗しました")

        except Exception as e:
            logger.error(f"ML設定の更新に失敗しました: {e}")
            return False

    def reset_config(self) -> bool:
        """
        設定をデフォルト値にリセット

        Returns:
            リセット成功フラグ
        """
        try:
            # デフォルト設定を作成
            self._ml_config = MLConfig()

            # ファイルに保存
            if self.save_config():
                logger.info("ML設定をデフォルト値にリセットしました")
                return True
            else:
                raise Exception("設定の保存に失敗しました")

        except Exception as e:
            logger.error(f"ML設定のリセットに失敗しました: {e}")
            return False

    def validate_config_updates(self, config_updates: Dict[str, Any]) -> List[str]:
        """
        設定更新のバリデーション

        Args:
            config_updates: 更新する設定項目

        Returns:
            バリデーションエラーのリスト
        """
        errors = []

        try:
            # データ処理設定のバリデーション
            if "data_processing" in config_updates:
                errors.extend(
                    self._validate_data_processing_config(
                        config_updates["data_processing"]
                    )
                )

            # モデル設定のバリデーション
            if "model" in config_updates:
                errors.extend(self._validate_model_config(config_updates["model"]))

            # 学習設定のバリデーション
            if "training" in config_updates:
                errors.extend(
                    self._validate_training_config(config_updates["training"])
                )

            # 予測設定のバリデーション
            if "prediction" in config_updates:
                errors.extend(
                    self._validate_prediction_config(config_updates["prediction"])
                )

            # アンサンブル設定のバリデーション
            if "ensemble" in config_updates:
                errors.extend(
                    self._validate_ensemble_config(config_updates["ensemble"])
                )

            # 再学習設定のバリデーション
            if "retraining" in config_updates:
                errors.extend(
                    self._validate_retraining_config(config_updates["retraining"])
                )

        except Exception as e:
            errors.append(f"バリデーション中にエラーが発生しました: {e}")

        return errors

    def _validate_data_processing_config(self, config: Dict[str, Any]) -> List[str]:
        """データ処理設定のバリデーション"""
        errors = []

        # 行数制限の検証
        for key in ["max_ohlcv_rows", "max_feature_rows"]:
            if key in config:
                value = config[key]
                if not isinstance(value, int) or value <= 0:
                    errors.append(f"{key} は正の整数である必要があります")
                elif value > 10000000:  # 1000万行制限
                    errors.append(f"{key} は1000万行以下である必要があります")

        # タイムアウト設定の検証
        for key in [
            "feature_calculation_timeout",
            "model_training_timeout",
            "model_prediction_timeout",
        ]:
            if key in config:
                value = config[key]
                if not isinstance(value, int) or value <= 0:
                    errors.append(f"{key} は正の整数である必要があります")
                elif value > 86400:  # 24時間制限
                    errors.append(f"{key} は24時間（86400秒）以下である必要があります")

        # メモリ閾値の検証
        for key in ["memory_warning_threshold", "memory_limit_threshold"]:
            if key in config:
                value = config[key]
                if not isinstance(value, int) or value <= 0:
                    errors.append(f"{key} は正の整数である必要があります")

        # ログレベルの検証
        if "log_level" in config:
            valid_levels = ["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"]
            if config["log_level"] not in valid_levels:
                errors.append(
                    f"log_level は {valid_levels} のいずれかである必要があります"
                )

        return errors

    def _validate_model_config(self, config: Dict[str, Any]) -> List[str]:
        """モデル設定のバリデーション"""
        errors = []

        # モデル保存パスの検証
        if "model_save_path" in config:
            path = config["model_save_path"]
            if not isinstance(path, str) or not path.strip():
                errors.append("model_save_path は空でない文字列である必要があります")

        # モデルバージョン数の検証
        if "max_model_versions" in config:
            value = config["max_model_versions"]
            if not isinstance(value, int) or value <= 0:
                errors.append("max_model_versions は正の整数である必要があります")
            elif value > 100:
                errors.append("max_model_versions は100以下である必要があります")

        # 保持日数の検証
        if "model_retention_days" in config:
            value = config["model_retention_days"]
            if not isinstance(value, int) or value <= 0:
                errors.append("model_retention_days は正の整数である必要があります")
            elif value > 365:
                errors.append("model_retention_days は365日以下である必要があります")

        return errors

    def _validate_training_config(self, config: Dict[str, Any]) -> List[str]:
        """学習設定のバリデーション"""
        errors = []

        # 分割比率の検証
        if "train_test_split" in config:
            value = config["train_test_split"]
            if not isinstance(value, (int, float)) or not (0.0 < value < 1.0):
                errors.append(
                    "train_test_split は0.0から1.0の間の数値である必要があります"
                )

        # クロスバリデーション分割数の検証
        if "cross_validation_folds" in config:
            value = config["cross_validation_folds"]
            if not isinstance(value, int) or value < 2:
                errors.append(
                    "cross_validation_folds は2以上の整数である必要があります"
                )
            elif value > 20:
                errors.append("cross_validation_folds は20以下である必要があります")

        # 予測期間の検証
        if "prediction_horizon" in config:
            value = config["prediction_horizon"]
            if not isinstance(value, int) or value <= 0:
                errors.append("prediction_horizon は正の整数である必要があります")
            elif value > 168:  # 1週間制限
                errors.append("prediction_horizon は168時間以下である必要があります")

        # 閾値の検証
        if "threshold_up" in config:
            value = config["threshold_up"]
            if not isinstance(value, (int, float)) or value <= 0:
                errors.append("threshold_up は正の数値である必要があります")

        if "threshold_down" in config:
            value = config["threshold_down"]
            if not isinstance(value, (int, float)) or value >= 0:
                errors.append("threshold_down は負の数値である必要があります")

        return errors

    def _validate_prediction_config(self, config: Dict[str, Any]) -> List[str]:
        """予測設定のバリデーション"""
        errors = []

        # 確率値の検証
        prob_keys = [
            "default_up_prob",
            "default_down_prob",
            "default_range_prob",
            "fallback_up_prob",
            "fallback_down_prob",
            "fallback_range_prob",
            "min_probability",
            "max_probability",
        ]

        for key in prob_keys:
            if key in config:
                value = config[key]
                if not isinstance(value, (int, float)) or not (0.0 <= value <= 1.0):
                    errors.append(f"{key} は0.0から1.0の間の数値である必要があります")

        # 確率合計範囲の検証
        for key in ["probability_sum_min", "probability_sum_max"]:
            if key in config:
                value = config[key]
                if not isinstance(value, (int, float)) or value <= 0:
                    errors.append(f"{key} は正の数値である必要があります")

        # 確率合計の整合性チェック
        if "probability_sum_min" in config and "probability_sum_max" in config:
            min_val = config["probability_sum_min"]
            max_val = config["probability_sum_max"]
            if min_val >= max_val:
                errors.append(
                    "probability_sum_min は probability_sum_max より小さい必要があります"
                )

        return errors

    def _validate_ensemble_config(self, config: Dict[str, Any]) -> List[str]:
        """アンサンブル設定のバリデーション"""
        errors = []

        # アルゴリズムリストの検証
        if "algorithms" in config:
            algorithms = config["algorithms"]
            if not isinstance(algorithms, list) or not algorithms:
                errors.append("algorithms は空でないリストである必要があります")
            else:
                valid_algorithms = ["lightgbm", "xgboost", "catboost", "tabnet", "knn"]
                for algo in algorithms:
                    if algo not in valid_algorithms:
                        errors.append(
                            f"無効なアルゴリズム: {algo}. 有効な値: {valid_algorithms}"
                        )

        # 投票方法の検証
        if "voting_method" in config:
            valid_methods = ["soft", "hard"]
            if config["voting_method"] not in valid_methods:
                errors.append(
                    f"voting_method は {valid_methods} のいずれかである必要があります"
                )

        # スタッキング設定の検証
        if "stacking_cv_folds" in config:
            value = config["stacking_cv_folds"]
            if not isinstance(value, int) or value < 2:
                errors.append("stacking_cv_folds は2以上の整数である必要があります")
            elif value > 10:
                errors.append("stacking_cv_folds は10以下である必要があります")

        return errors

    def _validate_retraining_config(self, config: Dict[str, Any]) -> List[str]:
        """再学習設定のバリデーション"""
        errors = []

        # 間隔設定の検証
        if "check_interval_seconds" in config:
            value = config["check_interval_seconds"]
            if not isinstance(value, int) or value <= 0:
                errors.append("check_interval_seconds は正の整数である必要があります")
            elif value < 60:  # 最小1分
                errors.append("check_interval_seconds は60秒以上である必要があります")

        # 並行ジョブ数の検証
        if "max_concurrent_jobs" in config:
            value = config["max_concurrent_jobs"]
            if not isinstance(value, int) or value <= 0:
                errors.append("max_concurrent_jobs は正の整数である必要があります")
            elif value > 10:
                errors.append("max_concurrent_jobs は10以下である必要があります")

        # 閾値の検証
        for key in ["performance_degradation_threshold", "data_drift_threshold"]:
            if key in config:
                value = config[key]
                if not isinstance(value, (int, float)) or not (0.0 <= value <= 1.0):
                    errors.append(f"{key} は0.0から1.0の間の数値である必要があります")

        return errors

    def _merge_config_updates(
        self, current_config: Dict[str, Any], updates: Dict[str, Any]
    ) -> Dict[str, Any]:
        """設定更新をマージ"""
        merged = current_config.copy()

        for section, section_updates in updates.items():
            if section in merged and isinstance(section_updates, dict):
                merged[section].update(section_updates)
            else:
                merged[section] = section_updates

        return merged


# グローバルインスタンス
ml_config_manager = MLConfigManager()
