"""
ML設定管理

ML関連のデフォルト設定定数、および設定の永続化、更新、リセット機能を提供します。
unified_config.ml と同期して動作します。
"""

import json
import logging
from datetime import datetime
from pathlib import Path
from typing import Any, Dict


logger = logging.getLogger(__name__)


# --- デフォルト設定 ---


def get_default_ensemble_config() -> Dict[str, Any]:
    """デフォルトのアンサンブル設定を取得"""
    return {
        "enabled": True,
        "method": "stacking",
        "stacking_params": {
            "base_models": ["lightgbm", "xgboost"],
            "meta_model": "lightgbm",
            "cv_folds": 5,
            "use_probas": True,
            "random_state": 42,
        },
    }


def get_default_single_model_config() -> Dict[str, Any]:
    """デフォルトの単一モデル設定を取得"""
    return {"model_type": "lightgbm"}


# --- 設定マネージャー ---


class MLConfigManager:
    """
    ML設定管理クラス

    設定の永続化、更新、リセット、バリデーション機能を提供します。
    unified_config.ml と同期して動作します。
    """

    def __init__(self, config_file_path: str = "config/ml_config.json"):
        """初期化"""
        self.config_file_path = Path(config_file_path)
        if self.config_file_path.exists():
            self.load_config()

    def get_config_dict(self) -> Dict[str, Any]:
        """設定を辞書形式で取得（エイリアス対応）"""
        from app.config.unified_config import unified_config

        return unified_config.ml.model_dump(by_alias=True)

    def save_config(self) -> bool:
        """現在の設定(unified_config.ml)をファイルに保存"""
        try:
            config_dict = self.get_config_dict()
            config_dict["_metadata"] = {
                "saved_at": datetime.now().isoformat(),
                "version": "1.0.0",
                "description": "ML設定ファイル",
            }

            if not self.config_file_path.parent.exists():
                try:
                    self.config_file_path.parent.mkdir(parents=True, exist_ok=True)
                except Exception as e:
                    logger.warning(f"設定ディレクトリの作成に失敗しました: {e}")

            with open(self.config_file_path, "w", encoding="utf-8") as f:
                json.dump(config_dict, f, indent=2, ensure_ascii=False)

            logger.info(f"ML設定を保存しました: {self.config_file_path}")
            return True
        except Exception as e:
            logger.error(f"ML設定の保存に失敗しました: {e}")
            return False

    def load_config(self) -> bool:
        """ファイルから設定を読み込み、unified_config.ml に適用"""
        try:
            if not self.config_file_path.exists():
                logger.warning(f"設定ファイルが存在しません: {self.config_file_path}")
                return False

            with open(self.config_file_path, "r", encoding="utf-8") as f:
                config_dict = json.load(f)

            config_dict.pop("_metadata", None)
            self._apply_config_dict(config_dict)
            logger.info(f"ML設定を読み込み適用しました: {self.config_file_path}")
            return True
        except Exception as e:
            logger.error(f"ML設定の読み込みに失敗しました: {e}")
            return False

    def update_config(self, config_updates: Dict[str, Any]) -> bool:
        """設定を更新"""
        try:
            current_config = self.get_config_dict()
            updated_config = self._merge_config_updates(current_config, config_updates)
            self._apply_config_dict(updated_config)

            if self.save_config():
                logger.info(f"ML設定を更新しました: {list(config_updates.keys())}")
                return True
            else:
                raise Exception("設定の保存に失敗しました")
        except Exception as e:
            logger.error(f"ML設定の更新に失敗しました: {e}")
            return False

    def reset_config(self) -> bool:
        """設定をデフォルト値にリセット"""
        from app.config.unified_config import MLConfig, unified_config

        try:
            unified_config.ml = MLConfig()
            if self.save_config():
                logger.info("ML設定をデフォルト値にリセットしました")
                return True
            else:
                raise Exception("設定の保存に失敗しました")
        except Exception as e:
            logger.error(f"ML設定のリセットに失敗しました: {e}")
            return False

    def _merge_config_updates(
        self, current: Dict[str, Any], updates: Dict[str, Any]
    ) -> Dict[str, Any]:
        """設定更新をマージ"""
        res = current.copy()

        for k, v in updates.items():
            if isinstance(v, dict) and isinstance(res.get(k), dict):
                res[k] = {**res[k], **v}
            else:
                res[k] = v
        return res

    def _apply_config_dict(self, config_dict: Dict[str, Any]) -> None:
        """辞書から設定をunified_config.mlに適用"""
        from app.config.unified_config import MLConfig, unified_config

        unified_config.ml = MLConfig(**config_dict)


# グローバルインスタンス
ml_config_manager = MLConfigManager()
