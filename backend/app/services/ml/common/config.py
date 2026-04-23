"""
ML設定管理

ML関連のデフォルト設定定数、および設定の永続化、更新、リセット機能を提供します。
設定の正本はこのモジュール内で保持します。
"""

import json
import logging
from datetime import datetime
from pathlib import Path
from typing import Any, Dict

from .ml_config import MLConfig

logger = logging.getLogger(__name__)


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


class MLConfigManager:
    """
    MLシステム全体の設定（ディレクトリパス、学習パラメータ、モデル構成等）を一括管理し、永続化を担当するクラスです。

    主な責務:
    - 設定の正本管理: アプリケーション実行中の最新設定を `MLConfig` オブジェクトとして保持します。
    - 永続化: 設定を JSON ファイルとして保存・読み込みし、再起動後も設定を維持します。
    - 動的更新: 実行時に API 等を通じて設定を安全に更新し、即座にシステムへ反映させます。
    - 安全なマージ: 部分的な設定更新（ネストされた辞書）を既存の設定と正しく統合します。
    """

    def __init__(self, config_file_path: str = "config/ml_config.json"):
        """
        設定マネージャーを初期化します。

        Args:
            config_file_path (str): 設定ファイルの保存先パス。
        """
        self.config_file_path = Path(config_file_path)
        self._config = MLConfig()
        if self.config_file_path.exists():
            self.load_config()

    @property
    def config(self) -> MLConfig:
        """現在の ML 設定を返します。"""
        return self._config

    @config.setter
    def config(self, value: MLConfig) -> None:
        """現在の ML 設定を差し替えます。"""
        self._config = value

    def get_config_dict(self) -> Dict[str, Any]:
        """設定を辞書形式で取得（エイリアス対応）"""
        return self.config.model_dump(by_alias=True)

    def save_config(self) -> bool:
        """
        現在のメモリ上の設定をファイルに保存します。

        ディレクトリが存在しない場合は自動的に作成し、
        保存日時やバージョン情報を含むメタデータを付与して JSON 形式で書き出します。

        Returns:
            bool: 保存に成功した場合は True。
        """
        try:
            config_dict = self.get_config_dict()
            config_dict["_metadata"] = {
                "saved_at": datetime.now().isoformat(),
                "version": "1.0.0",
                "description": "ML設定ファイル",
            }

            if not self.config_file_path.parent.exists():
                try:
                    self.config_file_path.parent.mkdir(
                        parents=True, exist_ok=True
                    )
                except Exception as e:
                    logger.warning(
                        f"設定ディレクトリの作成に失敗しました: {e}"
                    )

            with open(self.config_file_path, "w", encoding="utf-8") as f:
                json.dump(config_dict, f, indent=2, ensure_ascii=False)

            logger.info(f"ML設定を保存しました: {self.config_file_path}")
            return True
        except Exception as e:
            logger.error(f"ML設定の保存に失敗しました: {e}")
            return False

    def load_config(self) -> bool:
        """ファイルから設定を読み込み、現在の設定に適用"""
        try:
            if not self.config_file_path.exists():
                logger.warning(
                    f"設定ファイルが存在しません: {self.config_file_path}"
                )
                return False

            with open(self.config_file_path, "r", encoding="utf-8") as f:
                config_dict = json.load(f)

            config_dict.pop("_metadata", None)
            self._apply_config_dict(config_dict)
            logger.info(
                f"ML設定を読み込み適用しました: {self.config_file_path}"
            )
            return True
        except Exception as e:
            logger.error(f"ML設定の読み込みに失敗しました: {e}")
            return False

    def update_config(self, config_updates: Dict[str, Any]) -> bool:
        """
        既存の設定を指定された更新内容で上書きし、ファイルに保存します。

        辞書構造を再帰的にマージするため、一部のパラメータのみを変更することが可能です。

        Args:
            config_updates (Dict[str, Any]): 更新したい項目の辞書。

        Returns:
            bool: 更新と保存に成功した場合は True。
        """
        try:
            current_config = self.get_config_dict()
            updated_config = self._merge_config_updates(
                current_config, config_updates
            )
            self._apply_config_dict(updated_config)

            if self.save_config():
                logger.info(
                    f"ML設定を更新しました: {list(config_updates.keys())}"
                )
                return True
            else:
                logger.error("ML設定の保存に失敗しました")
                return False
        except Exception as e:
            logger.error(f"ML設定の更新に失敗しました: {e}")
            return False

    def reset_config(self) -> bool:
        """設定をデフォルト値にリセット"""
        try:
            self.config = MLConfig()
            if self.save_config():
                logger.info("ML設定をデフォルト値にリセットしました")
                return True
            else:
                logger.error("ML設定の保存に失敗しました")
                return False
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
        """辞書から設定を現在の設定に適用"""
        self.config = MLConfig(**config_dict)


# グローバルインスタンス
ml_config_manager = MLConfigManager()
