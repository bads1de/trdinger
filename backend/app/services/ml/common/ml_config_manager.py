"""
ML設定管理サービス

ML設定の永続化、更新、リセット機能を提供します。
設定はJSONファイルに保存され、アプリケーション起動時に読み込まれます。
読み込まれた設定は unified_config.ml に反映され、システム全体で共有されます。
"""

import json
import logging
from datetime import datetime
from pathlib import Path
from typing import Any, Dict

from app.config.unified_config import MLConfig, unified_config

logger = logging.getLogger(__name__)


class MLConfigManager:
    """
    ML設定管理クラス

    設定の永続化、更新、リセット、バリデーション機能を提供します。
    unified_config.ml と同期して動作します。
    """

    def __init__(self, config_file_path: str = "config/ml_config.json"):
        """
        初期化

        Args:
            config_file_path: 設定ファイルのパス
        """
        self.config_file_path = Path(config_file_path)

        # ディレクトリの自動作成は行わない（勝手にconfigディレクトリを作成しない方針）
        # 必要に応じて呼び出し側でディレクトリを用意すること

        # 設定ファイルが存在する場合は読み込み、unified_configに反映
        if self.config_file_path.exists():
            self.load_config()

    def get_config_dict(self) -> Dict[str, Any]:
        """
        設定を辞書形式で取得（Pydantic自動シリアライゼーション使用）
        unified_config.ml から最新の状態を取得します。
        """
        # model_dump() を使用して辞書化
        # by_alias=True により、エイリアス（大文字）で出力してAPI互換性を維持
        return unified_config.ml.model_dump(by_alias=True)

    def save_config(self) -> bool:
        """
        現在の設定(unified_config.ml)をファイルに保存

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

            # ディレクトリが存在しない場合は作成を試みる（保存時のみ）
            if not self.config_file_path.parent.exists():
                try:
                    self.config_file_path.parent.mkdir(parents=True, exist_ok=True)
                except Exception as e:
                    logger.warning(f"設定ディレクトリの作成に失敗しました: {e}")
                    # 続行して open でエラーになるかもしれないが、一応試みる

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
        ファイルから設定を読み込み、unified_config.ml に適用

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

            logger.info(
                f"ML設定を読み込み、unified_configに適用しました: {self.config_file_path}"
            )
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
            # 現在の設定を取得
            current_config = self.get_config_dict()

            # 設定を更新
            updated_config = self._merge_config_updates(current_config, config_updates)

            # 更新された設定を適用（Pydanticのバリデーションが自動実行される）
            self._apply_config_dict(updated_config)

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
            # デフォルト設定（新しいインスタンス）をunified_configにセット
            unified_config.ml = MLConfig()

            # ファイルに保存
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
        """
        辞書から設定をunified_config.mlに適用

        Args:
            config_dict: 適用する設定辞書
        """
        # Pydanticモデルの再構築を行い、シングルトンインスタンスの属性を更新
        unified_config.ml = MLConfig(**config_dict)


# グローバルインスタンス
ml_config_manager = MLConfigManager()



