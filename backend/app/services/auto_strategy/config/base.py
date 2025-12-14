"""
BaseConfigクラス

設定クラスの基底クラスを提供します。
"""

import json
import logging
from abc import ABC
from dataclasses import MISSING, dataclass, field, fields
from typing import Any, Dict

logger = logging.getLogger(__name__)


@dataclass
class BaseConfig(ABC):
    """設定クラスの基底クラス"""

    enabled: bool = True
    validation_rules: Dict[str, Any] = field(default_factory=dict)

    def get_default_values(self) -> Dict[str, Any]:
        """デフォルト値を取得"""
        return self.get_default_values_from_fields()

    @classmethod
    def get_default_values_from_fields(cls) -> Dict[str, Any]:
        """
        フィールド定義に基づいてデフォルト値を自動生成

        このメソッドはリフレクションを使って各フィールドのデフォルト値を収集します。
        複雑なデフォルト値（field(default_factory=...)）も処理できます。
        """
        defaults = {}
        for field_info in fields(cls):
            if field_info.default is not MISSING:  # _MISSING以外
                defaults[field_info.name] = field_info.default
            if field_info.default_factory is not MISSING:
                try:
                    # default_factoryがcallableの場合呼び出し
                    if callable(field_info.default_factory):
                        defaults[field_info.name] = field_info.default_factory()
                    else:
                        defaults[field_info.name] = field_info.default_factory
                except Exception as e:
                    logger.warning(f"デフォルト値生成失敗: {field_info.name}, {e}")
                    defaults[field_info.name] = None
        return defaults

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "BaseConfig":
        """辞書から設定オブジェクトを作成"""
        try:
            # まずデフォルトインスタンスを作成
            instance = cls()

            # データで更新
            for key, value in data.items():
                if hasattr(instance, key):
                    try:
                        setattr(instance, key, value)
                    except Exception as e:
                        logger.warning(f"Field設定エラー: {key} = {value}, {e}")

            return instance
        except Exception as e:
            logger.error(f"設定オブジェクト作成エラー: {e}", exc_info=True)
            raise ValueError(f"設定の作成に失敗しました: {e}")

    def to_dict(self) -> Dict[str, Any]:
        """設定オブジェクトを辞書に変換"""
        try:
            result = {}
            for field_info in fields(self):
                value = getattr(self, field_info.name)
                # 複雑なオブジェクトの場合は文字列化
                if hasattr(value, "__dict__"):
                    result[field_info.name] = str(value)
                else:
                    result[field_info.name] = value
            return result
        except Exception as e:
            logger.error(f"設定辞書変換エラー: {e}", exc_info=True)
            return {}

    def to_json(self) -> str:
        """設定オブジェクトをJSON文字列に変換"""
        try:
            return json.dumps(self.to_dict(), ensure_ascii=False, indent=2)
        except Exception as e:
            logger.error(f"JSON変換エラー: {e}", exc_info=True)
            return "{}"

    @classmethod
    def from_json(cls, json_str: str) -> "BaseConfig":
        """JSON文字列から設定オブジェクトを復元"""
        try:
            data = json.loads(json_str)
            return cls.from_dict(data)
        except json.JSONDecodeError as e:
            logger.error(f"JSON復元エラー: {e}", exc_info=True)
            return cls()
        except Exception as e:
            logger.error(f"JSON復元エラー: {e}", exc_info=True)
            raise ValueError(f"JSON からの復元に失敗しました: {e}")
