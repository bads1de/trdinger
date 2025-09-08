"""
BaseConfigクラス

設定クラスの基底クラスを提供します。
"""

import json
import logging
from abc import ABC
from dataclasses import dataclass, field, fields
from typing import Any, Dict, List, Tuple


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
            if field_info.default is not field_info.default_factory:  # sentinel以外
                defaults[field_info.name] = field_info.default
            elif field_info.default_factory is not None:
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

    def validate(self) -> Tuple[bool, List[str]]:
        """共通検証ロジック"""
        errors = []

        try:
            # 必須フィールドチェック
            required_fields = self.validation_rules.get("required_fields", [])
            for field_name in required_fields:
                if not hasattr(self, field_name) or getattr(self, field_name) is None:
                    errors.append(f"必須フィールド '{field_name}' が設定されていません")

            # 範囲チェック
            range_rules = self.validation_rules.get("ranges", {})
            for field_name, (min_val, max_val) in range_rules.items():
                if hasattr(self, field_name):
                    value = getattr(self, field_name)
                    if isinstance(value, (int, float)) and not (
                        min_val <= value <= max_val
                    ):
                        errors.append(
                            f"'{field_name}' は {min_val} から {max_val} の範囲で設定してください"
                        )

            # 型チェック
            type_rules = self.validation_rules.get("types", {})
            for field_name, expected_type in type_rules.items():
                if hasattr(self, field_name):
                    value = getattr(self, field_name)
                    if value is not None and not isinstance(value, expected_type):
                        errors.append(
                            f"'{field_name}' は {expected_type.__name__} 型である必要があります"
                        )

            # カスタム検証
            custom_errors = self._custom_validation()
            errors.extend(custom_errors)

        except Exception as e:
            logger.error(f"設定検証中にエラーが発生: {e}", exc_info=True)
            errors.append(f"検証処理エラー: {e}")

        return len(errors) == 0, errors

    def _custom_validation(self) -> List[str]:
        """サブクラスでオーバーライド可能なカスタム検証"""
        return []

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "BaseConfig":
        """辞書から設定オブジェクトを作成"""
        try:
            # まずデフォルトインスタンスを作成
            instance = cls()

            # データで更新
            for key, value in data.items():
                if hasattr(instance, key):
                    setattr(instance, key, value)

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
        except Exception as e:
            logger.error(f"JSON復元エラー: {e}", exc_info=True)
            raise ValueError(f"JSON からの復元に失敗しました: {e}")
