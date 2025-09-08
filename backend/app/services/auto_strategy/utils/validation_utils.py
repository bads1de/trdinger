"""
バリデーション関連ユーティリティ
"""

import logging
from typing import Any, Dict, List
from typing import Tuple

logger = logging.getLogger(__name__)


class ValidationUtils:
    """バリデーションユーティリティ"""

    @staticmethod
    def validate_range(
        value: float,
        min_val: float,
        max_val: float,
        name: str = "値",
    ) -> bool:
        """範囲バリデーション"""
        if not (min_val <= value <= max_val):
            logger.warning(f"{name}が範囲外です: {value} (範囲: {min_val}-{max_val})")
            return False
        return True

    @staticmethod
    def validate_required_fields(
        data: Dict[str, Any], required_fields: List[str]
    ) -> Tuple[bool, List[str]]:
        """必須フィールドバリデーション"""
        missing_fields = []
        for field in required_fields:
            if field not in data or data[field] is None:
                missing_fields.append(field)

        if missing_fields:
            logger.warning(f"必須フィールドが不足しています: {missing_fields}")
            return False, missing_fields

        return True, []