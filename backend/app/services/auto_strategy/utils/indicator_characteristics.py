"""
指標特性データベース
定数定義から分離された動的指標特性管理モジュール
"""


def _get_merged_characteristics(original):
    # Late import to avoid circular imports
    from app.services.auto_strategy.utils.yaml_utils import YamlIndicatorUtils

    return YamlIndicatorUtils.initialize_yaml_based_characteristics(original)


# YAML設定に基づいて特性を生成
INDICATOR_CHARACTERISTICS = _get_merged_characteristics({})
