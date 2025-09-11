"""
データ処理モジュールの統合インターフェース

transformers, pipelines, validatorsモジュールを統合した高レベルAPIを提供。
"""

from .data_processor import DataProcessor, data_processor

# 後方互換性のために既存の関数もエクスポート
from . import transformers
from . import pipelines
from . import validators

__all__ = [
    'DataProcessor',
    'data_processor',
    'transformers',
    'pipelines',
    'validators'
]