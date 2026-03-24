"""
評価関数ラッパーモジュール

並列処理で使用する評価関数のラッパークラスを提供します。
"""

import logging
from typing import Any

from ..config.ga import GAConfig
from .individual_evaluator import IndividualEvaluator

logger = logging.getLogger(__name__)


class EvaluatorWrapper:
    """
    評価関数のラッパー（Pickle化対応）

    並列処理（ProcessPoolExecutor）で個体評価を行う際に、
    評価器と設定を一緒に配信するためのクラスです。
    """

    def __init__(self, evaluator: IndividualEvaluator, config: GAConfig):
        """
        初期化

        Args:
            evaluator: 個体評価器（IndividualEvaluatorインスタンス）
            config: GA設定
        """
        self.evaluator = evaluator
        self.config = config

    def __call__(self, individual: Any) -> tuple:
        """
        評価実行

        Args:
            individual: 評価対象の個体

        Returns:
            フィットネス値のタプル
        """
        return self.evaluator.evaluate(individual, self.config)
