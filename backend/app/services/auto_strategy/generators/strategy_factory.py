"""
戦略ファクトリー

StrategyGeneから動的にbacktesting.py互換のStrategy継承クラスを生成します。
"""

import logging
from typing import Tuple, Type

from backtesting import Strategy

from ..models import StrategyGene

logger = logging.getLogger(__name__)


class StrategyFactory:
    """
    戦略ファクトリー

    StrategyGeneからUniversalStrategyクラスを返します。
    """

    def __init__(self):
        """初期化"""

    def create_strategy_class(self, gene: StrategyGene) -> Type[Strategy]:
        """
        遺伝子から実行可能な戦略クラスを取得

        Args:
            gene: 戦略遺伝子

        Returns:
            backtesting.py互換のStrategy継承クラス（UniversalStrategy）

        Raises:
            ValueError: 遺伝子が無効な場合
        """

        # 遺伝子の妥当性検証
        is_valid, errors = gene.validate()
        if not is_valid:
            raise ValueError(f"Invalid strategy gene: {', '.join(errors)}")

        # 汎用戦略クラスを返す
        # backtesting.pyのBacktestクラスは、Strategyクラスを受け取り、
        # 実行時にそのparamsを渡してインスタンス化する。
        # 呼び出し元がparams={'strategy_gene': gene}として渡す必要がある。
        from ..strategies.universal_strategy import UniversalStrategy

        # 遺伝子IDに応じたクラス名を動的に設定（ログの可読性のため）
        # ただし、クラス自体はユニバーサルなものを使い回す
        short_id = str(gene.id).split("-")[0] if gene.id else "Unknown"
        # 注意: UniversalStrategy.__name__ を書き換えると、
        # 並列実行時に問題が起きる可能性があるため、ここではログ出力のみに留めるか、
        # 派生クラスを作るのではなく、パラメータで識別する。
        logger.debug(f"戦略クラス準備完了: UniversalStrategy for Gene {short_id}")

        return UniversalStrategy

    def validate_gene(self, gene: StrategyGene) -> Tuple[bool, list]:
        """
        戦略遺伝子の妥当性を検証

        Args:
            gene: 検証する戦略遺伝子

        Returns:
            (is_valid, error_messages)
        """
        try:
            return gene.validate()
        except Exception as e:
            logger.error(f"遺伝子検証エラー: {e}")
            return False, [f"検証エラー: {str(e)}"]
