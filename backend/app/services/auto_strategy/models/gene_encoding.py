"""
遺伝子エンコーディング ファサード

GA用の戦略遺伝子エンコード/デコード機能への統一インターフェースを提供します。
内部的にエンコーダーとデコーダーを呼び出します。
"""

import logging
from typing import Dict, List

from . import gene_utils
from .gene_decoder import GeneDecoder as Decoder
from .gene_encoder import GeneEncoder as Encoder

logger = logging.getLogger(__name__)


class GeneEncoder:
    """
    遺伝子エンコード/デコードのファサードクラス。

    後方互換性を維持しつつ、内部実装を新しいエンコーダー/デコーダークラスに委譲します。
    """

    def __init__(self):
        """初期化"""
        self._encoder = Encoder()
        self._decoder = Decoder()
        # 依存関係を明確にするため、decoderから取得
        self.indicator_ids = self._decoder.indicator_ids
        self.id_to_indicator = self._decoder.id_to_indicator

    def encode_strategy_gene_to_list(self, strategy_gene) -> List[float]:
        """
        戦略遺伝子を固定長の数値リストにエンコードします。
        """
        return self._encoder.encode_strategy_gene_to_list(strategy_gene)

    def decode_list_to_strategy_gene(self, encoded: List[float], strategy_gene_class):
        """
        数値リストから戦略遺伝子にデコードします。
        """
        return self._decoder.decode_list_to_strategy_gene(encoded, strategy_gene_class)

    def get_encoding_info(self) -> Dict:
        """
        エンコーディングに関する基本情報を取得します。
        """
        return gene_utils.get_encoding_info(self.indicator_ids)
