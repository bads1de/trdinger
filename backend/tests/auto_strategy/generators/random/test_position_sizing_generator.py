"""
PositionSizingGene Factory Function Tests

Test logic for create_random_position_sizing_gene
"""

import pytest
from unittest.mock import Mock
from app.services.auto_strategy.genes import PositionSizingGene, PositionSizingMethod, create_random_position_sizing_gene

class TestCreateRandomPositionSizingGene:
    """create_random_position_sizing_geneのテスト"""

    def test_returns_valid_gene(self):
        """有効な遺伝子を返す"""
        config = Mock()
        result = create_random_position_sizing_gene(config)
        assert isinstance(result, PositionSizingGene)
        assert isinstance(result.method, PositionSizingMethod)
        assert isinstance(result.fixed_ratio, float)
        assert isinstance(result.max_position_size, float)
        assert isinstance(result.enabled, bool)

    def test_passes_config(self):
        """設定オブジェクトを受け取れる"""
        # 現在の実装ではconfigを使っていないが、インターフェースとしては受け取る
        config = Mock()
        result = create_random_position_sizing_gene(config)
        assert result is not None