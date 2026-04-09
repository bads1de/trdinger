"""
PositionSizingGene Factory Function Tests

Test logic for create_random_position_sizing_gene
"""

from unittest.mock import Mock

from app.services.auto_strategy.config import GAConfig
from app.services.auto_strategy.genes import (
    PositionSizingGene,
    PositionSizingMethod,
    create_random_position_sizing_gene,
)


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
        config = object()
        result = create_random_position_sizing_gene(config)
        assert result is not None

    def test_applies_position_sizing_constraints(self):
        """GAConfig の position sizing 制約を生成時に反映する"""
        config = GAConfig(
            position_sizing_method_constraints=["fixed_quantity"],
            position_sizing_fixed_quantity_range=[2.0, 2.5],
            position_sizing_max_size_range=[40.0, 45.0],
            position_sizing_var_confidence_range=[0.9, 0.92],
            position_sizing_var_lookback_range=[80, 90],
        )

        result = create_random_position_sizing_gene(config)

        assert result.method == PositionSizingMethod.FIXED_QUANTITY
        assert 2.0 <= result.fixed_quantity <= 2.5
        assert 40.0 <= result.max_position_size <= 45.0
        assert 0.9 <= result.var_confidence <= 0.92
        assert 80 <= result.var_lookback <= 90
