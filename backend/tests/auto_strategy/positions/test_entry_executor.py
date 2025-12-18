"""
EntryExecutorのユニットテスト
"""

import pytest
from unittest.mock import Mock

from app.services.auto_strategy.positions.entry_executor import EntryExecutor
from app.services.auto_strategy.genes.entry import EntryGene
from app.services.auto_strategy.config.constants import EntryType


class TestEntryExecutor:
    """EntryExecutorのテストクラス"""

    @pytest.fixture
    def executor(self):
        return EntryExecutor()

    def test_calculate_entry_prices_market(self, executor):
        """成行注文の価格計算"""
        gene = EntryGene(entry_type=EntryType.MARKET)
        current_price = 100.0
        
        # 成行の場合は空の辞書
        params = executor.calculate_entry_params(gene, current_price, direction=1.0)
        
        assert params == {}

    def test_calculate_entry_prices_limit_long(self, executor):
        """指値注文（ロング）の価格計算"""
        # 1% 下に指値を置く設定 (pct単位なので 1.0)
        gene = EntryGene(
            entry_type=EntryType.LIMIT,
            limit_offset_pct=1.0
        )
        current_price = 100.0
        
        params = executor.calculate_entry_params(gene, current_price, direction=1.0)
        
        # 100 * (1 - 0.01) = 99.0
        assert params["limit"] == pytest.approx(99.0)

    def test_calculate_entry_prices_limit_short(self, executor):
        """指値注文（ショート）の価格計算"""
        gene = EntryGene(
            entry_type=EntryType.LIMIT,
            limit_offset_pct=1.0
        )
        current_price = 100.0
        
        params = executor.calculate_entry_params(gene, current_price, direction=-1.0)
        
        # 100 * (1 + 0.01) = 101.0
        assert params["limit"] == pytest.approx(101.0)

    def test_calculate_entry_prices_stop_limit_long(self, executor):
        """逆指値指値注文（ロング）の価格計算"""
        gene = EntryGene(
            entry_type=EntryType.STOP_LIMIT,
            stop_offset_pct=2.0,
            limit_offset_pct=1.0
        )
        current_price = 100.0
        
        params = executor.calculate_entry_params(gene, current_price, direction=1.0)
        
        # Stop: 100 * (1 + 0.02) = 102.0
        # Limit: 102 * (1 - 0.01) = 100.98
        assert params["stop"] == pytest.approx(102.0)
        assert params["limit"] == pytest.approx(100.98)