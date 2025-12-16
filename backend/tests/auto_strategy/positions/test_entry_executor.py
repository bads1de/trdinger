"""
EntryExecutor サービスのテスト

エントリー注文パラメータ計算のユニットテスト
"""

import pytest
from app.services.auto_strategy.genes.entry import EntryGene
from app.services.auto_strategy.config.constants import EntryType
from app.services.auto_strategy.positions.entry_executor import EntryExecutor


class TestEntryExecutor:
    """EntryExecutor のテストクラス"""

    @pytest.fixture
    def executor(self):
        """EntryExecutor インスタンス"""
        return EntryExecutor()

    @pytest.fixture
    def market_gene(self):
        """成行注文遺伝子"""
        return EntryGene(entry_type=EntryType.MARKET, enabled=True)

    @pytest.fixture
    def limit_gene(self):
        """指値注文遺伝子"""
        return EntryGene(
            entry_type=EntryType.LIMIT,
            limit_offset_pct=1.0,  # 1%
            enabled=True,
        )

    @pytest.fixture
    def stop_gene(self):
        """逆指値注文遺伝子"""
        return EntryGene(
            entry_type=EntryType.STOP,
            stop_offset_pct=2.0,  # 2%
            enabled=True,
        )

    @pytest.fixture
    def stop_limit_gene(self):
        """逆指値指値注文遺伝子"""
        return EntryGene(
            entry_type=EntryType.STOP_LIMIT,
            limit_offset_pct=0.5,  # 0.5%
            stop_offset_pct=1.0,  # 1%
            enabled=True,
        )

    def test_market_order_returns_empty_params(self, executor, market_gene):
        """成行注文の場合、空のパラメータを返す"""
        result = executor.calculate_entry_params(market_gene, 100.0, 1.0)
        assert result == {}

    def test_none_gene_returns_empty_params(self, executor):
        """遺伝子がNoneの場合、空のパラメータを返す"""
        result = executor.calculate_entry_params(None, 100.0, 1.0)
        assert result == {}

    def test_disabled_gene_returns_empty_params(self, executor):
        """無効な遺伝子の場合、空のパラメータを返す"""
        disabled_gene = EntryGene(entry_type=EntryType.LIMIT, enabled=False)
        result = executor.calculate_entry_params(disabled_gene, 100.0, 1.0)
        assert result == {}

    def test_limit_order_long(self, executor, limit_gene):
        """ロング指値注文: 現在価格より低い価格"""
        result = executor.calculate_entry_params(limit_gene, 100.0, 1.0)
        assert "limit" in result
        # Long: 100 * (1 - 0.01) = 99
        assert result["limit"] == pytest.approx(99.0)

    def test_limit_order_short(self, executor, limit_gene):
        """ショート指値注文: 現在価格より高い価格"""
        result = executor.calculate_entry_params(limit_gene, 100.0, -1.0)
        assert "limit" in result
        # Short: 100 * (1 + 0.01) = 101
        assert result["limit"] == pytest.approx(101.0)

    def test_stop_order_long(self, executor, stop_gene):
        """ロング逆指値注文: 現在価格より高い価格でブレイクアウト"""
        result = executor.calculate_entry_params(stop_gene, 100.0, 1.0)
        assert "stop" in result
        # Long: 100 * (1 + 0.02) = 102
        assert result["stop"] == pytest.approx(102.0)

    def test_stop_order_short(self, executor, stop_gene):
        """ショート逆指値注文: 現在価格より低い価格でブレイクアウト"""
        result = executor.calculate_entry_params(stop_gene, 100.0, -1.0)
        assert "stop" in result
        # Short: 100 * (1 - 0.02) = 98
        assert result["stop"] == pytest.approx(98.0)

    def test_stop_limit_order_long(self, executor, stop_limit_gene):
        """ロング逆指値指値注文: stopとlimit両方が設定される"""
        result = executor.calculate_entry_params(stop_limit_gene, 100.0, 1.0)
        assert "stop" in result
        assert "limit" in result
        # Stop: 100 * (1 + 0.01) = 101
        assert result["stop"] == pytest.approx(101.0)
        # Limit: stop価格からオフセット: 101 * (1 - 0.005) = 100.495
        assert result["limit"] == pytest.approx(101.0 * 0.995)

    def test_stop_limit_order_short(self, executor, stop_limit_gene):
        """ショート逆指値指値注文: stopとlimit両方が設定される"""
        result = executor.calculate_entry_params(stop_limit_gene, 100.0, -1.0)
        assert "stop" in result
        assert "limit" in result
        # Stop: 100 * (1 - 0.01) = 99
        assert result["stop"] == pytest.approx(99.0)
        # Limit: stop価格からオフセット: 99 * (1 + 0.005) = 99.495
        assert result["limit"] == pytest.approx(99.0 * 1.005)


class TestEntryGene:
    """EntryGene モデルのテストクラス"""

    def test_default_values(self):
        """デフォルト値のテスト"""
        gene = EntryGene()
        assert gene.entry_type == EntryType.MARKET
        assert gene.limit_offset_pct == 0.005
        assert gene.stop_offset_pct == 0.005
        assert gene.order_validity_bars == 5
        assert gene.enabled is True

    def test_validate_valid_gene(self):
        """有効な遺伝子のバリデーション"""
        gene = EntryGene(
            entry_type=EntryType.LIMIT,
            limit_offset_pct=0.01,
            stop_offset_pct=0.02,
            order_validity_bars=10,
        )
        is_valid, errors = gene.validate()
        assert is_valid is True
        assert len(errors) == 0

    def test_validate_invalid_limit_offset(self):
        """無効な limit_offset_pct のバリデーション"""
        gene = EntryGene(limit_offset_pct=0.15)  # 15% > 10%
        is_valid, errors = gene.validate()
        assert is_valid is False
        assert any("limit_offset_pct" in e for e in errors)

    def test_validate_invalid_stop_offset(self):
        """無効な stop_offset_pct のバリデーション"""
        gene = EntryGene(stop_offset_pct=-0.01)  # 負の値
        is_valid, errors = gene.validate()
        assert is_valid is False
        assert any("stop_offset_pct" in e for e in errors)

    def test_validate_invalid_order_validity_bars(self):
        """無効な order_validity_bars のバリデーション"""
        gene = EntryGene(order_validity_bars=-5)
        is_valid, errors = gene.validate()
        assert is_valid is False
        assert any("order_validity_bars" in e for e in errors)

    def test_to_dict(self):
        """to_dict メソッドのテスト"""
        gene = EntryGene(
            entry_type=EntryType.LIMIT,
            limit_offset_pct=0.01,
            stop_offset_pct=0.02,
            order_validity_bars=10,
        )
        result = gene.to_dict()
        assert result["entry_type"] == "limit"
        assert result["limit_offset_pct"] == 0.01
        assert result["stop_offset_pct"] == 0.02
        assert result["order_validity_bars"] == 10

    def test_from_dict(self):
        """from_dict メソッドのテスト"""
        data = {
            "entry_type": "stop",
            "limit_offset_pct": 0.005,
            "stop_offset_pct": 0.015,
            "order_validity_bars": 8,
            "enabled": True,
        }
        gene = EntryGene.from_dict(data)
        assert gene.entry_type == EntryType.STOP
        assert gene.limit_offset_pct == 0.005
        assert gene.stop_offset_pct == 0.015
        assert gene.order_validity_bars == 8

    def test_round_trip_dict(self):
        """辞書変換のラウンドトリップテスト"""
        original = EntryGene(
            entry_type=EntryType.STOP_LIMIT,
            limit_offset_pct=0.008,
            stop_offset_pct=0.012,
            order_validity_bars=15,
        )
        dict_data = original.to_dict()
        restored = EntryGene.from_dict(dict_data)

        assert restored.entry_type == original.entry_type
        assert restored.limit_offset_pct == original.limit_offset_pct
        assert restored.stop_offset_pct == original.stop_offset_pct
        assert restored.order_validity_bars == original.order_validity_bars
