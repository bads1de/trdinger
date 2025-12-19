from unittest.mock import MagicMock

import pytest

from app.services.auto_strategy.config.constants import EntryType
from app.services.auto_strategy.genes.entry import EntryGene, create_random_entry_gene


class TestEntryGene:

    def test_init_default(self):
        gene = EntryGene()
        assert gene.entry_type == EntryType.MARKET
        assert gene.limit_offset_pct == 0.005
        assert gene.stop_offset_pct == 0.005
        assert gene.order_validity_bars == 5
        assert gene.enabled is True

    def test_validate_valid(self):
        gene = EntryGene(
            entry_type=EntryType.LIMIT,
            limit_offset_pct=0.01,
            stop_offset_pct=0.01,
            order_validity_bars=10,
        )
        is_valid, errors = gene.validate()
        assert is_valid is True
        assert len(errors) == 0

    def test_validate_invalid_limit_offset(self):
        gene = EntryGene(limit_offset_pct=-0.01)
        is_valid, errors = gene.validate()
        assert is_valid is False
        assert "limit_offset_pct" in errors[0]

        gene = EntryGene(limit_offset_pct=0.11)
        is_valid, errors = gene.validate()
        assert is_valid is False
        assert "limit_offset_pct" in errors[0]

    def test_validate_invalid_stop_offset(self):
        gene = EntryGene(stop_offset_pct=-0.01)
        is_valid, errors = gene.validate()
        assert is_valid is False
        assert "stop_offset_pct" in errors[0]

        gene = EntryGene(stop_offset_pct=0.11)
        is_valid, errors = gene.validate()
        assert is_valid is False
        assert "stop_offset_pct" in errors[0]

    def test_validate_invalid_validity_bars(self):
        gene = EntryGene(order_validity_bars=-1)
        is_valid, errors = gene.validate()
        assert is_valid is False
        assert "order_validity_bars" in errors[0]

    def test_validate_invalid_entry_type(self):
        # EntryType以外の値を無理やり入れる
        gene = EntryGene()
        gene.entry_type = "invalid_type"
        is_valid, errors = gene.validate()
        assert is_valid is False
        assert "entry_type" in errors[0]

    def test_to_dict(self):
        gene = EntryGene(entry_type=EntryType.STOP)
        data = gene.to_dict()
        assert data["entry_type"] == "stop"
        assert data["limit_offset_pct"] == 0.005

    def test_from_dict(self):
        data = {
            "entry_type": "stop_limit",
            "limit_offset_pct": 0.02,
            "stop_offset_pct": 0.03,
            "order_validity_bars": 20,
            "enabled": False,
            "priority": 2.0,
        }
        gene = EntryGene.from_dict(data)
        assert gene.entry_type == EntryType.STOP_LIMIT
        assert gene.limit_offset_pct == 0.02
        assert gene.stop_offset_pct == 0.03
        assert gene.order_validity_bars == 20
        assert gene.enabled is False
        assert gene.priority == 2.0

    def test_from_dict_enum_handling(self):
        # Enumオブジェクトをそのまま渡した場合
        data = {"entry_type": EntryType.MARKET}
        gene = EntryGene.from_dict(data)
        assert gene.entry_type == EntryType.MARKET

        # 不正な文字列の場合 -> MARKETにフォールバック
        data = {"entry_type": "unknown"}
        # EntryType("unknown") で ValueError になるはずだが、コードでは catch していないように見える？
        # コードを確認すると:
        # if isinstance(entry_type_value, str): entry_type = EntryType(entry_type_value)
        # とあるので、不正な値だとここで ValueError になる。
        # 呼び出し元が責任を持つ設計か、あるいはテストで確認する。
        with pytest.raises(ValueError):
            EntryGene.from_dict(data)

    def test_create_random_entry_gene_defaults(self):
        gene = create_random_entry_gene()
        assert isinstance(gene, EntryGene)
        assert isinstance(gene.entry_type, EntryType)
        assert 0.001 <= gene.limit_offset_pct <= 0.02
        assert 1 <= gene.order_validity_bars <= 20

    def test_create_random_entry_gene_with_config(self):
        # Configオブジェクトのモック
        config = MagicMock()
        config.entry_type_weights = {"limit": 1.0, "market": 0.0}

        gene = create_random_entry_gene(config)
        assert gene.entry_type == EntryType.LIMIT

    def test_create_random_entry_gene_fallback(self):
        # config.entry_type_weights に不正なキー（EntryTypeに変換できない文字列）を含める
        config = MagicMock()
        config.entry_type_weights = {"invalid_type_string": 1.0}

        # create_random_entry_gene 内で EntryType("invalid_type_string") が呼ばれ
        # ValueError が発生し、except ブロックでキャッチされてデフォルト値が返るはず
        gene = create_random_entry_gene(config)

        # フォールバックして MARKET になっているはず
        assert gene.entry_type == EntryType.MARKET
        assert gene.limit_offset_pct == 0.005


from unittest.mock import PropertyMock
