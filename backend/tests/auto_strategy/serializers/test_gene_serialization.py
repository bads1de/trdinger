"""
テストコード：GeneSerialization クラスのテスト
TDDアプローチでバグを洗い出して修正する
"""

import pytest
import json
from unittest.mock import Mock, MagicMock, patch
import uuid

from app.services.auto_strategy.serializers.gene_serialization import GeneSerializer
from app.services.auto_strategy.models.strategy_models import (
    IndicatorGene, Condition, TPSLGene, TPSLMethod, PositionSizingGene, PositionSizingMethod
)


class TestGeneSerializer:
    """GeneSerializerクラスのテスト"""

    @pytest.fixture
    def serializer(self):
        """GeneSerializerのフィーチャー"""
        return GeneSerializer(enable_smart_generation=False)

    @pytest.fixture
    def mock_strategy_gene(self):
        """モック戦略遺伝子"""
        strategy_gene = Mock()
        strategy_gene.id = str(uuid.uuid4())
        strategy_gene.indicators = [
            Mock(type="SMA", parameters={"period": 20}, enabled=True),
            Mock(type="RSI", parameters={"period": 14}, enabled=True)
        ]
        strategy_gene.entry_conditions = []
        strategy_gene.long_entry_conditions = [
            Mock(left_operand="close", operator=">", right_operand="sma")
        ]
        strategy_gene.short_entry_conditions = [
            Mock(left_operand="close", operator="<", right_operand="sma")
        ]
        strategy_gene.exit_conditions = []
        strategy_gene.risk_management = {
            "stop_loss": 0.03,
            "take_profit": 0.06,
            "position_size": 0.1
        }
        strategy_gene.tpsl_gene = Mock()
        strategy_gene.tpsl_gene.to_dict.return_value = {
            "enabled": True, "method": "RISK_REWARD_RATIO", "stop_loss_pct": 0.03, "take_profit_pct": 0.06
        }
        strategy_gene.position_sizing_gene = Mock()
        strategy_gene.position_sizing_gene.to_dict.return_value = {
            "enabled": True, "method": "VOLATILITY_BASED", "risk_per_trade": 0.02
        }
        strategy_gene.metadata = {"test": True}
        return strategy_gene

    @pytest.fixture
    def mock_indicator_gene(self):
        """モック指標遺伝子"""
        gene = Mock()
        gene.type = "SMA"
        gene.parameters = {"period": 20}
        gene.enabled = True
        return gene

    @pytest.fixture
    def mock_condition(self):
        """モック条件"""
        condition = Mock()
        condition.left_operand = "close"
        condition.operator = ">"
        condition.right_operand = "sma"
        return condition

    def test_initialization(self, serializer):
        """初期化テスト"""
        assert serializer.enable_smart_generation is False
        assert serializer._smart_condition_generator is None
        assert isinstance(serializer.indicator_ids, dict)
        assert isinstance(serializer.id_to_indicator, dict)

    def test_indicator_gene_to_dict(self, serializer, mock_indicator_gene):
        """指標遺伝子辞書変換テスト"""
        result = serializer.indicator_gene_to_dict(mock_indicator_gene)

        assert result["type"] == "SMA"
        assert result["parameters"] == {"period": 20}
        assert result["enabled"] is True

    def test_indicator_gene_to_dict_missing_attributes(self, serializer):
        """欠損属性の場合のテスト"""
        # 異常な indicator_gene のテスト
        mock_gene = Mock()
        mock_gene.type = "INVALID"
        # attributes を欠損させる
        del mock_gene.parameters

        with pytest.raises(ValueError):
            serializer.indicator_gene_to_dict(mock_gene)

    def test_dict_to_indicator_gene(self, serializer):
        """辞書から指標遺伝子復元テスト"""
        data = {
            "type": "RSI",
            "parameters": {"period": 14},
            "enabled": False  # explicit False
        }

        gene = serializer.dict_to_indicator_gene(data)

        assert gene.type == "RSI"
        assert gene.parameters == {"period": 14}
        assert gene.enabled is False

    def test_dict_to_indicator_gene_default_enabled(self, serializer):
        """enabled属性が欠損の場合のデフォルトテスト"""
        data = {
            "type": "MACD",
            "parameters": {"fast_period": 12}
            # enabled を欠損
        }

        gene = serializer.dict_to_indicator_gene(data)

        assert gene.enabled is True  # default

    def test_condition_to_dict(self, serializer, mock_condition):
        """条件辞書変換テスト"""
        result = serializer.condition_to_dict(mock_condition)

        expected = {
            "left_operand": "close",
            "operator": ">",
            "right_operand": "sma"
        }
        assert result == expected

    def test_condition_to_dict_invalid(self, serializer):
        """無効な条件変換テスト"""
        mock_invalid = Mock()
        # attributes を欠損
        del mock_invalid.left_operand

        with pytest.raises(ValueError):
            serializer.condition_to_dict(mock_invalid)

    def test_tpsl_gene_to_dict(self, serializer, mock_strategy_gene):
        """TP/SL遺伝子辞書変換テスト"""
        result = serializer.tpsl_gene_to_dict(mock_strategy_gene.tpsl_gene)

        assert "enabled" in result
        assert "method" in result

    def test_tpsl_gene_to_dict_none(self, serializer):
        """TP/SL遺伝子がNoneのテスト"""
        result = serializer.tpsl_gene_to_dict(None)

        assert result is None

    def test_dict_to_tpsl_gene(self, serializer):
        """辞書からTP/SL遺伝子復元テスト"""
        data = {
            "enabled": True,
            "method": "risk_reward_ratio",
            "stop_loss_pct": 0.02,
            "take_profit_pct": 0.04,
            "atr_multiplier_sl": 1.5,
            "atr_multiplier_tp": 2.0
        }

        tpsl_gene = serializer.dict_to_tpsl_gene(data)

        assert tpsl_gene.enabled is True
        assert tpsl_gene.stop_loss_pct == 0.02

    def test_dict_to_tpsl_gene_none_data(self, serializer):
        """データがNoneの場合のTP/SL遺伝子復元テスト"""
        result = serializer.dict_to_tpsl_gene(None)

        assert result is None

    def test_dict_to_tpsl_gene_invalid_data(self, serializer):
        """無効なデータでのTP/SL遺伝子復元テスト"""
        # from_dictが例外を投げる場合をテスト
        with patch('app.services.auto_strategy.models.strategy_models.TPSLGene.from_dict') as mock_from_dict:
            mock_from_dict.side_effect = Exception("Test error")

            data = {"invalid": "data"}
            with pytest.raises(ValueError):
                serializer.dict_to_tpsl_gene(data)

    def test_position_sizing_gene_to_dict(self, serializer, mock_strategy_gene):
        """ポジションサイジング遺伝子辞書変換テスト"""
        result = serializer.position_sizing_gene_to_dict(mock_strategy_gene.position_sizing_gene)

        assert isinstance(result, dict)
        assert "enabled" in result

    def test_dict_to_position_sizing_gene(self, serializer):
        """辞書からポジションサイジング遺伝子復元テスト"""
        data = {
            "enabled": True,
            "method": "volatility_based",
            "risk_per_trade": 0.01,
            "atr_multiplier": 1.8
        }

        ps_gene = serializer.dict_to_position_sizing_gene(data)

        assert ps_gene.enabled is True
        assert ps_gene.risk_per_trade == 0.01

    def test_clean_risk_management(self, serializer):
        """リスク管理クリーンアップテスト"""
        risk_management = {
            "stop_loss": 0.03,
            "take_profit": 0.15,
            "position_size": 0.1,
            "stop_loss_pct": 0.03,  # should be removed
            "take_profit_pct": 0.15,  # should be removed
            "tpsl_strategy": "test"  # should be removed
        }

        result = serializer._clean_risk_management(risk_management)

        assert "stop_loss" not in result  # removed
        assert "take_profit" not in result  # removed
        assert "position_size" in result
        assert round(result["position_size"], 6) == 0.100000  # rounded to 6 digits

    def test_strategy_gene_to_dict(self, serializer, mock_strategy_gene):
        """戦略遺伝子辞書変換テスト"""
        result = serializer.strategy_gene_to_dict(mock_strategy_gene)

        assert "id" in result
        assert "indicators" in result
        assert "entry_conditions" in result
        assert "risk_management" in result
        assert "tpsl_gene" in result
        assert "position_sizing_gene" in result

    def test_strategy_gene_to_json(self, serializer, mock_strategy_gene):
        """戦略遺伝子JSON変換テスト"""
        with patch.object(serializer, 'strategy_gene_to_dict') as mock_to_dict:
            mock_to_dict.return_value = {"test": "data"}

            result = serializer.strategy_gene_to_json(mock_strategy_gene)

            assert isinstance(result, str)
            json.loads(result)  # valid JSON

    def test_to_list_basic(self, serializer, mock_strategy_gene):
        """基本エンコードテスト"""
        with patch.object(serializer, 'MAX_INDICATORS', 2, create=True):
            encoded = serializer.to_list(mock_strategy_gene)

            assert isinstance(encoded, list)
            assert len(encoded) == 32  # 2*2 + 6 + 8*4

    def test_to_list_none_strategy_gene(self, serializer):
        """None戦略遺伝子のエンコードテスト"""
        result = serializer.to_list(None)
        assert isinstance(result, list)
        assert len(result) == 32
        assert all(isinstance(x, float) for x in result)

    def test_from_list_invalid_encoding(self, serializer):
        """無効なエンコーディングリストからのデテスト"""
        invalid_encodings = [[], None]

        for invalid in invalid_encodings:
            decoded = serializer.from_list(invalid, Mock)
            assert decoded is not None  # default gene returned

    def test_encode_decode_round_trip(self, serializer, mock_strategy_gene):
        """エンコード・デコードのラウンドトリップテスト"""
        # このテストはバグを洗い出すために失敗するはず
        with patch.object(serializer, 'MAX_INDICATORS', 2, create=True):
            encoded = serializer.to_list(mock_strategy_gene)
            decoded = serializer.from_list(encoded, type(mock_strategy_gene))

            assert decoded is not None

    # TODO: 追加のテストケースを続けて記述