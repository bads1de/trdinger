"""
auto_strategy リファクタリング検証テスト

リファクタリング後の動作確認を行うテストスイート
"""

import pytest
import sys
import os

# プロジェクトルートをパスに追加
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from app.services.auto_strategy.utils.constants import (
    OPERATORS,
    DATA_SOURCES,
    VALID_INDICATOR_TYPES,
)
from app.services.auto_strategy.models.gene_validation import GeneValidator
from app.services.auto_strategy.core.indicator_name_resolver import (
    IndicatorNameResolver,
)
from app.services.auto_strategy.services.tpsl_service import (
    TPSLService,
)
from app.services.auto_strategy.services.position_sizing_service import (
    PositionSizingService,
)
from app.services.auto_strategy.models.gene_serialization import GeneSerializer
from app.services.auto_strategy.models.gene_strategy import StrategyGene, IndicatorGene
from app.services.auto_strategy.models.gene_tpsl import TPSLGene, TPSLMethod
from app.services.auto_strategy.models.gene_position_sizing import (
    PositionSizingGene,
    PositionSizingMethod,
)
from app.services.indicators.config.indicator_config import indicator_registry


class TestConstantsUnification:
    """定数管理の一元化テスト"""

    def test_constants_availability(self):
        """定数が正しく定義されているかテスト"""
        assert len(OPERATORS) > 0
        assert len(DATA_SOURCES) > 0
        assert len(VALID_INDICATOR_TYPES) > 0

        # 基本的な演算子が含まれているかチェック
        assert ">" in OPERATORS
        assert "<" in OPERATORS
        assert ">=" in OPERATORS
        assert "<=" in OPERATORS

        # 基本的なデータソースが含まれているかチェック
        assert "close" in DATA_SOURCES
        assert "open" in DATA_SOURCES
        assert "high" in DATA_SOURCES
        assert "low" in DATA_SOURCES

    def test_gene_validator_uses_unified_constants(self):
        """GeneValidatorが統一定数を使用しているかテスト"""
        validator = GeneValidator()

        # 統一定数から取得されているかチェック
        assert validator.valid_operators == OPERATORS
        assert validator.valid_data_sources == DATA_SOURCES
        assert validator.valid_indicator_types == VALID_INDICATOR_TYPES


class TestIndicatorNameResolver:
    """指標名解決ロジックの改善テスト"""

    def test_dynamic_resolution(self):
        """動的解決が機能するかテスト"""

        # モックのstrategy_instanceを作成
        class MockStrategy:
            def __init__(self):
                self.MACD_0 = [1.0, 2.0, 3.0]
                self.BB_1 = [50.0, 51.0, 52.0]
                self.RSI = [30.0, 40.0, 50.0]

        strategy = MockStrategy()

        # 動的解決のテスト
        resolved, value = IndicatorNameResolver.try_resolve_value("MACD_0", strategy)
        assert resolved is True
        assert value == 3.0  # 最後の値

        resolved, value = IndicatorNameResolver.try_resolve_value("BB_1", strategy)
        assert resolved is True
        assert value == 52.0

        resolved, value = IndicatorNameResolver.try_resolve_value("RSI", strategy)
        assert resolved is True
        assert value == 50.0

    def test_indicator_registry_integration(self):
        """指標レジストリとの統合テスト"""
        # 指標レジストリから解決できるかテスト
        resolved = indicator_registry.resolve_indicator_name("MACD")
        assert resolved == "MACD"

        resolved = indicator_registry.resolve_indicator_name("MACD_0")
        assert resolved == "MACD"

        # 出力インデックスの取得テスト
        index = indicator_registry.get_output_index("MACD_0")
        assert index == 0

        index = indicator_registry.get_output_index("BB_1")
        assert index == 1


class TestCalculationServices:
    """計算ロジックの集約テスト"""

    def test_tpsl_calculator_service(self):
        """TP/SL計算サービステスト"""
        service = TPSLService()

        # 基本的なTP/SL計算テスト
        current_price = 50000.0
        sl_price, tp_price = service.calculate_tpsl_prices(
            current_price=current_price,
            stop_loss_pct=0.03,
            take_profit_pct=0.06,
            position_direction=1.0,  # ロング
        )

        assert sl_price is not None
        assert tp_price is not None
        assert sl_price < current_price  # ロングのSLは現在価格より低い
        assert tp_price > current_price  # ロングのTPは現在価格より高い

    def test_tpsl_calculator_with_gene(self):
        """TP/SL遺伝子を使用した計算テスト"""
        service = TPSLService()

        tpsl_gene = TPSLGene(
            method=TPSLMethod.FIXED_PERCENTAGE,
            stop_loss_pct=0.02,
            take_profit_pct=0.04,
            enabled=True,
        )

        current_price = 50000.0
        sl_price, tp_price = service.calculate_tpsl_prices(
            current_price=current_price, tpsl_gene=tpsl_gene, position_direction=1.0
        )

        assert sl_price is not None
        assert tp_price is not None
        assert abs(sl_price - current_price * 0.98) < 1.0  # 2%のSL
        assert abs(tp_price - current_price * 1.04) < 1.0  # 4%のTP

    def test_position_sizing_service(self):
        """ポジションサイジングサービステスト"""
        service = PositionSizingService()

        # 基本的なポジションサイズ計算テスト（簡易版）
        position_size = service.calculate_position_size_simple(
            method="fixed_ratio",
            account_balance=100000.0,
            current_price=50000.0,
            fixed_ratio=0.1,
        )

        assert position_size > 0

    def test_position_sizing_with_gene(self):
        """ポジションサイジング遺伝子を使用した計算テスト"""
        service = PositionSizingService()

        position_sizing_gene = PositionSizingGene(
            method=PositionSizingMethod.FIXED_RATIO, fixed_ratio=0.1, enabled=True
        )

        result = service.calculate_position_size(
            gene=position_sizing_gene,
            account_balance=100000.0,
            current_price=50000.0,
        )

        assert result.position_size > 0
        assert result.method_used == "fixed_ratio"


class TestEncodingDecoding:
    """エンコード/デコード層の簡素化テスト"""

    def test_direct_encoder_usage(self):
        """Encoderクラスの直接使用テスト"""
        serializer = GeneSerializer()

        # テスト用の戦略遺伝子を作成
        strategy_gene = StrategyGene(
            indicators=[
                IndicatorGene(type="RSI", parameters={"period": 14}, enabled=True)
            ]
        )

        # エンコードテスト
        encoded = serializer.to_list(strategy_gene)
        assert isinstance(encoded, list)
        assert len(encoded) > 0
        assert all(isinstance(x, (int, float)) for x in encoded)

    def test_direct_decoder_usage(self):
        """Decoderクラスの直接使用テスト"""
        serializer = GeneSerializer()

        # テスト用のエンコードデータ
        encoded = [0.5, 0.7] + [0.0] * 30  # 32要素のテストデータ

        # デコードテスト
        decoded_gene = serializer.from_list(encoded, StrategyGene)
        assert isinstance(decoded_gene, StrategyGene)
        assert len(decoded_gene.indicators) >= 0

    def test_encode_decode_roundtrip(self):
        """エンコード→デコードの往復テスト"""
        serializer = GeneSerializer()

        # 元の戦略遺伝子
        original_gene = StrategyGene(
            indicators=[
                IndicatorGene(type="RSI", parameters={"period": 14}, enabled=True),
                IndicatorGene(
                    type="MACD", parameters={"fast": 12, "slow": 26}, enabled=True
                ),
            ]
        )

        # エンコード→デコード
        encoded = serializer.to_list(original_gene)
        decoded_gene = serializer.from_list(encoded, StrategyGene)

        # 基本的な構造が保持されているかチェック
        assert isinstance(decoded_gene, StrategyGene)
        assert len(decoded_gene.indicators) > 0


class TestIntegration:
    """統合テスト"""

    def test_full_workflow(self):
        """フルワークフローテスト"""
        # 1. 戦略遺伝子の作成
        from app.services.auto_strategy.models.gene_strategy import Condition

        strategy_gene = StrategyGene(
            indicators=[
                IndicatorGene(type="RSI", parameters={"period": 14}, enabled=True)
            ],
            entry_conditions=[
                Condition(left_operand="RSI", operator=">", right_operand="30")
            ],
            long_entry_conditions=[
                Condition(left_operand="RSI", operator=">", right_operand="30")
            ],
            short_entry_conditions=[
                Condition(left_operand="RSI", operator="<", right_operand="70")
            ],
            tpsl_gene=TPSLGene(
                method=TPSLMethod.FIXED_PERCENTAGE,
                stop_loss_pct=0.02,
                take_profit_pct=0.04,
                enabled=True,
            ),
            position_sizing_gene=PositionSizingGene(
                method=PositionSizingMethod.FIXED_RATIO, fixed_ratio=0.1, enabled=True
            ),
        )

        # 2. バリデーション
        validator = GeneValidator()
        is_valid, errors = validator.validate_strategy_gene(strategy_gene)
        assert is_valid, f"Validation errors: {errors}"

        # 3. エンコード/デコード
        serializer = GeneSerializer()

        encoded = serializer.to_list(strategy_gene)
        decoded = serializer.from_list(encoded, StrategyGene)

        assert isinstance(decoded, StrategyGene)

        # 4. 計算サービスの使用
        tpsl_service = TPSLService()
        position_service = PositionSizingService()

        sl_price, tp_price = tpsl_service.calculate_tpsl_prices(
            current_price=50000.0, tpsl_gene=decoded.tpsl_gene, position_direction=1.0
        )

        position_result = position_service.calculate_position_size(
            gene=decoded.position_sizing_gene,
            account_balance=100000.0,
            current_price=50000.0,
        )

        assert sl_price is not None
        assert tp_price is not None
        assert position_result.position_size > 0


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
