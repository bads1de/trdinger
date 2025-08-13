"""
auto_strategy リファクタリング検証テスト

リファクタリング後の動作確認を行うテストスイート
統合されたエラーハンドリング、ユーティリティ、設定クラスをテストします。
"""

import pytest
import sys
import os
from unittest.mock import Mock, patch

# プロジェクトルートをパスに追加
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from app.services.auto_strategy.config.constants import (
    OPERATORS,
    DATA_SOURCES,
    VALID_INDICATOR_TYPES,
)
from app.services.auto_strategy.utils.error_handling import AutoStrategyErrorHandler
from app.services.auto_strategy.utils.auto_strategy_utils import AutoStrategyUtils
from app.services.auto_strategy.config.base_config import BaseConfig
from app.services.auto_strategy.config.constants import (
    validate_symbol,
    validate_timeframe,
)
from app.services.auto_strategy.models.ga_config import GAConfig
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


class TestRefactoredComponents:
    """リファクタリング後のコンポーネントテスト"""

    def test_auto_strategy_error_handler(self):
        """統合エラーハンドラーのテスト"""
        # GA エラーハンドリング
        error = ValueError("テストエラー")
        result = AutoStrategyErrorHandler.handle_ga_error(error, "テストコンテキスト")

        assert result["error_code"] == "GA_ERROR"
        assert "テストエラー" in result["message"]
        assert result["context"] == "テストコンテキスト"

        # 戦略生成エラーハンドリング
        strategy_data = {"test": "data"}
        result = AutoStrategyErrorHandler.handle_strategy_generation_error(
            error, strategy_data, "テスト戦略生成"
        )

        assert result["success"] is False
        assert result["strategy"] is None
        assert "テストエラー" in result["error"]

    def test_auto_strategy_utils(self):
        """統合ユーティリティのテスト"""
        # データ変換テスト
        assert AutoStrategyUtils.safe_convert_to_float("123.45") == 123.45
        assert AutoStrategyUtils.safe_convert_to_float("invalid", 0.0) == 0.0

        # シンボル正規化テスト
        assert AutoStrategyUtils.normalize_symbol("BTC") == "BTC:USDT"
        assert AutoStrategyUtils.normalize_symbol("BTC:USDT") == "BTC:USDT"

        # 検証テスト
        assert AutoStrategyUtils.validate_range(5, 1, 10) is True
        assert AutoStrategyUtils.validate_range(15, 1, 10) is False

    def test_base_config_integration(self):
        """BaseConfig統合のテスト"""

        class TestConfig(BaseConfig):
            def __init__(self, test_field=None, enabled=True):
                super().__init__(enabled=enabled)
                self.test_field = test_field
                self.validation_rules = {
                    "required_fields": ["test_field"],
                    "ranges": {"test_field": (1, 10)},
                }

            def get_default_values(self):
                return {"enabled": True, "test_field": 5}

        # 正常ケース
        config = TestConfig(test_field=5)
        is_valid, errors = config.validate()
        assert is_valid is True

        # 辞書からの作成
        data = {"test_field": 7, "enabled": False}
        config = TestConfig.from_dict(data)
        assert config.test_field == 7
        assert config.enabled is False

    def test_ga_config_base_config_inheritance(self):
        """GAConfigのBaseConfig継承テスト"""
        config = GAConfig()

        # BaseConfigのメソッドが使用可能か確認
        assert hasattr(config, "validate")
        assert hasattr(config, "get_default_values")
        assert hasattr(config, "to_dict")

        # 検証機能のテスト
        is_valid, errors = config.validate()
        assert is_valid is True

        # デフォルト値の取得
        defaults = config.get_default_values()
        assert "population_size" in defaults
        assert defaults["population_size"] == 10

    def test_shared_constants_integration(self):
        """共通定数統合のテスト"""
        # 定数の存在確認
        assert ">" in OPERATORS
        assert "close" in DATA_SOURCES
        assert "SMA" in VALID_INDICATOR_TYPES

        # 検証関数のテスト
        assert validate_symbol("BTC/USDT:USDT") is True
        assert validate_symbol("INVALID") is False
        assert validate_timeframe("1h") is True
        assert validate_timeframe("5m") is False

    @patch("app.services.indicators.TechnicalIndicatorService")
    def test_indicator_id_integration(self, mock_service):
        """指標ID統合のテスト"""
        mock_instance = Mock()
        mock_instance.get_supported_indicators.return_value = {"SMA": {}, "RSI": {}}
        mock_service.return_value = mock_instance

        result = AutoStrategyUtils.get_all_indicator_ids()

        assert "" in result
        assert result[""] == 0
        assert "SMA" in result
        assert "ML_UP_PROB" in result

    def test_error_handling_backward_compatibility(self):
        """エラーハンドリングの後方互換性テスト"""

        # 既存のsafe_execute関数が動作するか確認
        def success_func():
            return "成功"

        result = AutoStrategyErrorHandler.safe_execute(success_func)
        assert result == "成功"

        # エラーケース
        def error_func():
            raise ValueError("テストエラー")

        result = AutoStrategyErrorHandler.safe_execute(
            error_func, fallback_value="フォールバック"
        )
        assert result == "フォールバック"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
