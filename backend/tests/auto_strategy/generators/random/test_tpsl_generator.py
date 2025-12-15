"""
TPSLGenerator（generators/random）のテスト

TP/SL遺伝子生成ロジックのテスト
"""

import pytest
from unittest.mock import Mock, patch


class TestTPSLGeneratorInit:
    """初期化のテスト"""

    def test_init_stores_config(self):
        """設定が保存される"""
        from app.services.auto_strategy.generators.random.tpsl_generator import (
            TPSLGenerator,
        )

        config = Mock()
        generator = TPSLGenerator(config)

        assert generator.config == config


class TestGenerateTPSLGene:
    """generate_tpsl_geneのテスト"""

    @pytest.fixture
    def generator(self):
        """基本設定のジェネレータ"""
        from app.services.auto_strategy.generators.random.tpsl_generator import (
            TPSLGenerator,
        )

        config = Mock()
        # 必要な属性を設定（存在しない場合のハンドリングをテスト）
        config.tpsl_method_constraints = None
        config.tpsl_sl_range = None
        config.tpsl_tp_range = None
        config.tpsl_rr_range = None
        config.tpsl_atr_multiplier_range = None
        return TPSLGenerator(config)

    def test_returns_tpsl_gene(self, generator):
        """TPSLGeneを返す"""
        from app.services.auto_strategy.models import TPSLGene

        result = generator.generate_tpsl_gene()

        assert isinstance(result, TPSLGene)

    def test_applies_method_constraints(self):
        """メソッド制約を適用"""
        from app.services.auto_strategy.generators.random.tpsl_generator import (
            TPSLGenerator,
        )
        from app.services.auto_strategy.models import (
            TPSLGene,
            TPSLMethod,
        )

        config = Mock()
        config.tpsl_method_constraints = ["risk_reward_ratio", "volatility_based"]
        config.tpsl_sl_range = None
        config.tpsl_tp_range = None
        config.tpsl_rr_range = None
        config.tpsl_atr_multiplier_range = None

        generator = TPSLGenerator(config)

        # 複数回実行して制約が適用されることを確認
        methods_used = set()
        for _ in range(20):
            with patch(
                "app.services.auto_strategy.generators.random.tpsl_generator.create_random_tpsl_gene",
                return_value=TPSLGene(
                    method=TPSLMethod.FIXED_PERCENTAGE,
                    stop_loss_pct=0.02,
                    take_profit_pct=0.04,
                    risk_reward_ratio=2.0,
                    base_stop_loss=0.02,
                    enabled=True,
                ),
            ):
                result = generator.generate_tpsl_gene()
                methods_used.add(result.method)

        # 使用されたメソッドが制約内であることを確認
        for method in methods_used:
            assert method.value in ["risk_reward_ratio", "volatility_based"]

    def test_applies_sl_range_constraints(self):
        """SL範囲制約を適用"""
        from app.services.auto_strategy.generators.random.tpsl_generator import (
            TPSLGenerator,
        )

        config = Mock()
        config.tpsl_method_constraints = None
        config.tpsl_sl_range = (0.01, 0.02)
        config.tpsl_tp_range = None
        config.tpsl_rr_range = None
        config.tpsl_atr_multiplier_range = None

        generator = TPSLGenerator(config)
        result = generator.generate_tpsl_gene()

        # SL値が範囲内であることを確認
        assert 0.01 <= result.stop_loss_pct <= 0.02
        assert 0.01 <= result.base_stop_loss <= 0.02

    def test_applies_tp_range_constraints(self):
        """TP範囲制約を適用"""
        from app.services.auto_strategy.generators.random.tpsl_generator import (
            TPSLGenerator,
        )

        config = Mock()
        config.tpsl_method_constraints = None
        config.tpsl_sl_range = None
        config.tpsl_tp_range = (0.03, 0.05)
        config.tpsl_rr_range = None
        config.tpsl_atr_multiplier_range = None

        generator = TPSLGenerator(config)
        result = generator.generate_tpsl_gene()

        # TP値が範囲内であることを確認
        assert 0.03 <= result.take_profit_pct <= 0.05

    def test_applies_rr_range_constraints(self):
        """リスクリワード比範囲制約を適用"""
        from app.services.auto_strategy.generators.random.tpsl_generator import (
            TPSLGenerator,
        )

        config = Mock()
        config.tpsl_method_constraints = None
        config.tpsl_sl_range = None
        config.tpsl_tp_range = None
        config.tpsl_rr_range = (1.5, 3.0)
        config.tpsl_atr_multiplier_range = None

        generator = TPSLGenerator(config)
        result = generator.generate_tpsl_gene()

        # RR値が範囲内であることを確認
        assert 1.5 <= result.risk_reward_ratio <= 3.0

    def test_applies_atr_multiplier_range_constraints(self):
        """ATR倍率範囲制約を適用"""
        from app.services.auto_strategy.generators.random.tpsl_generator import (
            TPSLGenerator,
        )

        config = Mock()
        config.tpsl_method_constraints = None
        config.tpsl_sl_range = None
        config.tpsl_tp_range = None
        config.tpsl_rr_range = None
        config.tpsl_atr_multiplier_range = (1.0, 2.0)

        generator = TPSLGenerator(config)
        result = generator.generate_tpsl_gene()

        # ATR倍率SLが範囲内であることを確認
        assert 1.0 <= result.atr_multiplier_sl <= 2.0
        # ATR倍率TP範囲（1.5x ~ 2.0x）
        assert 1.5 <= result.atr_multiplier_tp <= 4.0

    def test_returns_fallback_on_error(self, generator):
        """エラー時にフォールバック遺伝子を返す"""
        from app.services.auto_strategy.models import TPSLMethod

        with patch(
            "app.services.auto_strategy.generators.random.tpsl_generator.create_random_tpsl_gene",
            side_effect=Exception("Generation failed"),
        ):
            result = generator.generate_tpsl_gene()

        # フォールバック値を確認
        assert result.method == TPSLMethod.RISK_REWARD_RATIO
        assert result.stop_loss_pct == 0.03
        assert result.take_profit_pct == 0.06
        assert result.risk_reward_ratio == 2.0
        assert result.base_stop_loss == 0.03
        assert result.enabled is True

    def test_logs_error_on_failure(self, generator):
        """エラー時にログを出力"""
        with patch(
            "app.services.auto_strategy.generators.random.tpsl_generator.create_random_tpsl_gene",
            side_effect=Exception("Test error"),
        ):
            with patch(
                "app.services.auto_strategy.generators.random.tpsl_generator.logger"
            ) as mock_logger:
                generator.generate_tpsl_gene()

        mock_logger.error.assert_called_once()
        assert "Test error" in str(mock_logger.error.call_args)


class TestTPSLGeneConstraintCombinations:
    """複数の制約の組み合わせテスト"""

    def test_all_constraints_applied(self):
        """全ての制約が同時に適用される"""
        from app.services.auto_strategy.generators.random.tpsl_generator import (
            TPSLGenerator,
        )
        from app.services.auto_strategy.models import (
            TPSLGene,
            TPSLMethod,
        )

        config = Mock()
        config.tpsl_method_constraints = ["risk_reward_ratio"]
        config.tpsl_sl_range = (0.01, 0.015)
        config.tpsl_tp_range = (0.04, 0.05)
        config.tpsl_rr_range = (2.5, 3.5)
        config.tpsl_atr_multiplier_range = (1.5, 2.5)

        # モック遺伝子を作成
        mock_gene = TPSLGene(
            method=TPSLMethod.VOLATILITY_BASED,  # 最初は別のメソッド
            stop_loss_pct=0.05,  # 範囲外の値
            take_profit_pct=0.10,  # 範囲外の値
            risk_reward_ratio=1.0,  # 範囲外の値
            base_stop_loss=0.05,
            atr_multiplier_sl=1.0,  # 範囲外の値
            atr_multiplier_tp=2.0,
            enabled=True,
        )

        with patch(
            "app.services.auto_strategy.generators.random.tpsl_generator.create_random_tpsl_gene",
            return_value=mock_gene,
        ):
            generator = TPSLGenerator(config)
            result = generator.generate_tpsl_gene()

        # メソッド制約が適用される
        assert result.method == TPSLMethod.RISK_REWARD_RATIO

        # SL範囲制約が適用される
        assert 0.01 <= result.stop_loss_pct <= 0.015
        assert 0.01 <= result.base_stop_loss <= 0.015

        # TP範囲制約が適用される
        assert 0.04 <= result.take_profit_pct <= 0.05

        # RR範囲制約が適用される
        assert 2.5 <= result.risk_reward_ratio <= 3.5

        # ATR倍率範囲制約が適用される
        assert 1.5 <= result.atr_multiplier_sl <= 2.5
        # atr_multiplier_tpは1.5x〜2.0xの範囲でスケール
        assert 2.25 <= result.atr_multiplier_tp <= 5.0

    def test_no_constraints_uses_random_values(self):
        """制約なしの場合はランダム値を使用"""
        from app.services.auto_strategy.generators.random.tpsl_generator import (
            TPSLGenerator,
        )
        from app.services.auto_strategy.models import TPSLGene

        config = Mock()
        config.tpsl_method_constraints = None
        config.tpsl_sl_range = None
        config.tpsl_tp_range = None
        config.tpsl_rr_range = None
        config.tpsl_atr_multiplier_range = None

        mock_gene = TPSLGene(
            method="RISK_REWARD_RATIO",
            stop_loss_pct=0.025,
            take_profit_pct=0.05,
            risk_reward_ratio=2.0,
            base_stop_loss=0.025,
            enabled=True,
        )

        with patch(
            "app.services.auto_strategy.generators.random.tpsl_generator.create_random_tpsl_gene",
            return_value=mock_gene,
        ):
            generator = TPSLGenerator(config)
            result = generator.generate_tpsl_gene()

        assert result.stop_loss_pct == 0.025
        assert result.take_profit_pct == 0.05


class TestTPSLGeneMethodConstraintsEdgeCases:
    """メソッド制約のエッジケース"""

    def test_empty_method_constraints_list(self):
        """空のメソッド制約リスト"""
        from app.services.auto_strategy.generators.random.tpsl_generator import (
            TPSLGenerator,
        )

        config = Mock()
        config.tpsl_method_constraints = []  # 空リスト
        config.tpsl_sl_range = None
        config.tpsl_tp_range = None
        config.tpsl_rr_range = None
        config.tpsl_atr_multiplier_range = None

        generator = TPSLGenerator(config)

        # 空リストの場合、元のメソッドがそのまま使用される
        result = generator.generate_tpsl_gene()
        assert result is not None

    def test_hasattr_check_for_missing_attributes(self):
        """属性が存在しない場合のhasattrチェック"""
        from app.services.auto_strategy.generators.random.tpsl_generator import (
            TPSLGenerator,
        )

        # spec=[]で属性を持たないモックを作成
        config = Mock(spec=[])
        generator = TPSLGenerator(config)

        # 属性がなくてもエラーにならないことを確認
        result = generator.generate_tpsl_gene()
        assert result is not None


