"""
TPSLService のテスト
"""

import pytest
from unittest.mock import MagicMock, patch

from app.services.auto_strategy.tpsl.tpsl_service import TPSLService
from app.services.auto_strategy.models.tpsl_gene import TPSLGene
from app.services.auto_strategy.config.enums import TPSLMethod
from app.services.auto_strategy.models import TPSLResult


class TestTPSLService:
    """TPSLService のテストクラス"""

    @pytest.fixture
    def service(self):
        return TPSLService()

    @pytest.fixture
    def mock_market_data(self):
        return {"atr": 100.0, "current_price": 50000.0}

    def test_calculate_tpsl_prices_fixed_percentage(self, service):
        """固定割合方式での計算テスト"""
        current_price = 10000.0
        # 3% SL, 6% TP
        gene = TPSLGene(
            method=TPSLMethod.FIXED_PERCENTAGE,
            stop_loss_pct=0.03,
            take_profit_pct=0.06,
            enabled=True,
        )

        # Long
        sl, tp = service.calculate_tpsl_prices(
            current_price, tpsl_gene=gene, position_direction=1.0
        )
        assert sl == 9700.0  # 10000 * (1 - 0.03)
        assert tp == 10600.0  # 10000 * (1 + 0.06)

        # Short
        sl, tp = service.calculate_tpsl_prices(
            current_price, tpsl_gene=gene, position_direction=-1.0
        )
        assert sl == 10300.0  # 10000 * (1 + 0.03)
        assert tp == 9400.0  # 10000 * (1 - 0.06)

    def test_calculate_tpsl_prices_risk_reward(self, service, mock_market_data):
        """リスクリワード方式での計算テスト"""
        current_price = 50000.0

        gene = TPSLGene(
            method=TPSLMethod.RISK_REWARD_RATIO,
            stop_loss_pct=0.02,
            base_stop_loss=0.02,
            risk_reward_ratio=2.0,
            enabled=True,
        )

        # Long
        sl, tp = service.calculate_tpsl_prices(
            current_price,
            tpsl_gene=gene,
            market_data=mock_market_data,
            position_direction=1.0,
        )
        # 50000 * 0.98 = 49000.0
        assert sl == 49000.0
        # 50000 * 1.04 = 52000.0
        assert tp == 52000.0

    def test_calculate_tpsl_prices_legacy_fallback(self, service):
        """Geneなし（従来パラメータ）での計算テスト"""
        current_price = 10000.0

        sl, tp = service.calculate_tpsl_prices(
            current_price,
            stop_loss_pct=0.05,
            take_profit_pct=0.10,
            position_direction=1.0,
        )

        assert sl == 9500.0
        assert tp == 11000.0

    def test_calculate_tpsl_prices_disabled_gene(self, service):
        """無効なGeneが渡された場合のフォールバック（レガシーパラメータ使用）テスト"""
        current_price = 10000.0
        gene = TPSLGene(enabled=False)

        # Geneは無効だがレガシーパラメータがある
        sl, tp = service.calculate_tpsl_prices(
            current_price,
            tpsl_gene=gene,
            stop_loss_pct=0.02,
            take_profit_pct=0.04,
            position_direction=1.0,
        )

        assert sl == 9800.0
        assert tp == 10400.0

    def test_calculate_tpsl_prices_total_fallback(self, service):
        """Geneもレガシーパラメータもない場合の挙動テスト

        パラメータがない場合は計算不能としてNoneが返されるのが正しい仕様。
        （例外が発生しない限りsafe_operationのフォールバックは発動しない）
        """
        current_price = 10000.0

        sl, tp = service.calculate_tpsl_prices(current_price, position_direction=1.0)

        assert sl is None
        assert tp is None

    def test_calculate_from_gene_unknown_method(self, service):
        """未知のメソッドの場合のフォールバックテスト"""
        gene = TPSLGene(enabled=True)
        gene.method = "UNKNOWN_METHOD"

        current_price = 10000.0

        # ここはコード内で明示的に _calculate_fallback を呼んでいるのでデフォルト値が返るはず
        sl, tp = service.calculate_tpsl_prices(
            current_price, tpsl_gene=gene, position_direction=1.0
        )

        assert sl == 9700.0
        assert tp == 10600.0

    def test_calculate_advanced_tpsl_prices(self, service):
        """リスク管理辞書による高度なTP/SL計算（RR比調整）のテスト"""
        current_price = 10000.0
        risk_management = {"_tpsl_strategy": "risk_reward", "_risk_reward_ratio": 3.0}

        sl, tp = service.calculate_tpsl_prices(
            current_price,
            stop_loss_pct=0.05,
            take_profit_pct=0.10,
            risk_management=risk_management,
            position_direction=1.0,
        )

        assert sl == 9500.0
        assert tp == 11500.0

    def test_validate_price_and_percentage(self, service):
        """バリデーションメソッドのテスト"""
        assert service._validate_price(100.0) is True
        assert service._validate_price(-1.0) is False
        assert service._validate_price(0) is False

        assert service._validate_percentage(0.1, "SL") is True
        assert service._validate_percentage(-0.1, "SL") is False


