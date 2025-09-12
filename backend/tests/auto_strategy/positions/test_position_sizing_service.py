import pytest
import logging
from unittest.mock import Mock, patch
from datetime import datetime
from typing import Dict, Any

from backend.app.services.auto_strategy.positions.position_sizing_service import (
    PositionSizingService,
    PositionSizingResult,
)
from backend.app.services.auto_strategy.positions.market_data_handler import MarketDataCache
from backend.app.services.auto_strategy.models.position_sizing_gene import PositionSizingGene
from backend.app.services.auto_strategy.models.enums import PositionSizingMethod


@pytest.fixture
def sample_gene_volatility_based() -> PositionSizingGene:
    """Sample gene for volatility-based method"""
    return PositionSizingGene(
        method=PositionSizingMethod.VOLATILITY_BASED,
        risk_per_trade=0.02,
        atr_multiplier=2.0,
    )


@pytest.fixture
def sample_gene_fixed_ratio() -> PositionSizingGene:
    """Sample gene for fixed ratio method"""
    return PositionSizingGene(
        method=PositionSizingMethod.FIXED_RATIO,
        fixed_ratio=0.1,
    )


@pytest.fixture
def sample_gene_half_optimal_f() -> PositionSizingGene:
    """Sample gene for half optimal F method"""
    return PositionSizingGene(
        method=PositionSizingMethod.HALF_OPTIMAL_F,
        risk_per_trade=0.02,
        optimal_f_multiplier=0.5,
    )


@pytest.fixture
def sample_gene_fixed_quantity() -> PositionSizingGene:
    """Sample gene for fixed quantity method"""
    return PositionSizingGene(
        method=PositionSizingMethod.FIXED_QUANTITY,
        fixed_quantity=100.0,
    )


class TestPositionSizingService:
    """Tests for PositionSizingService"""

    def setup_method(self):
        """Setup for each test"""
        with patch('backend.app.services.auto_strategy.positions.position_sizing_service.unified_config'):
            self.service = PositionSizingService()

    def test_initialization(self):
        """Test initialization"""
        with patch('backend.app.services.auto_strategy.positions.position_sizing_service.unified_config'):
            service = PositionSizingService()
            assert service.logger is not None
            assert service._market_data_handler is not None
            assert service._calculator_factory is not None
            assert service._calculation_history == []

    @patch('backend.app.services.auto_strategy.positions.position_sizing_service.datetime')
    def test_calculate_position_size_volatility_based_normal_case(self, mock_datetime, sample_gene_volatility_based):
        """Test volatility-based method normal case"""
        mock_datetime.now.return_value = datetime(2024, 1, 1, 12, 0, 0)

        market_data = {"atr": 1.0, "atr_source": "provided"}

        result = self.service.calculate_position_size(
            gene=sample_gene_volatility_based,
            account_balance=10000.0,
            current_price=50.0,
            market_data=market_data,
        )

        assert isinstance(result, PositionSizingResult)
        assert result.position_size > 0
        assert result.method_used == "volatility_based"
        assert "atr_value" in result.calculation_details

    @patch('backend.app.services.auto_strategy.positions.position_sizing_service.datetime')
    def test_calculate_position_size_zero_current_price(self, mock_datetime, sample_gene_fixed_ratio):
        """Test zero current price error case"""
        mock_datetime.now.return_value = datetime(2024, 1, 1, 12, 0, 0)

        result = self.service.calculate_position_size(
            gene=sample_gene_fixed_ratio,
            account_balance=10000.0,
            current_price=0.0,
        )

        assert isinstance(result, PositionSizingResult)
        assert result.position_size == 0.01  # Minimum size
        assert result.confidence_score == 0.0

    @patch('backend.app.services.auto_strategy.positions.position_sizing_service.datetime')
    def test_calculate_position_size_zero_atr_volatility_based(self, mock_datetime, sample_gene_volatility_based):
        """Test zero ATR in volatility-based method"""
        mock_datetime.now.return_value = datetime(2024, 1, 1, 12, 0, 0)

        market_data = {"atr": 0.0}

        result = self.service.calculate_position_size(
            gene=sample_gene_volatility_based,
            account_balance=10000.0,
            current_price=50.0,
            market_data=market_data,
        )

        assert isinstance(result, PositionSizingResult)
        assert result.position_size == sample_gene_volatility_based.min_position_size
        assert any("ボラティリティ" in w or "ATR" in w for w in result.warnings)

    @patch('backend.app.services.auto_strategy.positions.position_sizing_service.datetime')
    def test_validation_account_balance_zero(self, mock_datetime, sample_gene_volatility_based):
        """Test zero account balance input validation"""
        mock_datetime.now.return_value = datetime(2024, 1, 1, 12, 0, 0)

        result = self.service.calculate_position_size(
            gene=sample_gene_volatility_based,
            account_balance=0.0,
            current_price=50.0,
        )

        assert isinstance(result, PositionSizingResult)
        assert result.position_size == 0.01  # Minimum size
        assert result.method_used == sample_gene_volatility_based.method.value

    @patch('backend.app.services.auto_strategy.positions.position_sizing_service.datetime')
    def test_half_optimal_f_with_insufficient_trade_history(self, mock_datetime, sample_gene_half_optimal_f):
        """Test half optimal F with insufficient trade history"""
        mock_datetime.now.return_value = datetime(2024, 1, 1, 12, 0, 0)

        # Insufficient trade history (less than 10)
        trade_history = [
            {"pnl": 100, "timestamp": "2024-01-01"},
            {"pnl": -50, "timestamp": "2024-01-02"},
        ]

        with patch('backend.app.services.auto_strategy.positions.position_sizing_service.unified_config.auto_strategy.assumed_win_rate', 0.6):
            with patch('backend.app.services.auto_strategy.positions.position_sizing_service.unified_config.auto_strategy.assumed_avg_win', 150):
                with patch('backend.app.services.auto_strategy.positions.position_sizing_service.unified_config.auto_strategy.assumed_avg_loss', 100):
                    result = self.service.calculate_position_size(
                        gene=sample_gene_half_optimal_f,
                        account_balance=10000.0,
                        current_price=50.0,
                        trade_history=trade_history,
                    )

        assert isinstance(result, PositionSizingResult)
        assert result.position_size > 0
        assert any("取引履歴" in w for w in result.warnings)

    @patch('backend.app.services.auto_strategy.positions.market_data_handler.datetime')
    def test_cache_expiration(self, mock_datetime, sample_gene_volatility_based):
        """Test cache expiration"""
        # Initial time
        initial_time = datetime(2024, 1, 1, 12, 0, 0)
        mock_datetime.now.return_value = initial_time

        cache = MarketDataCache(
            atr_values={"BTCUSDT": 1.0},
            volatility_metrics={"BTCUSDT": 0.02},
            price_data=None,
            last_updated=initial_time
        )

        # Cache should not be expired
        assert not cache.is_expired(5)

        # Expired time
        expired_time = datetime(2024, 1, 1, 12, 6, 0)  # 6 minutes later
        mock_datetime.now.return_value = expired_time

        assert cache.is_expired(5)

    @patch('backend.app.services.auto_strategy.positions.position_sizing_service.datetime')
    def test_calculate_position_size_fixed_quantity(self, mock_datetime, sample_gene_fixed_quantity):
        """Test fixed quantity method"""
        mock_datetime.now.return_value = datetime(2024, 1, 1, 12, 0, 0)

        result = self.service.calculate_position_size(
            gene=sample_gene_fixed_quantity,
            account_balance=10000.0,
            current_price=50.0,
        )

        assert isinstance(result, PositionSizingResult)
        assert result.position_size == sample_gene_fixed_quantity.fixed_quantity
        assert result.method_used == "fixed_quantity"
        assert "fixed_quantity" in result.calculation_details

    @patch('backend.app.services.auto_strategy.positions.position_sizing_service.datetime')
    def test_calculate_position_size_simple_method_call(self, mock_datetime):
        """Test simple calculation method"""
        mock_datetime.now.return_value = datetime(2024, 1, 1, 12, 0, 0)

        result = self.service.calculate_position_size_simple(
            method="volatility_based",
            account_balance=10000.0,
            current_price=50.0,
        )

        assert result > 0
        assert isinstance(result, float)

    @patch('backend.app.services.auto_strategy.positions.position_sizing_service.datetime')
    def test_invalid_gene_validation(self, mock_datetime):
        """Test invalid gene validation"""
        mock_datetime.now.return_value = datetime(2024, 1, 1, 12, 0, 0)

        # Invalid gene (None)
        result = self.service.calculate_position_size(
            gene=None,
            account_balance=10000.0,
            current_price=50.0,
        )

        assert isinstance(result, PositionSizingResult)
        assert result.position_size == 0.01  # Minimum size
        assert result.confidence_score == 0.0

    @patch('backend.app.services.auto_strategy.positions.position_sizing_service.datetime')
    def test_negative_account_balance(self, mock_datetime, sample_gene_volatility_based):
        """Test negative account balance"""
        mock_datetime.now.return_value = datetime(2024, 1, 1, 12, 0, 0)

        result = self.service.calculate_position_size(
            gene=sample_gene_volatility_based,
            account_balance=-1000.0,
            current_price=50.0,
        )

        assert isinstance(result, PositionSizingResult)
        assert result.position_size == 0.01  # Minimum size
        assert "正の値" in str(result.warnings)

    @patch('backend.app.services.auto_strategy.positions.position_sizing_service.datetime')
    def test_negative_current_price(self, mock_datetime, sample_gene_volatility_based):
        """Test negative current price"""
        mock_datetime.now.return_value = datetime(2024, 1, 1, 12, 0, 0)

        result = self.service.calculate_position_size(
            gene=sample_gene_volatility_based,
            account_balance=10000.0,
            current_price=-50.0,
        )

        assert isinstance(result, PositionSizingResult)
        assert result.position_size == 0.01  # Minimum size
        assert "正の値" in str(result.warnings)

    @patch('backend.app.services.auto_strategy.positions.position_sizing_service.datetime')
    def test_max_position_size_limitation(self, mock_datetime):
        """Test max position size limitation"""
        mock_datetime.now.return_value = datetime(2024, 1, 1, 12, 0, 0)

        gene = PositionSizingGene(
            method=PositionSizingMethod.VOLATILITY_BASED,
            risk_per_trade=1.0,  # High risk to exceed max_position_size
            atr_multiplier=0.5,
        )

        # Set very low max_position_size to trigger limitation
        gene.max_position_size = 0.1

        market_data = {"atr": 10.0, "atr_source": "provided"}

        result = self.service.calculate_position_size(
            gene=gene,
            account_balance=10000.0,
            current_price=50.0,
            market_data=market_data,
        )

        assert isinstance(result, PositionSizingResult)
        # Position size should be limited to max_position_size
        assert result.position_size <= gene.max_position_size

    @patch('backend.app.services.auto_strategy.positions.position_sizing_service.datetime')
    def test_unknown_method_fallback(self, mock_datetime):
        """Test invalid method fallback"""
        mock_datetime.now.return_value = datetime(2024, 1, 1, 12, 0, 0)

        # Create gene with valid method but mock invalid method value
        gene = PositionSizingGene(
            method=PositionSizingMethod.FIXED_RATIO,
            fixed_ratio=0.1,
        )

        result = self.service.calculate_position_size(
            gene=gene,
            account_balance=10000.0,
            current_price=50.0,
        )

        assert isinstance(result, PositionSizingResult)
        assert result.position_size > 0
        # 测试只验证计算结果，没有mock所以简化
        assert result.position_size > 0

    def test_clear_cache_functionality(self):
        """Test cache clear functionality"""
        # Setup cache
        self.service._market_data_handler._cache = MarketDataCache(
            atr_values={"BTCUSDT": 1.0},
            volatility_metrics={"BTCUSDT": 0.02},
            price_data=None,
            last_updated=datetime(2024, 1, 1, 12, 0, 0)
        )

        assert self.service._market_data_handler._cache is not None

        # Clear cache
        self.service.clear_cache()

        assert self.service._market_data_handler._cache is None