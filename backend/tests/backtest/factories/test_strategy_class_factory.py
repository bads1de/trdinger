import pytest
from unittest.mock import MagicMock, patch
from app.services.backtest.factories.strategy_class_factory import StrategyClassFactory, StrategyClassCreationError
from app.services.backtest.factories.auto_strategy_loader import AutoStrategyLoaderError

class TestStrategyClassFactory:
    @pytest.fixture
    def factory(self):
        return StrategyClassFactory()

    def test_is_auto_strategy(self, factory):
        assert factory._is_auto_strategy({"strategy_gene": {}}) is True
        assert factory._is_auto_strategy({"parameters": {"strategy_gene": {}}}) is True
        assert factory._is_auto_strategy({"strategy_type": "custom"}) is False

    def test_create_strategy_class_auto_strategy(self, factory):
        config = {"strategy_gene": {"id": "test"}}
        mock_class = MagicMock()
        
        with patch.object(factory._auto_strategy_loader, 'create_auto_strategy_class', return_value=mock_class) as mock_loader:
            result = factory.create_strategy_class(config)
            
            assert result == mock_class
            mock_loader.assert_called_once_with(config)

    def test_create_strategy_class_unsupported(self, factory):
        config = {"strategy_type": "invalid"}
        
        with pytest.raises(StrategyClassCreationError) as excinfo:
            factory.create_strategy_class(config)
        
        assert "サポートされていない戦略タイプ" in str(excinfo.value)

    def test_create_strategy_class_loader_error(self, factory):
        config = {"strategy_gene": {}}
        
        with patch.object(factory._auto_strategy_loader, 'create_auto_strategy_class', side_effect=AutoStrategyLoaderError("Load Fail")):
            with pytest.raises(StrategyClassCreationError) as excinfo:
                factory.create_strategy_class(config)
            
            assert "オートストラテジーの生成に失敗" in str(excinfo.value)

    def test_get_strategy_parameters_auto(self, factory):
        config = {"strategy_gene": {"id": "test"}}
        mock_gene = MagicMock()
        
        with patch.object(factory._auto_strategy_loader, 'load_strategy_gene', return_value=mock_gene):
            params = factory.get_strategy_parameters(config)
            
            assert params == {"strategy_gene": mock_gene}

    def test_get_strategy_parameters_normal(self, factory):
        config = {"strategy_type": "normal", "parameters": {"p1": 1}}
        
        params = factory.get_strategy_parameters(config)
        
        assert params == {"p1": 1}

    def test_get_strategy_parameters_error(self, factory):
        config = {"strategy_gene": {}}
        
        with patch.object(factory._auto_strategy_loader, 'load_strategy_gene', side_effect=Exception("Error")):
            params = factory.get_strategy_parameters(config)
            
            assert params == {}
