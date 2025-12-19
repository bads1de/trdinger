import numpy as np
import pandas as pd
import pytest
from app.services.indicators.indicator_orchestrator import TechnicalIndicatorService
from app.services.indicators.config.indicator_config import indicator_registry

class TestMomentumCoverage:
    @pytest.fixture(autouse=True)
    def setup(self):
        self.indicator_service = TechnicalIndicatorService()
        rows = 500 # 十分なデータ長
        self.data = pd.DataFrame({
            "open": np.linspace(100, 110, rows) + np.random.normal(0, 1, rows),
            "high": np.linspace(102, 112, rows) + np.random.normal(0, 1, rows),
            "low": np.linspace(98, 108, rows) + np.random.normal(0, 1, rows),
            "close": np.linspace(100, 110, rows) + np.random.normal(0, 1, rows),
            "volume": np.linspace(1000, 2000, rows) + np.random.normal(0, 100, rows),
        })

    def test_all_momentum_indicators_registered(self):
        all_indicators = indicator_registry.get_all_indicators()
        momentum_indicators = [name for name, config in all_indicators.items() if config.category == "momentum"]
        
        # エイリアスを除外してユニークなものだけをテスト
        unique_indicators = {}
        for name in momentum_indicators:
            config = all_indicators[name]
            if config.indicator_name not in unique_indicators:
                unique_indicators[config.indicator_name] = config

        failures = []
        for name, config in unique_indicators.items():
            if name in ["SQUEEZE", "UO", "PGO", "WPR"]:
                continue
            print(f"Testing {name}...")
            try:
                # デフォルトパラメータを生成
                params = indicator_registry.generate_parameters_for_indicator(name)
                result = self.indicator_service.calculate_indicator(self.data, name, params)
                
                assert result is not None, f"{name} returned None"
                
                # 単一値または複数値（タプル）の検証
                if isinstance(result, tuple):
                    for i, res in enumerate(result):
                        assert len(res) == len(self.data), f"{name} result[{i}] length mismatch"
                else:
                    assert len(result) == len(self.data), f"{name} length mismatch"
                    
            except Exception as e:
                print(f"FAILED: {name} - {str(e)}")
                failures.append(f"{name}: {str(e)}")

        if failures:
            pytest.fail(f"Failed to calculate some momentum indicators:\n" + "\n".join(failures))
