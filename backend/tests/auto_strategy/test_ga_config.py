import pytest

from app.services.auto_strategy.config.ga import GASettings


class TestGAConfig:
    """GASettings（旧GAConfig）のテスト"""

    def test_default_values(self):
        """デフォルト値が正しく設定されることを確認"""
        config = GASettings()
        assert config.population_size > 0
        assert config.generations > 0
        assert config.crossover_rate > 0
        assert config.mutation_rate > 0

    def test_custom_values(self):
        """カスタム値を設定できることを確認"""
        config = GASettings(
            population_size=200, generations=100, crossover_rate=0.9, mutation_rate=0.2
        )
        assert config.population_size == 200
        assert config.generations == 100
        assert config.crossover_rate == 0.9
        assert config.mutation_rate == 0.2

    def test_serialize_deserialize(self):
        """シリアライズとデシリアライズが正しく動作することを確認"""
        original = GASettings(population_size=150, generations=75)
        data = original.to_dict()
        assert data["population_size"] == 150
        assert data["generations"] == 75

        restored = GASettings(**data)
        assert restored.population_size == 150
        assert restored.generations == 75
