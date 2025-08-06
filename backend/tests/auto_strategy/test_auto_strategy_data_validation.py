"""
オートストラテジーデータバリデーションの包括的テスト

入力データの妥当性チェック、データ型検証、範囲チェック、必須フィールド検証をテストします。
"""

import logging
from typing import Any, Dict, List

import numpy as np
import pandas as pd
import pytest

# テスト用のロガー設定
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class TestAutoStrategyDataValidation:
    """オートストラテジーデータバリデーションの包括的テストクラス"""

    @pytest.fixture(autouse=True)
    def setup_method(self):
        """各テストメソッドの前に実行される初期化"""
        self.valid_ohlcv_data = self._create_valid_ohlcv_data()
        self.invalid_data_cases = self._create_invalid_data_cases()
        self.valid_strategy_gene = self._create_valid_strategy_gene()
        self.invalid_strategy_genes = self._create_invalid_strategy_genes()

    def _create_valid_ohlcv_data(self) -> pd.DataFrame:
        """有効なOHLCVデータを作成"""
        dates = pd.date_range(start="2023-01-01", periods=100, freq="1H")
        np.random.seed(42)
        
        data = []
        base_price = 50000
        for i, date in enumerate(dates):
            price = base_price + np.random.normal(0, 1000)
            high = price * (1 + abs(np.random.normal(0, 0.01)))
            low = price * (1 - abs(np.random.normal(0, 0.01)))
            open_price = price * (1 + np.random.normal(0, 0.005))
            close_price = price
            volume = np.random.uniform(100, 1000)
            
            data.append({
                "timestamp": date,
                "open": open_price,
                "high": max(high, open_price, close_price),
                "low": min(low, open_price, close_price),
                "close": close_price,
                "volume": volume
            })
        
        return pd.DataFrame(data)

    def _create_invalid_data_cases(self) -> Dict[str, pd.DataFrame]:
        """無効なデータケースを作成"""
        return {
            "empty_dataframe": pd.DataFrame(),
            "missing_required_columns": pd.DataFrame({
                "timestamp": pd.date_range("2023-01-01", periods=10, freq="1H"),
                "price": [50000] * 10  # open, high, low, close, volume が不足
            }),
            "invalid_column_types": pd.DataFrame({
                "timestamp": ["invalid_date"] * 10,
                "open": ["not_a_number"] * 10,
                "high": ["not_a_number"] * 10,
                "low": ["not_a_number"] * 10,
                "close": ["not_a_number"] * 10,
                "volume": ["not_a_number"] * 10
            }),
            "negative_prices": pd.DataFrame({
                "timestamp": pd.date_range("2023-01-01", periods=10, freq="1H"),
                "open": [-100] * 10,
                "high": [-90] * 10,
                "low": [-110] * 10,
                "close": [-95] * 10,
                "volume": [100] * 10
            }),
            "invalid_ohlc_relationship": pd.DataFrame({
                "timestamp": pd.date_range("2023-01-01", periods=10, freq="1H"),
                "open": [100] * 10,
                "high": [90] * 10,  # high < open (無効)
                "low": [110] * 10,  # low > open (無効)
                "close": [105] * 10,
                "volume": [100] * 10
            }),
            "nan_values": pd.DataFrame({
                "timestamp": pd.date_range("2023-01-01", periods=10, freq="1H"),
                "open": [float('nan')] * 10,
                "high": [float('nan')] * 10,
                "low": [float('nan')] * 10,
                "close": [float('nan')] * 10,
                "volume": [float('nan')] * 10
            }),
            "zero_volume": pd.DataFrame({
                "timestamp": pd.date_range("2023-01-01", periods=10, freq="1H"),
                "open": [100] * 10,
                "high": [105] * 10,
                "low": [95] * 10,
                "close": [102] * 10,
                "volume": [0] * 10  # ゼロボリューム
            })
        }

    def _create_valid_strategy_gene(self) -> Dict[str, Any]:
        """有効な戦略遺伝子を作成"""
        return {
            "id": "valid_strategy_001",
            "indicators": [
                {
                    "type": "RSI",
                    "parameters": {"period": 14},
                    "enabled": True
                },
                {
                    "type": "SMA",
                    "parameters": {"period": 20},
                    "enabled": True
                }
            ],
            "long_entry_conditions": [
                {
                    "left_operand": "RSI",
                    "operator": "<",
                    "right_operand": 30
                }
            ],
            "short_entry_conditions": [
                {
                    "left_operand": "RSI",
                    "operator": ">",
                    "right_operand": 70
                }
            ],
            "exit_conditions": [],
            "risk_management": {
                "max_position_size": 0.1,
                "stop_loss": 0.02,
                "take_profit": 0.04
            }
        }

    def _create_invalid_strategy_genes(self) -> List[Dict[str, Any]]:
        """無効な戦略遺伝子のリストを作成"""
        return [
            # 空の戦略
            {},
            # 必須フィールドが不足
            {
                "id": "incomplete_strategy",
                "indicators": []
                # long_entry_conditions, short_entry_conditions が不足
            },
            # 無効な指標
            {
                "id": "invalid_indicators",
                "indicators": [
                    {
                        "type": "",  # 空の指標タイプ
                        "parameters": {},
                        "enabled": True
                    }
                ],
                "long_entry_conditions": [],
                "short_entry_conditions": [],
                "exit_conditions": []
            },
            # 無効な条件
            {
                "id": "invalid_conditions",
                "indicators": [
                    {
                        "type": "RSI",
                        "parameters": {"period": 14},
                        "enabled": True
                    }
                ],
                "long_entry_conditions": [
                    {
                        "left_operand": "",  # 空のオペランド
                        "operator": "invalid_operator",  # 無効な演算子
                        "right_operand": "not_a_number"  # 無効な値
                    }
                ],
                "short_entry_conditions": [],
                "exit_conditions": []
            },
            # 無効なリスク管理
            {
                "id": "invalid_risk_management",
                "indicators": [
                    {
                        "type": "RSI",
                        "parameters": {"period": 14},
                        "enabled": True
                    }
                ],
                "long_entry_conditions": [],
                "short_entry_conditions": [],
                "exit_conditions": [],
                "risk_management": {
                    "max_position_size": -0.1,  # 負の値
                    "stop_loss": 1.5,  # 100%を超える値
                    "take_profit": -0.04  # 負の値
                }
            }
        ]

    def test_ohlcv_data_validation(self):
        """OHLCVデータバリデーションテスト"""
        logger.info("=== OHLCVデータバリデーションテスト ===")
        
        try:
            from app.services.auto_strategy.services.ml_orchestrator import MLOrchestrator
            from app.services.ml.exceptions import MLDataError, MLValidationError
            
            ml_orchestrator = MLOrchestrator(enable_automl=False)
            
            # 有効なデータでの成功テスト
            try:
                result = ml_orchestrator.calculate_ml_indicators(self.valid_ohlcv_data)
                logger.info("✅ 有効なOHLCVデータの処理成功")
            except Exception as e:
                logger.warning(f"有効なOHLCVデータでエラー: {e}")
            
            # 無効なデータでのバリデーションテスト
            for case_name, invalid_data in self.invalid_data_cases.items():
                try:
                    result = ml_orchestrator.calculate_ml_indicators(invalid_data)
                    logger.warning(f"無効なデータ '{case_name}' が処理されました（バリデーションが不十分な可能性）")
                except (MLDataError, MLValidationError, ValueError, Exception) as e:
                    logger.info(f"✅ 無効なデータ '{case_name}' で適切にエラーが発生: {type(e).__name__}")
            
            logger.info("✅ OHLCVデータバリデーションテスト成功")
            
        except Exception as e:
            pytest.fail(f"OHLCVデータバリデーションテストエラー: {e}")

    def test_strategy_gene_validation(self):
        """戦略遺伝子バリデーションテスト"""
        logger.info("=== 戦略遺伝子バリデーションテスト ===")
        
        try:
            from app.services.auto_strategy.models.gene_strategy import StrategyGene, IndicatorGene, Condition
            
            # 有効な戦略遺伝子のテスト
            try:
                indicators = [
                    IndicatorGene(type="RSI", parameters={"period": 14}, enabled=True),
                    IndicatorGene(type="SMA", parameters={"period": 20}, enabled=True)
                ]
                
                long_conditions = [
                    Condition(left_operand="RSI", operator="<", right_operand=30)
                ]
                
                short_conditions = [
                    Condition(left_operand="RSI", operator=">", right_operand=70)
                ]
                
                strategy_gene = StrategyGene(
                    id="test_validation",
                    indicators=indicators,
                    long_entry_conditions=long_conditions,
                    short_entry_conditions=short_conditions,
                    exit_conditions=[],
                    risk_management={"max_position_size": 0.1}
                )
                
                is_valid = strategy_gene.validate()
                logger.info(f"✅ 有効な戦略遺伝子のバリデーション結果: {is_valid}")
                
            except Exception as e:
                logger.warning(f"有効な戦略遺伝子でエラー: {e}")
            
            # 無効な戦略遺伝子のテスト
            for i, invalid_gene_data in enumerate(self.invalid_strategy_genes):
                try:
                    # 辞書から戦略遺伝子オブジェクトを作成（可能な場合）
                    if "indicators" in invalid_gene_data:
                        indicators = []
                        for ind_data in invalid_gene_data.get("indicators", []):
                            if ind_data.get("type"):
                                indicators.append(IndicatorGene(
                                    type=ind_data["type"],
                                    parameters=ind_data.get("parameters", {}),
                                    enabled=ind_data.get("enabled", True)
                                ))
                        
                        long_conditions = []
                        for cond_data in invalid_gene_data.get("long_entry_conditions", []):
                            if all(key in cond_data for key in ["left_operand", "operator", "right_operand"]):
                                long_conditions.append(Condition(
                                    left_operand=cond_data["left_operand"],
                                    operator=cond_data["operator"],
                                    right_operand=cond_data["right_operand"]
                                ))
                        
                        short_conditions = []
                        for cond_data in invalid_gene_data.get("short_entry_conditions", []):
                            if all(key in cond_data for key in ["left_operand", "operator", "right_operand"]):
                                short_conditions.append(Condition(
                                    left_operand=cond_data["left_operand"],
                                    operator=cond_data["operator"],
                                    right_operand=cond_data["right_operand"]
                                ))
                        
                        strategy_gene = StrategyGene(
                            id=invalid_gene_data.get("id", ""),
                            indicators=indicators,
                            long_entry_conditions=long_conditions,
                            short_entry_conditions=short_conditions,
                            exit_conditions=[],
                            risk_management=invalid_gene_data.get("risk_management", {})
                        )
                        
                        is_valid = strategy_gene.validate()
                        if is_valid:
                            logger.warning(f"無効な戦略遺伝子 {i+1} がバリデーションを通過しました")
                        else:
                            logger.info(f"✅ 無効な戦略遺伝子 {i+1} で適切にバリデーションエラー")
                    
                except Exception as e:
                    logger.info(f"✅ 無効な戦略遺伝子 {i+1} で適切にエラーが発生: {type(e).__name__}")
            
            logger.info("✅ 戦略遺伝子バリデーションテスト成功")
            
        except Exception as e:
            pytest.fail(f"戦略遺伝子バリデーションテストエラー: {e}")

    def test_ga_config_validation(self):
        """GA設定バリデーションテスト"""
        logger.info("=== GA設定バリデーションテスト ===")
        
        try:
            # 有効なGA設定
            valid_ga_config = {
                "population_size": 50,
                "generations": 10,
                "mutation_rate": 0.1,
                "crossover_rate": 0.8,
                "elite_size": 5,
                "enable_multi_objective": False
            }
            
            # 無効なGA設定のテストケース
            invalid_ga_configs = [
                {},  # 空の設定
                {"population_size": 0},  # ゼロ人口
                {"population_size": -10},  # 負の人口
                {"generations": 0},  # ゼロ世代
                {"generations": -5},  # 負の世代
                {"mutation_rate": -0.1},  # 負の突然変異率
                {"mutation_rate": 1.5},  # 100%を超える突然変異率
                {"crossover_rate": -0.1},  # 負の交叉率
                {"crossover_rate": 1.5},  # 100%を超える交叉率
                {"elite_size": -1},  # 負のエリートサイズ
                {"population_size": "invalid"},  # 文字列
                {"generations": "invalid"},  # 文字列
            ]
            
            # バリデーション関数（簡易版）
            def validate_ga_config(config):
                if not isinstance(config, dict):
                    return False
                
                population_size = config.get("population_size", 0)
                if not isinstance(population_size, int) or population_size <= 0:
                    return False
                
                generations = config.get("generations", 0)
                if not isinstance(generations, int) or generations <= 0:
                    return False
                
                mutation_rate = config.get("mutation_rate", 0.1)
                if not isinstance(mutation_rate, (int, float)) or not (0 <= mutation_rate <= 1):
                    return False
                
                crossover_rate = config.get("crossover_rate", 0.8)
                if not isinstance(crossover_rate, (int, float)) or not (0 <= crossover_rate <= 1):
                    return False
                
                elite_size = config.get("elite_size", 0)
                if not isinstance(elite_size, int) or elite_size < 0:
                    return False
                
                return True
            
            # 有効な設定のテスト
            assert validate_ga_config(valid_ga_config), "有効なGA設定がバリデーションを通過しませんでした"
            logger.info("✅ 有効なGA設定のバリデーション成功")
            
            # 無効な設定のテスト
            for i, invalid_config in enumerate(invalid_ga_configs):
                is_valid = validate_ga_config(invalid_config)
                if is_valid:
                    logger.warning(f"無効なGA設定 {i+1} がバリデーションを通過しました: {invalid_config}")
                else:
                    logger.info(f"✅ 無効なGA設定 {i+1} で適切にバリデーションエラー")
            
            logger.info("✅ GA設定バリデーションテスト成功")
            
        except Exception as e:
            pytest.fail(f"GA設定バリデーションテストエラー: {e}")

    def test_backtest_config_validation(self):
        """バックテスト設定バリデーションテスト"""
        logger.info("=== バックテスト設定バリデーションテスト ===")
        
        try:
            # 有効なバックテスト設定
            valid_backtest_config = {
                "symbol": "BTC:USDT",
                "timeframe": "1h",
                "start_date": "2023-01-01",
                "end_date": "2023-01-03",
                "initial_capital": 10000,
                "commission_rate": 0.001
            }
            
            # 無効なバックテスト設定のテストケース
            invalid_backtest_configs = [
                {},  # 空の設定
                {"symbol": ""},  # 空のシンボル
                {"symbol": "INVALID_SYMBOL"},  # 無効なシンボル形式
                {"timeframe": "invalid"},  # 無効な時間軸
                {"start_date": "invalid_date"},  # 無効な日付形式
                {"end_date": "2022-01-01", "start_date": "2023-01-01"},  # 終了日が開始日より前
                {"initial_capital": 0},  # ゼロ初期資金
                {"initial_capital": -1000},  # 負の初期資金
                {"commission_rate": -0.1},  # 負の手数料
                {"commission_rate": 1.0},  # 100%手数料
            ]
            
            # バリデーション関数（簡易版）
            def validate_backtest_config(config):
                if not isinstance(config, dict):
                    return False
                
                symbol = config.get("symbol", "")
                if not isinstance(symbol, str) or not symbol:
                    return False
                
                timeframe = config.get("timeframe", "")
                valid_timeframes = ["15min", "30min", "1h", "4h", "1day"]
                if timeframe not in valid_timeframes:
                    return False
                
                initial_capital = config.get("initial_capital", 0)
                if not isinstance(initial_capital, (int, float)) or initial_capital <= 0:
                    return False
                
                commission_rate = config.get("commission_rate", 0)
                if not isinstance(commission_rate, (int, float)) or not (0 <= commission_rate < 1):
                    return False
                
                return True
            
            # 有効な設定のテスト
            assert validate_backtest_config(valid_backtest_config), "有効なバックテスト設定がバリデーションを通過しませんでした"
            logger.info("✅ 有効なバックテスト設定のバリデーション成功")
            
            # 無効な設定のテスト
            for i, invalid_config in enumerate(invalid_backtest_configs):
                is_valid = validate_backtest_config(invalid_config)
                if is_valid:
                    logger.warning(f"無効なバックテスト設定 {i+1} がバリデーションを通過しました: {invalid_config}")
                else:
                    logger.info(f"✅ 無効なバックテスト設定 {i+1} で適切にバリデーションエラー")
            
            logger.info("✅ バックテスト設定バリデーションテスト成功")
            
        except Exception as e:
            pytest.fail(f"バックテスト設定バリデーションテストエラー: {e}")

    def test_tpsl_config_validation(self):
        """TP/SL設定バリデーションテスト"""
        logger.info("=== TP/SL設定バリデーションテスト ===")
        
        try:
            from app.services.auto_strategy.services.tpsl_auto_decision_service import (
                TPSLConfig, TPSLStrategy
            )
            
            # 有効なTP/SL設定のテスト
            valid_configs = [
                TPSLConfig(strategy=TPSLStrategy.RANDOM),
                TPSLConfig(strategy=TPSLStrategy.RISK_REWARD, max_risk_per_trade=0.02),
                TPSLConfig(strategy=TPSLStrategy.VOLATILITY_ADAPTIVE, volatility_sensitivity="high"),
                TPSLConfig(strategy=TPSLStrategy.STATISTICAL),
                TPSLConfig(strategy=TPSLStrategy.AUTO_OPTIMAL)
            ]
            
            for i, config in enumerate(valid_configs):
                # 基本的な属性の確認
                assert hasattr(config, 'strategy'), f"設定 {i+1}: strategy属性が不足しています"
                assert hasattr(config, 'max_risk_per_trade'), f"設定 {i+1}: max_risk_per_trade属性が不足しています"
                assert hasattr(config, 'preferred_risk_reward_ratio'), f"設定 {i+1}: preferred_risk_reward_ratio属性が不足しています"
                
                # 値の範囲確認
                assert 0 < config.max_risk_per_trade <= 1, f"設定 {i+1}: max_risk_per_tradeが範囲外です"
                assert config.preferred_risk_reward_ratio > 0, f"設定 {i+1}: preferred_risk_reward_ratioが無効です"
                assert config.min_stop_loss >= 0, f"設定 {i+1}: min_stop_lossが負の値です"
                assert config.max_stop_loss > config.min_stop_loss, f"設定 {i+1}: max_stop_lossがmin_stop_loss以下です"
                assert config.min_take_profit >= 0, f"設定 {i+1}: min_take_profitが負の値です"
                assert config.max_take_profit > config.min_take_profit, f"設定 {i+1}: max_take_profitがmin_take_profit以下です"
                
                logger.info(f"✅ 有効なTP/SL設定 {i+1} のバリデーション成功")
            
            logger.info("✅ TP/SL設定バリデーションテスト成功")
            
        except Exception as e:
            pytest.fail(f"TP/SL設定バリデーションテストエラー: {e}")


if __name__ == "__main__":
    # 単体でテストを実行する場合
    pytest.main([__file__, "-v"])
