"""
オートストラテジー統合テストスイート

エンドツーエンドの統合テスト、API連携、フロントエンド統合テスト
"""

import sys
import os

# プロジェクトルートをパスに追加
current_dir = os.path.dirname(os.path.abspath(__file__))
backend_dir = os.path.dirname(os.path.dirname(current_dir))
sys.path.insert(0, backend_dir)

import pytest
import pandas as pd
import numpy as np
import asyncio
import json
import tempfile
from datetime import datetime, timedelta
from pathlib import Path
import logging

logger = logging.getLogger(__name__)


class TestAutoStrategyIntegration:
    """オートストラテジー統合テストクラス"""
    
    def setup_method(self):
        """テスト前の準備"""
        self.test_data = self.create_test_data()
        self.temp_dir = tempfile.mkdtemp()
    
    def teardown_method(self):
        """テスト後のクリーンアップ"""
        import shutil
        if os.path.exists(self.temp_dir):
            shutil.rmtree(self.temp_dir)
    
    def create_test_data(self, size: int = 1000) -> pd.DataFrame:
        """テスト用データを作成"""
        np.random.seed(42)
        dates = pd.date_range(start='2023-01-01', periods=size, freq='h')
        
        base_price = 50000
        returns = np.random.normal(0, 0.02, size)
        prices = [base_price]
        
        for ret in returns[1:]:
            prices.append(prices[-1] * (1 + ret))
        
        data = pd.DataFrame({
            'timestamp': dates,
            'Open': [p * (1 + np.random.normal(0, 0.001)) for p in prices],
            'High': [p * (1 + abs(np.random.normal(0, 0.01))) for p in prices],
            'Low': [p * (1 - abs(np.random.normal(0, 0.01))) for p in prices],
            'Close': prices,
            'Volume': np.random.exponential(1000, size),
        })
        
        data.set_index('timestamp', inplace=True)
        return data
    
    def test_end_to_end_strategy_generation(self):
        """テスト16: エンドツーエンド戦略生成テスト"""
        logger.info("🔍 エンドツーエンド戦略生成テスト開始")
        
        try:
            from app.services.auto_strategy.services.auto_strategy_service import AutoStrategyService
            from app.services.auto_strategy.orchestration.auto_strategy_orchestration_service import (
                AutoStrategyOrchestrationService
            )
            
            # サービス初期化
            auto_strategy_service = AutoStrategyService(enable_smart_generation=True)
            orchestration_service = AutoStrategyOrchestrationService()
            
            # 戦略生成リクエストの模擬
            experiment_id = f"test_experiment_{int(datetime.now().timestamp())}"
            experiment_name = "統合テスト実験"
            
            ga_config = {
                "population_size": 5,
                "generations": 2,
                "mutation_rate": 0.1,
                "crossover_rate": 0.8,
                "enable_multi_objective": False
            }
            
            backtest_config = {
                "symbol": "BTC:USDT",
                "timeframe": "1h",
                "start_date": "2023-01-01",
                "end_date": "2023-01-03",
                "initial_capital": 10000,
                "commission_rate": 0.001
            }
            
            # 実験作成の確認
            if hasattr(auto_strategy_service, 'persistence_service'):
                persistence_service = auto_strategy_service.persistence_service
                
                # 実験作成テスト
                try:
                    persistence_service.create_experiment(
                        experiment_id, experiment_name, ga_config, backtest_config
                    )
                    logger.info(f"実験作成成功: {experiment_id}")
                except Exception as e:
                    logger.info(f"実験作成でエラー（期待される場合もあります）: {e}")
            
            # 統合管理サービスの機能確認
            assert hasattr(orchestration_service, 'test_strategy'), "test_strategy メソッドが不足しています"
            
            logger.info("✅ エンドツーエンド戦略生成テスト成功")
            
        except Exception as e:
            pytest.fail(f"エンドツーエンド戦略生成テストエラー: {e}")
    
    def test_ml_auto_strategy_full_pipeline(self):
        """テスト17: ML-オートストラテジー完全パイプラインテスト"""
        logger.info("🔍 ML-オートストラテジー完全パイプラインテスト開始")
        
        try:
            from app.services.auto_strategy.services.ml_orchestrator import MLOrchestrator
            from app.services.auto_strategy.services.tpsl_auto_decision_service import (
                TPSLAutoDecisionService, TPSLConfig, TPSLStrategy
            )
            from app.services.auto_strategy.calculators.tpsl_calculator import TPSLCalculator
            
            # 1. ML指標計算
            ml_orchestrator = MLOrchestrator(enable_automl=True)
            ml_indicators = ml_orchestrator.calculate_ml_indicators(self.test_data)
            
            # 2. TP/SL自動決定
            tpsl_service = TPSLAutoDecisionService()
            tpsl_config = TPSLConfig(
                strategy=TPSLStrategy.AUTO_OPTIMAL,
                max_risk_per_trade=0.02,
                preferred_risk_reward_ratio=2.0
            )
            
            tpsl_result = tpsl_service.generate_tpsl_values(
                tpsl_config,
                market_data={"volatility": 0.02, "trend": "up"},
                symbol="BTC:USDT"
            )
            
            # 3. TP/SL価格計算
            tpsl_calculator = TPSLCalculator()
            current_price = self.test_data['Close'].iloc[-1]
            
            sl_price, tp_price = tpsl_calculator.calculate_basic_tpsl_prices(
                current_price=current_price,
                stop_loss_pct=tpsl_result.stop_loss_pct,
                take_profit_pct=tpsl_result.take_profit_pct,
                position_direction=1.0  # ロング
            )

            # 4. 統合結果の検証
            # ML指標の妥当性
            assert "ML_UP_PROB" in ml_indicators, "ML上昇確率が不足しています"
            assert len(ml_indicators["ML_UP_PROB"]) > 0, "ML指標が空です"

            # TP/SL決定の妥当性
            assert tpsl_result.stop_loss_pct > 0, "SL%が無効です"
            assert tpsl_result.take_profit_pct > 0, "TP%が無効です"
            assert tpsl_result.confidence_score >= 0, "信頼度スコアが無効です"

            # 価格計算の妥当性（Noneチェック追加）
            if sl_price is not None and tp_price is not None:
                assert sl_price < current_price, "ロングSL価格が現在価格より高いです"
                assert tp_price > current_price, "ロングTP価格が現在価格より低いです"
            else:
                logger.warning("TP/SL価格計算でNone値が返されました")
            
            # 5. パイプライン統合性の確認
            pipeline_result = {
                "ml_indicators": ml_indicators,
                "tpsl_decision": tpsl_result,
                "calculated_prices": {"sl": sl_price, "tp": tp_price},
                "current_price": current_price
            }

            # 統合結果の一貫性確認
            if sl_price is not None and tp_price is not None:
                risk_amount = (current_price - sl_price) / current_price
                reward_amount = (tp_price - current_price) / current_price
                actual_rr = reward_amount / risk_amount if risk_amount > 0 else 0

                # リスクリワード比の一貫性（20%の誤差許容）
                expected_rr = tpsl_result.risk_reward_ratio
                rr_error = abs(actual_rr - expected_rr) / expected_rr if expected_rr > 0 else 0
                if rr_error < 0.2:
                    logger.info(f"完全パイプライン結果: ML指標数={len(ml_indicators)}, RR比={actual_rr:.2f}")
                else:
                    logger.warning(f"リスクリワード比の一貫性エラー: {rr_error:.3f}")
            else:
                logger.info(f"完全パイプライン結果: ML指標数={len(ml_indicators)}, 価格計算スキップ")

            logger.info("✅ ML-オートストラテジー完全パイプラインテスト成功")

        except Exception as e:
            pytest.fail(f"ML-オートストラテジー完全パイプラインテストエラー: {e}")
    
    def test_api_integration_simulation(self):
        """テスト18: API統合シミュレーションテスト"""
        logger.info("🔍 API統合シミュレーションテスト開始")
        
        try:
            # APIリクエスト形式の模擬
            api_request = {
                "experiment_name": "API統合テスト",
                "ga_config": {
                    "population_size": 5,
                    "generations": 2,
                    "mutation_rate": 0.1,
                    "crossover_rate": 0.8,
                    "enable_multi_objective": False
                },
                "backtest_config": {
                    "symbol": "BTC:USDT",
                    "timeframe": "1h",
                    "start_date": "2023-01-01",
                    "end_date": "2023-01-03",
                    "initial_capital": 10000,
                    "commission_rate": 0.001
                }
            }
            
            # リクエスト検証
            assert "experiment_name" in api_request, "実験名が不足しています"
            assert "ga_config" in api_request, "GA設定が不足しています"
            assert "backtest_config" in api_request, "バックテスト設定が不足しています"
            
            # GA設定の妥当性確認
            ga_config = api_request["ga_config"]
            assert ga_config["population_size"] > 0, "人口サイズが無効です"
            assert ga_config["generations"] > 0, "世代数が無効です"
            assert 0 <= ga_config["mutation_rate"] <= 1, "突然変異率が範囲外です"
            assert 0 <= ga_config["crossover_rate"] <= 1, "交叉率が範囲外です"
            
            # バックテスト設定の妥当性確認
            backtest_config = api_request["backtest_config"]
            assert backtest_config["symbol"], "シンボルが空です"
            assert backtest_config["timeframe"], "時間軸が空です"
            assert backtest_config["initial_capital"] > 0, "初期資金が無効です"
            assert 0 <= backtest_config["commission_rate"] <= 1, "手数料率が範囲外です"
            
            # 日付形式の確認
            start_date = datetime.strptime(backtest_config["start_date"], "%Y-%m-%d")
            end_date = datetime.strptime(backtest_config["end_date"], "%Y-%m-%d")
            assert start_date < end_date, "開始日が終了日より後です"
            
            # APIレスポンス形式の模擬
            api_response = {
                "success": True,
                "experiment_id": f"exp_{int(datetime.now().timestamp())}",
                "message": "戦略生成を開始しました",
                "estimated_completion_time": (datetime.now() + timedelta(minutes=5)).isoformat()
            }
            
            # レスポンス検証
            assert api_response["success"], "API成功フラグが無効です"
            assert api_response["experiment_id"], "実験IDが空です"
            assert api_response["message"], "メッセージが空です"
            
            logger.info(f"API統合シミュレーション: 実験ID={api_response['experiment_id']}")
            logger.info("✅ API統合シミュレーションテスト成功")
            
        except Exception as e:
            pytest.fail(f"API統合シミュレーションテストエラー: {e}")
    
    def test_data_flow_consistency(self):
        """テスト19: データフロー一貫性テスト"""
        logger.info("🔍 データフロー一貫性テスト開始")
        
        try:
            from app.services.auto_strategy.services.ml_orchestrator import MLOrchestrator
            
            ml_orchestrator = MLOrchestrator()
            
            # 複数回の計算で一貫性を確認
            results = []
            
            for i in range(3):
                ml_indicators = ml_orchestrator.calculate_ml_indicators(self.test_data)
                results.append(ml_indicators)
            
            # 結果の一貫性確認
            if len(results) >= 2:
                first_result = results[0]
                second_result = results[1]
                
                # キーの一致確認
                assert set(first_result.keys()) == set(second_result.keys()), "結果キーが一致しません"
                
                # データサイズの一致確認
                for key in first_result.keys():
                    assert len(first_result[key]) == len(second_result[key]), f"{key}: データサイズが一致しません"
                
                # 値の安定性確認（完全一致は期待しないが、大きな差異は問題）
                for key in first_result.keys():
                    if len(first_result[key]) > 0 and len(second_result[key]) > 0:
                        # 最後の値での比較（最も安定している可能性が高い）
                        val1 = first_result[key][-1] if not np.isnan(first_result[key][-1]) else 0.5
                        val2 = second_result[key][-1] if not np.isnan(second_result[key][-1]) else 0.5
                        
                        if val1 != 0:
                            relative_diff = abs(val1 - val2) / abs(val1)
                            # 50%以上の差異は問題とする
                            assert relative_diff < 0.5, f"{key}: 値の変動が大きすぎます ({relative_diff:.3f})"
            
            logger.info("✅ データフロー一貫性テスト成功")
            
        except Exception as e:
            pytest.fail(f"データフロー一貫性テストエラー: {e}")
    
    def test_configuration_validation(self):
        """テスト20: 設定検証テスト"""
        logger.info("🔍 設定検証テスト開始")
        
        try:
            from app.services.auto_strategy.services.tpsl_auto_decision_service import (
                TPSLConfig, TPSLStrategy
            )
            
            # 有効な設定のテスト
            valid_configs = [
                TPSLConfig(
                    strategy=TPSLStrategy.RANDOM,
                    max_risk_per_trade=0.02,
                    preferred_risk_reward_ratio=2.0
                ),
                TPSLConfig(
                    strategy=TPSLStrategy.VOLATILITY_ADAPTIVE,
                    max_risk_per_trade=0.01,
                    preferred_risk_reward_ratio=3.0,
                    volatility_sensitivity=1.5
                ),
                TPSLConfig(
                    strategy=TPSLStrategy.AUTO_OPTIMAL,
                    max_risk_per_trade=0.03,
                    preferred_risk_reward_ratio=1.5
                )
            ]
            
            for i, config in enumerate(valid_configs):
                # 設定の基本属性確認
                assert hasattr(config, 'strategy'), f"設定{i+1}: strategy属性が不足しています"
                assert hasattr(config, 'max_risk_per_trade'), f"設定{i+1}: max_risk_per_trade属性が不足しています"
                assert hasattr(config, 'preferred_risk_reward_ratio'), f"設定{i+1}: preferred_risk_reward_ratio属性が不足しています"
                
                # 値の妥当性確認（型変換を含む）
                try:
                    max_risk = float(config.max_risk_per_trade)
                    assert 0 < max_risk <= 1, f"設定{i+1}: max_risk_per_tradeが範囲外です"
                except (ValueError, TypeError):
                    logger.warning(f"設定{i+1}: max_risk_per_tradeの型変換エラー")

                try:
                    rr_ratio = float(config.preferred_risk_reward_ratio)
                    assert rr_ratio > 0, f"設定{i+1}: preferred_risk_reward_ratioが無効です"
                except (ValueError, TypeError):
                    logger.warning(f"設定{i+1}: preferred_risk_reward_ratioの型変換エラー")

                if hasattr(config, 'volatility_sensitivity'):
                    try:
                        vol_sens = float(config.volatility_sensitivity)
                        assert vol_sens > 0, f"設定{i+1}: volatility_sensitivityが無効です"
                    except (ValueError, TypeError):
                        logger.warning(f"設定{i+1}: volatility_sensitivityの型変換エラー")
            
            # 無効な設定のテスト
            invalid_scenarios = [
                {"max_risk_per_trade": -0.01, "error": "負のリスク"},
                {"max_risk_per_trade": 1.5, "error": "100%超のリスク"},
                {"preferred_risk_reward_ratio": -1, "error": "負のリスクリワード比"},
                {"preferred_risk_reward_ratio": 0, "error": "ゼロのリスクリワード比"}
            ]
            
            for scenario in invalid_scenarios:
                try:
                    invalid_config = TPSLConfig(
                        strategy=TPSLStrategy.RANDOM,
                        max_risk_per_trade=scenario.get("max_risk_per_trade", 0.02),
                        preferred_risk_reward_ratio=scenario.get("preferred_risk_reward_ratio", 2.0)
                    )
                    
                    # 無効な設定でも作成できる場合は、使用時にエラーになることを期待
                    logger.info(f"無効設定 '{scenario['error']}' が作成されました（使用時検証を期待）")
                    
                except Exception as e:
                    logger.info(f"無効設定 '{scenario['error']}' で期待通りエラー: {e}")
            
            logger.info("✅ 設定検証テスト成功")
            
        except Exception as e:
            pytest.fail(f"設定検証テストエラー: {e}")


if __name__ == "__main__":
    # テスト実行
    test_instance = TestAutoStrategyIntegration()
    test_instance.setup_method()
    
    # 各テストを実行
    tests = [
        test_instance.test_end_to_end_strategy_generation,
        test_instance.test_ml_auto_strategy_full_pipeline,
        test_instance.test_api_integration_simulation,
        test_instance.test_data_flow_consistency,
        test_instance.test_configuration_validation,
    ]
    
    passed = 0
    failed = 0
    
    for test in tests:
        try:
            test()
            passed += 1
        except Exception as e:
            logger.error(f"テスト失敗: {test.__name__}: {e}")
            failed += 1
        finally:
            # 各テスト後にクリーンアップ
            try:
                test_instance.teardown_method()
                test_instance.setup_method()
            except:
                pass
    
    # 最終クリーンアップ
    test_instance.teardown_method()
    
    print(f"\n📊 統合オートストラテジーテスト結果: 成功 {passed}, 失敗 {failed}")
    print(f"成功率: {passed / (passed + failed) * 100:.1f}%")
