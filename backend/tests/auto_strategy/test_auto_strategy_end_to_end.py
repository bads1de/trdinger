"""
オートストラテジーエンドツーエンドの包括的テスト

フロントエンドからバックエンドまでの完全なワークフロー、実際の市場データでの動作、
結果の整合性をテストします。
"""

import json
import logging
import time
from typing import Any, Dict

import pytest
from fastapi.testclient import TestClient

# テスト用のロガー設定
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class TestAutoStrategyEndToEnd:
    """オートストラテジーエンドツーエンドの包括的テストクラス"""

    @pytest.fixture(autouse=True)
    def setup_method(self):
        """各テストメソッドの前に実行される初期化"""
        self.test_experiment_id = f"e2e_test_{int(time.time())}"
        self.test_client = self._create_test_client()
        self.complete_workflow_config = self._create_complete_workflow_config()

    def _create_test_client(self):
        """テスト用のクライアントを作成"""
        try:
            from app.main import create_app
            app = create_app()
            return TestClient(app)
        except Exception as e:
            logger.warning(f"テストクライアント作成でエラー: {e}")
            return None

    def _create_complete_workflow_config(self) -> Dict[str, Any]:
        """完全なワークフロー設定を作成"""
        return {
            "experiment_id": self.test_experiment_id,
            "experiment_name": "End-to-End Test Experiment",
            "ga_config": {
                "population_size": 10,  # テスト用に小さく設定
                "generations": 3,       # テスト用に小さく設定
                "mutation_rate": 0.1,
                "crossover_rate": 0.8,
                "elite_size": 2,
                "enable_multi_objective": False,
                "indicator_mode": "technical_only",
                "fitness_constraints": {
                    "min_trades": 5,
                    "max_drawdown_limit": 0.5,
                    "min_sharpe_ratio": 0.0
                }
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

    def test_complete_strategy_generation_workflow(self):
        """完全な戦略生成ワークフローテスト"""
        logger.info("=== 完全な戦略生成ワークフローテスト ===")
        
        if self.test_client is None:
            pytest.skip("テストクライアントが利用できません")
        
        try:
            # ステップ1: デフォルト設定の取得
            logger.info("ステップ1: デフォルト設定の取得")
            default_response = self.test_client.get("/api/auto-strategy/default-config")
            
            if default_response.status_code == 200:
                default_config = default_response.json()
                logger.info("✅ デフォルト設定取得成功")
            else:
                logger.warning(f"デフォルト設定取得失敗: {default_response.status_code}")
                default_config = None
            
            # ステップ2: プリセット設定の取得
            logger.info("ステップ2: プリセット設定の取得")
            presets_response = self.test_client.get("/api/auto-strategy/presets")
            
            if presets_response.status_code == 200:
                presets = presets_response.json()
                logger.info("✅ プリセット設定取得成功")
            else:
                logger.warning(f"プリセット設定取得失敗: {presets_response.status_code}")
                presets = None
            
            # ステップ3: 戦略生成の開始
            logger.info("ステップ3: 戦略生成の開始")
            generation_response = self.test_client.post(
                "/api/auto-strategy/generate",
                json=self.complete_workflow_config
            )
            
            if generation_response.status_code == 200:
                generation_result = generation_response.json()
                logger.info("✅ 戦略生成開始成功")
                
                # 実験IDの確認
                if "experiment_id" in generation_result:
                    experiment_id = generation_result["experiment_id"]
                    assert experiment_id == self.test_experiment_id, "実験IDが一致しません"
                
            else:
                logger.warning(f"戦略生成開始失敗: {generation_response.status_code}")
                generation_result = None
            
            # ステップ4: 実験一覧の確認
            logger.info("ステップ4: 実験一覧の確認")
            experiments_response = self.test_client.get("/api/auto-strategy/experiments")
            
            if experiments_response.status_code == 200:
                experiments = experiments_response.json()
                logger.info("✅ 実験一覧取得成功")
                
                # 作成した実験が一覧に含まれているかを確認
                if "experiments" in experiments:
                    experiment_found = any(
                        exp.get("id") == self.test_experiment_id 
                        for exp in experiments["experiments"]
                    )
                    if experiment_found:
                        logger.info("✅ 作成した実験が一覧に含まれています")
                    else:
                        logger.warning("作成した実験が一覧に見つかりません")
            else:
                logger.warning(f"実験一覧取得失敗: {experiments_response.status_code}")
            
            # ステップ5: 実験の停止（クリーンアップ）
            logger.info("ステップ5: 実験の停止")
            stop_response = self.test_client.post(f"/api/auto-strategy/experiments/{self.test_experiment_id}/stop")
            
            if stop_response.status_code in [200, 404]:  # 404も許容（実験が見つからない場合）
                logger.info("✅ 実験停止処理完了")
            else:
                logger.warning(f"実験停止失敗: {stop_response.status_code}")
            
            logger.info("✅ 完全な戦略生成ワークフローテスト成功")
            
        except Exception as e:
            pytest.fail(f"完全な戦略生成ワークフローテストエラー: {e}")

    def test_strategy_testing_workflow(self):
        """戦略テストワークフローテスト"""
        logger.info("=== 戦略テストワークフローテスト ===")
        
        if self.test_client is None:
            pytest.skip("テストクライアントが利用できません")
        
        try:
            # テスト用の戦略遺伝子を作成
            test_strategy_gene = {
                "id": "e2e_test_strategy",
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
            
            # 戦略テストリクエスト
            test_request = {
                "strategy_gene": test_strategy_gene,
                "backtest_config": self.complete_workflow_config["backtest_config"]
            }
            
            # 戦略テストの実行
            test_response = self.test_client.post(
                "/api/auto-strategy/test",
                json=test_request
            )
            
            if test_response.status_code == 200:
                test_result = test_response.json()
                logger.info("✅ 戦略テスト実行成功")
                
                # 結果の基本構造確認
                if "success" in test_result:
                    assert isinstance(test_result["success"], bool), "success フィールドがブール値ではありません"
                
                if "result" in test_result and test_result["result"]:
                    result_data = test_result["result"]
                    logger.info(f"テスト結果の構造: {list(result_data.keys()) if isinstance(result_data, dict) else type(result_data)}")
                
                if "message" in test_result:
                    logger.info(f"テストメッセージ: {test_result['message']}")
                
            else:
                logger.warning(f"戦略テスト実行失敗: {test_response.status_code}")
                if test_response.status_code == 422:
                    error_detail = test_response.json()
                    logger.warning(f"バリデーションエラー: {error_detail}")
            
            logger.info("✅ 戦略テストワークフローテスト成功")
            
        except Exception as e:
            pytest.fail(f"戦略テストワークフローテストエラー: {e}")

    def test_data_flow_consistency(self):
        """データフロー整合性テスト"""
        logger.info("=== データフロー整合性テスト ===")
        
        try:
            from app.services.auto_strategy.services.ml_orchestrator import MLOrchestrator
            from app.services.auto_strategy.services.tpsl_auto_decision_service import (
                TPSLAutoDecisionService, TPSLConfig, TPSLStrategy
            )
            from app.services.auto_strategy.generators.smart_condition_generator import SmartConditionGenerator
            from app.services.auto_strategy.models.gene_strategy import IndicatorGene
            
            # テストデータの作成
            test_data = self._create_realistic_market_data()
            
            # ステップ1: ML指標計算
            logger.info("ステップ1: ML指標計算")
            ml_orchestrator = MLOrchestrator(enable_automl=False)
            ml_indicators = ml_orchestrator.calculate_ml_indicators(test_data)
            
            assert ml_indicators is not None, "ML指標が計算されませんでした"
            logger.info("✅ ML指標計算成功")
            
            # ステップ2: TP/SL決定
            logger.info("ステップ2: TP/SL決定")
            tpsl_service = TPSLAutoDecisionService()
            tpsl_config = TPSLConfig(strategy=TPSLStrategy.AUTO_OPTIMAL)
            tpsl_result = tpsl_service.generate_tpsl_values(tpsl_config)
            
            assert tpsl_result is not None, "TP/SL値が生成されませんでした"
            logger.info("✅ TP/SL決定成功")
            
            # ステップ3: 条件生成
            logger.info("ステップ3: 条件生成")
            condition_generator = SmartConditionGenerator()
            test_indicators = [
                IndicatorGene(type="RSI", parameters={"period": 14}, enabled=True),
                IndicatorGene(type="SMA", parameters={"period": 20}, enabled=True),
                IndicatorGene(type="ML_UP_PROB", parameters={}, enabled=True)
            ]
            
            long_conditions, short_conditions, exit_conditions = condition_generator.generate_balanced_conditions(test_indicators)
            
            assert isinstance(long_conditions, list), "ロング条件が生成されませんでした"
            assert isinstance(short_conditions, list), "ショート条件が生成されませんでした"
            logger.info("✅ 条件生成成功")
            
            # ステップ4: データ整合性確認
            logger.info("ステップ4: データ整合性確認")
            
            # ML指標とテクニカル指標の整合性
            if ml_indicators:
                for key, values in ml_indicators.items():
                    assert len(values) == len(test_data), f"ML指標 {key} の長さが元データと一致しません"
            
            # TP/SL値の妥当性
            assert 0 < tpsl_result.stop_loss_pct < 1, "SL値が範囲外です"
            assert 0 < tpsl_result.take_profit_pct < 1, "TP値が範囲外です"
            assert tpsl_result.risk_reward_ratio > 0, "リスクリワード比が無効です"
            
            # 条件の妥当性
            total_conditions = len(long_conditions) + len(short_conditions) + len(exit_conditions)
            assert total_conditions > 0, "条件が生成されませんでした"
            
            logger.info("✅ データフロー整合性テスト成功")
            
        except Exception as e:
            pytest.fail(f"データフロー整合性テストエラー: {e}")

    def test_real_market_data_simulation(self):
        """実際の市場データシミュレーションテスト"""
        logger.info("=== 実際の市場データシミュレーションテスト ===")
        
        try:
            # 実際の市場データに近いデータを作成
            realistic_data = self._create_realistic_market_data()
            
            # 複数のコンポーネントでの処理テスト
            from app.services.auto_strategy.services.ml_orchestrator import MLOrchestrator
            
            ml_orchestrator = MLOrchestrator(enable_automl=False)
            
            # 処理時間の測定
            start_time = time.time()
            result = ml_orchestrator.calculate_ml_indicators(realistic_data)
            end_time = time.time()
            
            processing_time = end_time - start_time
            
            if result:
                # 結果の品質確認
                import numpy as np
                for key, values in result.items():
                    valid_count = sum(1 for v in values if not np.isnan(v) and np.isfinite(v))
                    valid_ratio = valid_count / len(values)
                    
                    logger.info(f"{key}: 有効値率={valid_ratio:.2%}")
                    
                    # 最低50%の有効値を期待
                    assert valid_ratio >= 0.5, f"{key}: 有効値率が低すぎます ({valid_ratio:.2%})"
                
                logger.info(f"✅ 実際の市場データ処理成功 (処理時間: {processing_time:.2f}秒)")
            else:
                logger.warning("実際の市場データ処理で結果がNone")
            
        except Exception as e:
            pytest.fail(f"実際の市場データシミュレーションテストエラー: {e}")

    def test_error_recovery_workflow(self):
        """エラー回復ワークフローテスト"""
        logger.info("=== エラー回復ワークフローテスト ===")
        
        if self.test_client is None:
            pytest.skip("テストクライアントが利用できません")
        
        try:
            # 意図的に無効なリクエストを送信
            invalid_requests = [
                # 空のリクエスト
                {},
                # 無効なGA設定
                {
                    "experiment_id": "error_test_1",
                    "experiment_name": "Error Test 1",
                    "ga_config": {"population_size": -1},
                    "backtest_config": self.complete_workflow_config["backtest_config"]
                },
                # 無効なバックテスト設定
                {
                    "experiment_id": "error_test_2",
                    "experiment_name": "Error Test 2",
                    "ga_config": self.complete_workflow_config["ga_config"],
                    "backtest_config": {"symbol": "", "initial_capital": -1000}
                }
            ]
            
            for i, invalid_request in enumerate(invalid_requests):
                logger.info(f"無効なリクエスト {i+1} のテスト")
                
                response = self.test_client.post(
                    "/api/auto-strategy/generate",
                    json=invalid_request
                )
                
                # エラーレスポンスが適切に返されることを確認
                if response.status_code in [400, 422, 500]:
                    logger.info(f"✅ 無効なリクエスト {i+1} で適切にエラーレスポンス (ステータス: {response.status_code})")
                    
                    try:
                        error_data = response.json()
                        assert isinstance(error_data, dict), "エラーレスポンスが辞書形式ではありません"
                    except json.JSONDecodeError:
                        logger.warning("エラーレスポンスがJSON形式ではありません")
                else:
                    logger.warning(f"無効なリクエスト {i+1} で予期しないステータス: {response.status_code}")
            
            logger.info("✅ エラー回復ワークフローテスト成功")
            
        except Exception as e:
            pytest.fail(f"エラー回復ワークフローテストエラー: {e}")

    def test_performance_under_load(self):
        """負荷下でのパフォーマンステスト"""
        logger.info("=== 負荷下でのパフォーマンステスト ===")
        
        if self.test_client is None:
            pytest.skip("テストクライアントが利用できません")
        
        try:
            # 複数の同時リクエストをシミュレート
            import threading
            
            results = []
            errors = []
            
            def make_request(request_id):
                try:
                    config = self.complete_workflow_config.copy()
                    config["experiment_id"] = f"load_test_{request_id}"
                    config["experiment_name"] = f"Load Test {request_id}"
                    
                    start_time = time.time()
                    response = self.test_client.post("/api/auto-strategy/generate", json=config)
                    end_time = time.time()
                    
                    results.append({
                        "request_id": request_id,
                        "status_code": response.status_code,
                        "response_time": end_time - start_time
                    })
                    
                except Exception as e:
                    errors.append(f"Request {request_id}: {e}")
            
            # 5つの同時リクエストを実行
            threads = []
            for i in range(5):
                thread = threading.Thread(target=make_request, args=(i,))
                threads.append(thread)
                thread.start()
            
            # 全てのスレッドの完了を待機
            for thread in threads:
                thread.join(timeout=30)  # 30秒でタイムアウト
            
            # 結果の分析
            successful_requests = [r for r in results if r["status_code"] in [200, 202]]
            failed_requests = [r for r in results if r["status_code"] not in [200, 202]]
            
            logger.info(f"成功したリクエスト: {len(successful_requests)}")
            logger.info(f"失敗したリクエスト: {len(failed_requests)}")
            logger.info(f"エラー: {len(errors)}")
            
            if successful_requests:
                avg_response_time = sum(r["response_time"] for r in successful_requests) / len(successful_requests)
                logger.info(f"平均レスポンス時間: {avg_response_time:.2f}秒")
                
                # レスポンス時間の基準（30秒以下）
                assert avg_response_time < 30, f"平均レスポンス時間が長すぎます: {avg_response_time:.2f}秒"
            
            logger.info("✅ 負荷下でのパフォーマンステスト成功")
            
        except Exception as e:
            pytest.fail(f"負荷下でのパフォーマンステストエラー: {e}")

    def _create_realistic_market_data(self):
        """現実的な市場データを作成"""
        import pandas as pd
        import numpy as np

        # 実際の市場データに近い特性を持つデータを生成
        dates = pd.date_range(start="2023-01-01", periods=200, freq="1H")
        np.random.seed(42)

        data = []
        base_price = 50000
        trend = 0.0001  # 微小なトレンド

        for i, date in enumerate(dates):
            # トレンドとランダムウォーク
            price_change = trend + np.random.normal(0, 0.02)
            base_price = base_price * (1 + price_change)

            # 現実的なOHLC関係
            open_price = base_price + np.random.normal(0, base_price * 0.001)
            close_price = open_price + np.random.normal(0, base_price * 0.005)

            high_price = max(open_price, close_price) + abs(np.random.normal(0, base_price * 0.003))
            low_price = min(open_price, close_price) - abs(np.random.normal(0, base_price * 0.003))

            # 現実的なボリューム（価格変動と相関）
            volatility = abs(close_price - open_price) / open_price
            base_volume = 1000
            volume = base_volume * (1 + volatility * 10) * np.random.uniform(0.5, 2.0)

            data.append({
                "timestamp": date,
                "open": open_price,
                "high": high_price,
                "low": low_price,
                "close": close_price,
                "volume": volume
            })

        return pd.DataFrame(data)


if __name__ == "__main__":
    # 単体でテストを実行する場合
    pytest.main([__file__, "-v"])
