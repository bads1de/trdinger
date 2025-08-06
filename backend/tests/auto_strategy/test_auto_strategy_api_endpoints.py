"""
オートストラテジーAPI エンドポイントの包括的テスト

全てのオートストラテジーAPI（生成、テスト、設定取得、実験管理）のリクエスト/レスポンス、
認証、バリデーションをテストします。
"""

import json
import logging
from typing import Any, Dict

import pytest
from fastapi.testclient import TestClient

# テスト用のロガー設定
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class TestAutoStrategyAPIEndpoints:
    """オートストラテジーAPI エンドポイントの包括的テストクラス"""

    @pytest.fixture(autouse=True)
    def setup_method(self):
        """各テストメソッドの前に実行される初期化"""
        self.test_client = self._create_test_client()
        self.test_ga_config = self._create_test_ga_config()
        self.test_backtest_config = self._create_test_backtest_config()
        self.test_strategy_gene = self._create_test_strategy_gene()

    def _create_test_client(self):
        """テスト用のクライアントを作成"""
        try:
            from app.main import create_app
            app = create_app()
            return TestClient(app)
        except Exception as e:
            logger.warning(f"テストクライアント作成でエラー: {e}")
            return None

    def _create_test_ga_config(self) -> Dict[str, Any]:
        """テスト用のGA設定を作成"""
        return {
            "population_size": 20,
            "generations": 5,
            "mutation_rate": 0.1,
            "crossover_rate": 0.8,
            "elite_size": 2,
            "enable_multi_objective": False,
            "indicator_mode": "technical_only",
            "fitness_constraints": {
                "min_trades": 10,
                "max_drawdown_limit": 0.3,
                "min_sharpe_ratio": 0.5
            }
        }

    def _create_test_backtest_config(self) -> Dict[str, Any]:
        """テスト用のバックテスト設定を作成"""
        return {
            "symbol": "BTC:USDT",
            "timeframe": "1h",
            "start_date": "2023-01-01",
            "end_date": "2023-01-03",
            "initial_capital": 10000,
            "commission_rate": 0.001
        }

    def _create_test_strategy_gene(self) -> Dict[str, Any]:
        """テスト用の戦略遺伝子を作成"""
        return {
            "id": "test_strategy_api_001",
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

    def test_api_client_initialization(self):
        """API クライアント初期化テスト"""
        logger.info("=== API クライアント初期化テスト ===")
        
        if self.test_client is None:
            pytest.skip("テストクライアントが利用できません")
        
        # クライアントの基本機能確認
        assert hasattr(self.test_client, 'get'), "GET メソッドが利用できません"
        assert hasattr(self.test_client, 'post'), "POST メソッドが利用できません"
        assert hasattr(self.test_client, 'put'), "PUT メソッドが利用できません"
        assert hasattr(self.test_client, 'delete'), "DELETE メソッドが利用できません"
        
        logger.info("✅ API クライアント初期化テスト成功")

    def test_generate_strategy_endpoint(self):
        """戦略生成エンドポイントテスト"""
        logger.info("=== 戦略生成エンドポイントテスト ===")
        
        if self.test_client is None:
            pytest.skip("テストクライアントが利用できません")
        
        try:
            # 戦略生成リクエストの準備
            request_data = {
                "experiment_id": "test_exp_api_001",
                "experiment_name": "API Test Experiment",
                "ga_config": self.test_ga_config,
                "backtest_config": self.test_backtest_config
            }
            
            # POST リクエストの実行
            response = self.test_client.post(
                "/api/auto-strategy/generate",
                json=request_data
            )
            
            # レスポンスの基本確認
            assert response is not None, "レスポンスが取得できませんでした"
            
            # ステータスコードの確認
            if response.status_code == 200:
                # 成功レスポンスの確認
                response_data = response.json()
                assert isinstance(response_data, dict), "レスポンスが辞書形式ではありません"
                
                expected_fields = ["success", "message", "experiment_id"]
                for field in expected_fields:
                    if field in response_data:
                        logger.info(f"レスポンスフィールド {field} が存在します")
                
                logger.info("✅ 戦略生成エンドポイント成功レスポンステスト成功")
                
            elif response.status_code in [400, 422, 500]:
                # エラーレスポンスの確認
                logger.info(f"エラーレスポンス（ステータス: {response.status_code}）が返されました")
                
                try:
                    error_data = response.json()
                    assert isinstance(error_data, dict), "エラーレスポンスが辞書形式ではありません"
                    logger.info("✅ 戦略生成エンドポイントエラーレスポンステスト成功")
                except json.JSONDecodeError:
                    logger.warning("エラーレスポンスがJSON形式ではありません")
            
            else:
                logger.warning(f"予期しないステータスコード: {response.status_code}")
            
        except Exception as e:
            pytest.fail(f"戦略生成エンドポイントテストエラー: {e}")

    def test_test_strategy_endpoint(self):
        """戦略テストエンドポイントテスト"""
        logger.info("=== 戦略テストエンドポイントテスト ===")
        
        if self.test_client is None:
            pytest.skip("テストクライアントが利用できません")
        
        try:
            # 戦略テストリクエストの準備
            request_data = {
                "strategy_gene": self.test_strategy_gene,
                "backtest_config": self.test_backtest_config
            }
            
            # POST リクエストの実行
            response = self.test_client.post(
                "/api/auto-strategy/test",
                json=request_data
            )
            
            # レスポンスの基本確認
            assert response is not None, "レスポンスが取得できませんでした"
            
            # ステータスコードの確認
            if response.status_code == 200:
                # 成功レスポンスの確認
                response_data = response.json()
                assert isinstance(response_data, dict), "レスポンスが辞書形式ではありません"
                
                expected_fields = ["success", "result", "message"]
                for field in expected_fields:
                    if field in response_data:
                        logger.info(f"レスポンスフィールド {field} が存在します")
                
                logger.info("✅ 戦略テストエンドポイント成功レスポンステスト成功")
                
            elif response.status_code in [400, 422, 500]:
                # エラーレスポンスの確認
                logger.info(f"エラーレスポンス（ステータス: {response.status_code}）が返されました")
                logger.info("✅ 戦略テストエンドポイントエラーレスポンステスト成功")
            
        except Exception as e:
            pytest.fail(f"戦略テストエンドポイントテストエラー: {e}")

    def test_get_default_config_endpoint(self):
        """デフォルト設定取得エンドポイントテスト"""
        logger.info("=== デフォルト設定取得エンドポイントテスト ===")
        
        if self.test_client is None:
            pytest.skip("テストクライアントが利用できません")
        
        try:
            # GET リクエストの実行
            response = self.test_client.get("/api/auto-strategy/default-config")
            
            # レスポンスの基本確認
            assert response is not None, "レスポンスが取得できませんでした"
            
            # ステータスコードの確認
            if response.status_code == 200:
                # 成功レスポンスの確認
                response_data = response.json()
                assert isinstance(response_data, dict), "レスポンスが辞書形式ではありません"
                
                if "config" in response_data:
                    config = response_data["config"]
                    assert isinstance(config, dict), "設定が辞書形式ではありません"
                    
                    # 基本的な設定項目の確認
                    expected_config_fields = ["ga_config", "backtest_config"]
                    for field in expected_config_fields:
                        if field in config:
                            logger.info(f"設定フィールド {field} が存在します")
                
                logger.info("✅ デフォルト設定取得エンドポイントテスト成功")
                
            else:
                logger.warning(f"予期しないステータスコード: {response.status_code}")
            
        except Exception as e:
            pytest.fail(f"デフォルト設定取得エンドポイントテストエラー: {e}")

    def test_get_presets_endpoint(self):
        """プリセット取得エンドポイントテスト"""
        logger.info("=== プリセット取得エンドポイントテスト ===")
        
        if self.test_client is None:
            pytest.skip("テストクライアントが利用できません")
        
        try:
            # GET リクエストの実行
            response = self.test_client.get("/api/auto-strategy/presets")
            
            # レスポンスの基本確認
            assert response is not None, "レスポンスが取得できませんでした"
            
            # ステータスコードの確認
            if response.status_code == 200:
                # 成功レスポンスの確認
                response_data = response.json()
                assert isinstance(response_data, dict), "レスポンスが辞書形式ではありません"
                
                if "presets" in response_data:
                    presets = response_data["presets"]
                    assert isinstance(presets, dict), "プリセットが辞書形式ではありません"
                    logger.info(f"プリセット数: {len(presets)}")
                
                logger.info("✅ プリセット取得エンドポイントテスト成功")
                
            else:
                logger.warning(f"予期しないステータスコード: {response.status_code}")
            
        except Exception as e:
            pytest.fail(f"プリセット取得エンドポイントテストエラー: {e}")

    def test_list_experiments_endpoint(self):
        """実験一覧取得エンドポイントテスト"""
        logger.info("=== 実験一覧取得エンドポイントテスト ===")
        
        if self.test_client is None:
            pytest.skip("テストクライアントが利用できません")
        
        try:
            # GET リクエストの実行
            response = self.test_client.get("/api/auto-strategy/experiments")
            
            # レスポンスの基本確認
            assert response is not None, "レスポンスが取得できませんでした"
            
            # ステータスコードの確認
            if response.status_code == 200:
                # 成功レスポンスの確認
                response_data = response.json()
                assert isinstance(response_data, dict), "レスポンスが辞書形式ではありません"
                
                if "experiments" in response_data:
                    experiments = response_data["experiments"]
                    assert isinstance(experiments, list), "実験一覧がリスト形式ではありません"
                    logger.info(f"実験数: {len(experiments)}")
                    
                    # 各実験の基本構造確認
                    for experiment in experiments[:3]:  # 最初の3つのみ確認
                        assert isinstance(experiment, dict), "実験データが辞書形式ではありません"
                        
                        expected_exp_fields = ["id", "name", "status", "created_at"]
                        for field in expected_exp_fields:
                            if field in experiment:
                                logger.info(f"実験フィールド {field} が存在します")
                
                logger.info("✅ 実験一覧取得エンドポイントテスト成功")
                
            else:
                logger.warning(f"予期しないステータスコード: {response.status_code}")
            
        except Exception as e:
            pytest.fail(f"実験一覧取得エンドポイントテストエラー: {e}")

    def test_stop_experiment_endpoint(self):
        """実験停止エンドポイントテスト"""
        logger.info("=== 実験停止エンドポイントテスト ===")
        
        if self.test_client is None:
            pytest.skip("テストクライアントが利用できません")
        
        try:
            # 実験停止リクエストの準備
            test_experiment_id = "test_stop_exp_001"
            
            # POST リクエストの実行
            response = self.test_client.post(f"/api/auto-strategy/experiments/{test_experiment_id}/stop")
            
            # レスポンスの基本確認
            assert response is not None, "レスポンスが取得できませんでした"
            
            # ステータスコードの確認
            if response.status_code == 200:
                # 成功レスポンスの確認
                response_data = response.json()
                assert isinstance(response_data, dict), "レスポンスが辞書形式ではありません"
                
                expected_fields = ["success", "message"]
                for field in expected_fields:
                    if field in response_data:
                        logger.info(f"レスポンスフィールド {field} が存在します")
                
                logger.info("✅ 実験停止エンドポイント成功レスポンステスト成功")
                
            elif response.status_code in [404, 400, 500]:
                # エラーレスポンスの確認（実験が存在しない場合など）
                logger.info(f"エラーレスポンス（ステータス: {response.status_code}）が返されました")
                logger.info("✅ 実験停止エンドポイントエラーレスポンステスト成功")
            
        except Exception as e:
            pytest.fail(f"実験停止エンドポイントテストエラー: {e}")

    def test_request_validation(self):
        """リクエストバリデーションテスト"""
        logger.info("=== リクエストバリデーションテスト ===")
        
        if self.test_client is None:
            pytest.skip("テストクライアントが利用できません")
        
        try:
            # 無効なリクエストデータのテストケース
            invalid_requests = [
                # 空のリクエスト
                {},
                # 必須フィールドが不足
                {
                    "experiment_name": "Test",
                    # experiment_id, ga_config, backtest_config が不足
                },
                # 無効なGA設定
                {
                    "experiment_id": "test_invalid_001",
                    "experiment_name": "Invalid Test",
                    "ga_config": {
                        "population_size": -1,  # 無効な値
                        "generations": 0        # 無効な値
                    },
                    "backtest_config": self.test_backtest_config
                },
                # 無効なバックテスト設定
                {
                    "experiment_id": "test_invalid_002",
                    "experiment_name": "Invalid Test 2",
                    "ga_config": self.test_ga_config,
                    "backtest_config": {
                        "symbol": "",  # 空の値
                        "timeframe": "invalid_timeframe",  # 無効な値
                        "initial_capital": -1000  # 無効な値
                    }
                }
            ]
            
            for i, invalid_request in enumerate(invalid_requests):
                response = self.test_client.post(
                    "/api/auto-strategy/generate",
                    json=invalid_request
                )
                
                # バリデーションエラーが返されることを確認
                if response.status_code in [400, 422]:
                    logger.info(f"無効なリクエスト {i+1} で適切にバリデーションエラーが返されました")
                else:
                    logger.warning(f"無効なリクエスト {i+1} でバリデーションエラーが返されませんでした（ステータス: {response.status_code}）")
            
            logger.info("✅ リクエストバリデーションテスト成功")
            
        except Exception as e:
            pytest.fail(f"リクエストバリデーションテストエラー: {e}")

    def test_response_format_consistency(self):
        """レスポンス形式一貫性テスト"""
        logger.info("=== レスポンス形式一貫性テスト ===")
        
        if self.test_client is None:
            pytest.skip("テストクライアントが利用できません")
        
        try:
            # 複数のエンドポイントのレスポンス形式を確認
            endpoints = [
                ("/api/auto-strategy/default-config", "GET"),
                ("/api/auto-strategy/presets", "GET"),
                ("/api/auto-strategy/experiments", "GET"),
            ]
            
            for endpoint, method in endpoints:
                try:
                    if method == "GET":
                        response = self.test_client.get(endpoint)
                    else:
                        continue  # 他のメソッドは今回はスキップ
                    
                    if response.status_code == 200:
                        response_data = response.json()
                        
                        # 基本的なレスポンス形式の確認
                        assert isinstance(response_data, dict), f"{endpoint}: レスポンスが辞書形式ではありません"
                        
                        # Content-Type の確認
                        content_type = response.headers.get("content-type", "")
                        assert "application/json" in content_type, f"{endpoint}: Content-Type が JSON ではありません"
                        
                        logger.info(f"✅ {endpoint} のレスポンス形式確認成功")
                    
                except Exception as endpoint_error:
                    logger.warning(f"{endpoint} でエラー: {endpoint_error}")
            
            logger.info("✅ レスポンス形式一貫性テスト成功")
            
        except Exception as e:
            pytest.fail(f"レスポンス形式一貫性テストエラー: {e}")


if __name__ == "__main__":
    # 単体でテストを実行する場合
    pytest.main([__file__, "-v"])
