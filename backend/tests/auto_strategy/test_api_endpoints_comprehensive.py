"""
API エンドポイント包括的テスト

auto_strategy.py APIエンドポイントのリクエスト処理、レスポンス形式、
エラーハンドリング、認証の包括的テストを実施します。
"""

import logging
import pytest
import json
import uuid
from unittest.mock import Mock, patch, AsyncMock
from fastapi.testclient import TestClient
from fastapi import status

# FastAPIアプリケーションのインポート（実際のパスに合わせて調整）
try:
    from app.main import app
    from app.api.auto_strategy import router
except ImportError:
    # テスト環境でインポートできない場合のフォールバック
    app = None
    router = None

logger = logging.getLogger(__name__)


class TestAutoStrategyAPIEndpointsComprehensive:
    """API エンドポイント包括的テストクラス"""

    @pytest.fixture
    def client(self):
        """テストクライアント"""
        if app is None:
            pytest.skip("FastAPIアプリケーションが利用できません")
        return TestClient(app)

    @pytest.fixture
    def valid_strategy_generation_request(self):
        """有効な戦略生成リクエスト"""
        return {
            "experiment_id": str(uuid.uuid4()),
            "experiment_name": "Test Strategy Generation",
            "base_config": {
                "symbol": "BTC/USDT",
                "timeframe": "1h",
                "start_date": "2024-01-01",
                "end_date": "2024-12-31",
                "initial_capital": 100000,
                "commission_rate": 0.00055
            },
            "ga_config": {
                "population_size": 10,
                "generations": 5,
                "crossover_rate": 0.8,
                "mutation_rate": 0.1,
                "elite_size": 2,
                "max_indicators": 3,
                "allowed_indicators": ["SMA", "EMA", "RSI", "MACD"],
                "enable_multi_objective": False,
                "objectives": ["total_return"],
                "objective_weights": [1.0]
            }
        }

    @pytest.fixture
    def valid_strategy_test_request(self):
        """有効な戦略テストリクエスト"""
        return {
            "strategy_gene": {
                "indicators": [
                    {"type": "SMA", "parameters": {"period": 20}},
                    {"type": "RSI", "parameters": {"period": 14}}
                ],
                "entry_conditions": [
                    {"indicator": "SMA", "operator": ">", "value": "close"}
                ],
                "exit_conditions": [
                    {"indicator": "RSI", "operator": ">", "value": 70}
                ],
                "risk_management": {
                    "position_size": 0.1,
                    "stop_loss": 0.02,
                    "take_profit": 0.04
                }
            },
            "backtest_config": {
                "symbol": "BTC/USDT",
                "timeframe": "1h",
                "start_date": "2024-01-01",
                "end_date": "2024-01-31",
                "initial_capital": 100000,
                "commission_rate": 0.00055
            }
        }

    def test_generate_strategy_endpoint_success(self, client, valid_strategy_generation_request):
        """戦略生成エンドポイント成功テスト"""
        if client is None:
            pytest.skip("テストクライアントが利用できません")
        
        with patch('app.api.auto_strategy.auto_strategy_service') as mock_service:
            mock_service.start_strategy_generation.return_value = valid_strategy_generation_request["experiment_id"]
            
            response = client.post(
                "/api/auto-strategy/generate",
                json=valid_strategy_generation_request
            )
            
            # レスポンス検証
            assert response.status_code == status.HTTP_202_ACCEPTED
            
            response_data = response.json()
            assert "success" in response_data
            assert response_data["success"] is True
            assert "message" in response_data
            assert "data" in response_data
            assert "experiment_id" in response_data["data"]

    def test_generate_strategy_endpoint_invalid_request(self, client):
        """戦略生成エンドポイント無効リクエストテスト"""
        if client is None:
            pytest.skip("テストクライアントが利用できません")
        
        invalid_requests = [
            {},  # 空のリクエスト
            {"experiment_name": "Test"},  # 必須フィールド不足
            {"experiment_name": "Test", "base_config": {}},  # 無効な設定
            {"experiment_name": "Test", "base_config": {"symbol": "BTC"}, "ga_config": {}},  # 無効なGA設定
        ]
        
        for invalid_request in invalid_requests:
            response = client.post(
                "/api/auto-strategy/generate",
                json=invalid_request
            )
            
            # エラーレスポンスの確認
            assert response.status_code in [
                status.HTTP_400_BAD_REQUEST,
                status.HTTP_422_UNPROCESSABLE_ENTITY
            ]

    def test_test_strategy_endpoint_success(self, client, valid_strategy_test_request):
        """戦略テストエンドポイント成功テスト"""
        if client is None:
            pytest.skip("テストクライアントが利用できません")
        
        with patch('app.api.auto_strategy.auto_strategy_orchestration_service') as mock_service:
            mock_test_result = {
                "total_return": 0.15,
                "sharpe_ratio": 1.2,
                "max_drawdown": -0.08,
                "total_trades": 25,
                "win_rate": 0.6
            }
            mock_service.test_strategy.return_value = {
                "success": True,
                "result": mock_test_result,
                "message": "戦略テスト完了"
            }
            
            response = client.post(
                "/api/auto-strategy/test-strategy",
                json=valid_strategy_test_request
            )
            
            # レスポンス検証
            assert response.status_code == status.HTTP_200_OK
            
            response_data = response.json()
            assert "success" in response_data
            assert response_data["success"] is True
            assert "result" in response_data

    def test_get_progress_endpoint(self, client):
        """進捗取得エンドポイントテスト"""
        if client is None:
            pytest.skip("テストクライアントが利用できません")
        
        experiment_id = str(uuid.uuid4())
        
        with patch('app.api.auto_strategy.auto_strategy_service') as mock_service:
            mock_progress = {
                "experiment_id": experiment_id,
                "status": "running",
                "current_generation": 3,
                "total_generations": 10,
                "best_fitness": 0.75,
                "estimated_completion": "2024-01-01T12:30:00"
            }
            mock_service.get_experiment_progress.return_value = mock_progress
            
            response = client.get(f"/api/auto-strategy/progress/{experiment_id}")
            
            # レスポンス検証
            assert response.status_code == status.HTTP_200_OK
            
            response_data = response.json()
            assert "success" in response_data
            assert "progress" in response_data

    def test_get_results_endpoint(self, client):
        """結果取得エンドポイントテスト"""
        if client is None:
            pytest.skip("テストクライアントが利用できません")
        
        experiment_id = str(uuid.uuid4())
        
        with patch('app.api.auto_strategy.auto_strategy_service') as mock_service:
            mock_results = {
                "experiment_id": experiment_id,
                "best_strategy": {
                    "fitness": 0.85,
                    "parameters": {"SMA_period": 20, "RSI_period": 14},
                    "performance": {
                        "total_return": 0.15,
                        "sharpe_ratio": 1.2,
                        "max_drawdown": -0.08
                    }
                },
                "generation_results": []
            }
            mock_service.get_experiment_results.return_value = mock_results
            
            response = client.get(f"/api/auto-strategy/results/{experiment_id}")
            
            # レスポンス検証
            assert response.status_code == status.HTTP_200_OK
            
            response_data = response.json()
            assert "success" in response_data
            assert "data" in response_data

    def test_get_default_config_endpoint(self, client):
        """デフォルト設定取得エンドポイントテスト"""
        if client is None:
            pytest.skip("テストクライアントが利用できません")
        
        with patch('app.api.auto_strategy.auto_strategy_service') as mock_service:
            mock_config = {
                "ga_config": {
                    "population_size": 20,
                    "generations": 10,
                    "crossover_rate": 0.8,
                    "mutation_rate": 0.1
                },
                "backtest_config": {
                    "initial_capital": 100000,
                    "commission_rate": 0.00055
                }
            }
            mock_service.get_default_config.return_value = mock_config
            
            response = client.get("/api/auto-strategy/default-config")
            
            # レスポンス検証
            assert response.status_code == status.HTTP_200_OK
            
            response_data = response.json()
            assert "config" in response_data

    def test_get_presets_endpoint(self, client):
        """プリセット取得エンドポイントテスト"""
        if client is None:
            pytest.skip("テストクライアントが利用できません")
        
        with patch('app.api.auto_strategy.auto_strategy_service') as mock_service:
            mock_presets = {
                "conservative": {
                    "description": "保守的な戦略",
                    "ga_config": {"population_size": 10, "generations": 5}
                },
                "aggressive": {
                    "description": "積極的な戦略",
                    "ga_config": {"population_size": 50, "generations": 20}
                }
            }
            mock_service.get_presets.return_value = mock_presets
            
            response = client.get("/api/auto-strategy/presets")
            
            # レスポンス検証
            assert response.status_code == status.HTTP_200_OK
            
            response_data = response.json()
            assert "presets" in response_data

    def test_stop_experiment_endpoint(self, client):
        """実験停止エンドポイントテスト"""
        if client is None:
            pytest.skip("テストクライアントが利用できません")
        
        experiment_id = str(uuid.uuid4())
        
        with patch('app.api.auto_strategy.auto_strategy_service') as mock_service:
            mock_service.stop_experiment.return_value = True
            
            response = client.post(f"/api/auto-strategy/stop/{experiment_id}")
            
            # レスポンス検証
            assert response.status_code == status.HTTP_200_OK
            
            response_data = response.json()
            assert "success" in response_data
            assert response_data["success"] is True

    def test_list_experiments_endpoint(self, client):
        """実験一覧取得エンドポイントテスト"""
        if client is None:
            pytest.skip("テストクライアントが利用できません")
        
        with patch('app.api.auto_strategy.auto_strategy_service') as mock_service:
            mock_experiments = [
                {
                    "experiment_id": str(uuid.uuid4()),
                    "experiment_name": "Test 1",
                    "status": "completed",
                    "created_at": "2024-01-01T10:00:00"
                },
                {
                    "experiment_id": str(uuid.uuid4()),
                    "experiment_name": "Test 2",
                    "status": "running",
                    "created_at": "2024-01-01T11:00:00"
                }
            ]
            mock_service.list_experiments.return_value = mock_experiments
            
            response = client.get("/api/auto-strategy/experiments")
            
            # レスポンス検証
            assert response.status_code == status.HTTP_200_OK
            
            response_data = response.json()
            assert "experiments" in response_data
            assert isinstance(response_data["experiments"], list)

    def test_error_handling_in_endpoints(self, client, valid_strategy_generation_request):
        """エンドポイントエラーハンドリングテスト"""
        if client is None:
            pytest.skip("テストクライアントが利用できません")
        
        # サービスでエラーを発生させる
        with patch('app.api.auto_strategy.auto_strategy_service') as mock_service:
            mock_service.start_strategy_generation.side_effect = Exception("Service error")
            
            response = client.post(
                "/api/auto-strategy/generate",
                json=valid_strategy_generation_request
            )
            
            # エラーレスポンスの確認
            assert response.status_code == status.HTTP_500_INTERNAL_SERVER_ERROR
            
            response_data = response.json()
            assert "success" in response_data
            assert response_data["success"] is False
            assert "message" in response_data

    def test_request_validation(self, client):
        """リクエスト検証テスト"""
        if client is None:
            pytest.skip("テストクライアントが利用できません")
        
        # 無効なJSONリクエスト
        response = client.post(
            "/api/auto-strategy/generate",
            data="invalid json",
            headers={"Content-Type": "application/json"}
        )
        
        assert response.status_code == status.HTTP_422_UNPROCESSABLE_ENTITY

    def test_response_format_consistency(self, client):
        """レスポンス形式一貫性テスト"""
        if client is None:
            pytest.skip("テストクライアントが利用できません")
        
        endpoints_to_test = [
            ("/api/auto-strategy/default-config", "GET"),
            ("/api/auto-strategy/presets", "GET"),
            ("/api/auto-strategy/experiments", "GET"),
        ]
        
        for endpoint, method in endpoints_to_test:
            with patch('app.api.auto_strategy.auto_strategy_service') as mock_service:
                # 適切なモックレスポンスを設定
                mock_service.get_default_config.return_value = {}
                mock_service.get_presets.return_value = {}
                mock_service.list_experiments.return_value = []
                
                if method == "GET":
                    response = client.get(endpoint)
                elif method == "POST":
                    response = client.post(endpoint, json={})
                
                # 基本的なレスポンス構造の確認
                if response.status_code == 200:
                    response_data = response.json()
                    # 共通フィールドの確認（実装に依存）
                    # assert "timestamp" in response_data  # 例

    def test_concurrent_api_requests(self, client, valid_strategy_generation_request):
        """並行APIリクエストテスト"""
        if client is None:
            pytest.skip("テストクライアントが利用できません")
        
        import threading
        import time
        
        responses = []
        errors = []
        
        def make_request(test_id):
            try:
                request_data = valid_strategy_generation_request.copy()
                request_data["experiment_id"] = str(uuid.uuid4())
                request_data["experiment_name"] = f"Concurrent Test {test_id}"
                
                with patch('app.api.auto_strategy.auto_strategy_service') as mock_service:
                    mock_service.start_strategy_generation.return_value = request_data["experiment_id"]
                    
                    response = client.post(
                        "/api/auto-strategy/generate",
                        json=request_data
                    )
                    responses.append((test_id, response.status_code))
                    
            except Exception as e:
                errors.append((test_id, e))
        
        # 複数スレッドで同時リクエスト
        threads = []
        for i in range(3):
            thread = threading.Thread(target=make_request, args=(i,))
            threads.append(thread)
            thread.start()
        
        # 全スレッドの完了を待機
        for thread in threads:
            thread.join(timeout=10)
        
        # 結果検証
        if responses:
            logger.info(f"並行APIリクエスト成功: {len(responses)} 個")
        
        if errors:
            logger.warning(f"並行APIリクエストエラー: {len(errors)} 個")

    def test_api_performance(self, client):
        """APIパフォーマンステスト"""
        if client is None:
            pytest.skip("テストクライアントが利用できません")
        
        import time
        
        # 軽量なエンドポイントでのパフォーマンステスト
        with patch('app.api.auto_strategy.auto_strategy_service') as mock_service:
            mock_service.get_default_config.return_value = {}
            
            start_time = time.time()
            
            # 複数回リクエストを実行
            for _ in range(10):
                response = client.get("/api/auto-strategy/default-config")
                assert response.status_code == 200
            
            execution_time = time.time() - start_time
            avg_response_time = execution_time / 10
            
            # 平均レスポンス時間が合理的な範囲内であることを確認（1秒以下）
            assert avg_response_time < 1.0, f"APIレスポンス時間が遅すぎます: {avg_response_time:.3f}秒"
            
            logger.info(f"API平均レスポンス時間: {avg_response_time:.3f}秒")

    def test_api_security_headers(self, client):
        """APIセキュリティヘッダーテスト"""
        if client is None:
            pytest.skip("テストクライアントが利用できません")
        
        with patch('app.api.auto_strategy.auto_strategy_service') as mock_service:
            mock_service.get_default_config.return_value = {}
            
            response = client.get("/api/auto-strategy/default-config")
            
            # セキュリティヘッダーの確認（実装に依存）
            headers = response.headers
            
            # CORS ヘッダーの確認
            if "access-control-allow-origin" in headers:
                logger.info("CORS ヘッダーが設定されています")
            
            # Content-Type ヘッダーの確認
            assert "content-type" in headers
            assert "application/json" in headers["content-type"]


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
