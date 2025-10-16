"""
AutoStrategy APIテスト（修正版）
"""

import pytest
from starlette.testclient import TestClient
from unittest.mock import MagicMock, patch

from app.main import app  # FastAPIアプリケーションインスタンスをインポート
from app.api.dependencies import get_auto_strategy_service
from app.services.auto_strategy import AutoStrategyService

# AutoStrategyServiceのモックを作成
mock_auto_strategy_service = MagicMock(spec=AutoStrategyService)

# 依存関係のオーバーライド
def override_get_auto_strategy_service():
    return mock_auto_strategy_service

app.dependency_overrides[get_auto_strategy_service] = override_get_auto_strategy_service


@pytest.fixture(autouse=True)
def reset_mocks():
    """各テストの前にモックをリセット"""
    mock_auto_strategy_service.reset_mock()


# --- テストケース ---


def test_generate_strategy_success():
    """正常系: POST /generate - 戦略生成が正常に受け付けられる"""
    # 準備
    mock_auto_strategy_service.start_strategy_generation.return_value = "test-exp-id"
    request_body = {
        "experiment_id": "test-exp-id",
        "experiment_name": "Test Experiment",
        "base_config": {"symbol": "BTC/USDT"},
        "ga_config": {"population_size": 10},
    }

    # TestClientを関数内で作成
    client = TestClient(app)

    # 実行
    response = client.post("/api/auto-strategy/generate", json=request_body)

    # 検証
    assert response.status_code == 202
    assert response.json()["success"] is True
    assert response.json()["data"]["experiment_id"] == "test-exp-id"
    mock_auto_strategy_service.start_strategy_generation.assert_called_once()


def test_generate_strategy_validation_error():
    """異常系: POST /generate - 不正なリクエストボディ"""
    # 準備
    request_body = {"experiment_name": "Test"}  # 必須フィールドが不足

    # TestClientを関数内で作成
    client = TestClient(app)

    # 実行
    response = client.post("/api/auto-strategy/generate", json=request_body)

    # 検証
    assert response.status_code == 422  # Unprocessable Entity


def test_generate_strategy_service_exception():
    """異常系: POST /generate - サービスレイヤーで例外発生"""
    # 準備
    mock_auto_strategy_service.start_strategy_generation.side_effect = Exception(
        "Service Error"
    )
    request_body = {
        "experiment_id": "err-exp-id",
        "experiment_name": "Error Experiment",
        "base_config": {},
        "ga_config": {},
    }

    # TestClientを関数内で作成
    client = TestClient(app)

    # 実行
    # NOTE: safe_execute_asyncが例外をキャッチし、エラーレスポンスを返すため、
    # TestClientは例外を発生させない。レスポンスの内容をチェックする。
    response = client.post("/api/auto-strategy/generate", json=request_body)

    # 検証
    assert response.status_code == 202  # 例外はハンドラ内で処理されるため202のまま
    json_response = response.json()
    assert json_response["success"] is False
    assert "Service Error" in json_response["message"]


def test_list_experiments_success():
    """正常系: GET /experiments - 実験一覧の取得"""
    # 準備
    mock_experiments = [{"id": "exp1"}]
    mock_auto_strategy_service.list_experiments.return_value = mock_experiments

    # TestClientを関数内で作成
    client = TestClient(app)

    # 実行
    response = client.get("/api/auto-strategy/experiments")

    # 検証
    assert response.status_code == 200
    assert response.json()["experiments"] == mock_experiments
    mock_auto_strategy_service.list_experiments.assert_called_once()


def test_stop_experiment_success():
    """正常系: POST /experiments/{experiment_id}/stop - 実験の停止"""
    # 準備
    experiment_id = "stop-me"
    mock_auto_strategy_service.stop_experiment.return_value = {
        "success": True,
        "message": "Stopped",
    }

    # TestClientを関数内で作成
    client = TestClient(app)

    # 実行
    response = client.post(f"/api/auto-strategy/experiments/{experiment_id}/stop")

    # 検証
    assert response.status_code == 200
    assert response.json()["success"] is True
    mock_auto_strategy_service.stop_experiment.assert_called_with(experiment_id)


def test_stop_experiment_not_found():
    """異常系: POST /experiments/{experiment_id}/stop - 存在しない実験"""
    # 準備
    experiment_id = "not-found"
    mock_auto_strategy_service.stop_experiment.side_effect = ValueError(
        "Experiment not found"
    )

    # TestClientを関数内で作成
    client = TestClient(app)

    # 実行
    response = client.post(f"/api/auto-strategy/experiments/{experiment_id}/stop")

    # 検証
    assert response.status_code == 400
    assert "Experiment not found" in response.json()["detail"]