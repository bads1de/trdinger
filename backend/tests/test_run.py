import pytest
import httpx
import subprocess
import time
import os
import signal

# APIサーバーのベースURL
BASE_URL = "http://127.0.0.1:8000"


@pytest.fixture(scope="session", autouse=True)
def start_server():
    # APIサーバーをバックグラウンドで起動
    process = subprocess.Popen(["python", "backend/run.py"])
    # サーバー起動を待機
    time.sleep(2)  # 起動時間に依存するため、適宜調整してください
    yield
    # テスト終了後にサーバーを停止
    os.kill(process.pid, signal.SIGTERM)
    process.wait()


def test_health_check_status_code():
    """
    /health エンドポイントがステータスコード 200 を返すことを確認するテスト
    """
    response = httpx.get(f"{BASE_URL}/health")
    assert response.status_code == 200


def test_health_check_response_body():
    """
    /health エンドポイントのレスポンスボディが {"status": "ok"} であることを確認するテスト
    """
    response = httpx.get(f"{BASE_URL}/health")
    assert response.json() == {"status": "ok"}
