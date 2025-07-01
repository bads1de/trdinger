#!/usr/bin/env python3
"""
Preview エンドポイントのテスト（TDD Red Phase）
"""

import pytest
import requests
import json


class TestPreviewEndpoint:
    """Preview エンドポイントのテストクラス"""

    BASE_URL = "http://localhost:8001"

    def test_preview_endpoint_exists(self):
        """previewエンドポイントが存在することをテスト"""
        strategy_config = {
            "indicators": [
                {"type": "SMA", "parameters": {"period": 20}, "enabled": True}
            ],
            "entry_conditions": [
                {"type": "threshold", "indicator": "SMA", "operator": ">", "value": 100}
            ],
            "exit_conditions": [
                {"type": "threshold", "indicator": "SMA", "operator": "<", "value": 95}
            ],
        }

        try:
            response = requests.post(
                f"{self.BASE_URL}/api/strategy-builder/preview",
                json={"strategy_config": strategy_config},
                headers={"Content-Type": "application/json"},
                timeout=10,
            )

            # エンドポイントが存在し、404でないことを確認
            assert response.status_code != 404, "previewエンドポイントが見つかりません"

        except requests.exceptions.RequestException as e:
            pytest.fail(f"previewエンドポイントへのリクエストが失敗しました: {e}")

    def test_preview_endpoint_returns_valid_response(self):
        """previewエンドポイントが有効なレスポンスを返すことをテスト"""
        strategy_config = {
            "indicators": [
                {"type": "RSI", "parameters": {"period": 14}, "enabled": True}
            ],
            "entry_conditions": [
                {"type": "threshold", "indicator": "RSI", "operator": "<", "value": 30}
            ],
            "exit_conditions": [
                {"type": "threshold", "indicator": "RSI", "operator": ">", "value": 70}
            ],
        }

        try:
            response = requests.post(
                f"{self.BASE_URL}/api/strategy-builder/preview",
                json={"strategy_config": strategy_config},
                headers={"Content-Type": "application/json"},
                timeout=10,
            )

            # 成功レスポンスを期待
            assert (
                response.status_code == 200
            ), f"期待されるステータスコード: 200, 実際: {response.status_code}"

            # JSONレスポンスを期待
            data = response.json()
            assert (
                "success" in data
            ), "レスポンスに'success'フィールドが含まれていません"
            assert data["success"] is True, "レスポンスのsuccessがTrueではありません"
            assert "data" in data, "レスポンスに'data'フィールドが含まれていません"

            # プレビューデータの構造を確認
            preview_data = data["data"]
            assert (
                "strategy_summary" in preview_data
            ), "プレビューデータに'strategy_summary'が含まれていません"
            assert (
                "indicators_used" in preview_data
            ), "プレビューデータに'indicators_used'が含まれていません"
            assert (
                "conditions_summary" in preview_data
            ), "プレビューデータに'conditions_summary'が含まれていません"

        except requests.exceptions.RequestException as e:
            pytest.fail(f"previewエンドポイントへのリクエストが失敗しました: {e}")
        except json.JSONDecodeError:
            pytest.fail("レスポンスが有効なJSONではありません")

    def test_preview_endpoint_handles_invalid_config(self):
        """previewエンドポイントが無効な設定を適切に処理することをテスト"""
        invalid_config = {
            "indicators": [],  # 空の指標リスト
            "entry_conditions": [],
            "exit_conditions": [],
        }

        try:
            response = requests.post(
                f"{self.BASE_URL}/api/strategy-builder/preview",
                json={"strategy_config": invalid_config},
                headers={"Content-Type": "application/json"},
                timeout=10,
            )

            # エラーレスポンスを期待（400 Bad Request または 500 Internal Server Error）
            assert response.status_code in [
                400,
                500,
            ], f"期待されるステータスコード: 400 or 500, 実際: {response.status_code}"

            # JSONレスポンスを期待
            data = response.json()

            # 400エラーの場合
            if response.status_code == 400:
                assert (
                    "success" in data
                ), "レスポンスに'success'フィールドが含まれていません"
                assert (
                    data["success"] is False
                ), "レスポンスのsuccessがFalseではありません"
                assert (
                    "message" in data
                ), "レスポンスに'message'フィールドが含まれていません"
            # 500エラーの場合
            else:
                assert (
                    "detail" in data
                ), "レスポンスに'detail'フィールドが含まれていません"

        except requests.exceptions.RequestException as e:
            pytest.fail(f"previewエンドポイントへのリクエストが失敗しました: {e}")
        except json.JSONDecodeError:
            pytest.fail("レスポンスが有効なJSONではありません")

    def test_preview_endpoint_with_complex_strategy(self):
        """複雑な戦略設定でのpreviewエンドポイントテスト"""
        complex_config = {
            "indicators": [
                {"type": "SMA", "parameters": {"period": 20}, "enabled": True},
                {"type": "RSI", "parameters": {"period": 14}, "enabled": True},
                {
                    "type": "MACD",
                    "parameters": {
                        "fast_period": 12,
                        "slow_period": 26,
                        "signal_period": 9,
                    },
                    "enabled": True,
                },
            ],
            "entry_conditions": [
                {
                    "type": "threshold",
                    "indicator": "SMA",
                    "operator": ">",
                    "value": 100,
                },
                {"type": "threshold", "indicator": "RSI", "operator": "<", "value": 30},
            ],
            "exit_conditions": [
                {"type": "threshold", "indicator": "RSI", "operator": ">", "value": 70}
            ],
        }

        try:
            response = requests.post(
                f"{self.BASE_URL}/api/strategy-builder/preview",
                json={"strategy_config": complex_config},
                headers={"Content-Type": "application/json"},
                timeout=10,
            )

            # 成功レスポンスを期待
            assert (
                response.status_code == 200
            ), f"期待されるステータスコード: 200, 実際: {response.status_code}"

            data = response.json()
            assert data["success"] is True, "複雑な戦略設定でのプレビューが失敗しました"

            # 複数の指標が認識されていることを確認
            preview_data = data["data"]
            indicators_used = preview_data["indicators_used"]
            assert (
                len(indicators_used) == 3
            ), f"期待される指標数: 3, 実際: {len(indicators_used)}"

        except requests.exceptions.RequestException as e:
            pytest.fail(f"previewエンドポイントへのリクエストが失敗しました: {e}")


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
