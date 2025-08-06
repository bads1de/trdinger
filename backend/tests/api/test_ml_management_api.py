"""
ML管理API統合テスト

ML設定の更新・リセットAPIエンドポイントをテストします。
"""

import pytest
import json
from fastapi.testclient import TestClient
from unittest.mock import patch, MagicMock

from app.main import app


class TestMLManagementAPI:
    """ML管理API統合テストクラス"""

    def setup_method(self):
        """各テストメソッドの前に実行される初期化処理"""
        self.client = TestClient(app)

    def test_get_ml_config(self):
        """ML設定取得APIのテスト"""
        response = self.client.get("/api/ml/config")

        assert response.status_code == 200
        data = response.json()

        # 必要なセクションが存在することを確認
        assert "data_processing" in data
        assert "model" in data
        assert "training" in data
        assert "prediction" in data
        assert "ensemble" in data
        assert "retraining" in data

        # データ処理設定の確認
        dp_config = data["data_processing"]
        assert "max_ohlcv_rows" in dp_config
        assert "feature_calculation_timeout" in dp_config
        assert isinstance(dp_config["max_ohlcv_rows"], int)
        assert dp_config["max_ohlcv_rows"] > 0

    @patch("app.services.ml.config.ml_config_manager.ml_config_manager")
    def test_update_ml_config_success(self, mock_config_manager):
        """ML設定更新API成功テスト"""
        # モックの設定
        mock_config_manager.update_config.return_value = True
        mock_config_manager.get_config_dict.return_value = {
            "data_processing": {"max_ohlcv_rows": 500000},
            "prediction": {"default_up_prob": 0.4},
        }

        # 更新データ
        update_data = {
            "data_processing": {
                "max_ohlcv_rows": 500000,
                "feature_calculation_timeout": 1800,
            },
            "prediction": {
                "default_up_prob": 0.4,
                "default_down_prob": 0.3,
                "default_range_prob": 0.3,
            },
        }

        response = self.client.put("/api/ml/config", json=update_data)

        assert response.status_code == 200
        data = response.json()
        assert data["success"] is True
        assert "設定が正常に更新されました" in data["message"]
        assert "data" in data

        # モックが正しく呼ばれたことを確認
        mock_config_manager.update_config.assert_called_once_with(update_data)

    @patch("app.services.ml.config.ml_config_manager.ml_config_manager")
    def test_update_ml_config_failure(self, mock_config_manager):
        """ML設定更新API失敗テスト"""
        # モックの設定（更新失敗）
        mock_config_manager.update_config.return_value = False

        # 無効な更新データ
        invalid_data = {"data_processing": {"max_ohlcv_rows": -1}}  # 無効な値

        response = self.client.put("/api/ml/config", json=invalid_data)

        assert response.status_code == 200
        data = response.json()
        assert data["success"] is False
        assert "更新に失敗しました" in data["message"]

    @patch("app.services.ml.config.ml_config_manager.ml_config_manager")
    def test_reset_ml_config_success(self, mock_config_manager):
        """ML設定リセットAPI成功テスト"""
        # モックの設定
        mock_config_manager.reset_config.return_value = True
        mock_config_manager.get_config_dict.return_value = {
            "data_processing": {"max_ohlcv_rows": 1000000},  # デフォルト値
            "prediction": {"default_up_prob": 0.33},  # デフォルト値
        }

        response = self.client.post("/api/ml/config/reset")

        assert response.status_code == 200
        data = response.json()
        assert data["success"] is True
        assert "デフォルト値にリセットされました" in data["message"]
        assert "data" in data

        # デフォルト値が返されることを確認
        config_data = data["data"]
        assert config_data["data_processing"]["max_ohlcv_rows"] == 1000000
        assert config_data["prediction"]["default_up_prob"] == 0.33

        # モックが正しく呼ばれたことを確認
        mock_config_manager.reset_config.assert_called_once()

    @patch("app.services.ml.config.ml_config_manager.ml_config_manager")
    def test_reset_ml_config_failure(self, mock_config_manager):
        """ML設定リセットAPI失敗テスト"""
        # モックの設定（リセット失敗）
        mock_config_manager.reset_config.return_value = False

        response = self.client.post("/api/ml/config/reset")

        assert response.status_code == 200
        data = response.json()
        assert data["success"] is False
        assert "リセットに失敗しました" in data["message"]

    def test_update_ml_config_validation_error(self):
        """ML設定更新バリデーションエラーテスト"""
        # 明らかに無効なデータ
        invalid_data = {
            "data_processing": {
                "max_ohlcv_rows": "invalid_string",  # 文字列は無効
                "feature_calculation_timeout": -100,  # 負の値は無効
            },
            "prediction": {
                "default_up_prob": 2.0,  # 1.0を超える値は無効
                "default_down_prob": -0.5,  # 負の値は無効
            },
        }

        response = self.client.put("/api/ml/config", json=invalid_data)

        # バリデーションエラーが適切に処理されることを確認
        assert response.status_code == 200
        data = response.json()
        assert data["success"] is False

    def test_update_ml_config_partial_update(self):
        """ML設定部分更新テスト"""
        # 一部の設定のみ更新
        partial_update = {"prediction": {"default_up_prob": 0.35}}

        response = self.client.put("/api/ml/config", json=partial_update)

        # 部分更新が正常に処理されることを確認
        assert response.status_code == 200

    def test_update_ml_config_empty_data(self):
        """ML設定空データ更新テスト"""
        # 空のデータで更新
        empty_data = {}

        response = self.client.put("/api/ml/config", json=empty_data)

        # 空データでも正常に処理されることを確認
        assert response.status_code == 200

    def test_update_ml_config_invalid_json(self):
        """ML設定無効JSON更新テスト"""
        # 無効なJSONデータ
        response = self.client.put(
            "/api/ml/config",
            data="invalid json",
            headers={"Content-Type": "application/json"},
        )

        # JSONパースエラーが適切に処理されることを確認
        assert response.status_code == 422  # Unprocessable Entity

    def test_ml_config_endpoints_error_handling(self):
        """ML設定エンドポイントのエラーハンドリングテスト"""
        # 存在しないエンドポイント
        response = self.client.get("/api/ml/config/nonexistent")
        assert response.status_code == 404

        # 不正なHTTPメソッド
        response = self.client.delete("/api/ml/config")
        assert response.status_code == 405  # Method Not Allowed

    @patch(
        "app.services.ml.orchestration.ml_management_orchestration_service.MLManagementOrchestrationService"
    )
    def test_delete_all_models_success(self, mock_service_class):
        """全モデル削除API成功テスト"""
        # モックサービスインスタンス
        mock_service = MagicMock()
        mock_service_class.return_value = mock_service

        # 全削除のモック
        mock_service.delete_all_models.return_value = {
            "success": True,
            "message": "すべてのモデルが削除されました",
            "deleted_count": 5,
        }

        response = self.client.delete("/api/ml/models/all")

        assert response.status_code == 200
        data = response.json()
        assert data["success"] is True
        assert "すべてのモデルが削除されました" in data["message"]
        assert data["deleted_count"] == 5

        # モックが正しく呼ばれたことを確認
        mock_service.delete_all_models.assert_called_once()

    @patch(
        "app.services.ml.orchestration.ml_management_orchestration_service.MLManagementOrchestrationService"
    )
    def test_delete_all_models_no_models(self, mock_service_class):
        """全モデル削除API（モデルなし）テスト"""
        # モックサービスインスタンス
        mock_service = MagicMock()
        mock_service_class.return_value = mock_service

        # モデルなしの場合のモック
        mock_service.delete_all_models.return_value = {
            "success": True,
            "message": "削除するモデルがありませんでした",
            "deleted_count": 0,
        }

        response = self.client.delete("/api/ml/models/all")

        assert response.status_code == 200
        data = response.json()
        assert data["success"] is True
        assert "削除するモデルがありませんでした" in data["message"]
        assert data["deleted_count"] == 0

    @patch(
        "app.services.ml.orchestration.ml_management_orchestration_service.MLManagementOrchestrationService"
    )
    def test_delete_all_models_error(self, mock_service_class):
        """全モデル削除APIエラーテスト"""
        # モックサービスインスタンス
        mock_service = MagicMock()
        mock_service_class.return_value = mock_service

        # エラーが発生する場合のモック
        mock_service.delete_all_models.side_effect = Exception(
            "削除処理中にエラーが発生しました"
        )

        response = self.client.delete("/api/ml/models/all")

        # エラーハンドリングが適切に処理されることを確認
        assert response.status_code == 500 or response.status_code == 200
        # UnifiedErrorHandlerによってエラーが適切に処理される

    @patch(
        "app.services.ml.orchestration.ml_management_orchestration_service.MLManagementOrchestrationService"
    )
    def test_service_layer_integration(self, mock_service_class):
        """サービス層統合テスト"""
        # モックサービスインスタンス
        mock_service = MagicMock()
        mock_service_class.return_value = mock_service

        # 設定取得のモック
        mock_service.get_ml_config_dict.return_value = {
            "data_processing": {"max_ohlcv_rows": 1000000}
        }

        # 設定更新のモック
        mock_service.update_ml_config.return_value = {
            "success": True,
            "message": "設定が更新されました",
            "updated_config": {"data_processing": {"max_ohlcv_rows": 500000}},
        }

        # 設定リセットのモック
        mock_service.reset_ml_config.return_value = {
            "success": True,
            "message": "設定がリセットされました",
            "config": {"data_processing": {"max_ohlcv_rows": 1000000}},
        }

        with patch(
            "app.api.dependencies.get_ml_management_orchestration_service",
            return_value=mock_service,
        ):
            # 設定取得テスト
            response = self.client.get("/api/ml/config")
            assert response.status_code == 200

            # 設定更新テスト
            update_data = {"data_processing": {"max_ohlcv_rows": 500000}}
            response = self.client.put("/api/ml/config", json=update_data)
            assert response.status_code == 200

            # 設定リセットテスト
            response = self.client.post("/api/ml/config/reset")
            assert response.status_code == 200

            # サービスメソッドが呼ばれたことを確認
            mock_service.get_ml_config_dict.assert_called()
            mock_service.update_ml_config.assert_called_with(update_data)
            mock_service.reset_ml_config.assert_called()


if __name__ == "__main__":
    pytest.main([__file__])
