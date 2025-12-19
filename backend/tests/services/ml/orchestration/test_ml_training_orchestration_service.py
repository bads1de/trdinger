import pytest
import pandas as pd
import numpy as np
from unittest.mock import MagicMock, patch, AsyncMock
from datetime import datetime, timedelta
from app.services.ml.orchestration.ml_training_orchestration_service import MLTrainingOrchestrationService, training_status

class TestMLTrainingOrchestrationService:
    @pytest.fixture
    def service(self):
        # グローバルな状態をリセット
        training_status.update({
            "is_training": False,
            "progress": 0,
            "status": "idle",
            "message": "待機中",
            "start_time": None,
            "end_time": None,
            "model_info": None,
            "error": None,
        })
        return MLTrainingOrchestrationService()

    @pytest.fixture
    def mock_config(self):
        config = MagicMock()
        config.symbol = "BTC/USDT"
        config.timeframe = "1h"
        config.start_date = "2023-01-01T00:00:00"
        config.end_date = "2023-01-31T00:00:00"
        config.ensemble_config = MagicMock(enabled=True)
        config.ensemble_config.model_dump.return_value = {"enabled": True}
        config.single_model_config = None
        config.optimization_settings = MagicMock(enabled=False)
        return config

    def test_validate_training_config_already_training(self, service, mock_config):
        """既にトレーニング中の場合にエラーを投げるか"""
        training_status["is_training"] = True
        with pytest.raises(ValueError, match="既にトレーニングが実行中です"):
            service.validate_training_config(mock_config)

    def test_validate_training_config_invalid_dates(self, service, mock_config):
        """日付が不正な場合にエラーを投げるか"""
        mock_config.start_date = "2023-01-31T00:00:00"
        mock_config.end_date = "2023-01-01T00:00:00"
        with pytest.raises(ValueError, match="開始日は終了日より前である必要があります"):
            service.validate_training_config(mock_config)

    def test_validate_training_config_too_short(self, service, mock_config):
        """期間が短すぎる場合にエラーを投げるか"""
        mock_config.start_date = "2023-01-01T00:00:00"
        mock_config.end_date = "2023-01-05T00:00:00" # 4 days
        with pytest.raises(ValueError, match="トレーニング期間は最低7日間必要です"):
            service.validate_training_config(mock_config)

    @pytest.mark.asyncio
    async def test_get_model_info_exception_fallback(self, service):
        """モデル情報取得で例外が発生した際のフォールバック"""
        with patch('app.services.ml.orchestration.ml_training_orchestration_service.get_latest_model_with_info', side_effect=Exception("DB Error")):
            response = await service.get_model_info()
            assert response["success"] is True
            assert response["data"]["model_status"]["is_loaded"] is False
            assert "モデル情報を取得しました（デフォルト値）" in response["message"]

    @pytest.mark.asyncio
    async def test_stop_training_not_running(self, service):
        """実行中でないトレーニングを停止しようとした場合"""
        with pytest.raises(ValueError, match="実行中のトレーニングがありません"):
            await service.stop_training()

    @pytest.mark.asyncio
    async def test_stop_training_success(self, service):
        """トレーニング停止の成功"""
        training_status["is_training"] = True
        with patch('app.services.ml.orchestration.ml_training_orchestration_service.background_task_manager.cleanup_all_tasks'):
            response = await service.stop_training()
            assert response["success"] is True
            assert training_status["is_training"] is False
            assert training_status["status"] == "stopped"

    @pytest.mark.asyncio
    async def test_train_ml_model_background_error_handling(self, service, mock_config):
        """バックグラウンドトレーニング中のエラーハンドリング"""
        db = MagicMock()
        with patch.object(service, 'get_data_service', side_effect=Exception("Data error")):
            # バックグラウンド処理を直接呼ぶ
            await service._train_ml_model_background(mock_config, db)
            
            assert training_status["is_training"] is False
            assert training_status["status"] == "error"
            assert "Data error" in training_status["error"]

    def test_determine_trainer_config_exception(self, service, mock_config):
        """設定決定中の例外ハンドリング"""
        mock_config.ensemble_config.model_dump.side_effect = Exception("Dump error")
        
        # 例外が発生してもデフォルト設定が返されるべき
        trainer_type, ensemble_dict, single_dict = service._determine_trainer_config(mock_config)
        assert trainer_type == "ensemble"
        assert ensemble_dict is not None

    @pytest.mark.asyncio
    async def test_start_training_flow(self, service, mock_config):
        """トレーニング開始からバックグラウンド実行までのフロー"""
        background_tasks = MagicMock()
        db = MagicMock()
        
        with patch.object(service, '_train_ml_model_background') as mock_bg:
            response = await service.start_training(mock_config, background_tasks, db)
            
            assert response["success"] is True
            assert "training_" in response["data"]["training_id"]
            # バックグラウンドタスクが追加されたか
            background_tasks.add_task.assert_called_once()

    @pytest.mark.asyncio
    async def test_train_ml_model_background_full_success(self, service, mock_config):
        """バックグラウンド学習の正常終了フロー"""
        db = MagicMock()
        mock_data_service = MagicMock()
        mock_training_data = pd.DataFrame({"close": [1]*100})
        mock_data_service.get_ml_training_data.return_value = mock_training_data
        
        with patch.object(service, 'get_data_service', return_value=mock_data_service):
            with patch('app.services.ml.orchestration.ml_training_orchestration_service.MLTrainingService') as mock_ml_svc:
                mock_instance = mock_ml_svc.return_value
                mock_instance.train_model.return_value = {"accuracy": 0.9}
                
                # 実際にバックグラウンド処理を実行
                await service._train_ml_model_background(mock_config, db)
                
                # ステータスが完了になっているか
                assert training_status["status"] == "completed"
                assert training_status["progress"] == 100
                assert training_status["model_info"]["accuracy"] == 0.9

    @pytest.mark.asyncio
    async def test_get_training_status(self, service):
        """ステータス取得のテスト"""
        training_status["status"] = "training"
        training_status["progress"] = 50
        
        status = await service.get_training_status()
        assert status["status"] == "training"
        assert status["progress"] == 50

    @pytest.mark.asyncio
    async def test_get_model_info_success(self, service):
        """モデル情報取得の成功ケース"""
        mock_info = {
            "path": "/path/to/model",
            "metadata": {"model_type": "lightgbm"}
        }
        with patch('app.services.ml.orchestration.ml_training_orchestration_service.get_latest_model_with_info', return_value=mock_info):
            with patch('app.services.ml.orchestration.ml_training_orchestration_service.get_model_info_with_defaults', return_value={"model_type": "lightgbm"}):
                response = await service.get_model_info()
                assert response["success"] is True
                assert response["data"]["model_status"]["is_loaded"] is True
                assert response["data"]["model_status"]["model_path"] == "/path/to/model"

    def test_execute_ml_training_with_error_handling_init_fail(self, service, mock_config):
        """MLTrainingServiceの初期化失敗時のハンドリング"""
        with patch('app.services.ml.orchestration.ml_training_orchestration_service.MLTrainingService', side_effect=Exception("Init fail")):
            # @safe_ml_operation がついているため、例外はキャッチされるが
            # デフォルトで raise_error=True のため、例外は再スローされる
            training_data = MagicMock()
            with pytest.raises(Exception, match="Init fail"):
                service._execute_ml_training_with_error_handling(
                    "ensemble", {}, {}, mock_config, training_data
                )
