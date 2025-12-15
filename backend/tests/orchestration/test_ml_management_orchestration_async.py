from unittest.mock import AsyncMock, patch

import pytest

from app.services.ml.orchestration.ml_management_orchestration_service import (
    MLManagementOrchestrationService,
)


@pytest.fixture
def orchestration_service():
    return MLManagementOrchestrationService()


class TestAsyncExecution:
    """非同期実行のテスト"""

    @pytest.mark.asyncio
    async def test_load_model_runs_in_threadpool(
        self, orchestration_service: MLManagementOrchestrationService
    ):
        """
        検証: load_modelが重い処理をrun_in_threadpoolで実行するか
        """
        sample_models = [{"name": "test.pkl", "path": "/models/test.pkl"}]

        with (
            patch(
                "app.services.ml.orchestration.ml_management_orchestration_service.model_manager"
            ) as mock_manager,
            patch(
                "app.services.ml.orchestration.ml_management_orchestration_service.ml_training_service"
            ) as mock_training,
            patch(
                "app.services.ml.orchestration.ml_management_orchestration_service.run_in_threadpool",
                side_effect=lambda func, *args, **kwargs: (
                    func(*args, **kwargs)
                    if not isinstance(func, AsyncMock)
                    else func(*args, **kwargs)
                ),
            ) as mock_run_in_threadpool,
            patch(
                "app.services.ml.orchestration.ml_management_orchestration_service.MLManagementOrchestrationService.get_current_model_info",
                new_callable=AsyncMock,
            ) as mock_get_info,
        ):
            mock_manager.list_models.return_value = sample_models
            mock_training.load_model.return_value = True
            mock_get_info.return_value = {"loaded": True}

            # run_in_threadpoolのモックを調整して、awaitできるようにする
            # 実際の実装では await run_in_threadpool(...) となるため、
            # モックはコルーチンを返すか、side_effectで処理する必要がある
            async def async_side_effect(func, *args, **kwargs):
                return func(*args, **kwargs)

            mock_run_in_threadpool.side_effect = async_side_effect

            await orchestration_service.load_model("test.pkl")

            # run_in_threadpoolが呼ばれたことを確認
            # 少なくとも2回呼ばれるべき (list_models, load_model)
            assert mock_run_in_threadpool.call_count >= 2

            # 呼び出し引数の確認
            calls = mock_run_in_threadpool.call_args_list

            # list_modelsの呼び出し確認
            list_models_called = any(
                call.args[0] == mock_manager.list_models for call in calls
            )
            assert (
                list_models_called
            ), "model_manager.list_models should be run in threadpool"

            # load_modelの呼び出し確認
            load_model_called = any(
                call.args[0] == mock_training.load_model for call in calls
            )
            assert (
                load_model_called
            ), "ml_training_service.load_model should be run in threadpool"

    @pytest.mark.asyncio
    async def test_get_formatted_models_runs_in_threadpool(
        self, orchestration_service: MLManagementOrchestrationService
    ):
        """
        検証: get_formatted_modelsが重い処理をrun_in_threadpoolで実行するか
        """
        with (
            patch(
                "app.services.ml.orchestration.ml_management_orchestration_service.model_manager"
            ) as mock_manager,
            patch(
                "app.services.ml.orchestration.ml_management_orchestration_service.run_in_threadpool"
            ) as mock_run_in_threadpool,
        ):
            # モックの設定
            async def async_side_effect(func, *args, **kwargs):
                return func(*args, **kwargs)

            mock_run_in_threadpool.side_effect = async_side_effect
            mock_manager.list_models.return_value = []

            await orchestration_service.get_formatted_models()

            # list_modelsがthreadpoolで実行されたか確認
            assert mock_run_in_threadpool.call_count >= 1
            assert mock_run_in_threadpool.call_args[0][0] == mock_manager.list_models




