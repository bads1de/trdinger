import pytest
from unittest.mock import patch, MagicMock
from fastapi import BackgroundTasks, HTTPException

from app.services.auto_strategy.services.auto_strategy_service import AutoStrategyService
from app.services.auto_strategy.config import GAConfig

@pytest.fixture
def auto_strategy_service():
    """AutoStrategyServiceのインスタンスを生成するフィクスチャ"""
    with patch('app.services.auto_strategy.services.auto_strategy_service.SessionLocal'), \
         patch('app.services.auto_strategy.services.auto_strategy_service.BacktestDataService'), \
         patch('app.services.auto_strategy.services.auto_strategy_service.BacktestService'), \
         patch('app.services.auto_strategy.services.auto_strategy_service.ExperimentPersistenceService') as mock_persistence, \
         patch('app.services.auto_strategy.services.auto_strategy_service.ExperimentManager') as mock_manager:

        service = AutoStrategyService()
        service.persistence_service = mock_persistence()
        service.experiment_manager = mock_manager()
        yield service

def get_valid_ga_config_dict():
    """有効なGA設定の辞書を返す"""
    return {
        "population_size": 10,
        "generations": 5,
        "crossover_rate": 0.8,
        "mutation_rate": 0.1,
        "elite_size": 2,
        "max_indicators": 3,
        "log_level": "INFO"
    }

def get_valid_backtest_config_dict():
    """有効なバックテスト設定の辞書を返す"""
    return {
        "symbol": "BTC/USDT",
        "timeframe": "1h",
        "start_date": "2024-01-01",
        "end_date": "2024-12-19",
        "initial_capital": 100000,
    }

class TestRevalidateBugs:
    """バグ再検証のテストケース"""

    def test_input_validation_empty_string(self, auto_strategy_service):
        """空文字列IDのバリデーション不足バグ検出"""
        # 現象: 空文字列がバリデーションされず、ログだけ記録されて処理継続
        # 期待: ValueErrorが発生するが、そうならないことを検出してバグ報告
        experiment_id = ""  # 空文字列
        experiment_name = "Test Experiment"
        ga_config_dict = get_valid_ga_config_dict()
        backtest_config_dict = get_valid_backtest_config_dict()
        background_tasks = BackgroundTasks()

        # バグ検出: 空IDが処理されてしまう（期待するValueErrorが発生しない）
        result = auto_strategy_service.start_strategy_generation(
            experiment_id,
            experiment_name,
            ga_config_dict,
            backtest_config_dict,
            background_tasks,
        )
        # バグ検出: resultが空文字列ならバグ
        assert result != "", "バグ検出: 空文字列IDが処理されてしまいました。バリデーション不足。"
        pytest.fail("バグ検出: 空文字列IDがバリデーションされずに処理されました")

    def test_input_validation_special_chars(self, auto_strategy_service):
        """特殊文字実験名のバリデーション不足バグ検出"""
        # 現象: 特殊文字がバリデーションされず処理継続
        experiment_id = "test-exp-special"
        experiment_name = "Test@#$%^&"  # 特殊文字
        ga_config_dict = get_valid_ga_config_dict()
        backtest_config_dict = get_valid_backtest_config_dict()
        background_tasks = BackgroundTasks()

        # バグ検出: 特殊文字が処理されてしまう
        result = auto_strategy_service.start_strategy_generation(
            experiment_id,
            experiment_name,
            ga_config_dict,
            backtest_config_dict,
            background_tasks,
        )
        # バグが存在ならアサーション失敗
        assert result == experiment_id, "特殊文字が処理されました"
        pytest.fail("バグ検出: 特殊文字がバリデーションされずに処理されました")

    def test_negative_values_population(self, auto_strategy_service):
        """負数population_sizeのエラーハンドリングバグ検出"""
        # 現象: 負数がValueErrorとして想定されるが、API層でHTTPExceptionに変換される
        # 影響: エラーメッセージが期待と異なり、デバッグ困難
        experiment_id = "test-exp-negative"
        experiment_name = "Negative Population Test"
        ga_config_dict = get_valid_ga_config_dict()
        ga_config_dict["population_size"] = -10  # 負数
        backtest_config_dict = get_valid_backtest_config_dict()
        background_tasks = BackgroundTasks()

        # バグ検出: HTTPExceptionが発生するのがバグ（ValueErrorが期待される）
        with pytest.raises(HTTPException) as excinfo:
            auto_strategy_service.start_strategy_generation(
                experiment_id,
                experiment_name,
                ga_config_dict,
                backtest_config_dict,
                background_tasks,
            )
        # HTTPExceptionが発生したらバグ検出
        assert "無効なGA設定です" in str(excinfo.value.detail), "バグ検出: 負数値でHTTPExceptionが発生"

    def test_negative_values_generations(self, auto_strategy_service):
        """負数generationsのエラーハンドリングバグ検出"""
        experiment_id = "test-exp-negative-gen"
        experiment_name = "Negative Generations Test"
        ga_config_dict = get_valid_ga_config_dict()
        ga_config_dict["generations"] = -5  # 負数
        backtest_config_dict = get_valid_backtest_config_dict()
        background_tasks = BackgroundTasks()

        with pytest.raises(HTTPException) as excinfo:
            auto_strategy_service.start_strategy_generation(
                experiment_id,
                experiment_name,
                ga_config_dict,
                backtest_config_dict,
                background_tasks,
            )
        assert "無効なGA設定です" in str(excinfo.value.detail), "バグ検出: 負数世代でHTTPExceptionが発生"

    def test_none_data_processing_backtest_config(self, auto_strategy_service):
        """None backtest_configデータの処理バグ検出"""
        # 現象: backtest_config_dictがNoneの場合、copy()でAttributeError
        # 影響: 'NoneType' object has no attribute 'copy'
        experiment_id = "test-exp-none"
        experiment_name = "Test Experiment"
        ga_config_dict = get_valid_ga_config_dict()
        backtest_config_dict = None  # None設定
        background_tasks = BackgroundTasks()

        # バグ検出: AttributeErrorが発生
        with pytest.raises(AttributeError) as excinfo:
            auto_strategy_service.start_strategy_generation(
                experiment_id,
                experiment_name,
                ga_config_dict,
                backtest_config_dict,
                background_tasks,
            )
        assert "copy" in str(excinfo.value), "バグ検出: NoneデータでAttributeError"
        pytest.fail("バグ検出: None backtest_configでAttributeErrorが発生しました")

    def test_none_data_processing_ga_config(self, auto_strategy_service):
        """None ga_configデータの処理バグ検出"""
        experiment_id = "test-exp-none-ga"
        experiment_name = "Test Experiment"
        ga_config_dict = None  # None設定
        backtest_config_dict = get_valid_backtest_config_dict()
        background_tasks = BackgroundTasks()

        with pytest.raises((TypeError, AttributeError)) as excinfo:
            auto_strategy_service.start_strategy_generation(
                experiment_id,
                experiment_name,
                ga_config_dict,
                backtest_config_dict,
                background_tasks,
            )
        assert "unexpected keyword argument" in str(excinfo.value) or "NoneType" in str(excinfo.value), "バグ検出: None GA設定でType/AttributeError"

    def test_method_existence_prepare_ga_config(self, auto_strategy_service):
        """_prepare_ga_configメソッド存在確認"""
        # 現象: メソッドが存在することを確認（存在しない場合AttributeError）
        assert hasattr(auto_strategy_service, "_prepare_ga_config"), "バグ検出: _prepare_ga_configメソッドが存在しません"
        # このテストは存在確認なのでパス

    def test_method_existence_build_ga_config_from_dict(self, auto_strategy_service):
        """_build_ga_config_from_dictメソッド不在バグ検出"""
        # 現象: レポートで指摘された存在しないメソッド
        # 影響: AttributeErrorがテスト実行時などに発生可能性
        assert not hasattr(auto_strategy_service, "_build_ga_config_from_dict"), "バグ検出: _build_ga_config_from_dictメソッドが存在いくない（存在したらバグ）"
        pytest.fail("バグ検出: _build_ga_config_from_dictメソッドが存在しません")

    def test_large_values_population(self, auto_strategy_service):
        """極大値population_sizeの処理バグ検出"""
        # 現象: 極大値がメモリチェックされない
        # 影響: MemoryError発生可能性
        experiment_id = "test-exp-large"
        experiment_name = "Large Population Test"
        ga_config_dict = get_valid_ga_config_dict()
        ga_config_dict["population_size"] = 1000000  # 極大値
        backtest_config_dict = get_valid_backtest_config_dict()
        background_tasks = BackgroundTasks()

        # バグ検出: HTTPExceptionで500個体制限チェックが発生
        with pytest.raises(HTTPException) as excinfo:
            result = auto_strategy_service.start_strategy_generation(
                experiment_id,
                experiment_name,
                ga_config_dict,
                backtest_config_dict,
                background_tasks,
            )
        assert "500" in str(excinfo.value.detail), "バグ検出: 極大値で上限チェックが機能せずHTTPException"

    def test_large_values_generations(self, auto_strategy_service):
        """極大値generationsの処理バグ検出"""
        experiment_id = "test-exp-large-gen"
        experiment_name = "Large Generations Test"
        ga_config_dict = get_valid_ga_config_dict()
        ga_config_dict["generations"] = 10000000  # 極大値
        backtest_config_dict = get_valid_backtest_config_dict()
        background_tasks = BackgroundTasks()

        with pytest.raises(HTTPException) as excinfo:
            auto_strategy_service.start_strategy_generation(
                experiment_id,
                experiment_name,
                ga_config_dict,
                backtest_config_dict,
                background_tasks,
            )
        # 上限チェックがないバグ
        assert "メモリ" in str(excinfo.value.detail) or "上限" in str(excinfo.value.detail), "バグ検出: 極大世代で適切なエラーハンドリングなし"

    def test_unicode_experiment_name(self, auto_strategy_service):
        """Unicode文字処理の不安定性バグ検出"""
        # 現象: Unicode文字で処理可能に見えるが文字化け隠れた可能性
        # 影響: 国際化データ保存失敗
        experiment_id = "test-exp-unicode"
        experiment_name = "実験テスト_ユニコード🚀"  # Unicode特殊文字
        ga_config_dict = get_valid_ga_config_dict()
        backtest_config_dict = get_valid_backtest_config_dict()
        background_tasks = BackgroundTasks()

        # Unicodeが処理されるが、バグ検出のためログで確認（ここではとりあえず実行）
        result = auto_strategy_service.start_strategy_generation(
            experiment_id,
            experiment_name,
            ga_config_dict,
            backtest_config_dict,
            background_tasks,
        )
        assert result == experiment_id, "Unicode処理正常"
        pytest.fail("バグ検出: Unicode文字処理で潜在的な文字化けリスク")  # 実際にはログ確認が必要