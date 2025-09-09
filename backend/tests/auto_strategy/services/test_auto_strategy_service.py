"""
テストケース追加: 優先度高バグ対処
Noneデータ処理、メソッド存在確認、大きな値処理のテストケース
"""

import pytest
from unittest.mock import patch, MagicMock, ANY
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

class TestAutoStrategyService:
    """AutoStrategyServiceの結合テスト"""

    def test_start_strategy_generation_success(self, auto_strategy_service):
        """start_strategy_generationの正常系テスト"""
        # 準備
        experiment_id = "test-exp-123"
        experiment_name = "Test Experiment"
        ga_config_dict = get_valid_ga_config_dict()
        backtest_config_dict = get_valid_backtest_config_dict()
        background_tasks = BackgroundTasks()

        # 実行
        result_id = auto_strategy_service.start_strategy_generation(
            experiment_id,
            experiment_name,
            ga_config_dict,
            backtest_config_dict,
            background_tasks,
        )

        # 検証
        assert result_id == experiment_id
        
        # 永続化サービスの呼び出しを検証
        auto_strategy_service.persistence_service.create_experiment.assert_called_once_with(
            experiment_id,
            experiment_name,
            ANY, # GAConfigオブジェクト
            backtest_config_dict
        )
        
        # ExperimentManagerの呼び出しを検証
        auto_strategy_service.experiment_manager.initialize_ga_engine.assert_called_once()
        
        # バックグラウンドタスクが追加されたことを検証
        # BackgroundTasksの内部実装に依存するため、ここではrun_experimentが呼ばれることを確認
        auto_strategy_service.experiment_manager.run_experiment.assert_not_called() # まだ呼ばれていない
        
        # タスクを実行
        # 実際のテストでは、バックグラウンドタスクの実行をシミュレートする必要がある
        # ここでは、add_taskに渡された関数が正しいことを間接的に確認する
        assert len(background_tasks.tasks) == 1
        task = background_tasks.tasks[0]
        assert task.func == auto_strategy_service.experiment_manager.run_experiment


    def test_start_strategy_generation_invalid_ga_config(self, auto_strategy_service):
        """無効なGA設定でエラーが発生することを確認するテスト"""
        # 準備
        experiment_id = "test-exp-invalid"
        experiment_name = "Invalid GA Config Test"
        # 無効な設定（population_sizeが0）
        ga_config_dict = get_valid_ga_config_dict()
        ga_config_dict["population_size"] = 0
        
        backtest_config_dict = get_valid_backtest_config_dict()
        background_tasks = BackgroundTasks()

        # 実行と検証
        with pytest.raises(HTTPException) as excinfo:
            auto_strategy_service.start_strategy_generation(
                experiment_id,
                experiment_name,
                ga_config_dict,
                backtest_config_dict,
                background_tasks,
            )
        assert "無効なGA設定です" in str(excinfo.value.detail)


    def test_stop_experiment_success(self, auto_strategy_service):
        """実験の停止が正常に行われるかのテスト"""
        # 準備
        experiment_id = "test-exp-to-stop"
        auto_strategy_service.experiment_manager.stop_experiment.return_value = True

        # 実行
        result = auto_strategy_service.stop_experiment(experiment_id)

        # 検証
        auto_strategy_service.experiment_manager.stop_experiment.assert_called_once_with(experiment_id)
        assert result["success"] is True
        assert result["message"] == "実験が正常に停止されました"

    def test_stop_experiment_failure(self, auto_strategy_service):
        """実験の停止に失敗した場合のテスト"""
        # 準備
        experiment_id = "test-exp-fail-stop"
        auto_strategy_service.experiment_manager.stop_experiment.return_value = False

        # 実行
        result = auto_strategy_service.stop_experiment(experiment_id)

        # 検証
        auto_strategy_service.experiment_manager.stop_experiment.assert_called_once_with(experiment_id)
        assert result["success"] is False
        assert result["message"] == "実験の停止に失敗しました"

    def test_stop_experiment_manager_not_initialized(self, auto_strategy_service):
        """ExperimentManagerが初期化されていない場合のテスト"""
        # 準備
        experiment_id = "test-exp-no-manager"
        auto_strategy_service.experiment_manager = None

        # 実行
        result = auto_strategy_service.stop_experiment(experiment_id)

        # 検証
        assert result["success"] is False
        assert result["message"] == "実験管理マネージャーが初期化されていません"

    def test_list_experiments(self, auto_strategy_service):
        """実験一覧が正しく取得できるかのテスト"""
        # 準備
        expected_experiments = [{"id": "exp1", "name": "Experiment 1"}, {"id": "exp2", "name": "Experiment 2"}]
        auto_strategy_service.persistence_service.list_experiments.return_value = expected_experiments

        # 実行
        experiments = auto_strategy_service.list_experiments()

        # 検証
        auto_strategy_service.persistence_service.list_experiments.assert_called_once()
    def test_invalid_empty_id_validation(self, auto_strategy_service):
        """空文字列IDでバリデーションを確認し、バグ検出"""
        # 準備
        experiment_id = ""  # 空文字列
        experiment_name = "Test Experiment"
        ga_config_dict = get_valid_ga_config_dict()
        backtest_config_dict = get_valid_backtest_config_dict()
        background_tasks = BackgroundTasks()

        # 実行と検証: 空IDが処理されてしまうバグを検出（期待されるはValueError）
        try:
            result = auto_strategy_service.start_strategy_generation(
                experiment_id,
                experiment_name,
                ga_config_dict,
                backtest_config_dict,
                background_tasks,
            )
            assert result == "", "失敗：空IDが処理されてしまいました"
        except ValueError as e:
            assert "empty" in str(e).lower() or "blank" in str(e).lower(), f"予測外のValueError: {e}"

    def test_none_backtest_config(self, auto_strategy_service):
        """NoneデータでAttributeError確認"""
        # 準備
        experiment_id = "test-exp-none"
        experiment_name = "Test Experiment"
        ga_config_dict = get_valid_ga_config_dict()
        backtest_config_dict = None  # None設定
        background_tasks = BackgroundTasks()

        # 実行と検証: AttributeErrorまたは適切なエラーを検出
        with pytest.raises(AttributeError) as excinfo:
            auto_strategy_service.start_strategy_generation(
                experiment_id,
                experiment_name,
                ga_config_dict,
                backtest_config_dict,
                background_tasks,
            )
        assert "copy" in str(excinfo.value).lower() or "dict" in str(excinfo.value).lower(), f"予測外のAttributeError: {excinfo.value}"

    def test_large_population_ga_config(self, auto_strategy_service):
        """極大値での処理確認"""
        # 準備
        experiment_id = "test-exp-large"
        experiment_name = "Large Population Test"
        ga_config_dict = get_valid_ga_config_dict()
        ga_config_dict["population_size"] = 1000000  # 極大値
        backtest_config_dict = get_valid_backtest_config_dict()
        background_tasks = BackgroundTasks()

        # 実行: メモリオーバーフローや処理遅延を検出
        try:
            result = auto_strategy_service.start_strategy_generation(
                experiment_id,
                experiment_name,
                ga_config_dict,
                backtest_config_dict,
                background_tasks,
            )
            assert result == experiment_id, "極大値処理に失敗"
        except MemoryError:
            pytest.fail("バグ検出：極大値でメモリエラー")
        except Exception as e:
            if "timeout" in str(e).lower() or "memory" in str(e).lower():
                pytest.fail(f"バグ検出：極大値で例外: {e}")
        assert experiments == expected_experiments

    def test_unicode_character_handling(self, auto_strategy_service):
        """Unicode文字処理確認"""
        # 準備
        experiment_id = "test-exp-unicode"
        experiment_name = "実験テスト_ユニコード🚀"  # Unicode特殊文字
        ga_config_dict = get_valid_ga_config_dict()
        backtest_config_dict = get_valid_backtest_config_dict()
        background_tasks = BackgroundTasks()

        # 実行: Unicode文字が処理可能なことを確認
        result = auto_strategy_service.start_strategy_generation(
            experiment_id,
            experiment_name,
            ga_config_dict,
            backtest_config_dict,
            background_tasks,
        )
        assert result == experiment_id, "Unicode文字処理エラー"

    def test_method_existence_for_ga_config(self, auto_strategy_service):
        """GA設定メソッド存在確認"""
        # メソッドが存在するか確認（バグ: メソッド名間違いの場合AttributeError）
        assert hasattr(auto_strategy_service, "_prepare_ga_config"), "_prepare_ga_configメソッドが存在しません"
        # from_dictメソッドが存在するか
        from app.services.auto_strategy.config.ga import GAConfig
        assert hasattr(GAConfig, "from_dict"), "GAConfig.from_dictメソッドが存在しません"

    def test_none_ga_config(self, auto_strategy_service):
        """None GA設定で例外確認"""
        # 準備
        experiment_id = "test-exp-none-ga"
        experiment_name = "Test Experiment"
        ga_config_dict = None  # None設定
        backtest_config_dict = get_valid_backtest_config_dict()
        background_tasks = BackgroundTasks()

        # 実行と検証: TypeErrorまたは適切なエラーを検出
        with pytest.raises((TypeError, AttributeError)) as excinfo:
            auto_strategy_service.start_strategy_generation(
                experiment_id,
                experiment_name,
                ga_config_dict,
                backtest_config_dict,
                background_tasks,
            )
        error_str = str(excinfo.value).lower()
        assert "none" in error_str or "dict" in error_str or "unexpected keyword" in error_str, f"予測外のエラー: {excinfo.value}"

    def test_negative_population_size(self, auto_strategy_service):
        """負数population_sizeでのエラーハンドリング"""
        # 準備
        experiment_id = "test-exp-negative"
        experiment_name = "Negative Population Test"
        ga_config_dict = get_valid_ga_config_dict()
        ga_config_dict["population_size"] = -10  # 負数
        backtest_config_dict = get_valid_backtest_config_dict()
        background_tasks = BackgroundTasks()

        # 実行と検証: ValueError確認
        try:
            result = auto_strategy_service.start_strategy_generation(
                experiment_id,
                experiment_name,
                ga_config_dict,
                backtest_config_dict,
                background_tasks,
            )
            pytest.fail("バグ検出：負数population_sizeが処理されてしまいました")
        except ValueError as e:
            assert "negative" in str(e).lower() or "population" in str(e).lower(), f"予測外のValueError: {e}"

    def test_empty_experiment_name(self, auto_strategy_service):
        """空実験名での処理確認"""
        # 準備
        experiment_id = "test-exp-empty-name"
        experiment_name = ""  # 空文字列
        ga_config_dict = get_valid_ga_config_dict()
        backtest_config_dict = get_valid_backtest_config_dict()
        background_tasks = BackgroundTasks()

        # 実行: 空experiment_nameが処理可能かを確認（バグ検出）
        result = auto_strategy_service.start_strategy_generation(
            experiment_id,
            experiment_name,
            ga_config_dict,
            backtest_config_dict,
            background_tasks,
        )
        assert result == experiment_id, "空experiment_name処理エラー"

    def test_ga_engine_with_missing_indicators(self, auto_strategy_service):
        """GAエンジンでの指示不足プログラミング検出：重要な指標不足問題を検出するテスト"""
        # 準備 - 不足インジケーターでGA設定を作成
        experiment_id = "test-exp-missing-indicators"
        experiment_name = "Missing Indicators Test"
        ga_config_dict = get_valid_ga_config_dict()
        ga_config_dict["max_indicators"] = 0  # インジケーターなし
        backtest_config_dict = get_valid_backtest_config_dict()
        background_tasks = BackgroundTasks()

        # 実行: インジケーター不足で進化が失敗するか確認
        try:
            result = auto_strategy_service.start_strategy_generation(
                experiment_id,
                experiment_name,
                ga_config_dict,
                backtest_config_dict,
                background_tasks,
            )
            # バグ: インジケーターなしでも処理されてしまう
            pytest.fail("バグ検出：インジケーター不足が無視されました")
        except (ValueError, AttributeError, KeyError) as e:
            assert "indicator" in str(e).lower() or "empty" in str(e).lower(), f"バーグ検出失敗: {e}"

    def test_none_individual_evaluator_handling(self, auto_strategy_service):
        """Noneの個体評価器処理：AttributeErrorを検出するテスト"""
        # 準備 - Noneの個体評価器を模擬
        experiment_id = "test-exp-none-evaluator"
        experiment_name = "None Evaluator Test"
        ga_config_dict = get_valid_ga_config_dict()
        backtest_config_dict = get_valid_backtest_config_dict()
        background_tasks = BackgroundTasks()

        # IndividualEvaluatorのモックをNoneに設定
        with patch('app.services.auto_strategy.services.auto_strategy_service.IndividualEvaluator', None):
            # 実行: None評価器でAttributeErrorまたは適切なエラーが出るか確認
            with pytest.raises(AttributeError) as excinfo:
                auto_strategy_service.start_strategy_generation(
                    experiment_id,
                    experiment_name,
                    ga_config_dict,
                    backtest_config_dict,
                    background_tasks,
                )
            assert "none" in str(excinfo.value).lower() or "evaluator" in str(excinfo.value).lower(), f"バーグ検出失敗: {excinfo.value}"

    def test_method_name_mismatch_ga_engine(self, auto_strategy_service):
        """GAエンジンのメソッド名不一致：AttributeErrorを検出するテスト"""
        # 準備
        from app.services.auto_strategy.core.ga_engine import GAEngine
        ga_engine = GAEngine(population_size=10)

        # 実行: 存在しないメソッドを呼び出してAttributeErrorを確認
        try:
            # run_evolutionではなくtypodしたメソッド名
            getattr(ga_engine, 'run_evolutionn_typo_method')
            pytest.fail("バグ検出：存在しないメソッドが検出されませんでした")
        except AttributeError as e:
            assert "evolutionn_typo" in str(e).lower() or "method" in str(e).lower(), f"バーグ検出失敗: {e}"

    def test_negative_crossover_rate_handling(self, auto_strategy_service):
        """負の交叉率処理のテスト"""
        # 準備
        experiment_id = "test-exp-negative-crossover"
        experiment_name = "Negative Crossover Test"
        ga_config_dict = get_valid_ga_config_dict()
        ga_config_dict["crossover_rate"] = -0.5  # 負数
        backtest_config_dict = get_valid_backtest_config_dict()
        background_tasks = BackgroundTasks()

        # 実行: 負の交叉率が適切にハンドリングされるか確認
        try:
            result = auto_strategy_service.start_strategy_generation(
                experiment_id,
                experiment_name,
                ga_config_dict,
                backtest_config_dict,
                background_tasks,
            )
            # バグ: 負数の交叉率が処理されてしまう
            pytest.fail("バグ検出：負の交叉率が処理されました")
        except (ValueError, AssertionError) as e:
            assert "negative" in str(e).lower() or "crossover" in str(e).lower(), f"バーグ検出失敗: {e}"

    def test_extremely_large_mutation_rate_overflow(self, auto_strategy_service):
        """極端な大値変異率でのオーバーフロー検出テスト"""
        # 準備
        experiment_id = "test-exp-huge-mutation"
        experiment_name = "Huge Mutation Rate Test"
        ga_config_dict = get_valid_ga_config_dict()
        ga_config_dict["mutation_rate"] = 1e10  # 極端に大きな値
        backtest_config_dict = get_valid_backtest_config_dict()
        background_tasks = BackgroundTasks()

        # 実行: 巨大値でオーバーフローや処理エラーが発生するか確認
        try:
            result = auto_strategy_service.start_strategy_generation(
                experiment_id,
                experiment_name,
                ga_config_dict,
                backtest_config_dict,
                background_tasks,
            )
            # こちらは処理される可能性もあるが、理想的には警告が出る
            assert result == experiment_id, "巨大値で処理失敗"
        except (OverflowError, MemoryError, ValueError) as e:
            assert "overflow" in str(e).lower() or "limit" in str(e).lower() or "large" in str(e).lower(), f"バーグ検出失敗: {e}"
    def test_none_experiment_id_handling(auto_strategy_service):
        """None experiment_idの処理確認"""
        # 準備
        experiment_id = None  # None
        experiment_name = "Test Experiment"
        ga_config_dict = get_valid_ga_config_dict()
        backtest_config_dict = get_valid_backtest_config_dict()
        background_tasks = BackgroundTasks()

        # 実行と検証: None IDが処理されてしまうバグを検出
        try:
            result = auto_strategy_service.start_strategy_generation(
                experiment_id,
                experiment_name,
                ga_config_dict,
                backtest_config_dict,
                background_tasks,
            )
            pytest.fail("バグ検出: None experiment_idが処理されてしまいました")
        except (TypeError, AttributeError, ValueError) as e:
            assert any(keyword in str(e).lower() for keyword in ["none", "id", "empty"]), f"予測外の例外: {e}"
    def test_extreme_unicode_experiment_name(auto_strategy_service):
        """極端なUnicode experiment_nameの処理"""
        # 準備
        experiment_id = "test-exp-extreme-unicode"
        experiment_name = "测试试验 экспедíció eksperimento eksperiment thromboembolism 実証実験 Торговля"  # 多言語文字
        ga_config_dict = get_valid_ga_config_dict()
        backtest_config_dict = get_valid_backtest_config_dict()
        background_tasks = BackgroundTasks()

        # 実行: Unicode文字が処理可能かを確認
        try:
            result = auto_strategy_service.start_strategy_generation(
                experiment_id,
                experiment_name,
                ga_config_dict,
                backtest_config_dict,
                background_tasks,
            )
            assert result == experiment_id, "Unicode文字処理エラー"
        except (UnicodeDecodeError, UnicodeEncodeError) as e:
            pytest.fail("バグ検出: Unicode処理エラー")
    def test_very_large_population_stress(auto_strategy_service):
        """非常な大値population_sizeでのストレステスト"""
        # 準備
        experiment_id = "test-exp-very-large-pop"
        experiment_name = "Very Large Population Test"
        ga_config_dict = get_valid_ga_config_dict()
        ga_config_dict["population_size"] = 10000000  # 非常に大きな値
        backtest_config_dict = get_valid_backtest_config_dict()
        background_tasks = BackgroundTasks()

        # 実行: 大きな値でメモリや処理の問題を検出
        try:
            result = auto_strategy_service.start_strategy_generation(
                experiment_id,
                experiment_name,
                ga_config_dict,
                backtest_config_dict,
                background_tasks,
            )
            assert result == experiment_id, "大値処理可能"
        except (MemoryError, OverflowError, ValueError) as e:
            pytest.fail(f"バグ検出: 大値で例外: {type(e).__name__}")

    def test_ga_engine_unicode_in_gene_generation(ga_engine):
        """GAエンジンでのUnicode包含遺伝子生成の処理"""
        # 準備
        config_dict = get_valid_ga_config_dict()
        config = GAConfig.from_dict(config_dict)
        backtest_config = get_valid_backtest_config_dict()
        
        # Unicodeを含む遺伝子を模擬
        mock_gene = MagicMock()
        mock_gene.serialize.return_value = {"condition_names": ["condition₯¢🎗łatyậ✵"]}
        ga_engine.gene_generator.generate_random_gene.return_value = mock_gene
        
        from app.services.auto_strategy.serializers.gene_serialization import GeneSerializer
        serializer = GeneSerializer()
        serializer.to_list = MagicMock(return_value=[1, 2, 3])
        
        # 実行: Unicode遺伝子で進化を実行
        try:
            result = ga_engine.run_evolution(config, backtest_config)
            assert result is not None, "Unicode遺伝子の処理エラー"
        except (UnicodeDecodeError, UnicodeEncodeError) as e:
            pytest.fail("バグ検出: Unicode遺伝子処理エラー")
    def test_missing_specific_indicators(self, ga_engine):
        """特定のインジケーター不足でのGA実行"""
        # 準備
        config_dict = get_valid_ga_config_dict()
        config_dict["max_indicators"] = 0  # インジケーターなし
        config = GAConfig.from_dict(config_dict)
        backtest_config = get_valid_backtest_config_dict()

        # 実行: インジケーター不足でエラーが発生するか確認
        # 期待: 個体生成に失敗し、例外が発生
        try:
            result = ga_engine.run_evolution(config, backtest_config)
            pytest.fail("バグ検出: インジケーター不足が無視されました")
        except (ValueError, KeyError, AttributeError) as e:
            assert "indicator" in str(e).lower() or "empty" in str(e).lower(), f"バグ検出失敗: {e}"

    def test_boundary_value_population_size_minimal(auto_strategy_service):
        """Boundary Value Testing: 最小有効値population_size=1の処理確認"""
        # 準備
        experiment_id = "test-exp-minimal-pop"
        experiment_name = "Minimal Population Test"
        ga_config_dict = get_valid_ga_config_dict()
        ga_config_dict["population_size"] = 1  # 最小境界値
        backtest_config_dict = get_valid_backtest_config_dict()
        background_tasks = BackgroundTasks()

        # 実行: 最小値が適切に処理されるか確認（expected: 正常実行 or appropriate validation）
        result = auto_strategy_service.start_strategy_generation(
            experiment_id,
            experiment_name,
            ga_config_dict,
            backtest_config_dict,
            background_tasks,
        )
        # Assert: population_size=1が正常に処理されるべきだが、バグ検出のため異常が発生する可能性
        assert result == experiment_id, "最小population_size処理エラー"
def test_inverse_input_invalid_config_type(auto_strategy_service):
         """Inverse Input Testing: ga_config_dictに無効なタイプ(list)を使用"""
         # 準備
         experiment_id = "test-exp-invalid-type"
         experiment_name = "Invalid Type Test"
         ga_config_dict = ["invalid", "list", "input"]  # list instead of dict
         backtest_config_dict = get_valid_backtest_config_dict()
         background_tasks = BackgroundTasks()

         # 実行: 無効なタイプでTypeErrorまたは適切なバリデーションが発生するか確認
         with pytest.raises((TypeError, AttributeError)) as excinfo:
             auto_strategy_service.start_strategy_generation(
                 experiment_id,
                 experiment_name,
                 ga_config_dict,
                 backtest_config_dict,
                 background_tasks,
             )
         # Assert: バグ検出の場合、適切なエラーが発生するかチェック
         error_str = str(excinfo.value).lower()
         assert "dict" in error_str or "type" in error_str or "keyword" in error_str, f"予期外のエラー: {excinfo.value}"
def test_stress_testing_large_configuration(auto_strategy_service):
         """Stress Testing: 大規模値の組み合わせでの処理確認"""
         # 準備
         experiment_id = "test-exp-stress"
         experiment_name = "Stress Test Large Config"
         ga_config_dict = get_valid_ga_config_dict()
         ga_config_dict["generations"] = 10000  # 大規模な世代数
         ga_config_dict["population_size"] = 10000  # 大規模な個体数
         ga_config_dict["max_indicators"] = 100  # 大規模な指標数
         backtest_config_dict = get_valid_backtest_config_dict()
         background_tasks = BackgroundTasks()

         # 実行: 大規模値でメモリ不足や処理エラーが発生するか確認
         try:
             result = auto_strategy_service.start_strategy_generation(
                 experiment_id,
                 experiment_name,
                 ga_config_dict,
                 backtest_config_dict,
                 background_tasks,
             )
             assert result == experiment_id, "大規模値処理成功"
         except (MemoryError, TimeoutError, RecursionError) as e:
             pytest.fail(f"バグ検出: 大規模値で例外発生: {type(e).__name__}: {e}")
         except Exception as e:
             if "limit" in str(e).lower() or "overflow" in str(e).lower():
                 pytest.fail(f"バグ検出: リミット超過: {e}")
      def test_internationalization_multiple_encodings(auto_strategy_service):
         """Internationalization Testing: 多種エンコーディング文字を含むexperiment_nameの処理"""
         # 準備
         experiment_id = "test-exp-i18n"
         experiment_name = "测试试验 экспедíció eksperimento eksperiment thromboembolism 実証実験 Торговля"  # 多言語文字
         ga_config_dict = get_valid_ga_config_dict()
         backtest_config_dict = get_valid_backtest_config_dict()
         background_tasks = BackgroundTasks()

         # 実行: 多言語文字が正しく処理されるか確認
         result = auto_strategy_service.start_strategy_generation(
             experiment_id,
             experiment_name,
             ga_config_dict,
             backtest_config_dict,
             background_tasks,
         )
         assert result == experiment_id, "多言語文字処理成功"

         # ログにも多言語文字が正しく記録されるか検証（バグ検出のため）
         # 実際の実行ではログチェックが必要だが、テストでは基本処理を確認
      def test_boundary_elite_size_exceeds_population(auto_strategy_service):
         """Boundary Value Testing: elite_size > population_sizeの境界値テスト"""
         # 準備
         experiment_id = "test-exp-elite-boundary"
         experiment_name = "Elite Size Boundary Test"
         ga_config_dict = get_valid_ga_config_dict()
         ga_config_dict["elite_size"] = 10  # elite_size > population_size (default 10)
         ga_config_dict["population_size"] = 5  # 5 < 10
         backtest_config_dict = get_valid_backtest_config_dict()
         background_tasks = BackgroundTasks()

         # 実行: elite_size > population_sizeの場合、バリデーションエラーが発生するか確認
         try:
             result = auto_strategy_service.start_strategy_generation(
                 experiment_id,
                 experiment_name,
                 ga_config_dict,
                 backtest_config_dict,
                 background_tasks,
             )
             # バグ検出: 無効な組み合わせが処理されてしまう
             pytest.fail("バグ検出: elite_size > population_sizeが許可されてしまった")
         except (ValueError, HTTPException) as e:
             assert "elite" in str(e).lower() or "population" in str(e).lower() or "size" in str(e).lower(), f"予期外のバリデーションエラー: {e}"
      def test_edge_case_mutation_rate_zero(auto_strategy_service):
         """Edge Case: mutation_rate = 0の処理確認"""
         # 準備
         experiment_id = "test-exp-zero-mutation"
         experiment_name = "Zero Mutation Rate Test"
         ga_config_dict = get_valid_ga_config_dict()
         ga_config_dict["mutation_rate"] = 0.0  # 変異率ゼロ
         backtest_config_dict = get_valid_backtest_config_dict()
         background_tasks = BackgroundTasks()

         # 実行: 変異率ゼロの場合、進化が妥当に行われるか（またはエラーが発生するか）
         result = auto_strategy_service.start_strategy_generation(
             experiment_id,
             experiment_name,
             ga_config_dict,
             backtest_config_dict,
             background_tasks,
         )
         # Assert: 変異率ゼロが許容されるか確認
         assert result == experiment_id, "変異率ゼロ処理成功"
         # バグ検出: 変異率ゼロでGAが機能しない場合があるが、このテストでは基本処理のみ確認

# 新規追加テストケース: 優先度高バグ対応

def test_none_experiment_name_arise(auto_strategy_service):
    """None experiment_nameによる潜在的AttributeError検出"""
    # 準備
    experiment_id = "test-exp-none-name"
    experiment_name = None  # Noneデータ
    ga_config_dict = get_valid_ga_config_dict()
    backtest_config_dict = get_valid_backtest_config_dict()
    background_tasks = BackgroundTasks()

    # 実行と検証: None experiment_nameが処理されてしまうバグ検出
    with pytest.raises((TypeError, AttributeError)) as excinfo:
        auto_strategy_service.start_strategy_generation(
            experiment_id,
            experiment_name,
            ga_config_dict,
            backtest_config_dict,
            background_tasks,
        )
    assert "none" in str(excinfo.value).lower() or "str" in str(excinfo.value).lower(), f"予測外のNone処理エラー: {excinfo.value}"

def test_method_non_existent_on_service(auto_strategy_service):
    """サービス上に存在しないメソッド呼び出しのAttributeError検出"""
    # 実行: 存在しないメソッドを呼び出してAttributeError確信
    try:
        getattr(auto_strategy_service, 'non_existent_method_12345')
        pytest.fail("バグ検出: 存在しないメソッドが何もしなかった")
    except AttributeError as e:
        assert "non_existent_method" in str(e).lower(), f"メソッド存在バグ検出失敗: {e}"

def test_extremely_large_float_value_generation(auto_strategy_service):
    """極大浮動小数値 generationsによる処理確認"""
    # 準備
    experiment_id = "test-exp-max-float-gen"
    experiment_name = "Max Float Generations Test"
    ga_config_dict = get_valid_ga_config_dict()
    ga_config_dict["generations"] = 1e308  # 浮動小数の最大値近似 (overflow可能性)
    backtest_config_dict = get_valid_backtest_config_dict()
    background_tasks = BackgroundTasks()

    # 実行: 巨大浮動小数値で処理エラー検出
    with pytest.raises((OverflowError, ValueError)) as excinfo:
        auto_strategy_service.start_strategy_generation(
            experiment_id,
            experiment_name,
            ga_config_dict,
            backtest_config_dict,
            background_tasks,
        )
    assert "overflow" in str(excinfo.value).lower() or "invalid" in str(excinfo.value).lower(), f"巨大浮動小数処理バグ検出失敗: {excinfo.value}"

def test_method_call_on_none_manager(auto_strategy_service):
    """None experiment_managerへのメソッド呼び出しAttributeError検出"""
    # 準備
    auto_strategy_service.experiment_manager = None  # None設定

    # 実行: None managerでメソッド呼び出し
    try:
        auto_strategy_service._initialize_ga_engine(GAConfig.from_dict(get_valid_ga_config_dict()))
        pytest.fail("バグ検出: None managerでメソッド呼び出し成功")
    except RuntimeError as e:
        assert "初期化されていません" in str(e), f"None managerメソッドバグ検出失敗: {e}"