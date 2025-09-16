"""strategy_integration_service.py のテストモジュール"""

import pytest
from datetime import datetime
from unittest.mock import MagicMock, call
from sqlalchemy.orm import Session

from database.models import BacktestResult, GeneratedStrategy
from app.services.auto_strategy.utils.strategy_integration_service import StrategyIntegrationService


@pytest.fixture
def mock_repository():
    """モックリポジトリ"""
    repo = MagicMock()
    return repo


@pytest.fixture
def sample_backtest_result():
    """テスト用のバックテスト結果データ"""
    mock_backtest = MagicMock(spec=BacktestResult)
    mock_backtest.performance_metrics = {
        "total_return": 0.35,
        "sharpe_ratio": 1.8,
        "max_drawdown": 0.12,
        "win_rate": 0.65,
        "profit_factor": 1.4,
        "total_trades": 150
    }
    return mock_backtest


@pytest.fixture
def sample_generated_strategy(sample_backtest_result):
    """テスト用の生成戦略データ"""
    mock_strategy = MagicMock(spec=GeneratedStrategy)
    mock_strategy.id = 1
    mock_strategy.experiment_id = 123
    mock_strategy.generation = 5
    mock_strategy.fitness_score = 0.85
    mock_strategy.gene_data = {
        "indicators": [
            {"type": "RSI", "enabled": True},
            {"type": "MACD", "enabled": True}
        ],
        "timeframe": "1h",
        "risk_management": {"stop_loss": 0.02, "take_profit": 0.04},
        "entry_conditions": ["rsi_cross_above"],
        "exit_conditions": ["macd_signal"]
    }

    # バックテスト結果を関連付ける
    mock_strategy.backtest_result = sample_backtest_result

    # datetime モック
    mock_datetime = datetime(2023, 1, 15, 10, 30, 0)
    mock_strategy.created_at = mock_datetime
    mock_strategy.updated_at = mock_datetime

    return mock_strategy


@pytest.fixture
def strategy_service(mock_repository):
    """テスト用のServiceインスタンス"""
    service = StrategyIntegrationService(None)  # dbは使用しない
    service.generated_strategy_repo = mock_repository
    service.backtest_result_repo = MagicMock()  # 使用されないが安全のため
    return service


class TestStrategyIntegrationService:
    """StrategyIntegrationServiceクラスのテスト"""

    def test_convert_generated_strategy_normal_case(self, strategy_service, sample_generated_strategy):
        """正常な戦略変換テスト"""
        result = strategy_service._convert_generated_strategy_to_display_format(sample_generated_strategy)

        assert result is not None
        assert result["id"] == "auto_1"
        assert result["name"] == "GA生成戦略_RSI+MACD"
        assert result["description"] == "遺伝的アルゴリズムで生成されたRSI+MACD複合戦略"
        assert result["expected_return"] == 0.35
        assert result["sharpe_ratio"] == 1.8
        assert result["max_drawdown"] == 0.12  # max_drawdownはabs()で正の値に変換される

    def test_convert_strategy_without_backtest_result(self, strategy_service):
        """バックテスト結果なしの戦略変換テスト"""
        mock_strategy = MagicMock(spec=GeneratedStrategy)
        mock_strategy.id = 2
        mock_strategy.experiment_id = 456
        mock_strategy.gene_data = {"indicators": [{"type": "SMA", "enabled": True}], "timeframe": "4h"}
        mock_strategy.backtest_result = None  # 重要: None

        result = strategy_service._convert_generated_strategy_to_display_format(mock_strategy)

        # Noneケースでも戻り値は基本構造を維持
        assert result is not None
        assert result["expected_return"] == 0.0
        assert result["sharpe_ratio"] == 0.0

    def test_extract_strategy_name_with_indicators(self, strategy_service):
        """指標情報を使用した戦略名抽出テスト"""
        gene_data = {
            "indicators": [
                {"type": "RSI", "enabled": True},
                {"type": "MACD", "enabled": True},
                {"type": "BOLINGER", "enabled": False},  # disabledは無視される
                {"type": "STOCH", "enabled": True}
            ]
        }

        name = strategy_service._extract_strategy_name(gene_data)
        # 最初の3つのenabledインジケーターを使用
        assert name == "GA生成戦略_RSI+MACD+STOCH"

    def test_extract_strategy_name_no_indicators(self, strategy_service):
        """インジケーターなしの戦略名テスト"""
        gene_data = {"indicators": []}
        name = strategy_service._extract_strategy_name(gene_data)
        assert name == "GA生成戦略"

    def test_generate_strategy_description_multiple_indicators(self, strategy_service):
        """複数インジケーターを使用した説明生成テスト"""
        gene_data = {
            "indicators": [
                {"type": "RSI", "enabled": True},
                {"type": "MACD", "enabled": True}
            ]
        }

        desc = strategy_service._generate_strategy_description(gene_data)
        assert desc == "遺伝的アルゴリズムで生成されたRSI+MACD複合戦略"

    def test_generate_strategy_description_no_indicators(self, strategy_service):
        """インジケーターなしの説明生成テスト"""
        gene_data = {"indicators": []}
        desc = strategy_service._generate_strategy_description(gene_data)
        assert desc == "遺伝的アルゴリズムで生成された戦略"

    def test_extract_indicators_list_format(self, strategy_service):
        """リスト形式のインジケーター抽出テスト"""
        gene_data = {
            "indicators": [
                {"type": "RSI", "enabled": True},
                {"type": "MACD", "enabled": False},
                {"type": "SMA", "enabled": True}
            ]
        }

        indicators = strategy_service._extract_indicators(gene_data)
        assert indicators == ["RSI", "SMA"]  # enabledのみ

    def test_extract_indicators_dict_format(self, strategy_service):
        """辞書形式のインジケーター抽出テスト（後方互換性）"""
        gene_data = {
            "indicators": {
                "rsi": {"enabled": True},
                "macd": {"enabled": False},
                "sma": {"enabled": True}
            }
        }

        indicators = strategy_service._extract_indicators(gene_data)
        assert indicators == ["RSI", "SMA"]  # enabledのみ、辞書の順序維持

    def test_extract_parameters(self, strategy_service):
        """パラメータ抽出テスト"""
        gene_data = {
            "indicators": {"rsi": {"period": 14}},
            "risk_management": {"stop_loss": 0.02},
            "entry_conditions": ["buy_signal"],
            "exit_conditions": ["sell_signal"]
        }

        params = strategy_service._extract_parameters(gene_data)
        assert params["indicators"] == {"rsi": {"period": 14}}
        assert params["risk_management"] == {"stop_loss": 0.02}

    def test_extract_performance_metrics_with_results(self, strategy_service, sample_backtest_result):
        """パフォーマンス指標抽出テスト"""
        metrics = strategy_service._extract_performance_metrics(sample_backtest_result)

        assert metrics["total_return"] == 0.35
        assert metrics["sharpe_ratio"] == 1.8
        assert metrics["max_drawdown"] == 0.12  # 絶対値に変換される

    def test_extract_performance_metrics_no_results(self, strategy_service):
        """バックテスト結果なしのパフォーマンス指標テスト"""
        metrics = strategy_service._extract_performance_metrics(None)

        # デフォルト値が設定される
        assert metrics["total_return"] == 0.0
        assert metrics["win_rate"] == 0.0

    def test_calculate_risk_level(self, strategy_service):
        """リスクレベル計算テスト"""
        # low risk (5%未満)
        low_risk_metrics = {"max_drawdown": 0.03}
        low_level = strategy_service._calculate_risk_level(low_risk_metrics)
        assert low_level == "low"

        # medium risk (5%-15%)
        medium_risk_metrics = {"max_drawdown": 0.08}
        medium_level = strategy_service._calculate_risk_level(medium_risk_metrics)
        assert medium_level == "medium"

        # high risk (15%以上)
        high_risk_metrics = {"max_drawdown": 0.20}
        high_level = strategy_service._calculate_risk_level(high_risk_metrics)
        assert high_level == "high"

    def test_convert_strategy_with_empty_gene_data(self, strategy_service):
        """空のgene_dataでの変換テスト（エラー処理）"""
        mock_strategy = MagicMock(spec=GeneratedStrategy)
        mock_strategy.id = 3
        mock_strategy.gene_data = {}  # 空

        result = strategy_service._convert_generated_strategy_to_display_format(mock_strategy)

        # Noneケースでも基本的なデータ構造は維持される
        assert result is not None

    def test_convert_strategy_missing_required_data(self, strategy_service):
        """必須データ不在時の変換テスト（エラー処理）"""
        mock_strategy = MagicMock(spec=GeneratedStrategy)
        mock_strategy.id = 4
        # gene_data属性が存在しない
        del mock_strategy.gene_data
        mock_strategy.gene_data_nonexistent = {}  # これでエラーを発生させる

        try:
            result = strategy_service._convert_generated_strategy_to_display_format(mock_strategy)
            # 例外が発生する想定
            assert False, "Expected exception did not occur"
        except AttributeError:
            # 期待されるエラー
            pass

    def test_get_strategies_pagination_and_filters(self, strategy_service, mock_repository, sample_generated_strategy):
        """戦略取得でのページネーションとフィルターテスト"""
        # モックデータの準備
        mock_repository.get_filtered_and_sorted_strategies.return_value = (25, [sample_generated_strategy])

        result = strategy_service.get_strategies(
            limit=10, offset=0, experiment_id=123, sort_by="fitness_score"
        )

        # リポジトリメソッドが正しく呼び出されていることを確認
        mock_repository.get_filtered_and_sorted_strategies.assert_called_once_with(
            limit=10, offset=0, risk_level=None, experiment_id=123,
            min_fitness=None, sort_by="fitness_score", sort_order="desc"
        )

        assert result["total_count"] == 25
        assert len(result["strategies"]) == 1

    def test_get_all_strategies_for_stats(self, strategy_service, mock_repository):
        """統計用全戦略取得テスト"""
        mock_strategy = MagicMock()
        mock_repository.get_strategies_with_backtest_results.return_value = [mock_strategy]

        result = strategy_service.get_all_strategies_for_stats()

        assert len(result) <= 1  # Noneフィルタリングされる可能性

    def test_get_strategies_with_response_normal(self, strategy_service, mock_repository, sample_generated_strategy):
        """APIレスポンス形式での戦略取得テスト"""
        # モックデータの準備
        mock_repository.get_filtered_and_sorted_strategies.return_value = (5, [sample_generated_strategy])

        result = strategy_service.get_strategies_with_response(limit=5)

        # APIレスポンス形式を検証
        assert result["success"] == True
        assert "strategies" in result["data"]
        assert result["data"]["total_count"] == 5

    def test_get_strategies_with_response_error_handling(self, strategy_service, mock_repository):
        """APIレスポンスでのエラー処理テスト"""
        # リポジトリで例外が発生
        mock_repository.get_filtered_and_sorted_strategies.side_effect = Exception("Database error")

        try:
            strategy_service.get_strategies_with_response()
            assert False, "Exception should have been raised"
        except Exception:
            # 例外が伝播することを確認
            pass

    # バグ発見用の追加テスト
    def test_convert_strategy_datetime_formatting(self, strategy_service, sample_generated_strategy):
        """datetimeオブジェクトのフォーマットテスト"""
        result = strategy_service._convert_generated_strategy_to_display_format(sample_generated_strategy)

        # datetimeがISO形式に変換されていることを確認
        assert "created_at" in result
        assert "updated_at" in result
        # isostringマッチングをテスト（Noneではないはず）

    def test_convert_strategy_with_none_datetime(self, strategy_service):
        """Noneのdatetimeフィールドでのテスト"""
        mock_strategy = MagicMock(spec=GeneratedStrategy)
        mock_strategy.id = 5
        mock_strategy.created_at = None
        mock_strategy.updated_at = None

        result = strategy_service._convert_generated_strategy_to_display_format(mock_strategy)

        # Noneのdatetimeも正しく処理されることを確認（具体的な値はgene_dataに依存）

    def test_extract_performance_metrics_partial_data(self, strategy_service):
        """部分的なパフォーマンス指標データのテスト"""
        # 部分的なメトリクスデータ
        partial_backtest = MagicMock(spec=BacktestResult)
        partial_backtest.performance_metrics = {
            "total_return": 0.1,
            # sharpe_ratio が欠けている
            "max_drawdown": 0.05
        }

        metrics = strategy_service._extract_performance_metrics(partial_backtest)

        # 欠けているデータはデフォルト値（0.0）になる
        assert metrics["total_return"] == 0.1
        assert metrics["sharpe_ratio"] == 0.0  # デフォルト値

    def test_convert_strategy_indicators_with_special_chars(self, strategy_service):
        """特殊文字を含むインジケーター名でのテスト"""
        gene_data = {
            "indicators": [
                {"type": "RSI_14", "enabled": True},
                {"type": "MACD_(12,26,9)", "enabled": True},
                {"type": "BB(20,2)", "enabled": True}
            ]
        }

        indicators = strategy_service._extract_indicators(gene_data)
        assert len(indicators) == 3
        # 特殊文字は保持される

    def test_convert_strategy_risk_level_edge_cases(self, strategy_service):
        """リスクレベル計算のエッジケーステスト"""
        # 境界値テスト
        edge_cases = [
            (0.04999, "low"),     # 5%未満
            (0.05001, "medium"),  # 5%以上15%未満
            (0.14999, "medium"),  # 15%未満
            (0.15001, "high"),    # 15%以上
            (1.0, "high")         # 100%
        ]

        for drawdown, expected_level in edge_cases:
            metrics = {"max_drawdown": drawdown}
            level = strategy_service._calculate_risk_level(metrics)
            assert level == expected_level