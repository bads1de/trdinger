from datetime import datetime
from unittest.mock import MagicMock, Mock, patch

import pytest

from app.services.auto_strategy.services.generated_strategy_service import (
    GeneratedStrategyService,
)
from database.models import BacktestResult, GeneratedStrategy


class TestGeneratedStrategyService:
    @pytest.fixture
    def mock_db_session(self):
        return MagicMock()

    @pytest.fixture
    def service(self, mock_db_session):
        # コンストラクタ内でリポジトリを初期化しているため、
        # リポジトリのモック化はpatchで行う必要があるかもしれないが、
        # ここでは依存関係注入の観点から単純に初期化し、
        # 内部のリポジトリメソッド呼び出しをモックすることを検討する。
        # ただし、__init__でインスタンス化しているので、
        # patch.objectを使ってインスタンスの属性を置き換えるのが安全。
        service = GeneratedStrategyService(mock_db_session)
        service.generated_strategy_repo = MagicMock()
        service.backtest_result_repo = MagicMock()
        return service

    @pytest.fixture
    def sample_strategy(self):
        strategy = GeneratedStrategy(
            id=1,
            experiment_id=100,
            generation=10,
            fitness_score=1.5,
            gene_data={
                "indicators": [
                    {"type": "rsi", "enabled": True},
                    {"type": "sma", "enabled": True},
                    {"type": "macd", "enabled": False},
                ],
                "timeframe": "15m",
                "risk_management": {"stop_loss": 0.02},
                "long_entry_conditions": {},
                "short_entry_conditions": {},
                "tpsl_gene": None,
                "long_tpsl_gene": None,
                "short_tpsl_gene": None,
                "position_sizing_gene": None,
            },
            created_at=datetime(2023, 1, 1, 12, 0, 0),
        )

        # バックテスト結果のモック
        backtest = BacktestResult(
            id=1,
            strategy_name="Test Strategy",
            symbol="BTC/USDT",
            timeframe="15m",
            start_date=datetime(2023, 1, 1),
            end_date=datetime(2023, 2, 1),
            initial_capital=10000.0,
            config_json={},
            equity_curve=[],
            trade_history=[],
            performance_metrics={
                "total_return": 0.15,
                "sharpe_ratio": 2.5,
                "max_drawdown": 0.12,  # 12% -> Medium Risk
                "win_rate": 0.65,
                "profit_factor": 1.8,
                "total_trades": 50,
            },
        )
        strategy.backtest_result = backtest
        return strategy

    def test_get_strategies_basic(self, service, sample_strategy):
        # Arrange
        service.generated_strategy_repo.get_filtered_and_sorted_strategies.return_value = (
            1,
            [sample_strategy],
        )

        # Act
        result = service.get_strategies(limit=10, offset=0)

        # Assert
        assert result["total_count"] == 1
        assert result["has_more"] is False
        assert len(result["strategies"]) == 1

        strategy_data = result["strategies"][0]
        assert strategy_data["id"] == "auto_1"
        assert strategy_data["name"] == "GA生成戦略_RSI+SMA"
        assert strategy_data["risk_level"] == "medium"
        assert strategy_data["fitness_score"] == 1.5

        # リポジトリ呼び出しの確認
        service.generated_strategy_repo.get_filtered_and_sorted_strategies.assert_called_once_with(
            limit=10,
            offset=0,
            risk_level=None,
            experiment_id=None,
            min_fitness=None,
            sort_by="fitness_score",
            sort_order="desc",
        )

    def test_get_strategies_with_filters(self, service):
        # Arrange
        service.generated_strategy_repo.get_filtered_and_sorted_strategies.return_value = (
            0,
            [],
        )

        # Act
        result = service.get_strategies(
            limit=20,
            offset=10,
            risk_level="low",
            experiment_id=5,
            min_fitness=1.0,
            sort_by="total_return",
            sort_order="asc",
        )

        # Assert
        assert result["total_count"] == 0
        assert result["strategies"] == []

        service.generated_strategy_repo.get_filtered_and_sorted_strategies.assert_called_once_with(
            limit=20,
            offset=10,
            risk_level="low",
            experiment_id=5,
            min_fitness=1.0,
            sort_by="total_return",
            sort_order="asc",
        )

    def test_convert_strategy_legacy_dict_format(self, service):
        # gene_dataのindicatorsが辞書形式（旧形式）の場合のテスト
        strategy = GeneratedStrategy(
            id=2,
            experiment_id=101,
            generation=5,
            fitness_score=1.2,
            gene_data={
                "indicators": {
                    "rsi": {"enabled": True},
                    "ema": {"enabled": True},
                    "adx": {"enabled": False},
                },
                "timeframe": "1h",
            },
            created_at=datetime.now(),
        )
        strategy.backtest_result = None

        converted = service._convert_generated_strategy_to_display_format(strategy)

        assert converted is not None
        assert "RSI" in converted["name"]
        assert "EMA" in converted["name"]
        assert "ADX" not in converted["name"]
        assert "RSI" in converted["indicators"]
        assert "EMA" in converted["indicators"]

        # バックテスト結果がない場合のデフォルト値確認
        assert converted["expected_return"] == 0.0
        assert converted["max_drawdown"] == 0.0
        assert converted["risk_level"] == "low"  # 0.0 <= 0.05 -> low

    def test_convert_strategy_error_handling(self, service):
        # 変換中にエラーが発生するケース（例: gene_dataがNoneでアクセス時にエラーなど）
        # ここではgene_dataがNoneだと仮定して、_extract_strategy_nameなどでエラーになるか、
        # あるいはNoneチェックが入っているかを確認。
        # 実装を見ると cast(Dict, strategy.gene_data) しているので、NoneだとAttributeErrorやTypeErrorの可能性。
        # ただし、モデル定義上 gene_data は nullable かもしれないが、通常は入っている。
        # ここでは意図的に例外を発生させるために不正なオブジェクトを渡してみる。

        bad_strategy = MagicMock()
        type(bad_strategy).gene_data = PropertyMock(
            side_effect=Exception("Database Error")
        )

        # プロパティへのアクセスで例外が出るようにモックするのは少し複雑なので、
        # 内部メソッド呼び出しで例外が出るようにパッチするアプローチを取る
        with patch.object(
            service, "_extract_strategy_name", side_effect=Exception("Conversion Error")
        ):
            result = service._convert_generated_strategy_to_display_format(
                MagicMock(gene_data={})
            )
            assert result is None

    def test_extract_strategy_name_empty(self, service):
        # インジケータがない場合のデフォルト名
        gene_data = {"indicators": []}
        name = service._extract_strategy_name(gene_data)
        assert name == "GA生成戦略"

    def test_generate_strategy_description_empty(self, service):
        # インジケータがない場合の説明文
        gene_data = {"indicators": []}
        desc = service._generate_strategy_description(gene_data)
        assert desc == "遺伝的アルゴリズムで生成された戦略"

    def test_calculate_risk_level(self, service):
        # Low risk
        assert service._calculate_risk_level({"max_drawdown": 0.03}) == "low"
        assert service._calculate_risk_level({"max_drawdown": 0.05}) == "low"

        # Medium risk
        assert service._calculate_risk_level({"max_drawdown": 0.051}) == "medium"
        assert service._calculate_risk_level({"max_drawdown": 0.15}) == "medium"

        # High risk
        assert service._calculate_risk_level({"max_drawdown": 0.151}) == "high"
        assert service._calculate_risk_level({"max_drawdown": 0.50}) == "high"

    def test_get_strategies_with_response_success(self, service, sample_strategy):
        # Arrange
        service.generated_strategy_repo.get_filtered_and_sorted_strategies.return_value = (
            1,
            [sample_strategy],
        )

        # Act
        # api_responseの依存関係をモックする代わりに、戻り値の構造を確認する
        # ただし、関数内で import しているので、sys.modulesへのパッチが必要かもしれない
        # あるいは、統合テストとして動作させる

        # ここでは単純にメソッドを実行し、例外が出ないことと戻り値を確認する
        # api_responseはFastAPIのJSONResponseなどを返す可能性があるが、
        # コードを見ると app.utils.response.api_response を呼んでいる。
        # 実際にどうなるか試す。
        with patch("app.utils.response.api_response") as mock_api_response:
            mock_api_response.return_value = {"status": "mocked"}

            response = service.get_strategies_with_response(limit=5)

            assert response == {"status": "mocked"}
            mock_api_response.assert_called_once()
            args, kwargs = mock_api_response.call_args
            assert kwargs["success"] is True
            assert kwargs["data"]["total_count"] == 1
            assert len(kwargs["data"]["strategies"]) == 1

    def test_get_strategies_exception(self, service):
        # リポジトリが例外を投げた場合
        service.generated_strategy_repo.get_filtered_and_sorted_strategies.side_effect = Exception(
            "DB Error"
        )

        with pytest.raises(Exception) as excinfo:
            service.get_strategies()
        assert "DB Error" in str(excinfo.value)


from unittest.mock import PropertyMock
