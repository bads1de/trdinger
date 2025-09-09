
import pytest
from unittest.mock import MagicMock, patch

from app.services.auto_strategy.core.individual_evaluator import IndividualEvaluator
from app.services.auto_strategy.config import GAConfig

# テスト用のGA設定を作成するヘルパー関数
def create_test_ga_config(enable_multi_objective=False, objectives=['total_return'], min_trades=0, max_drawdown_limit=1.0):
    config_dict = {
        "enable_multi_objective": enable_multi_objective,
        "objectives": objectives,
        "fitness_constraints": {
            "min_trades": min_trades,
            "max_drawdown_limit": max_drawdown_limit
        },
        "fitness_weights": { # 単一目的用の重み
            "total_return": 0.5,
            "sharpe_ratio": 0.3,
            "max_drawdown": 0.1,
            "win_rate": 0.05,
            "balance_score": 0.05
        }
    }
    return GAConfig.from_dict(config_dict)

# テスト用の個体（遺伝子リスト）を作成するヘルパー関数
def create_test_individual():
    """
    to_list のエンコードロジックを模倣した、正規化済みのfloatリストを返す。
    """
    # ダミーの指標IDとパラメータ
    # indicator_ids は約20種類と仮定
    total_indicators = 20
    sma_id = 1
    rsi_id = 5

    # パラメータの正規化 (period: 1-200 -> 0-1)
    def normalize(p):
        return (p - 1) / (200 - 1)

    encoded_list = []

    # Indicator 1: SMA(10)
    encoded_list.extend([sma_id / total_indicators, normalize(10)])
    # Indicator 2: RSI(14)
    encoded_list.extend([rsi_id / total_indicators, normalize(14)])
    # 残りは空
    encoded_list.extend([0.0, 0.0] * 3) # max_indicators = 5 と仮定

    # Conditions (ダミー)
    encoded_list.extend([1.0, 0.0, 0.0]) # entry: >
    encoded_list.extend([0.0, 1.0, 0.0]) # exit: <

    # TP/SL (ダミー)
    encoded_list.extend([0.0] * 8)

    # Position Sizing (ダミー)
    encoded_list.extend([0.0] * 8)
    
    # to_listの出力長は32なので合わせる
    # 5*2 (indicators) + 3 (entry) + 3 (exit) + 8 (tpsl) + 8 (pos_sizing) = 32
    return encoded_list


@pytest.fixture
def mock_backtest_service():
    return MagicMock()

@pytest.fixture
def individual_evaluator(mock_backtest_service):
    evaluator = IndividualEvaluator(mock_backtest_service)
    evaluator.set_backtest_config({"symbol": "BTC/USDT", "timeframe": "1h"})
    return evaluator

class TestIndividualEvaluator:

    # 正常系テスト
    def test_evaluate_individual_single_objective_success(self, individual_evaluator, mock_backtest_service):
        """単一目的評価が正常に成功するケース"""
        # 準備
        mock_backtest_service.run_backtest.return_value = {
            "performance_metrics": {
                "total_return": 0.25,
                "sharpe_ratio": 1.5,
                "max_drawdown": 0.1,
                "win_rate": 0.6,
                "total_trades": 50
            },
            "trade_history": [{"size": 1, "pnl": 10}, {"size": -1, "pnl": 5}] # long/short
        }
        config = create_test_ga_config()
        individual = create_test_individual()

        # 実行
        fitness = individual_evaluator.evaluate_individual(individual, config)

        # 検証
        assert isinstance(fitness, tuple)
        assert len(fitness) == 1
        assert fitness[0] > 0 # 何らかの正の適応度が計算されること
        mock_backtest_service.run_backtest.assert_called_once()

    def test_evaluate_individual_multi_objective_success(self, individual_evaluator, mock_backtest_service):
        """多目的評価が正常に成功するケース"""
        # 準備
        mock_backtest_service.run_backtest.return_value = {
            "performance_metrics": {
                "total_return": 0.3,
                "sharpe_ratio": 1.8,
                "max_drawdown": 0.08,
                "total_trades": 60
            }
        }
        config = create_test_ga_config(enable_multi_objective=True, objectives=['total_return', 'sharpe_ratio', 'max_drawdown'])
        individual = create_test_individual()

        # 実行
        fitness = individual_evaluator.evaluate_individual(individual, config)

        # 検証
        assert isinstance(fitness, tuple)
        assert len(fitness) == 3
        assert fitness[0] == 0.3  # total_return
        assert fitness[1] == 1.8  # sharpe_ratio
        assert fitness[2] == 0.08 # max_drawdown
        mock_backtest_service.run_backtest.assert_called_once()

    # 異常系・エッジケース
    def test_evaluate_individual_backtest_exception(self, individual_evaluator, mock_backtest_service):
        """バックテスト実行時に例外が発生するケース"""
        # 準備
        mock_backtest_service.run_backtest.side_effect = Exception("Backtest failed")
        config_single = create_test_ga_config()
        config_multi = create_test_ga_config(enable_multi_objective=True, objectives=['total_return', 'sharpe_ratio'])
        individual = create_test_individual()

        # 実行と検証 (単一目的)
        fitness_single = individual_evaluator.evaluate_individual(individual, config_single)
        assert fitness_single == (0.0,)

        # 実行と検証 (多目的)
        fitness_multi = individual_evaluator.evaluate_individual(individual, config_multi)
        assert fitness_multi == (0.0, 0.0)

    def test_evaluate_individual_no_trades(self, individual_evaluator, mock_backtest_service):
        """取引回数が0回のケース"""
        # 準備
        mock_backtest_service.run_backtest.return_value = {
            "performance_metrics": {"total_trades": 0}
        }
        config = create_test_ga_config()
        individual = create_test_individual()

        # 実行
        fitness = individual_evaluator.evaluate_individual(individual, config)

        # 検証
        assert fitness == (0.1,) # 取引回数0の場合はペナルティ値

    def test_evaluate_individual_violates_constraints(self, individual_evaluator, mock_backtest_service):
        """制約条件（最小取引回数、最大ドローダウン）に違反するケース"""
        # ケース1: 最小取引回数違反
        mock_backtest_service.run_backtest.return_value = {
            "performance_metrics": {"total_trades": 5, "total_return": 0.5, "sharpe_ratio": 2.0, "max_drawdown": 0.1}
        }
        config_min_trades = create_test_ga_config(min_trades=10)
        individual = create_test_individual()
        fitness = individual_evaluator.evaluate_individual(individual, config_min_trades)
        assert fitness == (0.0,)

        # ケース2: 最大ドローダウン違反
        mock_backtest_service.run_backtest.return_value = {
            "performance_metrics": {"total_trades": 20, "total_return": 0.5, "sharpe_ratio": 2.0, "max_drawdown": 0.5}
        }
        config_max_dd = create_test_ga_config(max_drawdown_limit=0.4)
        fitness = individual_evaluator.evaluate_individual(individual, config_max_dd)
        assert fitness == (0.0,)

    def test_extract_performance_metrics_invalid_values(self, individual_evaluator):
        """パフォーマンスメトリクスに無効な値が含まれるケース"""
        # 準備
        backtest_result = {
            "performance_metrics": {
                "total_return": None,       # 無効値
                "sharpe_ratio": float('inf'), # 無効値
                "max_drawdown": float('-inf'),# 無効値
                "win_rate": float('nan'),   # 無効値
                "profit_factor": "invalid", # 無効な型
                "total_trades": None
            }
        }

        # 実行
        metrics = individual_evaluator._extract_performance_metrics(backtest_result)

        # 検証
        assert metrics["total_return"] == 0.0
        assert metrics["sharpe_ratio"] == 0.0
        assert metrics["max_drawdown"] == 1.0 # -inf は最大リスクとして 1.0 になる
        assert metrics["win_rate"] == 0.0
        assert metrics["profit_factor"] == 0.0
        assert metrics["total_trades"] == 0

    def test_calculate_long_short_balance(self, individual_evaluator):
        """ロング・ショートのバランススコア計算のテスト"""
        # ケース1: バランスが取れている
        balanced_history = {"trade_history": [{"size": 1, "pnl": 10}, {"size": -1, "pnl": 10}]}
        score1 = individual_evaluator._calculate_long_short_balance(balanced_history)
        assert score1 > 0.8 # 理想的なので高いスコア

        # ケース2: ロングのみ
        long_only_history = {"trade_history": [{"size": 1, "pnl": 10}, {"size": 1, "pnl": 5}]}
        score2 = individual_evaluator._calculate_long_short_balance(long_only_history)
        assert 0.2 < score2 < 0.6 # 取引回数バランスが悪いのでスコアは下がる

        # ケース3: 取引なし
        no_trade_history = {"trade_history": []}
        score3 = individual_evaluator._calculate_long_short_balance(no_trade_history)
        assert score3 == 0.5 # 中立スコア

        # ケース4: 両方損失
        loss_history = {"trade_history": [{"size": 1, "pnl": -5}, {"size": -1, "pnl": -5}]}
        score4 = individual_evaluator._calculate_long_short_balance(loss_history)
        assert score4 < 0.65 # 利益バランスが悪いのでスコアは下がる (trade_balance=1.0, profit_balance=0.1 -> 0.64)
