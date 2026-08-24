"""資金費率 PnL 補正のテスト."""

from __future__ import annotations

import pandas as pd
import pytest

from app.services.auto_strategy.core.evaluation.funding_adjustment import (
    apply_funding_adjustment,
    compute_buy_hold_funding_drag,
    compute_trade_funding_drag,
)


@pytest.fixture
def funding_data() -> pd.DataFrame:
    """4バー・各バー資金費 0.01% のデータ（Close は期首100で漸増）。"""
    index = pd.date_range("2024-01-01", periods=4, freq="8h")
    return pd.DataFrame(
        {
            "Close": [100.0, 110.0, 120.0, 130.0],
            "funding_rate": [0.0001, 0.0001, 0.0001, 0.0001],
        },
        index=index,
    )


class TestBuyHoldDrag:
    """B&H ドラッグ計算"""

    def test_weighted_by_price_path(self, funding_data: pd.DataFrame) -> None:
        # rate 合計 0.0004、価格重み平均 = (1+1.1+1.2+1.3)/4 = 1.15
        drag = compute_buy_hold_funding_drag(funding_data)
        assert drag == pytest.approx(0.0001 * (1 + 1.1 + 1.2 + 1.3))

    def test_missing_column_returns_none(self) -> None:
        df = pd.DataFrame(
            {"Close": [1.0, 2.0]}, index=pd.date_range("2024-01-01", periods=2)
        )
        assert compute_buy_hold_funding_drag(df) is None

    def test_all_zero_rates_returns_none(self, funding_data: pd.DataFrame) -> None:
        df = funding_data.copy()
        df["funding_rate"] = 0.0
        assert compute_buy_hold_funding_drag(df) is None

    def test_four_hour_bars_halve_drag_due_to_merge_duplication(
        self, funding_data: pd.DataFrame
    ) -> None:
        """8hイベントが4h足に複製される場合、重複係数2で割り引かれる"""
        # 同じデータを 4h 間隔に再サンプルした体で index だけ置換
        df = funding_data.copy()
        df.index = pd.date_range("2024-01-01", periods=4, freq="4h")
        drag_4h = compute_buy_hold_funding_drag(df)
        drag_8h = compute_buy_hold_funding_drag(funding_data)
        assert drag_4h == pytest.approx(drag_8h / 2.0)


class TestTradeDrag:
    """トレード別ドラッグ計算"""

    def test_long_pays_positive_funding(self, funding_data: pd.DataFrame) -> None:
        trade = {
            "size": 2.0,
            "entry_price": 100.0,
            "entry_time": funding_data.index[0],
            "exit_time": funding_data.index[2],  # 2バー保有
        }
        cost, applied = compute_trade_funding_drag([trade], funding_data)
        # 名目 200 x (0.0001 * 2)
        assert cost == pytest.approx(200.0 * 0.0002)
        assert applied == 1

    def test_short_receives_funding(self, funding_data: pd.DataFrame) -> None:
        trade = {
            "size": -1.0,
            "entry_price": 100.0,
            "entry_time": funding_data.index[0],
            "exit_time": funding_data.index[1],
        }
        cost, _ = compute_trade_funding_drag([trade], funding_data)
        assert cost == pytest.approx(-100.0 * 0.0001)

    def test_open_trade_at_end_accrues_entry_bar(
        self, funding_data: pd.DataFrame
    ) -> None:
        """exit 未設定（保有継続）は最終バーまで加算される"""
        trade = {
            "size": 1.0,
            "entry_price": 130.0,
            "entry_time": funding_data.index[3],
            "exit_time": None,
        }
        cost, applied = compute_trade_funding_drag([trade], funding_data)
        assert cost == pytest.approx(130.0 * 0.0001)
        assert applied == 1

    def test_invalid_trade_skipped(self, funding_data: pd.DataFrame) -> None:
        trades = [
            {"size": 0.0, "entry_price": 100.0},
            {"size": 1.0, "entry_price": 0.0},
            {},
        ]
        cost, applied = compute_trade_funding_drag(trades, funding_data)
        assert cost == 0.0
        assert applied == 0


class TestApplyFundingAdjustment:
    """メトリクス適用"""

    def test_adjusts_total_and_buy_hold(self, funding_data: pd.DataFrame) -> None:
        result = {
            "performance_metrics": {"total_return": 10.0, "buy_hold_return": -20.0},
            "trade_history": [
                {
                    "size": 1.0,
                    "entry_price": 100.0,
                    "entry_time": funding_data.index[0],
                    "exit_time": funding_data.index[4]
                    if len(funding_data) > 4
                    else funding_data.index[-1],
                }
            ],
        }
        adjusted = apply_funding_adjustment(
            result, funding_data, initial_capital=1000.0
        )
        m = adjusted["performance_metrics"]
        # 戦略: 名目100 x 0.0003(3バー, exit=index[3]) / 1000 = 0.003%
        assert m["total_return"] == pytest.approx(10.0 - 0.003)
        # B&H: 全バー保持の加重分
        bh = compute_buy_hold_funding_drag(funding_data)
        assert m["buy_hold_return"] == pytest.approx(-20.0 - bh * 100.0)
        drag = m["funding_drag"]
        assert drag["trades_adjusted"] == 1

    def test_original_result_not_mutated(self, funding_data: pd.DataFrame) -> None:
        result = {"performance_metrics": {"total_return": 5.0}, "trade_history": []}
        apply_funding_adjustment(result, funding_data, initial_capital=1000.0)
        assert result["performance_metrics"]["total_return"] == 5.0
        assert "funding_drag" not in result["performance_metrics"]

    def test_noop_when_no_funding_column(self) -> None:
        df = pd.DataFrame(
            {"Close": [1.0]}, index=pd.date_range("2024-01-01", periods=1)
        )
        result = {"performance_metrics": {"total_return": 5.0}}
        adjusted = apply_funding_adjustment(result, df, initial_capital=1000.0)
        assert adjusted is result

    def test_noop_without_capital(self, funding_data: pd.DataFrame) -> None:
        result = {"performance_metrics": {"total_return": 5.0}}
        adjusted = apply_funding_adjustment(result, funding_data, initial_capital=0.0)
        assert adjusted is result

    def test_no_trades_still_adjusts_buy_hold(self, funding_data: pd.DataFrame) -> None:
        result = {"performance_metrics": {"total_return": 0.0, "buy_hold_return": 30.0}}
        adjusted = apply_funding_adjustment(
            result, funding_data, initial_capital=1000.0
        )
        m = adjusted["performance_metrics"]
        assert m["total_return"] == 0.0
        bh = compute_buy_hold_funding_drag(funding_data)
        assert m["buy_hold_return"] == pytest.approx(30.0 - bh * 100.0)
