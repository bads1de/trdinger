"""資金費率(PnL反映)補正ユーティリティ.

backtesting.py は資金費のような外部キャッシュフローを直接サポートしないため、
バックテスト実行後にトレード保有期間と Buy&Hold 全期間の資金費コストを
算出し、performance_metrics の total_return / buy_hold_return を補正する。

単位に関する注記:
- FractionalBacktest は価格を FRACTIONAL_UNIT 倍しサイズを逆数にするため、
  size x price (= 名目額) と PnL は実 USDT 空間でスケール不変。
- よって資金費コスト = rate x 名目額 の合計は PnL と同一単位で比較できる。
"""

from __future__ import annotations

import logging
from typing import Any

import pandas as pd

logger = logging.getLogger(__name__)

_METRIC_KEYS_ADJUSTED = ("total_return", "buy_hold_return")


def _funding_rate_series(data: pd.DataFrame) -> pd.Series | None:
    """資金費率系列を取得する（無い場合や全て無効な場合は None）。"""
    if data is None or "funding_rate" not in getattr(data, "columns", ()):
        return None
    series = pd.to_numeric(data["funding_rate"], errors="coerce").fillna(0.0)
    if float(series.abs().sum()) <= 0.0:
        return None
    return series


def _event_duplication_factor(index: pd.Index) -> float:
    """マージ済み資金費の重複係数を推定する。

    fr_merger は 8h イベントを tolerance=8h の merge_asof で結合するため、
    8h より短いタイムフレームでは同一イベントが複数バーに複製される。
    資金費は標準的に 8h 間隔で発生する前提で、
    factor = max(1.0, 8h / バー間隔) として合算値を割り引く。
    """
    try:
        dt_index = pd.DatetimeIndex(index)
        if len(dt_index) < 3:
            return 1.0
        median_step = dt_index.to_series().diff().median()
        hours = float(median_step / pd.Timedelta(hours=1))
        if not pd.notna(hours) or hours <= 0:
            return 1.0
        return max(1.0, 8.0 / hours)
    except Exception:  # noqa: BLE001
        return 1.0


def _close_series(data: pd.DataFrame) -> pd.Series | None:
    """終値系列を大文字小文字の揺れに耐えて取得する。"""
    columns = set(map(str, getattr(data, "columns", ())))
    for name in ("Close", "close"):
        if name in columns:
            return pd.to_numeric(data[name], errors="coerce")
    return None


def compute_buy_hold_funding_drag(data: pd.DataFrame) -> float | None:
    """期首購入して最後まで保持した場合の資金費ドラッグ（比率）。

    drag = sum(rate_t * Close_t / Close_0) / 重複係数

    スケール不変のため、Close が FRACTIONAL_UNIT 倍されていても影響しない。
    """
    series = _funding_rate_series(data)
    if series is None:
        return None
    close = _close_series(data)
    if close is None or close.empty or close.iloc[0] <= 0:
        return None
    weight = close / close.iloc[0]
    factor = _event_duplication_factor(data.index)
    return float((series * weight).sum() / factor)


def compute_trade_funding_drag(
    trades: list[dict[str, Any]], data: pd.DataFrame
) -> tuple[float, int]:
    """トレード毎の資金費コスト合計（実 USDT 相当）と適用件数を返す。

    ロング(size>0)は正の資金費を支払い、ショート(size<0)は受け取る。
    cost_i = sum(rate_t for t in [entry_time, exit_time)) * size_i * entry_price_i
    （size は符号付き。名目額はスケール不変のため PnL と同一単位で扱える）
    エントリーバー自体も保有開始としてカウントする（[entry, exit) 半開区間）。
    マージ済みデータの重複イベントは重複係数で割り引く。
    """
    series = _funding_rate_series(data)
    if series is None or not trades:
        return 0.0, 0

    index = pd.DatetimeIndex(data.index)
    rates = series.to_numpy(dtype=float)
    factor = _event_duplication_factor(data.index)

    total_cost = 0.0
    applied = 0
    for trade in trades:
        try:
            size = float(trade.get("size") or 0.0)
            entry_price = float(trade.get("entry_price") or 0.0)
            entry_time = trade.get("entry_time")
            exit_time = trade.get("exit_time")
        except (TypeError, ValueError):
            continue
        if size == 0.0 or entry_price <= 0.0:
            continue
        entry_ts = pd.Timestamp(entry_time) if entry_time is not None else None
        exit_ts = pd.Timestamp(exit_time) if exit_time is not None else None
        if entry_ts is None or pd.isna(entry_ts):
            continue

        start_pos = int(index.searchsorted(entry_ts, side="left"))
        if exit_ts is not None and not pd.isna(exit_ts):
            end_pos = int(index.searchsorted(exit_ts, side="left"))
        else:
            end_pos = len(index)
        if end_pos <= start_pos:
            continue
        held_rates = rates[start_pos:end_pos]
        if held_rates.size == 0:
            continue

        notional = size * entry_price
        total_cost += float(held_rates.sum()) * notional / factor
        applied += 1

    return total_cost, applied


def apply_funding_adjustment(
    result: dict[str, Any],
    data: pd.DataFrame | None,
    initial_capital: float,
) -> dict[str, Any]:
    """バックテスト結果に資金費補正を適用したコピーを返す。

    - total_return から戦略が保有中に支払った資金費を差し引く
    - buy_hold_return から期首購入ホールドの資金費を差し引く
    - 補正量は performance_metrics["funding_drag"] に記録する
    適用要件を満たさない場合は元の辞書をそのまま返す。
    """
    if not isinstance(result, dict) or not isinstance(data, pd.DataFrame):
        return result
    metrics = result.get("performance_metrics")
    if not isinstance(metrics, dict):
        return result
    if initial_capital <= 0:
        return result

    bh_drag = compute_buy_hold_funding_drag(data)
    trades = result.get("trade_history")
    trade_cost, applied = compute_trade_funding_drag(
        trades if isinstance(trades, list) else [], data
    )
    if bh_drag is None:
        return result

    strategy_drag_fraction = trade_cost / initial_capital
    adjusted = dict(result)
    adjusted_metrics = dict(metrics)

    raw_total = adjusted_metrics.get("total_return")
    raw_bh = adjusted_metrics.get("buy_hold_return")

    # メトリクスは百分率で保持されているため 100 を乗じる
    if isinstance(raw_total, (int, float)):
        adjusted_metrics["total_return"] = float(raw_total) - (
            strategy_drag_fraction * 100.0
        )
    if isinstance(raw_bh, (int, float)):
        adjusted_metrics["buy_hold_return"] = float(raw_bh) - (bh_drag * 100.0)

    drag_info = {
        "strategy_drag_pct": round(strategy_drag_fraction * 100.0, 6),
        "buy_hold_drag_pct": round(bh_drag * 100.0, 6),
        "trade_cost_usdt": round(trade_cost, 6),
        "trades_adjusted": applied,
        "adjusted_keys": list(_METRIC_KEYS_ADJUSTED),
    }
    adjusted_metrics["funding_drag"] = drag_info

    adjusted["performance_metrics"] = adjusted_metrics
    # extract_performance_metrics はホワイトリストのため、監査用に
    # トップレベルにも記録し、ScenarioEvaluation の metadata 経由で永続化する
    adjusted["_funding_adjustment"] = drag_info
    logger.debug(
        "資金費補正適用: strategy=%.4f%% bh=%.4f%% trades=%d",
        strategy_drag_fraction * 100.0,
        bh_drag * 100.0,
        applied,
    )
    return adjusted
