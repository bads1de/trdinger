from dataclasses import dataclass
from typing import Dict, Any


@dataclass
class SuccessStats:
    total: int
    with_trades: int

    @property
    def success_rate(self) -> float:
        return (self.with_trades / self.total) if self.total > 0 else 0.0


@dataclass
class QualityMetrics:
    return_pct: float
    sharpe: float
    profit_factor: float
    max_drawdown_pct: float
    win_rate_pct: float


def aggregate_success(cases: list[int]) -> SuccessStats:
    total = len(cases)
    with_trades = sum(1 for x in cases if x > 0)
    return SuccessStats(total=total, with_trades=with_trades)


def score_strategy_quality(stats: Dict[str, Any]) -> float:
    """
    戦略品質スコアの算出
    - 基本: 収益性（Return [%]）
    - リスク調整: Sharpe, MaxDD
    - 効率: Profit Factor
    - 安定性: Win Rate
    重みは経験的に設定、0-100スケールで返す
    """
    r = float(stats.get("Return [%]", stats.get("total_return", 0)))
    s = float(stats.get("Sharpe Ratio", stats.get("sharpe_ratio", 0)))
    pf = float(stats.get("Profit Factor", stats.get("profit_factor", 1)))
    dd = float(stats.get("Max. Drawdown [%]", stats.get("max_drawdown", 0)))
    wr = float(stats.get("Win Rate [%]", stats.get("win_rate", 0)))

    # 正規化/変換
    pf_c = max(min(pf, 3.0), 0.0) / 3.0  # 0..3 を 0..1
    s_c = max(min(s, 3.0), -1.0)
    s_c = (s_c + 1.0) / 4.0  # -1..3 を 0..1
    dd_c = max(min(dd, 50.0), 0.0)
    dd_c = 1.0 - (dd_c / 50.0)  # 0%が1、50%が0
    r_c = max(min(r, 100.0), -50.0)
    r_c = (r_c + 50.0) / 150.0  # -50..100 を 0..1
    wr_c = max(min(wr, 100.0), 0.0) / 100.0

    score_0_1 = 0.35 * r_c + 0.25 * s_c + 0.2 * pf_c + 0.1 * dd_c + 0.1 * wr_c
    return round(score_0_1 * 100.0, 2)


def passes_quality_threshold(stats: Dict[str, Any]) -> bool:
    """
    最低品質判定（成立率>=60%の母集団での二次選別用）
    - Sharpe >= 0.3
    - Profit Factor >= 1.05
    - MaxDD <= 35%
    """
    s = float(stats.get("Sharpe Ratio", stats.get("sharpe_ratio", 0)))
    pf = float(stats.get("Profit Factor", stats.get("profit_factor", 1)))
    dd = float(stats.get("Max. Drawdown [%]", stats.get("max_drawdown", 0)))
    return (s >= 0.3) and (pf >= 1.05) and (dd <= 35.0)
