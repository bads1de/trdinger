import glob
import json
import logging
import sys
from pathlib import Path

logging.basicConfig(level=logging.WARNING)

sys.path.insert(0, str(Path(__file__).parent))

from app.services.auto_strategy.serializers.serialization import GeneSerializer
from app.services.auto_strategy.strategies.entry_decision_engine import (
    EntryDecisionEngine,
)
from app.services.backtest.services.backtest_service import BacktestService

files = sorted(glob.glob("results/auto_strategy/*.json"))
d = json.load(open(files[-1], encoding="utf-8"))
raw = d["best_strategy"]["raw_gene"]

serializer = GeneSerializer()
gene = serializer.dict_to_strategy_gene(raw, object)

calls = []

_orig = EntryDecisionEngine.calculate_position_size


def patched(self):
    equity = getattr(self.strategy, "equity", "N/A")
    try:
        equity_f = float(equity)
    except (TypeError, ValueError):
        equity_f = "N/A"
    price = self.strategy.data.Close[-1]
    result = _orig(self)
    calls.append({"equity": equity_f, "price": float(price), "result": float(result)})
    return result


EntryDecisionEngine.calculate_position_size = patched

bt_config = {
    "strategy_name": "universal_strategy",
    "symbol": "BTC/USDT:USDT",
    "timeframe": "4h",
    "start_date": "2024-01-01",
    "end_date": "2024-06-30",
    "initial_capital": 100000.0,
    "strategy_config": {
        "strategy_type": "GENERATED_GA",
        "parameters": {"strategy_gene": raw},
    },
    "commission_rate": 0.0004,
    "slippage": 0.0001,
}

service = BacktestService()
result = service.run_backtest(bt_config)

trades = result.get("trade_history", [])
print("trades:", len(trades))
print("first 3 trades sizes:", [t.get("size") for t in trades[:3]])
print("first 3 call logs:", calls[:3])
print("last call log:", calls[-1] if calls else None)
print("total sizing calls:", len(calls))

from collections import Counter

results = Counter(round(c["result"], 6) for c in calls)
print("sizing result distribution:", dict(list(results.items())[:10]))
