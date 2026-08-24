"""compute_trade_fingerprint のテスト."""

from app.services.auto_strategy.core.evaluation.individual_evaluator import (
    compute_trade_fingerprint,
)


def _trade(entry: str, pnl: float) -> dict:
    return {
        "entry_time": entry,
        "exit_time": entry,
        "entry_price": 100.0,
        "exit_price": 101.0,
        "size": 1.0,
        "pnl": pnl,
        "return_pct": 0.01,
        "duration": 3600,
    }


class TestComputeTradeFingerprint:
    def test_identical_trades_produce_identical_hash(self):
        trades = [_trade("2026-01-01", 10.0), _trade("2026-01-02", -5.0)]
        assert compute_trade_fingerprint(trades) == compute_trade_fingerprint(
            [_trade("2026-01-01", 10.0), _trade("2026-01-02", -5.0)]
        )

    def test_trade_order_matters(self):
        a = [_trade("2026-01-01", 10.0), _trade("2026-01-02", -5.0)]
        b = list(reversed(a))
        assert compute_trade_fingerprint(a) != compute_trade_fingerprint(b)

    def test_tiny_float_noise_is_ignored(self):
        a = [_trade("2026-01-01", 10.0)]
        b = [_trade("2026-01-01", 10.0 + 1e-12)]
        assert compute_trade_fingerprint(a) == compute_trade_fingerprint(b)

    def test_different_pnl_changes_hash(self):
        assert compute_trade_fingerprint([_trade("2026-01-01", 10.0)]) != (
            compute_trade_fingerprint([_trade("2026-01-01", 11.0)])
        )

    def test_empty_or_invalid_inputs_return_none(self):
        assert compute_trade_fingerprint([]) is None
        assert compute_trade_fingerprint(None) is None
        assert compute_trade_fingerprint("not-a-list") is None
        assert compute_trade_fingerprint(["not-a-dict"]) is None

    def test_hash_is_short_hex(self):
        fp = compute_trade_fingerprint([_trade("2026-01-01", 1.0)])
        assert fp is not None
        assert len(fp) == 16
        int(fp, 16)  # 16進数であること
