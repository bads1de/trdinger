from app.services.auto_strategy.utils.metrics import aggregate_success


def test_aggregate_success_rate_calculation():
    stats = aggregate_success([0, 3, 1, 0, 2, 5])
    assert stats.total == 6
    assert stats.with_trades == 4
    assert abs(stats.success_rate - (4/6)) < 1e-9

