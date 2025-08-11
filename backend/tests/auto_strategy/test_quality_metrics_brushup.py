from app.services.auto_strategy.utils.metrics import score_strategy_quality, passes_quality_threshold


def test_score_monotonicity_and_thresholds():
    base = {
        "Return [%]": 10.0,
        "Sharpe Ratio": 0.5,
        "Profit Factor": 1.2,
        "Max. Drawdown [%]": 20.0,
        "Win Rate [%]": 55.0,
    }
    s0 = score_strategy_quality(base)

    # 収益/シャープ/プロフィットファクターが上がればスコアは非減少
    better = base | {"Return [%]": 20.0, "Sharpe Ratio": 1.0, "Profit Factor": 1.5}
    s1 = score_strategy_quality(better)
    assert s1 >= s0

    # DDが悪化すればスコアは非増加（他一定）
    worse_dd = base | {"Max. Drawdown [%]": 40.0}
    s2 = score_strategy_quality(worse_dd)
    assert s2 <= s0

    # 閾値チェック
    assert passes_quality_threshold(base) is True
    low_pf = base | {"Profit Factor": 1.01}
    assert passes_quality_threshold(low_pf) is False
    high_dd = base | {"Max. Drawdown [%]": 40.1}
    assert passes_quality_threshold(high_dd) is False

