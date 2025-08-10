import numpy as np
import pandas as pd
import pytest

from app.services.indicators import TechnicalIndicatorService
from app.services.indicators.config import indicator_registry
from app.services.auto_strategy.generators.smart_condition_generator import (
    SmartConditionGenerator,
)
from app.services.auto_strategy.models.gene_strategy import IndicatorGene, Condition


def make_ohlcv(n: int = 400) -> pd.DataFrame:
    # 合理的なレンジのダミーOHLCVを作成（必須カラムは先頭大文字）
    rng = np.random.default_rng(42)
    base = np.cumsum(rng.normal(0.0, 0.5, size=n)) + 100.0
    high = base + rng.uniform(0.2, 1.0, size=n)
    low = base - rng.uniform(0.2, 1.0, size=n)
    open_ = base + rng.normal(0, 0.3, size=n)
    close = base + rng.normal(0, 0.3, size=n)
    volume = rng.integers(100, 10000, size=n)
    idx = pd.date_range("2024-01-01", periods=n, freq="15min")
    df = pd.DataFrame(
        {
            "Open": open_.astype(float),
            "High": high.astype(float),
            "Low": low.astype(float),
            "Close": close.astype(float),
            "Volume": volume.astype(float),
        },
        index=idx,
    )
    return df


@pytest.mark.integration
def test_all_registered_indicators_calculate_without_error():
    df = make_ohlcv(300)
    svc = TechnicalIndicatorService()

    # 登録済みインジケータ一覧
    names = indicator_registry.list_indicators()

    # ML系はモデル非搭載前提のためスキップ
    names = [n for n in names if not n.startswith("ML_")]

    failures = []

    for name in names:
        cfg = indicator_registry.get_indicator_config(name)
        if cfg is None or cfg.adapter_function is None:
            # 未対応 or アダプタなしはスキップ
            continue

        # パラメータ: レジストリ定義のデフォルトを使用
        params = {pname: pconf.default_value for pname, pconf in cfg.parameters.items()}

        try:
            res = svc.calculate_indicator(df, name, params)
        except Exception as e:
            failures.append((name, f"exception: {e}"))
            continue

        # 返り値検証: 単一またはタプル、各要素の長さがlen(df)
        try:
            if isinstance(res, tuple):
                assert all(
                    hasattr(arr, "__len__") and len(arr) == len(df) for arr in res
                )
            else:
                assert hasattr(res, "__len__") and len(res) == len(df)
        except AssertionError:
            failures.append((name, f"invalid shape: {getattr(res, 'shape', None)}"))

    if failures:
        msg_lines = [f"{nm}: {reason}" for nm, reason in failures]
        pytest.fail(
            "Indicator calculation failures (first errors):\n" + "\n".join(msg_lines)
        )


@pytest.mark.integration
def test_smart_condition_generator_integration_with_registry():
    # レジストリに存在する（高確率で）基本的な指標とパターンを使用
    # 存在しない場合はスキップ
    candidate_genes = []
    for ind in [
        ("SMA", {"period": 20}),
        ("RSI", {"period": 14}),
        ("BB", {"period": 20}),
        ("CDL_ENGULFING", {}),
        ("CDL_DOJI", {}),
    ]:
        if indicator_registry.get_indicator_config(ind[0]) is not None:
            candidate_genes.append(
                IndicatorGene(type=ind[0], parameters=ind[1], enabled=True)
            )

    # 最低2つは使えるようにしておく
    if len(candidate_genes) < 2:
        pytest.skip("Not enough indicators available in registry for integration test")

    generator = SmartConditionGenerator(enable_smart_generation=True)
    long_conds, short_conds, exit_conds = generator.generate_balanced_conditions(
        candidate_genes
    )

    # 生成結果がリストであること
    assert isinstance(long_conds, list)
    assert isinstance(short_conds, list)
    assert isinstance(exit_conds, list)

    # 返された条件のうち、インジケータ名（str）のものはレジストリに存在すること
    def _is_valid_operand(x):
        if isinstance(x, str):
            base_fields = {"open", "high", "low", "close", "volume"}
            if x.strip().lower() in base_fields:
                return True
            return (
                indicator_registry.get_indicator_config(x) is not None
                or x.replace(" ", "").upper() in indicator_registry.list_indicators()
            )
        return True  # 数値や辞書はここでは許容

    for cond_list in (long_conds, short_conds):
        for c in cond_list:
            assert isinstance(c, Condition)
            assert c.operator in {">", "<", ">=", "<=", "==", "!=", "above", "below"}
            assert _is_valid_operand(c.left_operand)

    # ここではエラーなく条件生成が行われることを重視
