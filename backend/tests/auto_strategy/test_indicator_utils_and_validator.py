import importlib
import os
import sys
import types

import pytest

# Ensure backend cwd is on path when tests are run from various contexts
sys.path.insert(0, os.path.abspath("."))


def test_category_lists_dynamic():
    from app.services.auto_strategy.utils.indicator_utils import (
        get_volume_indicators,
        get_momentum_indicators,
        get_trend_indicators,
        get_volatility_indicators,
    )

    vols = get_volume_indicators()
    moms = get_momentum_indicators()
    trends = get_trend_indicators()
    volats = get_volatility_indicators()

    # 代表的な既存登録が含まれること（レジストリ初期化の検証も兼ねる）
    assert "OBV" in vols
    assert "RSI" in moms
    assert "SMA" in trends
    assert "ATR" in volats


def test_get_all_indicators_includes_ml():
    from app.services.auto_strategy.utils.indicator_utils import get_all_indicators

    all_inds = get_all_indicators()
    assert "ML_UP_PROB" in all_inds
    assert "ML_DOWN_PROB" in all_inds
    assert "ML_RANGE_PROB" in all_inds


def test_gene_validator_uses_dynamic_list(monkeypatch):
    # constants 側の VALID_INDICATOR_TYPES を空にしても、
    # GeneValidator が utils の動的リストに基づいて "SMA" を有効と判断できることを期待
    import app.services.auto_strategy.constants as consts

    monkeypatch.setattr(consts, "VALID_INDICATOR_TYPES", [], raising=False)

    # strategy_models を再読込して最新の依存を反映
    sm = importlib.import_module("app.services.auto_strategy.models.strategy_models")
    importlib.reload(sm)

    IndicatorGene = sm.IndicatorGene
    GeneValidator = sm.GeneValidator

    gene = IndicatorGene(type="SMA", parameters={})
    validator = GeneValidator()
    assert validator.validate_indicator_gene(gene), "SMA should be valid via dynamic registry"
