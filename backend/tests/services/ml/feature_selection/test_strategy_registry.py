"""
feature_selection.strategy_registry のテスト
"""

from app.services.ml.feature_selection.config import FeatureSelectionConfig, SelectionMethod
from app.services.ml.feature_selection.strategy_registry import (
    build_selection_strategy_map,
    build_staged_strategy_map,
    default_staged_methods,
)
from app.services.ml.feature_selection.strategies import (
    RFECVStrategy,
    StagedStrategy,
    TreeBasedStrategy,
    UnivariateStrategy,
    VarianceStrategy,
)


def test_default_staged_methods_are_centralized():
    assert default_staged_methods() == [
        SelectionMethod.VARIANCE,
        SelectionMethod.MUTUAL_INFO,
        SelectionMethod.RFECV,
    ]
    assert FeatureSelectionConfig().staged_methods == default_staged_methods()


def test_build_staged_strategy_map_excludes_staged():
    strategy_map = build_staged_strategy_map()

    assert SelectionMethod.STAGED not in strategy_map
    assert strategy_map[SelectionMethod.VARIANCE].__class__ is VarianceStrategy
    assert strategy_map[SelectionMethod.UNIVARIATE_F].__class__ is UnivariateStrategy
    assert strategy_map[SelectionMethod.RFECV].__class__ is RFECVStrategy
    assert strategy_map[SelectionMethod.RANDOM_FOREST].__class__ is TreeBasedStrategy


def test_build_selection_strategy_map_adds_staged():
    strategy_map = build_selection_strategy_map()

    assert SelectionMethod.STAGED in strategy_map
    assert strategy_map[SelectionMethod.STAGED].__class__ is StagedStrategy
    assert set(strategy_map) == set(build_staged_strategy_map()) | {
        SelectionMethod.STAGED
    }
