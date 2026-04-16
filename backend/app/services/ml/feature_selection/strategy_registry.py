"""
特徴量選択戦略のレジストリ

SelectionMethod と実際の戦略クラスの対応を1か所に集約します。
FeatureSelector と StagedStrategy はこのレジストリを参照します。
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Dict, List

from .config import SelectionMethod

if TYPE_CHECKING:
    from .strategies.base import BaseSelectionStrategy


def default_staged_methods() -> List[SelectionMethod]:
    """Staged 選択のデフォルト段階を返す。"""
    return SelectionMethod.default_staged_methods()


def build_staged_strategy_map() -> Dict[SelectionMethod, "BaseSelectionStrategy"]:
    """Staged 選択で使う戦略の対応表を返す。"""
    from .strategies.lasso import LassoStrategy
    from .strategies.permutation import PermutationStrategy
    from .strategies.rfecv import RFECVStrategy
    from .strategies.shadow import ShadowFeatureStrategy
    from .strategies.tree_based import TreeBasedStrategy
    from .strategies.univariate import UnivariateStrategy
    from .strategies.variance import VarianceStrategy

    return {
        SelectionMethod.VARIANCE: VarianceStrategy(),
        SelectionMethod.UNIVARIATE_F: UnivariateStrategy("f_classif"),
        SelectionMethod.UNIVARIATE_CHI2: UnivariateStrategy("chi2"),
        SelectionMethod.MUTUAL_INFO: UnivariateStrategy("mutual_info"),
        SelectionMethod.RFE: RFECVStrategy(),
        SelectionMethod.RFECV: RFECVStrategy(),
        SelectionMethod.LASSO: LassoStrategy(),
        SelectionMethod.RANDOM_FOREST: TreeBasedStrategy(),
        SelectionMethod.PERMUTATION: PermutationStrategy(),
        SelectionMethod.SHADOW: ShadowFeatureStrategy(),
    }


def build_selection_strategy_map() -> Dict[SelectionMethod, "BaseSelectionStrategy"]:
    """FeatureSelector 用の戦略対応表を返す。"""
    from .strategies.staged import StagedStrategy

    strategy_map = build_staged_strategy_map()
    strategy_map[SelectionMethod.STAGED] = StagedStrategy()
    return strategy_map


def get_selection_strategy(method: SelectionMethod) -> "BaseSelectionStrategy":
    """指定された手法に対応する戦略インスタンスを返す。"""
    strategy_map = build_selection_strategy_map()
    if method not in strategy_map:
        raise ValueError(f"Unknown selection method: {method}")
    return strategy_map[method]
