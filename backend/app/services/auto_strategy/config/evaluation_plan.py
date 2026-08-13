"""オートストラテジー評価計画の正規モデル。"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any, cast


@dataclass
class SelectionPlan:
    method: str = "is"
    holdout_ratio: float | None = None


@dataclass
class ValidationPlan:
    enabled: bool = True
    method: str = "rolling_holdout"
    folds: int = 5
    train_ratio: float = 0.7


@dataclass
class RobustnessPlan:
    enabled: bool = False
    scenarios: list[dict[str, Any]] = field(default_factory=list)
    policy: str = "gate_only"
    failure_policy: str = "fail_closed"


@dataclass
class EvaluationPlan:
    selection: SelectionPlan = field(default_factory=SelectionPlan)
    validation: ValidationPlan = field(default_factory=ValidationPlan)
    robustness: RobustnessPlan = field(default_factory=RobustnessPlan)
    warnings: list[str] = field(default_factory=list)

    def __post_init__(self) -> None:
        self._validate()

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> EvaluationPlan:
        if not isinstance(data, dict):
            raise ValueError("evaluation_plan は辞書である必要があります")
        return cls(
            selection=_build_dataclass(SelectionPlan, data.get("selection")),
            validation=_build_dataclass(ValidationPlan, data.get("validation")),
            robustness=_build_dataclass(RobustnessPlan, data.get("robustness")),
            warnings=[str(value) for value in data.get("warnings", []) or []],
        )

    @classmethod
    def from_legacy_config(
        cls,
        config: object,
    ) -> EvaluationPlan:
        evaluation_config = getattr(config, "evaluation_config", None)
        validation_config = getattr(config, "validation_config", None)
        robustness_config = getattr(config, "robustness_config", None)

        oos_ratio = _float_value(
            getattr(evaluation_config, "oos_split_ratio", 0.0), 0.0
        )
        enable_walk_forward = _bool_value(
            getattr(evaluation_config, "enable_walk_forward", False)
        )
        enable_purged_kfold = _bool_value(getattr(config, "enable_purged_kfold", False))
        validation_enabled = _bool_value(getattr(validation_config, "enabled", False))

        warnings: list[str] = []
        if (
            sum(
                int(value)
                for value in (enable_purged_kfold, enable_walk_forward, oos_ratio > 0.0)
            )
            > 1
        ):
            warnings.append(
                "legacy評価設定で複数の評価方式が有効です。EvaluationPlanではValidation方式を優先します。"
            )

        if validation_enabled:
            validation_method = "rolling_holdout"
        elif enable_purged_kfold:
            validation_method = "purged_kfold_diagnostic"
        elif enable_walk_forward:
            validation_method = "rolling_holdout"
        elif oos_ratio > 0.0:
            validation_method = "oos"
        else:
            validation_method = "none"

        if enable_purged_kfold:
            warnings.append(
                "現在のPurgedKFoldは診断用途として扱い、主Validation方式には使用しません。"
            )
        if (
            _float_value(getattr(evaluation_config, "oos_fitness_weight", 0.5), 0.5)
            != 0.5
            and oos_ratio > 0.0
        ):
            warnings.append("oos_fitness_weightはGAのIS-only評価では使用されません。")

        scenarios = _legacy_robustness_scenarios(robustness_config)
        explicit_robustness_enabled = _bool_value(
            getattr(robustness_config, "enabled", False)
        )
        validation_folds = _int_value(getattr(validation_config, "wfa_n_folds", 5), 5)
        validation_train_ratio = _float_value(
            getattr(validation_config, "wfa_train_ratio", 0.7), 0.7
        )
        return cls(
            selection=SelectionPlan(
                method="is",
                holdout_ratio=oos_ratio if oos_ratio > 0.0 else None,
            ),
            validation=ValidationPlan(
                enabled=validation_enabled,
                method=validation_method,
                folds=validation_folds,
                train_ratio=validation_train_ratio,
            ),
            robustness=RobustnessPlan(
                enabled=bool(scenarios) or explicit_robustness_enabled,
                scenarios=scenarios,
            ),
            warnings=warnings,
        )

    def apply_to_ga_config(self, config: object) -> None:
        """評価計画のrobustness設定をGAConfigの運用設定へ反映する。

        StrategyValidationService の robustness gate は
        ``config.robustness_config.enabled`` を参照するため、
        計画の ``robustness.enabled`` とシナリオを運用設定へ同期します。
        """
        robustness_config = getattr(config, "robustness_config", None)
        if robustness_config is None:
            return

        plan_robustness = self.robustness
        robustness_config.enabled = bool(plan_robustness.enabled)

        scenarios = plan_robustness.scenarios or []
        if not scenarios:
            return

        symbols: list[str] = []
        regime_windows: list[dict[str, Any]] = []
        slippage: list[float] = []
        commission: list[float] = []
        for scenario in scenarios:
            kind = str(scenario.get("type", ""))
            if kind == "symbol" and scenario.get("symbol"):
                symbols.append(str(scenario["symbol"]))
            elif kind == "regime":
                regime_windows.append(
                    {key: value for key, value in scenario.items() if key != "type"}
                )
            elif kind == "slippage":
                slippage.append(float(scenario.get("delta", 0.0)))
            elif kind == "commission":
                commission.append(float(scenario.get("multiplier", 1.0)))

        if symbols:
            robustness_config.validation_symbols = symbols
        if regime_windows:
            robustness_config.regime_windows = regime_windows
        if slippage:
            robustness_config.stress_slippage = slippage
        if commission:
            robustness_config.stress_commission_multipliers = commission

    def to_dict(self) -> dict[str, Any]:
        return cast(dict[str, Any], _json_safe(asdict(self)))

    def _validate(self) -> None:
        if self.selection.method != "is":
            raise ValueError("selection.method は 'is' である必要があります")
        if self.validation.method not in {
            "none",
            "oos",
            "rolling_holdout",
            "true_wfa",
            "purged_kfold_diagnostic",
        }:
            raise ValueError(f"未対応のValidation方式です: {self.validation.method}")
        if not 0.0 <= self.validation.train_ratio <= 1.0:
            raise ValueError("validation.train_ratioは0から1の範囲である必要があります")
        if self.validation.folds < 1:
            raise ValueError("validation.foldsは1以上である必要があります")
        if self.selection.holdout_ratio is not None and not (
            0.0 < self.selection.holdout_ratio < 1.0
        ):
            raise ValueError(
                "selection.holdout_ratioは0より大きく1未満である必要があります"
            )
        if self.robustness.policy != "gate_only":
            raise ValueError("robustness.policyは'gate_only'である必要があります")
        if self.robustness.failure_policy != "fail_closed":
            raise ValueError(
                "robustness.failure_policyは'fail_closed'である必要があります"
            )


def _build_dataclass(type_: type[Any], value: object) -> Any:
    if isinstance(value, type_):
        return value
    if not isinstance(value, dict):
        return type_()
    return type_(**value)


def _legacy_robustness_scenarios(config: object) -> list[dict[str, Any]]:
    if config is None:
        return []
    scenarios: list[dict[str, Any]] = []
    for symbol in getattr(config, "validation_symbols", None) or []:
        scenarios.append({"type": "symbol", "symbol": str(symbol)})
    for window in getattr(config, "regime_windows", None) or []:
        if isinstance(window, dict):
            scenarios.append({"type": "regime", **window})
    for delta in getattr(config, "stress_slippage", None) or []:
        scenarios.append({"type": "slippage", "delta": float(delta)})
    for multiplier in getattr(config, "stress_commission_multipliers", None) or []:
        scenarios.append({"type": "commission", "multiplier": float(multiplier)})
    return scenarios


def _json_safe(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(key): _json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_safe(item) for item in value]
    return value


def _bool_value(value: object) -> bool:
    if isinstance(value, bool):
        return value
    if isinstance(value, (int, float)):
        return bool(value)
    if isinstance(value, str):
        return value.strip().lower() in {"1", "true", "yes", "on"}
    return False


def _float_value(value: object, default: float) -> float:
    try:
        return float(cast(Any, value))
    except (TypeError, ValueError):
        return default


def _int_value(value: object, default: int) -> int:
    try:
        return int(cast(Any, value))
    except (TypeError, ValueError):
        return default
