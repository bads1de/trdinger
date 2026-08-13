"""オートストラテジー評価計画の正規モデル（robustness 設定専用）。

GA 実行時には evaluation_config / validation_config / robustness_config が
実設定として使用されるため、EvaluationPlan は robustness gate 設定の
受け渡し・反映のみを担う軽量モデルです。
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field, fields
from typing import Any, cast


@dataclass
class RobustnessPlan:
    enabled: bool = False
    scenarios: list[dict[str, Any]] = field(default_factory=list)
    policy: str = "gate_only"
    failure_policy: str = "fail_closed"


@dataclass
class EvaluationPlan:
    """robustness gate 設定を運搬する評価計画。"""

    robustness: RobustnessPlan = field(default_factory=RobustnessPlan)

    def __post_init__(self) -> None:
        self._validate()

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> EvaluationPlan:
        """辞書から EvaluationPlan を生成する。未知キーは無視される。"""
        if not isinstance(data, dict):
            raise ValueError("evaluation_plan は辞書である必要があります")
        robustness = data.get("robustness")
        if isinstance(robustness, RobustnessPlan):
            return cls(robustness=robustness)
        if isinstance(robustness, dict):
            known = {field_info.name for field_info in fields(RobustnessPlan)}
            return cls(
                robustness=RobustnessPlan(
                    **{key: value for key, value in robustness.items() if key in known}
                )
            )
        return cls()

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
        if self.robustness.policy != "gate_only":
            raise ValueError("robustness.policyは'gate_only'である必要があります")
        if self.robustness.failure_policy != "fail_closed":
            raise ValueError(
                "robustness.failure_policyは'fail_closed'である必要があります"
            )


def _json_safe(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(key): _json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_safe(item) for item in value]
    return value
