"""Indicator decorator consistency tests."""

from __future__ import annotations

import ast
from pathlib import Path

import pytest

INDICATORS_DIR = (
    Path(__file__).resolve().parents[3]
    / "app"
    / "services"
    / "indicators"
    / "technical_indicators"
)

PANDAS_TA_DIR = INDICATORS_DIR / "pandas_ta"

CLASS_MODULES = [
    ("momentum.py", "MomentumIndicators", set()),
    ("volatility.py", "VolatilityIndicators", set()),
    ("overlap.py", "OverlapIndicators", set()),
    ("trend.py", "TrendIndicators", set()),
    ("volume.py", "VolumeIndicators", set()),
    (
        "advanced_features.py",
        "AdvancedFeatures",
        {"get_weights_ffd", "z_score"},
    ),
]


def _decorator_name(node: ast.AST) -> str | None:
    if isinstance(node, ast.Name):
        return node.id
    if isinstance(node, ast.Attribute):
        return node.attr
    if isinstance(node, ast.Call):
        return _decorator_name(node.func)
    return None


def _public_methods_with_decorators(
    source: str, class_name: str
) -> list[tuple[str, set[str]]]:
    module = ast.parse(source)
    for class_node in module.body:
        if isinstance(class_node, ast.ClassDef) and class_node.name == class_name:
            methods: list[tuple[str, set[str]]] = []
            for node in class_node.body:
                if isinstance(node, ast.FunctionDef) and not node.name.startswith("_"):
                    decorators = {
                        name
                        for name in (_decorator_name(d) for d in node.decorator_list)
                        if name is not None
                    }
                    methods.append((node.name, decorators))
            return methods
    raise AssertionError(f"{class_name} が {INDICATORS_DIR} に見つかりません")


@pytest.mark.parametrize("filename,class_name,exempt_methods", CLASS_MODULES)
def test_public_indicator_methods_use_shared_error_decorator(
    filename: str,
    class_name: str,
    exempt_methods: set[str],
):
    module_dir = PANDAS_TA_DIR if filename != "advanced_features.py" else INDICATORS_DIR
    source = (module_dir / filename).read_text(encoding="utf-8")

    for method_name, decorators in _public_methods_with_decorators(source, class_name):
        if method_name in exempt_methods:
            continue

        assert "handle_pandas_ta_errors" in decorators, (
            f"{filename}:{class_name}.{method_name} に "
            f"@handle_pandas_ta_errors がありません: {sorted(decorators)}"
        )


@pytest.mark.parametrize(
    "module_path",
    sorted(
        path
        for path in (INDICATORS_DIR / "original").glob("*.py")
        if path.name not in {"__init__.py", "_window_helpers.py"}
    ),
)
def test_original_indicator_functions_use_shared_error_decorator(module_path: Path):
    source = module_path.read_text(encoding="utf-8")
    module = ast.parse(source)

    for node in module.body:
        if isinstance(node, ast.FunctionDef) and not node.name.startswith("_"):
            decorators = {
                name
                for name in (_decorator_name(d) for d in node.decorator_list)
                if name is not None
            }
            assert "handle_pandas_ta_errors" in decorators, (
                f"{module_path.name}:{node.name} に "
                f"@handle_pandas_ta_errors がありません: {sorted(decorators)}"
            )
