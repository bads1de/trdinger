"""
ML gate 設定ヘルパー関数

volatility gate 設定の正規化・解決ヘルパーを提供します。
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any, Optional


@dataclass(frozen=True)
class MLGateSettings:
    """
    ML gate の有効状態とモデルパスを表す正規化済み設定

    volatility gateの設定を正規化して保持するデータクラスです。

    Attributes:
        enabled: ML gateが有効かどうか
        model_path: MLモデルのパス（無効時はNone）
    """

    enabled: bool
    model_path: Optional[str]


def _read_value(source: object, key: str) -> object:
    """
    dict / オブジェクトのどちらからでも値を取得する

    ソースがMappingの場合はgetメソッドを使用し、
    オブジェクトの場合はgetattrを使用して値を取得します。

    Args:
        source: 値を取得するソース（dictまたはオブジェクト）
        key: 取得するキー名

    Returns:
        Any: 取得した値、存在しない場合はNone
    """
    if isinstance(source, Mapping):
        return source.get(key)
    return getattr(source, key, None)


def _resolve_model_path(*candidates: Any) -> Optional[str]:
    """
    最初に見つかった有効なモデルパスを返す

    複数の候補から、最初に見つかった有効なモデルパスを返します。
    Noneまたは空文字列はスキップされます。

    Args:
        *candidates: モデルパスの候補リスト

    Returns:
        Optional[str]: 有効なモデルパス、見つからない場合はNone
    """
    for candidate in candidates:
        if candidate in (None, ""):
            continue
        return str(candidate)
    return None


def resolve_ml_gate_settings(source: Any) -> MLGateSettings:
    """
    volatility gate 設定を共通形に解決する

    ソースからvolatility gateの設定を解決して、
    正規化されたMLGateSettingsを返します。
    hybrid_configからの読み取りも試行します。

    Args:
        source: 設定ソース（dictまたはオブジェクト）

    Returns:
        MLGateSettings: 正規化されたML gate設定

    Note:
        hybrid_configが存在する場合は、そこからも設定を読み取ります。
    """
    gate_enabled = bool(_read_value(source, "volatility_gate_enabled"))
    model_path = _resolve_model_path(
        _read_value(source, "volatility_model_path"),
    )

    # hybrid_configからも読み取り（優先）
    hybrid_config = _read_value(source, "hybrid_config")
    if hybrid_config is not None:
        gate_enabled = gate_enabled or bool(
            _read_value(hybrid_config, "volatility_gate_enabled")
        )
        model_path = model_path or _resolve_model_path(
            _read_value(hybrid_config, "volatility_model_path"),
        )

    return MLGateSettings(enabled=gate_enabled, model_path=model_path)


def normalize_ml_gate_fields(source: Any) -> dict[str, Optional[str] | bool]:
    """
    volatility gate 設定を正規化する

    volatility gate設定を標準的な辞書形式に正規化します。

    Args:
        source: 設定ソース（dictまたはオブジェクト）

    Returns:
        dict[str, Optional[str] | bool]: 正規化された設定辞書
            - volatility_gate_enabled: ML gateが有効かどうか
            - volatility_model_path: MLモデルのパス
    """
    settings = resolve_ml_gate_settings(source)
    return {
        "volatility_gate_enabled": settings.enabled,
        "volatility_model_path": settings.model_path,
    }
