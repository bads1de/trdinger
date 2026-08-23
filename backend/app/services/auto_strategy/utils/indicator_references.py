"""
インジケーター参照名の共通ヘルパー
"""

from __future__ import annotations


def build_indicator_reference_name(
    indicator: object,
    output_index: int | None = None,
) -> str:
    """
    実行時に使う一意なインジケーター参照名を生成する。

    規約:
    - `TYPE`
    - `TYPE_TIMEFRAME`
    - `TYPE_TIMEFRAME_ID[:8]`
    - `TYPE_TIMEFRAME_ID[:8]_OUTPUT_INDEX`
    """
    parts = [str(getattr(indicator, "type", ""))]

    timeframe = getattr(indicator, "timeframe", None)
    if timeframe:
        parts.append(str(timeframe))

    indicator_id = getattr(indicator, "id", None)
    if indicator_id:
        parts.append(str(indicator_id)[:8])

    if output_index is not None:
        parts.append(str(output_index))

    return "_".join(part for part in parts if part)


def collect_indicator_references(node: object, out: set[str]) -> None:
    """条件ツリーから指標参照（{"type": "indicator", "name": ...}）を収集する。

    dataclass（Condition/ConditionGroup）と dict/list を再帰的に辿る。
    それ以外のオブジェクト（Mock 等のテストダブル含む）には立ち入らない。
    slots=True のデータクラスにも対応するため fields 経由で読む。
    """
    from dataclasses import fields, is_dataclass

    if isinstance(node, dict):
        name = node.get("name")
        if node.get("type") == "indicator" and isinstance(name, str):
            out.add(name)
        for value in node.values():
            collect_indicator_references(value, out)
    elif isinstance(node, (list, tuple)):
        for item in node:
            collect_indicator_references(item, out)
    elif is_dataclass(node) and not isinstance(node, type):
        for field_info in fields(node):
            value = getattr(node, field_info.name, None)
            collect_indicator_references(value, out)


def indicator_reference_names(indicator: object) -> set[str]:
    """指標遺伝子が取り得る参照名（単出力・複数出力インデックス込み）の集合。"""
    names = {build_indicator_reference_name(indicator)}
    names |= {
        build_indicator_reference_name(indicator, output_index)
        for output_index in range(4)
    }
    return names


def is_dead_indicator(indicator: object, referenced_names: set[str]) -> bool:
    """指標が条件式から参照されていない「死んだ遺伝子」か判定する。

    referenced_names が空（条件式が何も参照しない特殊なケース）の場合は
    判定不能として死んでいない扱いにする。
    """
    if not getattr(indicator, "enabled", True):
        return False
    if not referenced_names:
        return False
    return not (indicator_reference_names(indicator) & referenced_names)
