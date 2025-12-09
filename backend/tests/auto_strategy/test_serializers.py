import json
from typing import Any, Dict

import pytest

from backend.app.services.auto_strategy.models.strategy_models import (
    Condition,
    IndicatorGene,
    PositionSizingGene,
    PositionSizingMethod,
    StrategyGene,
    TPSLGene,
    TPSLMethod,
)
from backend.app.services.auto_strategy.serializers.gene_serialization import (
    GeneSerializer,
)


@pytest.fixture
def serializer() -> GeneSerializer:
    # enable_smart_generation=True がデフォルト挙動のため、それに従う
    return GeneSerializer(enable_smart_generation=True)


@pytest.fixture
def base_strategy_gene() -> StrategyGene:
    """シンプルだが代表的なStrategyGene."""
    indicators = [
        IndicatorGene(type="SMA", parameters={"period": 10}, enabled=True),
        IndicatorGene(type="EMA", parameters={"period": 20}, enabled=True),
    ]
    entry_conditions = [
        Condition(left_operand="close", operator=">", right_operand="sma"),
    ]
    exit_conditions = [
        Condition(left_operand="close", operator="<", right_operand="ema"),
    ]
    long_entry_conditions = entry_conditions
    short_entry_conditions = exit_conditions

    tpsl_gene = TPSLGene(
        enabled=True,
        method=TPSLMethod.RISK_REWARD_RATIO,
        stop_loss_pct=0.01,
        take_profit_pct=0.02,
        risk_reward_ratio=2.0,
        atr_multiplier_sl=1.5,
        atr_multiplier_tp=2.5,
        atr_period=14,
        lookback_period=100,
    )

    position_sizing_gene = PositionSizingGene(
        enabled=True,
        method=PositionSizingMethod.FIXED_QUANTITY,
        risk_per_trade=0.02,
        fixed_ratio=0.1,
        fixed_quantity=1000,
        atr_multiplier=1.5,
        optimal_f_multiplier=0.5,
        lookback_period=30,
        min_position_size=0.001,
    )

    return StrategyGene(
        id="test-strategy",
        indicators=indicators,
        entry_conditions=entry_conditions,
        exit_conditions=exit_conditions,
        long_entry_conditions=long_entry_conditions,
        short_entry_conditions=short_entry_conditions,
        risk_management={"position_size": 0.1234567, "max_risk": 0.05},
        tpsl_gene=tpsl_gene,
        position_sizing_gene=position_sizing_gene,
        metadata={"tag": "unit-test", "version": 1},
    )


# a. Round-trip 一貫性: Dict / JSON


def test_strategy_gene_round_trip_dict(
    serializer: GeneSerializer, base_strategy_gene: StrategyGene
) -> None:
    """DictConverter経由のラウンドトリップで主要フィールドが保持されること."""
    data = serializer.strategy_gene_to_dict(base_strategy_gene)
    restored = serializer.dict_to_strategy_gene(data, StrategyGene)

    assert isinstance(restored, StrategyGene)
    # ID
    assert restored.id == base_strategy_gene.id
    # 指標
    assert len(restored.indicators) == len(base_strategy_gene.indicators)
    for orig, got in zip(base_strategy_gene.indicators, restored.indicators):
        assert got.type == orig.type
        assert got.parameters == orig.parameters
        assert got.enabled is orig.enabled
    # 条件
    assert len(restored.entry_conditions) == len(base_strategy_gene.entry_conditions)
    assert len(restored.exit_conditions) == len(base_strategy_gene.exit_conditions)
    # TP/SL Gene
    assert isinstance(restored.tpsl_gene, TPSLGene)
    assert restored.tpsl_gene.enabled is True
    # 現行実装では Enum ではなく文字列として復元されるため、value ベースで比較
    assert str(restored.tpsl_gene.method) == str(
        base_strategy_gene.tpsl_gene.method.value
        if hasattr(base_strategy_gene.tpsl_gene.method, "value")
        else base_strategy_gene.tpsl_gene.method
    )
    assert restored.tpsl_gene.stop_loss_pct == pytest.approx(
        base_strategy_gene.tpsl_gene.stop_loss_pct
    )
    # PositionSizingGene
    assert isinstance(restored.position_sizing_gene, PositionSizingGene)
    # 現行実装では Enum ではなく文字列として復元されるため、value ベースで比較
    assert str(restored.position_sizing_gene.method) == str(
        base_strategy_gene.position_sizing_gene.method.value
        if hasattr(base_strategy_gene.position_sizing_gene.method, "value")
        else base_strategy_gene.position_sizing_gene.method
    )
    assert restored.position_sizing_gene.fixed_quantity == pytest.approx(
        base_strategy_gene.position_sizing_gene.fixed_quantity
    )
    # risk_management は DictConverter._clean_risk_management により加工される仕様
    assert "position_size" in data["risk_management"]
    assert data["risk_management"]["position_size"] == pytest.approx(0.123457, rel=1e-6)
    assert restored.risk_management.get("max_risk") == 0.05
    # metadata
    assert restored.metadata == base_strategy_gene.metadata


def test_strategy_gene_round_trip_json(
    serializer: GeneSerializer, base_strategy_gene: StrategyGene
) -> None:
    """JsonConverter経由のラウンドトリップ."""
    json_str = serializer.strategy_gene_to_json(base_strategy_gene)
    loaded: Dict[str, Any] = json.loads(json_str)
    assert loaded["id"] == base_strategy_gene.id

    restored = serializer.json_to_strategy_gene(json_str, StrategyGene)
    assert isinstance(restored, StrategyGene)
    assert restored.id == base_strategy_gene.id
    assert len(restored.indicators) == len(base_strategy_gene.indicators)
    assert restored.metadata == base_strategy_gene.metadata


def test_tpsl_gene_round_trip_via_dict(serializer: GeneSerializer) -> None:
    """TPSLGene.to_dict / from_dict を DictConverter 経由で固定."""
    gene = TPSLGene(
        enabled=True,
        method=TPSLMethod.RISK_REWARD_RATIO,
        stop_loss_pct=0.01,
        take_profit_pct=0.03,
        risk_reward_ratio=3.0,
        atr_multiplier_sl=1.2,
        atr_multiplier_tp=2.4,
        atr_period=20,
        lookback_period=200,
    )

    data = serializer.tpsl_gene_to_dict(gene)
    assert data is not None
    restored = serializer.dict_to_tpsl_gene(data)
    assert isinstance(restored, TPSLGene)
    assert restored.enabled is True
    # from_dict 側が method を文字列として扱う現行挙動を固定（後方互換のため）
    assert str(restored.method) == str(
        gene.method.value if hasattr(gene.method, "value") else gene.method
    )
    assert restored.stop_loss_pct == pytest.approx(gene.stop_loss_pct)
    assert restored.take_profit_pct == pytest.approx(gene.take_profit_pct)
    assert restored.risk_reward_ratio == pytest.approx(gene.risk_reward_ratio)


def test_position_sizing_gene_round_trip_via_dict(serializer: GeneSerializer) -> None:
    """PositionSizingGene.to_dict / from_dict を DictConverter 経由で固定."""
    gene = PositionSizingGene(
        enabled=True,
        method=PositionSizingMethod.VOLATILITY_BASED,
        risk_per_trade=0.03,
        fixed_ratio=0.2,
        fixed_quantity=0.0,
        atr_multiplier=1.8,
        optimal_f_multiplier=0.7,
        lookback_period=40,
        min_position_size=0.002,
    )

    data = serializer.position_sizing_gene_to_dict(gene)
    assert data is not None

    restored = serializer.dict_to_position_sizing_gene(data)
    assert isinstance(restored, PositionSizingGene)
    assert restored.enabled is True
    # from_dict 側が method を文字列として扱う現行挙動を固定（後方互換のため）
    assert str(restored.method) == str(
        gene.method.value if hasattr(gene.method, "value") else gene.method
    )
    assert restored.risk_per_trade == pytest.approx(gene.risk_per_trade)
    assert restored.min_position_size == pytest.approx(gene.min_position_size)


# b. 部分的/古いスキーマへの耐性


@pytest.mark.parametrize(
    "extra_field_key, extra_field_value",
    [
        ("unknown_field", "should_be_ignored"),
        ("legacy_flag", True),
    ],
)
def test_decoder_ignores_unknown_fields_and_applies_defaults(
    serializer: GeneSerializer,
    base_strategy_gene: StrategyGene,
    extra_field_key: str,
    extra_field_value: Any,
) -> None:
    """未知フィールド無視 + デフォルト補完挙動をドキュメント化."""
    data = serializer.strategy_gene_to_dict(base_strategy_gene)
    data[extra_field_key] = extra_field_value

    restored = serializer.dict_to_strategy_gene(data, StrategyGene)

    # 未知フィールドは StrategyGene コンストラクタに渡されない想定
    assert not hasattr(restored, extra_field_key)

    # long/short_entry_conditions 後方互換:
    # long/short が空で entry_conditions があれば entry_conditions をコピーする仕様
    # ここでは既に long/short があるケースも維持されることを確認
    assert len(restored.long_entry_conditions) >= 1
    assert len(restored.short_entry_conditions) >= 0


def test_decoder_uses_default_when_data_empty(serializer: GeneSerializer) -> None:
    """空dict入力時は GeneUtils.create_default_strategy_gene にフォールバックする仕様."""
    restored = serializer.dict_to_strategy_gene({}, StrategyGene)
    # 仕様上「デフォルトStrategyGene」を返す
    assert isinstance(restored, StrategyGene)
    assert restored.indicators  # デフォルトで少なくとも1つは存在することを期待


# c. エラー・バリデーション系


def test_invalid_tpsl_dict_raises_or_falls_back(serializer: GeneSerializer) -> None:
    """
    TPSL dict が明らかに不正な場合の挙動を固定.

    現行実装では TPSLGene.from_dict 内部仕様に依存するため、
    ValueError またはフォールバック(None)のいずれかを許容しつつ、
    例外が握りつぶされないことを確認する。
    """
    invalid_data = {"enabled": True, "method": "UNKNOWN"}  # 想定外
    try:
        result = serializer.dict_to_tpsl_gene(invalid_data)
    except ValueError:
        # 明示的なバリデーション例外が投げられる実装も許容
        return

    # 例外を投げない実装の場合、None やデフォルトTPSLGeneにフォールバックしていることを許容
    # TODO: clarify expected behavior
    assert result is None or isinstance(result, TPSLGene)


def test_invalid_position_sizing_dict_raises_or_falls_back(
    serializer: GeneSerializer,
) -> None:
    """
    PositionSizing dict が不正な場合の挙動.

    仕様が曖昧なため、ValueError または None / デフォルトへのフォールバックを許容。
    """
    invalid_data = {"enabled": True, "method": "UNKNOWN"}
    try:
        result = serializer.dict_to_position_sizing_gene(invalid_data)
    except ValueError:
        return

    # TODO: clarify expected behavior
    assert result is None or isinstance(result, PositionSizingGene)


# d. List/ネスト構造: ListEncoder/ListDecoder 安定性


def test_strategy_list_encoding_and_decoding_is_stable(
    serializer: GeneSerializer, base_strategy_gene: StrategyGene
) -> None:
    """
    ListEncoder.to_list / ListDecoder.from_list の往復で
    少なくとも基本構造が破綻しないこと。

    注意:
    - ListEncoder/ListDecoder は情報圧縮・ヒューリスティック生成を行うため、
      完全な同一オブジェクトにはならない設計。
    - ここでは「有効なStrategyGeneとして復元されること」「指標やTP/SL情報が一貫していること」を確認。
    """
    encoded = serializer.to_list(base_strategy_gene)
    assert isinstance(encoded, list)
    assert len(encoded) >= 10  # デフォルト設計に基づく最低長

    restored = serializer.from_list(encoded, StrategyGene)
    assert isinstance(restored, StrategyGene)

    # ListDecoderは必要に応じてデフォルト条件/リスク管理を生成する仕様
    assert restored.indicators  # 少なくとも1つ
    assert restored.entry_conditions
    assert restored.long_entry_conditions
    assert restored.short_entry_conditions
    assert isinstance(restored.risk_management, dict)

    # TP/SL および PositionSizing の有無はエンコード長等に依存するため、
    # enabled な場合は構造が妥当であることのみ確認
    if restored.tpsl_gene:
        assert isinstance(restored.tpsl_gene, TPSLGene)
    if restored.position_sizing_gene:
        assert isinstance(restored.position_sizing_gene, PositionSizingGene)


def test_list_decoder_fallback_to_default_on_short_list(
    serializer: GeneSerializer,
) -> None:
    """短すぎるエンコードリストではデフォルトStrategyGeneにフォールバックする仕様を固定."""
    restored = serializer.from_list([0.1, 0.2], StrategyGene)
    assert isinstance(restored, StrategyGene)
    # GeneUtils.create_default_strategy_gene による生成が想定されるため最低限の構造をチェック
    assert restored.indicators
    assert isinstance(restored.risk_management, dict)


def test_list_decoder_generates_metadata(serializer: GeneSerializer) -> None:
    """ListDecoder.from_list がメタデータを設定する現行仕様をテストで固定."""
    encoded = [0.5] * 40  # 十分な長さ
    restored = serializer.from_list(encoded, StrategyGene)

    assert isinstance(restored.metadata, dict)
    assert "generated_by" in restored.metadata
    assert "decoded_from_length" in restored.metadata


# 追加: indicator/condition 単体の変換


def test_indicator_gene_single_round_trip(serializer: GeneSerializer) -> None:
    """indicator_gene_to_dict / dict_to_indicator_gene のラウンドトリップ."""
    indicator = IndicatorGene(type="RSI", parameters={"period": 14}, enabled=False)
    data = serializer.indicator_gene_to_dict(indicator)
    restored = serializer.dict_to_indicator_gene(data)

    assert isinstance(restored, IndicatorGene)
    assert restored.type == "RSI"
    assert restored.parameters == {"period": 14}
    assert restored.enabled is False


def test_condition_round_trip(serializer: GeneSerializer) -> None:
    """condition_to_dict / dict_to_condition の整合性."""
    cond = Condition(left_operand="close", operator=">", right_operand="open")
    data = serializer.condition_to_dict(cond)
    restored = serializer.dict_to_condition(data)

    assert isinstance(restored, Condition)
    assert restored.left_operand == cond.left_operand
    assert restored.operator == cond.operator
    assert restored.right_operand == cond.right_operand


# e. マルチタイムフレーム（MTF）サポート


def test_indicator_gene_with_timeframe_round_trip(serializer: GeneSerializer) -> None:
    """タイムフレーム付き指標遺伝子のラウンドトリップ."""
    indicator = IndicatorGene(
        type="SMA", parameters={"period": 50}, enabled=True, timeframe="4h"
    )
    data = serializer.indicator_gene_to_dict(indicator)

    # timeframe がシリアライズされていることを確認
    assert "timeframe" in data
    assert data["timeframe"] == "4h"

    restored = serializer.dict_to_indicator_gene(data)

    assert isinstance(restored, IndicatorGene)
    assert restored.type == "SMA"
    assert restored.parameters == {"period": 50}
    assert restored.enabled is True
    assert restored.timeframe == "4h"


def test_indicator_gene_without_timeframe_round_trip(
    serializer: GeneSerializer,
) -> None:
    """タイムフレームなし（None）の指標遺伝子のラウンドトリップ."""
    indicator = IndicatorGene(
        type="RSI", parameters={"period": 14}, enabled=True, timeframe=None
    )
    data = serializer.indicator_gene_to_dict(indicator)

    # timeframe が None の場合はシリアライズに含まれない（後方互換性）
    assert "timeframe" not in data

    restored = serializer.dict_to_indicator_gene(data)

    assert isinstance(restored, IndicatorGene)
    assert restored.type == "RSI"
    assert restored.timeframe is None


def test_strategy_gene_with_mtf_indicators_round_trip(
    serializer: GeneSerializer,
) -> None:
    """マルチタイムフレーム指標を含む戦略遺伝子のラウンドトリップ."""
    indicators = [
        IndicatorGene(
            type="EMA", parameters={"period": 20}, enabled=True, timeframe=None
        ),  # デフォルトTF
        IndicatorGene(
            type="SMA", parameters={"period": 200}, enabled=True, timeframe="1d"
        ),  # 日足トレンド
        IndicatorGene(
            type="RSI", parameters={"period": 14}, enabled=True, timeframe="4h"
        ),  # 4時間足
    ]
    entry_conditions = [
        Condition(left_operand="close", operator=">", right_operand="EMA_20"),
    ]
    strategy_gene = StrategyGene(
        id="mtf-strategy-test",
        indicators=indicators,
        entry_conditions=entry_conditions,
        long_entry_conditions=entry_conditions,
        short_entry_conditions=[],
        exit_conditions=[],
        risk_management={},
        tpsl_gene=TPSLGene(enabled=True, method=TPSLMethod.FIXED_PERCENTAGE),
        metadata={"mtf_enabled": True},
    )

    data = serializer.strategy_gene_to_dict(strategy_gene)
    restored = serializer.dict_to_strategy_gene(data, StrategyGene)

    assert isinstance(restored, StrategyGene)
    assert len(restored.indicators) == 3

    # 各指標のタイムフレームが正しく復元されていることを確認
    assert restored.indicators[0].timeframe is None  # デフォルト
    assert restored.indicators[1].timeframe == "1d"  # 日足
    assert restored.indicators[2].timeframe == "4h"  # 4時間足
