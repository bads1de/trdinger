"""
戦略遺伝子モデル

遺伝的アルゴリズムで使用する戦略の遺伝子表現を定義します。
v1仕様: 最大5指標、単純比較条件のみ
"""

from dataclasses import dataclass, field
from typing import List, Dict, Any, Union
import json
import logging

logger = logging.getLogger(__name__)


@dataclass
class IndicatorGene:
    """
    指標遺伝子

    単一のテクニカル指標の設定を表現します。
    v2拡張: OI/FRベースの指標を含む
    """

    type: str  # "SMA", "EMA", "RSI", "MACD", "OpenInterest", "FundingRate", etc.
    parameters: Dict[str, float]  # {"period": 20, "source": "close"}
    enabled: bool = True  # 使用するかどうか

    def validate(self) -> bool:
        """指標遺伝子の妥当性を検証"""
        if not self.type or not isinstance(self.type, str):
            return False
        if not isinstance(self.parameters, dict):
            return False

        # 有効な指標タイプの確認
        valid_indicator_types = [
            # 従来のテクニカル指標（価格・出来高ベース）
            "SMA",
            "EMA",
            "RSI",
            "MACD",
            "BB",
            "STOCH",
            "CCI",
            "WILLIAMS",
            "ADX",
            "AROON",
            "MFI",
            "MOMENTUM",
            "ROC",
            "ATR",
            "NATR",
            "TRANGE",
            "OBV",
            "AD",
            "ADOSC",
            "TEMA",
            "DEMA",
            "T3",
            "WMA",
            "KAMA",
            # Phase 3 新規追加指標
            "BOP",
            "PPO",
            "MIDPOINT",
            "MIDPRICE",
            "TRIMA",
            # Phase 4 新規追加指標
            "PLUS_DI",
            "MINUS_DI",
            "ROCP",
            "ROCR",
            "STOCHF",
            # 注意: OpenInterest, FundingRateは指標ではなく判断材料として条件で使用
        ]

        if self.type not in valid_indicator_types:
            return False

        # パラメータの妥当性確認
        if "period" in self.parameters:
            period = self.parameters["period"]
            if not isinstance(period, (int, float)) or period <= 0:
                return False

        return True


@dataclass
class Condition:
    """
    売買条件

    v2拡張: Open Interest (OI) と Funding Rate (FR) データソースを含む
    """

    left_operand: str  # "SMA_20", "RSI_14", "price", "OpenInterest", "FundingRate"
    operator: str  # ">", "<", "cross_above", "cross_below"
    right_operand: Union[
        str, float
    ]  # "SMA_50", 70, "OpenInterest", "FundingRate", etc.

    def validate(self) -> bool:
        """条件の妥当性を検証"""
        valid_operators = [">", "<", ">=", "<=", "==", "cross_above", "cross_below"]
        valid_data_sources = [
            "close",
            "open",
            "high",
            "low",
            "volume",  # 基本OHLCV
            "OpenInterest",
            "FundingRate",  # 新規追加データソース
        ]

        # オペレーターの検証
        if self.operator not in valid_operators:
            return False

        # オペランドの検証（指標名またはデータソース名）
        if isinstance(self.left_operand, str):
            # 指標名（例: "SMA_20"）またはデータソース名の場合
            if not (
                self._is_indicator_name(self.left_operand)
                or self.left_operand in valid_data_sources
            ):
                return False

        if isinstance(self.right_operand, str):
            # 指標名（例: "SMA_20"）またはデータソース名の場合
            if not (
                self._is_indicator_name(self.right_operand)
                or self.right_operand in valid_data_sources
            ):
                return False

        return True

    def _is_indicator_name(self, name: str) -> bool:
        """指標名かどうかを判定"""
        # 特別な指標名パターン
        if name == "BOP":
            # BOPは期間を使用しない
            return True

        # PPOの複数パラメータパターン: "PPO_12_26"
        if name.startswith("PPO_") and len(name.split("_")) == 3:
            parts = name.split("_")
            try:
                # 数値パラメータかチェック
                int(parts[1])  # fastperiod
                int(parts[2])  # slowperiod
                return True
            except ValueError:
                pass

        # STOCHFの複数パラメータパターン: "STOCHF_K_5_3", "STOCHF_D_5_3"
        if name.startswith("STOCHF_") and len(name.split("_")) == 4:
            parts = name.split("_")
            try:
                # STOCHF_K_fastk_fastd または STOCHF_D_fastk_fastd
                if parts[1] in ["K", "D"]:
                    int(parts[2])  # fastk_period
                    int(parts[3])  # fastd_period
                    return True
            except ValueError:
                pass

        # 指標名のパターン: "TYPE_PERIOD" (例: "SMA_20", "RSI_14")
        # または複合指標名: "OI_SMA_10", "FR_EMA_5"
        parts = name.split("_")
        if len(parts) >= 2:
            # PLUS_DI_14, MINUS_DI_14 のような3部構成の指標名
            if len(parts) == 3 and parts[0] in ["PLUS", "MINUS"] and parts[1] == "DI":
                return True

            # 通常の指標: "SMA_20"
            if len(parts) == 2:
                indicator_type = parts[0]
                valid_indicators = [
                    "SMA",
                    "EMA",
                    "RSI",
                    "MACD",
                    "BB",
                    "STOCH",
                    "CCI",
                    "WILLIAMS",
                    "ADX",
                    "AROON",
                    "MFI",
                    "MOMENTUM",
                    "ROC",
                    "ATR",
                    "NATR",
                    "TRANGE",
                    "OBV",
                    "AD",
                    "ADOSC",
                    "TEMA",
                    "DEMA",
                    "T3",
                    "WMA",
                    "KAMA",
                    # Phase 3 新規追加指標
                    "MIDPOINT",
                    "MIDPRICE",
                    "TRIMA",
                    # Phase 4 新規追加指標
                    "ROCP",
                    "ROCR",
                ]
                return indicator_type in valid_indicators

            # 注意: OI/FRベースの複合指標は使用しない
            # OI/FRは生の値として判断材料に使用

        return False


@dataclass
class StrategyGene:
    """
    戦略遺伝子

    完全な取引戦略の遺伝子表現。
    v1仕様: 最大5指標、単純条件のみ
    """

    id: str = ""
    indicators: List[IndicatorGene] = field(default_factory=list)
    entry_conditions: List[Condition] = field(default_factory=list)
    exit_conditions: List[Condition] = field(default_factory=list)
    risk_management: Dict[str, float] = field(default_factory=dict)
    metadata: Dict[str, Any] = field(default_factory=dict)

    # v1制約
    MAX_INDICATORS = 5

    def __post_init__(self):
        """初期化後の処理"""
        if not self.id:
            import uuid

            self.id = str(uuid.uuid4())[:8]

    def validate(self) -> tuple[bool, List[str]]:
        """
        戦略遺伝子の妥当性を検証

        Returns:
            (is_valid, error_messages)
        """
        errors = []

        # 指標数の制約チェック
        if len(self.indicators) > self.MAX_INDICATORS:
            errors.append(
                f"指標数が上限({self.MAX_INDICATORS})を超えています: {len(self.indicators)}"
            )

        # 指標の妥当性チェック
        for i, indicator in enumerate(self.indicators):
            if not indicator.validate():
                errors.append(f"指標{i}が無効です: {indicator.type}")

        # 条件の妥当性チェック
        for i, condition in enumerate(self.entry_conditions):
            if not condition.validate():
                errors.append(f"エントリー条件{i}が無効です")

        for i, condition in enumerate(self.exit_conditions):
            if not condition.validate():
                errors.append(f"イグジット条件{i}が無効です")

        # 最低限の条件チェック
        if not self.entry_conditions:
            errors.append("エントリー条件が設定されていません")
        if not self.exit_conditions:
            errors.append("イグジット条件が設定されていません")

        return len(errors) == 0, errors

    def to_dict(self) -> Dict[str, Any]:
        """辞書形式に変換（データベース保存用）"""
        return {
            "id": self.id,
            "indicators": [
                {"type": ind.type, "parameters": ind.parameters, "enabled": ind.enabled}
                for ind in self.indicators
            ],
            "entry_conditions": [
                {
                    "left_operand": cond.left_operand,
                    "operator": cond.operator,
                    "right_operand": cond.right_operand,
                }
                for cond in self.entry_conditions
            ],
            "exit_conditions": [
                {
                    "left_operand": cond.left_operand,
                    "operator": cond.operator,
                    "right_operand": cond.right_operand,
                }
                for cond in self.exit_conditions
            ],
            "risk_management": self.risk_management,
            "metadata": self.metadata,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "StrategyGene":
        """辞書から復元"""
        try:
            indicators = [
                IndicatorGene(
                    type=ind["type"],
                    parameters=ind["parameters"],
                    enabled=ind.get("enabled", True),
                )
                for ind in data.get("indicators", [])
            ]

            entry_conditions = [
                Condition(
                    left_operand=cond["left_operand"],
                    operator=cond["operator"],
                    right_operand=cond["right_operand"],
                )
                for cond in data.get("entry_conditions", [])
            ]

            exit_conditions = [
                Condition(
                    left_operand=cond["left_operand"],
                    operator=cond["operator"],
                    right_operand=cond["right_operand"],
                )
                for cond in data.get("exit_conditions", [])
            ]

            return cls(
                id=data.get("id", ""),
                indicators=indicators,
                entry_conditions=entry_conditions,
                exit_conditions=exit_conditions,
                risk_management=data.get("risk_management", {}),
                metadata=data.get("metadata", {}),
            )

        except Exception as e:
            logger.error(f"戦略遺伝子の復元に失敗: {e}")
            raise ValueError(f"Invalid strategy gene data: {e}")

    def to_json(self) -> str:
        """JSON文字列に変換"""
        return json.dumps(self.to_dict(), ensure_ascii=False, indent=2)

    @classmethod
    def from_json(cls, json_str: str) -> "StrategyGene":
        """JSON文字列から復元"""
        try:
            data = json.loads(json_str)
            return cls.from_dict(data)
        except json.JSONDecodeError as e:
            logger.error(f"JSON解析エラー: {e}")
            raise ValueError(f"Invalid JSON: {e}")


# v1仕様用のエンコード/デコード関数


def encode_gene_to_list(gene: StrategyGene) -> List[float]:
    """
    戦略遺伝子を固定長の数値リストにエンコード

    v1仕様: 最大5指標、単純条件のみ
    エンコード形式: [indicator1_id, param1_val, ..., entry_condition, exit_condition]
    """
    # 利用可能な指標ID（24種類）
    INDICATOR_IDS = {
        "": 0,  # 未使用
        "SMA": 1,
        "EMA": 2,
        "RSI": 3,
        "MACD": 4,
        "BB": 5,
        "STOCH": 6,
        "CCI": 7,
        "WILLIAMS": 8,
        "ADX": 9,
        "AROON": 10,
        "MFI": 11,
        "MOMENTUM": 12,
        "ROC": 13,
        "ATR": 14,
        "NATR": 15,
        "TRANGE": 16,
        "OBV": 17,
        "AD": 18,
        "ADOSC": 19,
        "TEMA": 20,
        "DEMA": 21,
        "T3": 22,
        "WMA": 23,
        "KAMA": 24,
    }

    encoded = []

    # 指標部分（5指標 × 2値 = 10要素）
    for i in range(StrategyGene.MAX_INDICATORS):
        if i < len(gene.indicators) and gene.indicators[i].enabled:
            indicator = gene.indicators[i]
            indicator_id = INDICATOR_IDS.get(indicator.type, 0)
            # パラメータを正規化（期間の場合は1-200を0-1に変換）
            param_val = _normalize_parameter(indicator.parameters.get("period", 20))
        else:
            indicator_id = 0  # 未使用
            param_val = 0.0

        encoded.extend([indicator_id, param_val])

    # エントリー条件（簡略化: 最初の条件のみ）
    if gene.entry_conditions:
        entry_cond = gene.entry_conditions[0]
        entry_encoded = _encode_condition(entry_cond)
    else:
        entry_encoded = [0, 0, 0]  # デフォルト

    # イグジット条件（簡略化: 最初の条件のみ）
    if gene.exit_conditions:
        exit_cond = gene.exit_conditions[0]
        exit_encoded = _encode_condition(exit_cond)
    else:
        exit_encoded = [0, 0, 0]  # デフォルト

    encoded.extend(entry_encoded)
    encoded.extend(exit_encoded)

    return encoded


def decode_list_to_gene(encoded: List[float]) -> StrategyGene:
    """
    固定長数値リストから戦略遺伝子にデコード
    """
    # 指標IDの逆引き（安全で実用的な指標のみ）
    ID_TO_INDICATOR = {
        0: "",
        1: "SMA",
        2: "EMA",
        3: "RSI",
        4: "WMA",
        5: "MOMENTUM",
        6: "ROC",
        # 複雑な指標は一旦除外
        # 7: "MACD",
        # 8: "BB",
        # 9: "STOCH",
        # 10: "CCI",
        # 11: "WILLIAMS",
        # 12: "ADX",
    }

    indicators = []

    # 指標部分をデコード
    for i in range(StrategyGene.MAX_INDICATORS):
        idx = i * 2
        if idx + 1 < len(encoded):
            # 指標IDを1-6の範囲にマッピング（安全な指標のみ）
            indicator_id = max(1, int(encoded[idx] * 6) + 1)
            param_val = encoded[idx + 1]

            indicator_type = ID_TO_INDICATOR.get(indicator_id, "")
            if indicator_type:
                period = _denormalize_parameter(
                    param_val, min_val=5, max_val=50
                )  # 5-50の範囲
                indicators.append(
                    IndicatorGene(
                        type=indicator_type,
                        parameters={"period": period},
                        enabled=True,
                    )
                )

    # 条件部分をデコード（確実に条件を生成）
    entry_start = StrategyGene.MAX_INDICATORS * 2
    exit_start = entry_start + 3

    # 生成された指標に基づいて条件を作成
    entry_conditions = []
    exit_conditions = []

    if indicators:
        # 最初の指標を使用して条件を作成
        first_indicator = indicators[0]
        indicator_name = (
            f"{first_indicator.type}_{first_indicator.parameters.get('period', 20)}"
        )

        # 指標タイプに応じた条件を生成
        if first_indicator.type == "RSI":
            entry_conditions = [
                Condition(left_operand=indicator_name, operator="<", right_operand=30)
            ]
            exit_conditions = [
                Condition(left_operand=indicator_name, operator=">", right_operand=70)
            ]
        elif first_indicator.type in ["SMA", "EMA", "WMA"]:
            # 移動平均の場合は価格との比較
            entry_conditions = [
                Condition(
                    left_operand="close", operator=">", right_operand=indicator_name
                )
            ]
            exit_conditions = [
                Condition(
                    left_operand="close", operator="<", right_operand=indicator_name
                )
            ]
        else:
            # その他の指標の場合は汎用的な条件
            entry_conditions = [
                Condition(
                    left_operand="close", operator=">", right_operand=indicator_name
                )
            ]
            exit_conditions = [
                Condition(
                    left_operand="close", operator="<", right_operand=indicator_name
                )
            ]
    else:
        # 指標がない場合はデフォルト条件（価格ベース）
        entry_conditions = [
            Condition(left_operand="close", operator=">", right_operand="close")
        ]
        exit_conditions = [
            Condition(left_operand="close", operator="<", right_operand="close")
        ]

    return StrategyGene(
        indicators=indicators,
        entry_conditions=entry_conditions,
        exit_conditions=exit_conditions,
    )


def _normalize_parameter(
    value: float, min_val: float = 1, max_val: float = 200
) -> float:
    """パラメータを0-1範囲に正規化"""
    return (value - min_val) / (max_val - min_val)


def _denormalize_parameter(
    normalized: float, min_val: float = 1, max_val: float = 200
) -> int:
    """正規化されたパラメータを元の範囲に戻す"""
    return int(min_val + normalized * (max_val - min_val))


def _encode_condition(condition: Condition) -> List[float]:
    """条件を数値リストにエンコード（簡略化）"""
    # 簡略化: 固定の条件パターンのみ
    return [1.0, 0.0, 1.0]  # プレースホルダー


def _decode_condition(encoded: List[float]) -> Condition:
    """数値リストから条件にデコード（実用的な条件を生成）"""
    import random

    # 実用的な条件パターンを生成（OI/FR対応版）
    patterns = [
        # RSI条件
        Condition(
            left_operand="RSI_14", operator="<", right_operand=30
        ),  # RSI oversold
        Condition(
            left_operand="RSI_14", operator=">", right_operand=70
        ),  # RSI overbought
        # SMA条件
        Condition(
            left_operand="SMA_10", operator=">", right_operand="SMA_20"
        ),  # SMA crossover
        Condition(
            left_operand="SMA_20", operator=">", right_operand="SMA_50"
        ),  # SMA trend
        # 価格条件
        Condition(
            left_operand="close", operator=">", right_operand="SMA_20"
        ),  # Price above SMA
        Condition(
            left_operand="close", operator="<", right_operand="SMA_20"
        ),  # Price below SMA
        # OI/FR条件（新規追加）
        Condition(
            left_operand="FundingRate", operator=">", right_operand=0.0005
        ),  # Funding rate bullish sentiment
        Condition(
            left_operand="FundingRate", operator="<", right_operand=-0.0005
        ),  # Funding rate bearish sentiment
        Condition(
            left_operand="OpenInterest", operator=">", right_operand=10000000
        ),  # High open interest
        Condition(
            left_operand="OpenInterest", operator="<", right_operand=5000000
        ),  # Low open interest
    ]

    # ランダムに条件を選択
    return random.choice(patterns)
