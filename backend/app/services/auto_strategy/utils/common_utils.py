"""
共通ユーティリティ関数

auto_strategy全体で使用される共通機能を提供します。
"""

import logging
import random
import yaml
from pathlib import Path
from typing import Any, Dict, List, Optional, Union, Tuple
from datetime import datetime
from dataclasses import dataclass
from abc import ABC, abstractmethod

from app.utils.error_handler import (
    ErrorHandler,
)


logger = logging.getLogger(__name__)


# =============================================================================
# Gene関連クラスの基底クラスと共通機能
# =============================================================================


@dataclass
class GeneParameter:
    """遺伝子パラメータ定義"""
    name: str
    default_value: Any
    param_type: type
    min_value: Optional[Union[int, float]] = None
    max_value: Optional[Union[int, float]] = None
    description: str = ""


class BaseGene(ABC):
    """
    遺伝子クラスの基底クラス

    PositionSizingGeneとTPSLGeneの共通機能を統合した抽象基底クラスです。
    to_dict(), from_dict(), validate() の共通実装を提供します。
    """

    def to_dict(self) -> Dict[str, Any]:
        """オブジェクトを辞書形式に変換"""
        result = {}
        for key, value in self.__dict__.items():
            if key.startswith('_'):
                continue  # プライベート属性は除外

            # Enumの処理
            if hasattr(value, 'value'):
                result[key] = value.value
            # datetimeの処理
            elif isinstance(value, datetime):
                result[key] = value.isoformat()
            # その他の値
            else:
                result[key] = value

        return result

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> Any:
        """辞書形式からオブジェクトを復元"""
        init_params = {}

        # クラスアノテーションからパラメータ情報を取得
        if hasattr(cls, '__annotations__'):
            annotations = cls.__annotations__

            for param_name, param_type in annotations.items():
                if param_name in data:
                    value = data[param_name]

                    # Enum型への変換
                    if hasattr(param_type, '__members__'):
                        # Enum型の場合はvalueから取得
                        if isinstance(value, str):
                            try:
                                init_params[param_name] = param_type(value)
                            except ValueError:
                                logger.warning(f"無効なEnum値 {value} を無視")
                        else:
                            init_params[param_name] = value
                    # datetime型への変換
                    elif param_type == datetime:
                        if isinstance(value, str):
                            try:
                                init_params[param_name] = datetime.fromisoformat(value)
                            except ValueError:
                                logger.warning(f"無効なdatetime値 {value} を無視")
                        else:
                            init_params[param_name] = value
                    # その他の型
                    else:
                        init_params[param_name] = value
                else:
                    logger.debug(f"パラメータ {param_name} がデータに存在しないためスキップ")

        return cls(**init_params)

    def validate(self) -> Tuple[bool, List[str]]:
        """遺伝子の妥当性を検証"""
        errors = []

        try:
            # 基本的な属性チェック
            if hasattr(self, 'enabled') and not isinstance(getattr(self, 'enabled', True), bool):
                errors.append("enabled属性がbool型である必要があります")

            # サブクラス固有の検証を呼び出し
            self._validate_parameters(errors)

        except Exception as e:
            errors.append(f"検証処理でエラーが発生: {e}")

        return len(errors) == 0, errors

    @abstractmethod
    def _validate_parameters(self, errors: List[str]) -> None:
        """サブクラスで固有のパラメータ検証を実装"""
        pass

    def _validate_range(self, value: Union[int, float], min_val: Union[int, float],
                       max_val: Union[int, float], param_name: str, errors: List[str]) -> bool:
        """範囲検証のヘルパー関数"""
        if not (min_val <= value <= max_val):
            errors.append(f"{param_name}は{min_val}-{max_val}の範囲である必要があります")
            return False
        return True


class DataConverter:
    """データ変換ユーティリティ"""

    @staticmethod
    def ensure_float(value: Any, default: float = 0.0) -> float:
        """値をfloatに安全に変換"""
        try:
            return float(value)
        except (ValueError, TypeError):
            logger.warning(f"float変換失敗: {value}, デフォルト値 {default} を使用")
            return default

    @staticmethod
    def ensure_int(value: Any, default: int = 0) -> int:
        """値をintに安全に変換"""
        try:
            return int(value)
        except (ValueError, TypeError):
            logger.warning(f"int変換失敗: {value}, デフォルト値 {default} を使用")
            return default

    @staticmethod
    def ensure_list(value: Any, default: Optional[List] = None) -> List:
        """値をリストに安全に変換"""
        if default is None:
            default = []

        if isinstance(value, list):
            return value
        elif value is None:
            return default
        else:
            return [value]

    @staticmethod
    def ensure_dict(value: Any, default: Optional[Dict] = None) -> Dict:
        """値を辞書に安全に変換"""
        if default is None:
            default = {}

        if isinstance(value, dict):
            return value
        else:
            return default

    @staticmethod
    def normalize_symbol(symbol: str) -> str:
        """シンボルを正規化"""
        if not symbol:
            return "BTC:USDT"

        if ":" not in symbol:
            return f"{symbol}:USDT"

        return symbol


class LoggingUtils:
    """ログ出力ユーティリティ"""

    @staticmethod
    def log_performance(operation: str, duration: float, **metrics):
        """パフォーマンスログ"""
        metrics_str = ", ".join([f"{k}={v}" for k, v in metrics.items()])
        logger.info(f"[PERF] {operation}: {duration:.3f}s, {metrics_str}")


class ValidationUtils:
    """バリデーションユーティリティ"""

    @staticmethod
    def validate_range(
        value: Union[int, float],
        min_val: Union[int, float],
        max_val: Union[int, float],
        name: str = "値",
    ) -> bool:
        """範囲バリデーション"""
        if not (min_val <= value <= max_val):
            logger.warning(f"{name}が範囲外です: {value} (範囲: {min_val}-{max_val})")
            return False
        return True

    @staticmethod
    def validate_required_fields(
        data: Dict[str, Any], required_fields: List[str]
    ) -> tuple[bool, List[str]]:
        """必須フィールドバリデーション"""
        missing_fields = []
        for field in required_fields:
            if field not in data or data[field] is None:
                missing_fields.append(field)

        if missing_fields:
            logger.warning(f"必須フィールドが不足しています: {missing_fields}")
            return False, missing_fields

        return True, []


class PerformanceUtils:
    """パフォーマンス測定ユーティリティ"""

    @staticmethod
    def time_function(func):
        """関数実行時間測定デコレータ"""
        import time
        from functools import wraps

        @wraps(func)
        def wrapper(*args, **kwargs):
            start_time = time.time()
            try:
                result = func(*args, **kwargs)
                duration = time.time() - start_time
                LoggingUtils.log_performance(func.__name__, duration)
                return result
            except Exception:
                duration = time.time() - start_time
                LoggingUtils.log_performance(f"{func.__name__} (ERROR)", duration)
                raise

        return wrapper


class CacheUtils:
    """キャッシュユーティリティ"""

    _cache = {}

    @classmethod
    def get(cls, key: str, default: Any = None) -> Any:
        """キャッシュから値を取得"""
        return cls._cache.get(key, default)

    @classmethod
    def set(cls, key: str, value: Any, ttl: Optional[int] = None) -> None:
        """キャッシュに値を設定"""
        cls._cache[key] = {"value": value, "timestamp": datetime.now(), "ttl": ttl}

    @classmethod
    def clear(cls, pattern: Optional[str] = None) -> None:
        """キャッシュをクリア"""
        if pattern:
            keys_to_remove = [k for k in cls._cache.keys() if pattern in k]
            for key in keys_to_remove:
                del cls._cache[key]
        else:
            cls._cache.clear()


class GeneticUtils:
    """遺伝的アルゴリズム関連ユーティリティ"""

    @staticmethod
    def create_child_metadata(
        parent1_metadata: Dict[str, Any],
        parent2_metadata: Dict[str, Any],
        parent1_id: str,
        parent2_id: str
    ) -> Tuple[Dict[str, Any], Dict[str, Any]]:
        """
        交叉子のメタデータを生成

        Args:
            parent1_metadata: 親1のメタデータ
            parent2_metadata: 親2のメタデータ
            parent1_id: 親1のID
            parent2_id: 親2のID

        Returns:
            (child1_metadata, child2_metadata) のタプル
        """
        # 子1のメタデータ：親1のメタデータを継承し、親情報を追加
        child1_metadata = parent1_metadata.copy()
        child1_metadata["crossover_parent1"] = parent1_id
        child1_metadata["crossover_parent2"] = parent2_id

        # 子2のメタデータ：親2のメタデータを継承し、親情報を追加
        child2_metadata = parent2_metadata.copy()
        child2_metadata["crossover_parent1"] = parent1_id
        child2_metadata["crossover_parent2"] = parent2_id

        return child1_metadata, child2_metadata

    @staticmethod
    def prepare_crossover_metadata(strategy_parent1, strategy_parent2):
        """
        戦略遺伝子交叉用のメタデータ作成ユーティリティ

        Args:
            strategy_parent1: 親1のStrategyGene
            strategy_parent2: 親2のStrategyGene

        Returns:
            (child1_metadata, child2_metadata) のタプル
        """
        return GeneticUtils.create_child_metadata(
            parent1_metadata=strategy_parent1.metadata,
            parent2_metadata=strategy_parent2.metadata,
            parent1_id=strategy_parent1.id,
            parent2_id=strategy_parent2.id
        )

    @staticmethod
    def crossover_generic_genes(
        parent1_gene,
        parent2_gene,
        gene_class,
        numeric_fields: Optional[List[str]] = None,
        enum_fields: Optional[List[str]] = None,
        choice_fields: Optional[List[str]] = None,
        list_fields: Optional[List[str]] = None
    ):
        """
        ジェネリック遺伝子交叉関数

        Args:
            parent1_gene: 親1遺伝子
            parent2_gene: 親2遺伝子
            gene_class: 遺伝子クラス
            numeric_fields: 数値フィールドのリスト（平均化）
            enum_fields: Enumフィールドのリスト（ランダム選択）
            choice_fields: どちらかを選択するフィールドのリスト
            list_fields: リストフィールドのリスト（未使用）

        Returns:
            (child1_gene, child2_gene) のタプル
        """
        if numeric_fields is None:
            numeric_fields = []
        if enum_fields is None:
            enum_fields = []
        if choice_fields is None:
            choice_fields = []

        # 共通のフィールド処理
        parent1_dict = parent1_gene.__dict__.copy()
        parent2_dict = parent2_gene.__dict__.copy()

        child1_params = {}
        child2_params = {}

        # 全フィールドを取得
        all_fields = set(parent1_dict.keys()) & set(parent2_dict.keys())
        all_fields.discard('_')  # プライベート属性を除外

        for field in all_fields:
            if field in choice_fields:
                # どちらかを選択
                child1_params[field] = random.choice([parent1_dict[field], parent2_dict[field]])
                child2_params[field] = random.choice([parent1_dict[field], parent2_dict[field]])
            elif field in numeric_fields:
                # 数値フィールドは平均化
                val1 = parent1_dict[field]
                val2 = parent2_dict[field]
                if isinstance(val1, (int, float)) and isinstance(val2, (int, float)):
                    child1_params[field] = (val1 + val2) / 2
                    child2_params[field] = (val1 + val2) / 2
                else:
                    child1_params[field] = val1
                    child2_params[field] = val2
            elif field in enum_fields:
                # Enumフィールドはランダム選択
                child1_params[field] = random.choice([parent1_dict[field], parent2_dict[field]])
                child2_params[field] = random.choice([parent1_dict[field], parent2_dict[field]])
            else:
                # デフォルト：交互に選択
                if random.random() < 0.5:
                    child1_params[field] = parent1_dict[field]
                    child2_params[field] = parent2_dict[field]
                else:
                    child1_params[field] = parent2_dict[field]
                    child2_params[field] = parent1_dict[field]

        # 子遺伝子作成
        child1 = gene_class(**child1_params)
        child2 = gene_class(**child2_params)

        return child1, child2

    @staticmethod
    def mutate_generic_gene(
        gene,
        gene_class,
        mutation_rate: float = 0.1,
        numeric_fields: Optional[List[str]] = None,
        enum_fields: Optional[List[str]] = None,
        numeric_ranges: Optional[Dict[str, Tuple[float, float]]] = None
    ):
        """
        ジェネリック遺伝子突然変異関数

        Args:
            gene: 突然変異対象の遺伝子
            gene_class: 遺伝子クラス
            mutation_rate: 突然変異率
            numeric_fields: 数値フィールドのリスト
            enum_fields: Enumフィールドのリスト
            numeric_ranges: 数値フィールドの範囲 {"field_name": (min_val, max_val)}

        Returns:
            突然変異後の遺伝子
        """
        import random

        if numeric_fields is None:
            numeric_fields = []
        if enum_fields is None:
            enum_fields = []
        if numeric_ranges is None:
            numeric_ranges = {}

        # 遺伝子のコピーを作成
        mutated_params = gene.__dict__.copy()

        # 数値フィールドの突然変異
        for field in numeric_fields:
            if random.random() < mutation_rate and field in mutated_params:
                current_value = mutated_params[field]
                if isinstance(current_value, (int, float)):
                    # 値の範囲を取得（指定がない場合はデフォルト範囲）
                    min_val, max_val = numeric_ranges.get(field, (0, 100))

                    # 現在の値を中心とした範囲で突然変異
                    mutation_factor = random.uniform(0.8, 1.2)
                    new_value = current_value * mutation_factor

                    # 範囲内に制限
                    new_value = max(min_val, min(max_val, new_value))

                    mutated_params[field] = new_value

                    # int型だった場合はintに変換
                    if isinstance(current_value, int):
                        mutated_params[field] = int(new_value)

        # Enumフィールドの突然変異
        for field in enum_fields:
            if random.random() < mutation_rate and field in mutated_params:
                current_enum = mutated_params[field]
                if hasattr(current_enum, '__class__'):
                    enum_class = current_enum.__class__
                    # ランダムに別の値を選択
                    mutated_params[field] = random.choice(list(enum_class))

        return gene_class(**mutated_params)

class YamlLoadUtils:
    """YAMLローディングユーティリティ"""

    @staticmethod
    def load_yaml_config(config_path: Union[str, Path], fallback: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """YAML設定ファイルを安全に読み込み

        Args:
            config_path: YAMLファイルのパス
            fallback: 読み込み失敗時のフォールバックデータ

        Returns:
            読み込んだ設定データ
        """
        if fallback is None:
            fallback = {"indicators": {}}

        try:
            path = Path(config_path) if isinstance(config_path, str) else config_path

            if not path.exists():
                logger.warning(f"YAML設定ファイルが見つかりません: {path}")
                return fallback

            with open(path, 'r', encoding='utf-8') as f:
                config = yaml.safe_load(f)

            if config is None:
                logger.warning(f"YAML設定ファイルが空です: {path}")
                return fallback

            # 基本構造の検証
            if not isinstance(config, dict):
                logger.error(f"無効なYAML構造: {path}")
                return fallback

            return config

        except yaml.YAMLError as e:
            logger.error(f"YAML構文エラー: {path}, {e}")
            return fallback
        except Exception as e:
            logger.error(f"YAML読み込みエラー: {path}, {e}")
            return fallback

    @staticmethod
    def validate_yaml_config(config: Dict[str, Any]) -> tuple[bool, List[str]]:
        """YAML設定データの妥当性を検証

        Args:
            config: 検証対象の設定データ

        Returns:
            (妥当性, エラーメッセージリスト) のタプル
        """
        errors = []

        try:
            # indicators セクションの存在確認
            if "indicators" not in config:
                errors.append("indicatorsセクションが必須です")

            indicators = config.get("indicators", {})

            # 各indicatorの構造検証
            for indicator_name, indicator_config in indicators.items():
                if not isinstance(indicator_config, dict):
                    errors.append(f"indicator {indicator_name}: 辞書形式である必要があります")
                    continue

                # 必須フィールドチェック
                required_fields = ["type", "scale_type", "thresholds", "conditions"]
                for field in required_fields:
                    if field not in indicator_config:
                        errors.append(f"indicator {indicator_name}: {field}フィールドが必須です")

                # conditionsの検証
                conditions = indicator_config.get("conditions", {})
                if isinstance(conditions, dict):
                    for side in ["long", "short"]:
                        if side not in conditions:
                            continue  # オプション
                        condition_template = conditions[side]
                        if not isinstance(condition_template, str):
                            errors.append(f"indicator {indicator_name}: {side}条件は文字列テンプレートである必要があります")

        except Exception as e:
            errors.append(f"設定検証エラー: {e}")

        return len(errors) == 0, errors

    @staticmethod
    def get_indicator_config(yaml_config: Dict[str, Any], indicator_name: str) -> Optional[Dict[str, Any]]:
        """YAML設定から特定のindicator設定を取得

        Args:
            yaml_config: YAML設定データ
            indicator_name: 取得対象のindicator名

        Returns:
            indicator設定
        """
        indicators = yaml_config.get("indicators", {})
        return indicators.get(indicator_name)

    @staticmethod
    def get_all_indicator_names(yaml_config: Dict[str, Any]) -> List[str]:
        """YAML設定に含まれる全indicator名を取得

        Args:
            yaml_config: YAML設定データ

        Returns:
            indicator名リスト
        """
        indicators = yaml_config.get("indicators", {})
        return list(indicators.keys())


class YamlTestUtils:
    """YAML条件生成テストユーティリティ"""

    @staticmethod
    def test_yaml_based_conditions(
        yaml_config: Dict[str, Any],
        condition_generator_class,
        test_indicators: Optional[List[str]] = None
    ) -> Dict[str, Any]:
        """YAMLベースの条件生成をテスト

        Args:
            yaml_config: YAML設定データ
            condition_generator_class: テスト対象の条件generatorクラス
            test_indicators: テスト対象のindicators（指定なしの場合は全て）

        Returns:
            テスト結果
        """
        result = {
            "success": False,
            "tested_indicators": [],
            "errors": [],
            "summary": {}
        }

        try:
            # ConditionGeneratorの作成
            generator = condition_generator_class()
            generator.yaml_config = yaml_config  # YAML設定をセット

            # テスト対象のindicators決定
            if test_indicators:
                indicators_to_test = test_indicators
            else:
                indicators_to_test = YamlLoadUtils.get_all_indicator_names(yaml_config)

            # 各indicatorでテスト実行
            successful_tests = 0
            total_tests = 0

            for indicator_name in indicators_to_test:
                try:
                    # Mock IndicatorGene作成
                    mock_indicator = MockIndicatorGene(indicator_name, enabled=True)

                    # YAML設定取得
                    config = YamlLoadUtils.get_indicator_config(yaml_config, indicator_name)
                    if not config:
                        result["errors"].append(f"YAML設定なし: {indicator_name}")
                        continue

                    # 条件生成テスト
                    long_conditions = generator._generate_yaml_based_conditions(mock_indicator, "long")
                    short_conditions = generator._generate_yaml_based_conditions(mock_indicator, "short")

                    result["tested_indicators"].append({
                        "name": indicator_name,
                        "long_conditions_count": len(long_conditions),
                        "short_conditions_count": len(short_conditions),
                        "type": config.get("type", "unknown")
                    })

                    successful_tests += 1
                    total_tests += 1

                except Exception as e:
                    result["errors"].append(f"{indicator_name} テスト失敗: {e}")
                    total_tests += 1

            result["success"] = successful_tests == total_tests and total_tests > 0
            result["summary"] = {
                "total_tested": total_tests,
                "successful": successful_tests,
                "success_rate": successful_tests / total_tests if total_tests > 0 else 0.0
            }

        except Exception as e:
            result["errors"].append(f"全体テスト失敗: {e}")

        return result


class MockIndicatorGene:
    """テスト用のモックIndicatorGene"""
    def __init__(self, type: str, enabled: bool = True, parameters: Dict[str, Any] = None):
        self.type = type
        self.enabled = enabled
        self.parameters = parameters or {}


class GeneUtils:
    """遺伝子関連ユーティリティ"""

    @staticmethod
    def normalize_parameter(
        value: float, min_val: float = 1, max_val: float = 200
    ) -> float:
        """期間パラメータを正規化"""
        return max(0.0, min(1.0, (value - min_val) / (max_val - min_val)))

    @staticmethod
    def create_default_strategy_gene(strategy_gene_class):
        """デフォルトの戦略遺伝子を作成"""
        try:
            # 動的インポートを避けるため、引数として渡すか、呼び出し側でインポートする
            # ここでは基本的な構造のみを提供
            from ..models.strategy_models import Condition, IndicatorGene

            # デフォルト指標
            indicators = [
                IndicatorGene(type="SMA", parameters={"period": 20}, enabled=True)
            ]

            # デフォルト条件
            long_entry_conditions = [
                Condition(left_operand="close", operator=">", right_operand="open")
            ]
            short_entry_conditions = [
                Condition(left_operand="close", operator="<", right_operand="open")
            ]
            exit_conditions = [
                Condition(left_operand="close", operator="==", right_operand="open")
            ]
            entry_conditions = long_entry_conditions

            # デフォルトリスク管理
            risk_management = {
                "stop_loss": 0.03,
                "take_profit": 0.15,
                "position_size": 0.1,
            }

            # メタデータ
            metadata = {
                "generated_by": "create_default_strategy_gene",
                "source": "fallback",
                "indicators_count": len(indicators),
                "tpsl_gene_included": False,
                "position_sizing_gene_included": False,
            }

            return strategy_gene_class(
                indicators=indicators,
                entry_conditions=entry_conditions,
                long_entry_conditions=long_entry_conditions,
                short_entry_conditions=short_entry_conditions,
                exit_conditions=exit_conditions,
                risk_management=risk_management,
                tpsl_gene=None,
                position_sizing_gene=None,
                metadata=metadata,
            )
        except Exception as inner_e:
            logger.error(f"デフォルト戦略遺伝子作成エラー: {inner_e}")
            raise ValueError(f"デフォルト戦略遺伝子の作成に失敗: {inner_e}")


safe_execute = ErrorHandler.safe_execute
ensure_float = DataConverter.ensure_float
ensure_int = DataConverter.ensure_int
ensure_list = DataConverter.ensure_list
ensure_dict = DataConverter.ensure_dict
normalize_symbol = DataConverter.normalize_symbol
validate_range = ValidationUtils.validate_range
validate_required_fields = ValidationUtils.validate_required_fields
time_function = PerformanceUtils.time_function

# GeneUtilsの便利関数
normalize_parameter = GeneUtils.normalize_parameter
create_default_strategy_gene = GeneUtils.create_default_strategy_gene

# GeneticUtilsの便利関数
create_child_metadata = GeneticUtils.create_child_metadata
prepare_crossover_metadata = GeneticUtils.prepare_crossover_metadata
