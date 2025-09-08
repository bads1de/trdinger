"""
共通ユーティリティ関数

auto_strategy全体で使用される共通機能を提供します。
"""

import logging
import os
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
            if key.startswith("_"):
                continue  # プライベート属性は除外

            # Enumの処理
            if hasattr(value, "value"):
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
        if hasattr(cls, "__annotations__"):
            annotations = cls.__annotations__

            for param_name, param_type in annotations.items():
                if param_name in data:
                    value = data[param_name]

                    # Enum型への変換
                    if hasattr(param_type, "__members__"):
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

        return cls(**init_params)

    def validate(self) -> Tuple[bool, List[str]]:
        """遺伝子の妥当性を検証"""
        errors = []

        try:
            # 基本的な属性チェック
            if hasattr(self, "enabled") and not isinstance(
                getattr(self, "enabled", True), bool
            ):
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

    def _validate_range(
        self,
        value: Union[int, float],
        min_val: Union[int, float],
        max_val: Union[int, float],
        param_name: str,
        errors: List[str],
    ) -> bool:
        """範囲検証のヘルパー関数"""
        if not (min_val <= value <= max_val):
            errors.append(
                f"{param_name}は{min_val}-{max_val}の範囲である必要があります"
            )
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
    def normalize_symbol(symbol: Optional[str], provider: str = "generic") -> str:
        """シンボルを正規化（統一サービス経由）"""
        if not symbol:
            return "BTC:USDT"
        return symbol.strip().upper()


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


class GeneticUtils:
    """遺伝的アルゴリズム関連ユーティリティ"""

    @staticmethod
    def create_child_metadata(
        parent1_metadata: Dict[str, Any],
        parent2_metadata: Dict[str, Any],
        parent1_id: str,
        parent2_id: str,
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
            parent2_id=strategy_parent2.id,
        )

    @staticmethod
    def crossover_generic_genes(
        parent1_gene,
        parent2_gene,
        gene_class,
        numeric_fields: Optional[List[str]] = None,
        enum_fields: Optional[List[str]] = None,
        choice_fields: Optional[List[str]] = None,
        list_fields: Optional[List[str]] = None,
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
        all_fields.discard("_")  # プライベート属性を除外

        for field in all_fields:
            if field in choice_fields:
                # どちらかを選択
                child1_params[field] = random.choice(
                    [parent1_dict[field], parent2_dict[field]]
                )
                child2_params[field] = random.choice(
                    [parent1_dict[field], parent2_dict[field]]
                )
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
                child1_params[field] = random.choice(
                    [parent1_dict[field], parent2_dict[field]]
                )
                child2_params[field] = random.choice(
                    [parent1_dict[field], parent2_dict[field]]
                )
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
        numeric_ranges: Optional[Dict[str, Tuple[float, float]]] = None,
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
                if hasattr(current_enum, "__class__"):
                    enum_class = current_enum.__class__
                    # ランダムに別の値を選択
                    mutated_params[field] = random.choice(list(enum_class))

        return gene_class(**mutated_params)


class YamlIndicatorUtils:
    """YAMLベースの指標特性ユーティリティ"""

    @classmethod
    def generate_characteristics_from_yaml(cls, yaml_file_path: str) -> Dict[str, Dict]:
        """
        YAMLファイルから指標特性を動的に生成

        Args:
            yaml_file_path: YAML設定ファイルのパス

        Returns:
            各指標の特性定義を含む辞書
        """
        characteristics = {}

        try:
            with open(yaml_file_path, "r", encoding="utf-8") as file:
                config = yaml.safe_load(file)

            if "indicators" not in config:
                print(f"警告: {yaml_file_path}に'indicators'セクションが見つかりません")
                return characteristics

            config["indicators"]

            for indicator_name, indicator_config in config["indicators"].items():
                if not isinstance(indicator_config, dict):
                    continue

                # 基本構造
                char = {
                    "type": indicator_config.get("type", "unknown"),
                    "scale_type": indicator_config.get("scale_type", "price_absolute"),
                }

                # thresholdsの処理
                thresholds = indicator_config.get("thresholds", {})
                if thresholds:
                    if isinstance(thresholds, dict):
                        # riskレベルの処理
                        for risk_level, risk_config in thresholds.items():
                            if risk_level in ["aggressive", "normal", "conservative"]:
                                char[f"{risk_level}_config"] = risk_config
                            elif risk_level == "all":
                                char.update(cls._process_thresholds(risk_config))
                            else:
                                # その他の特殊なしきい値設定
                                char.update(
                                    cls._process_thresholds({risk_level: risk_config})
                                )

                    # 既存の互換性維持
                    char.update(
                        cls._extract_oscillator_settings(
                            char, indicator_config, thresholds
                        )
                    )

                # 特性辞書に追加
                characteristics[indicator_name] = char
        except FileNotFoundError:
            print(f"エラー: YAMLファイルが見つかりません: {yaml_file_path}")
        except yaml.YAMLError as e:
            print(f"エラー: YAMLファイルの解析に失敗しました: {e}")
        except Exception as e:
            print(f"エラー: 特性生成中に予期しないエラーが発生しました: {e}")

        return characteristics

    @classmethod
    def _process_thresholds(cls, thresholds: Dict[str, Any]) -> Dict[str, Any]:
        """しきい値の設定を処理"""
        processed = {}

        for key, value in thresholds.items():
            if key.endswith("_lt"):
                processed[f'{key.rstrip("_lt")}_oversold'] = value
            elif key.endswith("_gt"):
                processed[f'{key.rstrip("_gt")}_overbought'] = value
            elif key in ["long_gt", "short_lt"]:
                processed[key.replace("_", "_signal_")] = value
            else:
                processed[key] = value

        return processed

    @classmethod
    def _extract_oscillator_settings(
        cls, char: Dict, indicator_config: Dict, thresholds: Dict
    ) -> Dict[str, Any]:
        """オシレーター設定を抽出して互換性のための形式に変換"""
        settings = {}

        # スケールタイプに基づいてデフォルトの設定を適用
        scale_type = indicator_config.get("scale_type", "price_absolute")

        if scale_type == "oscillator_0_100":
            settings.update(
                {
                    "range": (0, 100),
                    "oversold_threshold": 30,
                    "overbought_threshold": 70,
                    "neutral_zone": (40, 60),
                }
            )
        elif scale_type == "oscillator_plus_minus_100":
            settings.update({"range": (-100, 100), "neutral_zone": (-20, 20)})
        elif scale_type == "momentum_zero_centered":
            settings.update({"range": None, "zero_cross": True})

        # 条件に基づいてゾーン設定を上書き
        conditions = indicator_config.get("conditions", {})
        if conditions:
            cls._apply_condition_based_settings(settings, conditions, thresholds)

        return settings

    @classmethod
    def _apply_condition_based_settings(
        cls, settings: Dict, conditions: Dict, thresholds: Dict
    ) -> Dict[str, Any]:
        """条件ベースの設定を適用"""
        if "long_lt" in str(conditions.get("long", "")) and "short_gt" in str(
            conditions.get("short", "")
        ):
            # オーバーソールド/オーバーバウト設定
            settings["oversold_based"] = True
            settings["overbought_based"] = True
        elif "long_gt" in str(conditions.get("long", "")) and "short_lt" in str(
            conditions.get("short", "")
        ):
            # ゼロクロス設定
            settings["zero_cross"] = True

        return settings

    @classmethod
    def _merge_characteristics(cls, existing: Dict, yaml_based: Dict) -> Dict:
        """既存の特性とYAMLベースの特性をマージ"""
        merged = existing.copy()

        for indicator_name, yaml_config in yaml_based.items():
            if indicator_name in merged:
                # 既存のエントリをYAMLの設定で更新
                merged[indicator_name].update(yaml_config)
            else:
                # 新しいエントリを追加
                merged[indicator_name] = yaml_config

        return merged

    @classmethod
    def initialize_yaml_based_characteristics(
        cls, existing_characteristics: Dict[str, Dict[str, Any]]
    ) -> Dict[str, Dict[str, Any]]:
        """
        YAML設定に基づいて特性を生成してマージ

        Args:
            existing_characteristics: 既存のINDICATOR_CHARACTERISTICS

        Returns:
            マージされた特性データ
        """
        CONFIG_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        YAML_CONFIG_PATH = os.path.join(
            CONFIG_DIR, "config", "technical_indicators_config.yaml"
        )

        if os.path.exists(YAML_CONFIG_PATH):
            YAML_BASED_CHARACTERISTICS = cls.generate_characteristics_from_yaml(
                YAML_CONFIG_PATH
            )
            # 既存の特性とマージして動的に更新
            return cls._merge_characteristics(
                existing_characteristics, YAML_BASED_CHARACTERISTICS
            )
        else:
            print(f"警告: YAML設定ファイルが見つかりません: {YAML_CONFIG_PATH}")
            return existing_characteristics

    @classmethod
    def load_yaml_config_for_indicators(cls) -> Dict[str, Any]:
        """技術指標のYAML設定を読み込み（ConditionGenerator用）"""
        config_path = (
            Path(__file__).parent.parent / "config" / "technical_indicators_config.yaml"
        )
        # YamlLoadUtilsを使用して読み込み
        config = YamlLoadUtils.load_yaml_config(config_path)
        return config

    @classmethod
    def get_indicator_config_from_yaml(
        cls, yaml_config: Dict[str, Any], indicator_name: str
    ) -> Optional[Dict[str, Any]]:
        """YAMLから指標設定を取得"""
        import logging

        logger = logging.getLogger(__name__)
        logger.debug(f"Looking for indicator config: {indicator_name}")
        indicators_config = yaml_config.get("indicators", {})
        logger.debug(f"Indicators config keys: {list(indicators_config.keys())}")
        config = indicators_config.get(indicator_name)
        if config is None:
            logger.warning(f"No config found for indicator: {indicator_name}")
        else:
            logger.debug(f"Found config for {indicator_name}: {config}")
        return config

    @classmethod
    def get_threshold_from_yaml(
        cls,
        yaml_config: Dict[str, Any],
        config: Dict[str, Any],
        side: str,
        context: Dict[str, Any],
    ) -> Any:
        """YAMLから閾値取得"""
        import logging

        logger = logging.getLogger(__name__)

        if not config:
            logger.debug("YAML config is None")
            return None

        thresholds = config.get("thresholds", {})
        if not thresholds:
            logger.debug("thresholds not found in config")
            return None

        profile = context.get("threshold_profile", "normal")
        logger.debug(f"Using profile: {profile}, side: {side}")

        if profile in thresholds and thresholds[profile]:
            profile_config = thresholds[profile]
            logger.debug(f"Found profile_config: {profile_config}")
            if side == "long" and (
                "long_gt" in profile_config or "long_lt" in profile_config
            ):
                threshold = profile_config.get("long_gt", profile_config.get("long_lt"))
                logger.debug(f"Long threshold: {threshold}")
                return threshold
            elif side == "short" and (
                "short_lt" in profile_config or "short_gt" in profile_config
            ):
                threshold = profile_config.get(
                    "short_lt", profile_config.get("short_gt")
                )
                logger.debug(f"Short threshold: {threshold}")
                return threshold

        if "all" in thresholds and thresholds["all"]:
            all_config = thresholds["all"]
            logger.debug(f"Using 'all' config: {all_config}")
            if side == "long":
                threshold = all_config.get("pos_threshold")
                logger.debug(f"Long threshold from all: {threshold}")
                return threshold
            else:
                threshold = all_config.get("neg_threshold")
                logger.debug(f"Short threshold from all: {threshold}")
                return threshold

        logger.debug("No threshold found")
        return None

    @classmethod
    def test_yaml_conditions_with_generator(
        cls, yaml_config: Dict[str, Any], test_indicators: Optional[List[str]] = None
    ) -> Dict[str, Any]:
        """YAMLベースの条件生成テスト（ConditionGenerator用）"""
        try:
            return {"error": "Generator class must be passed externally to avoid circular import"}
        except Exception as e:
            import logging

            logger = logging.getLogger(__name__)
            logger.error(f"YAMLテストエラー: {e}")
            return {"error": str(e)}


class YamlLoadUtils:
    """YAMLローディングユーティリティ"""

    @staticmethod
    def load_yaml_config(
        config_path: Union[str, Path], fallback: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
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

            with open(path, "r", encoding="utf-8") as f:
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
            logger.error(f"YAML構文エラー: {config_path}, {e}")
            return fallback
        except Exception as e:
            logger.error(f"YAML読み込みエラー: {config_path}, {e}")
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
                    errors.append(
                        f"indicator {indicator_name}: 辞書形式である必要があります"
                    )
                    continue

                # 必須フィールドチェック
                required_fields = ["type", "scale_type", "thresholds", "conditions"]
                for field in required_fields:
                    if field not in indicator_config:
                        errors.append(
                            f"indicator {indicator_name}: {field}フィールドが必須です"
                        )

                # conditionsの検証
                conditions = indicator_config.get("conditions", {})
                if isinstance(conditions, dict):
                    for side in ["long", "short"]:
                        if side not in conditions:
                            continue  # オプション
                        condition_template = conditions[side]
                        if not isinstance(condition_template, str):
                            errors.append(
                                f"indicator {indicator_name}: {side}条件は文字列テンプレートである必要があります"
                            )

        except Exception as e:
            errors.append(f"設定検証エラー: {e}")

        return len(errors) == 0, errors

    @staticmethod
    def get_indicator_config(
        yaml_config: Dict[str, Any], indicator_name: str
    ) -> Optional[Dict[str, Any]]:
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
        test_indicators: Optional[List[str]] = None,
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
            "summary": {},
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
                    config = YamlLoadUtils.get_indicator_config(
                        yaml_config, indicator_name
                    )
                    if not config:
                        result["errors"].append(f"YAML設定なし: {indicator_name}")
                        continue

                    # 条件生成テスト
                    long_conditions = generator._generate_yaml_based_conditions(
                        mock_indicator, "long"
                    )
                    short_conditions = generator._generate_yaml_based_conditions(
                        mock_indicator, "short"
                    )

                    result["tested_indicators"].append(
                        {
                            "name": indicator_name,
                            "long_conditions_count": len(long_conditions),
                            "short_conditions_count": len(short_conditions),
                            "type": config.get("type", "unknown"),
                        }
                    )

                    successful_tests += 1
                    total_tests += 1

                except Exception as e:
                    result["errors"].append(f"{indicator_name} テスト失敗: {e}")
                    total_tests += 1

            result["success"] = successful_tests == total_tests and total_tests > 0
            result["summary"] = {
                "total_tested": total_tests,
                "successful": successful_tests,
                "success_rate": (
                    successful_tests / total_tests if total_tests > 0 else 0.0
                ),
            }

        except Exception as e:
            result["errors"].append(f"全体テスト失敗: {e}")

        return result


class MockIndicatorGene:
    """テスト用のモックIndicatorGene"""

    def __init__(
        self,
        type: str,
        enabled: bool = True,
        parameters: Optional[Dict[str, Any]] = None,
    ):
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
