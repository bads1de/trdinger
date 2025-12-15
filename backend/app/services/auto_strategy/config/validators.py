"""
設定バリデーター

設定クラスの検証ロジックを分離・集約します。
"""

import logging
from typing import Dict, List, Tuple

from .auto_strategy import AutoStrategyConfig
from .base import BaseConfig
from .ga_runtime import GAConfig

logger = logging.getLogger(__name__)


class ConfigValidator:
    """設定バリデーター"""

    @staticmethod
    def validate(config: BaseConfig) -> Tuple[bool, List[str]]:
        """設定オブジェクトを検証"""
        errors = ConfigValidator._validate_base(config)

        # クラスごとの追加検証
        if isinstance(config, GAConfig):
            errors.extend(ConfigValidator._validate_ga_config(config))
        elif isinstance(config, AutoStrategyConfig):
            errors.extend(ConfigValidator._validate_auto_strategy_config(config))

        # 将来的に TradingSettings などの検証が必要になったらここに追加

        return len(errors) == 0, errors

    @staticmethod
    def _validate_base(config: BaseConfig) -> List[str]:
        """共通検証ロジック"""
        errors = []
        try:
            # 必須フィールドチェック
            required_fields = config.validation_rules.get("required_fields", [])
            for field_name in required_fields:
                if not hasattr(config, field_name) or not getattr(config, field_name):
                    errors.append(f"必須フィールド '{field_name}' が設定されていません")

            # 範囲チェック
            range_rules = config.validation_rules.get("ranges", {})
            for field_name, (min_val, max_val) in range_rules.items():
                if hasattr(config, field_name):
                    value = getattr(config, field_name)
                    if isinstance(value, (int, float)) and not (
                        min_val <= value <= max_val
                    ):
                        errors.append(
                            f"'{field_name}' は {min_val} から {max_val} の範囲で設定してください"
                        )

            # 型チェック
            type_rules = config.validation_rules.get("types", {})
            for field_name, expected_type in type_rules.items():
                if hasattr(config, field_name):
                    value = getattr(config, field_name)
                    if value is not None and not isinstance(value, expected_type):
                        errors.append(
                            f"'{field_name}' は {expected_type.__name__} 型である必要があります"
                        )

        except Exception as e:
            logger.error(f"基本検証中にエラーが発生: {e}", exc_info=True)
            errors.append(f"検証処理エラー: {e}")

        return errors

    @staticmethod
    def _validate_ga_config(config: GAConfig) -> List[str]:
        """GAConfig固有の検証"""
        errors = []

        # 進化設定の検証
        try:
            if config.population_size <= 0:
                errors.append("個体数は正の整数である必要があります")
            elif config.population_size > 1000:
                errors.append(
                    "個体数は1000以下である必要があります（パフォーマンス上の制約）"
                )
        except TypeError:
            errors.append("個体数は数値である必要があります")

        try:
            if config.generations <= 0:
                errors.append("世代数は正の整数である必要があります")
            elif config.generations > 500:
                errors.append(
                    "世代数は500以下である必要があります（パフォーマンス上の制約）"
                )
        except TypeError:
            errors.append("世代数は数値である必要があります")

        try:
            if not 0 <= config.crossover_rate <= 1:
                errors.append("交叉率は0-1の範囲である必要があります")
        except (TypeError, ValueError):
            errors.append("交叉率は数値である必要があります")

        try:
            if not 0 <= config.mutation_rate <= 1:
                errors.append("突然変異率は0-1の範囲である必要があります")
        except (TypeError, ValueError):
            errors.append("突然変異率は数値である必要があります")

        try:
            if config.elite_size < 0 or config.elite_size >= config.population_size:
                errors.append("エリート保存数は0以上、個体数未満である必要があります")
        except (TypeError, ValueError):
            errors.append("elite_size と population_size は数値である必要があります")

        # OOS設定の検証
        if not 0.0 <= config.oos_split_ratio < 1.0:
            errors.append("OOS分割比率は0.0以上1.0未満である必要があります")

        # 評価設定の検証
        if abs(sum(config.fitness_weights.values()) - 1.0) > 0.01:
            errors.append("フィットネス重みの合計は1.0である必要があります")

        required_metrics = {"total_return", "sharpe_ratio", "max_drawdown", "win_rate"}
        missing_metrics = required_metrics - set(config.fitness_weights.keys())
        if missing_metrics:
            errors.append(f"必要なメトリクスが不足しています: {missing_metrics}")

        if config.primary_metric not in config.fitness_weights:
            errors.append(
                f"プライマリメトリクス '{config.primary_metric}' がフィットネス重みに含まれていません"
            )

        # 指標設定の検証
        try:
            if config.max_indicators <= 0:
                errors.append("最大指標数は正の整数である必要があります")
            elif config.max_indicators > 10:
                errors.append(
                    "最大指標数は10以下である必要があります（パフォーマンス上の制約）"
                )
        except TypeError:
            errors.append("最大指標数は数値である必要があります")

        # パラメータ設定の検証
        for param_name, range_values in config.parameter_ranges.items():
            if not isinstance(range_values, list) or len(range_values) != 2:
                errors.append(
                    f"パラメータ '{param_name}' の範囲は [min, max] の形式である必要があります"
                )
            elif range_values[0] >= range_values[1]:
                errors.append(
                    f"パラメータ '{param_name}' の最小値は最大値より小さい必要があります"
                )

        # 実行設定の検証
        valid_log_levels = {"DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"}
        if config.log_level not in valid_log_levels:
            errors.append(
                f"無効なログレベル: {config.log_level}. 有効な値: {valid_log_levels}"
            )

        if config.parallel_processes is not None:
            if config.parallel_processes <= 0:
                errors.append("並列プロセス数は正の整数である必要があります")
            elif config.parallel_processes > 32:
                errors.append("並列プロセス数は32以下である必要があります")

        return errors

    @staticmethod
    def _validate_auto_strategy_config(config: AutoStrategyConfig) -> List[str]:
        """AutoStrategyConfig固有の検証"""
        errors = []

        # cache_ttl_hoursの検証
        if (
            isinstance(config.cache_ttl_hours, (int, float))
            and config.cache_ttl_hours < 0
        ):
            errors.append("キャッシュTTLは正の数である必要があります")

        # log_levelの検証
        valid_log_levels = {"DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"}
        if config.log_level not in valid_log_levels:
            errors.append(f"無効なログレベル: {config.log_level}")

        return errors

    @staticmethod
    def validate_all(config: AutoStrategyConfig) -> Tuple[bool, Dict[str, List[str]]]:
        """AutoStrategyConfigの全階層を検証"""
        all_errors = {}
        is_valid = True

        # 各設定グループの検証
        settings_groups = {
            "trading": config.trading,
            "indicators": config.indicators,
            "ga": config.ga,
            "tpsl": config.tpsl,
            "position_sizing": config.position_sizing,
        }

        for group_name, group_config in settings_groups.items():
            valid, errors = ConfigValidator.validate(group_config)
            if not valid:
                all_errors[group_name] = errors
                is_valid = False

        # メイン設定の検証
        main_valid, main_errors = ConfigValidator.validate(config)
        if not main_valid:
            all_errors["main"] = main_errors
            is_valid = False

        return is_valid, all_errors





