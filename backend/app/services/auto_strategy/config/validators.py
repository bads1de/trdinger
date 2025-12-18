"""
設定バリデーター

設定クラスの検証ロジックを分離・集約します。
"""

import logging
from typing import List, Tuple

from .base import BaseConfig
from .ga import GAConfig

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

        def check_range(val, min_v, max_v, name, is_int=True):
            try:
                if not isinstance(val, (int, float)):
                    errors.append(f"{name}は数値である必要があります")
                    return False
                if not (min_v <= val <= max_v):
                    if is_int:
                        if val > max_v:
                            errors.append(f"{name}は{max_v}以下である必要があります（パフォーマンス上の制約）")
                        else:
                            errors.append(f"{name}は正の整数である必要があります")
                    else:
                        errors.append(f"{name}は{min_v}-{max_v}の範囲である必要があります")
                    return False
                return True
            except (TypeError, ValueError):
                errors.append(f"{name}は数値である必要があります")
                return False

        # 進化設定
        check_range(config.population_size, 1, 1000, "個体数")
        check_range(config.generations, 1, 500, "世代数")
        check_range(config.crossover_rate, 0, 1, "交叉率", False)
        check_range(config.mutation_rate, 0, 1, "突然変異率", False)
        
        # 数値であることを確認してから比較
        if isinstance(config.elite_size, (int, float)) and isinstance(config.population_size, (int, float)):
            if config.elite_size < 0 or config.elite_size >= config.population_size:
                errors.append("エリート保存数は0以上、個体数未満である必要があります")
        else:
            errors.append("elite_size と population_size は数値である必要があります")

        # OOS設定
        if not isinstance(config.oos_split_ratio, (int, float)) or not 0.0 <= config.oos_split_ratio < 1.0:
            errors.append("OOS分割比率は0.0以上1.0未満である必要があります")

        # 評価設定
        if abs(sum(config.fitness_weights.values()) - 1.0) > 0.01:
            errors.append("フィットネス重みの合計は1.0である必要があります")

        required_metrics = {"total_return", "sharpe_ratio", "max_drawdown", "win_rate"}
        missing_metrics = required_metrics - set(config.fitness_weights.keys())
        if missing_metrics:
            errors.append(f"必要なメトリクスが不足しています: {missing_metrics}")

        if config.primary_metric not in config.fitness_weights:
            errors.append(f"プライマリメトリクス '{config.primary_metric}' がフィットネス重みに含まれていません")

        # 指標設定
        check_range(config.max_indicators, 1, 10, "最大指標数")

        # パラメータ範囲の検証
        for param, r in config.parameter_ranges.items():
            if not isinstance(r, list) or len(r) != 2:
                errors.append(f"パラメータ '{param}' の範囲は [min, max] の形式である必要があります")
            elif r[0] >= r[1]:
                errors.append(f"パラメータ '{param}' の最小値は最大値より小さい必要があります")

        # ログレベル
        if config.log_level not in {"DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"}:
            errors.append(f"無効なログレベル: {config.log_level}. 有効な値: {{'DEBUG', 'INFO', 'WARNING', 'ERROR', 'CRITICAL'}}")

        if config.parallel_processes is not None:
            if not isinstance(config.parallel_processes, (int, float)) or config.parallel_processes <= 0:
                errors.append("並列プロセス数は正の整数である必要があります")
            elif config.parallel_processes > 32:
                errors.append("並列プロセス数は32以下である必要があります")

        return errors
