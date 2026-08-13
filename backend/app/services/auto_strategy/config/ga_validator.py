"""
設定バリデーター

ConfigValidator クラスを提供します。
GAConfigを含む設定オブジェクトの妥当性を検証します。
"""

import logging
from collections.abc import Mapping
from typing import Any

from .evaluation_plan import EvaluationPlan
from .ga_config import GAConfig, RobustnessConfig
from .helpers import validate_robustness_regime_window

logger = logging.getLogger(__name__)


class ConfigValidator:
    """GA設定バリデーター

    GAConfigを含む設定オブジェクトの妥当性を検証します。
    数値パラメータの範囲チェック、設定間の整合性検証など、
    遺伝的アルゴリズム実行前の設定検証を包括的に実施します。
    """

    VALID_LOG_LEVELS = {"DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"}

    @staticmethod
    def validate(config: GAConfig) -> tuple[bool, list[str]]:
        """
        設定オブジェクトの妥当性を検証

        Args:
            config: 検証対象の設定インスタンス

        Returns:
            (妥当であればTrue, エラーメッセージのリスト) のタプル
        """
        errors = []

        validation_rules = getattr(config, "validation_rules", None)
        if isinstance(validation_rules, dict):
            errors.extend(
                ConfigValidator._validate_generic_rules(config, validation_rules)
            )

        # GAConfigの属性を持っている場合のみGA固有の検証を実行
        if hasattr(config, "population_size"):
            errors.extend(ConfigValidator._validate_ga_config(config))

        return len(errors) == 0, errors

    @staticmethod
    def _validate_generic_rules(config: GAConfig, rules: dict[str, Any]) -> list[str]:
        """validation_rules ベースの汎用検証を行う。"""
        errors: list[str] = []

        try:
            required_fields = rules.get("required_fields", [])
            for field_name in required_fields:
                try:
                    value = getattr(config, field_name)
                except Exception as exc:  # pragma: no cover - defensive
                    errors.append(f"検証処理エラー: {exc}")
                    continue
                if value is None:
                    errors.append(f"必須フィールド '{field_name}' が不足しています")

            for field_name, value_range in rules.get("ranges", {}).items():
                try:
                    value = getattr(config, field_name)
                except Exception as exc:  # pragma: no cover - defensive
                    errors.append(f"検証処理エラー: {exc}")
                    continue

                if not isinstance(value_range, (list, tuple)) or len(value_range) != 2:
                    errors.append(
                        f"フィールド '{field_name}' の範囲設定は [min, max] の形式である必要があります"
                    )
                    continue

                min_v, max_v = value_range
                if not isinstance(value, (int, float)):
                    errors.append(f"'{field_name}' は数値である必要があります")
                    continue
                if not (min_v <= value <= max_v):
                    errors.append(
                        f"'{field_name}' は {min_v} から {max_v} の範囲である必要があります"
                    )

            for field_name, expected_type in rules.get("types", {}).items():
                try:
                    value = getattr(config, field_name)
                except Exception as exc:  # pragma: no cover - defensive
                    errors.append(f"検証処理エラー: {exc}")
                    continue

                if not isinstance(value, expected_type):
                    type_name = getattr(expected_type, "__name__", str(expected_type))
                    errors.append(
                        f"'{field_name}' は {type_name} 型である必要があります"
                    )
        except Exception as exc:  # pragma: no cover - defensive
            errors.append(f"検証処理エラー: {exc}")

        return errors

    @staticmethod
    def _validate_ga_config(config: GAConfig) -> list[str]:
        """
        GAConfig固有の検証を実行

        Args:
            config: 検証対象のGAConfigインスタンス

        Returns:
            エラーメッセージのリスト
        """
        errors = []
        errors.extend(ConfigValidator._validate_ga_evolution_settings(config))
        errors.extend(ConfigValidator._validate_ga_fitness_settings(config))
        errors.extend(ConfigValidator._validate_ga_parameter_settings(config))
        errors.extend(ConfigValidator._validate_ga_multi_fidelity_settings(config))
        errors.extend(ConfigValidator._validate_ga_early_termination_settings(config))
        errors.extend(ConfigValidator._validate_ga_two_stage_settings(config))
        errors.extend(ConfigValidator._validate_ga_robustness_settings(config))
        errors.extend(ConfigValidator._validate_ga_validation_settings(config))
        errors.extend(ConfigValidator._validate_evaluation_plan(config))
        errors.extend(
            ConfigValidator._validate_ga_iterative_improvement_settings(config)
        )
        return errors

    @staticmethod
    def _validate_numeric_range(
        val: Any, min_v: float, max_v: float, name: str, is_int: bool = True
    ) -> list[str]:
        """
        汎用的な数値範囲検証

        Args:
            val: 検証対象の値
            min_v: 最小値
            max_v: 最大値
            name: パラメータ名（エラー表示用）
            is_int: 整数として検証するかどうか

        Returns:
            エラーメッセージのリスト
        """
        try:
            if not isinstance(val, (int, float)):
                return [f"{name}は数値である必要があります"]
            if not (min_v <= val <= max_v):
                if is_int:
                    if val > max_v:
                        return [
                            f"{name}は{max_v}以下である必要があります（パフォーマンス上の制約）"
                        ]
                    return [f"{name}は正の整数である必要があります"]
                return [f"{name}は{min_v}-{max_v}の範囲である必要があります"]
            return []
        except (TypeError, ValueError):
            return [f"{name}は数値である必要があります"]

    @staticmethod
    def _validate_ga_evolution_settings(config: GAConfig) -> list[str]:
        """
        GA進化パラメータ（個体数、世代数、交叉率、突然変異率等）の検証

        Args:
            config: 検証対象のGAConfigインスタンス

        Returns:
            エラーメッセージのリスト
        """
        errors = []
        errors.extend(
            ConfigValidator._validate_numeric_range(
                config.population_size, 1, 1000, "個体数"
            )
        )
        errors.extend(
            ConfigValidator._validate_numeric_range(
                config.generations, 1, 500, "世代数"
            )
        )
        errors.extend(
            ConfigValidator._validate_numeric_range(
                config.crossover_rate, 0, 1, "交叉率", False
            )
        )
        errors.extend(
            ConfigValidator._validate_numeric_range(
                config.mutation_rate, 0, 1, "突然変異率", False
            )
        )

        elite_size: Any = config.elite_size
        population_size: Any = config.population_size
        if isinstance(elite_size, (int, float)) and isinstance(
            population_size, (int, float)
        ):
            if elite_size < 0 or elite_size >= population_size:
                errors.append("エリート保存数は0以上、個体数未満である必要があります")
        else:
            errors.append("elite_size と population_size は数値である必要があります")

        return errors

    @staticmethod
    def _validate_ga_fitness_settings(config: GAConfig) -> list[str]:
        """
        フィットネス計算設定（重み、メトリクス、多目的最適化等）の検証

        Args:
            config: 検証対象のGAConfigインスタンス

        Returns:
            エラーメッセージのリスト
        """
        errors = []
        fitness_weights = getattr(config, "fitness_weights", {})
        if not isinstance(fitness_weights, dict):
            errors.append("fitness_weights は辞書である必要があります")
            return errors

        weights_are_numeric = not any(
            not isinstance(weight, (int, float)) for weight in fitness_weights.values()
        )
        if not weights_are_numeric:
            errors.append("フィットネス重みは数値である必要があります")
        else:
            if abs(sum(fitness_weights.values()) - 1.0) > 0.01:
                errors.append("フィットネス重みの合計は1.0である必要があります")

        required_metrics = {
            "total_return",
            "sharpe_ratio",
            "max_drawdown",
            "win_rate",
        }
        missing_metrics = required_metrics - set(fitness_weights.keys())
        if missing_metrics:
            errors.append(f"必要なメトリクスが不足しています: {missing_metrics}")

        if "prediction_score" in fitness_weights:
            errors.append(
                "prediction_score はボラ回帰化に伴い fitness_weights ではサポートされません"
            )

        objectives = getattr(config, "objectives", None)
        if objectives is None:
            objectives = []
        elif not isinstance(objectives, (list, tuple)):
            errors.append("objectives はリストである必要があります")
            objectives = []

        objective_weights = getattr(config, "objective_weights", None)
        if objective_weights is None:
            objective_weights = []
        elif not isinstance(objective_weights, (list, tuple)):
            errors.append("objective_weights はリストである必要があります")
            objective_weights = []

        if len(objectives) != len(objective_weights):
            errors.append(
                "objective_weights の数は objectives と一致する必要があります "
                f"(objectives={len(objectives)}, "
                f"objective_weights={len(objective_weights)})"
            )
        elif objective_weights and not all(
            isinstance(weight, (int, float)) for weight in objective_weights
        ):
            errors.append("objective_weights は数値である必要があります")

        if "prediction_score" in objectives:
            errors.append(
                "prediction_score はボラ回帰化に伴い objectives ではサポートされません"
            )

        return errors

    @staticmethod
    def _validate_parameter_ranges(
        parameter_ranges: dict[str, list[float]],
    ) -> list[str]:
        """
        パラメータ探索範囲設定の検証

        Args:
            parameter_ranges: パラメータ名と [min, max] リストの辞書

        Returns:
            エラーメッセージのリスト
        """
        errors = []
        raw_ranges: Any = parameter_ranges
        if not isinstance(raw_ranges, dict):
            errors.append("パラメータ範囲は辞書である必要があります")
            return errors

        for param, value_range in parameter_ranges.items():
            if not isinstance(value_range, list) or len(value_range) != 2:
                errors.append(
                    f"パラメータ '{param}' の範囲は [min, max] の形式である必要があります"
                )
            else:
                try:
                    if value_range[0] >= value_range[1]:
                        errors.append(
                            f"パラメータ '{param}' の最小値は最大値より小さい必要があります"
                        )
                except TypeError:
                    errors.append(
                        f"パラメータ '{param}' の最小値は最大値より小さい必要があります"
                    )
        return errors

    @staticmethod
    def _validate_ga_parameter_settings(config: GAConfig) -> list[str]:
        """
        GA戦略パラメータ（指標数、探索範囲、ログレベル等）の検証

        Args:
            config: 検証対象のGAConfigインスタンス

        Returns:
            エラーメッセージのリスト
        """
        errors = []
        errors.extend(
            ConfigValidator._validate_numeric_range(
                config.max_indicators, 1, 10, "最大指標数"
            )
        )
        errors.extend(
            ConfigValidator._validate_parameter_ranges(config.parameter_ranges)
        )

        if (
            not isinstance(config.log_level, str)
            or config.log_level not in ConfigValidator.VALID_LOG_LEVELS
        ):
            errors.append(
                f"無効なログレベル: {config.log_level}. "
                "有効な値: {'DEBUG', 'INFO', 'WARNING', "
                "'ERROR', 'CRITICAL'}"
            )

        return errors

    @staticmethod
    def _validate_ga_multi_fidelity_settings(config: GAConfig) -> list[str]:
        """multi-fidelity 評価設定の検証"""
        errors: list[str] = []
        evaluation_config = config.evaluation_config
        if not getattr(evaluation_config, "enable_multi_fidelity_evaluation", False):
            return errors

        if (
            not isinstance(evaluation_config.multi_fidelity_window_ratio, (int, float))
            or not 0.0 < float(evaluation_config.multi_fidelity_window_ratio) <= 1.0
        ):
            errors.append(
                "evaluation_config.multi_fidelity_window_ratio "
                "は0より大きく1.0以下である必要があります"
            )

        if (
            not isinstance(evaluation_config.multi_fidelity_oos_ratio, (int, float))
            or not 0.0 < float(evaluation_config.multi_fidelity_oos_ratio) < 1.0
        ):
            errors.append(
                "evaluation_config.multi_fidelity_oos_ratio "
                "は0より大きく1.0未満である必要があります"
            )

        if (
            not isinstance(
                evaluation_config.multi_fidelity_candidate_ratio, (int, float)
            )
            or not 0.0 < float(evaluation_config.multi_fidelity_candidate_ratio) <= 1.0
        ):
            errors.append(
                "evaluation_config.multi_fidelity_candidate_ratio "
                "は0より大きく1.0以下である必要があります"
            )

        if (
            not isinstance(
                evaluation_config.multi_fidelity_min_candidates, (int, float)
            )
            or int(evaluation_config.multi_fidelity_min_candidates) <= 0
        ):
            errors.append(
                "evaluation_config.multi_fidelity_min_candidates "
                "は正の整数である必要があります"
            )

        return errors

    @staticmethod
    def _validate_ga_early_termination_settings(config: GAConfig) -> list[str]:
        """早期打ち切り設定の検証"""
        errors: list[str] = []
        settings = config.evaluation_config.early_termination_settings
        if not settings.enabled:
            return errors

        max_drawdown = settings.max_drawdown
        if max_drawdown is not None and (
            not isinstance(max_drawdown, (int, float))
            or not 0.0 < float(max_drawdown) <= 1.0
        ):
            errors.append(
                "evaluation_config.early_termination_settings."
                "max_drawdown は0より大きく1.0以下である必要があります"
            )

        min_trades = settings.min_trades
        if min_trades is not None and (
            not isinstance(min_trades, (int, float)) or int(min_trades) <= 0
        ):
            errors.append(
                "evaluation_config.early_termination_settings."
                "min_trades は正の整数である必要があります"
            )

        expectancy: Any = settings.min_expectancy
        if expectancy is not None and not isinstance(expectancy, (int, float)):
            errors.append(
                "evaluation_config.early_termination_settings."
                "min_expectancy は数値である必要があります"
            )

        expectancy_min_trades = settings.expectancy_min_trades
        if (
            not isinstance(expectancy_min_trades, (int, float))
            or int(expectancy_min_trades) <= 0
        ):
            errors.append(
                "evaluation_config.early_termination_settings."
                "expectancy_min_trades は正の整数である必要があります"
            )

        for field_name in (
            "min_trade_check_progress",
            "trade_pace_tolerance",
            "expectancy_progress",
        ):
            value = getattr(settings, field_name, None)
            if not isinstance(value, (int, float)) or not 0.0 < float(value) <= 1.0:
                errors.append(
                    "evaluation_config.early_termination_settings."
                    f"{field_name} は0より大きく1.0以下である必要があります"
                )

        return errors

    @staticmethod
    def _validate_ga_two_stage_settings(config: GAConfig) -> list[str]:
        """二段階選抜設定の検証

        二段階選抜（Two-Stage Selection）の設定が正しいか検証します。
        エリート数、候補プールサイズ、スクリーニングしきい値などの
        整合性をチェックします。

        Args:
            config: 検証対象のGAConfigインスタンス

        Returns:
            List[str]: エラーメッセージのリスト。問題がなければ空リスト。
        """
        errors: list[str] = []
        two_stage_config = config.two_stage_selection_config
        if not two_stage_config.enabled:
            return errors

        elite_count = two_stage_config.elite_count
        population_size = config.population_size
        candidate_pool_size = two_stage_config.candidate_pool_size

        if not isinstance(elite_count, (int, float)) or int(elite_count) <= 0:
            errors.append("二段階選抜エリート数は正の整数である必要があります")
        elif isinstance(population_size, (int, float)) and int(elite_count) >= int(
            population_size
        ):
            errors.append("二段階選抜エリート数は個体数未満である必要があります")

        if (
            not isinstance(candidate_pool_size, (int, float))
            or int(candidate_pool_size) <= 0
        ):
            errors.append("二段階選抜候補数は正の整数である必要があります")
        elif isinstance(elite_count, (int, float)) and int(candidate_pool_size) < int(
            elite_count
        ):
            errors.append(
                "二段階選抜候補数は二段階選抜エリート数以上である必要があります"
            )

        # 新規: candidate_pool_sizeがpopulation_sizeを超えないことを検証
        if (
            isinstance(candidate_pool_size, (int, float))
            and isinstance(population_size, (int, float))
            and int(candidate_pool_size) > int(population_size)
        ):
            errors.append("二段階選抜候補数は個体数以下である必要があります")

        if (
            not isinstance(two_stage_config.min_pass_rate, (int, float))
            or not 0.0 <= float(two_stage_config.min_pass_rate) <= 1.0
        ):
            errors.append("二段階選抜 pass rate は0.0-1.0の範囲である必要があります")

        return errors

    @staticmethod
    def _validate_robustness_config_validation_symbols(
        config: GAConfig,
    ) -> list[str]:
        """ロバストネス検証用通貨ペア設定の検証

        ロバストネス検証（Robustness Validation）で使用する通貨ペアの
        設定が正しいか検証します。validation_symbolsがリスト形式であるか、
        適切な通貨ペア名が含まれているかを確認します。

        Args:
            config: 検証対象のGAConfigインスタンス

        Returns:
            List[str]: エラーメッセージのリスト。問題がなければ空リスト。
        """
        errors: list[str] = []
        validation_symbols: Any = config.robustness_config.validation_symbols
        if validation_symbols is not None and not isinstance(validation_symbols, list):
            errors.append(
                "robustness_config.validation_symbols はリストである必要があります"
            )
        return errors

    @staticmethod
    def _validate_robustness_window(window: Mapping[str, Any]) -> list[str]:
        """
        robustness 検証用期間設定の検証

        Args:
            window: 検証期間設定を含む辞書

        Returns:
            エラーメッセージのリスト
        """
        return validate_robustness_regime_window(window)

    @staticmethod
    def _validate_robustness_regime_windows(config: GAConfig) -> list[str]:
        """
        robustness 検証用全期間リストの検証

        Args:
            config: 検証対象のGAConfigインスタンス

        Returns:
            エラーメッセージのリスト
        """
        errors = []
        regime_windows: Any = config.robustness_config.regime_windows
        if not isinstance(regime_windows, list):
            errors.append(
                "robustness_config.regime_windows はリストである必要があります"
            )
            return errors

        for window in regime_windows:
            window_errors = ConfigValidator._validate_robustness_window(window)
            if window_errors:
                errors.extend(window_errors)
                break

        return errors

    @staticmethod
    def _validate_non_negative_numeric_list(
        values: list[float], label: str
    ) -> list[str]:
        """
        非負数値リストの検証

        Args:
            values: 検証対象のリスト
            label: エラー表示用ラベル

        Returns:
            エラーメッセージのリスト
        """
        errors = []
        raw_values: Any = values
        if not isinstance(raw_values, list):
            errors.append(f"{label} はリストである必要があります")
            return errors

        for value in values:
            if not isinstance(value, (int, float)) or float(value) < 0.0:
                errors.append(f"{label} は0以上の数値である必要があります")
                break

        return errors

    @staticmethod
    def _validate_positive_numeric_list(values: list[float], label: str) -> list[str]:
        """
        正の数値リストの検証

        Args:
            values: 検証対象のリスト
            label: エラー表示用ラベル

        Returns:
            エラーメッセージのリスト
        """
        errors = []
        raw_values: Any = values
        if not isinstance(raw_values, list):
            errors.append(f"{label} はリストである必要があります")
            return errors

        for value in values:
            if not isinstance(value, (int, float)) or float(value) <= 0.0:
                errors.append(f"{label} は正の数値である必要があります")
                break

        return errors

    @staticmethod
    def _validate_aggregate_method(config: GAConfig) -> list[str]:
        """
        robustness 評価集計方法の検証

        Args:
            config: 検証対象のGAConfigインスタンス

        Returns:
            エラーメッセージのリスト
        """
        aggregate_method = config.robustness_config.aggregate_method
        if not isinstance(aggregate_method, str) or aggregate_method not in {
            "robust",
            "mean",
        }:
            return [
                "robustness_config.aggregate_method "
                "は {'robust', 'mean'} "
                "のいずれかである必要があります"
            ]
        return []

    @staticmethod
    def _validate_ga_robustness_settings(config: GAConfig) -> list[str]:
        """
        robustness 検証全設定の検証

        Args:
            config: 検証対象のGAConfigインスタンス

        Returns:
            エラーメッセージのリスト
        """
        errors = []
        robustness_config = config.robustness_config
        if isinstance(robustness_config, RobustnessConfig):
            if (
                not isinstance(robustness_config.min_pass_rate, (int, float))
                or not 0.0 <= float(robustness_config.min_pass_rate) <= 1.0
            ):
                errors.append(
                    "robustness_config.min_pass_rate は0.0-1.0の範囲である必要があります"
                )
        errors.extend(
            ConfigValidator._validate_robustness_config_validation_symbols(config)
        )
        errors.extend(ConfigValidator._validate_robustness_regime_windows(config))
        slippage = config.robustness_config.stress_slippage
        errors.extend(
            ConfigValidator._validate_non_negative_numeric_list(
                slippage,
                "robustness の slippage",
            )
        )
        commission_multipliers = config.robustness_config.stress_commission_multipliers
        errors.extend(
            ConfigValidator._validate_positive_numeric_list(
                commission_multipliers,
                "robustness の commission multiplier",
            )
        )
        errors.extend(ConfigValidator._validate_aggregate_method(config))
        return errors

    @staticmethod
    def _validate_ga_validation_settings(config: GAConfig) -> list[str]:
        """自動検証パイプライン設定の検証。"""
        errors: list[str] = []
        validation_config = getattr(config, "validation_config", None)
        if validation_config is None:
            return errors
        if not validation_config.enabled:
            return errors

        min_pass_rate: Any = validation_config.min_pass_rate
        if (
            not isinstance(min_pass_rate, (int, float))
            or not 0.0 <= float(min_pass_rate) <= 1.0
        ):
            errors.append(
                "validation_config.min_pass_rate は0.0-1.0の範囲である必要があります"
            )

        wfa_n_folds: Any = validation_config.wfa_n_folds
        if not isinstance(wfa_n_folds, (int, float)) or int(wfa_n_folds) <= 0:
            errors.append(
                "validation_config.wfa_n_folds は正の整数である必要があります"
            )

        wfa_train_ratio: Any = validation_config.wfa_train_ratio
        if (
            not isinstance(wfa_train_ratio, (int, float))
            or not 0.0 < float(wfa_train_ratio) < 1.0
        ):
            errors.append(
                "validation_config.wfa_train_ratio は0より大きく1.0未満である必要があります"
            )

        min_trades: Any = validation_config.min_trades
        if min_trades is not None and (
            not isinstance(min_trades, (int, float)) or int(min_trades) <= 0
        ):
            errors.append("validation_config.min_trades は正の整数である必要があります")

        max_drawdown: Any = validation_config.max_drawdown
        if max_drawdown is not None and (
            not isinstance(max_drawdown, (int, float))
            or not 0.0 < float(max_drawdown) <= 1.0
        ):
            errors.append(
                "validation_config.max_drawdown は0より大きく1.0以下である必要があります"
            )

        min_primary_fitness: Any = validation_config.min_primary_fitness
        if min_primary_fitness is not None and not isinstance(
            min_primary_fitness, (int, float)
        ):
            errors.append(
                "validation_config.min_primary_fitness は数値である必要があります"
            )

        max_candidates: Any = validation_config.max_candidates
        if not isinstance(max_candidates, (int, float)) or int(max_candidates) <= 0:
            errors.append(
                "validation_config.max_candidates は正の整数である必要があります"
            )

        return errors

    @staticmethod
    def _validate_evaluation_plan(config: GAConfig) -> list[str]:
        """評価計画の robustness 設定の安全性を検証する。"""
        plan = getattr(config, "evaluation_plan", None)
        if not isinstance(plan, EvaluationPlan):
            return []

        errors: list[str] = []
        if getattr(plan.robustness, "policy", None) != "gate_only":
            errors.append(
                "evaluation_plan.robustness.policy は'gate_only'である必要があります"
            )
        if getattr(plan.robustness, "failure_policy", None) != "fail_closed":
            errors.append(
                "evaluation_plan.robustness.failure_policy は'fail_closed'である必要があります"
            )
        return errors

    @staticmethod
    def _validate_ga_iterative_improvement_settings(config: GAConfig) -> list[str]:
        """反復改善ループ設定の検証。"""
        errors: list[str] = []
        iterative_config = getattr(config, "iterative_improvement_config", None)
        if iterative_config is None:
            return errors
        if not iterative_config.enabled:
            return errors

        max_seed_strategies: Any = iterative_config.max_seed_strategies
        if (
            not isinstance(max_seed_strategies, (int, float))
            or int(max_seed_strategies) <= 0
        ):
            errors.append(
                "iterative_improvement_config.max_seed_strategies "
                "は正の整数である必要があります"
            )

        min_fitness: Any = iterative_config.min_fitness
        if min_fitness is not None and not isinstance(min_fitness, (int, float)):
            errors.append(
                "iterative_improvement_config.min_fitness は数値である必要があります"
            )

        return errors
