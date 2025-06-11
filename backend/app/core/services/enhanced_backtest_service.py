"""
拡張バックテストサービス

backtesting.py内蔵最適化機能を活用した高度なバックテスト最適化機能を提供します。
scipyなどの外部ライブラリは不要で、backtesting.py単体で高度な最適化が可能です。
"""

import pandas as pd
import numpy as np
from datetime import datetime
from typing import Dict, Any, List, Tuple, Callable, Union
from backtesting import Backtest
from backtesting.lib import plot_heatmaps

from .backtest_service import BacktestService
from .backtest_data_service import BacktestDataService
from database.repositories.ohlcv_repository import OHLCVRepository
from database.connection import SessionLocal


class EnhancedBacktestService(BacktestService):
    """
    拡張されたバックテストサービス

    backtesting.py内蔵最適化機能を活用した高度な最適化機能とヒートマップ可視化を提供
    scipyなどの外部ライブラリは不要
    """

    def __init__(self, data_service: BacktestDataService = None):
        """
        初期化

        Args:
            data_service: データ変換サービス（テスト時にモックを注入可能）
        """
        super().__init__(data_service)
        self.constraint_functions = self._create_constraint_functions()

    def optimize_strategy_enhanced(
        self, config: Dict[str, Any], optimization_params: Dict[str, Any]
    ) -> Dict[str, Any]:
        """
        拡張された戦略最適化

        backtesting.py内蔵最適化機能を使用した戦略最適化

        Args:
            config: 基本バックテスト設定
            optimization_params: 最適化パラメータ
                - method: 'grid' | 'sambo' (推奨: sambo)
                - max_tries: int | float | None
                - maximize: str | callable
                - constraint: callable | None | str
                - return_heatmap: bool
                - return_optimization: bool
                - random_state: int | None
                - parameters: Dict[str, range]
                - save_heatmap: bool
                - heatmap_filename: str

        Returns:
            最適化結果の辞書
        """
        try:
            # 設定の検証
            self._validate_optimization_config(config, optimization_params)

            # データサービスの初期化（必要に応じて）
            if self.data_service is None:
                db = SessionLocal()
                try:
                    ohlcv_repo = OHLCVRepository(db)
                    self.data_service = BacktestDataService(ohlcv_repo)
                finally:
                    db.close()

            # データ取得
            data = self._get_backtest_data(config)
            strategy_class = self._create_strategy_class(config["strategy_config"])

            # バックテスト設定
            bt = Backtest(
                data,
                strategy_class,
                cash=config["initial_capital"],
                commission=config["commission_rate"],
                exclusive_orders=True,
                trade_on_close=True,
            )

            # 最適化パラメータの構築
            optimize_kwargs = self._build_optimize_kwargs(optimization_params)

            # 最適化実行（マルチプロセシング有効）
            print(f"最適化開始: {optimization_params.get('method', 'grid')} method")
            print("マルチプロセシング: 有効")

            result = bt.optimize(**optimize_kwargs)

            # 結果の処理
            processed_result = self._process_optimization_results(
                result, config, optimization_params
            )

            # ヒートマップの保存
            if (
                optimization_params.get("save_heatmap", False)
                and "heatmap_data" in processed_result
            ):
                self._save_heatmap(
                    processed_result["heatmap_data"],
                    optimization_params.get(
                        "heatmap_filename", "optimization_heatmap.html"
                    ),
                )

            print("最適化完了")
            return processed_result

        except Exception as e:
            print(f"最適化エラー: {str(e)}")
            raise

    def _validate_optimization_config(
        self, config: Dict[str, Any], optimization_params: Dict[str, Any]
    ) -> None:
        """最適化設定の検証"""
        # 基本設定の検証
        self._validate_config(config)

        # 最適化パラメータの検証
        required_fields = ["parameters"]
        for field in required_fields:
            if field not in optimization_params:
                raise ValueError(f"Missing required optimization field: {field}")

        # パラメータ範囲の検証
        parameters = optimization_params["parameters"]
        if not parameters:
            raise ValueError("At least one parameter range must be specified")

        for param_name, param_range in parameters.items():
            if not hasattr(param_range, "__iter__"):
                raise ValueError(f"Parameter {param_name} must be iterable")

    def _build_optimize_kwargs(
        self, optimization_params: Dict[str, Any]
    ) -> Dict[str, Any]:
        """最適化パラメータの構築"""
        optimize_kwargs = {
            "method": optimization_params.get("method", "grid"),
            "maximize": optimization_params.get("maximize", "Sharpe Ratio"),
            "return_heatmap": optimization_params.get("return_heatmap", False),
            "return_optimization": optimization_params.get(
                "return_optimization", False
            ),
        }

        # オプションパラメータの追加
        optional_params = ["max_tries", "random_state"]
        for param in optional_params:
            if param in optimization_params:
                optimize_kwargs[param] = optimization_params[param]

        # 制約条件の処理
        constraint = optimization_params.get("constraint")
        if constraint:
            if isinstance(constraint, str):
                # 事前定義された制約条件を使用
                if constraint in self.constraint_functions:
                    optimize_kwargs["constraint"] = self.constraint_functions[
                        constraint
                    ]
                else:
                    raise ValueError(f"Unknown constraint: {constraint}")
            elif callable(constraint):
                optimize_kwargs["constraint"] = constraint
            else:
                raise ValueError("Constraint must be callable or predefined string")

        # パラメータ範囲を追加
        for param_name, param_range in optimization_params["parameters"].items():
            optimize_kwargs[param_name] = param_range

        return optimize_kwargs

    def _create_constraint_functions(self) -> Dict[str, Callable]:
        """事前定義された制約条件関数"""

        def sma_cross_constraint(params):
            """SMAクロス戦略: 短期SMA < 長期SMA"""
            return params.n1 < params.n2

        def rsi_constraint(params):
            """RSI戦略: 適切な閾値範囲"""
            return (
                hasattr(params, "rsi_lower")
                and hasattr(params, "rsi_upper")
                and params.rsi_lower < params.rsi_upper
                and params.rsi_lower >= 10
                and params.rsi_upper <= 90
            )

        def macd_constraint(params):
            """MACD戦略: 適切なパラメータ関係"""
            return (
                hasattr(params, "fast")
                and hasattr(params, "slow")
                and hasattr(params, "signal")
                and params.fast < params.slow
                and params.signal < params.slow
            )

        def risk_management_constraint(params):
            """リスク管理: ストップロス < テイクプロフィット"""
            return (
                hasattr(params, "stop_loss")
                and hasattr(params, "take_profit")
                and 0.01 <= params.stop_loss <= 0.1
                and 0.01 <= params.take_profit <= 0.2
                and params.stop_loss < params.take_profit
            )

        return {
            "sma_cross": sma_cross_constraint,
            "rsi": rsi_constraint,
            "macd": macd_constraint,
            "risk_management": risk_management_constraint,
        }

    def _get_backtest_data(self, config: Dict[str, Any]) -> pd.DataFrame:
        """バックテスト用データの取得"""
        return self.data_service.get_ohlcv_for_backtest(
            symbol=config["symbol"],
            timeframe=config["timeframe"],
            start_date=config["start_date"],
            end_date=config["end_date"],
        )

    def _process_optimization_results(
        self,
        result: Union[pd.Series, Tuple],
        config: Dict[str, Any],
        optimization_params: Dict[str, Any],
    ) -> Dict[str, Any]:
        """最適化結果の処理"""

        # 結果の分解
        if optimization_params.get("return_heatmap", False):
            if optimization_params.get("return_optimization", False):
                stats, heatmap, optimization_result = result
            else:
                stats, heatmap = result
                optimization_result = None
        else:
            stats = result
            heatmap = None
            optimization_result = None

        # 基本結果の変換
        processed_result = self._convert_backtest_results(
            stats,
            config["strategy_name"],
            config["symbol"],
            config["timeframe"],
            config["initial_capital"],
            config.get("start_date"),
            config.get("end_date"),
        )

        # 最適化されたパラメータを追加
        optimized_strategy = stats.get("_strategy")
        if optimized_strategy:
            processed_result["optimized_parameters"] = {}
            for param_name in optimization_params["parameters"].keys():
                if hasattr(optimized_strategy, param_name):
                    processed_result["optimized_parameters"][param_name] = getattr(
                        optimized_strategy, param_name
                    )

        # ヒートマップデータを追加
        if heatmap is not None:
            processed_result["heatmap_data"] = heatmap
            processed_result["heatmap_summary"] = self._analyze_heatmap(heatmap)

        # SAMBO最適化結果を追加
        if optimization_result is not None:
            processed_result["optimization_details"] = {
                "method": "sambo",
                "n_calls": len(optimization_result.func_vals),
                "best_value": optimization_result.fun,
                "convergence": self._analyze_convergence(optimization_result),
            }

        # 最適化メタデータを追加
        processed_result["optimization_metadata"] = {
            "method": optimization_params.get("method", "grid"),
            "maximize": optimization_params.get("maximize", "Sharpe Ratio"),
            "max_tries": optimization_params.get("max_tries"),
            "parameter_space_size": self._calculate_parameter_space_size(
                optimization_params["parameters"]
            ),
            "optimization_timestamp": datetime.now().isoformat(),
        }

        return processed_result

    def _analyze_heatmap(self, heatmap: pd.Series) -> Dict[str, Any]:
        """ヒートマップデータの分析"""
        return {
            "best_combination": heatmap.idxmax(),
            "best_value": heatmap.max(),
            "worst_combination": heatmap.idxmin(),
            "worst_value": heatmap.min(),
            "mean_value": heatmap.mean(),
            "std_value": heatmap.std(),
            "total_combinations": len(heatmap),
        }

    def _analyze_convergence(self, optimization_result) -> Dict[str, Any]:
        """SAMBO最適化の収束分析"""
        func_vals = optimization_result.func_vals
        return {
            "initial_value": func_vals[0] if func_vals else None,
            "final_value": func_vals[-1] if func_vals else None,
            "improvement": (func_vals[-1] - func_vals[0]) if len(func_vals) > 1 else 0,
            "convergence_rate": self._calculate_convergence_rate(func_vals),
            "plateau_detection": self._detect_plateau(func_vals),
        }

    def _calculate_convergence_rate(self, func_vals: List[float]) -> float:
        """収束率の計算"""
        if len(func_vals) < 10:
            return 0.0

        # 最後の10回の改善率を計算
        recent_vals = func_vals[-10:]
        improvements = [
            recent_vals[i] - recent_vals[i - 1] for i in range(1, len(recent_vals))
        ]
        return np.mean(improvements) if improvements else 0.0

    def _detect_plateau(self, func_vals: List[float], threshold: float = 1e-6) -> bool:
        """プラトー（収束停滞）の検出"""
        if len(func_vals) < 20:
            return False

        # 最後の20回の変動を確認
        recent_vals = func_vals[-20:]
        variance = np.var(recent_vals)
        return variance < threshold

    def _calculate_parameter_space_size(self, parameters: Dict[str, Any]) -> int:
        """パラメータ空間のサイズを計算"""
        total_size = 1
        for param_range in parameters.values():
            if hasattr(param_range, "__len__"):
                total_size *= len(param_range)
            else:
                # rangeオブジェクトの場合
                try:
                    total_size *= len(list(param_range))
                except:
                    total_size *= 100  # デフォルト推定値
        return total_size

    def _save_heatmap(self, heatmap_data: pd.Series, filename: str) -> str:
        """ヒートマップの保存"""
        try:
            plot_heatmaps(
                heatmap_data, agg="mean", filename=filename, open_browser=False
            )
            print(f"ヒートマップを保存しました: {filename}")
            return filename
        except Exception as e:
            print(f"ヒートマップ保存エラー: {str(e)}")
            return ""

    def multi_objective_optimization(
        self,
        config: Dict[str, Any],
        objectives: List[str],
        weights: List[float] = None,
        optimization_params: Dict[str, Any] = None,
    ) -> Dict[str, Any]:
        """
        複数指標での最適化

        Args:
            config: バックテスト設定
            objectives: 最適化対象の指標リスト
            weights: 各指標の重み（Noneの場合は均等重み）
            optimization_params: 追加の最適化パラメータ

        Returns:
            最適化結果
        """
        if weights is None:
            weights = [1.0] * len(objectives)

        if len(objectives) != len(weights):
            raise ValueError("Number of objectives must match number of weights")

        def combined_objective(stats):
            """複合目的関数"""
            score = 0
            for obj, weight in zip(objectives, weights):
                if obj.startswith("-"):  # 最小化したい指標（負の符号付き）
                    metric_name = obj[1:]
                    value = stats.get(metric_name, 0)
                    score -= weight * value
                else:  # 最大化したい指標
                    value = stats.get(obj, 0)
                    score += weight * value
            return score

        # 最適化パラメータの準備
        if optimization_params is None:
            optimization_params = {}

        optimization_params["maximize"] = combined_objective
        optimization_params.setdefault("method", "sambo")
        optimization_params.setdefault("max_tries", 300)

        # 最適化実行
        result = self.optimize_strategy_enhanced(config, optimization_params)

        # マルチ目的最適化の詳細を追加
        result["multi_objective_details"] = {
            "objectives": objectives,
            "weights": weights,
            "individual_scores": self._calculate_individual_scores(
                result.get("performance_metrics", {}), objectives
            ),
        }

        return result

    def _calculate_individual_scores(
        self, performance_metrics: Dict[str, float], objectives: List[str]
    ) -> Dict[str, float]:
        """各目的関数の個別スコアを計算"""
        scores = {}
        for obj in objectives:
            if obj.startswith("-"):
                metric_name = obj[1:]
                scores[obj] = -performance_metrics.get(metric_name, 0)
            else:
                scores[obj] = performance_metrics.get(obj, 0)
        return scores

    def robustness_test(
        self,
        config: Dict[str, Any],
        test_periods: List[Tuple[str, str]],
        optimization_params: Dict[str, Any],
    ) -> Dict[str, Any]:
        """
        複数期間でのロバストネステスト

        Args:
            config: 基本設定
            test_periods: テスト期間のリスト [(start_date, end_date), ...]
            optimization_params: 最適化パラメータ

        Returns:
            ロバストネステスト結果
        """
        results = {}
        all_optimized_params = []

        print(f"ロバストネステスト開始: {len(test_periods)}期間")

        for i, (start_date, end_date) in enumerate(test_periods):
            print(f"期間 {i+1}/{len(test_periods)}: {start_date} - {end_date}")

            period_config = config.copy()
            period_config.update({"start_date": start_date, "end_date": end_date})

            try:
                result = self.optimize_strategy_enhanced(
                    period_config, optimization_params
                )

                results[f"period_{i+1}"] = result

                # 最適化されたパラメータを収集
                if "optimized_parameters" in result:
                    all_optimized_params.append(result["optimized_parameters"])

            except Exception as e:
                print(f"期間 {i+1} でエラー: {str(e)}")
                results[f"period_{i+1}"] = {"error": str(e)}

        # 結果の統合と分析
        robustness_analysis = self._analyze_robustness_results(
            results, all_optimized_params
        )

        return {
            "individual_results": results,
            "robustness_analysis": robustness_analysis,
            "test_periods": test_periods,
            "total_periods": len(test_periods),
        }

    def _analyze_robustness_results(
        self, results: Dict[str, Any], all_optimized_params: List[Dict[str, Any]]
    ) -> Dict[str, Any]:
        """ロバストネステスト結果の分析"""

        # パフォーマンス指標の統計
        performance_stats = {}
        metrics = ["total_return", "sharpe_ratio", "max_drawdown", "win_rate"]

        for metric in metrics:
            values = []
            for period_result in results.values():
                if "performance_metrics" in period_result:
                    values.append(period_result["performance_metrics"].get(metric, 0))

            if values:
                performance_stats[metric] = {
                    "mean": np.mean(values),
                    "std": np.std(values),
                    "min": np.min(values),
                    "max": np.max(values),
                    "consistency_score": 1
                    - (np.std(values) / (np.mean(values) + 1e-8)),
                }

        # パラメータの安定性分析
        parameter_stability = {}
        if all_optimized_params:
            param_names = set()
            for params in all_optimized_params:
                param_names.update(params.keys())

            for param_name in param_names:
                param_values = []
                for params in all_optimized_params:
                    if param_name in params:
                        param_values.append(params[param_name])

                if param_values:
                    parameter_stability[param_name] = {
                        "mean": np.mean(param_values),
                        "std": np.std(param_values),
                        "min": np.min(param_values),
                        "max": np.max(param_values),
                        "coefficient_of_variation": np.std(param_values)
                        / (np.mean(param_values) + 1e-8),
                    }

        # 総合ロバストネススコア
        robustness_score = self._calculate_robustness_score(
            performance_stats, parameter_stability
        )

        return {
            "performance_statistics": performance_stats,
            "parameter_stability": parameter_stability,
            "robustness_score": robustness_score,
            "successful_periods": len(
                [r for r in results.values() if "error" not in r]
            ),
            "failed_periods": len([r for r in results.values() if "error" in r]),
        }

    def _calculate_robustness_score(
        self, performance_stats: Dict[str, Any], parameter_stability: Dict[str, Any]
    ) -> float:
        """総合ロバストネススコアの計算"""

        # パフォーマンス一貫性スコア（0-1）
        performance_consistency = 0
        if performance_stats:
            consistency_scores = [
                stats.get("consistency_score", 0)
                for stats in performance_stats.values()
            ]
            performance_consistency = np.mean(consistency_scores)

        # パラメータ安定性スコア（0-1）
        parameter_consistency = 0
        if parameter_stability:
            cv_scores = [
                1 / (1 + stats.get("coefficient_of_variation", 1))
                for stats in parameter_stability.values()
            ]
            parameter_consistency = np.mean(cv_scores)

        # 総合スコア（重み付き平均）
        robustness_score = 0.7 * performance_consistency + 0.3 * parameter_consistency

        return max(0, min(1, robustness_score))
