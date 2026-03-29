"""
個体評価器

遺伝的アルゴリズムの個体評価を担当します。
"""

import hashlib
import logging
import threading
from datetime import datetime
from math import ceil
from typing import (
    Any,
    Dict,
    Optional,
    Tuple,
)

from cachetools import LRUCache  # type: ignore[import-untyped]
import pandas as pd
from pydantic import ValidationError

from app.services.backtest.config.backtest_config import BacktestRunConfig
from app.services.backtest.conversion.backtest_result_converter import (
    BacktestResultConverter,
)
from app.services.backtest.services.backtest_service import BacktestService

from app.services.auto_strategy.config import GAConfig
from app.services.auto_strategy.genes import StrategyGene
from app.services.auto_strategy.serializers.serialization import GeneSerializer


from ..fitness.optimized_fitness_calculator import OptimizedFitnessCalculator
from .optimized_data_provider import OptimizedBacktestDataProvider
from .evaluation_report import EvaluationReport, ScenarioEvaluation
from .optimized_evaluation_strategies import OptimizedEvaluationStrategy
from .run_config_builder import RunConfigBuilder

logger = logging.getLogger(__name__)


class IndividualEvaluator:
    """
    遺伝的アルゴリズムの個体評価を管理するエンジン

    バックテストサービスを介して個体（戦略遺伝子）のシミュレーションを
    実行し、得られた統計データ（利益、ドローダウン、勝率等）から
    適応度（Fitness）を算出します。
    LRU キャッシュを用いて同一遺伝子の再評価をスキップし、
    並列実行時の効率を高めます。
    """

    # デフォルトのキャッシュサイズ上限
    DEFAULT_MAX_CACHE_SIZE = 100

    def __init__(
        self,
        backtest_service: BacktestService,
        max_cache_size: Optional[int] = None,
    ):
        """初期化

        Args:
            backtest_service: バックテストサービス
            max_cache_size: データキャッシュの最大サイズ（LRU方式で古いエントリを削除）
        """
        self.backtest_service = backtest_service
        self._fixed_backtest_config: Optional[Dict[str, Any]] = None
        self._max_cache_size = max_cache_size or self.DEFAULT_MAX_CACHE_SIZE
        # データキャッシュ（重いデータ用）
        self._data_cache: LRUCache = LRUCache(maxsize=self._max_cache_size)
        # 評価結果キャッシュ（軽量、ヒット率向上用）
        # データより軽量なのでサイズを大きめに取る
        self._result_cache: LRUCache = LRUCache(maxsize=self._max_cache_size * 100)
        self._report_cache: LRUCache = LRUCache(maxsize=self._max_cache_size * 20)
        self._robustness_report_cache: LRUCache = LRUCache(
            maxsize=self._max_cache_size * 10
        )
        self._lock = threading.Lock()
        self._last_evaluation_report: Optional[EvaluationReport] = None

        # 統計情報
        self._cache_hits = 0
        self._cache_misses = 0

        self._initialize_components()

    def _initialize_components(self) -> None:
        """委譲先コンポーネントを初期化する。"""
        # 最適化されたフィットネス計算を使用
        self._fitness_calculator = OptimizedFitnessCalculator()
        # 最適化された評価ストラテジーを使用
        self._evaluation_strategy = OptimizedEvaluationStrategy(self, max_workers=2)
        self._run_config_builder = RunConfigBuilder()
        # 最適化されたデータプロバイダーを使用
        self._data_provider = OptimizedBacktestDataProvider(
            backtest_service=self.backtest_service,
            data_cache=self._data_cache,  # type: ignore
            lock=self._lock,
            prefetch_enabled=True,
            max_prefetch_workers=2,
        )

    def _build_cache_key(self, gene: Any) -> str:
        """遺伝子構造に基づく安定したキャッシュキーを生成する。

        id属性が有効な場合はそれを使用し、空文字列やNoneの場合は
        シリアライズ結果のMD5ハッシュで補完する。
        """
        gene_id = getattr(gene, "id", "") or ""
        if gene_id:
            return gene_id
        try:
            serialized = GeneSerializer().strategy_gene_to_dict(gene)
            raw = str(sorted(serialized.items(), key=lambda x: str(x[0])))
            return hashlib.md5(raw.encode()).hexdigest()
        except Exception:
            return str(gene)

    def set_backtest_config(self, backtest_config: Dict[str, Any]):
        """バックテスト設定を設定"""
        # バリデーションのためPydanticモデルを通す
        # strategy_configは個体ごとに生成されるため、ここではダミーを入れておく
        temp_config = backtest_config.copy()
        if "strategy_config" not in temp_config or not temp_config.get(
            "strategy_config", {}
        ).get("strategy_type"):
            temp_config["strategy_config"] = {
                "strategy_type": "GENERATED_GA",
                "parameters": {"strategy_gene": {}},  # ダミー
            }
        if "strategy_name" not in temp_config:
            temp_config["strategy_name"] = "Dummy"

        try:
            # モデル変換を試行してバリデーション（型変換などもここで行われる）
            # ここでの目的は必須フィールドの欠落を早期検知すること
            BacktestRunConfig(**temp_config)
        except ValidationError as e:
            logger.warning(f"バックテスト設定の初期バリデーション警告: {e}")
            # ここでは警告に留める

        self._fixed_backtest_config = backtest_config.copy()

        # 設定変更時は結果キャッシュもクリア（前提が変わるため）
        with self._lock:
            self._result_cache.clear()
            self._report_cache.clear()
            self._robustness_report_cache.clear()

    def clear_cache(self) -> None:
        """データキャッシュをクリア"""
        with self._lock:
            self._data_cache.clear()
            self._result_cache.clear()
            self._report_cache.clear()
            self._robustness_report_cache.clear()
            self._cache_hits = 0
            self._cache_misses = 0
            logger.info("データキャッシュと評価結果キャッシュをクリアしました")

    def get_cache_info(self) -> Dict[str, Any]:
        """キャッシュの状態情報を取得"""
        with self._lock:
            return {
                "data_cache_size": len(self._data_cache),
                "data_cache_max": self._max_cache_size,
                "result_cache_size": len(self._result_cache),
                "result_cache_max": self._result_cache.maxsize,
                "report_cache_size": len(self._report_cache),
                "report_cache_max": self._report_cache.maxsize,
                "robustness_report_cache_size": len(self._robustness_report_cache),
                "robustness_report_cache_max": self._robustness_report_cache.maxsize,
                "cache_hits": self._cache_hits,
                "cache_misses": self._cache_misses,
            }

    def get_last_evaluation_report(self) -> Optional[EvaluationReport]:
        """直近の評価レポートを取得する。"""
        return self._last_evaluation_report

    def get_cached_evaluation_report(
        self, individual: Any
    ) -> Optional[EvaluationReport]:
        """キャッシュ済みの評価レポートを取得する。"""
        try:
            gene = self._resolve_gene(individual)
            cache_key = self._build_cache_key(gene)
        except Exception:
            return None

        return self._report_cache.get(cache_key)

    def get_cached_robustness_report(
        self, individual: Any, config: GAConfig
    ) -> Optional[EvaluationReport]:
        """キャッシュ済みの robustness 評価レポートを取得する。"""
        try:
            gene = self._resolve_gene(individual)
            cache_key = self._build_robustness_cache_key(gene, config)
        except Exception:
            return None

        return self._robustness_report_cache.get(cache_key)

    def build_parallel_worker_initargs(
        self, config: GAConfig
    ) -> Optional[Tuple[Dict[str, Any], GAConfig, Dict[str, Any]]]:
        """並列ワーカー初期化に必要な引数を構築する。"""
        if not self._fixed_backtest_config:
            return None

        backtest_config = self._fixed_backtest_config.copy()
        shared_data: Dict[str, Any] = {}
        main_key, minute_key = self._build_backtest_cache_keys(backtest_config)

        main_data = self._get_cached_data(backtest_config)
        if main_data is not None and not getattr(main_data, "empty", False):
            shared_data["main_data"] = {"key": main_key, "data": main_data}

        minute_data = self._get_cached_minute_data(backtest_config)
        if minute_data is not None:
            shared_data["minute_data"] = {"key": minute_key, "data": minute_data}

        return (backtest_config, config, shared_data)

    @staticmethod
    def _build_backtest_cache_keys(
        backtest_config: Dict[str, Any],
    ) -> Tuple[Tuple[Any, ...], Tuple[Any, ...]]:
        """メイン足・1分足キャッシュキーを構築する。"""
        symbol = backtest_config.get("symbol")
        timeframe = backtest_config.get("timeframe")
        start_date = backtest_config.get("start_date")
        end_date = backtest_config.get("end_date")
        main_key = (symbol, timeframe, str(start_date), str(end_date))
        minute_key = ("minute", symbol, "1m", str(start_date), str(end_date))
        return main_key, minute_key

    def __getstate__(self):
        """Pickle化時の状態取得（ロック、キャッシュ、委譲先を除外）"""
        state = self.__dict__.copy()
        # ロックはPickle不可
        if "_lock" in state:
            del state["_lock"]

        # キャッシュはプロセス間で共有せず、転送コスト削減のために除外する
        # ワーカープロセスは空のキャッシュで開始する
        if "_data_cache" in state:
            del state["_data_cache"]
        if "_result_cache" in state:
            del state["_result_cache"]
        if "_report_cache" in state:
            del state["_report_cache"]
        if "_robustness_report_cache" in state:
            del state["_robustness_report_cache"]

        # 委譲先コンポーネントは自己参照を含むため除外（復元時に再生成）
        if "_fitness_calculator" in state:
            del state["_fitness_calculator"]
        if "_evaluation_strategy" in state:
            del state["_evaluation_strategy"]
        if "_run_config_builder" in state:
            del state["_run_config_builder"]
        if "_data_provider" in state:
            del state["_data_provider"]

        return state

    def __setstate__(self, state):
        """Pickle復元時の状態設定（ロック、キャッシュ、委譲先を再生成）"""
        self.__dict__.update(state)
        self._lock = threading.Lock()

        # キャッシュの再初期化（空の状態）
        if not hasattr(self, "_data_cache"):
            self._data_cache = LRUCache(maxsize=self._max_cache_size)
        if not hasattr(self, "_result_cache"):
            self._result_cache = LRUCache(maxsize=self._max_cache_size * 100)
        if not hasattr(self, "_report_cache"):
            self._report_cache = LRUCache(maxsize=self._max_cache_size * 20)
        if not hasattr(self, "_robustness_report_cache"):
            self._robustness_report_cache = LRUCache(maxsize=self._max_cache_size * 10)
        if not hasattr(self, "_last_evaluation_report"):
            self._last_evaluation_report = None

        self._initialize_components()

    def evaluate(self, individual: Any, config: GAConfig) -> Tuple[float, ...]:
        """
        個体を評価し、適応度（Fitness）のタプルを返す

        DEAP から呼び出されるメインの評価関数です。
        遺伝子のデコード、キャッシュの確認、実際のバックテスト実行、
        および結果のキャッシュまでを一貫して行います。

        Args:
            individual: 評価対象（StrategyGene オブジェクト、または辞書/リスト）
            config: GA の目的関数、ペナルティ、期間分割等の設定

        Returns:
            複数の目的関数に対応した評価値のタプル
        """
        return self.evaluate_individual(individual, config)

    def evaluate_individual(
        self, individual: Any, config: GAConfig
    ) -> Tuple[float, ...]:
        """
        個体を評価し、適応度（Fitness）のタプルを返す（実体）
        """
        try:
            gene = self._resolve_gene(individual)

            # 安定したキャッシュキーの生成
            cache_key = self._build_cache_key(gene)

            # ロックフリー読み取り: キャッシュヒット時はロック不要
            cached = self._result_cache.get(cache_key)
            if cached is not None:
                self._cache_hits += 1
                self._last_evaluation_report = self._report_cache.get(cache_key)
                return cached
            self._cache_misses += 1

            # バックテスト設定のベースを取得
            base_backtest_config: Dict[str, Any] = (
                self._fixed_backtest_config.copy()
                if self._fixed_backtest_config
                else {}
            )

            # 評価実行（戦略ルーティングに委譲）
            report = self._evaluation_strategy.execute_report(
                gene, base_backtest_config, config
            )
            fitness = report.aggregated_fitness
            self._last_evaluation_report = report

            # 結果をキャッシュ（ロック付き書き込み）
            with self._lock:
                self._result_cache[cache_key] = fitness
                self._report_cache[cache_key] = report

            return fitness

        except Exception as e:
            logger.error(f"個体評価エラー: {e}")
            return self._fitness_calculator.get_penalty_values(config)

    def evaluate_robustness_report(
        self, individual: Any, config: GAConfig
    ) -> Optional[EvaluationReport]:
        """二段階選抜用の robustness 評価レポートを返す。"""
        try:
            gene = self._resolve_gene(individual)
            cache_key = self._build_robustness_cache_key(gene, config)
            cached = self._robustness_report_cache.get(cache_key)
            if cached is not None:
                self._last_evaluation_report = cached
                return cached

            base_backtest_config: Dict[str, Any] = (
                self._fixed_backtest_config.copy()
                if self._fixed_backtest_config
                else {}
            )
            report = self._evaluation_strategy.execute_robustness_report(
                gene, base_backtest_config, config
            )
            self._last_evaluation_report = report

            with self._lock:
                self._robustness_report_cache[cache_key] = report

            return report
        except Exception as e:
            logger.error(f"robustness 評価エラー: {e}")
            return None

    def _resolve_gene(self, individual: Any) -> Any:
        """評価対象から遺伝子本体を取り出す。"""
        if isinstance(individual, StrategyGene):
            return individual
        if isinstance(individual, dict):
            gene_serializer = GeneSerializer()
            return gene_serializer.dict_to_strategy_gene(individual, StrategyGene)
        if isinstance(individual, list):
            if len(individual) > 0 and isinstance(individual[0], StrategyGene):
                return individual[0]
            if len(individual) > 0 and hasattr(individual[0], "id"):
                return individual[0]
        return individual

    def _build_robustness_cache_key(self, gene: Any, config: GAConfig) -> str:
        """robustness 評価用のキャッシュキーを生成する。"""
        base_key = self._build_cache_key(gene)
        signature = (
            bool(getattr(config, "enable_two_stage_selection", True)),
            float(getattr(config, "two_stage_min_pass_rate", 0.0) or 0.0),
            tuple(getattr(config, "robustness_validation_symbols", None) or ()),
            tuple(
                (
                    str(window.get("name", "")),
                    str(window.get("start_date", "")),
                    str(window.get("end_date", "")),
                )
                for window in (getattr(config, "robustness_regime_windows", []) or [])
                if isinstance(window, dict)
            ),
            tuple(getattr(config, "robustness_stress_slippage", ()) or ()),
            tuple(
                getattr(config, "robustness_stress_commission_multipliers", ()) or ()
            ),
            str(getattr(config, "robustness_aggregate_method", "robust")),
            bool(getattr(config, "enable_purged_kfold", False)),
            int(getattr(config, "purged_kfold_splits", 0) or 0),
            float(getattr(config, "purged_kfold_embargo", 0.0) or 0.0),
            bool(getattr(config, "enable_walk_forward", False)),
            int(getattr(config, "wfa_n_folds", 0) or 0),
            float(getattr(config, "wfa_train_ratio", 0.0) or 0.0),
            bool(getattr(config, "wfa_anchored", False)),
            float(getattr(config, "oos_split_ratio", 0.0) or 0.0),
            float(getattr(config, "oos_fitness_weight", 0.0) or 0.0),
        )
        return f"{base_key}:robustness:{hash(signature)}"

    def _get_cached_data(self, backtest_config: Dict[str, Any]) -> Any:
        """キャッシュされたバックテストデータを取得"""
        return self._data_provider.get_cached_backtest_data(backtest_config)

    def _get_cached_minute_data(self, backtest_config: Dict[str, Any]) -> Any:
        """
        1分足データをキャッシュから取得

        1分足シミュレーション用に使用します。

        Args:
            backtest_config: バックテスト設定

        Returns:
            1分足のDataFrame、またはデータが存在しない場合はNone
        """
        return self._data_provider.get_cached_minute_data(backtest_config)

    def _get_cached_ohlcv_data(
        self,
        symbol: str,
        timeframe: str,
        start_date: Any,
        end_date: Any,
        cache_prefix: str = "ohlcv",
    ) -> Any:
        """
        OHLCVデータをキャッシュから汎用的に取得

        HybridIndividualEvaluatorなどサブクラスからの利用を想定した
        汎用的なデータ取得メソッドです。

        Args:
            symbol: 通貨ペア
            timeframe: 時間軸
            start_date: 開始日時
            end_date: 終了日時
            cache_prefix: キャッシュキーのプレフィックス

        Returns:
            OHLCVのDataFrame、またはデータが存在しない場合はNone
        """
        return self._data_provider.get_cached_ohlcv_data(
            symbol=symbol,
            timeframe=timeframe,
            start_date=start_date,
            end_date=end_date,
            cache_prefix=cache_prefix,
        )

    def _perform_single_evaluation(
        self, gene, backtest_config: Dict[str, Any], config: GAConfig
    ) -> Tuple[float, ...]:
        """単一期間評価を実行し、適応度タプルのみ返す。"""
        return self._perform_single_evaluation_report(
            gene,
            backtest_config,
            config,
        ).fitness

    def _perform_single_evaluation_report(
        self,
        gene,
        backtest_config: Dict[str, Any],
        config: GAConfig,
        *,
        scenario_name: str = "single",
        metadata: Optional[Dict[str, Any]] = None,
    ) -> ScenarioEvaluation:
        """
        単一期間でのバックテストとフィットネス評価を実行し、評価レポートを返す。

        Args:
            gene: 評価対象の戦略遺伝子
            backtest_config: バックテスト実行用設定
            config: GA設定

        Returns:
            単一シナリオの評価結果
        """
        try:
            prepared_backtest_config = self._prepare_backtest_config_for_evaluation(
                gene, backtest_config
            )

            # 1. 実行用設定の構築
            run_config = self._prepare_run_config(gene, prepared_backtest_config, config)
            if not run_config:
                return ScenarioEvaluation(
                    name=scenario_name,
                    fitness=tuple(0.0 for _ in config.objectives),
                    passed=False,
                    metadata=metadata.copy() if metadata else {},
                )

            # 2. データの準備
            data = self._get_cached_data(prepared_backtest_config)

            evaluation_start = prepared_backtest_config.get("_evaluation_start")
            if evaluation_start is not None:
                run_config["_include_raw_stats"] = True

            # モデル外のオブジェクトを注入
            self._inject_external_objects(run_config, backtest_config, config)

            # 3. バックテスト実行
            result = self.backtest_service.run_backtest(
                config=run_config, preloaded_data=data
            )

            if (
                evaluation_start is not None
                and isinstance(result, dict)
                and isinstance(data, pd.DataFrame)
            ):
                result = self._apply_evaluation_window_to_result(
                    result,
                    result.get("_raw_stats"),
                    data,
                    evaluation_start,
                    backtest_config.get("end_date"),
                )

            # 4. 追加のコンテキスト情報（ML予測シグナルなど）を取得
            evaluation_context = self._get_evaluation_context(
                gene, backtest_config, config
            )

            # 5. フィットネス計算（フィットネス計算機に委譲）
            fitness = self._calculate_multi_objective_fitness(
                result, config, **evaluation_context
            )
            performance_metrics = self._extract_performance_metrics(result)
            scenario_metadata = metadata.copy() if metadata else {}
            scenario_metadata.update(
                {
                    "start_date": str(backtest_config.get("start_date")),
                    "end_date": str(backtest_config.get("end_date")),
                }
            )
            if prepared_backtest_config.get("_evaluation_start") is not None:
                scenario_metadata["evaluation_start"] = str(
                    prepared_backtest_config["_evaluation_start"]
                )

            return ScenarioEvaluation(
                name=scenario_name,
                fitness=tuple(float(value) for value in fitness),
                passed=self._is_backtest_result_passing(result, config),
                metadata=scenario_metadata,
                performance_metrics=performance_metrics,
            )

        except Exception as e:
            logger.error(f"単一評価実行エラー: {e}")
            scenario_metadata = metadata.copy() if metadata else {}
            scenario_metadata["error"] = str(e)
            return ScenarioEvaluation(
                name=scenario_name,
                fitness=tuple(0.0 for _ in config.objectives),
                passed=False,
                metadata=scenario_metadata,
            )

    def _prepare_backtest_config_for_evaluation(
        self, gene: Any, backtest_config: Dict[str, Any]
    ) -> Dict[str, Any]:
        """指標 warmup 用に評価開始前の履歴を含む実行設定へ変換する。"""
        prepared_config = backtest_config.copy()
        start_date = backtest_config.get("start_date")
        timeframe = str(backtest_config.get("timeframe", ""))

        if not start_date or not timeframe:
            return prepared_config

        warmup_bars = self._estimate_required_warmup_bars(gene, timeframe)
        if warmup_bars <= 0:
            return prepared_config

        start_timestamp = pd.Timestamp(start_date)
        base_minutes = self._timeframe_to_minutes(timeframe)
        adjusted_start = start_timestamp - pd.to_timedelta(
            warmup_bars * base_minutes, unit="m"
        )

        prepared_config["_evaluation_start"] = start_date
        prepared_config["start_date"] = self._format_datetime_like(
            start_date, adjusted_start
        )
        return prepared_config

    def _is_backtest_result_passing(
        self, backtest_result: Dict[str, Any], config: GAConfig
    ) -> bool:
        """バックテスト結果が基本制約を満たすか判定する。"""
        try:
            metrics = self._extract_performance_metrics(backtest_result)
            constraints = getattr(config, "fitness_constraints", {}) or {}

            total_trades = int(metrics.get("total_trades", 0) or 0)
            if total_trades <= 0:
                return False

            min_trades = int(constraints.get("min_trades", 0) or 0)
            if total_trades < min_trades:
                return False

            total_return = float(metrics.get("total_return", 0.0) or 0.0)
            if total_return < 0.0:
                return False

            max_drawdown_limit = constraints.get("max_drawdown_limit")
            max_drawdown = float(metrics.get("max_drawdown", 0.0) or 0.0)
            if max_drawdown_limit is not None and max_drawdown > max_drawdown_limit:
                return False

            min_sharpe_ratio = float(constraints.get("min_sharpe_ratio", 0.0) or 0.0)
            sharpe_ratio = float(metrics.get("sharpe_ratio", 0.0) or 0.0)
            if sharpe_ratio < min_sharpe_ratio:
                return False

            return True
        except Exception:
            return False

    def _estimate_required_warmup_bars(self, gene: Any, base_timeframe: str) -> int:
        """戦略実行前に必要な warmup バー数を推定する。"""
        base_minutes = self._timeframe_to_minutes(base_timeframe)
        max_bars = 0

        indicators = getattr(gene, "indicators", []) or []
        for indicator in indicators:
            if not getattr(indicator, "enabled", True):
                continue

            lookback = self._extract_lookback_from_parameters(
                getattr(indicator, "parameters", {}) or {}
            )
            indicator_timeframe = getattr(indicator, "timeframe", None) or base_timeframe
            timeframe_scale = max(
                1, ceil(self._timeframe_to_minutes(indicator_timeframe) / base_minutes)
            )
            max_bars = max(max_bars, (lookback + 1) * timeframe_scale)

        position_sizing_gene = getattr(gene, "position_sizing_gene", None)
        if position_sizing_gene and getattr(position_sizing_gene, "enabled", True):
            max_bars = max(
                max_bars,
                int(
                    max(
                        getattr(position_sizing_gene, "lookback_period", 0),
                        getattr(position_sizing_gene, "atr_period", 0),
                    )
                )
                + 1,
            )

        for tpsl_attr in ("tpsl_gene", "long_tpsl_gene", "short_tpsl_gene"):
            tpsl_gene = getattr(gene, tpsl_attr, None)
            if tpsl_gene and getattr(tpsl_gene, "enabled", True):
                max_bars = max(
                    max_bars,
                    int(
                        max(
                            getattr(tpsl_gene, "lookback_period", 0),
                            getattr(tpsl_gene, "atr_period", 0),
                        )
                    )
                    + 1,
                )

        return int(max_bars)

    @staticmethod
    def _extract_lookback_from_parameters(parameters: Dict[str, Any]) -> int:
        """インジケーターパラメータから lookback 長を推定する。"""
        if not isinstance(parameters, dict):
            return 0

        excluded_tokens = ("multiplier", "threshold", "offset", "shift", "std")
        lookback_tokens = ("length", "period", "window", "lookback", "span")
        lookback = 0

        for key, value in parameters.items():
            key_str = str(key).lower()
            if any(token in key_str for token in excluded_tokens):
                continue
            if not any(token in key_str for token in lookback_tokens):
                continue
            if isinstance(value, (int, float)) and value > 0:
                lookback = max(lookback, int(ceil(float(value))))

        return lookback

    @staticmethod
    def _timeframe_to_minutes(timeframe: str) -> int:
        """timeframe 文字列を分単位へ変換する。"""
        timeframe_map = {
            "1m": 1,
            "5m": 5,
            "15m": 15,
            "30m": 30,
            "1h": 60,
            "4h": 240,
            "1d": 1440,
        }
        return timeframe_map.get(str(timeframe), 60)

    @staticmethod
    def _format_datetime_like(original_value: Any, timestamp: pd.Timestamp) -> Any:
        """元の入力型に合わせて Timestamp を整形する。"""
        if isinstance(original_value, pd.Timestamp):
            return timestamp
        if isinstance(original_value, datetime):
            return timestamp.to_pydatetime()
        if isinstance(original_value, str):
            return str(timestamp)
        return timestamp

    def _apply_evaluation_window_to_result(
        self,
        backtest_result: Dict[str, Any],
        raw_stats: Any,
        market_data: pd.DataFrame,
        evaluation_start: Any,
        evaluation_end: Any,
    ) -> Dict[str, Any]:
        """warmup を除外した評価窓だけでバックテスト結果を再計算する。"""
        if raw_stats is None or market_data is None or market_data.empty:
            return backtest_result

        raw_equity_curve = getattr(raw_stats, "_equity_curve", None)
        if raw_equity_curve is None or getattr(raw_equity_curve, "empty", True):
            return backtest_result

        market_df = market_data.copy()
        market_df = market_df.sort_index()

        start_ts = self._normalize_timestamp_to_index(
            evaluation_start, market_df.index
        )
        end_ts = self._normalize_timestamp_to_index(evaluation_end, market_df.index)

        start_pos = int(market_df.index.searchsorted(start_ts, side="left"))
        end_pos = int(market_df.index.searchsorted(end_ts, side="right"))

        if start_pos >= end_pos:
            logger.warning(
                "評価窓のトリミングに失敗しました: start=%s end=%s",
                evaluation_start,
                evaluation_end,
            )
            return backtest_result

        trimmed_market_data = market_df.iloc[start_pos:end_pos].copy()
        if trimmed_market_data.empty:
            return backtest_result

        trimmed_equity_curve = self._slice_equity_curve_for_window(
            raw_equity_curve,
            trimmed_market_data.index,
            start_pos,
            end_pos,
            backtest_result.get("initial_capital", 0.0),
        )
        equity_values = pd.to_numeric(
            trimmed_equity_curve["Equity"], errors="coerce"
        ).fillna(float(backtest_result.get("initial_capital", 0.0))).to_numpy()

        trimmed_trades = self._slice_trades_for_window(
            getattr(raw_stats, "_trades", None),
            start_pos,
            end_pos,
        )

        normalized_market_data = self._normalize_ohlc_data_for_stats(
            trimmed_market_data
        )
        window_stats = self._compute_window_stats(
            trimmed_trades,
            equity_values,
            normalized_market_data,
        )

        converter = BacktestResultConverter()
        adjusted_result = converter.convert_backtest_results(
            stats=window_stats,
            strategy_name=str(backtest_result.get("strategy_name", "")),
            symbol=str(backtest_result.get("symbol", "")),
            timeframe=str(backtest_result.get("timeframe", "")),
            initial_capital=float(backtest_result.get("initial_capital", 0.0)),
            start_date=start_ts.to_pydatetime(),
            end_date=end_ts.to_pydatetime(),
            config_json=backtest_result.get("config_json", {}),
        )
        adjusted_result["_raw_stats"] = window_stats
        return adjusted_result

    @staticmethod
    def _normalize_timestamp_to_index(
        value: Any, index: pd.Index
    ) -> pd.Timestamp:
        """インデックスのタイムゾーンに合わせて Timestamp を正規化する。"""
        timestamp = pd.Timestamp(value)
        index_tz = getattr(index, "tz", None)

        if index_tz is None and timestamp.tzinfo is not None:
            return timestamp.tz_localize(None)
        if index_tz is not None and timestamp.tzinfo is None:
            return timestamp.tz_localize(index_tz)
        if index_tz is not None and timestamp.tzinfo != index_tz:
            return timestamp.tz_convert(index_tz)
        return timestamp

    @staticmethod
    def _normalize_ohlc_data_for_stats(market_data: pd.DataFrame) -> pd.DataFrame:
        """backtesting.py が期待する大文字 OHLCV カラムへ正規化する。"""
        normalized = market_data.copy()
        normalized.columns = [
            column.capitalize() if isinstance(column, str) else column
            for column in normalized.columns
        ]
        if "Volume" not in normalized.columns:
            normalized["Volume"] = 0.0
        return normalized

    @staticmethod
    def _slice_equity_curve_for_window(
        raw_equity_curve: pd.DataFrame,
        target_index: pd.Index,
        start_pos: int,
        end_pos: int,
        initial_capital: float,
    ) -> pd.DataFrame:
        """評価窓に対応するエクイティカーブを切り出す。"""
        if not isinstance(raw_equity_curve, pd.DataFrame) or raw_equity_curve.empty:
            return pd.DataFrame({"Equity": [initial_capital] * len(target_index)}, index=target_index)

        if len(raw_equity_curve) >= end_pos:
            trimmed = raw_equity_curve.iloc[start_pos:end_pos].copy()
        else:
            trimmed = raw_equity_curve.reindex(target_index).copy()

        trimmed = trimmed.reindex(target_index).ffill().bfill()
        if "Equity" not in trimmed.columns:
            trimmed["Equity"] = float(initial_capital)
        if "DrawdownPct" not in trimmed.columns:
            trimmed["DrawdownPct"] = 0.0
        return trimmed

    @staticmethod
    def _slice_trades_for_window(
        raw_trades: Any,
        start_pos: int,
        end_pos: int,
    ) -> pd.DataFrame:
        """評価窓内のトレードだけを抽出し、バー番号を窓内基準へ補正する。"""
        if not isinstance(raw_trades, pd.DataFrame):
            return pd.DataFrame()

        trades_df = raw_trades.copy()
        if trades_df.empty:
            return trades_df

        if {"EntryBar", "ExitBar"}.issubset(trades_df.columns):
            entry_bars = pd.to_numeric(trades_df["EntryBar"], errors="coerce")
            exit_bars = pd.to_numeric(trades_df["ExitBar"], errors="coerce")
            mask = (
                entry_bars.notna()
                & exit_bars.notna()
                & (entry_bars >= start_pos)
                & (exit_bars < end_pos)
            )
            trades_df = trades_df.loc[mask].copy()
            if trades_df.empty:
                return trades_df
            trades_df["EntryBar"] = (
                pd.to_numeric(trades_df["EntryBar"], errors="coerce").astype(int)
                - start_pos
            )
            trades_df["ExitBar"] = (
                pd.to_numeric(trades_df["ExitBar"], errors="coerce").astype(int)
                - start_pos
            )
        return trades_df

    def _compute_window_stats(
        self,
        trades_df: pd.DataFrame,
        equity_values: Any,
        ohlc_data: pd.DataFrame,
    ) -> Any:
        """評価窓だけを対象に backtesting.py の統計を再計算する。"""
        from backtesting._stats import compute_stats

        return compute_stats(
            trades=trades_df,
            equity=equity_values,
            ohlc_data=ohlc_data,
            strategy_instance=None,
        )

    def _prepare_run_config(
        self, gene, backtest_config: Dict[str, Any], config: GAConfig
    ) -> Optional[Dict[str, Any]]:
        """バックテスト実行用設定の構築（高速化版）"""
        return self._run_config_builder.build_run_config(gene, backtest_config, config)

    def _inject_external_objects(
        self,
        run_config: Dict[str, Any],
        backtest_config: Dict[str, Any],
        config: GAConfig,
    ) -> None:
        """実行設定への外部オブジェクト注入（1分足データ）"""
        minute_data = self._get_cached_minute_data(backtest_config)
        self._run_config_builder.inject_external_objects(
            run_config,
            minute_data=minute_data,
        )

    def _get_evaluation_context(
        self, gene, backtest_config: Dict[str, Any], config: GAConfig
    ) -> Dict[str, Any]:
        """評価計算に 필요한追加コンテキストを取得（サブクラスでオーバーライド）"""
        return {}

    # --- EvaluationStrategy への委譲メソッド（バックワード互換性・テスト用） ---

    def _execute_evaluation_logic(
        self, gene: Any, base_backtest_config: Dict[str, Any], config: GAConfig
    ) -> Tuple[float, ...]:
        """具体的な評価プロセスを振り分け（EvaluationStrategyに委譲）"""
        return self._evaluation_strategy.execute(gene, base_backtest_config, config)

    def _evaluate_with_oos(
        self,
        gene,
        base_backtest_config: Dict[str, Any],
        config: GAConfig,
        oos_ratio: float,
        oos_weight: float,
    ) -> Tuple[float, ...]:
        """OOS検証を含む評価（EvaluationStrategyに委譲）"""
        return self._evaluation_strategy._evaluate_with_oos(
            gene, base_backtest_config, config, oos_ratio, oos_weight
        )

    def _evaluate_with_walk_forward(
        self,
        gene,
        base_backtest_config: Dict[str, Any],
        config: GAConfig,
    ) -> Tuple[float, ...]:
        """Walk-Forward Analysis による評価（EvaluationStrategyに委譲）"""
        return self._evaluation_strategy._evaluate_with_walk_forward(
            gene, base_backtest_config, config
        )

    # --- FitnessCalculator への委譲メソッド（バックワード互換性・サブクラス用） ---

    def _extract_performance_metrics(
        self, backtest_result: Dict[str, Any]
    ) -> Dict[str, Any]:
        """バックテスト結果からパフォーマンスメトリクスを抽出（FitnessCalculatorに委譲）"""
        return self._fitness_calculator.extract_performance_metrics(backtest_result)

    def _calculate_fitness(
        self, backtest_result: Dict[str, Any], config: GAConfig, **kwargs
    ) -> float:
        """フィットネス計算（FitnessCalculatorに委譲）"""
        return self._fitness_calculator.calculate_fitness(
            backtest_result, config, **kwargs
        )

    def _calculate_long_short_balance(self, backtest_result: Dict[str, Any]) -> float:
        """ロング・ショートバランススコア計算（FitnessCalculatorに委譲）"""
        return self._fitness_calculator.calculate_long_short_balance(backtest_result)

    def _calculate_multi_objective_fitness(
        self, backtest_result: Dict[str, Any], config: GAConfig, **kwargs
    ) -> Tuple[float, ...]:
        """多目的適応度計算（FitnessCalculatorに委譲）"""
        return self._fitness_calculator.calculate_multi_objective_fitness(
            backtest_result, config, **kwargs
        )


# 旧 shim モジュール向けの公開シンボルをここでも提供する
from .evaluation_metrics import (  # noqa: E402, F401
    REFERENCE_TRADES_PER_DAY,
    _calculate_ulcer_index_numba,
    _ensure_datetime,
    calculate_trade_frequency_penalty,
    calculate_ulcer_index,
)

__all__ = [
    "IndividualEvaluator",
    "calculate_trade_frequency_penalty",
    "calculate_ulcer_index",
]
