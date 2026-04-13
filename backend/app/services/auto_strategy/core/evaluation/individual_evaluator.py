"""
個体評価器

遺伝的アルゴリズムの個体評価を担当します。
"""

from __future__ import annotations

import hashlib
import logging
import threading
from typing import (
    Any,
    Dict,
    Optional,
    Tuple,
)

import pandas as pd
from cachetools import LRUCache
from pydantic import ValidationError

from app.types import SerializableValue

from app.services.auto_strategy.config import GAConfig
from app.services.auto_strategy.config.helpers import (
    normalize_robustness_regime_windows,
)
from app.services.auto_strategy.genes import StrategyGene
from app.services.auto_strategy.serializers.serialization import GeneSerializer
from app.services.backtest.config.backtest_config import BacktestRunConfig
from app.services.backtest.execution.backtest_executor import (
    BacktestEarlyTerminationError,
)
from app.services.backtest.services.backtest_service import BacktestService

from ..fitness.fitness_calculator import FitnessCalculator
from .evaluation_metrics import (
    calculate_trade_frequency_penalty,
    calculate_ulcer_index,
)
from .backtest_data_provider import BacktestDataProvider
from .evaluation_fidelity import adjust_backtest_config_for_fidelity, is_coarse_fidelity
from .evaluation_report import EvaluationReport, ScenarioEvaluation
from .evaluation_strategies import EvaluationStrategy
from .evaluation_window_service import EvaluationWindowService
from .run_config_builder import RunConfigBuilder

logger = logging.getLogger(__name__)


class IndividualEvaluator(EvaluationWindowService):
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
        self._fitness_calculator = FitnessCalculator()
        self._evaluation_strategy = EvaluationStrategy(self)
        self._run_config_builder = RunConfigBuilder()
        self._gene_serializer = GeneSerializer()
        self._data_provider = BacktestDataProvider(
            backtest_service=self.backtest_service,
            data_cache=self._data_cache,  # type: ignore
            lock=self._lock,
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
            serialized = self._gene_serializer.strategy_gene_to_dict(gene)
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
        if "_gene_serializer" in state:
            del state["_gene_serializer"]
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

    def evaluate(
        self,
        individual: Any,
        config: GAConfig,
        force_refresh: bool = False,
    ) -> Tuple[float, ...]:
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
        return self.evaluate_individual(
            individual,
            config,
            force_refresh=force_refresh,
        )

    def evaluate_individual(
        self,
        individual: Any,
        config: GAConfig,
        force_refresh: bool = False,
    ) -> Tuple[float, ...]:
        """
        個体を評価し、適応度（Fitness）のタプルを返します（評価ロジックの実体）。

        このメソッドは、以下のステップで個体を評価します：
        1. 引数 `individual` を `StrategyGene` オブジェクトに解決（`_resolve_gene`）。
        2. 遺伝子構造から一意なキャッシュキーを生成（`_build_cache_key`）。
        3. `force_refresh` が False の場合、結果キャッシュ（`_result_cache`）を確認。
        4. キャッシュミスの場合、`EvaluationStrategy` を使用してバックテストを実行し、`EvaluationReport` を生成。
        5. 生成された適応度（`aggregated_fitness`）をキャッシュに保存して返却。

        Args:
            individual (Any): 評価対象の個体。`StrategyGene` オブジェクト、またはそのシリアライズされた辞書/リスト。
            config (GAConfig): GAの実行設定。目的関数の定義やペナルティ、期間分割（OOS/WFA）の設定を含みます。
            force_refresh (bool): Trueの場合、キャッシュを無視して強制的に再評価を実行します。デフォルトはFalse。

        Returns:
            Tuple[float, ...]: 設定された目的関数に対応する評価値のタプル。
                多目的最適化の場合は複数の値、単一目的の場合は要素数1のタプルとなります。
                バックテスト中にエラーが発生した場合は、設定に基づいたペナルティ値が返されます。

        Note:
            - スレッドセーフ: キャッシュの書き込み時およびデータ提供サービスへのアクセス時にはロック制御を行います。
            - 効率化: LRUキャッシュを使用して同一遺伝子の重複評価を回避し、大規模な進化計算の高速化を図っています。
        """
        try:
            gene = self._resolve_gene(individual)

            cache_key = self._build_cache_key(gene)

            with self._lock:
                cached = None if force_refresh else self._result_cache.get(cache_key)
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
            report.metadata.setdefault(
                "evaluation_fidelity",
                "coarse" if is_coarse_fidelity(config) else "full",
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

    def _resolve_gene(self, individual: object) -> object:
        """評価対象から遺伝子本体を取り出す。"""
        if isinstance(individual, StrategyGene):
            return individual
        if isinstance(individual, dict):
            return self._gene_serializer.dict_to_strategy_gene(individual, StrategyGene)
        if isinstance(individual, list):
            if len(individual) > 0 and isinstance(individual[0], StrategyGene):
                return individual[0]
            if len(individual) > 0 and hasattr(individual[0], "id"):
                return individual[0]
        return individual

    def _build_robustness_cache_key(self, gene: Any, config: GAConfig) -> str:
        """robustness 評価用のキャッシュキーを生成する。"""
        base_key = self._build_cache_key(gene)
        evaluation_config = getattr(config, "evaluation_config", None)
        two_stage_config = getattr(config, "two_stage_selection_config", None)
        robustness_config = getattr(config, "robustness_config", None)
        signature = (
            bool(getattr(two_stage_config, "enabled", True)),
            float(getattr(two_stage_config, "min_pass_rate", 0.0) or 0.0),
            tuple(getattr(robustness_config, "validation_symbols", None) or ()),
            tuple(
                window.signature
                for window in normalize_robustness_regime_windows(
                    getattr(robustness_config, "regime_windows", [])
                )
            ),
            tuple(getattr(robustness_config, "stress_slippage", ()) or ()),
            tuple(
                getattr(robustness_config, "stress_commission_multipliers", ()) or ()
            ),
            str(getattr(robustness_config, "aggregate_method", "robust")),
            bool(getattr(config, "enable_purged_kfold", False)),
            int(getattr(config, "purged_kfold_splits", 0) or 0),
            float(getattr(config, "purged_kfold_embargo", 0.0) or 0.0),
            bool(getattr(evaluation_config, "enable_walk_forward", False)),
            int(getattr(evaluation_config, "wfa_n_folds", 0) or 0),
            float(getattr(evaluation_config, "wfa_train_ratio", 0.0) or 0.0),
            bool(getattr(evaluation_config, "wfa_anchored", False)),
            float(getattr(evaluation_config, "oos_split_ratio", 0.0) or 0.0),
            float(getattr(evaluation_config, "oos_fitness_weight", 0.0) or 0.0),
        )
        return f"{base_key}:robustness:{hash(signature)}"

    def _get_cached_data(
        self, backtest_config: Dict[str, SerializableValue]
    ) -> pd.DataFrame | None:
        """キャッシュされたバックテストデータを取得"""
        return self._data_provider.get_cached_backtest_data(backtest_config)

    def _get_cached_minute_data(
        self, backtest_config: Dict[str, SerializableValue]
    ) -> pd.DataFrame | None:
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
        start_date: object,
        end_date: object,
        cache_prefix: str = "ohlcv",
    ) -> pd.DataFrame | None:
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
            fidelity_backtest_config = adjust_backtest_config_for_fidelity(
                backtest_config,
                config,
            )
            prepared_backtest_config = self._prepare_backtest_config_for_evaluation(
                gene, fidelity_backtest_config
            )

            # 1. 実行用設定の構築
            run_config = self._prepare_run_config(
                gene, prepared_backtest_config, config
            )
            if not run_config:
                return ScenarioEvaluation(  # type: ignore[call-arg]
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
            self._inject_external_objects(
                run_config,
                prepared_backtest_config,
                config,
            )

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
                gene,
                fidelity_backtest_config,
                config,
            )

            # 5. フィットネス計算（フィットネス計算機に委譲）
            fitness = self._calculate_multi_objective_fitness(
                result, config, **evaluation_context
            )
            performance_metrics = self._extract_performance_metrics(result)
            scenario_metadata = metadata.copy() if metadata else {}
            scenario_metadata.update(
                {
                    "start_date": str(fidelity_backtest_config.get("start_date")),
                    "end_date": str(fidelity_backtest_config.get("end_date")),
                }
            )
            if prepared_backtest_config.get("_evaluation_start") is not None:
                scenario_metadata["evaluation_start"] = str(
                    prepared_backtest_config["_evaluation_start"]
                )

            return ScenarioEvaluation(  # type: ignore[call-arg]
                name=scenario_name,
                fitness=tuple(float(value) for value in fitness),
                passed=self._is_backtest_result_passing(result, config),
                metadata=scenario_metadata,
                performance_metrics=performance_metrics,
            )

        except BacktestEarlyTerminationError as e:
            logger.info("単一評価を早期終了しました: %s", e)
            scenario_metadata = metadata.copy() if metadata else {}
            scenario_metadata["early_terminated"] = True
            scenario_metadata["termination_reason"] = getattr(e, "reason", str(e))
            return ScenarioEvaluation(  # type: ignore[call-arg]
                name=scenario_name,
                fitness=self._fitness_calculator.get_penalty_values(config),
                passed=False,
                metadata=scenario_metadata,
                performance_metrics={"early_terminated": True},
            )
        except Exception as e:
            logger.error(f"単一評価実行エラー: {e}")
            scenario_metadata = metadata.copy() if metadata else {}
            scenario_metadata["error"] = str(e)
            return ScenarioEvaluation(  # type: ignore[call-arg]
                name=scenario_name,
                fitness=tuple(0.0 for _ in config.objectives),
                passed=False,
                metadata=scenario_metadata,
            )

    def _is_backtest_result_passing(
        self, backtest_result: Dict[str, Any], config: GAConfig
    ) -> bool:
        """バックテスト結果が基本制約を満たすか判定する。"""
        try:
            metrics = self._extract_performance_metrics(backtest_result)
            return self._fitness_calculator.meets_constraints(metrics, config)
        except Exception:
            return False

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
        """評価計算に必要な追加コンテキストを取得（サブクラスでオーバーライド）"""
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
        return self._evaluation_strategy._evaluate_with_oos_report(
            gene, base_backtest_config, config, oos_ratio, oos_weight
        ).aggregated_fitness

    def _evaluate_with_walk_forward(
        self,
        gene,
        base_backtest_config: Dict[str, Any],
        config: GAConfig,
    ) -> Tuple[float, ...]:
        """Walk-Forward Analysis による評価（EvaluationStrategyに委譲）"""
        return self._evaluation_strategy._evaluate_with_walk_forward_report(
            gene, base_backtest_config, config
        ).aggregated_fitness

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


__all__ = [
    "IndividualEvaluator",
    "calculate_trade_frequency_penalty",
    "calculate_ulcer_index",
]
