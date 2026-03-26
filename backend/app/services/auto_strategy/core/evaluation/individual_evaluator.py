"""
個体評価器

遺伝的アルゴリズムの個体評価を担当します。
"""

import logging
import threading
from typing import (
    Any,
    Dict,
    Optional,
    Tuple,
)

from cachetools import LRUCache
from pydantic import ValidationError

from app.services.backtest.config.backtest_config import BacktestRunConfig
from app.services.backtest.services.backtest_service import BacktestService

from app.services.auto_strategy.config import GAConfig
from app.services.auto_strategy.genes import StrategyGene
from app.services.auto_strategy.serializers.serialization import GeneSerializer


from ..fitness.fitness_calculator import FitnessCalculator
from .evaluation_strategies import EvaluationStrategy

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
        self._fixed_backtest_config = None
        self._max_cache_size = max_cache_size or self.DEFAULT_MAX_CACHE_SIZE
        # データキャッシュ（重いデータ用）
        self._data_cache: LRUCache = LRUCache(maxsize=self._max_cache_size)
        # 評価結果キャッシュ（軽量、ヒット率向上用）
        # データより軽量なのでサイズを大きめに取る
        self._result_cache: LRUCache = LRUCache(maxsize=self._max_cache_size * 100)
        self._lock = threading.Lock()

        # 統計情報
        self._cache_hits = 0
        self._cache_misses = 0

        # 委譲先コンポーネント
        self._fitness_calculator = FitnessCalculator()
        self._evaluation_strategy = EvaluationStrategy(self)

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

    def clear_cache(self) -> None:
        """データキャッシュをクリア"""
        with self._lock:
            self._data_cache.clear()
            self._result_cache.clear()
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
                "cache_hits": self._cache_hits,
                "cache_misses": self._cache_misses,
            }

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

        # 委譲先コンポーネントは自己参照を含むため除外（復元時に再生成）
        if "_fitness_calculator" in state:
            del state["_fitness_calculator"]
        if "_evaluation_strategy" in state:
            del state["_evaluation_strategy"]

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

        # 委譲先コンポーネントを再生成
        self._fitness_calculator = FitnessCalculator()
        self._evaluation_strategy = EvaluationStrategy(self)

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
            # 遺伝子デコード
            if isinstance(individual, StrategyGene):
                gene = individual
            elif isinstance(individual, dict):
                gene_serializer = GeneSerializer()
                gene = gene_serializer.dict_to_strategy_gene(individual, StrategyGene)
            elif isinstance(individual, list):
                if len(individual) > 0 and isinstance(individual[0], StrategyGene):
                    gene = individual[0]
                elif len(individual) > 0 and hasattr(individual[0], "id"):
                    gene = individual[0]
                else:
                    gene = individual
            else:
                gene = individual

            # 遺伝子の文字列表現（パラメータ全体を含む）を一意なキーとして使用
            cache_key = getattr(gene, "id", str(gene))

            with self._lock:
                if cache_key in self._result_cache:
                    self._cache_hits += 1
                    return self._result_cache[cache_key]
                self._cache_misses += 1

            # バックテスト設定のベースを取得
            base_backtest_config = (
                self._fixed_backtest_config.copy()
                if self._fixed_backtest_config
                else {}
            )

            # 評価実行（戦略ルーティングに委譲）
            fitness = self._evaluation_strategy.execute(
                gene, base_backtest_config, config
            )

            # 結果をキャッシュ
            with self._lock:
                self._result_cache[cache_key] = fitness

            return fitness

        except Exception as e:
            logger.error(f"個体評価エラー: {e}")
            return tuple(0.0 for _ in getattr(config, "objectives", []))

    def _get_cached_data(self, backtest_config: Dict[str, Any]) -> Any:
        """キャッシュされたバックテストデータを取得"""
        # 並列ワーカー内の共有データをチェック
        try:
            from .parallel_evaluator import get_worker_data

            worker_data = get_worker_data("main_data")
            if worker_data is not None:
                return worker_data
        except ImportError:
            pass

        symbol = backtest_config.get("symbol")
        timeframe = backtest_config.get("timeframe")
        start_date = backtest_config.get("start_date")
        end_date = backtest_config.get("end_date")

        # キーの作成（文字列化して一意性を確保）
        key = (symbol, timeframe, str(start_date), str(end_date))

        with self._lock:
            if key not in self._data_cache:
                import pandas as pd

                self.backtest_service.ensure_data_service_initialized()

                data = self.backtest_service.data_service.get_data_for_backtest(
                    symbol=symbol,
                    timeframe=timeframe,
                    start_date=pd.to_datetime(start_date),
                    end_date=pd.to_datetime(end_date),
                )

                self._data_cache[key] = data
                logger.debug(f"バックテストデータをキャッシュしました: {key}")

            return self._data_cache[key]

    def _get_cached_minute_data(self, backtest_config: Dict[str, Any]) -> Any:
        """
        1分足データをキャッシュから取得

        1分足シミュレーション用に使用します。

        Args:
            backtest_config: バックテスト設定

        Returns:
            1分足のDataFrame、またはデータが存在しない場合はNone
        """
        try:
            from .parallel_evaluator import get_worker_data

            worker_data = get_worker_data("minute_data")
            if worker_data is not None:
                return worker_data
        except ImportError:
            pass

        symbol = backtest_config.get("symbol")
        start_date = backtest_config.get("start_date")
        end_date = backtest_config.get("end_date")

        key = ("minute", symbol, "1m", str(start_date), str(end_date))

        with self._lock:
            if key not in self._data_cache:
                try:
                    import pandas as pd

                    self.backtest_service.ensure_data_service_initialized()

                    data = self.backtest_service.data_service.get_data_for_backtest(
                        symbol=symbol,
                        timeframe="1m",
                        start_date=pd.to_datetime(start_date),
                        end_date=pd.to_datetime(end_date),
                    )
                    if not data.empty:
                        self._data_cache[key] = data
                        logger.debug(f"1分足データをキャッシュしました: {key}")
                    else:
                        logger.debug(f"1分足データが空です: {key}")
                        return None
                except Exception as e:
                    logger.warning(f"1分足データ取得エラー: {e}")
                    return None

            return self._data_cache.get(key)

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
        import pandas as pd

        if not all([symbol, timeframe, start_date, end_date]):
            logger.warning(
                "OHLCVデータ取得: 必須パラメータが不足しています "
                f"(symbol={symbol}, timeframe={timeframe}, "
                f"start_date={start_date}, end_date={end_date})"
            )
            return None

        cache_key = (cache_prefix, symbol, timeframe, str(start_date), str(end_date))

        with self._lock:
            if cache_key in self._data_cache:
                cached_data = self._data_cache[cache_key]
                if hasattr(cached_data, "empty") and not cached_data.empty:
                    logger.debug(f"OHLCVデータ: キャッシュヒット (key={cache_key})")
                    return cached_data

        data_service = getattr(self.backtest_service, "data_service", None)
        if data_service is None:
            logger.warning("data_service が利用できません。")
            return None

        try:
            if hasattr(self.backtest_service, "ensure_data_service_initialized"):
                self.backtest_service.ensure_data_service_initialized()

            ohlcv_data = data_service.get_ohlcv_data(
                symbol, timeframe, start_date, end_date
            )

            if isinstance(ohlcv_data, pd.DataFrame) and not ohlcv_data.empty:
                with self._lock:
                    self._data_cache[cache_key] = ohlcv_data
                logger.debug(
                    f"OHLCVデータ: DB から取得・キャッシュ保存 (key={cache_key})"
                )
                return ohlcv_data
            else:
                logger.warning(
                    f"OHLCVデータが空または無効です: symbol={symbol}, "
                    f"timeframe={timeframe}"
                )
                return None

        except Exception as exc:
            logger.warning(f"OHLCVデータ取得エラー: {exc}")
            return None

    def _perform_single_evaluation(
        self, gene, backtest_config: Dict[str, Any], config: GAConfig
    ) -> Tuple[float, ...]:
        """
        単一期間でのバックテストとフィットネス評価を実行

        Args:
            gene: 評価対象の戦略遺伝子
            backtest_config: バックテスト実行用設定
            config: GA設定

        Returns:
            フィットネス値のタプル
        """
        try:
            # 1. 実行用設定の構築
            run_config = self._prepare_run_config(gene, backtest_config, config)
            if not run_config:
                return tuple(0.0 for _ in config.objectives)

            # 2. データの準備
            data = self._get_cached_data(backtest_config)

            # モデル外のオブジェクトを注入
            self._inject_external_objects(run_config, backtest_config, config)

            # 3. バックテスト実行
            result = self.backtest_service.run_backtest(
                config=run_config, preloaded_data=data
            )

            # 4. 追加のコンテキスト情報（ML予測シグナルなど）を取得
            evaluation_context = self._get_evaluation_context(
                gene, backtest_config, config
            )

            # 5. フィットネス計算（フィットネス計算機に委譲）
            return self._calculate_multi_objective_fitness(
                result, config, **evaluation_context
            )

        except Exception as e:
            logger.error(f"単一評価実行エラー: {e}")
            return tuple(0.0 for _ in config.objectives)

    def _prepare_run_config(
        self, gene, backtest_config: Dict[str, Any], config: GAConfig
    ) -> Optional[Dict[str, Any]]:
        """バックテスト実行用設定の構築（高速化版）"""
        try:
            config_dict = backtest_config.copy()

            strategy_parameters = {
                "strategy_gene": gene,
                "ml_filter_enabled": config.ml_filter_enabled,
                "ml_model_path": config.ml_model_path,
            }

            config_dict["strategy_config"] = {
                "strategy_type": "GENERATED_GA",
                "parameters": strategy_parameters,
            }
            gene_id = getattr(gene, "id", "unknown")[:8]
            config_dict["strategy_name"] = f"GA_Individual_{gene_id}"

            # 高速化フラグ: BacktestOrchestratorでのバリデーションをスキップ
            config_dict["_skip_validation"] = True

            return config_dict
        except Exception as e:
            logger.error(f"バックテスト設定生成エラー: {e}")
            return None

    def _inject_external_objects(
        self,
        run_config: Dict[str, Any],
        backtest_config: Dict[str, Any],
        config: GAConfig,
    ) -> None:
        """実行設定への外部オブジェクト注入（1分足データ）"""
        # 1分足データを取得（1分足シミュレーション用）
        minute_data = self._get_cached_minute_data(backtest_config)
        if minute_data is not None:
            run_config["strategy_config"]["parameters"]["minute_data"] = minute_data

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
