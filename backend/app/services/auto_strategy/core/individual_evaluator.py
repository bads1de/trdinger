"""
個体評価器

遺伝的アルゴリズムの個体評価を担当します。
"""

import functools
import logging
import math
import threading
from datetime import datetime
from typing import (
    Any,
    Dict,
    Iterable,
    Mapping,
    Optional,
    Sequence,
    Tuple,
)

import numpy as np
import pandas as pd
from numba import njit

from cachetools import LRUCache
from pydantic import ValidationError

from app.services.backtest.backtest_config import BacktestConfig
from app.services.backtest.backtest_service import BacktestService
from app.services.ml.models.model_manager import model_manager

from ..config import GAConfig
from ..genes import StrategyGene
from ..serializers.serialization import GeneSerializer

logger = logging.getLogger(__name__)


# 基準となる1日あたりの取引回数（これを超えると過剰取引とみなすペナルティが増加）
REFERENCE_TRADES_PER_DAY = 8.0


@functools.lru_cache(maxsize=1024)
def _ensure_datetime(value: Optional[object]) -> Optional[datetime]:
    """値をdatetimeオブジェクトに変換します（キャッシュ付き）。"""
    if isinstance(value, datetime):
        return value
    if isinstance(value, str):
        try:
            return datetime.fromisoformat(value.replace("Z", "+00:00"))
        except ValueError:
            return None
    return None


@njit(cache=True)
def _calculate_ulcer_index_numba(dd_array: np.ndarray) -> float:
    """
    NumbaでUlcer Indexの数値計算を高速化
    """
    if len(dd_array) == 0:
        return 0.0

    # 二乗和を計算
    squared_sum = 0.0
    count = 0
    for i in range(len(dd_array)):
        val = dd_array[i]
        if np.isnan(val):
            continue

        # 絶対値化
        val = abs(val)

        # 1.0より大きい場合はパーセンテージ(0-100)とみなして小数(0-1.0)に正規化
        if val > 1.0:
            val /= 100.0

        squared_sum += val * val
        count += 1

    if count == 0:
        return 0.0

    return np.sqrt(squared_sum / count)


def calculate_ulcer_index(equity_curve: Sequence[Mapping[str, Any]]) -> float:
    """
    Ulcer Index（ドローダウンの二乗平均平方根）を計算します。
    標準偏差よりも下落リスクを重視するリスク指標です。

    Args:
        equity_curve: バックテスト結果の資産曲線のポイント列。

    Returns:
        ドローダウン率の二乗平均平方根（小数値）。
    """
    if not equity_curve:
        return 0.0

    try:
        # ドローダウン値の抽出
        # 大規模データの場合、リスト内包表記 + np.array が一般的
        # カラムが多い場合は pd.DataFrame(equity_curve)['drawdown'] も速いが、メモリ消費を考慮
        dd_array = np.array(
            [
                float(p.get("drawdown", 0.0) or 0.0) if isinstance(p, Mapping) else 0.0
                for p in equity_curve
            ],
            dtype=np.float64,
        )

        # Numbaで高速計算
        return _calculate_ulcer_index_numba(dd_array)

    except Exception as e:
        logger.warning(f"Ulcer Index計算エラー: {e}")
        return 0.0


def calculate_trade_frequency_penalty(
    *,
    total_trades: int,
    start_date: Optional[object],
    end_date: Optional[object],
    trade_history: Optional[Iterable[Mapping[str, Any]]] = None,
) -> float:
    """
    過剰取引（高頻度取引）に対する正規化されたペナルティを返します。

    1日あたりの平均取引回数が増えるにつれてペナルティは増加し、
    双曲線正接（tanh）により ``[0, 1)`` の範囲に収まります。

    Args:
        total_trades: 総取引回数。
        start_date: 開始日時。
        end_date: 終了日時。
        trade_history: 取引履歴（total_tradesが0の場合にカウントに使用）。

    Returns:
        ペナルティ値（0.0〜1.0未満）。
    """

    trades = int(total_trades or 0)
    if trades <= 0 and trade_history is not None:
        trades = sum(1 for _ in trade_history)

    if trades <= 0:
        return 0.0

    parsed_start = _ensure_datetime(start_date)
    parsed_end = _ensure_datetime(end_date)

    if parsed_start is None or parsed_end is None or parsed_end <= parsed_start:
        # 期間が不明または無効な場合は1日として扱う
        duration_days = 1.0
    else:
        # 最低1時間（1/24日）として計算
        duration_days = max(
            (parsed_end - parsed_start).total_seconds() / 86_400.0,
            1.0 / 24.0,
        )

    trades_per_day = trades / duration_days

    return math.tanh(trades_per_day / REFERENCE_TRADES_PER_DAY)


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
            BacktestConfig(**temp_config)
        except ValidationError as e:
            logger.warning(f"バックテスト設定の初期バリデーション警告: {e}")
            # ここでは警告に留める

        self._fixed_backtest_config = self._select_timeframe_config(backtest_config)

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
        """Pickle化時の状態取得（ロックとキャッシュを除外）"""
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

        return state

    def __setstate__(self, state):
        """Pickle復元時の状態設定（ロックとキャッシュを再生成）"""
        self.__dict__.update(state)
        self._lock = threading.Lock()

        # キャッシュの再初期化（空の状態）
        if not hasattr(self, "_data_cache"):
            self._data_cache = LRUCache(maxsize=self._max_cache_size)
        if not hasattr(self, "_result_cache"):
            self._result_cache = LRUCache(maxsize=self._max_cache_size * 100)

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
                # 辞書形式から復元
                gene_serializer = GeneSerializer()
                gene = gene_serializer.dict_to_strategy_gene(individual, StrategyGene)
            elif isinstance(individual, list):
                # リスト形式の互換性维持（要素の最初をStrategyGeneとして扱う）
                # DEAP形式: [StrategyGene, ...] もしくは [float, ...] などの場合
                if len(individual) > 0 and isinstance(individual[0], StrategyGene):
                    gene = individual[0]
                elif len(individual) > 0 and hasattr(individual[0], "id"):
                    # StrategyGeneオブジェクトそのものと仮定
                    gene = individual[0]
                else:
                    # 互換性のため、StrategyGeneオブジェクトと仮定
                    gene = individual
            else:
                # その他はそのまま渡す（エラーハンドリングは下位で）
                gene = individual

            # キャッシュチェック
            # 遺伝子IDとバックテスト設定（期間など）をキーにする
            # config自体は複雑なので、ハッシュ化可能な要素のみ抽出するか、
            # gene.id が一意であればそれだけで十分だが、backtest_configが変われば結果も変わる。
            # 今回は _fixed_backtest_config は set_backtest_config で固定され、変更時にキャッシュクリアされるため
            # gene.id だけで十分かもしれないが、念のため安全策を取る。
            # geneにはハッシュメソッドがあるか不明だが、文字列表現はユニークと仮定。

            # 遺伝子の文字列表現（パラメータ全体を含む）を一意なキーとして使用
            # IDだけでは、パラメータが同じでIDが違う場合に対応できないが、
            # GAではパラメータが同じなら結果も同じはずなので、パラメータのハッシュが良い。
            # しかし実装が複雑になるため、ここでは gene.id を使用する（IDは生成時に付与される前提）。
            # IDがない場合はstr(gene)を使う。
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

            # 評価実行
            fitness = self._execute_evaluation_logic(gene, base_backtest_config, config)

            # 結果をキャッシュ
            with self._lock:
                self._result_cache[cache_key] = fitness

            return fitness

        except Exception as e:
            logger.error(f"個体評価エラー: {e}")
            # エラー発生時は目的プロトコルに合わせて0.0のタプルを返す
            return tuple(0.0 for _ in getattr(config, "objectives", []))

    def _execute_evaluation_logic(
        self, gene: Any, base_backtest_config: Dict[str, Any], config: GAConfig
    ) -> Tuple[float, ...]:
        """
        具体的な評価プロセス（OOS 分割や Walk-Forward 分割）を振り分け

        設定に応じて、通常のバックテスト、Out-of-Sample 検証（期間分割）、
        または Walk-Forward 分析のいずれかのロジックで評価を実行します。

        Args:
            gene: 評価対象の遺伝子
            base_backtest_config: 固定的なバックテスト設定
            config: GA 実行設定

        Returns:
            算出された適応度（タプル）
        """

        # Walk-Forward Analysis が有効な場合
        if getattr(config, "enable_walk_forward", False):
            return self._evaluate_with_walk_forward(gene, base_backtest_config, config)

        # OOS検証の有無を確認
        oos_ratio = getattr(config, "oos_split_ratio", 0.0)
        oos_weight = getattr(config, "oos_fitness_weight", 0.5)

        if oos_ratio > 0.0:
            # 期間分割とOOS評価
            return self._evaluate_with_oos(
                gene, base_backtest_config, config, oos_ratio, oos_weight
            )
        else:
            # 通常評価（全期間）
            return self._perform_single_evaluation(gene, base_backtest_config, config)

    def _evaluate_with_oos(
        self,
        gene,
        base_backtest_config: Dict[str, Any],
        config: GAConfig,
        oos_ratio: float,
        oos_weight: float,
    ) -> Tuple[float, ...]:
        """
        Out-of-Sample (OOS) 検証を含む評価を実行します。

        全体期間をトレーニング期間（In-Sample）とテスト期間（Out-of-Sample）に分割し、
        それぞれのフィットネス値を加重平均して最終的な評価値とします。

        Args:
            gene: 評価対象の戦略遺伝子
            base_backtest_config: ベースとなるバックテスト設定
            config: GA設定
            oos_ratio: OOS期間の割合 (0.0 - 1.0)
            oos_weight: OOSスコアの重み (0.0 - 1.0)

        Returns:
            加重平均されたフィットネス値のタプル
        """
        try:
            start_date = pd.to_datetime(base_backtest_config.get("start_date"))
            end_date = pd.to_datetime(base_backtest_config.get("end_date"))

            if start_date is None or end_date is None:
                # 期間が不明な場合は通常評価
                return self._perform_single_evaluation(
                    gene, base_backtest_config, config
                )

            total_duration = end_date - start_date
            train_duration = total_duration * (1.0 - oos_ratio)

            split_date = start_date + train_duration

            # 日付文字列に変換
            start_str = start_date.strftime("%Y-%m-%d %H:%M:%S")
            split_str = split_date.strftime("%Y-%m-%d %H:%M:%S")
            end_str = end_date.strftime("%Y-%m-%d %H:%M:%S")

            # In-Sample評価
            is_config = base_backtest_config.copy()
            is_config["start_date"] = start_str
            is_config["end_date"] = split_str
            is_fitness = self._perform_single_evaluation(gene, is_config, config)

            # Out-of-Sample評価
            oos_config = base_backtest_config.copy()
            oos_config["start_date"] = split_str
            oos_config["end_date"] = end_str
            oos_fitness = self._perform_single_evaluation(gene, oos_config, config)

            # フィットネス結合
            combined_fitness = []
            for f_is, f_oos in zip(is_fitness, oos_fitness):
                combined = f_is * (1.0 - oos_weight) + f_oos * oos_weight
                combined_fitness.append(max(0.0, combined))

            logger.info(
                f"OOS評価完了: IS={is_fitness}, OOS={oos_fitness}, Combined={combined_fitness}"
            )
            return tuple(combined_fitness)

        except Exception as e:
            logger.error(f"OOS評価中エラー: {e}")
            return self._perform_single_evaluation(gene, base_backtest_config, config)

    def _evaluate_with_walk_forward(
        self,
        gene,
        base_backtest_config: Dict[str, Any],
        config: GAConfig,
    ):
        """
        Walk-Forward Analysis による評価

        時系列をスライディングさせながら検証し、過学習を検出するための堅牢な評価手法。

        Args:
            gene: 評価する戦略遺伝子
            base_backtest_config: ベースとなるバックテスト設定
            config: GA設定

        Returns:
            フィットネス値のタプル（OOSスコアの平均）
        """
        try:
            start_date = pd.to_datetime(base_backtest_config.get("start_date"))
            end_date = pd.to_datetime(base_backtest_config.get("end_date"))

            if start_date is None or end_date is None:
                logger.warning("WFA: 期間が不明なため通常評価にフォールバック")
                return self._perform_single_evaluation(
                    gene, base_backtest_config, config
                )

            # WFA パラメータ取得
            n_folds = getattr(config, "wfa_n_folds", 5)
            train_ratio = getattr(config, "wfa_train_ratio", 0.7)
            anchored = getattr(config, "wfa_anchored", False)

            total_duration = end_date - start_date
            fold_duration = total_duration / n_folds

            oos_fitness_values = []  # 各フォールドのOOSスコアを保存

            for fold_idx in range(n_folds):
                # フォールド期間の計算
                if anchored:
                    # Anchored WFA: トレーニング開始は常に最初から
                    fold_train_start = start_date
                else:
                    # Rolling WFA: トレーニングウィンドウがスライド
                    fold_train_start = start_date + (fold_duration * fold_idx)

                fold_end = start_date + (fold_duration * (fold_idx + 1))

                # フォールド内をトレーニングとテストに分割
                fold_period = fold_end - fold_train_start
                train_duration = fold_period * train_ratio

                train_end = fold_train_start + train_duration
                test_start = train_end
                test_end = fold_end

                # トレーニング期間が短すぎる場合はスキップ
                if (train_end - fold_train_start).days < 7:
                    logger.debug(
                        "WFA Fold %s: トレーニング期間が短すぎるためスキップ", fold_idx
                    )
                    continue

                # テスト期間が短すぎる場合はスキップ
                if (test_end - test_start).days < 1:
                    logger.debug(
                        "WFA Fold %s: テスト期間が短すぎるためスキップ", fold_idx
                    )
                    continue

                # 日付文字列に変換
                train_start_str = fold_train_start.strftime("%Y-%m-%d %H:%M:%S")
                train_end_str = train_end.strftime("%Y-%m-%d %H:%M:%S")
                test_start_str = test_start.strftime("%Y-%m-%d %H:%M:%S")
                test_end_str = test_end.strftime("%Y-%m-%d %H:%M:%S")

                logger.debug(
                    "WFA Fold %s: Train=%s to %s, Test=%s to %s",
                    fold_idx,
                    train_start_str,
                    train_end_str,
                    test_start_str,
                    test_end_str,
                )

                # テスト期間で評価（WFAではOOSスコアのみを使用）
                test_config = base_backtest_config.copy()
                test_config["start_date"] = test_start_str
                test_config["end_date"] = test_end_str

                try:
                    oos_fitness = self._perform_single_evaluation(
                        gene, test_config, config
                    )
                    oos_fitness_values.append(oos_fitness)
                except Exception as fold_error:
                    logger.warning(f"WFA Fold {fold_idx} 評価エラー: {fold_error}")
                    continue

            if not oos_fitness_values:
                logger.warning(
                    "WFA: 有効なフォールドがないため通常評価にフォールバック"
                )
                return self._perform_single_evaluation(
                    gene, base_backtest_config, config
                )

            # 全フォールドのOOSスコアの平均を計算
            num_objectives = len(oos_fitness_values[0])
            averaged_fitness = []

            for obj_idx in range(num_objectives):
                obj_values = [f[obj_idx] for f in oos_fitness_values]
                avg_value = sum(obj_values) / len(obj_values)
                averaged_fitness.append(max(0.0, avg_value))

            logger.info(
                f"WFA評価完了: {len(oos_fitness_values)}フォールド, "
                f"平均OOS={tuple(round(v, 4) for v in averaged_fitness)}"
            )

            return tuple(averaged_fitness)

        except Exception as e:
            logger.error(f"WFA評価中エラー: {e}")
            return self._perform_single_evaluation(gene, base_backtest_config, config)

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
                # データサービスが初期化されていることを確認
                self.backtest_service.ensure_data_service_initialized()

                # データを取得
                data = self.backtest_service.data_service.get_data_for_backtest(
                    symbol=symbol,
                    timeframe=timeframe,
                    start_date=pd.to_datetime(start_date),
                    end_date=pd.to_datetime(end_date),
                )

                # 最適化: backtesting.py用に事前にカラム名を大文字化してキャッシュ
                # これによりBacktestExecutorでの毎回コピーとリネームを回避
                if not data.empty and "Open" not in data.columns:
                    # dataはここでしか使われないので直接書き換えても良いが
                    # 安全のためコピーしておく（get_data_for_backtestの実装次第では参照返しもありうる）
                    data = data.copy()
                    data.columns = data.columns.str.capitalize()

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
        # 並列ワーカー内の共有データをチェック
        try:
            from .parallel_evaluator import get_worker_data

            worker_data = get_worker_data("minute_data")
            # 1分足データは存在しない場合もある（None）ため、キーが存在するか確認するロジックが必要だが
            # get_worker_dataは存在しない場合にNoneを返す仕様。
            # "minute_data"キーが明示的にセットされていて中身がNoneなのか、セットされていないのか区別がつかない。
            # しかし、GAEngine側でminute_dataを使わない場合はセットしないので、Noneならキャッシュを見に行くフローで良い。
            if worker_data is not None:
                return worker_data
        except ImportError:
            pass

        symbol = backtest_config.get("symbol")
        start_date = backtest_config.get("start_date")
        end_date = backtest_config.get("end_date")

        # キーの作成（"minute_"プレフィックスで通常データと区別）
        key = ("minute", symbol, "1m", str(start_date), str(end_date))

        with self._lock:
            if key not in self._data_cache:
                try:
                    # データサービスが初期化されていることを確認
                    self.backtest_service.ensure_data_service_initialized()

                    # 1分足データを取得
                    data = self.backtest_service.data_service.get_data_for_backtest(
                        symbol=symbol,
                        timeframe="1m",
                        start_date=pd.to_datetime(start_date),
                        end_date=pd.to_datetime(end_date),
                    )
                    if not data.empty:
                        # 最適化: カラム名を大文字化してキャッシュ
                        if "Open" not in data.columns:
                            data = data.copy()
                            data.columns = data.columns.str.capitalize()

                        self._data_cache[key] = data
                        logger.debug(f"1分足データをキャッシュしました: {key}")
                    else:
                        logger.debug(f"1分足データが空です: {key}")
                        return None
                except Exception as e:
                    logger.warning(f"1分足データ取得エラー: {e}")
                    return None

            return self._data_cache.get(key)

    def _perform_single_evaluation(
        self, gene, backtest_config: Dict[str, Any], config: GAConfig
    ) -> Tuple[float, ...]:
        """
        単一期間でのバックテストとフィットネス評価を実行

        バックテスト設定の構築、データの準備、外部オブジェクトの注入を行い、
        バックテストエンジンを呼び出した後、多目的フィットネスを計算します。

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

            # 5. フィットネス計算
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
            # 高速化: シリアライザーとPydanticバリデーションをスキップ
            # geneオブジェクトをそのまま渡すことでシリアライズコストを削減

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
            # gene.id があれば使う
            gene_id = getattr(gene, "id", "unknown")[:8]
            config_dict["strategy_name"] = f"GA_Individual_{gene_id}"

            # 高速化フラグ: BacktestOrchestratorでのバリデーションをスキップ
            config_dict["_skip_validation"] = True

            # 辞書をそのまま返す（バリデーションは初期設定時に済ませている前提）
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
        """実行設定への外部オブジェクト注入（1分足データ、MLモデルなど）"""
        # 1分足データを取得（1分足シミュレーション用）
        minute_data = self._get_cached_minute_data(backtest_config)
        if minute_data is not None:
            run_config["strategy_config"]["parameters"]["minute_data"] = minute_data

        # MLフィルター設定
        if config.ml_filter_enabled and config.ml_model_path:
            try:
                ml_filter_model = model_manager.load_model(config.ml_model_path)
                if ml_filter_model:
                    run_config["strategy_config"]["parameters"][
                        "ml_predictor"
                    ] = ml_filter_model
                    run_config["strategy_config"]["parameters"][
                        "ml_filter_threshold"
                    ] = 0.5
            except Exception:
                # ロード失敗時は無効化
                run_config["strategy_config"]["parameters"]["ml_filter_enabled"] = False
        elif config.ml_filter_enabled:
            # パス指定なしなどの場合
            run_config["strategy_config"]["parameters"]["ml_filter_enabled"] = False

    def _get_evaluation_context(
        self, gene, backtest_config: Dict[str, Any], config: GAConfig
    ) -> Dict[str, Any]:
        """評価計算に必要な追加コンテキストを取得（サブクラスでオーバーライド）"""
        return {}

    def _extract_performance_metrics(
        self, backtest_result: Dict[str, Any]
    ) -> Dict[str, Any]:
        """
        バックテスト結果からパフォーマンスメトリクスを抽出

        Args:
            backtest_result: バックテスト結果

        Returns:
            抽出されたパフォーマンスメトリクス
        """
        performance_metrics = backtest_result.get("performance_metrics", {})

        # 主要メトリクスを安全に抽出（デフォルト値を設定）
        metrics = {
            "total_return": performance_metrics.get("total_return", 0.0),
            "sharpe_ratio": performance_metrics.get("sharpe_ratio", 0.0),
            "max_drawdown": performance_metrics.get("max_drawdown", 1.0),
            "win_rate": performance_metrics.get("win_rate", 0.0),
            "profit_factor": performance_metrics.get("profit_factor", 0.0),
            "sortino_ratio": performance_metrics.get("sortino_ratio", 0.0),
            "calmar_ratio": performance_metrics.get("calmar_ratio", 0.0),
            "total_trades": performance_metrics.get("total_trades", 0),
        }

        # 無効な値を処理（None, inf, nanなど）
        import math

        for key, value in metrics.items():

            def is_invalid_value(val):
                return (
                    val is None
                    or (isinstance(val, float) and not math.isfinite(val))
                    or not isinstance(val, (int, float))
                )

            if is_invalid_value(value):
                if key == "max_drawdown":
                    metrics[key] = 1.0  # 最大ドローダウンは1.0（100%）が上限
                elif key == "total_trades":
                    metrics[key] = 0
                else:
                    metrics[key] = 0.0
            elif (
                key == "max_drawdown" and isinstance(value, (int, float)) and value < 0
            ):
                metrics[key] = 0.0  # 負のドローダウンは0に修正

        equity_curve = backtest_result.get("equity_curve", [])
        metrics["ulcer_index"] = calculate_ulcer_index(equity_curve)

        trade_history = backtest_result.get("trade_history", [])
        metrics["trade_frequency_penalty"] = calculate_trade_frequency_penalty(
            total_trades=metrics["total_trades"],
            start_date=backtest_result.get("start_date"),
            end_date=backtest_result.get("end_date"),
            trade_history=trade_history,
        )

        return metrics

    def _calculate_fitness(
        self, backtest_result: Dict[str, Any], config: GAConfig, **kwargs
    ) -> float:
        """
        フィットネス計算（ロング・ショートバランス評価を含む）

        Args:
            backtest_result: バックテスト結果
            config: GA設定

        Returns:
            フィットネス値
        """
        try:
            # パフォーマンスメトリクスを抽出
            metrics = self._extract_performance_metrics(backtest_result)

            total_return = metrics["total_return"]
            sharpe_ratio = metrics["sharpe_ratio"]
            max_drawdown = metrics["max_drawdown"]
            win_rate = metrics["win_rate"]
            total_trades = metrics["total_trades"]
            ulcer_index = metrics.get("ulcer_index", 0.0)
            trade_penalty = metrics.get("trade_frequency_penalty", 0.0)

            # 取引回数が0の場合は低いフィットネス値を返す
            if total_trades == 0:
                logger.warning("取引回数が0のため、低いフィットネス値を設定")
                return config.zero_trades_penalty

            # 追加の制約チェック
            min_trades_req = int(config.fitness_constraints.get("min_trades", 0))
            if total_trades < min_trades_req:
                return config.constraint_violation_penalty

            max_dd_limit = config.fitness_constraints.get("max_drawdown_limit", None)
            if isinstance(max_dd_limit, (float, int)) and max_drawdown > float(
                max_dd_limit
            ):
                return config.constraint_violation_penalty

            if total_return < 0 or sharpe_ratio < config.fitness_constraints.get(
                "min_sharpe_ratio", 0
            ):
                return config.constraint_violation_penalty

            # ロング・ショートバランス評価を計算
            balance_score = self._calculate_long_short_balance(backtest_result)

            fitness_weights = config.fitness_weights.copy()

            # 重み付きフィットネス計算（バランススコアを追加）
            fitness = (
                fitness_weights.get("total_return", 0.3) * total_return
                + fitness_weights.get("sharpe_ratio", 0.4) * sharpe_ratio
                + fitness_weights.get("max_drawdown", 0.2) * (1 - max_drawdown)
                + fitness_weights.get("win_rate", 0.1) * win_rate
                + fitness_weights.get("balance_score", 0.1) * balance_score
            )

            ulcer_scale = 1.0
            trade_scale = 1.0
            if getattr(config, "dynamic_objective_reweighting", False):
                dynamic_scalars = getattr(config, "objective_dynamic_scalars", {})
                ulcer_scale = dynamic_scalars.get("ulcer_index", 1.0)
                trade_scale = dynamic_scalars.get("trade_frequency_penalty", 1.0)

            fitness -= (
                fitness_weights.get("ulcer_index_penalty", 0.0)
                * ulcer_scale
                * ulcer_index
            )
            fitness -= (
                fitness_weights.get("trade_frequency_penalty", 0.0)
                * trade_scale
                * trade_penalty
            )

            return max(0.0, fitness)

        except Exception as e:
            logger.error(f"フィットネス計算エラー: {e}")
            return config.constraint_violation_penalty

    def _calculate_long_short_balance(self, backtest_result: Dict[str, Any]) -> float:
        """
        ロング・ショートバランススコアを計算

        Args:
            backtest_result: バックテスト結果

        Returns:
            バランススコア（0.0-1.0）
        """
        try:
            trade_history = backtest_result.get("trade_history", [])
            if not trade_history:
                return 0.5  # 取引がない場合は中立スコア

            long_trades = []
            short_trades = []
            long_pnl = 0.0
            short_pnl = 0.0

            # 取引をロング・ショートに分類
            for trade in trade_history:
                size = trade.get("size", 0.0)
                pnl = trade.get("pnl", 0.0)

                if size > 0:  # ロング取引
                    long_trades.append(trade)
                    long_pnl += pnl
                elif size < 0:  # ショート取引
                    short_trades.append(trade)
                    short_pnl += pnl

            total_trades = len(long_trades) + len(short_trades)
            if total_trades == 0:
                return 0.5

            # 取引回数バランス（理想は50:50）
            long_ratio = len(long_trades) / total_trades
            short_ratio = len(short_trades) / total_trades
            trade_balance = 1.0 - abs(long_ratio - short_ratio)

            # 利益バランス（両方向で利益を出せているか）
            total_pnl = long_pnl + short_pnl
            profit_balance = 0.5  # デフォルト

            if total_pnl > 0:
                # 両方向で利益が出ている場合は高スコア
                if long_pnl > 0 and short_pnl > 0:
                    profit_balance = 1.0
                # 片方向のみで利益の場合は中程度
                elif long_pnl > 0 or short_pnl > 0:
                    profit_balance = 0.7
            # 両方で損失が出ている場合は低いスコア
            elif long_pnl < 0 and short_pnl < 0:
                profit_balance = 0.1
            else:
                profit_balance = 0.3

            # 総合バランススコア（取引回数バランス60%、利益バランス40%）
            balance_score = 0.6 * trade_balance + 0.4 * profit_balance

            return max(0.0, min(1.0, balance_score))

        except Exception as e:
            logger.error(f"ロング・ショートバランス計算エラー: {e}")
            return 0.5  # エラー時は中立スコア

    def _select_timeframe_config(
        self, backtest_config: Dict[str, Any]
    ) -> Dict[str, Any]:
        """
        タイムフレーム設定の選択

        Args:
            backtest_config: バックテスト設定

        Returns:
            選択されたタイムフレーム設定
        """
        if not backtest_config:
            return {}

        # 簡単な実装: 設定をそのまま返す
        return backtest_config.copy()

    def _calculate_multi_objective_fitness(
        self, backtest_result: Dict[str, Any], config: GAConfig, **kwargs
    ) -> Tuple[float, ...]:
        """
        多目的適応度の計算

        バックテスト結果からパフォーマンスメトリクスを抽出し、
        GA設定で定義された各目的関数（利益、リスク、取引回数など）
        に対応する値を算出してタプルとして返します。

        Args:
            backtest_result: バックテスト実行結果
            config: GA設定
            **kwargs: 追加の評価コンテキスト

        Returns:
            各目的関数の評価値を含むタプル
        """
        try:
            # パフォーマンスメトリクスを抽出
            metrics = self._extract_performance_metrics(backtest_result)
            total_trades = metrics["total_trades"]

            # 取引回数制約チェック
            min_trades_req = int(config.fitness_constraints.get("min_trades", 0))
            if total_trades < min_trades_req:
                penalty_values = []
                for obj in config.objectives:
                    # 最小化したい指標（悪いほど値が大きい）には最大ペナルティを設定
                    if obj in [
                        "max_drawdown",
                        "ulcer_index",
                        "trade_frequency_penalty",
                    ]:
                        penalty_values.append(1.0)
                    else:
                        penalty_values.append(config.constraint_violation_penalty)
                return tuple(penalty_values)

            fitness_values = []

            for objective in config.objectives:
                if objective == "weighted_score":
                    # 従来の重み付けスコア計算を利用
                    value = self._calculate_fitness(backtest_result, config, **kwargs)
                elif objective == "total_return":
                    value = metrics["total_return"]
                elif objective == "sharpe_ratio":
                    value = metrics["sharpe_ratio"]
                elif objective == "max_drawdown":
                    # ドローダウンは最小化したいので、DEAP側で-1.0の重みが設定される
                    value = metrics["max_drawdown"]
                elif objective == "win_rate":
                    value = metrics["win_rate"]
                elif objective == "profit_factor":
                    value = metrics["profit_factor"]
                elif objective == "sortino_ratio":
                    value = metrics["sortino_ratio"]
                elif objective == "calmar_ratio":
                    value = metrics["calmar_ratio"]
                elif objective == "balance_score":
                    value = self._calculate_long_short_balance(backtest_result)
                elif objective == "ulcer_index":
                    value = metrics.get("ulcer_index", 0.0)
                elif objective == "trade_frequency_penalty":
                    value = metrics.get("trade_frequency_penalty", 0.0)
                else:
                    logger.warning(f"未知の目的: {objective}")
                    value = 0.0

                dynamic_scalars = getattr(config, "objective_dynamic_scalars", {})
                scale = dynamic_scalars.get(objective, 1.0)
                fitness_values.append(float(value) * scale)

            return tuple(fitness_values)

        except Exception as e:
            logger.error(f"多目的フィットネス計算エラー: {e}")
            # エラー時は目的数に応じたデフォルト値を返す
            return tuple(0.0 for _ in config.objectives)
