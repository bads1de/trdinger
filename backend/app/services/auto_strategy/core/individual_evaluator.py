"""
個体評価器

遺伝的アルゴリズムの個体評価を担当します。
"""

import logging
import math
import threading
from datetime import datetime
from typing import Any, Dict, Iterable, Mapping, MutableSequence, Optional, Sequence, Tuple

import pandas as pd
from cachetools import LRUCache
from pydantic import ValidationError

from app.services.backtest.backtest_config import BacktestConfig
from app.services.backtest.backtest_service import BacktestService
from app.services.ml.model_manager import model_manager

from ..config import GAConfig

logger = logging.getLogger(__name__)


# 基準となる1日あたりの取引回数（これを超えると過剰取引とみなすペナルティが増加）
REFERENCE_TRADES_PER_DAY = 8.0


def _ensure_datetime(value: Optional[object]) -> Optional[datetime]:
    """値をdatetimeオブジェクトに変換します。"""
    if isinstance(value, datetime):
        return value
    if isinstance(value, str):
        try:
            return datetime.fromisoformat(value.replace("Z", "+00:00"))
        except ValueError:
            return None
    return None


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

    squared_drawdowns: MutableSequence[float] = []
    for point in equity_curve:
        raw_drawdown: Any = (
            point.get("drawdown") if isinstance(point, Mapping) else None
        )
        if raw_drawdown is None:
            continue
        try:
            drawdown = float(raw_drawdown)
        except (TypeError, ValueError):
            continue

        if math.isnan(drawdown):
            continue

        drawdown = abs(drawdown)
        # ドローダウンがパーセンテージ（>1.0）の場合、小数（0.0-1.0）に変換
        if drawdown > 1.0:
            drawdown /= 100.0

        squared_drawdowns.append(drawdown * drawdown)

    if not squared_drawdowns:
        return 0.0

    mean_square = sum(squared_drawdowns) / float(len(squared_drawdowns))
    return math.sqrt(mean_square)


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
    個体評価器

    遺伝的アルゴリズムの個体評価を担当します。
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
        if "strategy_config" not in temp_config:
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

    def evaluate_individual(self, individual, config: GAConfig):
        """
        個体評価（OOS検証/WFA対応版）

        Args:
            individual: 評価する個体
            config: GA設定

        Returns:
            フィットネス値のタプル
        """
        try:
            # 遺伝子デコード
            from ..genes import StrategyGene
            from ..serializers.gene_serialization import GeneSerializer

            if isinstance(individual, StrategyGene):
                gene = individual
            else:
                gene_serializer = GeneSerializer()
                gene = gene_serializer.from_list(individual, StrategyGene)

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
            return tuple(0.0 for _ in config.objectives)

    def _execute_evaluation_logic(
        self, gene, base_backtest_config, config
    ) -> Tuple[float, ...]:
        """実際の評価ロジック（OOS/WFA分岐）"""

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
    ):
        """OOS検証付き評価"""
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
                        f"WFA Fold {fold_idx}: トレーニング期間が短すぎるためスキップ"
                    )
                    continue

                # テスト期間が短すぎる場合はスキップ
                if (test_end - test_start).days < 1:
                    logger.debug(
                        f"WFA Fold {fold_idx}: テスト期間が短すぎるためスキップ"
                    )
                    continue

                # 日付文字列に変換
                train_start_str = fold_train_start.strftime("%Y-%m-%d %H:%M:%S")
                train_end_str = train_end.strftime("%Y-%m-%d %H:%M:%S")
                test_start_str = test_start.strftime("%Y-%m-%d %H:%M:%S")
                test_end_str = test_end.strftime("%Y-%m-%d %H:%M:%S")

                logger.debug(
                    f"WFA Fold {fold_idx}: "
                    f"Train={train_start_str} to {train_end_str}, "
                    f"Test={test_start_str} to {test_end_str}"
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
            from .worker_initializer import get_worker_data

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
            from .worker_initializer import get_worker_data

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
        """単一期間での評価実行"""
        try:
            # 遺伝子から戦略設定を生成
            from ..serializers.gene_serialization import GeneSerializer

            serializer = GeneSerializer()

            # Pydanticモデルを使用して設定を構築
            # 1. 辞書をコピーしてベースにする
            config_dict = backtest_config.copy()

            # 2. StrategyConfig部分を構築
            strategy_parameters = {
                "strategy_gene": serializer.strategy_gene_to_dict(gene),
                "ml_filter_enabled": config.ml_filter_enabled,
                "ml_model_path": config.ml_model_path,
            }

            config_dict["strategy_config"] = {
                "strategy_type": "GENERATED_GA",
                "parameters": strategy_parameters,
            }
            config_dict["strategy_name"] = f"GA_Individual_{gene.id[:8]}"

            # 3. モデル化（バリデーション）
            try:
                backtest_config_model = BacktestConfig(**config_dict)
            except ValidationError as e:
                logger.error(f"バックテスト設定モデル生成エラー: {e}")
                return tuple(0.0 for _ in config.objectives)

            # MLフィルター設定（モデル外のオブジェクト）
            ml_filter_model = None
            if config.ml_filter_enabled and config.ml_model_path:
                try:
                    ml_filter_model = model_manager.load_model(config.ml_model_path)
                except Exception:
                    # エラー時は設定を無効化
                    pass

            # データをキャッシュから取得または新規取得
            # キャッシュキー作成には辞書を使用（モデルでも良いが既存踏襲）
            data = self._get_cached_data(config_dict)

            # 1分足データを取得（1分足シミュレーション用）
            minute_data = self._get_cached_minute_data(config_dict)

            # 実行用辞書の作成
            run_config = backtest_config_model.model_dump()

            # モデル外のオブジェクトを注入
            if minute_data is not None:
                run_config["strategy_config"]["parameters"]["minute_data"] = minute_data

            if ml_filter_model:
                run_config["strategy_config"]["parameters"][
                    "ml_predictor"
                ] = ml_filter_model
                # ダマシ予測モデル用の閾値: 0.5 = 50%以上の確率で有効ならエントリー許可
                # 将来的にGAConfigに追加することを検討
                run_config["strategy_config"]["parameters"]["ml_filter_threshold"] = 0.5
            elif config.ml_filter_enabled:  # ロード失敗時
                run_config["strategy_config"]["parameters"]["ml_filter_enabled"] = False

            # バックテスト実行
            result = self.backtest_service.run_backtest(
                config=run_config, preloaded_data=data
            )

            # フィットネス計算（常に統一ロジックを使用）
            return self._calculate_multi_objective_fitness(result, config)

        except Exception as e:
            logger.error(f"単一評価実行エラー: {e}")
            return tuple(0.0 for _ in config.objectives)

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
        self,
        backtest_result: Dict[str, Any],
        config: GAConfig,
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
        self,
        backtest_result: Dict[str, Any],
        config: GAConfig,
    ) -> tuple:
        """
        多目的最適化用フィットネス計算

        Args:
            backtest_result: バックテスト結果
            config: GA設定

        Returns:
            各目的の評価値のタプル
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
                    value = self._calculate_fitness(backtest_result, config)
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





