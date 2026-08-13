"""
オートストラテジー実行スクリプト

遺伝的アルゴリズムを使用して取引戦略を自動生成し、
結果をJSON形式で出力します。

デフォルトでは backend/results/auto_strategy ディレクトリに
日付時刻付きのファイル名（例: strategy_2024-12-12_231030.json）で保存されます。

使用方法:
    python -m scripts.run_auto_strategy [オプション]

例:
    # デフォルト設定で実行（自動的にresults/auto_strategyに保存）
    python -m scripts.run_auto_strategy

    # 設定をカスタマイズして実行
    python -m scripts.run_auto_strategy --generations 20 --population 30

    # 結果を指定したファイルに保存
    python -m scripts.run_auto_strategy --output results/my_strategy.json

    # ファイル保存をスキップ（標準出力のみ）
    python -m scripts.run_auto_strategy --no-save
"""

import argparse
import json
import logging
import sys
from datetime import datetime
from pathlib import Path
from typing import Any

# プロジェクトルートをパスに追加
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from app.services.auto_strategy.config.ga import GAConfig  # noqa: E402
from app.services.auto_strategy.config.ga.nested_configs import (  # noqa: E402
    EvaluationConfig,
)
from app.services.auto_strategy.core import GeneticAlgorithmEngine  # noqa: E402
from app.services.auto_strategy.core.evaluation.individual_evaluator import (  # noqa: E402
    IndividualEvaluator,
)
from app.services.auto_strategy.generators.random_gene_generator import (  # noqa: E402
    RandomGeneGenerator,
)
from app.services.auto_strategy.genes.strategy import StrategyGene  # noqa: E402
from app.services.auto_strategy.serializers.serialization import (
    GeneSerializer,  # noqa: E402
)
from app.services.auto_strategy.services.strategy_validation_service import (  # noqa: E402
    StrategyValidationService,
)
from app.services.backtest.services.backtest_service import (
    BacktestService,  # noqa: E402
)

# ロギング設定
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    handlers=[
        logging.StreamHandler(sys.stdout),
    ],
)
logger = logging.getLogger(__name__)


def apply_smoke_mode(args: argparse.Namespace) -> argparse.Namespace:
    """高速なバグ確認向けの実行設定を反映します。

    `--smoke` が有効な場合に、実行時間に直結する副作用をまとめて抑制します。
    """
    if not getattr(args, "smoke", False):
        return args

    # できるだけ小さい構成に固定して、GA/バックテスト本体の動作確認に集中する。
    args.population = 2
    args.generations = 1
    args.elite_size = 1
    args.no_parallel = True
    args.no_save = True
    if getattr(args, "min_trades", None) is None:
        args.min_trades = 0

    return args


def parse_args() -> argparse.Namespace:
    """コマンドライン引数をパースします。"""
    parser = argparse.ArgumentParser(
        description="オートストラテジーで取引戦略を自動生成します",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
例:
  # 基本的な実行（自動的にresults/auto_strategy/に日付付きで保存）
  python -m scripts.run_auto_strategy

  # 高速テスト（少ない世代数・個体数）
  python -m scripts.run_auto_strategy --generations 5 --population 10

  # 本格的な探索
  python -m scripts.run_auto_strategy --generations 50 --population 100

  # 結果を指定したファイルに保存
  python -m scripts.run_auto_strategy --output results/my_strategy.json

  # ファイル保存をスキップ（標準出力のみ）
  python -m scripts.run_auto_strategy --no-save

  # 小さな世代数・個体数で高速に回すバグ確認モード
  python -m scripts.run_auto_strategy --smoke --start-date 2024-01-01 --end-date 2024-01-31
        """,
    )

    # GA設定
    parser.add_argument(
        "--generations",
        "-g",
        type=int,
        default=10,
        help="進化の世代数 (デフォルト: 10)",
    )
    parser.add_argument(
        "--population",
        "-p",
        type=int,
        default=20,
        help="個体数 (デフォルト: 20)",
    )
    parser.add_argument(
        "--elite-size",
        "-e",
        type=int,
        default=2,
        help="エリート保存数 (デフォルト: 2)",
    )
    parser.add_argument(
        "--crossover-rate",
        type=float,
        default=0.8,
        help="交叉率 (デフォルト: 0.8)",
    )
    parser.add_argument(
        "--mutation-rate",
        type=float,
        default=0.2,
        help="突然変異率 (デフォルト: 0.2)",
    )

    # バックテスト設定
    parser.add_argument(
        "--symbol",
        "-s",
        type=str,
        default="BTC/USDT:USDT",
        help="取引ペア (デフォルト: BTC/USDT:USDT)",
    )
    parser.add_argument(
        "--timeframe",
        "-t",
        type=str,
        default="4h",
        help="時間足 (デフォルト: 4h)",
    )
    parser.add_argument(
        "--start-date",
        type=str,
        default="2024-01-01",
        help="バックテスト開始日 (デフォルト: 2024-01-01)",
    )
    parser.add_argument(
        "--end-date",
        type=str,
        default="2024-06-30",
        help="バックテスト終了日 (デフォルト: 2024-06-30)",
    )
    parser.add_argument(
        "--initial-capital",
        type=float,
        default=100000.0,
        help="初期資本 (デフォルト: 100000)",
    )

    # 出力設定
    parser.add_argument(
        "--output",
        "-o",
        type=str,
        default=None,
        help="結果を保存するJSONファイルパス (未指定の場合は標準出力)",
    )
    parser.add_argument(
        "--verbose",
        "-v",
        action="store_true",
        help="詳細なログを出力",
    )
    parser.add_argument(
        "--no-parallel",
        action="store_true",
        help="並列評価を無効化",
    )
    parser.add_argument(
        "--no-save",
        action="store_true",
        help="ファイル保存をスキップし、標準出力のみに出力",
    )
    parser.add_argument(
        "--smoke",
        action="store_true",
        help="並列評価・詳細保存を無効化し、小さな世代数・個体数で高速に回すバグ確認モード",
    )
    parser.add_argument(
        "--min-trades",
        type=int,
        default=None,
        help="最小取引回数制約 (デフォルト: 設定ファイル依存, 0で無効化)",
    )
    parser.add_argument(
        "--no-validation",
        action="store_true",
        help="GA後のWFA自動検証を無効化（高速実行用）",
    )

    # 探索空間の拡張設定
    parser.add_argument(
        "--mtf",
        action="store_true",
        help="マルチタイムフレーム（MTF）指標の生成を有効化",
    )
    parser.add_argument(
        "--mtf-timeframes",
        type=str,
        default="1d",
        help="MTFで使用するタイムフレーム（カンマ区切り, 例: 1d,4h）。"
        "ベースタイムフレームより大きい整数倍の足のみ有効（例: 4h基準なら 1d のみ）",
    )
    parser.add_argument(
        "--mtf-probability",
        type=float,
        default=0.3,
        help="MTF指標が生成される確率 (0.0-1.0, デフォルト: 0.3)",
    )
    parser.add_argument(
        "--indicator-universe",
        type=str,
        choices=["curated", "experimental_all"],
        default="curated",
        help="インジケーターユニバース (デフォルト: curated, "
        "experimental_all は実装済み全220種を使用)。注意: experimental_all "
        "は検証済みカタログ外の指標を含むため、NaNや条件生成に不向きな指標が混入しやすい",
    )
    parser.add_argument(
        "--max-indicators",
        type=int,
        default=10,
        help="1戦略あたりの最大インジケーター数 (デフォルト: 10)",
    )
    parser.add_argument(
        "--min-non-price",
        type=int,
        default=0,
        help="1戦略あたりに含める非価格指標（OI/FR/LSR由来）の最低数 "
        "(デフォルト: 0 = 強制なし, 1以上で保証)",
    )
    parser.add_argument(
        "--non-price-probability",
        type=float,
        default=0.3,
        help="非価格指標の選択確率（公平な自由探索用, 0.0-1.0, デフォルト: 0.3）。"
        "トレンド70%優先バイアスで非価格指標が埋もれるのを防ぐ",
    )
    parser.add_argument(
        "--max-conditions",
        type=int,
        default=3,
        help="エントリー条件の最大数 (デフォルト: 3)",
    )

    return parser.parse_args()


def create_ga_config(args: argparse.Namespace) -> GAConfig:
    """引数からGAConfigを作成します。

    Args:
        args: コマンドライン引数

    Returns:
        GAConfig: GA設定オブジェクト

    Raises:
        ValueError: 引数が不正な場合
    """
    smoke_mode = getattr(args, "smoke", False)

    population = 2 if smoke_mode else args.population
    generations = 1 if smoke_mode else args.generations
    elite_size = 1 if smoke_mode else args.elite_size

    # 入力検証
    if population < 2:
        raise ValueError("個体数は2以上である必要があります")
    if generations < 1:
        raise ValueError("世代数は1以上である必要があります")
    if elite_size < 0:
        raise ValueError("エリート保存数は0以上である必要があります")
    if elite_size >= population:
        raise ValueError("エリート保存数は個体数未満である必要があります")
    if not 0 <= args.crossover_rate <= 1:
        raise ValueError("交叉率は0から1の範囲である必要があります")
    if not 0 <= args.mutation_rate <= 1:
        raise ValueError("突然変異率は0から1の範囲である必要があります")

    # fitness_constraints の上書き
    fitness_constraints = None
    min_trades = getattr(args, "min_trades", None)
    if min_trades is not None or smoke_mode:
        from app.services.auto_strategy.config.constants import (
            DEFAULT_FITNESS_CONSTRAINTS,
        )

        fitness_constraints = DEFAULT_FITNESS_CONSTRAINTS.copy()
        fitness_constraints["min_trades"] = min_trades if min_trades is not None else 0

    config_kwargs = {
        "population_size": population,
        "generations": generations,
        "crossover_rate": args.crossover_rate,
        "mutation_rate": args.mutation_rate,
        "elite_size": elite_size,
        "evaluation_config": EvaluationConfig(
            enable_parallel=not args.no_parallel and not smoke_mode
        ),
        "log_level": "DEBUG" if args.verbose else "INFO",
        "fallback_start_date": args.start_date,
        "fallback_end_date": args.end_date,
    }
    if smoke_mode:
        config_kwargs["max_indicators"] = 3
        config_kwargs["max_conditions"] = 2
        config_kwargs["use_seed_strategies"] = False
        config_kwargs["seed_injection_rate"] = 0.0
        config_kwargs["two_stage_selection_config"] = {"enabled": False}
    else:
        # 探索空間の拡張設定（通常モードのみ適用）
        config_kwargs["max_indicators"] = getattr(args, "max_indicators", 10)
        config_kwargs["max_conditions"] = getattr(args, "max_conditions", 3)
        min_non_price = getattr(args, "min_non_price", 0)
        if min_non_price:
            config_kwargs["min_non_price_indicators"] = min_non_price
        non_price_probability = getattr(args, "non_price_probability", 0.3)
        config_kwargs["non_price_indicator_probability"] = non_price_probability

        mtf_enabled = getattr(args, "mtf", False)
        if mtf_enabled:
            mtf_timeframes = [
                tf.strip()
                for tf in getattr(args, "mtf_timeframes", "1d").split(",")
                if tf.strip()
            ]
            mtf_probability = getattr(args, "mtf_probability", 0.3)
            if not 0.0 <= mtf_probability <= 1.0:
                raise ValueError("--mtf-probability は0から1の範囲である必要があります")
            # サポート外タイムフレームは早期に弾く（無効なMTF指標の生成を防ぐ）
            from app.config.constants import SUPPORTED_TIMEFRAMES

            invalid_tfs = [
                tf for tf in mtf_timeframes if tf not in SUPPORTED_TIMEFRAMES
            ]
            if invalid_tfs:
                raise ValueError(
                    "サポートされていないタイムフレーム: "
                    f"{', '.join(invalid_tfs)}. 有効: {SUPPORTED_TIMEFRAMES}"
                )
            config_kwargs["enable_multi_timeframe"] = True
            config_kwargs["available_timeframes"] = mtf_timeframes
            config_kwargs["mtf_indicator_probability"] = mtf_probability

        universe = getattr(args, "indicator_universe", "curated")
        if universe != "curated":
            config_kwargs["indicator_universe_mode"] = universe

    if smoke_mode or getattr(args, "no_validation", False):
        # 高速実行時はWFA自動検証を無効化
        config_kwargs["validation_config"] = {"enabled": False}
    if fitness_constraints is not None:
        config_kwargs["fitness_constraints"] = fitness_constraints

    return GAConfig(**config_kwargs)


def create_backtest_config(args: argparse.Namespace) -> dict[str, Any]:
    """引数からバックテスト設定を作成します。

    Args:
        args: コマンドライン引数

    Returns:
        Dict[str, Any]: バックテスト設定辞書

    Raises:
        ValueError: 引数が不正な場合
    """
    # 入力検証
    if args.initial_capital <= 0:
        raise ValueError("初期資本は0より大きい必要があります")

    return {
        "symbol": args.symbol,
        "timeframe": args.timeframe,
        "start_date": args.start_date,
        "end_date": args.end_date,
        "initial_capital": args.initial_capital,
        "commission_rate": 0.0004,  # 0.04%手数料
        "slippage": 0.0001,  # 0.01%スリッページ
    }


def strategy_gene_to_readable_dict(
    gene: StrategyGene, serializer: GeneSerializer
) -> dict[str, Any]:
    """StrategyGeneを可読性の高い辞書に変換します。"""
    # シリアライザを使って基本辞書を取得
    base_dict = serializer.strategy_gene_to_dict(gene)

    # 可読性を高めるための追加処理
    readable = {
        "strategy_name": base_dict.get("name", "GA Generated Strategy"),
        "description": "遺伝的アルゴリズムによって自動生成された取引戦略",
        "generated_at": datetime.now().isoformat(),
        "indicators": [],
        "entry_conditions": {
            "long": [],
            "short": [],
        },
        "exit_conditions": {
            "long": [],
            "short": [],
        },
        "risk_management": base_dict.get("risk_management", {}),
        "tpsl": None,
        "position_sizing": None,
        "raw_gene": base_dict,
    }

    # インジケーターの整形
    for ind in base_dict.get("indicators", []):
        # インジケーター名を取得（複数のキー名に対応）
        indicator_name = (
            ind.get("indicator") or ind.get("type") or ind.get("name") or "Unknown"
        )
        # パラメータを取得（複数のキー名に対応）
        params = ind.get("params") or ind.get("parameters") or {}
        readable["indicators"].append(
            {
                "name": indicator_name,
                "parameters": params,
                "timeframe": ind.get("timeframe"),
                "enabled": ind.get("enabled", True),
            }
        )

    # エントリー条件の整形
    if "long_entry_conditions" in base_dict:
        readable["entry_conditions"]["long"] = _format_conditions(
            base_dict["long_entry_conditions"]
        )
    if "short_entry_conditions" in base_dict:
        readable["entry_conditions"]["short"] = _format_conditions(
            base_dict["short_entry_conditions"]
        )

    # イグジット条件の整形
    if "long_exit_conditions" in base_dict:
        readable["exit_conditions"]["long"] = _format_conditions(
            base_dict["long_exit_conditions"]
        )
    if "short_exit_conditions" in base_dict:
        readable["exit_conditions"]["short"] = _format_conditions(
            base_dict["short_exit_conditions"]
        )

    # TPSLの整形
    if "long_tpsl_gene" in base_dict and base_dict["long_tpsl_gene"]:
        readable["tpsl"] = {
            "long": base_dict["long_tpsl_gene"],
            "short": base_dict.get("short_tpsl_gene"),
        }

    # ポジションサイジングの整形
    if "position_sizing_gene" in base_dict and base_dict["position_sizing_gene"]:
        readable["position_sizing"] = base_dict["position_sizing_gene"]

    return readable


def _format_conditions(conditions: Any) -> list:
    """条件リストを可読形式に整形します。"""
    if not conditions:
        return []

    formatted = []
    if isinstance(conditions, list):
        for cond in conditions:
            formatted.append(_format_single_condition(cond))
    elif isinstance(conditions, dict):
        formatted.append(_format_single_condition(conditions))

    return formatted


def _format_single_condition(cond: Any) -> dict[str, Any]:
    """単一条件を可読形式に整形します。"""
    if not isinstance(cond, dict):
        return {"raw": str(cond)}

    # ConditionGroupの場合
    if "conditions" in cond:
        return {
            "type": "group",
            "logic": cond.get("logic", "AND"),
            "conditions": [
                _format_single_condition(c) for c in cond.get("conditions", [])
            ],
        }

    # 通常のConditionの場合
    return {
        "left_operand": cond.get("left_operand", ""),
        "operator": cond.get("operator", ""),
        "right_operand": cond.get("right_operand", ""),
        "description": _create_condition_description(cond),
    }


def _create_condition_description(cond: dict[str, Any]) -> str:
    """条件の説明文を生成します。"""
    left = cond.get("left_operand", "?")
    op = cond.get("operator", "?")
    right = cond.get("right_operand", "?")

    # オペランドの文字列化
    def operand_to_str(operand: Any) -> str:
        if isinstance(operand, dict):
            return operand.get("indicator", operand.get("value", str(operand)))
        return str(operand)

    return f"{operand_to_str(left)} {op} {operand_to_str(right)}"


def run_auto_strategy(args: argparse.Namespace) -> dict[str, Any]:
    """オートストラテジーを実行します。"""
    args = apply_smoke_mode(args)

    logger.info("=" * 60)
    logger.info("オートストラテジー実行開始")
    logger.info("=" * 60)
    if getattr(args, "smoke", False):
        logger.info(
            "smokeモード: 並列評価・詳細保存を無効化し、小さな世代数・個体数で実行しています"
        )

    # 設定の作成
    ga_config = create_ga_config(args)
    backtest_config = create_backtest_config(args)

    logger.info(
        f"GA設定: 世代数={ga_config.generations}, 個体数={ga_config.population_size}"
    )
    symbol = backtest_config["symbol"]
    timeframe = backtest_config["timeframe"]
    logger.info(f"バックテスト設定: {symbol} / {timeframe}")
    logger.info(
        f"期間: {backtest_config['start_date']} ~ {backtest_config['end_date']}"
    )

    # サービスの初期化
    logger.info("サービスを初期化中...")
    backtest_service = BacktestService()
    gene_generator = RandomGeneGenerator(config=ga_config)

    # GAエンジンの作成
    logger.info("GAエンジンを初期化中...")
    ga_engine = GeneticAlgorithmEngine(
        backtest_service=backtest_service,
        gene_generator=gene_generator,
    )

    # 進化の実行
    logger.info("-" * 60)
    logger.info("進化を開始します...")
    logger.info("-" * 60)

    try:
        result = ga_engine.run_evolution(
            config=ga_config,
            backtest_config=backtest_config,
        )

        logger.info("-" * 60)
        logger.info("進化が完了しました!")
        logger.info("-" * 60)

        # 自動検証パイプライン（WFA + 合格判定）
        # 実験マネージャーと同じ品質基準をCLI実行にも適用する。
        # 合格した戦略だけが詳細バックテスト・出力の対象となる。
        if ga_config.validation_config.enabled:
            logger.info("自動検証パイプラインを開始します（WFA）")
            evaluator = getattr(ga_engine, "individual_evaluator", None)
            if evaluator is None:
                evaluator = IndividualEvaluator(backtest_service)
            validation_service = StrategyValidationService(evaluator)
            result = validation_service.validate_and_filter_result(
                result=result,
                ga_config=ga_config,
                backtest_config=backtest_config,
            )
            validation_results = result.get("validation_results", {})
            passed_count = sum(
                1 for v in validation_results.values() if v.get("passed", False)
            )
            logger.info(
                "自動検証結果: 検証 %d 件 / 合格 %d 件",
                len(validation_results),
                passed_count,
            )

        # 結果の整形
        best_gene = result.get("best_strategy")
        best_fitness = result.get("best_fitness")
        execution_time = result.get("execution_time", 0)

        logger.info(f"最良フィットネス: {best_fitness}")
        logger.info(f"実行時間: {execution_time:.2f}秒")

        # シリアライザの作成
        serializer = GeneSerializer()

        saved_backtest_path = None

        # 可読形式の戦略辞書を作成
        if best_gene is None:
            # 自動検証で全戦略が不合格の場合
            logger.warning(
                "合格した戦略が無いため、詳細バックテストと戦略出力をスキップします"
            )
            strategy_dict = None
        elif isinstance(best_gene, StrategyGene):
            strategy_dict = strategy_gene_to_readable_dict(best_gene, serializer)
            if not args.no_save:
                # === 最良戦略の詳細バックテスト結果を別途保存 ===
                try:
                    logger.info("最良戦略の詳細バックテストを実行して保存中...")
                    # 辞書形式に変換して渡す
                    gene_dict = serializer.strategy_gene_to_dict(best_gene)
                    full_bt_config = {
                        "strategy_name": "universal_strategy",
                        "symbol": backtest_config["symbol"],
                        "timeframe": backtest_config["timeframe"],
                        "start_date": backtest_config["start_date"],
                        "end_date": backtest_config["end_date"],
                        "initial_capital": backtest_config["initial_capital"],
                        "strategy_config": {
                            "strategy_type": "GENERATED_GA",
                            "parameters": {
                                "strategy_gene": gene_dict,
                                # 最終レポート用の詳細バックテストは早期終了を無効化する
                                # （trade_pace 早期終了が発動すると BacktestEarlyTerminationError
                                #  で結果が保存されず、実験サービスと挙動が異なるため）
                                "early_termination_settings": {"enabled": False},
                            },
                        },
                        "commission_rate": 0.0004,
                        "slippage": 0.0001,
                    }
                    full_result = backtest_service.run_backtest(full_bt_config)

                    # 保存ディレクトリの準備
                    bt_results_dir = project_root / "results" / "backtest"
                    bt_results_dir.mkdir(parents=True, exist_ok=True)

                    # ファイル名の生成（戦略IDやタイムスタンプを含む）
                    bt_filename = f"backtest_result_{datetime.now().strftime('%Y-%m-%d_%H%M%S')}.json"
                    bt_path = bt_results_dir / bt_filename

                    with open(bt_path, "w", encoding="utf-8") as f:
                        # datetimeなどを文字列化するために default=str を指定
                        json.dump(
                            full_result, f, indent=2, ensure_ascii=False, default=str
                        )

                    logger.info(f"詳細バックテスト結果を保存しました: {bt_path}")
                    saved_backtest_path = str(bt_path)
                except Exception as bt_err:
                    logger.warning(f"詳細バックテストの保存に失敗しました: {bt_err}")
        else:
            strategy_dict = {"raw": str(best_gene)}

        # 結果のまとめ
        output = {
            "success": True,
            "execution_summary": {
                "generations_completed": result.get(
                    "generations_completed", ga_config.generations
                ),
                "final_population_size": result.get(
                    "final_population_size", ga_config.population_size
                ),
                "execution_time_seconds": round(execution_time, 2),
                "best_fitness": (
                    best_fitness
                    if not isinstance(best_fitness, tuple)
                    else list(best_fitness)
                ),
            },
            "ga_config": ga_config.to_dict(),
            "backtest_config": backtest_config,
            "best_strategy": strategy_dict,
        }

        if saved_backtest_path:
            output["backtest_result_file"] = saved_backtest_path

        # 自動検証結果を出力に含める（実行された場合）
        if "validation_results" in result:
            output["validation_results"] = result["validation_results"]
        if "validation_report_summaries" in result:
            output["validation_report_summaries"] = result[
                "validation_report_summaries"
            ]

        # パレート最適解がある場合
        if "pareto_front" in result:
            output["pareto_front"] = []
            for item in result["pareto_front"]:
                strategy = item.get("strategy")
                if isinstance(strategy, StrategyGene):
                    strategy_readable = strategy_gene_to_readable_dict(
                        strategy, serializer
                    )
                else:
                    strategy_readable = {"raw": str(strategy)}
                output["pareto_front"].append(
                    {
                        "fitness_values": item.get("fitness_values", []),
                        "strategy": strategy_readable,
                    }
                )

        return output

    except Exception as e:
        logger.error(f"進化実行中にエラーが発生しました: {e}", exc_info=True)
        return {
            "success": False,
            "error": str(e),
            "ga_config": ga_config.to_dict(),
            "backtest_config": backtest_config,
        }
    finally:
        # クリーンアップ
        if hasattr(backtest_service, "cleanup"):
            backtest_service.cleanup()


def generate_output_filename() -> str:
    """日付時刻付きの出力ファイル名を生成します。

    Returns:
        str: 生成されたファイル名（例: strategy_2024-12-12_231030.json）
    """
    timestamp = datetime.now().strftime("%Y-%m-%d_%H%M%S")
    return f"strategy_{timestamp}.json"


def get_default_output_dir() -> Path:
    """デフォルトの出力ディレクトリを取得します。

    Returns:
        Path: backend/results/auto_strategy ディレクトリへのパス
    """
    return project_root / "results" / "auto_strategy"


def main():
    """メインエントリーポイント"""
    args = parse_args()
    args = apply_smoke_mode(args)

    # 詳細ログの設定
    if args.verbose:
        logging.getLogger().setLevel(logging.DEBUG)
        logging.getLogger("app.services.auto_strategy").setLevel(logging.DEBUG)

    # 実行
    result = run_auto_strategy(args)

    # JSON出力
    json_output = json.dumps(result, indent=2, ensure_ascii=False, default=str)

    # 出力先の決定
    if args.no_save:
        # 保存をスキップ、標準出力のみ
        output_path = None
    elif args.output:
        # 指定されたパスに保存
        output_path = Path(args.output)
    else:
        # デフォルトディレクトリに日付付きファイル名で保存
        output_dir = get_default_output_dir()
        output_path = output_dir / generate_output_filename()

    if output_path:
        # ファイルに保存
        output_path.parent.mkdir(parents=True, exist_ok=True)
        with open(output_path, "w", encoding="utf-8") as f:
            f.write(json_output)
        logger.info(f"結果を保存しました: {output_path}")

    # 標準出力にも表示
    print("\n" + "=" * 60)
    print("生成された戦略 (JSON)")
    print("=" * 60)
    print(json_output)

    if output_path:
        print("\n" + "-" * 60)
        print(f"保存先: {output_path}")
        print("-" * 60)

    return 0 if result.get("success") else 1


if __name__ == "__main__":
    sys.exit(main())
