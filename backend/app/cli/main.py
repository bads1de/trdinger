"""
Trdinger CLI

サーバー不要でオートストラテジー実験をターミナルから操作する CLI。

コマンド:
    trdinger exp run [options]    # GA実験を実行（同期、DB保存あり）
    trdinger exp list             # 実験一覧
    trdinger exp show <uuid>      # 実験詳細
    trdinger exp stop <uuid>      # 実行中実験の停止
    trdinger exp delete <uuid>    # 実験削除
    trdinger strategy list        # 生成済み戦略一覧
"""

from __future__ import annotations

import json
import logging
import sys
import uuid
from typing import Annotated, Any, cast

import typer

from app.cli._services import (
    SynchronousBackgroundTasks,
    build_data_collection_services,
    build_services,
    build_task_scheduler,
)
from app.cli.config_builder import (
    build_backtest_config_dict,
    build_ga_config_dict,
)
from app.services.auto_strategy.config.ga_config import GAConfig
from app.services.auto_strategy.services.experiment_application_service import (
    TaskScheduler,
)
from database.connection import SessionLocal

logger = logging.getLogger("trdinger")

app = typer.Typer(
    help="Trdinger オートストラテジー CLI",
    no_args_is_help=True,
    pretty_exceptions_show_locals=False,
)

exp_app = typer.Typer(help="GA実験の管理", no_args_is_help=True)
app.add_typer(exp_app, name="exp")

strategy_app = typer.Typer(help="生成済み戦略の閲覧", no_args_is_help=True)
app.add_typer(strategy_app, name="strategy")

data_app = typer.Typer(help="市場データの収集・管理", no_args_is_help=True)
app.add_typer(data_app, name="data")


def _setup_logging(verbose: bool) -> None:
    level = logging.DEBUG if verbose else logging.INFO
    try:
        from app.utils.logging import configure_logging

        configure_logging(log_file="cli.log")
    except Exception:  # インポート失敗時は従来どおりコンソールのみ
        logging.basicConfig(
            level=level,
            format="%(asctime)s %(levelname)s %(name)s: %(message)s",
            handlers=[logging.StreamHandler(sys.stdout)],
        )


def _new_experiment_id() -> str:
    return str(uuid.uuid4())


def _print_json(data: Any) -> None:
    typer.echo(json.dumps(data, indent=2, ensure_ascii=False, default=str))


@exp_app.command("run")
def exp_run(
    name: Annotated[
        str, typer.Option("--name", "-n", help="実験名")
    ] = "CLI experiment",
    population: Annotated[int, typer.Option("--population", "-p", help="個体数")] = 20,
    generations: Annotated[
        int, typer.Option("--generations", "-g", help="世代数")
    ] = 10,
    elite_size: Annotated[
        int, typer.Option("--elite-size", "-e", help="エリート保存数")
    ] = 2,
    crossover_rate: Annotated[
        float, typer.Option("--crossover-rate", help="交叉率")
    ] = 0.8,
    mutation_rate: Annotated[
        float, typer.Option("--mutation-rate", help="突然変異率")
    ] = 0.2,
    symbol: Annotated[
        str, typer.Option("--symbol", "-s", help="取引ペア")
    ] = "BTC/USDT:USDT",
    timeframe: Annotated[str, typer.Option("--timeframe", "-t", help="時間足")] = "4h",
    start_date: Annotated[
        str, typer.Option("--start-date", help="バックテスト開始日")
    ] = "2024-01-01",
    end_date: Annotated[
        str, typer.Option("--end-date", help="バックテスト終了日")
    ] = "2024-06-30",
    initial_capital: Annotated[
        float, typer.Option("--initial-capital", help="初期資本")
    ] = 100000.0,
    no_parallel: Annotated[
        bool, typer.Option("--no-parallel", help="並列評価を無効化")
    ] = False,
    no_validation: Annotated[
        bool, typer.Option("--no-validation", help="WFA自動検証を無効化")
    ] = False,
    no_seeds: Annotated[
        bool, typer.Option("--no-seeds", help="シード戦略注入を無効化")
    ] = False,
    min_trades: Annotated[
        int | None,
        typer.Option("--min-trades", help="最小取引回数制約（0で無効化）"),
    ] = None,
    smoke: Annotated[bool, typer.Option("--smoke", help="高速スモークモード")] = False,
    mtf: Annotated[
        bool, typer.Option("--mtf", help="マルチタイムフレーム指標を有効化")
    ] = False,
    mtf_timeframes: Annotated[
        str, typer.Option("--mtf-timeframes", help="MTFタイムフレーム（カンマ区切り）")
    ] = "1d",
    mtf_probability: Annotated[
        float, typer.Option("--mtf-probability", help="MTF指標生成確率")
    ] = 0.3,
    indicator_universe: Annotated[
        str,
        typer.Option(
            "--indicator-universe",
            help="インジケーターユニバース（curated | experimental_all）",
        ),
    ] = "curated",
    max_indicators: Annotated[
        int, typer.Option("--max-indicators", help="1戦略あたりの最大インジケーター数")
    ] = 10,
    min_non_price: Annotated[
        int, typer.Option("--min-non-price", help="非価格指標の最低数")
    ] = 0,
    non_price_probability: Annotated[
        float, typer.Option("--non-price-probability", help="非価格指標の選択確率")
    ] = 0.3,
    max_conditions: Annotated[
        int, typer.Option("--max-conditions", help="エントリー条件の最大数")
    ] = 3,
    verbose: Annotated[bool, typer.Option("--verbose", "-v", help="詳細ログ")] = False,
    json_output: Annotated[
        bool, typer.Option("--json", help="結果をJSONで出力")
    ] = False,
) -> None:
    """GA実験を実行する（サーバー不要・DB保存あり）。"""
    _setup_logging(verbose)

    try:
        ga_config_dict = build_ga_config_dict(
            population=population,
            generations=generations,
            crossover_rate=crossover_rate,
            mutation_rate=mutation_rate,
            elite_size=elite_size,
            start_date=start_date,
            end_date=end_date,
            no_parallel=no_parallel,
            verbose=verbose,
            smoke=smoke,
            min_trades=min_trades,
            no_validation=no_validation,
            no_seeds=no_seeds,
            mtf=mtf,
            mtf_timeframes=mtf_timeframes,
            mtf_probability=mtf_probability,
            indicator_universe=indicator_universe,
            max_indicators=max_indicators,
            min_non_price=min_non_price,
            non_price_probability=non_price_probability,
            max_conditions=max_conditions,
        )
        backtest_config_dict = build_backtest_config_dict(
            symbol=symbol,
            timeframe=timeframe,
            start_date=start_date,
            end_date=end_date,
            initial_capital=initial_capital,
        )
    except ValueError as e:
        raise typer.BadParameter(str(e)) from e

    ga_config = GAConfig.from_dict(ga_config_dict)
    experiment_id = _new_experiment_id()

    auto_strategy_service, _ = build_services()
    scheduler: TaskScheduler = build_task_scheduler()

    try:
        auto_strategy_service.start_strategy_generation(
            experiment_id=experiment_id,
            experiment_name=name,
            ga_config_dict=ga_config_dict,
            backtest_config_dict=backtest_config_dict,
            task_scheduler=scheduler,
        )
    except RuntimeError as e:
        typer.echo(f"実行失敗: {e}", err=True)
        raise typer.Exit(code=1) from e

    detail = auto_strategy_service.get_experiment_detail(experiment_id)
    summary = {
        "experiment_id": experiment_id,
        "status": detail.get("status") if detail else "unknown",
        "best_fitness": detail.get("best_fitness") if detail else None,
        "ga_config": ga_config.to_dict(),
    }
    if json_output:
        _print_json(summary)
    else:
        typer.echo(
            f"実験完了: {experiment_id} "
            f"(status={summary['status']}, fitness={summary['best_fitness']})"
        )


@exp_app.command("list")
def exp_list(
    json_output: Annotated[bool, typer.Option("--json", help="JSON形式で出力")] = False,
) -> None:
    """実験一覧を表示する。"""
    auto_strategy_service, _ = build_services()
    experiments = auto_strategy_service.list_experiments()

    if json_output:
        _print_json(experiments)
        return

    if not experiments:
        typer.echo("実験はまだありません")
        return
    for exp in experiments:
        typer.echo(
            f"[{exp['status']}] {exp['id']}  {exp['experiment_name']}  "
            f"fitness={exp.get('best_fitness')}  progress={exp.get('progress', 0):.0%}"
        )


@exp_app.command("show")
def exp_show(
    experiment_id: Annotated[str, typer.Argument(help="実験ID (UUID)")],
    json_output: Annotated[bool, typer.Option("--json", help="JSON形式で出力")] = False,
) -> None:
    """実験の詳細を表示する。"""
    auto_strategy_service, _ = build_services()
    detail = auto_strategy_service.get_experiment_detail(experiment_id)
    if detail is None:
        typer.echo(f"実験が見つかりません: {experiment_id}", err=True)
        raise typer.Exit(code=1)
    if json_output:
        _print_json(detail)
    else:
        typer.echo(f"ID: {detail['id']}")
        typer.echo(f"Name: {detail.get('experiment_name', detail.get('name'))}")
        typer.echo(f"Status: {detail.get('status')}")
        typer.echo(
            f"Progress: {detail.get('current_generation', 0)}/"
            f"{detail.get('total_generations', 0)}"
        )
        typer.echo(f"Best fitness: {detail.get('best_fitness')}")
        if detail.get("error_message"):
            typer.echo(f"Error: {detail['error_message']}")


@exp_app.command("stop")
def exp_stop(
    experiment_id: Annotated[str, typer.Argument(help="実験ID (UUID)")],
) -> None:
    """実行中の実験を停止する。"""
    auto_strategy_service, _ = build_services()
    result = auto_strategy_service.stop_experiment(experiment_id)
    if result.get("success"):
        typer.echo(f"停止しました: {experiment_id}")
    else:
        typer.echo(f"停止失敗: {result.get('message', '不明なエラー')}", err=True)
        raise typer.Exit(code=1)


@exp_app.command("delete")
def exp_delete(
    experiment_id: Annotated[str, typer.Argument(help="実験ID (UUID)")],
    yes: Annotated[bool, typer.Option("--yes", "-y", help="確認をスキップ")] = False,
) -> None:
    """実験を削除する（戦略・BT結果もカスケード削除）。"""
    if not yes:
        confirmed = typer.confirm(
            f"実験 {experiment_id} と関連する戦略・BT結果を削除しますか？"
        )
        if not confirmed:
            raise typer.Abort()

    auto_strategy_service, _ = build_services()
    result = auto_strategy_service.delete_experiment(experiment_id)
    if result.get("success"):
        typer.echo(result.get("message", "削除しました"))
    else:
        typer.echo(f"削除失敗: {result.get('message')}", err=True)
        raise typer.Exit(code=1)


@strategy_app.command("list")
def strategy_list(
    limit: Annotated[int, typer.Option("--limit", "-l", help="取得件数")] = 20,
    min_fitness: Annotated[
        float | None, typer.Option("--min-fitness", help="最小フィットネス")
    ] = None,
    json_output: Annotated[bool, typer.Option("--json", help="JSON形式で出力")] = False,
) -> None:
    """生成済み戦略の一覧を表示する。"""
    from app.services.auto_strategy.services.generated_strategy_service import (
        GeneratedStrategyService,
    )
    from database.connection import SessionLocal

    with SessionLocal() as db:
        service = GeneratedStrategyService(db)
        result = service.get_strategies(
            limit=limit,
            min_fitness=min_fitness,
        )

    if json_output:
        _print_json(result)
        return

    strategies = result["strategies"]
    if not strategies:
        typer.echo("戦略はまだありません")
        return
    for s in strategies:
        typer.echo(
            f"[{s['id']}] {s['name']}  "
            f"fitness={s.get('fitness_score')}  "
            f"ret={s.get('expected_return', 0):.2%}  "
            f"sharpe={s.get('sharpe_ratio', 0):.2f}  "
            f"risk={s.get('risk_level')}"
        )


def _print_ga_strategy(strategy: dict[str, Any]) -> None:
    typer.echo(f"ID: {strategy['id']}")
    typer.echo(f"Name: {strategy['name']}")
    typer.echo(f"Description: {strategy['description']}")
    typer.echo(f"Experiment: {strategy.get('experiment_id')}")
    typer.echo(f"Generation: {strategy.get('generation')}")
    typer.echo(f"Fitness: {strategy.get('fitness_score')}")
    typer.echo(f"Indicators: {', '.join(strategy.get('indicators', []))}")
    typer.echo(f"Expected return: {strategy.get('expected_return', 0):.2%}")
    typer.echo(f"Sharpe: {strategy.get('sharpe_ratio', 0):.2f}")
    typer.echo(f"Max DD: {strategy.get('max_drawdown', 0):.2%}")
    typer.echo(f"Win rate: {strategy.get('win_rate', 0):.2%}")
    typer.echo(f"Risk level: {strategy.get('risk_level')}")
    if strategy.get("validation_passed") is not None:
        typer.echo(f"Validation passed: {strategy['validation_passed']}")


@strategy_app.command("show")
def strategy_show(
    strategy_id: Annotated[str, typer.Argument(help="戦略ID (例: auto_42)")],
    json_output: Annotated[bool, typer.Option("--json", help="JSON形式で出力")] = False,
) -> None:
    """戦略の詳細を表示する。"""
    from app.services.auto_strategy.services.generated_strategy_service import (
        GeneratedStrategyService,
    )
    from database.connection import SessionLocal

    if not strategy_id.startswith("auto_"):
        strategy_id = f"auto_{strategy_id}"
    db_id = strategy_id.removeprefix("auto_")
    if not db_id.isdigit():
        typer.echo(f"不正な戦略IDです: {strategy_id}", err=True)
        raise typer.Exit(code=1)

    with SessionLocal() as db:
        service = GeneratedStrategyService(db)
        result = service.get_strategies(limit=1000)
        strategies = result["strategies"]
        strategy = next((s for s in strategies if s["id"] == strategy_id), None)

    if strategy is None:
        typer.echo(f"戦略が見つかりません: {strategy_id}", err=True)
        raise typer.Exit(code=1)

    if json_output:
        _print_json(strategy)
    else:
        _print_ga_strategy(strategy)


@data_app.command("fetch")
def data_fetch(
    symbol: Annotated[
        str, typer.Option("--symbol", "-s", help="取引ペア")
    ] = "BTC/USDT:USDT",
    timeframe: Annotated[str, typer.Option("--timeframe", "-t", help="時間足")] = "1h",
    force_update: Annotated[
        bool, typer.Option("--force", "-f", help="既存データを削除して再取得")
    ] = False,
    start_date: Annotated[
        str | None, typer.Option("--start-date", help="収集開始日（YYYY-MM-DD）")
    ] = None,
    json_output: Annotated[
        bool, typer.Option("--json", help="結果をJSONで出力")
    ] = False,
) -> None:
    """OHLCV履歴データを取引所から取得してDBに保存する（同期実行）。"""
    orchestration, _ = build_data_collection_services()
    background_tasks = SynchronousBackgroundTasks()

    from fastapi import BackgroundTasks

    with SessionLocal() as db:
        try:
            result = asyncio_run(
                orchestration.start_historical_data_collection(
                    symbol,
                    timeframe,
                    cast(BackgroundTasks, background_tasks),
                    db,
                    force_update,
                    start_date,
                )
            )
            asyncio_run(background_tasks.run_tasks())
        except (ValueError, RuntimeError) as e:
            typer.echo(f"収集失敗: {e}", err=True)
            raise typer.Exit(code=1) from e

    if json_output:
        _print_json(result)
    else:
        status = result.get("status", "unknown")
        typer.echo(f"[{status}] {result.get('message', '')}")


@data_app.command("update")
def data_update(
    symbol: Annotated[
        str, typer.Option("--symbol", "-s", help="取引ペア")
    ] = "BTC/USDT:USDT",
    json_output: Annotated[
        bool, typer.Option("--json", help="結果をJSONで出力")
    ] = False,
) -> None:
    """DB末尾から最新まで差分更新する（OHLCV・FR・OI一括）。"""
    orchestration, _ = build_data_collection_services()

    with SessionLocal() as db:
        try:
            result = asyncio_run(
                orchestration.execute_bulk_incremental_update(symbol, db)
            )
        except (ValueError, RuntimeError) as e:
            typer.echo(f"更新失敗: {e}", err=True)
            raise typer.Exit(code=1) from e

    if json_output:
        _print_json(result)
    else:
        typer.echo(result.get("message", "更新完了"))


@data_app.command("status")
def data_status(
    symbol: Annotated[
        str, typer.Option("--symbol", "-s", help="取引ペア")
    ] = "BTC/USDT:USDT",
    timeframe: Annotated[str, typer.Option("--timeframe", "-t", help="時間足")] = "1h",
    json_output: Annotated[
        bool, typer.Option("--json", help="結果をJSONで出力")
    ] = False,
) -> None:
    """指定シンボル・時間足の収集状況（件数・範囲）を確認する。"""
    orchestration, _ = build_data_collection_services()
    background_tasks = SynchronousBackgroundTasks()

    from fastapi import BackgroundTasks

    with SessionLocal() as db:
        try:
            result = asyncio_run(
                orchestration.get_collection_status(
                    symbol=symbol,
                    timeframe=timeframe,
                    background_tasks=cast(BackgroundTasks, background_tasks),
                    auto_fetch=False,
                    db=db,
                )
            )
        except (ValueError, RuntimeError) as e:
            typer.echo(f"確認失敗: {e}", err=True)
            raise typer.Exit(code=1) from e

    if json_output:
        _print_json(result)
    else:
        data = result.get("data", {}) if isinstance(result, dict) else {}
        count = data.get("data_count", 0)
        latest = data.get("latest_timestamp")
        oldest = data.get("oldest_timestamp")
        typer.echo(f"{symbol} {timeframe}: {count}件")
        if oldest and latest:
            typer.echo(f"  範囲: {oldest} 〜 {latest}")


@data_app.command("overview")
def data_overview(
    json_output: Annotated[
        bool, typer.Option("--json", help="結果をJSONで出力")
    ] = False,
) -> None:
    """全データ種別（OHLCV・FR・OI）の総件数と範囲を確認する。"""
    _, management = build_data_collection_services()

    try:
        result = asyncio_run(management.get_data_status())
    except (ValueError, RuntimeError) as e:
        typer.echo(f"確認失敗: {e}", err=True)
        raise typer.Exit(code=1) from e

    if json_output:
        _print_json(result)
    else:
        data = result.get("data", {}) if isinstance(result, dict) else {}
        counts = data.get("data_counts", {})
        total = data.get("total_records", 0)
        typer.echo(
            f"OHLCV: {counts.get('ohlcv', 0)}件 / "
            f"FR: {counts.get('funding_rates', 0)}件 / "
            f"OI: {counts.get('open_interest', 0)}件（計{total}件）"
        )


@data_app.command("reset")
def data_reset(
    target: Annotated[
        str,
        typer.Argument(
            help="リセット対象（all | ohlcv | funding-rates | open-interest）"
        ),
    ],
    symbol: Annotated[
        str | None, typer.Option("--symbol", "-s", help="シンボルを限定して削除")
    ] = None,
    yes: Annotated[
        bool, typer.Option("--yes", "-y", help="確認プロンプトをスキップ")
    ] = False,
    json_output: Annotated[
        bool, typer.Option("--json", help="結果をJSONで出力")
    ] = False,
) -> None:
    """DBの市場データを削除する（デフォルトは確認あり）。"""
    if not yes:
        scope = symbol or target
        answer = typer.confirm(f"{scope} のデータを削除します。よろしいですか？")
        if not answer:
            typer.echo("キャンセルしました")
            raise typer.Exit(code=0)

    _, management = build_data_collection_services()

    try:
        if symbol is not None:
            result = asyncio_run(management.reset_data_by_symbol(symbol))
        elif target == "all":
            result = asyncio_run(management.reset_all_data())
        elif target == "ohlcv":
            result = asyncio_run(management.reset_ohlcv_data())
        elif target == "funding-rates":
            result = asyncio_run(management.reset_funding_rate_data())
        elif target == "open-interest":
            result = asyncio_run(management.reset_open_interest_data())
        else:
            typer.echo(
                "対象は all | ohlcv | funding-rates | open-interest のいずれかです",
                err=True,
            )
            raise typer.Exit(code=2)
    except typer.Exit:
        raise
    except (ValueError, RuntimeError) as e:
        typer.echo(f"リセット失敗: {e}", err=True)
        raise typer.Exit(code=1) from e

    if json_output:
        _print_json(result)
    else:
        data = result.get("data", {}) if isinstance(result, dict) else {}
        typer.echo(
            f"{result.get('message', '完了')} "
            f"（削除: {data.get('total_deleted', data.get('deleted_count', '?'))}件）"
        )


def asyncio_run(coro: Any) -> Any:
    """CLI から async サービスを同期実行する共通ヘルパー。

    NOTE: データ収集サービス内部で生成される ccxt exchange は
    close されないままプロセス終了を迎えるため、aiohttp の
    ResourceWarning（Unclosed client session）が出る。
    既知のサービス層リソース管理問題であり、CLI では抑制する。
    """
    import asyncio
    import warnings

    with warnings.catch_warnings():
        warnings.simplefilter("ignore", ResourceWarning)
        return asyncio.run(coro)


if __name__ == "__main__":
    app()
