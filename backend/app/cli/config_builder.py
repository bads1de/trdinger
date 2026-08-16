"""
CLI とレガシースクリプトで共用する GA / BT 設定ビルダー。

Trdinger の設定構築ロジックを一本化するためのモジュール。
`trdinger exp run` と `scripts.run_auto_strategy` の両方が
このモジュールの関数を使用して設定を構築する。

設定の流れ:
    CLI引数 / argparse Namespace
        -> normalize_*_options()   (入力検証・smokeモード反映)
        -> build_ga_config_dict() / build_backtest_config_dict()
        -> GAConfig.from_dict() / dict
"""

from __future__ import annotations

from typing import Any

from app.services.auto_strategy.config.ga_config import GAConfig

DEFAULT_SYMBOL = "BTC/USDT:USDT"
DEFAULT_TIMEFRAME = "4h"
DEFAULT_START_DATE = "2024-01-01"
DEFAULT_END_DATE = "2024-06-30"
DEFAULT_INITIAL_CAPITAL = 100000.0
DEFAULT_COMMISSION_RATE = 0.0004
DEFAULT_SLIPPAGE = 0.0001


def normalize_run_options(
    *,
    population: int,
    generations: int,
    elite_size: int,
    crossover_rate: float,
    mutation_rate: float,
    initial_capital: float,
    mtf_probability: float | None = None,
    smoke: bool = False,
) -> dict[str, Any]:
    """
    実行オプションを検証し、smoke モードを反映した正規化値を返す。

    Args:
        population: 個体数
        generations: 世代数
        elite_size: エリート保存数
        crossover_rate: 交叉率
        mutation_rate: 突然変異率
        initial_capital: 初期資本
        mtf_probability: MTF指標生成確率（None ならチェックしない）
        smoke: smoke モードなら最小構成へ縮小する

    Returns:
        正規化済みオプションの辞書

    Raises:
        ValueError: 入力が不正な場合
    """
    normalized: dict[str, Any] = {
        "population": 2 if smoke else population,
        "generations": 1 if smoke else generations,
        "elite_size": 1 if smoke else elite_size,
        "crossover_rate": crossover_rate,
        "mutation_rate": mutation_rate,
        "initial_capital": initial_capital,
    }

    if normalized["population"] < 2:
        raise ValueError("個体数は2以上である必要があります")
    if normalized["generations"] < 1:
        raise ValueError("世代数は1以上である必要があります")
    if normalized["elite_size"] < 0:
        raise ValueError("エリート保存数は0以上である必要があります")
    if normalized["elite_size"] >= normalized["population"]:
        raise ValueError("エリート保存数は個体数未満である必要があります")
    if not 0 <= crossover_rate <= 1:
        raise ValueError("交叉率は0から1の範囲である必要があります")
    if not 0 <= mutation_rate <= 1:
        raise ValueError("突然変異率は0から1の範囲である必要があります")
    if initial_capital <= 0:
        raise ValueError("初期資本は0より大きい必要があります")
    if mtf_probability is not None and not 0.0 <= mtf_probability <= 1.0:
        raise ValueError("MTF指標確率は0から1の範囲である必要があります")

    return normalized


def build_ga_config_dict(
    *,
    population: int,
    generations: int,
    crossover_rate: float,
    mutation_rate: float,
    elite_size: int,
    start_date: str,
    end_date: str,
    no_parallel: bool = False,
    verbose: bool = False,
    smoke: bool = False,
    min_trades: int | None = None,
    no_validation: bool = False,
    no_seeds: bool = False,
    mtf: bool = False,
    mtf_timeframes: str = "1d",
    mtf_probability: float = 0.3,
    indicator_universe: str = "curated",
    max_indicators: int = 10,
    min_non_price: int = 0,
    non_price_probability: float = 0.3,
    max_conditions: int = 3,
) -> dict[str, Any]:
    """
    実行オプションから GA 設定辞書を構築する。

    `GAConfig.from_dict()` に渡せる辞書を返す。
    検証エラーは ValueError で通知する。

    Args:
        population: 個体数
        generations: 世代数
        crossover_rate: 交叉率
        mutation_rate: 突然変異率
        elite_size: エリート保存数
        start_date: バックテスト開始日
        end_date: バックテスト終了日
        no_parallel: 並列評価を無効化
        verbose: 詳細ログ
        smoke: 高速スモークモード（最小構成へ縮小）
        min_trades: 最小取引回数制約（None はデフォルト、0 で無効）
        no_validation: WFA 自動検証を無効化
        no_seeds: シード戦略注入を無効化
        mtf: マルチタイムフレーム指標を有効化
        mtf_timeframes: MTF タイムフレーム（カンマ区切り）
        mtf_probability: MTF 指標生成確率
        indicator_universe: インジケーターユニバース
        max_indicators: 1戦略あたりの最大インジケーター数
        min_non_price: 非価格指標の最低数
        non_price_probability: 非価格指標の選択確率
        max_conditions: エントリー条件の最大数

    Returns:
        GA 設定辞書

    Raises:
        ValueError: 入力が不正な場合
    """
    normalized = normalize_run_options(
        population=population,
        generations=generations,
        elite_size=elite_size,
        crossover_rate=crossover_rate,
        mutation_rate=mutation_rate,
        initial_capital=DEFAULT_INITIAL_CAPITAL,
        mtf_probability=mtf_probability if mtf else None,
        smoke=smoke,
    )

    # fitness_constraints の上書き
    fitness_constraints: dict[str, Any] | None = None
    if min_trades is not None or smoke:
        from app.services.auto_strategy.config.constants import (
            DEFAULT_FITNESS_CONSTRAINTS,
        )

        fitness_constraints = DEFAULT_FITNESS_CONSTRAINTS.copy()
        fitness_constraints["min_trades"] = min_trades if min_trades is not None else 0

    config_kwargs: dict[str, Any] = {
        "population_size": normalized["population"],
        "generations": normalized["generations"],
        "crossover_rate": crossover_rate,
        "mutation_rate": mutation_rate,
        "elite_size": normalized["elite_size"],
        "evaluation_config": {
            "enable_parallel": not no_parallel and not smoke,
        },
        "log_level": "DEBUG" if verbose else "INFO",
        "fallback_start_date": start_date,
        "fallback_end_date": end_date,
    }

    if smoke:
        config_kwargs["max_indicators"] = 3
        config_kwargs["max_conditions"] = 2
        config_kwargs["use_seed_strategies"] = False
        config_kwargs["seed_injection_rate"] = 0.0
        config_kwargs["two_stage_selection_config"] = {"enabled": False}
    else:
        config_kwargs["max_indicators"] = max_indicators
        config_kwargs["max_conditions"] = max_conditions
        if min_non_price:
            config_kwargs["min_non_price_indicators"] = min_non_price
        if no_seeds:
            config_kwargs["use_seed_strategies"] = False
            config_kwargs["seed_injection_rate"] = 0.0
        config_kwargs["non_price_indicator_probability"] = non_price_probability

        if mtf:
            timeframes = [tf.strip() for tf in mtf_timeframes.split(",") if tf.strip()]
            from app.config.constants import SUPPORTED_TIMEFRAMES

            invalid_tfs = [tf for tf in timeframes if tf not in SUPPORTED_TIMEFRAMES]
            if invalid_tfs:
                raise ValueError(
                    "サポートされていないタイムフレーム: "
                    f"{', '.join(invalid_tfs)}. 有効: {SUPPORTED_TIMEFRAMES}"
                )
            config_kwargs["enable_multi_timeframe"] = True
            config_kwargs["available_timeframes"] = timeframes
            config_kwargs["mtf_indicator_probability"] = mtf_probability

        if indicator_universe != "curated":
            config_kwargs["indicator_universe_mode"] = indicator_universe

    if smoke or no_validation:
        config_kwargs["validation_config"] = {"enabled": False}
    if fitness_constraints is not None:
        config_kwargs["fitness_constraints"] = fitness_constraints

    return config_kwargs


def build_backtest_config_dict(
    *,
    symbol: str = DEFAULT_SYMBOL,
    timeframe: str = DEFAULT_TIMEFRAME,
    start_date: str = DEFAULT_START_DATE,
    end_date: str = DEFAULT_END_DATE,
    initial_capital: float = DEFAULT_INITIAL_CAPITAL,
    commission_rate: float = DEFAULT_COMMISSION_RATE,
    slippage: float = DEFAULT_SLIPPAGE,
) -> dict[str, Any]:
    """
    実行オプションからバックテスト設定辞書を構築する。

    Raises:
        ValueError: 初期資本が不正な場合
    """
    if initial_capital <= 0:
        raise ValueError("初期資本は0より大きい必要があります")

    return {
        "symbol": symbol,
        "timeframe": timeframe,
        "start_date": start_date,
        "end_date": end_date,
        "initial_capital": initial_capital,
        "commission_rate": commission_rate,
        "slippage": slippage,
    }


def build_ga_config(**kwargs: Any) -> GAConfig:
    """設定辞書から GAConfig インスタンスを構築する。"""
    return GAConfig.from_dict(build_ga_config_dict(**kwargs))
