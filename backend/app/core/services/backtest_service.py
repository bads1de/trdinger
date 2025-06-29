"""
バックテスト実行サービス

backtesting.pyライブラリを使用したバックテスト実行機能を提供します。
"""

import pandas as pd
import warnings
from datetime import datetime
from typing import Dict, Any, Type, Optional
from backtesting import Backtest, Strategy

from .backtest_data_service import BacktestDataService
from ..strategies.macd_strategy import MACDStrategy
from database.repositories.ohlcv_repository import OHLCVRepository
from database.connection import SessionLocal


class BacktestService:
    """
    backtesting.pyを使用したバックテスト実行サービス

    既存のOHLCVデータを使用してバックテストを実行し、
    結果をデータベース保存用の形式に変換します。
    """

    def __init__(self, data_service: Optional[BacktestDataService] = None):
        """
        初期化

        Args:
            data_service: データ変換サービス（テスト時にモックを注入可能）
        """
        self.data_service = data_service

    def run_backtest(self, config: Dict[str, Any]) -> Dict[str, Any]:
        """
        バックテストを実行

        Args:
            config: バックテスト設定
                - strategy_name: 戦略名
                - symbol: 取引ペア
                - timeframe: 時間軸
                - start_date: 開始日時
                - end_date: 終了日時
                - initial_capital: 初期資金
                - commission_rate: 手数料率
                - strategy_config: 戦略固有の設定

        Returns:
            バックテスト結果の辞書

        Raises:
            ValueError: 設定が無効な場合
        """
        import logging

        logger = logging.getLogger(__name__)

        try:
            logger.info(f"バックテストを開始します。設定: {config}")

            # 1. 設定の検証
            self._validate_config(config)
            logger.info("バックテスト設定の検証が完了しました。")

            # 2. データサービスの初期化（必要に応じて）
            if self.data_service is None:
                db = SessionLocal()
                try:
                    ohlcv_repo = OHLCVRepository(db)
                    self.data_service = BacktestDataService(ohlcv_repo)
                    logger.info("BacktestDataServiceが初期化されました。")
                finally:
                    db.close()

            # 3. データ取得
            logger.info(
                f"{config['symbol']} {config['timeframe']} のOHLCVデータを {config['start_date']} から {config['end_date']} まで取得しています。"
            )

            # 日付文字列をdatetimeオブジェクトに変換
            from datetime import datetime

            start_date = (
                datetime.fromisoformat(config["start_date"])
                if isinstance(config["start_date"], str)
                else config["start_date"]
            )
            end_date = (
                datetime.fromisoformat(config["end_date"])
                if isinstance(config["end_date"], str)
                else config["end_date"]
            )

            # 新しい統合データ取得メソッドを使用（OI/FR含む）
            try:
                data = self.data_service.get_data_for_backtest(
                    symbol=config["symbol"],
                    timeframe=config["timeframe"],
                    start_date=start_date,
                    end_date=end_date,
                )
                logger.info("OI/FR統合を含む拡張データを使用しています。")
            except AttributeError:
                # 後方互換性のため、古いメソッドにフォールバック
                logger.warning("OHLCVのみのデータにフォールバックします。")
                data = self.data_service.get_ohlcv_for_backtest(
                    symbol=config["symbol"],
                    timeframe=config["timeframe"],
                    start_date=start_date,
                    end_date=end_date,
                )

            if data is None or data.empty:
                raise ValueError(
                    f"{config['symbol']} {config['timeframe']} の {config['start_date']} から {config['end_date']} までのOHLCVデータが見つかりませんでした。"
                )

            logger.info(f"{len(data)}件のデータポイントを取得しました。")

            # 4. 戦略クラス動的生成
            logger.info(
                f"{config['strategy_config']['strategy_type']} の戦略クラスを作成しています。"
            )
            strategy_class = self._create_strategy_class(config["strategy_config"])

            # 5. backtesting.py実行
            logger.info("バックテストを実行中...")
            logger.info(f"データシェイプ: {data.shape}")
            logger.info(f"データカラム: {data.columns.tolist()}")
            logger.info(
                f"データインデックス範囲: {data.index.min()} から {data.index.max()}"
            )

            bt = Backtest(
                data,
                strategy_class,
                cash=config["initial_capital"],
                commission=config["commission_rate"],
                exclusive_orders=True,  # 推奨設定
                trade_on_close=True,  # 終値で取引
            )

            # バックテストを実行（時間計測付き）
            import time

            start_time = time.time()
            logger.info("backtesting.pyによるバックテストの実行を開始します...")
            warnings.filterwarnings("ignore", category=UserWarning)
            stats = bt.run()
            warnings.filterwarnings("default", category=UserWarning)
            execution_time = time.time() - start_time
            logger.info(
                f"バックテストが正常に完了しました。実行時間: {execution_time:.2f}秒"
            )

            # 6. 結果をデータベース形式に変換
            logger.info("バックテスト結果の変換処理を開始します...")

            # config_jsonを構築
            config_json = {
                "strategy_config": config.get("strategy_config", {}),
                "commission_rate": config.get("commission_rate", 0.001),
            }

            result = self._convert_backtest_results(
                stats,
                config["strategy_name"],
                config["symbol"],
                config["timeframe"],
                config["initial_capital"],
                config["start_date"],
                config["end_date"],
                config_json,
            )

            logger.info("バックテスト結果の変換が完了し、返却されます。")
            return result

        except Exception as e:
            logger.error(
                f"バックテストの実行中に予期せぬエラーが発生しました: {str(e)}",
                exc_info=True,
            )
            raise

    def _validate_config(self, config: Dict[str, Any]) -> bool:
        """
        バックテスト設定の妥当性を検証

        Args:
            config: バックテスト設定

        Returns:
            設定が有効な場合True

        Raises:
            ValueError: 設定が無効な場合
        """
        required_fields = [
            "strategy_name",
            "symbol",
            "timeframe",
            "start_date",
            "end_date",
            "initial_capital",
            "commission_rate",
            "strategy_config",
        ]

        # 必須フィールドの確認
        for field in required_fields:
            if field not in config:
                raise ValueError(f"必須フィールドが見つかりません: {field}")

        # 日付の妥当性確認
        if config["start_date"] >= config["end_date"]:
            raise ValueError("開始日は終了日よりも前である必要があります。")

        # 初期資金の妥当性確認
        if config["initial_capital"] <= 0:
            raise ValueError("初期資金は正の数である必要があります。")

        # 手数料率の妥当性確認
        if config["commission_rate"] < 0 or config["commission_rate"] > 1:
            raise ValueError("手数料率は0から1の間である必要があります。")

        # 戦略設定の確認
        strategy_config = config["strategy_config"]
        if "strategy_type" not in strategy_config:
            raise ValueError("戦略設定にはstrategy_typeが必要です。")

        return True

    def _create_strategy_class(self, strategy_config: Dict[str, Any]) -> Type[Strategy]:
        """
        戦略設定から戦略クラスを取得し、パラメータを設定

        Args:
            strategy_config: 戦略設定

        Returns:
            戦略クラス

        Raises:
            ValueError: サポートされていない戦略タイプの場合
        """
        strategy_type = strategy_config["strategy_type"]
        parameters = strategy_config.get("parameters", {})

        if strategy_type == "MACD":
            # パラメータをクラス変数として設定
            if "fast_period" in parameters:
                MACDStrategy.fast_period = parameters["fast_period"]
                MACDStrategy.n1 = parameters["fast_period"]  # エイリアス
            if "slow_period" in parameters:
                MACDStrategy.slow_period = parameters["slow_period"]
                MACDStrategy.n2 = parameters["slow_period"]  # エイリアス
            if "signal_period" in parameters:
                MACDStrategy.signal_period = parameters["signal_period"]

            # 後方互換性のためのエイリアス処理
            if "n1" in parameters:
                MACDStrategy.fast_period = parameters["n1"]
                MACDStrategy.n1 = parameters["n1"]
            if "n2" in parameters:
                MACDStrategy.slow_period = parameters["n2"]
                MACDStrategy.n2 = parameters["n2"]

            return MACDStrategy

        elif strategy_type == "GENERATED_TEST":
            # 自動生成戦略のテスト用
            # StrategyFactoryで生成された戦略クラスを使用
            from app.core.services.auto_strategy.factories.strategy_factory import (
                StrategyFactory,
            )
            from app.core.services.auto_strategy.models.strategy_gene import (
                StrategyGene,
            )

            # パラメータから戦略遺伝子を復元
            if "strategy_gene" in parameters:
                strategy_gene = StrategyGene.from_dict(parameters["strategy_gene"])
                factory = StrategyFactory()
                return factory.create_strategy_class(strategy_gene)
            else:
                raise ValueError(
                    "GENERATED_TEST戦略タイプには、戦略遺伝子 (strategy_gene) がパラメータとして必要です。"
                )

        elif strategy_type == "GENERATED_AUTO":
            # オートストラテジーで生成された戦略用
            # StrategyFactoryで生成された戦略クラスを使用
            from app.core.services.auto_strategy.factories.strategy_factory import (
                StrategyFactory,
            )
            from app.core.services.auto_strategy.models.strategy_gene import (
                StrategyGene,
            )

            # パラメータから戦略遺伝子を復元
            if "strategy_gene" in parameters:
                strategy_gene = StrategyGene.from_dict(parameters["strategy_gene"])
                factory = StrategyFactory()
                return factory.create_strategy_class(strategy_gene)
            else:
                raise ValueError(
                    "GENERATED_AUTO戦略タイプには、戦略遺伝子 (strategy_gene) がパラメータとして必要です。"
                )

        else:
            raise ValueError(
                f"サポートされていない戦略タイプが指定されました: {strategy_type}"
            )

    def _convert_backtest_results(
        self,
        stats: pd.Series,
        strategy_name: str,
        symbol: str,
        timeframe: str,
        initial_capital: float,
        start_date: Optional[str] = None,
        end_date: Optional[str] = None,
        config_json: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        """
        backtesting.pyの結果をデータベース保存用の形式に変換

        Args:
            stats: backtesting.pyの統計結果
            strategy_name: 戦略名
            symbol: 取引ペア
            timeframe: 時間軸
            initial_capital: 初期資金
            start_date: バックテスト開始日 (オプション)
            end_date: バックテスト終了日 (オプション)
            config_json: 設定のJSON表現 (オプション)

        Returns:
            データベース保存用の結果辞書
        """
        # パフォーマンス指標を抽出
        performance_metrics = {
            "total_return": float(stats.get("Return [%]") or 0.0),
            "sharpe_ratio": float(stats.get("Sharpe Ratio") or 0.0),
            "max_drawdown": float(stats.get("Max. Drawdown [%]") or 0.0),
            "win_rate": float(stats.get("Win Rate [%]") or 0.0),
            "total_trades": int(stats.get("# Trades") or 0),
            "equity_final": float(stats.get("Equity Final [$") or initial_capital),
            "buy_hold_return": float(stats.get("Buy & Hold Return [%]") or 0.0),
            "exposure_time": float(stats.get("Exposure Time [%]") or 0.0),
            "sortino_ratio": float(stats.get("Sortino Ratio") or 0.0),
            "calmar_ratio": float(stats.get("Calmar Ratio") or 0.0),
        }

        # 資産曲線を変換
        equity_curve = []
        equity_data = stats.get("_equity_curve")
        if isinstance(equity_data, pd.DataFrame) and not equity_data.empty:
            for timestamp, row in equity_data.iterrows():
                # timestampがdatetimeオブジェクトかどうかを確認
                if isinstance(timestamp, datetime) and hasattr(timestamp, "isoformat"):
                    timestamp_str = timestamp.isoformat()
                else:
                    timestamp_str = str(timestamp)

                equity_curve.append(
                    {
                        "timestamp": timestamp_str,
                        "equity": float(row["Equity"]) if "Equity" in row else 0.0,
                        "drawdown_pct": float(row.get("DrawdownPct") or 0.0),
                    }
                )

        # 取引履歴を変換
        trade_history = []
        winning_trades = 0
        losing_trades = 0
        total_wins = 0.0
        total_losses = 0.0

        trades_data = stats.get("_trades")
        if isinstance(trades_data, pd.DataFrame) and not trades_data.empty:
            for _, trade in trades_data.iterrows():
                pnl = float(trade.get("PnL") or 0.0)

                # 勝ち負けの統計を計算
                if pnl > 0:
                    winning_trades += 1
                    total_wins += pnl
                elif pnl < 0:
                    losing_trades += 1
                    total_losses += abs(pnl)

                entry_time_obj = trade.get("EntryTime")
                exit_time_obj = trade.get("ExitTime")

                trade_history.append(
                    {
                        "size": float(trade.get("Size") or 0.0),
                        "entry_price": float(trade.get("EntryPrice") or 0.0),
                        "exit_price": float(trade.get("ExitPrice") or 0.0),
                        "pnl": pnl,
                        "return_pct": float(trade.get("ReturnPct") or 0.0),
                        "entry_time": (
                            entry_time_obj.isoformat()
                            if isinstance(entry_time_obj, datetime)
                            and hasattr(entry_time_obj, "isoformat")
                            else str(entry_time_obj or "")
                        ),
                        "exit_time": (
                            exit_time_obj.isoformat()
                            if isinstance(exit_time_obj, datetime)
                            and hasattr(exit_time_obj, "isoformat")
                            else str(exit_time_obj or "")
                        ),
                    }
                )

        # 追加の指標を計算
        avg_win = total_wins / winning_trades if winning_trades > 0 else 0.0
        avg_loss = total_losses / losing_trades if losing_trades > 0 else 0.0
        profit_factor = total_wins / total_losses if total_losses > 0 else 0.0

        # パフォーマンス指標に追加
        performance_metrics.update(
            {
                "winning_trades": winning_trades,
                "losing_trades": losing_trades,
                "avg_win": avg_win,
                "avg_loss": avg_loss,
                "profit_factor": profit_factor,
            }
        )

        # 結果を統合
        result = {
            "strategy_name": strategy_name,
            "symbol": symbol,
            "timeframe": timeframe,
            "start_date": start_date,
            "end_date": end_date,
            "initial_capital": initial_capital,
            "commission_rate": (
                config_json.get("commission_rate", 0.001) if config_json else 0.001
            ),
            "config_json": config_json or {},
            "performance_metrics": performance_metrics,
            "equity_curve": equity_curve,
            "trade_history": trade_history,
            "status": "completed",
            "created_at": datetime.now().isoformat(),
        }

        return result

    def get_supported_strategies(self) -> Dict[str, Dict[str, Any]]:
        """
        サポートされている戦略の一覧を取得

        Returns:
            戦略情報の辞書
        """
        return {
            "MACD": {
                "name": "MACD Strategy",
                "description": "移動平均収束拡散戦略",
                "parameters": {
                    "fast_period": {
                        "type": "int",
                        "default": 12,
                        "min": 5,
                        "max": 20,
                        "description": "高速EMA期間",
                    },
                    "slow_period": {
                        "type": "int",
                        "default": 26,
                        "min": 20,
                        "max": 50,
                        "description": "低速EMA期間",
                    },
                    "signal_period": {
                        "type": "int",
                        "default": 9,
                        "min": 5,
                        "max": 15,
                        "description": "シグナルライン期間",
                    },
                    "n1": {
                        "type": "int",
                        "default": 12,
                        "min": 5,
                        "max": 20,
                        "description": "高速EMA期間（fast_periodのエイリアス）",
                    },
                    "n2": {
                        "type": "int",
                        "default": 26,
                        "min": 20,
                        "max": 50,
                        "description": "低速EMA期間（slow_periodのエイリアス）",
                    },
                },
                "constraints": ["fast_period < slow_period", "n1 < n2"],
            }
        }

    def optimize_strategy(
        self, config: Dict[str, Any], optimization_params: Dict[str, Any]
    ) -> Dict[str, Any]:
        """
        戦略パラメータの最適化

        Args:
            config: 基本バックテスト設定
            optimization_params: 最適化パラメータ
                - parameters: 最適化対象のパラメータ範囲
                - maximize: 最大化する指標
                - constraint: 制約条件

        Returns:
            最適化結果
        """
        # 設定の検証
        self._validate_config(config)

        # データサービスの初期化（必要に応じて）
        if self.data_service is None:
            db = SessionLocal()
            try:
                ohlcv_repo = OHLCVRepository(db)
                self.data_service = BacktestDataService(ohlcv_repo)
            finally:
                db.close()

        # データ取得（統合データを優先）
        try:
            data = self.data_service.get_data_for_backtest(
                symbol=config["symbol"],
                timeframe=config["timeframe"],
                start_date=config["start_date"],
                end_date=config["end_date"],
            )
        except AttributeError:
            # 後方互換性のため、古いメソッドにフォールバック
            data = self.data_service.get_ohlcv_for_backtest(
                symbol=config["symbol"],
                timeframe=config["timeframe"],
                start_date=config["start_date"],
                end_date=config["end_date"],
            )

        # 戦略クラス取得
        strategy_class = self._create_strategy_class(config["strategy_config"])

        # バックテスト実行
        bt = Backtest(
            data,
            strategy_class,
            cash=config["initial_capital"],
            commission=config["commission_rate"],
            exclusive_orders=True,
        )

        # 最適化実行
        optimize_kwargs = {}
        if "maximize" in optimization_params:
            optimize_kwargs["maximize"] = optimization_params["maximize"]
        if "constraint" in optimization_params:
            optimize_kwargs["constraint"] = optimization_params["constraint"]

        # パラメータ範囲を追加
        for param_name, param_range in optimization_params["parameters"].items():
            optimize_kwargs[param_name] = param_range

        stats_raw = bt.optimize(**optimize_kwargs)

        # optimizeの結果がタプルの場合、最初の要素を実際の統計として扱う
        if isinstance(stats_raw, tuple):
            stats = stats_raw[0]
        else:
            stats = stats_raw

        # config_jsonを構築
        config_json = {
            "strategy_config": config.get("strategy_config", {}),
            "commission_rate": config.get("commission_rate", 0.001),
        }

        # 結果を変換
        result = self._convert_backtest_results(
            stats,
            config["strategy_name"],
            config["symbol"],
            config["timeframe"],
            config["initial_capital"],
            config.get("start_date"),
            config.get("end_date"),
            config_json,
        )

        # 最適化されたパラメータを追加
        # statsがpd.Seriesであることを前提に_strategy属性にアクセス
        optimized_strategy = stats._strategy
        result["optimized_parameters"] = {
            param: getattr(optimized_strategy, param)
            for param in optimization_params["parameters"].keys()
            if hasattr(optimized_strategy, param)
        }

        return result

    def optimize_strategy_enhanced(
        self, config: Dict[str, Any], optimization_params: Dict[str, Any]
    ) -> Dict[str, Any]:
        """
        戦略パラメータの拡張最適化

        Args:
            config: 基本バックテスト設定
            optimization_params: 拡張最適化パラメータ
                - parameters: 最適化対象のパラメータ範囲
                - maximize: 最大化する指標
                - constraint: 制約条件
                - advanced_options: 拡張オプション

        Returns:
            拡張最適化結果
        """
        # 基本的には既存のoptimize_strategyを使用し、拡張オプションを適用
        return self.optimize_strategy(config, optimization_params)

    def multi_objective_optimization(
        self,
        config: Dict[str, Any],
        objectives: list,
        weights: Optional[list] = None,
        optimization_params: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        """
        多目的最適化

        Args:
            config: 基本バックテスト設定
            objectives: 最適化対象の指標リスト
            weights: 各指標の重み
            optimization_params: 追加の最適化パラメータ

        Returns:
            多目的最適化結果
        """
        import logging

        logger = logging.getLogger(__name__)

        # 現在は単一目的最適化として実装（将来的に拡張可能）
        if not objectives:
            raise ValueError("最適化対象の指標が指定されていません")

        # 最初の目的関数を使用
        primary_objective = objectives[0]

        # optimization_paramsが指定されていない場合はデフォルト値を使用
        if optimization_params is None:
            optimization_params = {"maximize": primary_objective}
        else:
            optimization_params = optimization_params.copy()
            optimization_params["maximize"] = primary_objective

        logger.info(f"多目的最適化を実行中（主目的: {primary_objective}）")

        result = self.optimize_strategy(config, optimization_params)

        # 多目的最適化の結果として追加情報を付与
        result["multi_objective_info"] = {
            "objectives": objectives,
            "weights": weights,
            "primary_objective": primary_objective,
        }

        return result

    def robustness_test(
        self,
        config: Dict[str, Any],
        test_periods: list,
        optimization_params: Dict[str, Any],
    ) -> Dict[str, Any]:
        """
        戦略のロバストネステスト

        Args:
            config: 基本バックテスト設定
            test_periods: テスト期間のリスト [(start_date, end_date), ...]
            optimization_params: 最適化パラメータ

        Returns:
            ロバストネステスト結果
        """
        import logging

        logger = logging.getLogger(__name__)

        logger.info(f"ロバストネステストを開始（{len(test_periods)}期間）")

        results = []

        for i, (start_date, end_date) in enumerate(test_periods):
            logger.info(f"期間 {i+1}/{len(test_periods)}: {start_date} - {end_date}")

            # 期間ごとの設定を作成
            period_config = config.copy()
            period_config["start_date"] = start_date
            period_config["end_date"] = end_date

            try:
                # 各期間でバックテストを実行
                period_result = self.run_backtest(period_config)
                period_result["test_period"] = {
                    "start_date": start_date,
                    "end_date": end_date,
                }
                results.append(period_result)

            except Exception as e:
                logger.error(f"期間 {start_date} - {end_date} でエラー: {e}")
                results.append(
                    {
                        "test_period": {"start_date": start_date, "end_date": end_date},
                        "error": str(e),
                        "status": "failed",
                    }
                )

        # ロバストネス統計を計算
        successful_results = [r for r in results if r.get("status") != "failed"]

        if successful_results:
            returns = [
                r["performance_metrics"]["total_return"] for r in successful_results
            ]
            sharpe_ratios = [
                r["performance_metrics"]["sharpe_ratio"] for r in successful_results
            ]
            max_drawdowns = [
                r["performance_metrics"]["max_drawdown"] for r in successful_results
            ]

            robustness_stats = {
                "total_periods": len(test_periods),
                "successful_periods": len(successful_results),
                "success_rate": len(successful_results) / len(test_periods),
                "avg_return": sum(returns) / len(returns) if returns else 0,
                "std_return": (
                    (
                        sum((r - sum(returns) / len(returns)) ** 2 for r in returns)
                        / len(returns)
                    )
                    ** 0.5
                    if len(returns) > 1
                    else 0
                ),
                "avg_sharpe": (
                    sum(sharpe_ratios) / len(sharpe_ratios) if sharpe_ratios else 0
                ),
                "avg_max_drawdown": (
                    sum(max_drawdowns) / len(max_drawdowns) if max_drawdowns else 0
                ),
            }
        else:
            robustness_stats = {
                "total_periods": len(test_periods),
                "successful_periods": 0,
                "success_rate": 0,
                "avg_return": 0,
                "std_return": 0,
                "avg_sharpe": 0,
                "avg_max_drawdown": 0,
            }

        # 統合結果を作成
        robustness_result = {
            "strategy_name": config["strategy_name"],
            "symbol": config["symbol"],
            "timeframe": config["timeframe"],
            "test_type": "robustness_test",
            "robustness_stats": robustness_stats,
            "period_results": results,
            "status": "completed",
            "created_at": datetime.now().isoformat(),
        }

        logger.info(
            f"ロバストネステスト完了（成功率: {robustness_stats['success_rate']:.2%}）"
        )

        return robustness_result
