"""
バックテスト実行サービス

backtesting.pyライブラリを使用したバックテスト実行機能を提供します。
"""

import pandas as pd
from datetime import datetime
from typing import Dict, Any, Type, Optional
from backtesting import Backtest, Strategy

from .backtest_data_service import BacktestDataService
from ..strategies.sma_cross_strategy import SMACrossStrategy
from ..strategies.rsi_strategy import RSIStrategy
from ..strategies.macd_strategy import MACDStrategy
from ..strategies.sma_rsi_strategy import SMARSIStrategy
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
            logger.info(f"Starting backtest with config: {config}")

            # 1. 設定の検証
            self._validate_config(config)
            logger.info("Config validation passed")

            # 2. データサービスの初期化（必要に応じて）
            if self.data_service is None:
                db = SessionLocal()
                try:
                    ohlcv_repo = OHLCVRepository(db)
                    self.data_service = BacktestDataService(ohlcv_repo)
                    logger.info("Data service initialized")
                finally:
                    db.close()

            # 3. データ取得
            logger.info(
                f"Fetching OHLCV data for {config['symbol']} {config['timeframe']} from {config['start_date']} to {config['end_date']}"
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
                logger.info("Using extended data with OI/FR integration")
            except AttributeError:
                # 後方互換性のため、古いメソッドにフォールバック
                logger.warning("Falling back to OHLCV-only data")
                data = self.data_service.get_ohlcv_for_backtest(
                    symbol=config["symbol"],
                    timeframe=config["timeframe"],
                    start_date=start_date,
                    end_date=end_date,
                )

            if data is None or data.empty:
                raise ValueError(
                    f"No OHLCV data found for {config['symbol']} {config['timeframe']} from {config['start_date']} to {config['end_date']}"
                )

            logger.info(f"Retrieved {len(data)} data points")

            # 4. 戦略クラス動的生成
            logger.info(
                f"Creating strategy class for {config['strategy_config']['strategy_type']}"
            )
            strategy_class = self._create_strategy_class(config["strategy_config"])

            # 5. backtesting.py実行
            logger.info("Running backtest...")
            logger.info(f"Data shape: {data.shape}")
            logger.info(f"Data columns: {data.columns.tolist()}")
            logger.info(f"Data index range: {data.index.min()} to {data.index.max()}")

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
            logger.info("Starting backtest execution...")
            stats = bt.run()
            execution_time = time.time() - start_time
            logger.info(
                f"Backtest completed successfully in {execution_time:.2f} seconds"
            )

            # 6. 結果をデータベース形式に変換
            logger.info("Converting results...")

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

            logger.info("Backtest result conversion completed")
            return result

        except Exception as e:
            logger.error(f"Backtest failed: {str(e)}", exc_info=True)
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
                raise ValueError(f"Missing required field: {field}")

        # 日付の妥当性確認
        if config["start_date"] >= config["end_date"]:
            raise ValueError("Start date must be before end date")

        # 初期資金の妥当性確認
        if config["initial_capital"] <= 0:
            raise ValueError("Initial capital must be positive")

        # 手数料率の妥当性確認
        if config["commission_rate"] < 0 or config["commission_rate"] > 1:
            raise ValueError("Commission rate must be between 0 and 1")

        # 戦略設定の確認
        strategy_config = config["strategy_config"]
        if "strategy_type" not in strategy_config:
            raise ValueError("Strategy type is required in strategy_config")

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

        if strategy_type == "SMA_CROSS":
            # パラメータをクラス変数として設定
            if "n1" in parameters:
                SMACrossStrategy.n1 = parameters["n1"]
            if "n2" in parameters:
                SMACrossStrategy.n2 = parameters["n2"]
            return SMACrossStrategy

        elif strategy_type == "RSI":
            # パラメータをクラス変数として設定
            if "period" in parameters:
                RSIStrategy.period = parameters["period"]
            if "oversold" in parameters:
                RSIStrategy.oversold = parameters["oversold"]
            if "overbought" in parameters:
                RSIStrategy.overbought = parameters["overbought"]
            return RSIStrategy

        elif strategy_type == "MACD":
            # パラメータをクラス変数として設定
            if "fast_period" in parameters:
                MACDStrategy.fast_period = parameters["fast_period"]
            if "slow_period" in parameters:
                MACDStrategy.slow_period = parameters["slow_period"]
            if "signal_period" in parameters:
                MACDStrategy.signal_period = parameters["signal_period"]
            return MACDStrategy

        elif strategy_type == "SMA_RSI":
            # パラメータをクラス変数として設定
            if "sma_short" in parameters:
                SMARSIStrategy.sma_short = parameters["sma_short"]
            if "sma_long" in parameters:
                SMARSIStrategy.sma_long = parameters["sma_long"]
            if "rsi_period" in parameters:
                SMARSIStrategy.rsi_period = parameters["rsi_period"]
            if "oversold_threshold" in parameters:
                SMARSIStrategy.oversold_threshold = parameters["oversold_threshold"]
            if "overbought_threshold" in parameters:
                SMARSIStrategy.overbought_threshold = parameters["overbought_threshold"]
            if "use_risk_management" in parameters:
                SMARSIStrategy.use_risk_management = parameters["use_risk_management"]
            if "sl_pct" in parameters:
                SMARSIStrategy.sl_pct = parameters["sl_pct"]
            if "tp_pct" in parameters:
                SMARSIStrategy.tp_pct = parameters["tp_pct"]
            return SMARSIStrategy

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
                    "strategy_gene is required for GENERATED_TEST strategy type"
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
                    "strategy_gene is required for GENERATED_AUTO strategy type"
                )

        else:
            raise ValueError(f"Unsupported strategy type: {strategy_type}")

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

        Returns:
            データベース保存用の結果辞書
        """
        # パフォーマンス指標を抽出
        performance_metrics = {
            "total_return": float(stats.get("Return [%]", 0)),
            "sharpe_ratio": float(stats.get("Sharpe Ratio", 0)),
            "max_drawdown": float(stats.get("Max. Drawdown [%]", 0)),
            "win_rate": float(stats.get("Win Rate [%]", 0)),
            "total_trades": int(stats.get("# Trades", 0)),
            "equity_final": float(stats.get("Equity Final [$]", initial_capital)),
            "buy_hold_return": float(stats.get("Buy & Hold Return [%]", 0)),
            "exposure_time": float(stats.get("Exposure Time [%]", 0)),
            "sortino_ratio": float(stats.get("Sortino Ratio", 0)),
            "calmar_ratio": float(stats.get("Calmar Ratio", 0)),
        }

        # 資産曲線を変換
        equity_curve = []
        if "_equity_curve" in stats and not stats["_equity_curve"].empty:
            equity_df = stats["_equity_curve"]
            for timestamp, row in equity_df.iterrows():
                # timestampがdatetimeオブジェクトかどうかを確認
                if hasattr(timestamp, "isoformat"):
                    timestamp_str = timestamp.isoformat()
                else:
                    timestamp_str = str(timestamp)

                equity_curve.append(
                    {
                        "timestamp": timestamp_str,
                        "equity": float(row["Equity"]),
                        "drawdown_pct": float(row.get("DrawdownPct", 0)),
                    }
                )

        # 取引履歴を変換
        trade_history = []
        winning_trades = 0
        losing_trades = 0
        total_wins = 0.0
        total_losses = 0.0

        if "_trades" in stats and not stats["_trades"].empty:
            trades_df = stats["_trades"]
            for _, trade in trades_df.iterrows():
                pnl = float(trade.get("PnL", 0))

                # 勝ち負けの統計を計算
                if pnl > 0:
                    winning_trades += 1
                    total_wins += pnl
                elif pnl < 0:
                    losing_trades += 1
                    total_losses += abs(pnl)

                trade_history.append(
                    {
                        "size": float(trade.get("Size", 0)),
                        "entry_price": float(trade.get("EntryPrice", 0)),
                        "exit_price": float(trade.get("ExitPrice", 0)),
                        "pnl": pnl,
                        "return_pct": float(trade.get("ReturnPct", 0)),
                        "entry_time": (
                            trade.get("EntryTime", "").isoformat()
                            if hasattr(trade.get("EntryTime", ""), "isoformat")
                            else str(trade.get("EntryTime", ""))
                        ),
                        "exit_time": (
                            trade.get("ExitTime", "").isoformat()
                            if hasattr(trade.get("ExitTime", ""), "isoformat")
                            else str(trade.get("ExitTime", ""))
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
            "SMA_CROSS": {
                "name": "SMA Cross Strategy",
                "description": "Simple Moving Average Crossover Strategy",
                "parameters": {
                    "n1": {
                        "type": "int",
                        "default": 20,
                        "min": 5,
                        "max": 100,
                        "description": "Short-term SMA period",
                    },
                    "n2": {
                        "type": "int",
                        "default": 50,
                        "min": 20,
                        "max": 200,
                        "description": "Long-term SMA period",
                    },
                },
                "constraints": ["n1 < n2"],
            },
            "RSI": {
                "name": "RSI Strategy",
                "description": "Relative Strength Index Oscillator Strategy",
                "parameters": {
                    "period": {
                        "type": "int",
                        "default": 14,
                        "min": 5,
                        "max": 50,
                        "description": "RSI calculation period",
                    },
                    "oversold": {
                        "type": "int",
                        "default": 30,
                        "min": 10,
                        "max": 40,
                        "description": "Oversold threshold",
                    },
                    "overbought": {
                        "type": "int",
                        "default": 70,
                        "min": 60,
                        "max": 90,
                        "description": "Overbought threshold",
                    },
                },
                "constraints": ["oversold < overbought"],
            },
            "MACD": {
                "name": "MACD Strategy",
                "description": "Moving Average Convergence Divergence Strategy",
                "parameters": {
                    "fast_period": {
                        "type": "int",
                        "default": 12,
                        "min": 5,
                        "max": 20,
                        "description": "Fast EMA period",
                    },
                    "slow_period": {
                        "type": "int",
                        "default": 26,
                        "min": 20,
                        "max": 50,
                        "description": "Slow EMA period",
                    },
                    "signal_period": {
                        "type": "int",
                        "default": 9,
                        "min": 5,
                        "max": 15,
                        "description": "Signal line period",
                    },
                },
                "constraints": ["fast_period < slow_period"],
            },
            "SMA_RSI": {
                "name": "SMA + RSI Strategy",
                "description": "Combined SMA Crossover and RSI Momentum Strategy",
                "parameters": {
                    "sma_short": {
                        "type": "int",
                        "default": 20,
                        "min": 5,
                        "max": 50,
                        "description": "Short-term SMA period",
                    },
                    "sma_long": {
                        "type": "int",
                        "default": 50,
                        "min": 20,
                        "max": 200,
                        "description": "Long-term SMA period",
                    },
                    "rsi_period": {
                        "type": "int",
                        "default": 14,
                        "min": 5,
                        "max": 50,
                        "description": "RSI calculation period",
                    },
                    "oversold_threshold": {
                        "type": "int",
                        "default": 30,
                        "min": 10,
                        "max": 40,
                        "description": "RSI oversold threshold",
                    },
                    "overbought_threshold": {
                        "type": "int",
                        "default": 70,
                        "min": 60,
                        "max": 90,
                        "description": "RSI overbought threshold",
                    },
                    "use_risk_management": {
                        "type": "bool",
                        "default": True,
                        "description": "Enable risk management",
                    },
                    "sl_pct": {
                        "type": "float",
                        "default": 0.02,
                        "min": 0.005,
                        "max": 0.1,
                        "description": "Stop loss percentage",
                    },
                    "tp_pct": {
                        "type": "float",
                        "default": 0.05,
                        "min": 0.01,
                        "max": 0.2,
                        "description": "Take profit percentage",
                    },
                },
                "constraints": [
                    "sma_short < sma_long",
                    "oversold_threshold < overbought_threshold",
                ],
            },
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
