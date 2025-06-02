"""
バックテスト実行サービス

backtesting.pyライブラリを使用したバックテスト実行機能を提供します。
"""

import pandas as pd
from datetime import datetime
from typing import Dict, Any, Type
from backtesting import Backtest, Strategy

from .backtest_data_service import BacktestDataService
from ..strategies.sma_cross_strategy import SMACrossStrategy
from ..strategies.rsi_strategy import RSIStrategy
from ..strategies.macd_strategy import MACDStrategy
from database.repositories.ohlcv_repository import OHLCVRepository
from database.connection import SessionLocal


class BacktestService:
    """
    backtesting.pyを使用したバックテスト実行サービス

    既存のOHLCVデータを使用してバックテストを実行し、
    結果をデータベース保存用の形式に変換します。
    """

    def __init__(self, data_service: BacktestDataService = None):
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
        # 1. 設定の検証
        self._validate_config(config)

        # 2. データサービスの初期化（必要に応じて）
        if self.data_service is None:
            db = SessionLocal()
            try:
                ohlcv_repo = OHLCVRepository(db)
                self.data_service = BacktestDataService(ohlcv_repo)
            finally:
                db.close()

        # 3. データ取得
        data = self.data_service.get_ohlcv_for_backtest(
            symbol=config["symbol"],
            timeframe=config["timeframe"],
            start_date=config["start_date"],
            end_date=config["end_date"],
        )

        # 4. 戦略クラス動的生成
        strategy_class = self._create_strategy_class(config["strategy_config"])

        # 5. backtesting.py実行
        bt = Backtest(
            data,
            strategy_class,
            cash=config["initial_capital"],
            commission=config["commission_rate"],
            exclusive_orders=True,  # 推奨設定
            trade_on_close=True,  # 終値で取引
        )

        stats = bt.run()

        # 6. 結果をデータベース形式に変換
        result = self._convert_backtest_results(
            stats,
            config["strategy_name"],
            config["symbol"],
            config["timeframe"],
            config["initial_capital"],
        )

        return result

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
        戦略設定から動的に戦略クラスを生成

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
            # SMAクロス戦略のカスタムクラスを作成
            class CustomSMACrossStrategy(SMACrossStrategy):
                pass

            # パラメータを動的に設定
            if "n1" in parameters:
                CustomSMACrossStrategy.n1 = parameters["n1"]
            if "n2" in parameters:
                CustomSMACrossStrategy.n2 = parameters["n2"]

            return CustomSMACrossStrategy

        elif strategy_type == "RSI":
            # RSI戦略のカスタムクラスを作成
            class CustomRSIStrategy(RSIStrategy):
                pass

            # パラメータを動的に設定
            if "period" in parameters:
                CustomRSIStrategy.period = parameters["period"]
            if "oversold" in parameters:
                CustomRSIStrategy.oversold = parameters["oversold"]
            if "overbought" in parameters:
                CustomRSIStrategy.overbought = parameters["overbought"]

            return CustomRSIStrategy

        elif strategy_type == "MACD":
            # MACD戦略のカスタムクラスを作成
            class CustomMACDStrategy(MACDStrategy):
                pass

            # パラメータを動的に設定
            if "fast_period" in parameters:
                CustomMACDStrategy.fast_period = parameters["fast_period"]
            if "slow_period" in parameters:
                CustomMACDStrategy.slow_period = parameters["slow_period"]
            if "signal_period" in parameters:
                CustomMACDStrategy.signal_period = parameters["signal_period"]

            return CustomMACDStrategy

        else:
            raise ValueError(f"Unsupported strategy type: {strategy_type}")

    def _convert_backtest_results(
        self,
        stats: pd.Series,
        strategy_name: str,
        symbol: str,
        timeframe: str,
        initial_capital: float,
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
        if "_trades" in stats and not stats["_trades"].empty:
            trades_df = stats["_trades"]
            for _, trade in trades_df.iterrows():
                trade_history.append(
                    {
                        "size": float(trade.get("Size", 0)),
                        "entry_price": float(trade.get("EntryPrice", 0)),
                        "exit_price": float(trade.get("ExitPrice", 0)),
                        "pnl": float(trade.get("PnL", 0)),
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

        # 結果を統合
        result = {
            "strategy_name": strategy_name,
            "symbol": symbol,
            "timeframe": timeframe,
            "initial_capital": initial_capital,
            "commission_rate": 0.001,  # デフォルト値を追加
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

        # データ取得
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

        stats = bt.optimize(**optimize_kwargs)

        # 結果を変換
        result = self._convert_backtest_results(
            stats,
            config["strategy_name"],
            config["symbol"],
            config["timeframe"],
            config["initial_capital"],
        )

        # 最適化されたパラメータを追加
        optimized_strategy = stats["_strategy"]
        result["optimized_parameters"] = {
            param: getattr(optimized_strategy, param)
            for param in optimization_params["parameters"].keys()
            if hasattr(optimized_strategy, param)
        }

        return result
