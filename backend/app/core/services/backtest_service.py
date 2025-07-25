"""
バックテスト実行サービス

backtesting.pyライブラリを使用したバックテスト実行機能を提供します。
"""

import logging
import pandas as pd
import warnings

from datetime import datetime
from typing import Dict, Any, Type, Optional

from backtesting import Backtest, Strategy
from .backtest_data_service import BacktestDataService
from database.repositories.ohlcv_repository import OHLCVRepository
from database.repositories.backtest_result_repository import BacktestResultRepository
from database.connection import get_db
from sqlalchemy.orm import Session


logger = logging.getLogger(__name__)


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
        try:
            # ログに出力する設定情報から、長大なデータを除外する
            log_config = {
                k: v
                for k, v in config.items()
                if k not in ["strategy_config", "equity_curve", "trade_history"]
            }
            if "strategy_config" in config:
                log_config["strategy_config"] = {
                    "strategy_type": config["strategy_config"].get("strategy_type"),
                    "parameters": "...",  # strategy_geneはログに出力しない
                }
            # 1. 設定の検証
            self._validate_config(config)

            # 2. データサービスの初期化（必要に応じて）
            if self.data_service is None:
                db = next(get_db())
                try:
                    ohlcv_repo = OHLCVRepository(db)
                    self.data_service = BacktestDataService(ohlcv_repo)
                finally:
                    db.close()

            # 3. データ取得
            # 日付文字列をdatetimeオブジェクトに変換
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
            # 常にOI/FR統合データを使用
            data = self.data_service.get_data_for_backtest(
                symbol=config["symbol"],
                timeframe=config["timeframe"],
                start_date=start_date,
                end_date=end_date,
            )

            if data is None or data.empty:
                raise ValueError(
                    f"{config['symbol']} {config['timeframe']} の {config['start_date']} から {config['end_date']} までのOHLCVデータが見つかりませんでした。"
                )

            # 4. 戦略クラス取得または生成
            if "strategy_class" in config:
                # GAエンジンから直接戦略クラスが渡された場合
                strategy_class = config["strategy_class"]
                # パラメータは戦略クラス生成時に既に設定済み
                strategy_parameters = {}
            else:
                # 通常のstrategy_configから戦略クラスを生成する場合
                strategy_class = self._create_strategy_class(config["strategy_config"])
                # パラメータを取得
                strategy_parameters = config.get("strategy_config", {}).get(
                    "parameters", {}
                )

            # 5. backtesting.py実行
            bt = Backtest(
                data,
                strategy_class,
                cash=config["initial_capital"],
                commission=config["commission_rate"],
                exclusive_orders=False,  # マージン問題を回避
                trade_on_close=True,  # 終値で取引
                hedging=False,  # ヘッジングを無効化
                margin=1.0,  # マージン要件を1.0に設定（レバレッジなし）
            )

            # バックテストを実行（時間計測付き）
            import time

            start_time = time.time()
            warnings.filterwarnings("ignore", category=UserWarning)
            stats = bt.run(**strategy_parameters)
            warnings.filterwarnings("default", category=UserWarning)
            execution_time = time.time() - start_time
            logger.debug(f"バックテスト実行時間: {execution_time:.2f}秒")

            # 6. 結果をデータベース形式に変換

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

        return True

    def _create_strategy_class(self, strategy_config: Dict[str, Any]) -> Type[Strategy]:
        """
        戦略設定からオートストラテジーのクラスを生成します。

        Args:
            strategy_config: 戦略設定。'strategy_gene' を含む必要があります。

        Returns:
            生成された戦略クラス

        Raises:
            ValueError: 戦略遺伝子が見つからない場合
        """
        # オートストラテジーのみをサポート
        from app.core.services.auto_strategy.factories.strategy_factory import (
            StrategyFactory,
        )
        from app.core.services.auto_strategy.models.gene_strategy import (
            StrategyGene,
        )

        # パラメータから戦略遺伝子を復元
        gene_data = strategy_config.get("strategy_gene") or strategy_config.get(
            "parameters", {}
        ).get("strategy_gene")

        if not gene_data:
            raise ValueError(
                "オートストラテジーの実行には、戦略遺伝子 (strategy_gene) がパラメータとして必要です。"
            )

        from app.core.services.auto_strategy.models.gene_serialization import (
            GeneSerializer,
        )

        serializer = GeneSerializer()
        strategy_gene = serializer.dict_to_strategy_gene(gene_data, StrategyGene)
        factory = StrategyFactory()
        return factory.create_strategy_class(strategy_gene)

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
        サポートされている戦略の一覧を取得します。
        現在はオートストラテジーのみをサポートしています。

        Returns:
            戦略情報の辞書
        """
        return {
            "GENERATED_AUTO": {
                "name": "Auto-Generated Strategy",
                "description": "オートストラテジーシステムで自動生成された戦略",
                "parameters": {
                    "strategy_gene": {
                        "type": "dict",
                        "description": "戦略遺伝子データ",
                        "required": True,
                    }
                },
                "constraints": [],
            }
        }

    def execute_and_save_backtest(self, request, db_session: Session) -> Dict[str, Any]:
        """
        バックテストを実行し、結果をデータベースに保存

        Args:
            request: BacktestRequestオブジェクト
            db_session: データベースセッション

        Returns:
            実行結果の辞書
        """
        try:
            # リクエストから設定を作成
            config = {
                "strategy_name": request.strategy_name,
                "symbol": request.symbol,
                "timeframe": request.timeframe,
                "start_date": request.start_date,
                "end_date": request.end_date,
                "initial_capital": request.initial_capital,
                "commission_rate": request.commission_rate,
                "strategy_config": request.strategy_config.dict(),
            }

            # バックテストを実行
            result = self.run_backtest(config)

            # 結果をデータベースに保存
            backtest_repo = BacktestResultRepository(db_session)
            saved_result = backtest_repo.save_backtest_result(result)

            return {"success": True, "result": saved_result}

        except Exception as e:
            logger.error(f"バックテスト実行・保存エラー: {e}", exc_info=True)
            raise
