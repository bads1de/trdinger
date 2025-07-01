#!/usr/bin/env python3
"""
GENERATED_TEST戦略の詳細デバッグスクリプト

条件評価の実行過程を詳しく調査します。
"""

import sys
import os

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import pandas as pd
import logging
from datetime import datetime
from app.core.services.auto_strategy.models.strategy_gene import (
    StrategyGene,
    IndicatorGene,
    Condition,
)
from app.core.services.auto_strategy.factories.strategy_factory import StrategyFactory
from app.core.services.backtest_data_service import BacktestDataService
from database.repositories.ohlcv_repository import OHLCVRepository
from database.repositories.open_interest_repository import OpenInterestRepository
from database.repositories.funding_rate_repository import FundingRateRepository
from database.connection import SessionLocal

# ログレベルを詳細に設定
logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger(__name__)


def debug_generated_strategy():
    """GENERATED_TEST戦略の詳細デバッグ"""

    print("🔍 GENERATED_TEST戦略詳細デバッグ開始")
    print("=" * 60)

    # RSI戦略を作成
    strategy_gene = StrategyGene(
        id="debug_generated_detailed",
        indicators=[IndicatorGene(type="RSI", parameters={"period": 14}, enabled=True)],
        entry_conditions=[
            Condition(left_operand="RSI", operator="<", right_operand=30.0)
        ],
        exit_conditions=[
            Condition(left_operand="RSI", operator=">", right_operand=70.0)
        ],
        risk_management={"stop_loss": 0.05, "take_profit": 0.20, "position_size": 0.01},
    )

    print("📊 テスト戦略:")
    print(f"  指標: RSI(14)")
    print(f"  エントリー: RSI < 30")
    print(f"  エグジット: RSI > 70")
    print()

    try:
        # データベース接続
        db = SessionLocal()

        # データサービス
        ohlcv_repo = OHLCVRepository(db)
        oi_repo = OpenInterestRepository(db)
        fr_repo = FundingRateRepository(db)
        data_service = BacktestDataService(ohlcv_repo, oi_repo, fr_repo)

        # データ取得（短期間でテスト）
        print("📊 データ取得中...")
        data = data_service.get_data_for_backtest(
            symbol="BTC/USDT:USDT",
            timeframe="1h",
            start_date=datetime.strptime("2024-01-01", "%Y-%m-%d"),
            end_date=datetime.strptime("2024-01-31", "%Y-%m-%d"),  # 1ヶ月のみ
        )

        print(f"  データポイント数: {len(data)}")
        print(f"  データ期間: {data.index[0]} - {data.index[-1]}")
        print("  ✅ データ取得完了")

        # StrategyFactoryで戦略クラスを作成
        print("🏭 戦略クラス作成中...")
        try:
            factory = StrategyFactory()
            print("  → StrategyFactory作成完了")

            # IndicatorInitializerの状態確認
            print("  → IndicatorInitializer状態確認...")
            supported_indicators = (
                factory.indicator_initializer.get_supported_indicators()
            )
            print(f"    対応指標数: {len(supported_indicators)}")
            print(f"    対応指標: {supported_indicators[:5]}...")  # 最初の5個のみ表示

            strategy_class = factory.create_strategy_class(strategy_gene)
            print(f"  ✅ 戦略クラス作成成功: {strategy_class.__name__}")

        except Exception as e:
            print(f"  ❌ 戦略クラス作成エラー: {e}")
            import traceback

            traceback.print_exc()
            return

        # 戦略インスタンスを手動で作成してテスト
        print("🧪 戦略インスタンス手動テスト...")

        # backtesting.pyのBrokerとDataを模擬
        from backtesting import Backtest

        print("🔍 デバッグフェーズ開始...")

        # カスタム戦略クラスでデバッグ情報を追加
        class DebugGeneratedStrategy(strategy_class):
            def init(self):
                print("  🔧 init()メソッド実行中...")
                super().init()
                print(f"  📊 初期化された指標: {list(self.indicators.keys())}")

                # RSI値の統計を表示
                if "RSI" in self.indicators:
                    rsi_values = [x for x in self.indicators["RSI"] if not pd.isna(x)]
                    if rsi_values:
                        oversold_count = sum(1 for x in rsi_values if x < 30)
                        overbought_count = sum(1 for x in rsi_values if x > 70)
                        print(f"  📊 RSI統計:")
                        print(f"    有効値数: {len(rsi_values)}")
                        print(
                            f"    売られすぎ (< 30): {oversold_count} ({oversold_count/len(rsi_values)*100:.1f}%)"
                        )
                        print(
                            f"    買われすぎ (> 70): {overbought_count} ({overbought_count/len(rsi_values)*100:.1f}%)"
                        )

            def next(self):
                # 現在のRSI値を取得
                current_rsi = None
                if "RSI" in self.indicators and len(self.indicators["RSI"]) > 0:
                    # Pandas Seriesの場合はilocを使用
                    if hasattr(self.indicators["RSI"], "iloc"):
                        current_rsi = self.indicators["RSI"].iloc[-1]
                    else:
                        current_rsi = self.indicators["RSI"][-1]

                # 詳細ログ出力（最初の100回のみ）
                if len(self.data) <= 100:
                    rsi_str = (
                        f"{current_rsi:.2f}"
                        if current_rsi is not None and not pd.isna(current_rsi)
                        else "N/A"
                    )
                    print(
                        f"  📊 Bar {len(self.data)}: RSI={rsi_str}, Position={bool(self.position)}"
                    )

                # エントリー条件の詳細チェック
                if not self.position:
                    try:
                        entry_result = self._check_entry_conditions()
                        if len(self.data) <= 100:
                            print(f"    🔍 エントリー条件チェック: {entry_result}")

                        if entry_result:
                            rsi_str = (
                                f"{current_rsi:.2f}"
                                if current_rsi is not None and not pd.isna(current_rsi)
                                else "N/A"
                            )
                            print(
                                f"  🟢 エントリーシグナル発生! Bar {len(self.data)}, RSI={rsi_str}"
                            )
                            self.buy()
                    except Exception as e:
                        print(f"    ❌ エントリー条件チェックエラー: {e}")

                # エグジット条件の詳細チェック
                elif self.position:
                    try:
                        exit_result = self._check_exit_conditions()
                        if len(self.data) <= 100:
                            print(f"    🔍 エグジット条件チェック: {exit_result}")

                        if exit_result:
                            rsi_str = (
                                f"{current_rsi:.2f}"
                                if current_rsi is not None and not pd.isna(current_rsi)
                                else "N/A"
                            )
                            print(
                                f"  🔴 エグジットシグナル発生! Bar {len(self.data)}, RSI={rsi_str}"
                            )
                            self.sell()
                    except Exception as e:
                        print(f"    ❌ エグジット条件チェックエラー: {e}")

            def _check_entry_conditions(self):
                """エントリー条件チェック（デバッグ版）"""
                try:
                    result = super()._check_entry_conditions()

                    # 条件の詳細評価
                    for i, condition in enumerate(self.gene.entry_conditions):
                        try:
                            condition_result = self._evaluate_condition(condition)
                            left_value = self._get_condition_value(
                                condition.left_operand
                            )
                            right_value = self._get_condition_value(
                                condition.right_operand
                            )

                            if len(self.data) <= 100:
                                print(
                                    f"      条件{i+1}: {condition.left_operand}({left_value}) {condition.operator} {condition.right_operand}({right_value}) = {condition_result}"
                                )
                        except Exception as e:
                            print(f"      条件{i+1}評価エラー: {e}")

                    return result
                except Exception as e:
                    print(f"    エントリー条件チェック全体エラー: {e}")
                    return False

            def _check_exit_conditions(self):
                """エグジット条件チェック（デバッグ版）"""
                try:
                    result = super()._check_exit_conditions()

                    # 条件の詳細評価
                    for i, condition in enumerate(self.gene.exit_conditions):
                        try:
                            condition_result = self._evaluate_condition(condition)
                            left_value = self._get_condition_value(
                                condition.left_operand
                            )
                            right_value = self._get_condition_value(
                                condition.right_operand
                            )

                            if len(self.data) <= 100:
                                print(
                                    f"      条件{i+1}: {condition.left_operand}({left_value}) {condition.operator} {condition.right_operand}({right_value}) = {condition_result}"
                                )
                        except Exception as e:
                            print(f"      条件{i+1}評価エラー: {e}")

                    return result
                except Exception as e:
                    print(f"    エグジット条件チェック全体エラー: {e}")
                    return False

        # バックテスト実行
        print("⚡ デバッグバックテスト実行中...")
        print(f"  戦略クラス: {DebugGeneratedStrategy}")
        print(f"  データ形状: {data.shape}")
        print(f"  データ列: {list(data.columns)}")

        # 戦略インスタンスを手動で作成してテスト
        print("🧪 戦略インスタンス手動作成テスト...")
        try:
            print("  → 戦略インスタンス作成中...")
            strategy_instance = DebugGeneratedStrategy()
            print(f"  ✅ 戦略インスタンス作成成功")
            print(f"  gene: {hasattr(strategy_instance, 'gene')}")
            print(f"  indicators: {hasattr(strategy_instance, 'indicators')}")
            print(
                f"  gene内容: {strategy_instance.gene if hasattr(strategy_instance, 'gene') else 'なし'}"
            )

            # 手動でinit呼び出し
            print("  🔧 手動init呼び出し...")
            strategy_instance.data = data
            print("  → dataセット完了")
            strategy_instance.init()
            print(f"  ✅ init呼び出し成功")
            print(f"  指標数: {len(strategy_instance.indicators)}")
            print(f"  指標キー: {list(strategy_instance.indicators.keys())}")

        except Exception as e:
            print(f"  ❌ 戦略インスタンス手動テストエラー: {e}")
            import traceback

            traceback.print_exc()

        print("🚀 backtesting.pyバックテスト実行...")
        bt = Backtest(data, DebugGeneratedStrategy, cash=1000000, commission=0.001)
        result = bt.run()
        print("✅ バックテスト実行完了")

        print("📈 デバッグバックテスト結果:")
        print(f"  総取引数: {result['# Trades']}")
        print(f"  総リターン: {result['Return [%]']:.4f}%")
        print(f"  最終資産: {result['Equity Final [$]']:,.0f}")
        print(f"  勝率: {result['Win Rate [%]']:.2f}%")
        print(f"  最大ドローダウン: {result['Max. Drawdown [%]']:.4f}%")

        # 取引履歴
        trades = result._trades
        if len(trades) > 0:
            print(f"📋 取引履歴 ({len(trades)}件):")
            for i, trade in trades.iterrows():
                print(f"  取引 {i+1}:")
                print(
                    f"    エントリー: {trade['EntryTime']} @ {trade['EntryPrice']:.2f}"
                )
                print(f"    エグジット: {trade['ExitTime']} @ {trade['ExitPrice']:.2f}")
                print(f"    P&L: {trade['PnL']:.2f}")
        else:
            print("  ❌ 取引履歴なし")

        db.close()

    except Exception as e:
        print(f"❌ エラー: {e}")
        import traceback

        traceback.print_exc()


if __name__ == "__main__":
    # 強制的にログ出力
    import sys

    print("🚀 デバッグスクリプト開始...", flush=True)
    sys.stdout.flush()

    try:
        print("📞 debug_generated_strategy()関数呼び出し...", flush=True)
        debug_generated_strategy()
        print("✅ デバッグスクリプト完了", flush=True)
    except Exception as e:
        print(f"❌ デバッグスクリプトエラー: {e}", flush=True)
        import traceback

        traceback.print_exc()
        sys.stdout.flush()
