"""
メタラベリング バックテスト比較スクリプト

「ベースシグナルのみ」 vs 「MLでフィルタリング後」の損益を比較します。
Meta-Labelingの効果を定量的に評価するためのスクリプトです。

実行例:
    python -m scripts.meta_labeling_backtest_comparison \\
        --symbol BTCUSDT \\
        --start-date 2024-01-01 \\
        --end-date 2024-06-01 \\
        --signal-type bb_breakout
"""

import argparse
import logging
from datetime import datetime
from typing import Dict, Any, Tuple
import pandas as pd
import numpy as np
from pathlib import Path

from app.database.session import get_db
from app.database.repositories.ohlcv_repository import OHLCVRepository
from app.database.repositories.funding_rate_repository import FundingRateRepository
from app.database.repositories.open_interest_repository import OpenInterestRepository
from app.services.ml.label_generation import SignalGenerator, LabelGenerationService
from app.services.ml.feature_engineering.feature_engineering_service import (
    FeatureEngineeringService,
    FAKEOUT_DETECTION_ALLOWLIST,
)
from app.services.ml.common.meta_labeling_evaluation import (
    evaluate_meta_labeling,
    print_meta_labeling_report,
)

logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


class MetaLabelingBacktestComparison:
    """
    メタラベリングのバックテスト比較クラス

    ベースシグナルとMLフィルタリング後の損益を比較し、
    Meta-Labelingの有効性を評価します。
    """

    def __init__(
        self,
        symbol: str = "BTCUSDT",
        start_date: str = "2024-01-01",
        end_date: str = "2024-06-01",
        timeframe: str = "1h",
    ):
        """
        初期化

        Args:
            symbol: 銘柄シンボル
            start_date: 開始日（YYYY-MM-DD）
            end_date: 終了日（YYYY-MM-DD）
            timeframe: 時間足
        """
        self.symbol = symbol
        self.start_date = start_date
        self.end_date = end_date
        self.timeframe = timeframe

        # データベースセッション
        self.db = next(get_db())

        # リポジトリ
        self.ohlcv_repo = OHLCVRepository(self.db)
        self.fr_repo = FundingRateRepository(self.db)
        self.oi_repo = OpenInterestRepository(self.db)

        # サービス
        self.signal_generator = SignalGenerator()
        self.label_service = LabelGenerationService()
        self.feature_service = FeatureEngineeringService()

    def load_data(self) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
        """
        データを読み込む

        Returns:
            (OHLCV, FundingRate, OpenInterest) のタプル
        """
        logger.info(
            f"データ読み込み開始: {self.symbol} ({self.start_date} ~ {self.end_date})"
        )

        # OHLCV
        ohlcv_df = self.ohlcv_repo.get_ohlcv_data(
            symbol=self.symbol,
            timeframe=self.timeframe,
            start_date=self.start_date,
            end_date=self.end_date,
        )

        # Funding Rate
        fr_df = self.fr_repo.get_funding_rate_data(
            symbol=self.symbol, start_date=self.start_date, end_date=self.end_date
        )

        # Open Interest
        oi_df = self.oi_repo.get_open_interest_data(
            symbol=self.symbol, start_date=self.start_date, end_date=self.end_date
        )

        logger.info(
            f"✅ データ読み込み完了: OHLCV={len(ohlcv_df)}件, FR={len(fr_df)}件, OI={len(oi_df)}件"
        )

        return ohlcv_df, fr_df, oi_df

    def calculate_pnl(
        self,
        signals: pd.DatetimeIndex,
        ohlcv_df: pd.DataFrame,
        pt_factor: float = 1.5,
        sl_factor: float = 1.0,
        holding_periods: int = 4,
    ) -> Dict[str, Any]:
        """
        シグナルに基づいてPnLを計算

        Args:
            signals: エントリーシグナルのタイムスタンプ
            ohlcv_df: OHLCV DataFrame
            pt_factor: 利確倍率
            sl_factor: 損切倍率
            holding_periods: 保有期間（足数）

        Returns:
            PnL統計の辞書
        """
        if len(signals) == 0:
            return {
                "total_trades": 0,
                "winning_trades": 0,
                "losing_trades": 0,
                "win_rate": 0.0,
                "total_pnl": 0.0,
                "avg_win": 0.0,
                "avg_loss": 0.0,
                "profit_factor": 0.0,
                "sharpe_ratio": 0.0,
            }

        trades = []

        for signal_time in signals:
            if signal_time not in ohlcv_df.index:
                continue

            entry_idx = ohlcv_df.index.get_loc(signal_time)
            entry_price = ohlcv_df.iloc[entry_idx]["close"]

            # ボラティリティ（ATR）計算
            atr = ohlcv_df.iloc[max(0, entry_idx - 14) : entry_idx + 1]["close"].std()
            if pd.isna(atr) or atr == 0:
                atr = entry_price * 0.01  # デフォルト1%

            # 利確・損切レベル
            take_profit = entry_price + (atr * pt_factor)
            stop_loss = entry_price - (atr * sl_factor)

            # 保有期間のデータ
            exit_idx = min(entry_idx + holding_periods, len(ohlcv_df) - 1)
            holding_data = ohlcv_df.iloc[entry_idx : exit_idx + 1]

            # 利確・損切判定
            pnl = 0.0
            exit_reason = "time_limit"

            for i, (idx, row) in enumerate(holding_data.iterrows()):
                if i == 0:
                    continue  # エントリー足はスキップ

                # 利確ヒット
                if row["high"] >= take_profit:
                    pnl = (take_profit - entry_price) / entry_price
                    exit_reason = "take_profit"
                    break

                # 損切ヒット
                if row["low"] <= stop_loss:
                    pnl = (stop_loss - entry_price) / entry_price
                    exit_reason = "stop_loss"
                    break

            # 時間切れの場合
            if exit_reason == "time_limit":
                exit_price = holding_data.iloc[-1]["close"]
                pnl = (exit_price - entry_price) / entry_price

            trades.append(
                {
                    "entry_time": signal_time,
                    "entry_price": entry_price,
                    "pnl": pnl,
                    "exit_reason": exit_reason,
                }
            )

        # 統計計算
        if len(trades) == 0:
            return {
                "total_trades": 0,
                "winning_trades": 0,
                "losing_trades": 0,
                "win_rate": 0.0,
                "total_pnl": 0.0,
                "avg_win": 0.0,
                "avg_loss": 0.0,
                "profit_factor": 0.0,
                "sharpe_ratio": 0.0,
            }

        pnls = [t["pnl"] for t in trades]
        winning_trades = [p for p in pnls if p > 0]
        losing_trades = [p for p in pnls if p < 0]

        total_pnl = sum(pnls)
        win_rate = len(winning_trades) / len(trades) if len(trades) > 0 else 0.0
        avg_win = np.mean(winning_trades) if len(winning_trades) > 0 else 0.0
        avg_loss = np.mean(losing_trades) if len(losing_trades) > 0 else 0.0

        total_win = sum(winning_trades) if len(winning_trades) > 0 else 0.0
        total_loss = abs(sum(losing_trades)) if len(losing_trades) > 0 else 0.0
        profit_factor = total_win / total_loss if total_loss > 0 else 0.0

        sharpe_ratio = np.mean(pnls) / np.std(pnls) if np.std(pnls) > 0 else 0.0

        return {
            "total_trades": len(trades),
            "winning_trades": len(winning_trades),
            "losing_trades": len(losing_trades),
            "win_rate": win_rate,
            "total_pnl": total_pnl,
            "avg_win": avg_win,
            "avg_loss": avg_loss,
            "profit_factor": profit_factor,
            "sharpe_ratio": sharpe_ratio,
            "trades": trades,
        }

    def run_comparison(
        self,
        signal_type: str = "bb_breakout",
        use_ml_model: bool = False,
        model_path: str = None,
    ) -> Dict[str, Any]:
        """
        バックテスト比較を実行

        Args:
            signal_type: シグナルタイプ（"bb_breakout", "donchian", "volume_spike", "combined"）
            use_ml_model: MLモデルを使用するか
            model_path: MLモデルのパス（use_ml_model=Trueの場合）

        Returns:
            比較結果の辞書
        """
        # データ読み込み
        ohlcv_df, fr_df, oi_df = self.load_data()

        # シグナル生成
        logger.info(f"シグナル生成: {signal_type}")

        if signal_type == "bb_breakout":
            base_signals = self.signal_generator.get_bb_breakout_events(
                df=ohlcv_df, window=20, dev=2.0
            )
        elif signal_type == "donchian":
            base_signals = self.signal_generator.get_donchian_breakout_events(
                df=ohlcv_df, window=20
            )
        elif signal_type == "volume_spike":
            base_signals = self.signal_generator.get_volume_spike_events(
                df=ohlcv_df, window=20, multiplier=2.5
            )
        elif signal_type == "combined":
            base_signals = self.signal_generator.get_combined_events(
                df=ohlcv_df, use_bb=True, use_donchian=True, use_volume=True
            )
        else:
            raise ValueError(f"Unknown signal type: {signal_type}")

        logger.info(f"✅ ベースシグナル生成完了: {len(base_signals)}件")

        # ベースシグナルのみでバックテスト
        logger.info("ベースシグナルのみでバックテスト実行...")
        base_results = self.calculate_pnl(base_signals, ohlcv_df)

        # MLフィルタリング（オプション）
        if use_ml_model and model_path:
            logger.info("MLモデルでシグナルをフィルタリング...")

            # 特徴量生成
            features_df = self.feature_service.calculate_advanced_features(
                ohlcv_data=ohlcv_df, funding_rate_data=fr_df, open_interest_data=oi_df
            )

            # TODO: モデルロードと予測
            # filtered_signals = ...

            # 暫定: ダミー実装（実際はモデルで予測）
            logger.warning("MLモデル未実装のため、ベースシグナルの50%をランダムに選択")
            np.random.seed(42)
            n_filtered = int(len(base_signals) * 0.5)
            filtered_signals = pd.DatetimeIndex(
                np.random.choice(base_signals, size=n_filtered, replace=False)
            ).sort_values()

            logger.info(f"✅ MLフィルタリング完了: {len(filtered_signals)}件")

            # MLフィルタリング後でバックテスト
            logger.info("MLフィルタリング後でバックテスト実行...")
            ml_results = self.calculate_pnl(filtered_signals, ohlcv_df)
        else:
            ml_results = None

        # 結果をまとめる
        results = {
            "base_signal_results": base_results,
            "ml_filtered_results": ml_results,
            "comparison": (
                self._create_comparison(base_results, ml_results)
                if ml_results
                else None
            ),
        }

        return results

    def _create_comparison(self, base: Dict, ml: Dict) -> Dict[str, Any]:
        """
        ベースとMLの比較統計を作成

        Args:
            base: ベースシグナルの結果
            ml: MLフィルタリング後の結果

        Returns:
            比較統計の辞書
        """
        return {
            "win_rate_improvement": ml["win_rate"] - base["win_rate"],
            "pnl_improvement": ml["total_pnl"] - base["total_pnl"],
            "profit_factor_improvement": ml["profit_factor"] - base["profit_factor"],
            "sharpe_improvement": ml["sharpe_ratio"] - base["sharpe_ratio"],
            "signal_reduction_rate": (
                1 - (ml["total_trades"] / base["total_trades"])
                if base["total_trades"] > 0
                else 0.0
            ),
        }

    def print_results(self, results: Dict[str, Any]) -> None:
        """
        結果を出力

        Args:
            results: run_comparison() の戻り値
        """
        print("\n" + "=" * 80)
        print("📊 Meta-Labeling Backtest Comparison Results")
        print("=" * 80)

        base = results["base_signal_results"]
        ml = results["ml_filtered_results"]

        print("\n🎯 ベースシグナルのみ（Base Signal Only）:")
        print(f"  Total Trades:     {base['total_trades']}")
        print(f"  Win Rate:         {base['win_rate']:.2%}")
        print(f"  Total PnL:        {base['total_pnl']:.4f}")
        print(f"  Profit Factor:    {base['profit_factor']:.2f}")
        print(f"  Sharpe Ratio:     {base['sharpe_ratio']:.2f}")

        if ml:
            print("\n🤖 MLフィルタリング後（ML Filtered）:")
            print(f"  Total Trades:     {ml['total_trades']}")
            print(f"  Win Rate:         {ml['win_rate']:.2%}")
            print(f"  Total PnL:        {ml['total_pnl']:.4f}")
            print(f"  Profit Factor:    {ml['profit_factor']:.2f}")
            print(f"  Sharpe Ratio:     {ml['sharpe_ratio']:.2f}")

            comp = results["comparison"]
            print("\n📈 改善度（Improvement）:")
            print(f"  Win Rate:         {comp['win_rate_improvement']:+.2%}")
            print(f"  Total PnL:        {comp['pnl_improvement']:+.4f}")
            print(f"  Profit Factor:    {comp['profit_factor_improvement']:+.2f}")
            print(f"  Sharpe Ratio:     {comp['sharpe_improvement']:+.2f}")
            print(f"  Signal Reduction: {comp['signal_reduction_rate']:.1%}")

        print("\n" + "=" * 80 + "\n")


def main():
    """
    メイン関数
    """
    parser = argparse.ArgumentParser(description="Meta-Labeling Backtest Comparison")
    parser.add_argument(
        "--symbol", type=str, default="BTCUSDT", help="Symbol (default: BTCUSDT)"
    )
    parser.add_argument(
        "--start-date", type=str, default="2024-01-01", help="Start date (YYYY-MM-DD)"
    )
    parser.add_argument(
        "--end-date", type=str, default="2024-06-01", help="End date (YYYY-MM-DD)"
    )
    parser.add_argument(
        "--signal-type",
        type=str,
        default="bb_breakout",
        choices=["bb_breakout", "donchian", "volume_spike", "combined"],
        help="Signal type",
    )
    parser.add_argument(
        "--use-ml-model", action="store_true", help="Use ML model for filtering"
    )
    parser.add_argument("--model-path", type=str, default=None, help="Path to ML model")

    args = parser.parse_args()

    # バックテスト実行
    comparison = MetaLabelingBacktestComparison(
        symbol=args.symbol, start_date=args.start_date, end_date=args.end_date
    )

    results = comparison.run_comparison(
        signal_type=args.signal_type,
        use_ml_model=args.use_ml_model,
        model_path=args.model_path,
    )

    comparison.print_results(results)


if __name__ == "__main__":
    main()
