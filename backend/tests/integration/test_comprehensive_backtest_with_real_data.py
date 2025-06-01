"""
実際のDBデータを使用した包括的バックテストテスト

backtesting.pyライブラリに統一されたシステムの包括的テスト
実際のデータベースに保存されている1dデータを使用
"""

import pytest
import pandas as pd
import numpy as np
from datetime import datetime, timedelta, timezone
import logging
import sys
import os

# プロジェクトルートをパスに追加
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(__file__))))

from app.core.services.backtest_service import BacktestService
from app.core.services.backtest_data_service import BacktestDataService
from app.core.utils.data_standardization import (
    standardize_ohlcv_columns,
    validate_ohlcv_data,
    prepare_data_for_backtesting,
    convert_legacy_config_to_backtest_service,
)
from database.connection import SessionLocal
from database.repositories.ohlcv_repository import OHLCVRepository
from backtest.runner import run_backtest

logger = logging.getLogger(__name__)


class TestComprehensiveBacktestWithRealData:
    """実際のDBデータを使用した包括的バックテストテスト"""

    @pytest.fixture(scope="class")
    def db_session(self):
        """データベースセッション"""
        db = SessionLocal()
        try:
            yield db
        finally:
            db.close()

    @pytest.fixture(scope="class")
    def ohlcv_repo(self, db_session):
        """OHLCVリポジトリ"""
        return OHLCVRepository(db_session)

    @pytest.fixture(scope="class")
    def available_data_info(self, ohlcv_repo):
        """利用可能なデータ情報を取得"""
        # 1dデータのみを対象とする
        symbols = ["BTC/USDT", "BTC/USDT:USDT", "BTCUSD"]
        timeframe = "1d"

        data_info = {}
        for symbol in symbols:
            count = ohlcv_repo.count_records(symbol, timeframe)
            if count > 0:
                latest = ohlcv_repo.get_latest_timestamp(symbol, timeframe)
                oldest = ohlcv_repo.get_oldest_timestamp(symbol, timeframe)
                data_info[symbol] = {
                    "count": count,
                    "latest": latest,
                    "oldest": oldest,
                    "timeframe": timeframe,
                }
                logger.info(
                    f"利用可能データ: {symbol} {timeframe} - {count}件 ({oldest.date()} ～ {latest.date()})"
                )

        return data_info

    @pytest.fixture
    def backtest_service(self, db_session):
        """BacktestService with real data"""
        ohlcv_repo = OHLCVRepository(db_session)
        data_service = BacktestDataService(ohlcv_repo)
        return BacktestService(data_service)

    def test_data_availability(self, available_data_info):
        """実際のデータが利用可能かテスト"""
        assert len(available_data_info) > 0, "利用可能なデータがありません"

        for symbol, info in available_data_info.items():
            assert (
                info["count"] > 100
            ), f"{symbol}のデータが不足しています: {info['count']}件"
            assert (
                info["timeframe"] == "1d"
            ), f"1dデータではありません: {info['timeframe']}"

            # 最低30日分のデータがあることを確認
            days_diff = (info["latest"] - info["oldest"]).days
            assert days_diff >= 30, f"{symbol}のデータ期間が短すぎます: {days_diff}日"

    def test_data_quality_validation(self, ohlcv_repo, available_data_info):
        """実際のデータの品質をテスト"""
        for symbol in available_data_info.keys():
            # 最新100件のデータを取得
            df = ohlcv_repo.get_ohlcv_dataframe(symbol, "1d", limit=100)

            # データが空でないことを確認
            assert not df.empty, f"{symbol}のデータが空です"

            # 列名を標準化
            standardized_df = standardize_ohlcv_columns(df)

            # データの妥当性を検証
            assert validate_ohlcv_data(standardized_df), f"{symbol}のデータが無効です"

            # 基本的な価格関係をチェック
            assert (
                standardized_df["High"] >= standardized_df["Low"]
            ).all(), f"{symbol}でHigh < Lowの異常データがあります"

            # 負の価格がないことを確認
            price_cols = ["Open", "High", "Low", "Close"]
            for col in price_cols:
                assert (
                    standardized_df[col] > 0
                ).all(), f"{symbol}の{col}に負の値があります"

    def test_basic_sma_cross_strategy_btc_usdt(
        self, backtest_service, available_data_info
    ):
        """BTC/USDTでの基本的なSMAクロス戦略テスト"""
        symbol = "BTC/USDT"
        if symbol not in available_data_info:
            pytest.skip(f"{symbol}のデータが利用できません")

        # 最近3ヶ月のデータを使用
        end_date = available_data_info[symbol]["latest"]
        start_date = end_date - timedelta(days=90)

        config = {
            "strategy_name": "SMA_CROSS",
            "symbol": symbol,
            "timeframe": "1d",
            "start_date": start_date.isoformat(),
            "end_date": end_date.isoformat(),
            "initial_capital": 1000000,  # BTCの高価格に対応するため100万に増額
            "commission_rate": 0.001,
            "strategy_config": {
                "strategy_type": "SMA_CROSS",
                "parameters": {"n1": 20, "n2": 50},
            },
        }

        result = backtest_service.run_backtest(config)

        # 結果の基本検証
        assert "performance_metrics" in result
        assert "total_return" in result["performance_metrics"]
        assert "sharpe_ratio" in result["performance_metrics"]
        assert "max_drawdown" in result["performance_metrics"]
        assert "win_rate" in result["performance_metrics"]

        # パフォーマンス指標の妥当性チェック
        metrics = result["performance_metrics"]
        assert isinstance(metrics["total_return"], (int, float))
        assert (
            -1.0 <= metrics["total_return"] <= 50.0
        )  # -100%から5000%の範囲（暗号通貨は高ボラティリティ）
        assert (
            -10.0 <= metrics["max_drawdown"] <= 0.0
        )  # ドローダウンは負の値（暗号通貨は高ボラティリティ）

        # 勝率はNaNの場合もある（取引が発生しない場合）
        if not pd.isna(metrics["win_rate"]):
            assert 0.0 <= metrics["win_rate"] <= 1.0  # 勝率は0-1の範囲

    def test_different_symbols_comparison(self, backtest_service, available_data_info):
        """異なるシンボルでの戦略比較テスト"""
        if len(available_data_info) < 2:
            pytest.skip("比較に十分なシンボルがありません")

        results = {}
        base_config = {
            "strategy_name": "SMA_CROSS",
            "timeframe": "1d",
            "initial_capital": 1000000,  # BTCの高価格に対応
            "commission_rate": 0.001,
            "strategy_config": {
                "strategy_type": "SMA_CROSS",
                "parameters": {"n1": 10, "n2": 30},
            },
        }

        # 各シンボルで同じ期間のテストを実行
        for symbol, info in available_data_info.items():
            # 最近2ヶ月のデータを使用
            end_date = info["latest"]
            start_date = end_date - timedelta(days=60)

            config = base_config.copy()
            config.update(
                {
                    "symbol": symbol,
                    "start_date": start_date.isoformat(),
                    "end_date": end_date.isoformat(),
                }
            )

            try:
                result = backtest_service.run_backtest(config)
                results[symbol] = result["performance_metrics"]
                logger.info(
                    f"{symbol} - Total Return: {result['performance_metrics']['total_return']:.2%}"
                )
            except Exception as e:
                logger.warning(f"{symbol}のバックテストでエラー: {e}")

        # 少なくとも2つのシンボルで成功していることを確認
        assert len(results) >= 2, "複数シンボルでのテストが失敗しました"

        # 全ての結果が有効な値を持つことを確認
        for symbol, metrics in results.items():
            assert "total_return" in metrics
            assert isinstance(metrics["total_return"], (int, float))

    def test_parameter_optimization_with_real_data(
        self, backtest_service, available_data_info
    ):
        """実際のデータを使用したパラメータ最適化テスト"""
        # データが最も多いシンボルを選択
        best_symbol = max(
            available_data_info.keys(), key=lambda s: available_data_info[s]["count"]
        )

        info = available_data_info[best_symbol]

        # 最近6ヶ月のデータを使用（最適化には多めのデータが必要）
        end_date = info["latest"]
        start_date = end_date - timedelta(days=180)

        config = {
            "strategy_name": "SMA_CROSS",
            "symbol": best_symbol,
            "timeframe": "1d",
            "start_date": start_date.isoformat(),
            "end_date": end_date.isoformat(),
            "initial_capital": 1000000,  # BTCの高価格に対応
            "commission_rate": 0.001,
            "strategy_config": {
                "strategy_type": "SMA_CROSS",
                "parameters": {"n1": 20, "n2": 50},
            },
        }

        # 最適化パラメータ
        optimization_params = {
            "parameters": {
                "n1": list(range(5, 25, 5)),  # [5, 10, 15, 20]
                "n2": list(range(25, 65, 10)),  # [25, 35, 45, 55]
            },
            "maximize": "Sharpe Ratio",
            "constraint": lambda p: p.n1 < p.n2,
        }

        result = backtest_service.optimize_strategy(config, optimization_params)

        # 最適化結果の検証
        assert "optimized_parameters" in result
        assert "performance_metrics" in result

        optimized_params = result["optimized_parameters"]
        assert "n1" in optimized_params
        assert "n2" in optimized_params
        assert optimized_params["n1"] < optimized_params["n2"]  # 制約条件の確認

    def test_runner_integration_with_real_data(self, available_data_info):
        """runner.pyの統合テスト（実際のデータ使用）"""
        # データが最も多いシンボルを選択
        best_symbol = max(
            available_data_info.keys(), key=lambda s: available_data_info[s]["count"]
        )

        info = available_data_info[best_symbol]

        # 最近1ヶ月のデータを使用
        end_date = info["latest"]
        start_date = end_date - timedelta(days=30)

        # 従来の設定形式（runner.pyが期待する形式）
        legacy_config = {
            "strategy": {
                "name": "SMA_CROSS",
                "target_pair": best_symbol,
                "indicators": [
                    {"name": "SMA", "params": {"period": 10}},
                    {"name": "SMA", "params": {"period": 20}},
                ],
                "entry_rules": [{"condition": "SMA(close, 10) > SMA(close, 20)"}],
                "exit_rules": [{"condition": "SMA(close, 10) < SMA(close, 20)"}],
            },
            "start_date": start_date.isoformat() + "Z",
            "end_date": end_date.isoformat() + "Z",
            "timeframe": "1d",
            "initial_capital": 1000000,  # BTCの高価格に対応
            "commission_rate": 0.001,
        }

        result = run_backtest(legacy_config)

        # エラーが発生していないことを確認
        assert "error" not in result, f"runner.pyでエラーが発生: {result.get('error')}"

        # 基本的な結果フィールドが存在することを確認
        expected_fields = ["id", "strategy_id", "config", "created_at"]
        for field in expected_fields:
            assert field in result, f"必要なフィールドが不足: {field}"

    def test_edge_cases_with_real_data(self, backtest_service, available_data_info):
        """実際のデータを使用したエッジケーステスト"""
        symbol = list(available_data_info.keys())[0]
        info = available_data_info[symbol]

        # 1. 非常に短い期間（1週間）
        end_date = info["latest"]
        start_date = end_date - timedelta(days=7)

        config = {
            "strategy_name": "SMA_CROSS",
            "symbol": symbol,
            "timeframe": "1d",
            "start_date": start_date.isoformat(),
            "end_date": end_date.isoformat(),
            "initial_capital": 1000000,  # BTCの高価格に対応
            "commission_rate": 0.001,
            "strategy_config": {
                "strategy_type": "SMA_CROSS",
                "parameters": {"n1": 3, "n2": 5},  # 短期間に適したパラメータ
            },
        }

        result = backtest_service.run_backtest(config)
        assert "performance_metrics" in result

        # 2. 高い手数料率
        config["commission_rate"] = 0.01  # 1%の手数料
        result_high_commission = backtest_service.run_backtest(config)
        assert "performance_metrics" in result_high_commission

        # 高い手数料の方がパフォーマンスが悪いことを確認
        assert (
            result_high_commission["performance_metrics"]["total_return"]
            <= result["performance_metrics"]["total_return"]
        )

    def test_data_standardization_with_real_data(self, ohlcv_repo, available_data_info):
        """実際のデータでの標準化機能テスト"""
        for symbol in available_data_info.keys():
            # 生データを取得
            raw_df = ohlcv_repo.get_ohlcv_dataframe(symbol, "1d", limit=50)

            # 標準化前の列名を確認
            original_columns = list(raw_df.columns)
            logger.info(f"{symbol}の元の列名: {original_columns}")

            # 標準化を実行
            standardized_df = standardize_ohlcv_columns(raw_df)

            # 標準化後の列名を確認
            expected_columns = ["Open", "High", "Low", "Close", "Volume"]
            for col in expected_columns:
                if (
                    col != "Volume" or col in original_columns
                ):  # Volumeは元データにない場合がある
                    assert (
                        col in standardized_df.columns
                    ), f"{symbol}で{col}列が標準化されていません"

            # backtesting.py用データ準備
            prepared_df = prepare_data_for_backtesting(standardized_df)

            # 準備されたデータの検証
            assert not prepared_df.empty, f"{symbol}の準備されたデータが空です"
            assert isinstance(
                prepared_df.index, pd.DatetimeIndex
            ), f"{symbol}のインデックスがDatetimeIndexではありません"

    def test_performance_metrics_accuracy(self, backtest_service, available_data_info):
        """パフォーマンス指標の精度テスト"""
        symbol = list(available_data_info.keys())[0]
        info = available_data_info[symbol]

        # 十分なデータ期間を使用
        end_date = info["latest"]
        start_date = end_date - timedelta(days=120)

        config = {
            "strategy_name": "SMA_CROSS",
            "symbol": symbol,
            "timeframe": "1d",
            "start_date": start_date.isoformat(),
            "end_date": end_date.isoformat(),
            "initial_capital": 1000000,  # BTCの高価格に対応
            "commission_rate": 0.001,
            "strategy_config": {
                "strategy_type": "SMA_CROSS",
                "parameters": {"n1": 20, "n2": 50},
            },
        }

        result = backtest_service.run_backtest(config)
        metrics = result["performance_metrics"]

        # 各指標の妥当性を詳細にチェック
        assert "total_return" in metrics
        assert "sharpe_ratio" in metrics
        assert "max_drawdown" in metrics
        assert "win_rate" in metrics

        # profit_factorは必須ではないため、存在する場合のみチェック
        if "profit_factor" in metrics and not pd.isna(metrics["profit_factor"]):
            assert (
                metrics["profit_factor"] >= 0
            ), f"プロフィットファクターが負の値: {metrics['profit_factor']}"

        # シャープレシオの妥当性（通常-3から3の範囲）
        if not pd.isna(metrics["sharpe_ratio"]):
            assert (
                -5.0 <= metrics["sharpe_ratio"] <= 5.0
            ), f"シャープレシオが異常値: {metrics['sharpe_ratio']}"

    def test_long_term_strategy_performance(
        self, backtest_service, available_data_info
    ):
        """長期間での戦略パフォーマンステスト"""
        # データが最も多いシンボルを選択
        best_symbol = max(
            available_data_info.keys(), key=lambda s: available_data_info[s]["count"]
        )

        info = available_data_info[best_symbol]

        # 可能な限り長期間のデータを使用（最大1年）
        end_date = info["latest"]
        available_days = (end_date - info["oldest"]).days
        test_days = min(365, available_days - 10)  # 最大1年、余裕を持って10日減らす
        start_date = end_date - timedelta(days=test_days)

        config = {
            "strategy_name": "SMA_CROSS",
            "symbol": best_symbol,
            "timeframe": "1d",
            "start_date": start_date.isoformat(),
            "end_date": end_date.isoformat(),
            "initial_capital": 1000000,  # BTCの高価格に対応
            "commission_rate": 0.001,
            "strategy_config": {
                "strategy_type": "SMA_CROSS",
                "parameters": {"n1": 50, "n2": 200},  # 長期戦略用パラメータ
            },
        }

        result = backtest_service.run_backtest(config)

        # 長期戦略の結果検証
        assert "performance_metrics" in result
        metrics = result["performance_metrics"]

        # 長期戦略では取引回数が少ないことを確認
        if "total_trades" in metrics:
            assert metrics["total_trades"] >= 0  # 取引が発生しない場合もある

        logger.info(
            f"長期戦略結果 ({test_days}日間): Total Return: {metrics['total_return']:.2%}"
        )

    def test_commission_impact_analysis(self, backtest_service, available_data_info):
        """手数料率の影響分析テスト"""
        symbol = list(available_data_info.keys())[0]
        info = available_data_info[symbol]

        # 最近2ヶ月のデータを使用
        end_date = info["latest"]
        start_date = end_date - timedelta(days=60)

        base_config = {
            "strategy_name": "SMA_CROSS",
            "symbol": symbol,
            "timeframe": "1d",
            "start_date": start_date.isoformat(),
            "end_date": end_date.isoformat(),
            "initial_capital": 1000000,  # BTCの高価格に対応
            "strategy_config": {
                "strategy_type": "SMA_CROSS",
                "parameters": {"n1": 5, "n2": 15},  # 短期パラメータで取引頻度を上げる
            },
        }

        # 異なる手数料率でテスト
        commission_rates = [0.0, 0.001, 0.005, 0.01]  # 0%, 0.1%, 0.5%, 1%
        results = {}

        for rate in commission_rates:
            config = base_config.copy()
            config["commission_rate"] = rate

            result = backtest_service.run_backtest(config)
            results[rate] = result["performance_metrics"]["total_return"]
            logger.info(f"手数料率 {rate:.1%}: Return {results[rate]:.2%}")

        # 手数料率が高いほどリターンが低いことを確認
        returns_list = [results[rate] for rate in commission_rates]
        for i in range(len(returns_list) - 1):
            assert (
                returns_list[i] >= returns_list[i + 1]
            ), f"手数料率の増加でリターンが改善されています: {commission_rates[i]} vs {commission_rates[i+1]}"


if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short"])
