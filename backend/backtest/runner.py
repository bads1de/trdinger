#!/usr/bin/env python3
"""
バックテスト実行スクリプト
Next.js API Routeから呼び出される

backtesting.pyライブラリを使用した統一実装
"""
import sys
import json
import pandas as pd
import numpy as np
from datetime import datetime
from app.core.services.backtest_service import BacktestService
from database.connection import SessionLocal
from database.repositories.ohlcv_repository import OHLCVRepository


def generate_sample_data(
    start_date: str, end_date: str, symbol: str = "BTC/USD"
) -> pd.DataFrame:
    """
    サンプルデータを生成（実際の実装では外部APIやデータベースから取得）
    """
    start = datetime.fromisoformat(start_date.replace("Z", "+00:00"))
    end = datetime.fromisoformat(end_date.replace("Z", "+00:00"))

    # 1分足データを生成
    date_range = pd.date_range(start=start, end=end, freq="1min")

    # ランダムウォークで価格データを生成
    np.random.seed(42)  # 再現性のため
    base_price = 50000 if symbol.startswith("BTC") else 50000  # BTC only (ETH excluded)

    # より現実的な価格変動を生成
    returns = np.random.normal(0, 0.001, len(date_range))  # 0.1%の標準偏差
    price_changes = np.exp(returns.cumsum())
    prices = base_price * price_changes

    # OHLCV データを生成
    data = []
    for i, timestamp in enumerate(date_range):
        price = prices[i]
        # 高値・安値・始値・終値を生成
        high = price * (1 + abs(np.random.normal(0, 0.002)))
        low = price * (1 - abs(np.random.normal(0, 0.002)))
        open_price = price + np.random.normal(0, price * 0.001)
        close_price = price
        volume = np.random.randint(100, 1000)

        data.append(
            {
                "timestamp": timestamp,
                "Open": open_price,
                "High": max(open_price, high, close_price),
                "Low": min(open_price, low, close_price),
                "Close": close_price,
                "Volume": volume,
            }
        )

    df = pd.DataFrame(data)
    df.set_index("timestamp", inplace=True)
    return df


def get_real_data(
    symbol: str, start_date: str, end_date: str, timeframe: str = "1d"
) -> pd.DataFrame:
    """
    データベースから実際のOHLCVデータを取得
    """
    try:
        # 日付文字列をdatetimeに変換
        start_dt = datetime.fromisoformat(start_date.replace("Z", "+00:00"))
        end_dt = datetime.fromisoformat(end_date.replace("Z", "+00:00"))

        # データベースセッション
        db = SessionLocal()
        try:
            ohlcv_repo = OHLCVRepository(db)

            # データを取得
            df = ohlcv_repo.get_ohlcv_dataframe(
                symbol=symbol, timeframe=timeframe, start_time=start_dt, end_time=end_dt
            )

            if df.empty:
                print(
                    "警告: データベースにデータが見つかりません。サンプルデータを使用します。"
                )
                return generate_sample_data(start_date, end_date, symbol)

            print(f"データベースから{len(df)}件のデータを取得しました")
            return df

        finally:
            db.close()

    except Exception as e:
        print(f"データベースからのデータ取得エラー: {e}")
        print("サンプルデータを使用します")
        return generate_sample_data(start_date, end_date, symbol)


def run_backtest(config: dict) -> dict:
    """
    バックテストを実行
    """
    try:
        # データを取得（まずデータベースから、なければサンプルデータ）
        get_real_data(
            config["strategy"]["target_pair"],
            config["start_date"],
            config["end_date"],
            config.get("timeframe", "1d"),
        )

        # BacktestServiceを使用した新しい実装
        backtest_service = BacktestService()

        # 設定をBacktestService形式に変換
        backtest_config = {
            "strategy_name": config["strategy"].get("name", "SMA_CROSS"),
            "symbol": config["strategy"]["target_pair"],
            "timeframe": config.get("timeframe", "1d"),
            "start_date": config["start_date"],
            "end_date": config["end_date"],
            "initial_capital": config.get("initial_capital", 100000),
            "commission_rate": config.get("commission_rate", 0.001),
            "strategy_config": {
                "strategy_type": "SMA_CROSS",
                "parameters": {"n1": 20, "n2": 50},
            },
        }

        # バックテストを実行
        result = backtest_service.run_backtest(backtest_config)

        # BacktestServiceの結果は既に適切な形式なので、そのまま返す
        # 必要に応じて追加のフィールドを設定
        backtest_result = result.copy()
        backtest_result.update(
            {
                "id": str(int(datetime.now().timestamp())),
                "strategy_id": config["strategy"].get("id", "unknown"),
                "config": config,
                "created_at": datetime.now().isoformat(),
            }
        )

        return backtest_result

    except Exception as e:
        return {"error": str(e), "traceback": str(e.__traceback__)}


def main():
    """
    標準入力からJSONを読み取り、結果を標準出力に出力
    """
    try:
        # 標準入力からJSONを読み取り
        input_data = sys.stdin.read()
        config = json.loads(input_data)

        # バックテストを実行
        result = run_backtest(config)

        # 結果をJSONとして出力
        print(json.dumps(result, ensure_ascii=False, indent=2))

    except Exception as e:
        error_result = {
            "error": f"バックテスト実行エラー: {str(e)}",
            "type": type(e).__name__,
        }
        print(json.dumps(error_result, ensure_ascii=False))
        sys.exit(1)


if __name__ == "__main__":
    main()
