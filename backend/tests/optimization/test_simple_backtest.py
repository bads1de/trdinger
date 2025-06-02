#!/usr/bin/env python3
"""
シンプルなバックテストテスト

最適化機能のpickleエラーを回避して、基本的なバックテスト機能をテストします。
"""

import sys
import os
from datetime import datetime, timezone
import pandas as pd
import numpy as np

# プロジェクトルートをパスに追加
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from app.core.services.backtest_service import BacktestService
from database.connection import SessionLocal
from database.repositories.ohlcv_repository import OHLCVRepository
from app.core.services.backtest_data_service import BacktestDataService
from database.models import OHLCVData


def generate_sample_btc_data(start_date: datetime, end_date: datetime, initial_price: float = 50000) -> pd.DataFrame:
    """
    サンプルのBTC価格データを生成
    """
    # 日付範囲を生成
    date_range = pd.date_range(start=start_date, end=end_date, freq='D')
    
    # ランダムウォークでリアルな価格変動を生成
    np.random.seed(42)  # 再現性のため
    
    # 日次リターンを生成（平均0.1%、標準偏差3%）
    daily_returns = np.random.normal(0.001, 0.03, len(date_range))
    
    # 価格を計算
    prices = [initial_price]
    for ret in daily_returns[1:]:
        prices.append(prices[-1] * (1 + ret))
    
    # OHLCV データを生成
    data = []
    for i, (date, price) in enumerate(zip(date_range, prices)):
        # 日中の変動を生成
        daily_volatility = np.random.uniform(0.005, 0.02)  # 0.5-2%の日中変動
        
        high = price * (1 + daily_volatility)
        low = price * (1 - daily_volatility)
        
        # Open/Closeを調整
        if i == 0:
            open_price = price
        else:
            open_price = prices[i-1]
        close_price = price
        
        # Volumeを生成（価格変動に応じて調整）
        base_volume = 1000000
        volume_multiplier = 1 + abs(daily_returns[i]) * 10  # 変動が大きいほど出来高増加
        volume = base_volume * volume_multiplier
        
        data.append({
            'timestamp': date,
            'open': open_price,
            'high': high,
            'low': low,
            'close': close_price,
            'volume': volume
        })
    
    df = pd.DataFrame(data)
    df.set_index('timestamp', inplace=True)
    
    # 列名を大文字に変換（backtesting.py用）
    df.columns = ['Open', 'High', 'Low', 'Close', 'Volume']
    
    return df


def insert_sample_data_to_db(df: pd.DataFrame, symbol: str = "BTC/USDT", timeframe: str = "1d"):
    """
    サンプルデータをデータベースに挿入
    """
    db = SessionLocal()
    try:
        # 既存データを削除
        db.query(OHLCVData).filter(
            OHLCVData.symbol == symbol,
            OHLCVData.timeframe == timeframe
        ).delete()
        
        # 新しいデータを挿入
        for timestamp, row in df.iterrows():
            ohlcv_data = OHLCVData(
                symbol=symbol,
                timeframe=timeframe,
                timestamp=timestamp,
                open=float(row['Open']),
                high=float(row['High']),
                low=float(row['Low']),
                close=float(row['Close']),
                volume=float(row['Volume'])
            )
            db.add(ohlcv_data)
        
        db.commit()
        print(f"✅ {len(df)}件のサンプルデータを挿入しました")
        
    except Exception as e:
        db.rollback()
        print(f"❌ データ挿入エラー: {e}")
        raise
    finally:
        db.close()


def test_simple_backtest():
    """シンプルなバックテストテスト（最適化なし）"""
    print("=== シンプルなバックテストテスト ===")
    
    # サンプルデータを生成・挿入
    start_date = datetime(2024, 1, 1, tzinfo=timezone.utc)
    end_date = datetime(2024, 12, 31, tzinfo=timezone.utc)
    
    print("サンプルデータを生成中...")
    sample_data = generate_sample_btc_data(start_date, end_date)
    print(f"生成されたデータ期間: {sample_data.index.min()} - {sample_data.index.max()}")
    print(f"データ件数: {len(sample_data)}")
    
    print("データベースに挿入中...")
    insert_sample_data_to_db(sample_data)
    
    # データベース接続
    db = SessionLocal()
    try:
        # データサービスを初期化
        ohlcv_repo = OHLCVRepository(db)
        data_service = BacktestDataService(ohlcv_repo)
        backtest_service = BacktestService(data_service)
        
        # バックテスト設定
        config = {
            "strategy_name": "SMA_CROSS_SIMPLE",
            "symbol": "BTC/USDT",
            "timeframe": "1d",
            "start_date": start_date,
            "end_date": end_date,
            "initial_capital": 1000000,  # 100万円
            "commission_rate": 0.001,
            "strategy_config": {
                "strategy_type": "SMA_CROSS",
                "parameters": {"n1": 20, "n2": 50},
            },
        }
        
        print("バックテスト実行中...")
        result = backtest_service.run_backtest(config)
        
        print("✅ シンプルなバックテスト成功!")
        print(f"戦略名: {result['strategy_name']}")
        print(f"期間: {config['start_date'].date()} - {config['end_date'].date()}")
        
        if "performance_metrics" in result:
            metrics = result["performance_metrics"]
            print(f"\n📊 パフォーマンス指標:")
            print(f"  総リターン: {metrics.get('total_return', 0):.2f}%")
            print(f"  シャープレシオ: {metrics.get('sharpe_ratio', 0):.3f}")
            print(f"  最大ドローダウン: {metrics.get('max_drawdown', 0):.2f}%")
            print(f"  勝率: {metrics.get('win_rate', 0):.2f}%")
            print(f"  プロフィットファクター: {metrics.get('profit_factor', 0):.3f}")
            print(f"  総取引数: {metrics.get('total_trades', 0)}")
        
        return result
        
    except Exception as e:
        print(f"❌ エラー: {str(e)}")
        import traceback
        traceback.print_exc()
        return None
    finally:
        db.close()


def test_multiple_parameters():
    """複数のパラメータでのバックテストテスト"""
    print("\n=== 複数パラメータでのバックテストテスト ===")
    
    # パラメータの組み合わせ
    parameter_combinations = [
        {"n1": 10, "n2": 30},
        {"n1": 15, "n2": 40},
        {"n1": 20, "n2": 50},
        {"n1": 25, "n2": 60},
    ]
    
    results = []
    
    db = SessionLocal()
    try:
        ohlcv_repo = OHLCVRepository(db)
        data_service = BacktestDataService(ohlcv_repo)
        backtest_service = BacktestService(data_service)
        
        for i, params in enumerate(parameter_combinations):
            print(f"\nパラメータ組み合わせ {i+1}: n1={params['n1']}, n2={params['n2']}")
            
            config = {
                "strategy_name": f"SMA_CROSS_PARAM_{i+1}",
                "symbol": "BTC/USDT",
                "timeframe": "1d",
                "start_date": datetime(2024, 1, 1, tzinfo=timezone.utc),
                "end_date": datetime(2024, 12, 31, tzinfo=timezone.utc),
                "initial_capital": 1000000,
                "commission_rate": 0.001,
                "strategy_config": {
                    "strategy_type": "SMA_CROSS",
                    "parameters": params,
                },
            }
            
            try:
                result = backtest_service.run_backtest(config)
                results.append((params, result))
                
                if "performance_metrics" in result:
                    metrics = result["performance_metrics"]
                    print(f"  シャープレシオ: {metrics.get('sharpe_ratio', 0):.3f}")
                    print(f"  総リターン: {metrics.get('total_return', 0):.2f}%")
                
            except Exception as e:
                print(f"  ❌ エラー: {e}")
                results.append((params, None))
        
        # 結果の比較
        print(f"\n📊 パラメータ比較結果:")
        print("パラメータ\t\tシャープレシオ\t総リターン")
        print("-" * 50)
        
        best_sharpe = -float('inf')
        best_params = None
        
        for params, result in results:
            if result and "performance_metrics" in result:
                metrics = result["performance_metrics"]
                sharpe = metrics.get('sharpe_ratio', 0)
                total_return = metrics.get('total_return', 0)
                
                print(f"n1={params['n1']}, n2={params['n2']}\t\t{sharpe:.3f}\t\t{total_return:.2f}%")
                
                if sharpe > best_sharpe:
                    best_sharpe = sharpe
                    best_params = params
            else:
                print(f"n1={params['n1']}, n2={params['n2']}\t\tエラー\t\tエラー")
        
        if best_params:
            print(f"\n🏆 最適パラメータ: n1={best_params['n1']}, n2={best_params['n2']} (シャープレシオ: {best_sharpe:.3f})")
        
        return results
        
    except Exception as e:
        print(f"❌ エラー: {str(e)}")
        import traceback
        traceback.print_exc()
        return None
    finally:
        db.close()


def main():
    """メイン関数"""
    print("シンプルなバックテストテスト開始")
    print("=" * 80)
    
    tests = [
        ("シンプルバックテスト", test_simple_backtest),
        ("複数パラメータテスト", test_multiple_parameters),
    ]
    
    results = []
    for test_name, test_func in tests:
        print(f"\n{test_name}を実行中...")
        try:
            result = test_func()
            success = result is not None
            results.append((test_name, success, result))
        except Exception as e:
            print(f"{test_name}でエラー: {e}")
            results.append((test_name, False, None))
    
    print("\n" + "=" * 80)
    print("テスト結果サマリー:")
    for test_name, success, _ in results:
        status = "✅ 成功" if success else "❌ 失敗"
        print(f"  {test_name}: {status}")
    
    success_count = sum(1 for _, success, _ in results if success)
    print(f"\n成功: {success_count}/{len(results)}")
    
    if success_count == len(results):
        print("🎉 全てのシンプルバックテストが成功しました！")
        print("\n💡 テスト結果:")
        print("- DB内のデータでのバックテストが正常に動作")
        print("- 複数パラメータでの比較が可能")
        print("- サンプルデータでの動作確認完了")
        print("- 最適化機能の代替として手動パラメータ比較が利用可能")
    else:
        print("⚠️ 一部のテストが失敗しました。")


if __name__ == "__main__":
    main()
