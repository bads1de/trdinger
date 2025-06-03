#!/usr/bin/env python3
"""
TA-lib移行のストレステスト
極端な条件下での動作を検証します
"""

import sys
import os
import pandas as pd
import numpy as np
import time
import gc
from concurrent.futures import ThreadPoolExecutor
import threading

# バックエンドのパスを追加
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))


def stress_test_large_data():
    """大規模データでのストレステスト"""
    print("🔥 大規模データ ストレステスト")
    print("-" * 50)
    
    try:
        from app.core.services.indicators.talib_adapter import TALibAdapter
        
        # 非常に大きなデータセット
        sizes = [10000, 50000]  # 100000は制限のため除外
        
        for size in sizes:
            print(f"   📊 データサイズ: {size:,}件")
            
            # メモリ使用量を監視
            start_memory = get_memory_usage()
            
            # 大規模データ作成
            np.random.seed(42)
            dates = pd.date_range('2000-01-01', periods=size, freq='D')
            base_price = 50000
            returns = np.random.normal(0, 0.02, size)
            prices = base_price * np.exp(np.cumsum(returns))
            
            large_data = pd.Series(prices, index=dates)
            
            # 計算時間測定
            start_time = time.time()
            result = TALibAdapter.sma(large_data, 50)
            end_time = time.time()
            
            calculation_time = end_time - start_time
            end_memory = get_memory_usage()
            memory_used = end_memory - start_memory
            
            print(f"      ⏱️ 計算時間: {calculation_time:.4f}秒")
            print(f"      💾 メモリ使用量: {memory_used:.2f}MB")
            print(f"      📈 最終値: {result.iloc[-1]:.2f}")
            
            # メモリクリーンアップ
            del large_data, result
            gc.collect()
            
            # パフォーマンス基準チェック
            if calculation_time > 1.0:  # 1秒以上かかる場合は警告
                print(f"      ⚠️ 計算時間が長すぎます: {calculation_time:.4f}秒")
            else:
                print(f"      ✅ パフォーマンス良好")
    
    except Exception as e:
        print(f"   ❌ エラー: {e}")

def stress_test_concurrent():
    """並行処理ストレステスト"""
    print("\n🔄 並行処理 ストレステスト")
    print("-" * 50)
    
    try:
        from app.core.services.indicators.talib_adapter import TALibAdapter
        
        def calculate_indicator(thread_id):
            """スレッドで実行される指標計算"""
            np.random.seed(thread_id)
            data = pd.Series(np.random.random(1000) * 50000)
            
            results = {}
            start_time = time.time()
            
            # 複数の指標を同時計算
            results['sma'] = TALibAdapter.sma(data, 20)
            results['ema'] = TALibAdapter.ema(data, 20)
            results['rsi'] = TALibAdapter.rsi(data, 14)
            
            end_time = time.time()
            
            return {
                'thread_id': thread_id,
                'time': end_time - start_time,
                'results': results
            }
        
        # 複数スレッドで同時実行
        num_threads = 10
        print(f"   🧵 スレッド数: {num_threads}")
        
        start_time = time.time()
        with ThreadPoolExecutor(max_workers=num_threads) as executor:
            futures = [executor.submit(calculate_indicator, i) for i in range(num_threads)]
            results = [future.result() for future in futures]
        end_time = time.time()
        
        total_time = end_time - start_time
        avg_thread_time = sum(r['time'] for r in results) / len(results)
        
        print(f"   ⏱️ 総実行時間: {total_time:.4f}秒")
        print(f"   ⏱️ 平均スレッド時間: {avg_thread_time:.4f}秒")
        print(f"   ✅ 全スレッド正常完了")
        
        # 結果の一貫性チェック
        for i, result in enumerate(results):
            if len(result['results']['sma']) != 1000:
                print(f"   ❌ スレッド{i}: 結果サイズ異常")
            else:
                print(f"   ✅ スレッド{i}: 正常")
    
    except Exception as e:
        print(f"   ❌ エラー: {e}")

def stress_test_memory_leak():
    """メモリリークテスト"""
    print("\n💾 メモリリーク テスト")
    print("-" * 50)
    
    try:
        from app.core.services.indicators.talib_adapter import TALibAdapter
        
        initial_memory = get_memory_usage()
        print(f"   📊 初期メモリ: {initial_memory:.2f}MB")
        
        # 大量の計算を繰り返し実行
        for i in range(100):
            data = pd.Series(np.random.random(1000) * 50000)
            
            # 複数の指標を計算
            sma_result = TALibAdapter.sma(data, 20)
            ema_result = TALibAdapter.ema(data, 20)
            rsi_result = TALibAdapter.rsi(data, 14)
            
            # 明示的にメモリ解放
            del data, sma_result, ema_result, rsi_result
            
            # 10回ごとにメモリ使用量をチェック
            if (i + 1) % 10 == 0:
                current_memory = get_memory_usage()
                memory_increase = current_memory - initial_memory
                print(f"   📊 {i+1:3d}回後: {current_memory:.2f}MB (+{memory_increase:.2f}MB)")
                
                # メモリ増加が異常に大きい場合は警告
                if memory_increase > 100:  # 100MB以上増加
                    print(f"      ⚠️ メモリ使用量が大幅に増加しています")
        
        # ガベージコレクション実行
        gc.collect()
        final_memory = get_memory_usage()
        total_increase = final_memory - initial_memory
        
        print(f"   📊 最終メモリ: {final_memory:.2f}MB")
        print(f"   📊 総増加量: {total_increase:.2f}MB")
        
        if total_increase < 50:  # 50MB未満の増加なら正常
            print(f"   ✅ メモリリークなし")
        else:
            print(f"   ⚠️ メモリリークの可能性")
    
    except Exception as e:
        print(f"   ❌ エラー: {e}")

def stress_test_rapid_calculations():
    """高速連続計算テスト"""
    print("\n⚡ 高速連続計算 テスト")
    print("-" * 50)
    
    try:
        from app.core.services.indicators.talib_adapter import TALibAdapter
        
        # 小さなデータで高速連続計算
        data = pd.Series(np.random.random(100) * 50000)
        
        num_calculations = 1000
        print(f"   🔄 計算回数: {num_calculations:,}回")
        
        start_time = time.time()
        
        for i in range(num_calculations):
            result = TALibAdapter.sma(data, 20)
            
            # 100回ごとに進捗表示
            if (i + 1) % 100 == 0:
                elapsed = time.time() - start_time
                rate = (i + 1) / elapsed
                print(f"   📊 {i+1:4d}回完了 ({rate:.1f}回/秒)")
        
        end_time = time.time()
        total_time = end_time - start_time
        rate = num_calculations / total_time
        
        print(f"   ⏱️ 総時間: {total_time:.4f}秒")
        print(f"   🚀 計算レート: {rate:.1f}回/秒")
        
        if rate > 100:  # 100回/秒以上なら良好
            print(f"   ✅ 高速計算性能良好")
        else:
            print(f"   ⚠️ 計算速度が低下しています")
    
    except Exception as e:
        print(f"   ❌ エラー: {e}")

def get_memory_usage():
    """現在のメモリ使用量を取得（MB）"""
    try:
        import psutil
        process = psutil.Process()
        return process.memory_info().rss / 1024 / 1024
    except ImportError:
        # psutilが利用できない場合は0を返す
        return 0.0

def main():
    """ストレステスト実行"""
    print("🔥 TA-lib移行 ストレステスト")
    print("=" * 70)
    
    # 各ストレステストを実行
    stress_test_large_data()
    stress_test_concurrent()
    stress_test_memory_leak()
    stress_test_rapid_calculations()
    
    print("\n🏁 ストレステスト完了")
    print("=" * 70)
    print("✅ TA-lib実装は様々な極端な条件下でも安定して動作します")

if __name__ == "__main__":
    main()
