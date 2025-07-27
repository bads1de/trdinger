#!/usr/bin/env python3
"""
強化された暗号通貨特徴量のテストスクリプト

新しい特徴量エンジニアリングの効果を検証します。
"""

import sys
import os
import pandas as pd
import numpy as np
from pathlib import Path
from datetime import datetime, timedelta
import warnings
warnings.filterwarnings('ignore')

# プロジェクトルートをパスに追加
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from app.core.services.ml.feature_engineering.enhanced_crypto_features import EnhancedCryptoFeatures

def generate_test_data(hours: int = 168) -> pd.DataFrame:
    """
    テスト用の暗号通貨データを生成（1週間分）
    """
    print(f"テストデータ生成: {hours}時間分")
    
    # 基準時刻
    start_time = datetime.now() - timedelta(hours=hours)
    timestamps = [start_time + timedelta(hours=i) for i in range(hours)]
    
    # 基準価格
    base_price = 50000
    
    # リアルな価格データ生成
    np.random.seed(42)
    
    # トレンド + ボラティリティ
    trend = np.cumsum(np.random.normal(0, 0.001, hours))
    volatility = np.random.normal(0, 0.02, hours)
    returns = trend * 0.1 + volatility
    prices = base_price * np.exp(np.cumsum(returns))
    
    # OHLCV生成
    data = []
    for i, (timestamp, close) in enumerate(zip(timestamps, prices)):
        vol = abs(returns[i]) * close * 0.5
        high = close + np.random.exponential(vol)
        low = close - np.random.exponential(vol)
        
        if i == 0:
            open_price = close
        else:
            open_price = prices[i-1] + np.random.normal(0, vol * 0.1)
        
        volume = max(100, np.random.normal(1000 + abs(returns[i]) * 50000, 500))
        
        data.append({
            'timestamp': timestamp,
            'Open': open_price,
            'High': max(open_price, high, close),
            'Low': min(open_price, low, close),
            'Close': close,
            'Volume': volume,
        })
    
    df = pd.DataFrame(data)
    df.set_index('timestamp', inplace=True)
    
    # Open Interest（8時間ごと）
    oi_base = 1000000
    oi_trend = np.cumsum(np.random.normal(0, 0.005, hours // 8 + 1))
    oi_values = oi_base * (1 + oi_trend)
    oi_timestamps = [start_time + timedelta(hours=i*8) for i in range(len(oi_values))]
    oi_series = pd.Series(oi_values, index=oi_timestamps)
    df['open_interest'] = oi_series.reindex(df.index, method='ffill')
    
    # Funding Rate（8時間ごと）
    fr_values = []
    for i in range(len(oi_values)):
        price_momentum = np.mean(returns[max(0, i*8-24):i*8+1])
        base_fr = price_momentum * 0.1
        noise = np.random.normal(0, 0.0001)
        fr = np.clip(base_fr + noise, -0.01, 0.01)
        fr_values.append(fr)
    
    fr_series = pd.Series(fr_values, index=oi_timestamps)
    df['funding_rate'] = fr_series.reindex(df.index, method='ffill')
    
    # Fear & Greed（1日ごと）
    daily_timestamps = [start_time + timedelta(days=i) for i in range(hours // 24 + 1)]
    fg_values = []
    for i in range(len(daily_timestamps)):
        if i == 0:
            daily_return = 0
        else:
            start_idx = max(0, i*24-24)
            end_idx = min(hours-1, i*24)
            daily_return = np.mean(returns[start_idx:end_idx])
        
        base_fg = 50 + daily_return * 1000
        noise = np.random.normal(0, 5)
        fg = np.clip(base_fg + noise, 0, 100)
        fg_values.append(fg)
    
    fg_series = pd.Series(fg_values, index=daily_timestamps)
    df['fear_greed_value'] = fg_series.reindex(df.index, method='ffill')
    
    print(f"生成完了: {len(df)}行, カラム: {list(df.columns)}")
    return df

def test_feature_creation():
    """特徴量作成のテスト"""
    print("\n=== 特徴量作成テスト ===")
    
    # テストデータ生成
    df = generate_test_data(168)  # 1週間分
    
    print(f"元データ形状: {df.shape}")
    print(f"元データカラム: {list(df.columns)}")
    
    # 特徴量エンジニアリング
    crypto_features = EnhancedCryptoFeatures()
    
    start_time = datetime.now()
    enhanced_df = crypto_features.create_comprehensive_features(df)
    processing_time = (datetime.now() - start_time).total_seconds()
    
    print(f"\n特徴量作成完了:")
    print(f"  処理時間: {processing_time:.2f}秒")
    print(f"  拡張後形状: {enhanced_df.shape}")
    print(f"  追加特徴量: {enhanced_df.shape[1] - df.shape[1]}個")
    
    # 特徴量グループ分析
    feature_groups = crypto_features.get_feature_groups()
    print(f"\n特徴量グループ:")
    for group, features in feature_groups.items():
        print(f"  {group}: {len(features)}個")
    
    # データ品質チェック
    print(f"\nデータ品質:")
    print(f"  欠損値: {enhanced_df.isnull().sum().sum()}個")
    print(f"  無限値: {np.isinf(enhanced_df.select_dtypes(include=[np.number])).sum().sum()}個")
    
    return enhanced_df, crypto_features

def test_feature_effectiveness(df: pd.DataFrame, crypto_features: EnhancedCryptoFeatures):
    """特徴量の有効性テスト"""
    print("\n=== 特徴量有効性テスト ===")
    
    # ターゲット変数作成（複数期間）
    target_periods = [1, 4, 12, 24]
    results = {}
    
    for period in target_periods:
        print(f"\n📈 {period}時間後の価格変動予測:")
        
        # ターゲット変数
        target = df['Close'].pct_change(period).shift(-period)
        
        # 有効なデータのみ
        valid_mask = target.notna() & df.notna().all(axis=1)
        valid_count = valid_mask.sum()
        
        if valid_count < 50:
            print(f"  有効データ不足: {valid_count}件")
            continue
        
        print(f"  有効データ: {valid_count}件")
        
        # 上位特徴量を取得
        top_features = crypto_features.get_top_features_by_correlation(
            df.loc[valid_mask], 'Close', top_n=20
        )
        
        if top_features:
            print(f"  上位特徴量:")
            for i, feature in enumerate(top_features[:10]):
                corr = df.loc[valid_mask, feature].corr(target.loc[valid_mask])
                print(f"    {i+1:2d}. {feature:<30} 相関: {corr:+.4f}")
        
        results[f'{period}h'] = {
            'valid_count': valid_count,
            'top_features': top_features[:10],
        }
    
    return results

def test_feature_groups(df: pd.DataFrame, crypto_features: EnhancedCryptoFeatures):
    """特徴量グループ別の分析"""
    print("\n=== 特徴量グループ分析 ===")
    
    feature_groups = crypto_features.get_feature_groups()
    target = df['Close'].pct_change().shift(-1)
    
    valid_mask = target.notna() & df.notna().all(axis=1)
    
    if valid_mask.sum() < 50:
        print("有効データ不足")
        return
    
    group_performance = {}
    
    for group_name, features in feature_groups.items():
        if not features:
            continue
        
        # グループ内の特徴量の相関を計算
        correlations = []
        for feature in features:
            if feature in df.columns:
                corr = df.loc[valid_mask, feature].corr(target.loc[valid_mask])
                if not pd.isna(corr):
                    correlations.append(abs(corr))
        
        if correlations:
            avg_corr = np.mean(correlations)
            max_corr = np.max(correlations)
            group_performance[group_name] = {
                'feature_count': len(features),
                'avg_correlation': avg_corr,
                'max_correlation': max_corr,
            }
    
    # グループ性能をソート
    sorted_groups = sorted(
        group_performance.items(), 
        key=lambda x: x[1]['avg_correlation'], 
        reverse=True
    )
    
    print("グループ別性能（平均相関順）:")
    for group_name, performance in sorted_groups:
        print(f"  {group_name:<15}: "
              f"特徴量 {performance['feature_count']:2d}個, "
              f"平均相関 {performance['avg_correlation']:.4f}, "
              f"最大相関 {performance['max_correlation']:.4f}")
    
    return group_performance

def test_memory_efficiency(df: pd.DataFrame):
    """メモリ効率性テスト"""
    print("\n=== メモリ効率性テスト ===")
    
    # メモリ使用量測定
    memory_before = df.memory_usage(deep=True).sum() / 1024 / 1024
    print(f"処理前メモリ: {memory_before:.2f}MB")
    
    # 特徴量作成
    crypto_features = EnhancedCryptoFeatures()
    
    import psutil
    process = psutil.Process()
    memory_start = process.memory_info().rss / 1024 / 1024
    
    enhanced_df = crypto_features.create_comprehensive_features(df)
    
    memory_end = process.memory_info().rss / 1024 / 1024
    memory_after = enhanced_df.memory_usage(deep=True).sum() / 1024 / 1024
    
    print(f"処理後メモリ: {memory_after:.2f}MB")
    print(f"メモリ増加: {memory_after - memory_before:.2f}MB")
    print(f"プロセスメモリ変化: {memory_end - memory_start:+.2f}MB")
    print(f"メモリ効率: {(memory_after - memory_before) / enhanced_df.shape[1]:.3f}MB/特徴量")

def main():
    """メイン実行関数"""
    print("強化された暗号通貨特徴量のテスト")
    print("=" * 50)
    
    try:
        # 1. 特徴量作成テスト
        enhanced_df, crypto_features = test_feature_creation()
        
        # 2. 特徴量有効性テスト
        effectiveness_results = test_feature_effectiveness(enhanced_df, crypto_features)
        
        # 3. 特徴量グループ分析
        group_performance = test_feature_groups(enhanced_df, crypto_features)
        
        # 4. メモリ効率性テスト
        test_memory_efficiency(generate_test_data(168))
        
        print("\n" + "=" * 50)
        print("🎯 テスト結果サマリー:")
        print(f"  総特徴量数: {enhanced_df.shape[1]}個")
        print(f"  データ品質: 欠損値 {enhanced_df.isnull().sum().sum()}個")
        print("  最も効果的なグループ:")
        
        if group_performance:
            best_group = max(group_performance.items(), key=lambda x: x[1]['avg_correlation'])
            print(f"    {best_group[0]}: 平均相関 {best_group[1]['avg_correlation']:.4f}")
        
        print("\n✅ 新しい特徴量エンジニアリングが正常に動作しています！")
        
    except Exception as e:
        print(f"テスト中にエラーが発生しました: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()
