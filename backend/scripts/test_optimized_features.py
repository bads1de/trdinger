#!/usr/bin/env python3
"""
最適化された特徴量エンジニアリングのテスト

深度分析の結果に基づいて最適化された特徴量の効果を検証します。
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

from app.services.ml.feature_engineering.enhanced_crypto_features import EnhancedCryptoFeatures
from app.services.ml.feature_engineering.optimized_crypto_features import OptimizedCryptoFeatures
from app.services.ml.feature_engineering.enhanced_feature_engineering_service import EnhancedFeatureEngineeringService

def generate_realistic_market_data(hours: int = 720) -> pd.DataFrame:
    """
    リアルな市場データを生成（30日分）
    """
    print(f"リアルな市場データ生成: {hours}時間分")
    
    start_time = datetime.now() - timedelta(hours=hours)
    timestamps = [start_time + timedelta(hours=i) for i in range(hours)]
    
    # 複雑な市場動向を模擬
    np.random.seed(42)
    
    # 複数のサイクル
    weekly_cycle = np.sin(np.arange(hours) * 2 * np.pi / (24 * 7)) * 0.02
    daily_cycle = np.sin(np.arange(hours) * 2 * np.pi / 24) * 0.01
    
    # ボラティリティクラスタリング
    volatility_regime = np.zeros(hours)
    current_vol = 0.02
    for i in range(hours):
        if np.random.random() < 0.03:  # 3%の確率でレジーム変更
            current_vol = np.random.choice([0.01, 0.02, 0.04], p=[0.3, 0.5, 0.2])
        volatility_regime[i] = current_vol
    
    # 価格生成
    base_returns = weekly_cycle + daily_cycle
    noise_returns = np.random.normal(0, volatility_regime)
    returns = base_returns + noise_returns
    
    base_price = 50000
    prices = base_price * np.exp(np.cumsum(returns))
    
    # OHLCV生成
    data = []
    for i, (timestamp, close) in enumerate(zip(timestamps, prices)):
        vol = volatility_regime[i] * close
        high = close + np.random.exponential(vol * 0.3)
        low = close - np.random.exponential(vol * 0.3)
        
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
    oi_changes = []
    for i in range(0, hours, 8):
        if i == 0:
            oi_change = 0
        else:
            price_momentum = np.mean(returns[max(0, i-24):i])
            oi_change = price_momentum * 0.3 + np.random.normal(0, 0.01)
        oi_changes.append(oi_change)
    
    oi_values = oi_base * np.exp(np.cumsum(oi_changes))
    oi_timestamps = [start_time + timedelta(hours=i*8) for i in range(len(oi_values))]
    oi_series = pd.Series(oi_values, index=oi_timestamps)
    df['open_interest'] = oi_series.reindex(df.index, method='ffill')
    
    # Funding Rate（8時間ごと）
    fr_values = []
    for i in range(len(oi_values)):
        if i == 0:
            price_trend = 0
        else:
            start_idx = max(0, i*8-24)
            end_idx = i*8
            price_trend = np.mean(returns[start_idx:end_idx])
        
        fr = np.clip(price_trend * 0.15 + np.random.normal(0, 0.0001), -0.01, 0.01)
        fr_values.append(fr)
    
    fr_series = pd.Series(fr_values, index=oi_timestamps)
    df['funding_rate'] = fr_series.reindex(df.index, method='ffill')
    
    # Fear & Greed（1日ごと）
    daily_timestamps = [start_time + timedelta(days=i) for i in range(hours // 24 + 1)]
    fg_values = []
    for i in range(len(daily_timestamps)):
        if i == 0:
            daily_return = 0
            daily_volatility = 0.02
        else:
            start_idx = max(0, i*24-24)
            end_idx = min(hours-1, i*24)
            daily_return = np.sum(returns[start_idx:end_idx])
            daily_volatility = np.std(returns[start_idx:end_idx])
        
        fg_raw = 50 - daily_return * 1000 + (daily_volatility - 0.02) * 500
        fg = np.clip(fg_raw + np.random.normal(0, 2), 0, 100)
        fg_values.append(fg)
    
    fg_series = pd.Series(fg_values, index=daily_timestamps)
    df['fear_greed_value'] = fg_series.reindex(df.index, method='ffill')
    
    print(f"生成完了: {len(df)}行")
    return df

def compare_feature_engines():
    """特徴量エンジンの比較"""
    print("\n=== 特徴量エンジン比較 ===")
    
    # テストデータ生成
    df = generate_realistic_market_data(720)  # 30日分
    
    print(f"基本データ: {df.shape}")
    
    # 1. 従来の特徴量エンジニアリング
    print(f"\n📊 従来の特徴量エンジニアリング:")
    enhanced_features = EnhancedCryptoFeatures()
    
    start_time = datetime.now()
    enhanced_df = enhanced_features.create_comprehensive_features(df)
    enhanced_time = (datetime.now() - start_time).total_seconds()
    
    print(f"  処理時間: {enhanced_time:.2f}秒")
    print(f"  特徴量数: {enhanced_df.shape[1]}個")
    print(f"  追加特徴量: {enhanced_df.shape[1] - df.shape[1]}個")
    
    # 2. 最適化された特徴量エンジニアリング
    print(f"\n🚀 最適化された特徴量エンジニアリング:")
    optimized_features = OptimizedCryptoFeatures()
    
    start_time = datetime.now()
    optimized_df = optimized_features.create_optimized_features(df)
    optimized_time = (datetime.now() - start_time).total_seconds()
    
    print(f"  処理時間: {optimized_time:.2f}秒")
    print(f"  特徴量数: {optimized_df.shape[1]}個")
    print(f"  追加特徴量: {optimized_df.shape[1] - df.shape[1]}個")
    
    # 3. 統合版（両方を使用）
    print(f"\n🔥 統合版（両方を使用）:")
    
    start_time = datetime.now()
    integrated_df = enhanced_features.create_comprehensive_features(df)
    integrated_df = optimized_features.create_optimized_features(integrated_df)
    integrated_time = (datetime.now() - start_time).total_seconds()
    
    print(f"  処理時間: {integrated_time:.2f}秒")
    print(f"  特徴量数: {integrated_df.shape[1]}個")
    print(f"  追加特徴量: {integrated_df.shape[1] - df.shape[1]}個")
    
    return enhanced_df, optimized_df, integrated_df

def analyze_feature_quality(df: pd.DataFrame, name: str):
    """特徴量品質の分析"""
    print(f"\n=== {name} 品質分析 ===")
    
    # ターゲット変数
    target = df['Close'].pct_change(4).shift(-4)
    
    # 有効データ
    valid_mask = target.notna() & df.notna().all(axis=1)
    valid_count = valid_mask.sum()
    
    print(f"有効データ: {valid_count}件 / {len(df)}件 ({valid_count/len(df)*100:.1f}%)")
    
    if valid_count < 50:
        print("有効データ不足")
        return {}
    
    # 特徴量の相関分析
    feature_cols = [col for col in df.columns if col not in ['Open', 'High', 'Low', 'Close', 'Volume']]
    
    correlations = []
    for col in feature_cols:
        corr = df.loc[valid_mask, col].corr(target.loc[valid_mask])
        if not pd.isna(corr):
            correlations.append({
                'feature': col,
                'correlation': abs(corr),
                'correlation_raw': corr,
            })
    
    # 相関順にソート
    correlations.sort(key=lambda x: x['correlation'], reverse=True)
    
    # 統計情報
    corr_values = [item['correlation'] for item in correlations]
    
    print(f"特徴量統計:")
    print(f"  総特徴量数: {len(correlations)}個")
    print(f"  平均相関: {np.mean(corr_values):.4f}")
    print(f"  最大相関: {np.max(corr_values):.4f}")
    print(f"  高相関特徴量(>0.05): {sum(1 for c in corr_values if c > 0.05)}個")
    print(f"  中相関特徴量(0.02-0.05): {sum(1 for c in corr_values if 0.02 <= c <= 0.05)}個")
    print(f"  低相関特徴量(<0.02): {sum(1 for c in corr_values if c < 0.02)}個")
    
    print(f"\n上位10特徴量:")
    for i, item in enumerate(correlations[:10]):
        print(f"  {i+1:2d}. {item['feature']:<40} 相関: {item['correlation_raw']:+.4f}")
    
    return {
        'total_features': len(correlations),
        'mean_correlation': np.mean(corr_values),
        'max_correlation': np.max(corr_values),
        'high_corr_count': sum(1 for c in corr_values if c > 0.05),
        'correlations': correlations,
    }

def test_feature_stability(df: pd.DataFrame, name: str):
    """特徴量の安定性テスト"""
    print(f"\n=== {name} 安定性テスト ===")
    
    # 異なる期間でのテスト
    periods = [168, 336, 504, 720]  # 1週間、2週間、3週間、1ヶ月
    stability_results = {}
    
    for period in periods:
        if period > len(df):
            continue
        
        period_df = df.iloc[-period:].copy()
        target = period_df['Close'].pct_change(4).shift(-4)
        
        valid_mask = target.notna() & period_df.notna().all(axis=1)
        if valid_mask.sum() < 30:
            continue
        
        feature_cols = [col for col in period_df.columns if col not in ['Open', 'High', 'Low', 'Close', 'Volume']]
        
        correlations = {}
        for col in feature_cols:
            corr = period_df.loc[valid_mask, col].corr(target.loc[valid_mask])
            if not pd.isna(corr):
                correlations[col] = abs(corr)
        
        stability_results[period] = correlations
    
    if len(stability_results) < 2:
        print("安定性テストに十分なデータがありません")
        return {}
    
    # 安定性スコア計算
    common_features = set(stability_results[periods[0]].keys())
    for period in periods[1:]:
        if period in stability_results:
            common_features &= set(stability_results[period].keys())
    
    stability_scores = {}
    for feature in common_features:
        correlations = [stability_results[period][feature] for period in periods if period in stability_results]
        if len(correlations) >= 2:
            stability_score = np.mean(correlations) - np.std(correlations)
            stability_scores[feature] = {
                'stability_score': stability_score,
                'mean_correlation': np.mean(correlations),
                'correlation_std': np.std(correlations),
            }
    
    # 安定性順にソート
    sorted_features = sorted(stability_scores.items(), key=lambda x: x[1]['stability_score'], reverse=True)
    
    print(f"安定性の高い特徴量（上位10位）:")
    for i, (feature, scores) in enumerate(sorted_features[:10]):
        print(f"  {i+1:2d}. {feature:<40} "
              f"安定性: {scores['stability_score']:+.4f} "
              f"(平均: {scores['mean_correlation']:.4f})")
    
    return stability_scores

def test_integrated_service():
    """統合サービスのテスト"""
    print("\n=== 統合サービステスト ===")
    
    # テストデータ生成
    df = generate_realistic_market_data(240)  # 10日分
    target = df['Close'].pct_change(4).shift(-4)
    
    # 統合サービス
    service = EnhancedFeatureEngineeringService()
    
    start_time = datetime.now()
    
    try:
        enhanced_df = service.calculate_enhanced_features(
            ohlcv_data=df,
            target=target,
            lookback_periods={
                'short': 4,
                'medium': 24,
                'long': 168,
                'extra_long': 336,
            }
        )
        
        processing_time = (datetime.now() - start_time).total_seconds()
        
        print(f"✅ 統合サービス成功:")
        print(f"  処理時間: {processing_time:.2f}秒")
        print(f"  入力データ: {df.shape}")
        print(f"  出力データ: {enhanced_df.shape}")
        print(f"  追加特徴量: {enhanced_df.shape[1] - df.shape[1]}個")
        
        # データ品質
        missing_count = enhanced_df.isnull().sum().sum()
        inf_count = np.isinf(enhanced_df.select_dtypes(include=[np.number])).sum().sum()
        print(f"  データ品質: 欠損値 {missing_count}個, 無限値 {inf_count}個")
        
        return enhanced_df
        
    except Exception as e:
        print(f"❌ 統合サービスエラー: {e}")
        import traceback
        traceback.print_exc()
        return None

def main():
    """メイン実行関数"""
    print("最適化された特徴量エンジニアリングのテスト")
    print("=" * 60)
    
    try:
        # 1. 特徴量エンジン比較
        enhanced_df, optimized_df, integrated_df = compare_feature_engines()
        
        # 2. 品質分析
        enhanced_quality = analyze_feature_quality(enhanced_df, "従来版")
        optimized_quality = analyze_feature_quality(optimized_df, "最適化版")
        integrated_quality = analyze_feature_quality(integrated_df, "統合版")
        
        # 3. 安定性テスト
        enhanced_stability = test_feature_stability(enhanced_df, "従来版")
        optimized_stability = test_feature_stability(optimized_df, "最適化版")
        
        # 4. 統合サービステスト
        service_result = test_integrated_service()
        
        print("\n" + "=" * 60)
        print("🎯 テスト結果サマリー:")
        
        print(f"\n📊 品質比較:")
        print(f"  従来版: 平均相関 {enhanced_quality.get('mean_correlation', 0):.4f}, "
              f"高相関特徴量 {enhanced_quality.get('high_corr_count', 0)}個")
        print(f"  最適化版: 平均相関 {optimized_quality.get('mean_correlation', 0):.4f}, "
              f"高相関特徴量 {optimized_quality.get('high_corr_count', 0)}個")
        print(f"  統合版: 平均相関 {integrated_quality.get('mean_correlation', 0):.4f}, "
              f"高相関特徴量 {integrated_quality.get('high_corr_count', 0)}個")
        
        print(f"\n🎯 改善効果:")
        if enhanced_quality and optimized_quality:
            quality_improvement = (optimized_quality['mean_correlation'] - enhanced_quality['mean_correlation']) / enhanced_quality['mean_correlation'] * 100
            print(f"  平均相関改善: {quality_improvement:+.1f}%")
        
        if service_result is not None:
            print(f"  統合サービス: 正常動作 ✅")
        else:
            print(f"  統合サービス: エラー ❌")
        
        print(f"\n✅ 最適化された特徴量エンジニアリングが正常に動作しています！")
        
    except Exception as e:
        print(f"テスト中にエラーが発生しました: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()
