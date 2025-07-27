#!/usr/bin/env python3
"""
AutoML特徴量の深度分析

特徴量の品質、相関、重要度、安定性を詳細に分析し、
より効果的な特徴量生成のための改善点を特定します。
"""

import sys
import os
import pandas as pd
import numpy as np
from pathlib import Path
from datetime import datetime, timedelta
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.feature_selection import mutual_info_regression, f_regression
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LassoCV
from sklearn.preprocessing import StandardScaler
import warnings
warnings.filterwarnings('ignore')

# プロジェクトルートをパスに追加
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from app.services.ml.feature_engineering.enhanced_crypto_features import EnhancedCryptoFeatures

def generate_comprehensive_test_data(hours: int = 720) -> pd.DataFrame:
    """
    包括的なテストデータを生成（30日分）
    より複雑で現実的な市場動向を模擬
    """
    print(f"包括的テストデータ生成: {hours}時間分")
    
    start_time = datetime.now() - timedelta(hours=hours)
    timestamps = [start_time + timedelta(hours=i) for i in range(hours)]
    
    # 複数の市場レジーム
    np.random.seed(42)
    
    # 1. 長期トレンド（週単位）
    weekly_trend = np.sin(np.arange(hours) * 2 * np.pi / (24 * 7)) * 0.02
    
    # 2. 中期サイクル（日単位）
    daily_cycle = np.sin(np.arange(hours) * 2 * np.pi / 24) * 0.01
    
    # 3. 短期ノイズ
    short_noise = np.random.normal(0, 0.015, hours)
    
    # 4. ボラティリティクラスタリング
    volatility_regime = np.zeros(hours)
    current_vol = 0.02
    for i in range(hours):
        # ボラティリティの持続性
        if np.random.random() < 0.05:  # 5%の確率でレジーム変更
            current_vol = np.random.choice([0.01, 0.02, 0.04], p=[0.3, 0.5, 0.2])
        volatility_regime[i] = current_vol
    
    # 5. 突発的なイベント（ニュース等）
    event_impacts = np.zeros(hours)
    for _ in range(5):  # 5回のイベント
        event_time = np.random.randint(0, hours)
        event_magnitude = np.random.choice([-0.05, 0.05], p=[0.5, 0.5])
        event_duration = np.random.randint(1, 6)  # 1-5時間
        for j in range(event_duration):
            if event_time + j < hours:
                event_impacts[event_time + j] = event_magnitude * np.exp(-j * 0.5)
    
    # 総合的なリターン
    base_returns = weekly_trend + daily_cycle + event_impacts
    noise_returns = np.random.normal(0, volatility_regime)
    returns = base_returns + noise_returns
    
    # 価格生成
    base_price = 50000
    prices = base_price * np.exp(np.cumsum(returns))
    
    # OHLCV生成
    data = []
    for i, (timestamp, close) in enumerate(zip(timestamps, prices)):
        vol = volatility_regime[i] * close
        
        # より現実的な高値・安値
        high_factor = np.random.exponential(0.3) * vol
        low_factor = np.random.exponential(0.3) * vol
        high = close + high_factor
        low = close - low_factor
        
        if i == 0:
            open_price = close
        else:
            gap = np.random.normal(0, vol * 0.1)
            open_price = prices[i-1] + gap
        
        # 出来高（価格変動とボラティリティに相関）
        volume_base = 1000
        volume_volatility_factor = volatility_regime[i] * 50000
        volume_price_factor = abs(returns[i]) * 100000
        volume = max(100, np.random.normal(
            volume_base + volume_volatility_factor + volume_price_factor,
            volume_base * 0.3
        ))
        
        data.append({
            'timestamp': timestamp,
            'Open': open_price,
            'High': max(open_price, high, close),
            'Low': min(open_price, low, close),
            'Close': close,
            'Volume': volume,
            'volatility_regime': volatility_regime[i],
        })
    
    df = pd.DataFrame(data)
    df.set_index('timestamp', inplace=True)
    
    # Open Interest（より複雑なパターン）
    oi_base = 1000000
    oi_changes = []
    
    for i in range(0, hours, 8):  # 8時間ごと
        if i == 0:
            oi_change = 0
        else:
            # 複数要因の組み合わせ
            price_momentum = np.mean(returns[max(0, i-24):i])  # 24時間の価格勢い
            volatility_avg = np.mean(volatility_regime[max(0, i-24):i])  # 24時間の平均ボラティリティ
            
            # OI変動の要因
            momentum_factor = price_momentum * 0.3  # 価格勢いに追随
            volatility_factor = (volatility_avg - 0.02) * 0.5  # 高ボラティリティで増加
            mean_reversion = -0.1 * (len(oi_changes) > 0 and oi_changes[-1] or 0)  # 平均回帰
            noise = np.random.normal(0, 0.01)
            
            oi_change = momentum_factor + volatility_factor + mean_reversion + noise
        
        oi_changes.append(oi_change)
    
    oi_values = oi_base * np.exp(np.cumsum(oi_changes))
    oi_timestamps = [start_time + timedelta(hours=i*8) for i in range(len(oi_values))]
    oi_series = pd.Series(oi_values, index=oi_timestamps)
    df['open_interest'] = oi_series.reindex(df.index, method='ffill')
    
    # Funding Rate（市場構造を反映）
    fr_values = []
    for i in range(len(oi_values)):
        if i == 0:
            price_trend = 0
            oi_trend = 0
        else:
            start_idx = max(0, i*8-24)
            end_idx = i*8
            price_trend = np.mean(returns[start_idx:end_idx])
            oi_trend = oi_changes[i] if i < len(oi_changes) else 0
        
        # FR決定要因
        base_fr = price_trend * 0.15  # 価格トレンドの影響
        oi_pressure = oi_trend * 0.1  # OI変動の影響
        market_stress = (volatility_regime[min(i*8, hours-1)] - 0.02) * 0.002  # ボラティリティの影響
        
        # 極値制限とノイズ
        fr = np.clip(
            base_fr + oi_pressure + market_stress + np.random.normal(0, 0.0001),
            -0.01, 0.01
        )
        fr_values.append(fr)
    
    fr_series = pd.Series(fr_values, index=oi_timestamps)
    df['funding_rate'] = fr_series.reindex(df.index, method='ffill')
    
    # Fear & Greed Index（心理的要因を反映）
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
        
        # FG決定要因
        fear_from_loss = max(0, -daily_return * 1000)  # 損失からの恐怖
        fear_from_volatility = (daily_volatility - 0.02) * 500  # ボラティリティからの恐怖
        greed_from_gain = max(0, daily_return * 800)  # 利益からの強欲
        
        # 基準値50から調整
        fg_raw = 50 - fear_from_loss - fear_from_volatility + greed_from_gain
        
        # 慣性（前日からの変化を制限）
        if i > 0:
            max_change = 10
            fg_raw = fg_values[-1] + np.clip(fg_raw - fg_values[-1], -max_change, max_change)
        
        # ノイズと制限
        fg = np.clip(fg_raw + np.random.normal(0, 2), 0, 100)
        fg_values.append(fg)
    
    fg_series = pd.Series(fg_values, index=daily_timestamps)
    df['fear_greed_value'] = fg_series.reindex(df.index, method='ffill')
    
    # ボラティリティレジーム情報を削除（分析用のみ）
    df = df.drop('volatility_regime', axis=1)
    
    print(f"生成完了: {len(df)}行")
    print(f"価格変動: {(df['Close'].iloc[-1]/df['Close'].iloc[0]-1)*100:+.1f}%")
    print(f"最大ドローダウン: {((df['Close']/df['Close'].cummax()).min()-1)*100:.1f}%")
    print(f"平均ボラティリティ: {df['Close'].pct_change().std()*100:.2f}%")
    
    return df

def analyze_feature_stability(df: pd.DataFrame, crypto_features: EnhancedCryptoFeatures):
    """
    特徴量の安定性を分析
    """
    print("\n=== 特徴量安定性分析 ===")
    
    # 複数の期間で特徴量を計算
    periods = [168, 336, 504, 720]  # 1週間、2週間、3週間、1ヶ月
    stability_results = {}
    
    for period in periods:
        print(f"\n📊 {period}時間（{period//24}日）データでの分析:")
        
        # データを切り取り
        period_df = df.iloc[-period:].copy()
        
        # 特徴量生成
        enhanced_df = crypto_features.create_comprehensive_features(period_df)
        
        # ターゲット変数
        target = enhanced_df['Close'].pct_change(4).shift(-4)
        
        # 有効データ
        valid_mask = target.notna() & enhanced_df.notna().all(axis=1)
        if valid_mask.sum() < 50:
            print(f"  有効データ不足: {valid_mask.sum()}件")
            continue
        
        # 特徴量の相関計算
        feature_cols = [col for col in enhanced_df.columns 
                       if col not in ['Open', 'High', 'Low', 'Close', 'Volume']]
        
        correlations = {}
        for col in feature_cols:
            corr = enhanced_df.loc[valid_mask, col].corr(target.loc[valid_mask])
            if not pd.isna(corr):
                correlations[col] = abs(corr)
        
        stability_results[period] = correlations
        print(f"  有効特徴量: {len(correlations)}個")
        print(f"  平均相関: {np.mean(list(correlations.values())):.4f}")
    
    # 安定性スコア計算
    print(f"\n🎯 特徴量安定性ランキング:")
    
    # 全期間で存在する特徴量
    common_features = set(stability_results[periods[0]].keys())
    for period in periods[1:]:
        common_features &= set(stability_results[period].keys())
    
    stability_scores = {}
    for feature in common_features:
        correlations = [stability_results[period][feature] for period in periods]
        # 安定性スコア = 平均相関 - 相関の標準偏差
        stability_score = np.mean(correlations) - np.std(correlations)
        stability_scores[feature] = {
            'stability_score': stability_score,
            'mean_correlation': np.mean(correlations),
            'correlation_std': np.std(correlations),
            'correlations': correlations,
        }
    
    # 安定性順にソート
    sorted_features = sorted(stability_scores.items(), 
                           key=lambda x: x[1]['stability_score'], reverse=True)
    
    for i, (feature, scores) in enumerate(sorted_features[:20]):
        print(f"  {i+1:2d}. {feature:<35} "
              f"安定性: {scores['stability_score']:+.4f} "
              f"(平均: {scores['mean_correlation']:.4f}, "
              f"標準偏差: {scores['correlation_std']:.4f})")
    
    return stability_scores

def analyze_feature_interactions(df: pd.DataFrame, crypto_features: EnhancedCryptoFeatures):
    """
    特徴量間の相互作用を分析
    """
    print("\n=== 特徴量相互作用分析 ===")
    
    # 特徴量生成
    enhanced_df = crypto_features.create_comprehensive_features(df)
    
    # ターゲット変数
    target = enhanced_df['Close'].pct_change(4).shift(-4)
    
    # 有効データ
    valid_mask = target.notna() & enhanced_df.notna().all(axis=1)
    if valid_mask.sum() < 100:
        print("有効データ不足")
        return
    
    # 特徴量選択
    feature_cols = [col for col in enhanced_df.columns 
                   if col not in ['Open', 'High', 'Low', 'Close', 'Volume']]
    
    X = enhanced_df.loc[valid_mask, feature_cols]
    y = target.loc[valid_mask]
    
    # 1. 相互情報量分析
    print(f"\n📊 相互情報量分析:")
    mi_scores = mutual_info_regression(X, y, random_state=42)
    mi_results = list(zip(feature_cols, mi_scores))
    mi_results.sort(key=lambda x: x[1], reverse=True)
    
    print(f"上位10特徴量（相互情報量）:")
    for i, (feature, score) in enumerate(mi_results[:10]):
        print(f"  {i+1:2d}. {feature:<35} MI: {score:.4f}")
    
    # 2. F統計量分析
    print(f"\n📊 F統計量分析:")
    f_scores, _ = f_regression(X, y)
    f_results = list(zip(feature_cols, f_scores))
    f_results.sort(key=lambda x: x[1], reverse=True)
    
    print(f"上位10特徴量（F統計量）:")
    for i, (feature, score) in enumerate(f_results[:10]):
        print(f"  {i+1:2d}. {feature:<35} F: {score:.2f}")
    
    # 3. ランダムフォレスト重要度
    print(f"\n📊 ランダムフォレスト重要度:")
    rf = RandomForestRegressor(n_estimators=100, random_state=42, n_jobs=-1)
    rf.fit(X, y)
    
    rf_results = list(zip(feature_cols, rf.feature_importances_))
    rf_results.sort(key=lambda x: x[1], reverse=True)
    
    print(f"上位10特徴量（RF重要度）:")
    for i, (feature, importance) in enumerate(rf_results[:10]):
        print(f"  {i+1:2d}. {feature:<35} 重要度: {importance:.4f}")
    
    # 4. Lasso回帰による特徴量選択
    print(f"\n📊 Lasso回帰特徴量選択:")
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    
    lasso = LassoCV(cv=5, random_state=42, max_iter=1000)
    lasso.fit(X_scaled, y)
    
    lasso_coefs = np.abs(lasso.coef_)
    lasso_results = list(zip(feature_cols, lasso_coefs))
    lasso_results.sort(key=lambda x: x[1], reverse=True)
    
    selected_features = [feature for feature, coef in lasso_results if coef > 0]
    print(f"選択された特徴量: {len(selected_features)}個")
    
    print(f"上位10特徴量（Lasso係数）:")
    for i, (feature, coef) in enumerate(lasso_results[:10]):
        if coef > 0:
            print(f"  {i+1:2d}. {feature:<35} 係数: {coef:.4f}")
    
    return {
        'mutual_info': mi_results,
        'f_statistic': f_results,
        'random_forest': rf_results,
        'lasso': lasso_results,
        'selected_features': selected_features,
    }

def analyze_feature_groups_performance(df: pd.DataFrame, crypto_features: EnhancedCryptoFeatures):
    """
    特徴量グループ別の性能分析
    """
    print("\n=== 特徴量グループ性能分析 ===")
    
    # 特徴量生成
    enhanced_df = crypto_features.create_comprehensive_features(df)
    feature_groups = crypto_features.get_feature_groups()
    
    # 複数のターゲット期間で分析
    target_periods = [1, 4, 12, 24]
    group_performance = {}
    
    for period in target_periods:
        print(f"\n📈 {period}時間後予測での性能:")
        
        target = enhanced_df['Close'].pct_change(period).shift(-period)
        valid_mask = target.notna() & enhanced_df.notna().all(axis=1)
        
        if valid_mask.sum() < 50:
            print(f"  有効データ不足: {valid_mask.sum()}件")
            continue
        
        period_performance = {}
        
        for group_name, features in feature_groups.items():
            if not features:
                continue
            
            # グループ内特徴量の相関計算
            group_correlations = []
            for feature in features:
                if feature in enhanced_df.columns:
                    corr = enhanced_df.loc[valid_mask, feature].corr(target.loc[valid_mask])
                    if not pd.isna(corr):
                        group_correlations.append(abs(corr))
            
            if group_correlations:
                period_performance[group_name] = {
                    'feature_count': len(group_correlations),
                    'mean_correlation': np.mean(group_correlations),
                    'max_correlation': np.max(group_correlations),
                    'std_correlation': np.std(group_correlations),
                    'top_10_percent': np.percentile(group_correlations, 90),
                }
        
        group_performance[period] = period_performance
        
        # グループ性能をソート
        sorted_groups = sorted(period_performance.items(), 
                             key=lambda x: x[1]['mean_correlation'], reverse=True)
        
        for group_name, performance in sorted_groups:
            print(f"  {group_name:<15}: "
                  f"平均 {performance['mean_correlation']:.4f}, "
                  f"最大 {performance['max_correlation']:.4f}, "
                  f"90%ile {performance['top_10_percent']:.4f} "
                  f"({performance['feature_count']}個)")
    
    return group_performance

def identify_improvement_opportunities(stability_scores, interaction_results, group_performance):
    """
    改善機会の特定
    """
    print("\n=== 改善機会の特定 ===")
    
    # 1. 不安定な特徴量の特定
    print(f"\n🔍 不安定な特徴量（改善が必要）:")
    unstable_features = []
    for feature, scores in stability_scores.items():
        if scores['correlation_std'] > 0.02:  # 標準偏差が大きい
            unstable_features.append((feature, scores['correlation_std']))
    
    unstable_features.sort(key=lambda x: x[1], reverse=True)
    for feature, std in unstable_features[:10]:
        print(f"  - {feature:<35} 標準偏差: {std:.4f}")
    
    # 2. 一貫して高性能な特徴量
    print(f"\n✅ 一貫して高性能な特徴量:")
    stable_high_performers = []
    for feature, scores in stability_scores.items():
        if scores['mean_correlation'] > 0.05 and scores['correlation_std'] < 0.01:
            stable_high_performers.append((feature, scores['stability_score']))
    
    stable_high_performers.sort(key=lambda x: x[1], reverse=True)
    for feature, score in stable_high_performers[:10]:
        print(f"  - {feature:<35} 安定性スコア: {score:.4f}")
    
    # 3. 複数手法で高評価の特徴量
    print(f"\n🎯 複数手法で高評価の特徴量:")
    
    # 各手法の上位20%を取得
    mi_top = set([f for f, _ in interaction_results['mutual_info'][:len(interaction_results['mutual_info'])//5]])
    f_top = set([f for f, _ in interaction_results['f_statistic'][:len(interaction_results['f_statistic'])//5]])
    rf_top = set([f for f, _ in interaction_results['random_forest'][:len(interaction_results['random_forest'])//5]])
    lasso_selected = set(interaction_results['selected_features'])
    
    # 複数手法で選ばれた特徴量
    multi_method_features = mi_top & f_top & rf_top & lasso_selected
    
    for feature in list(multi_method_features)[:10]:
        print(f"  - {feature}")
    
    # 4. 改善提案
    print(f"\n💡 改善提案:")
    print(f"1. 不安定特徴量の改良:")
    print(f"   - 移動平均やスムージングの適用")
    print(f"   - より長期間での計算")
    print(f"   - 外れ値の除去")
    
    print(f"\n2. 新しい特徴量の開発:")
    print(f"   - 高性能特徴量の組み合わせ")
    print(f"   - 非線形変換の適用")
    print(f"   - 時間遅れの考慮")
    
    print(f"\n3. 特徴量グループの最適化:")
    print(f"   - 低性能グループの見直し")
    print(f"   - 高性能グループの拡張")
    print(f"   - グループ間の相互作用")
    
    return {
        'unstable_features': unstable_features,
        'stable_high_performers': stable_high_performers,
        'multi_method_features': multi_method_features,
    }

def main():
    """メイン実行関数"""
    print("AutoML特徴量の深度分析")
    print("=" * 60)
    
    try:
        # 1. 包括的テストデータ生成
        df = generate_comprehensive_test_data(720)  # 30日分
        
        # 2. 特徴量エンジニアリング
        crypto_features = EnhancedCryptoFeatures()
        
        # 3. 特徴量安定性分析
        stability_scores = analyze_feature_stability(df, crypto_features)
        
        # 4. 特徴量相互作用分析
        interaction_results = analyze_feature_interactions(df, crypto_features)
        
        # 5. 特徴量グループ性能分析
        group_performance = analyze_feature_groups_performance(df, crypto_features)
        
        # 6. 改善機会の特定
        improvement_opportunities = identify_improvement_opportunities(
            stability_scores, interaction_results, group_performance
        )
        
        print("\n" + "=" * 60)
        print("🎯 分析完了")
        print("詳細な改善提案に基づいて特徴量エンジニアリングを最適化してください。")
        
    except Exception as e:
        print(f"分析中にエラーが発生しました: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()
