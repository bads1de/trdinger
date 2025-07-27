#!/usr/bin/env python3
"""
効果的な特徴量の実装と検証

実際の取引データの特性を考慮した、より効果的な特徴量を実装します。
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

def generate_realistic_crypto_data(hours: int = 720) -> pd.DataFrame:
    """
    リアルな暗号通貨データを生成（30日分）
    
    Args:
        hours: 生成する時間数（デフォルト: 720時間 = 30日）
        
    Returns:
        OHLCV + OI + FR + FGデータ
    """
    print(f"リアルな暗号通貨データ生成: {hours}時間分")
    
    # 基準時刻
    start_time = datetime.now() - timedelta(hours=hours)
    timestamps = [start_time + timedelta(hours=i) for i in range(hours)]
    
    # 基準価格（BTC風）
    base_price = 50000
    
    # 価格データ生成（トレンド + ボラティリティ）
    np.random.seed(42)
    
    # トレンド成分
    trend = np.cumsum(np.random.normal(0, 0.001, hours))
    
    # ボラティリティ成分（時間帯による変動）
    hour_volatility = np.array([
        0.02 if 8 <= (start_time + timedelta(hours=i)).hour <= 16 else 0.015
        for i in range(hours)
    ])
    
    # 価格変動
    returns = np.random.normal(0, hour_volatility) + trend * 0.1
    prices = base_price * np.exp(np.cumsum(returns))
    
    # OHLCV生成
    ohlcv_data = []
    for i, (timestamp, close) in enumerate(zip(timestamps, prices)):
        # 高値・安値の生成
        volatility = hour_volatility[i] * close
        high = close + np.random.exponential(volatility * 0.5)
        low = close - np.random.exponential(volatility * 0.5)
        
        # 始値（前の終値 + 小さな変動）
        if i == 0:
            open_price = close
        else:
            open_price = prices[i-1] + np.random.normal(0, volatility * 0.1)
        
        # 出来高（価格変動と相関）
        price_change = abs(returns[i])
        base_volume = 1000 + price_change * 50000
        volume = max(100, np.random.normal(base_volume, base_volume * 0.3))
        
        ohlcv_data.append({
            'timestamp': timestamp,
            'Open': open_price,
            'High': max(open_price, high, close),
            'Low': min(open_price, low, close),
            'Close': close,
            'Volume': volume,
        })
    
    df = pd.DataFrame(ohlcv_data)
    df.set_index('timestamp', inplace=True)
    
    # Open Interest（8時間ごとに更新、トレンドと相関）
    oi_base = 1000000
    oi_trend = np.cumsum(np.random.normal(0, 0.005, hours // 8 + 1))
    oi_values = oi_base * (1 + oi_trend)
    
    # 8時間ごとのタイムスタンプ
    oi_timestamps = [start_time + timedelta(hours=i*8) for i in range(len(oi_values))]
    oi_series = pd.Series(oi_values, index=oi_timestamps)
    df['open_interest'] = oi_series.reindex(df.index, method='ffill')
    
    # Funding Rate（8時間ごと、価格トレンドと相関）
    fr_values = []
    for i in range(len(oi_values)):
        # 価格上昇時は正のFR、下降時は負のFR
        price_momentum = np.mean(returns[max(0, i*8-24):i*8+1])  # 過去24時間の平均
        base_fr = price_momentum * 0.1  # 価格変動の10%
        noise = np.random.normal(0, 0.0001)  # ノイズ
        fr = np.clip(base_fr + noise, -0.01, 0.01)  # -1%から1%に制限
        fr_values.append(fr)
    
    fr_series = pd.Series(fr_values, index=oi_timestamps)
    df['funding_rate'] = fr_series.reindex(df.index, method='ffill')
    
    # Fear & Greed Index（1日ごと、価格変動と逆相関）
    daily_timestamps = [start_time + timedelta(days=i) for i in range(hours // 24 + 1)]
    fg_values = []
    for i in range(len(daily_timestamps)):
        # 価格下落時は恐怖（低い値）、上昇時は強欲（高い値）
        if i == 0:
            daily_return = 0
        else:
            start_idx = max(0, i*24-24)
            end_idx = min(hours-1, i*24)
            daily_return = np.mean(returns[start_idx:end_idx])
        
        # 基準値50から価格変動に応じて調整
        base_fg = 50 + daily_return * 1000  # 価格変動を1000倍してFG値に
        noise = np.random.normal(0, 5)  # ノイズ
        fg = np.clip(base_fg + noise, 0, 100)
        fg_values.append(fg)
    
    fg_series = pd.Series(fg_values, index=daily_timestamps)
    df['fear_greed_value'] = fg_series.reindex(df.index, method='ffill')
    
    print(f"生成完了: {len(df)}行, カラム: {list(df.columns)}")
    return df

def create_advanced_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    高度な特徴量を作成
    
    Args:
        df: 基本データ
        
    Returns:
        特徴量が追加されたDataFrame
    """
    print("高度な特徴量を作成中...")
    
    result_df = df.copy()
    
    # 1. 価格関連特徴量
    print("  価格関連特徴量...")
    
    # 価格変動率（複数期間）
    for period in [1, 4, 12, 24]:
        result_df[f'price_change_{period}h'] = result_df['Close'].pct_change(period)
        result_df[f'price_volatility_{period}h'] = result_df['Close'].rolling(period).std() / result_df['Close'].rolling(period).mean()
    
    # 価格レンジ特徴量
    result_df['price_range'] = (result_df['High'] - result_df['Low']) / result_df['Close']
    result_df['upper_shadow'] = (result_df['High'] - np.maximum(result_df['Open'], result_df['Close'])) / result_df['Close']
    result_df['lower_shadow'] = (np.minimum(result_df['Open'], result_df['Close']) - result_df['Low']) / result_df['Close']
    
    # 2. 出来高関連特徴量
    print("  出来高関連特徴量...")
    
    # 出来高変動率
    for period in [1, 4, 12, 24]:
        result_df[f'volume_change_{period}h'] = result_df['Volume'].pct_change(period)
        result_df[f'volume_ma_{period}h'] = result_df['Volume'].rolling(period).mean()
    
    # 出来高加重平均価格（VWAP）
    for period in [12, 24, 48]:
        typical_price = (result_df['High'] + result_df['Low'] + result_df['Close']) / 3
        vwap = (typical_price * result_df['Volume']).rolling(period).sum() / result_df['Volume'].rolling(period).sum()
        result_df[f'vwap_{period}h'] = vwap
        result_df[f'price_vs_vwap_{period}h'] = (result_df['Close'] - vwap) / vwap
    
    # 3. Open Interest関連特徴量
    print("  Open Interest関連特徴量...")
    
    # OI変動率
    for period in [1, 8, 24]:
        result_df[f'oi_change_{period}h'] = result_df['open_interest'].pct_change(period)
    
    # OI vs 価格の関係
    result_df['oi_price_divergence'] = (
        result_df['open_interest'].pct_change() - result_df['Close'].pct_change()
    )
    
    # OI勢い
    result_df['oi_momentum_24h'] = result_df['open_interest'].rolling(24).apply(
        lambda x: (x.iloc[-1] - x.iloc[0]) / x.iloc[0] if x.iloc[0] != 0 else 0
    )
    
    # 4. Funding Rate関連特徴量
    print("  Funding Rate関連特徴量...")
    
    # FR変動
    result_df['fr_change'] = result_df['funding_rate'].diff()
    result_df['fr_abs'] = result_df['funding_rate'].abs()
    
    # FR累積（トレンドの強さ）
    for period in [24, 72, 168]:  # 1日、3日、1週間
        result_df[f'fr_cumsum_{period}h'] = result_df['funding_rate'].rolling(period).sum()
    
    # FR極値検出
    result_df['fr_extreme_positive'] = (result_df['funding_rate'] > 0.005).astype(int)
    result_df['fr_extreme_negative'] = (result_df['funding_rate'] < -0.005).astype(int)
    
    # 5. Fear & Greed関連特徴量
    print("  Fear & Greed関連特徴量...")
    
    # FG変動
    result_df['fg_change'] = result_df['fear_greed_value'].diff()
    result_df['fg_change_24h'] = result_df['fear_greed_value'].diff(24)
    
    # FG極値
    result_df['fg_extreme_fear'] = (result_df['fear_greed_value'] <= 25).astype(int)
    result_df['fg_extreme_greed'] = (result_df['fear_greed_value'] >= 75).astype(int)
    result_df['fg_neutral'] = ((result_df['fear_greed_value'] > 40) & (result_df['fear_greed_value'] < 60)).astype(int)
    
    # 6. 複合特徴量（相互作用）
    print("  複合特徴量...")
    
    # 価格 vs OI の関係
    result_df['price_oi_correlation'] = result_df['Close'].rolling(24).corr(result_df['open_interest'])
    
    # FR vs 価格変動の関係
    result_df['fr_price_alignment'] = (
        np.sign(result_df['funding_rate']) == np.sign(result_df['price_change_1h'])
    ).astype(int)
    
    # FG vs 価格変動の逆相関
    result_df['fg_price_contrarian'] = (
        (result_df['fear_greed_value'] < 30) & (result_df['price_change_1h'] > 0)
    ).astype(int)
    
    # 7. テクニカル指標
    print("  テクニカル指標...")
    
    # RSI
    for period in [14, 24]:
        delta = result_df['Close'].diff()
        gain = (delta.where(delta > 0, 0)).rolling(period).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(period).mean()
        rs = gain / loss
        result_df[f'rsi_{period}'] = 100 - (100 / (1 + rs))
    
    # ボリンジャーバンド
    for period in [20, 48]:
        ma = result_df['Close'].rolling(period).mean()
        std = result_df['Close'].rolling(period).std()
        result_df[f'bb_upper_{period}'] = ma + (std * 2)
        result_df[f'bb_lower_{period}'] = ma - (std * 2)
        result_df[f'bb_position_{period}'] = (result_df['Close'] - result_df[f'bb_lower_{period}']) / (result_df[f'bb_upper_{period}'] - result_df[f'bb_lower_{period}'])
    
    # 8. 時間関連特徴量
    print("  時間関連特徴量...")
    
    # 時間帯
    result_df['hour'] = result_df.index.hour
    result_df['day_of_week'] = result_df.index.dayofweek
    result_df['is_weekend'] = (result_df.index.dayofweek >= 5).astype(int)
    
    # アジア時間、欧州時間、米国時間
    result_df['asia_hours'] = ((result_df['hour'] >= 0) & (result_df['hour'] < 8)).astype(int)
    result_df['europe_hours'] = ((result_df['hour'] >= 8) & (result_df['hour'] < 16)).astype(int)
    result_df['us_hours'] = ((result_df['hour'] >= 16) & (result_df['hour'] < 24)).astype(int)
    
    print(f"特徴量作成完了: {len(result_df.columns)}個の特徴量")
    return result_df

def analyze_feature_importance(df: pd.DataFrame, target_periods: list = [1, 4, 12, 24]):
    """
    特徴量の重要度を分析
    
    Args:
        df: 特徴量データ
        target_periods: 予測対象期間（時間）
    """
    print("\n=== 特徴量重要度分析 ===")
    
    results = {}
    
    for period in target_periods:
        print(f"\n📈 {period}時間後の価格変動予測:")
        
        # ターゲット変数作成
        target = df['Close'].pct_change(period).shift(-period)
        
        # 特徴量選択（数値のみ）
        feature_cols = df.select_dtypes(include=[np.number]).columns
        feature_cols = [col for col in feature_cols if col not in ['Open', 'High', 'Low', 'Close', 'Volume']]
        
        # 有効なデータのみ
        valid_mask = target.notna() & df[feature_cols].notna().all(axis=1)
        if valid_mask.sum() < 50:
            print(f"  有効データ不足: {valid_mask.sum()}件")
            continue
        
        X = df.loc[valid_mask, feature_cols]
        y = target.loc[valid_mask]
        
        # 相関分析
        correlations = []
        for col in feature_cols:
            corr = X[col].corr(y)
            if not pd.isna(corr):
                correlations.append({
                    'feature': col,
                    'correlation': abs(corr),
                    'correlation_raw': corr,
                })
        
        # 相関順にソート
        correlations = sorted(correlations, key=lambda x: x['correlation'], reverse=True)
        
        print(f"  有効データ: {len(X)}件")
        print(f"  上位10特徴量:")
        for i, item in enumerate(correlations[:10]):
            print(f"    {i+1:2d}. {item['feature']:<30} 相関: {item['correlation_raw']:+.4f}")
        
        results[f'{period}h'] = correlations
    
    return results

def create_optimized_feature_set(df: pd.DataFrame, importance_results: dict):
    """
    最適化された特徴量セットを作成
    
    Args:
        df: 元データ
        importance_results: 重要度分析結果
        
    Returns:
        最適化された特徴量DataFrame
    """
    print("\n=== 最適化特徴量セット作成 ===")
    
    # 全期間で重要な特徴量を抽出
    all_important_features = set()
    
    for period, correlations in importance_results.items():
        # 上位20特徴量を選択
        top_features = [item['feature'] for item in correlations[:20] if item['correlation'] > 0.01]
        all_important_features.update(top_features)
        print(f"{period}: {len(top_features)}個の重要特徴量")
    
    print(f"統合重要特徴量: {len(all_important_features)}個")
    
    # 基本カラム + 重要特徴量
    base_cols = ['Open', 'High', 'Low', 'Close', 'Volume', 'open_interest', 'funding_rate', 'fear_greed_value']
    optimized_cols = base_cols + list(all_important_features)
    
    # 存在するカラムのみ選択
    available_cols = [col for col in optimized_cols if col in df.columns]
    
    optimized_df = df[available_cols].copy()
    
    print(f"最適化データ形状: {optimized_df.shape}")
    print(f"選択された特徴量: {len(available_cols)}個")
    
    return optimized_df

def main():
    """メイン実行関数"""
    print("効果的な特徴量の実装と検証")
    print("=" * 50)
    
    try:
        # 1. リアルなデータ生成
        df = generate_realistic_crypto_data(720)  # 30日分
        
        # 2. 高度な特徴量作成
        df_with_features = create_advanced_features(df)
        
        # 3. 特徴量重要度分析
        importance_results = analyze_feature_importance(df_with_features)
        
        # 4. 最適化特徴量セット作成
        optimized_df = create_optimized_feature_set(df_with_features, importance_results)
        
        print("\n" + "=" * 50)
        print("🎯 推奨改善策:")
        print("1. 高相関特徴量を優先的に実装")
        print("2. 複合特徴量（相互作用）の活用")
        print("3. 時間帯別の特徴量エンジニアリング")
        print("4. データ頻度の違いを考慮した補間")
        print("5. ドメイン知識を活用した特徴量設計")
        
        # サンプルデータを保存
        output_path = Path(__file__).parent / "sample_optimized_features.csv"
        optimized_df.to_csv(output_path)
        print(f"\n📁 サンプルデータ保存: {output_path}")
        
    except Exception as e:
        print(f"実行中にエラーが発生しました: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()
