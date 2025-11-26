import sys
from pathlib import Path
import pandas as pd
import numpy as np
import logging

# プロジェクトルートをパスに追加
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from app.utils.label_generation.presets import triple_barrier_method_preset, forward_classification_preset
from app.utils.label_generation.enums import ThresholdMethod

# ログ設定
logging.basicConfig(level=logging.INFO, format='%(message)s')
logger = logging.getLogger(__name__)

def generate_sample_data(n_samples=1000):
    """
    ボラティリティが変化するサンプルデータを生成
    """
    np.random.seed(42)
    dates = pd.date_range(start="2023-01-01", periods=n_samples, freq="1h")
    
    # 価格生成プロセス (幾何ブラウン運動 + ボラティリティ変化)
    returns = []
    price = 10000.0
    prices = [price]
    highs = [price]
    lows = [price]
    
    # 3つのフェーズ: 
    # 1. 低ボラティリティ (レンジ)
    # 2. 高ボラティリティ (トレンド発生しやすい)
    # 3. 中ボラティリティ
    
    vol_regimes = [0.002] * 300 + [0.01] * 400 + [0.005] * 300
    
    for i in range(1, n_samples):
        vol = vol_regimes[i]
        ret = np.random.normal(0, vol)
        # トレンド成分を少し加える (ランダムウォークより少しトレンドしやすくする)
        if i > 300 and i < 700:
            ret += np.random.normal(0.001, vol) # 上昇トレンド気味
            
        price = price * (1 + ret)
        prices.append(price)
        returns.append(ret)
        
        # High/Low 生成 (ATR計算用)
        high = price * (1 + abs(np.random.normal(0, vol/2)))
        low = price * (1 - abs(np.random.normal(0, vol/2)))
        highs.append(high)
        lows.append(low)
        
    df = pd.DataFrame({
        'open': prices,
        'high': highs,
        'low': lows,
        'close': prices,
        'volume': [1000] * n_samples
    }, index=dates)
    
    return df

def evaluate_labels(df, labels, name):
    """
    ラベルの品質を評価
    """
    # ラベルと次期リターンを結合
    # horizon_n=4 (4時間後) のリターンを見る
    horizon = 4
    future_return = df['close'].shift(-horizon) / df['close'] - 1
    
    analysis_df = pd.DataFrame({
        'label': labels,
        'return': future_return
    })
    
    # 評価
    # トレンド(UP/1)と判定された箇所のパフォーマンス
    if isinstance(labels.iloc[0], str):
        # 文字列ラベル (UP, RANGE, DOWN)
        trend_mask = analysis_df['label'] == 'UP'
    else:
        # バイナリラベル (1, 0)
        trend_mask = analysis_df['label'] == 1
        
    trend_trades = analysis_df[trend_mask]
    n_trades = len(trend_trades)
    
    if n_trades == 0:
        logger.info(f"--- {name} ---")
        logger.info("トレンド判定なし")
        return

    avg_return = trend_trades['return'].mean()
    win_rate = (trend_trades['return'] > 0).mean()
    
    logger.info(f"--- {name} ---")
    logger.info(f"総サンプル数: {len(df)}")
    logger.info(f"エントリー回数(トレンド判定): {n_trades} ({n_trades/len(df):.1%})")
    logger.info(f"平均リターン(4H後): {avg_return:.4%}")
    logger.info(f"勝率(4H後プラス): {win_rate:.1%}")
    logger.info("-" * 20)

def run_comparison():
    logger.info("データ生成中...")
    df = generate_sample_data()
    
    # 1. 旧ロジック: 固定閾値 (Forward Classification)
    # 4時間後の変化率が 0.5% 以上ならトレンド
    logger.info("\n評価1: 旧ロジック (固定閾値 0.5%)")
    labels_old = forward_classification_preset(
        df,
        timeframe="1h",
        horizon_n=4,
        threshold=0.005, # 0.5%
        threshold_method=ThresholdMethod.FIXED
    )
    evaluate_labels(df, labels_old, "Old Logic (Fixed 0.5%)")
    
    # 2. 新ロジック: ATRベース First-Exit (Meta-Labeling)
    # PT=1.0ATR, SL=0.5ATR, 時間切れ=8時間 (パラメータ緩和)
    # binary_label=True で 1(Trend) か 0(Range/Fake) か判定
    logger.info("\n評価2: 新ロジック (ATRベース First-Exit)")
    labels_new = triple_barrier_method_preset(
        df,
        timeframe="1h",
        horizon_n=8,
        pt=1.0,
        sl=0.5,
        use_atr=True,
        atr_period=14,
        binary_label=True
    )
    evaluate_labels(df, labels_new, "New Logic (ATR First-Exit)")

if __name__ == "__main__":
    run_comparison()
