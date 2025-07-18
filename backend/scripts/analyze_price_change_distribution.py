"""
価格変化率の分布を分析するスクリプト

BTCUSDTの1時間足データの価格変化率分布を調査し、
適切なラベル生成閾値を決定するためのデータを提供します。
"""

import logging
import pandas as pd
import numpy as np
from datetime import datetime, timezone
from typing import Dict, Any
import sys
import os

# プロジェクトルートをパスに追加
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from database.connection import SessionLocal
from database.repositories.ohlcv_repository import OHLCVRepository

# ログ設定
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def analyze_price_change_distribution(
    symbol: str = "BTC/USDT:USDT",
    timeframe: str = "1h",
    start_date: datetime = datetime(2020, 3, 5, tzinfo=timezone.utc),
    end_date: datetime = datetime(2025, 7, 15, tzinfo=timezone.utc)
) -> Dict[str, Any]:
    """
    価格変化率の分布を分析
    
    Args:
        symbol: 取引ペア
        timeframe: 時間軸
        start_date: 開始日時
        end_date: 終了日時
        
    Returns:
        分析結果の辞書
    """
    logger.info(f"価格変化率分布分析開始: {symbol} {timeframe}")
    logger.info(f"期間: {start_date} ～ {end_date}")
    
    try:
        with SessionLocal() as db:
            ohlcv_repo = OHLCVRepository(db)
            
            # OHLCVデータを取得
            ohlcv_data = ohlcv_repo.get_ohlcv_data(
                symbol=symbol,
                timeframe=timeframe,
                start_time=start_date,
                end_time=end_date
            )
            
            if not ohlcv_data:
                logger.error(f"データが見つかりません: {symbol} {timeframe}")
                return {}
            
            logger.info(f"取得データ件数: {len(ohlcv_data)}")
            
            # DataFrameに変換
            df = pd.DataFrame([
                {
                    'timestamp': data.timestamp,
                    'open': data.open,
                    'high': data.high,
                    'low': data.low,
                    'close': data.close,
                    'volume': data.volume
                }
                for data in ohlcv_data
            ])
            
            # 価格変化率を計算
            df['price_change'] = df['close'].pct_change()
            
            # NaNを除去
            price_changes = df['price_change'].dropna()
            
            if len(price_changes) == 0:
                logger.error("価格変化率データが空です")
                return {}
            
            # 基本統計
            stats = {
                'count': len(price_changes),
                'mean': price_changes.mean(),
                'std': price_changes.std(),
                'min': price_changes.min(),
                'max': price_changes.max(),
                'median': price_changes.median(),
            }
            
            # 分位数
            percentiles = [1, 5, 10, 25, 50, 75, 90, 95, 99]
            for p in percentiles:
                stats[f'percentile_{p}'] = price_changes.quantile(p / 100)
            
            # 現在の閾値（±2%）での分布
            current_threshold_up = 0.02
            current_threshold_down = -0.02
            
            up_count = (price_changes > current_threshold_up).sum()
            down_count = (price_changes < current_threshold_down).sum()
            range_count = ((price_changes >= current_threshold_down) & 
                          (price_changes <= current_threshold_up)).sum()
            
            current_distribution = {
                'up': up_count,
                'down': down_count,
                'range': range_count,
                'up_ratio': up_count / len(price_changes),
                'down_ratio': down_count / len(price_changes),
                'range_ratio': range_count / len(price_changes)
            }
            
            # 推奨閾値の計算（各クラスが約33%になるように）
            recommended_thresholds = {}
            
            # 分位数ベースの閾値（33%、67%）
            threshold_33 = price_changes.quantile(0.33)
            threshold_67 = price_changes.quantile(0.67)
            
            recommended_thresholds['quantile_based'] = {
                'threshold_down': threshold_33,
                'threshold_up': threshold_67,
                'description': '33%/67%分位数ベース'
            }
            
            # 標準偏差ベースの閾値
            std_multipliers = [0.25, 0.5, 0.75, 1.0]
            for mult in std_multipliers:
                threshold_down = -mult * stats['std']
                threshold_up = mult * stats['std']
                
                up_count_std = (price_changes > threshold_up).sum()
                down_count_std = (price_changes < threshold_down).sum()
                range_count_std = ((price_changes >= threshold_down) & 
                                  (price_changes <= threshold_up)).sum()
                
                recommended_thresholds[f'std_{mult}'] = {
                    'threshold_down': threshold_down,
                    'threshold_up': threshold_up,
                    'up_count': up_count_std,
                    'down_count': down_count_std,
                    'range_count': range_count_std,
                    'up_ratio': up_count_std / len(price_changes),
                    'down_ratio': down_count_std / len(price_changes),
                    'range_ratio': range_count_std / len(price_changes),
                    'description': f'{mult}標準偏差ベース'
                }
            
            # 固定閾値での分析
            fixed_thresholds = [0.005, 0.01, 0.015, 0.02, 0.025, 0.03]
            for threshold in fixed_thresholds:
                threshold_down = -threshold
                threshold_up = threshold
                
                up_count_fixed = (price_changes > threshold_up).sum()
                down_count_fixed = (price_changes < threshold_down).sum()
                range_count_fixed = ((price_changes >= threshold_down) & 
                                    (price_changes <= threshold_up)).sum()
                
                recommended_thresholds[f'fixed_{threshold}'] = {
                    'threshold_down': threshold_down,
                    'threshold_up': threshold_up,
                    'up_count': up_count_fixed,
                    'down_count': down_count_fixed,
                    'range_count': range_count_fixed,
                    'up_ratio': up_count_fixed / len(price_changes),
                    'down_ratio': down_count_fixed / len(price_changes),
                    'range_ratio': range_count_fixed / len(price_changes),
                    'description': f'固定閾値±{threshold*100:.1f}%'
                }
            
            result = {
                'basic_stats': stats,
                'current_distribution': current_distribution,
                'recommended_thresholds': recommended_thresholds,
                'data_period': {
                    'start': start_date.isoformat(),
                    'end': end_date.isoformat(),
                    'symbol': symbol,
                    'timeframe': timeframe
                }
            }
            
            return result
            
    except Exception as e:
        logger.error(f"価格変化率分布分析エラー: {e}")
        raise


def print_analysis_results(results: Dict[str, Any]):
    """分析結果を見やすく表示"""
    if not results:
        logger.error("分析結果が空です")
        return
    
    print("\n" + "="*80)
    print("価格変化率分布分析結果")
    print("="*80)
    
    # 基本統計
    stats = results['basic_stats']
    print(f"\n【基本統計】")
    print(f"データ件数: {stats['count']:,}")
    print(f"平均: {stats['mean']:.6f} ({stats['mean']*100:.4f}%)")
    print(f"標準偏差: {stats['std']:.6f} ({stats['std']*100:.4f}%)")
    print(f"最小値: {stats['min']:.6f} ({stats['min']*100:.4f}%)")
    print(f"最大値: {stats['max']:.6f} ({stats['max']*100:.4f}%)")
    print(f"中央値: {stats['median']:.6f} ({stats['median']*100:.4f}%)")
    
    # 分位数
    print(f"\n【分位数】")
    percentiles = [1, 5, 10, 25, 50, 75, 90, 95, 99]
    for p in percentiles:
        value = stats[f'percentile_{p}']
        print(f"{p:2d}%: {value:.6f} ({value*100:.4f}%)")
    
    # 現在の閾値での分布
    current = results['current_distribution']
    print(f"\n【現在の閾値（±2%）での分布】")
    print(f"上昇: {current['up']:,}件 ({current['up_ratio']*100:.2f}%)")
    print(f"下落: {current['down']:,}件 ({current['down_ratio']*100:.2f}%)")
    print(f"レンジ: {current['range']:,}件 ({current['range_ratio']*100:.2f}%)")
    
    # 推奨閾値
    print(f"\n【推奨閾値の分析】")
    thresholds = results['recommended_thresholds']
    
    # バランスの良い閾値を見つける
    best_balance = None
    best_score = float('inf')
    
    for name, data in thresholds.items():
        if 'up_ratio' in data:
            # 各クラスが33%に近いほど良いスコア
            target_ratio = 1/3
            score = (abs(data['up_ratio'] - target_ratio) + 
                    abs(data['down_ratio'] - target_ratio) + 
                    abs(data['range_ratio'] - target_ratio))
            
            if score < best_score:
                best_score = score
                best_balance = (name, data)
            
            print(f"\n{data['description']}:")
            print(f"  閾値: {data['threshold_down']:.6f} ～ {data['threshold_up']:.6f}")
            print(f"  閾値(%): {data['threshold_down']*100:.4f}% ～ {data['threshold_up']*100:.4f}%")
            print(f"  上昇: {data['up_count']:,}件 ({data['up_ratio']*100:.2f}%)")
            print(f"  下落: {data['down_count']:,}件 ({data['down_ratio']*100:.2f}%)")
            print(f"  レンジ: {data['range_count']:,}件 ({data['range_ratio']*100:.2f}%)")
            print(f"  バランススコア: {score:.6f}")
    
    if best_balance:
        name, data = best_balance
        print(f"\n【最もバランスの良い閾値】")
        print(f"方法: {data['description']}")
        print(f"閾値: {data['threshold_down']:.6f} ～ {data['threshold_up']:.6f}")
        print(f"閾値(%): {data['threshold_down']*100:.4f}% ～ {data['threshold_up']*100:.4f}%")
        print(f"分布: 上昇{data['up_ratio']*100:.1f}% / 下落{data['down_ratio']*100:.1f}% / レンジ{data['range_ratio']*100:.1f}%")


if __name__ == "__main__":
    try:
        # 分析実行
        results = analyze_price_change_distribution()
        
        # 結果表示
        print_analysis_results(results)
        
    except Exception as e:
        logger.error(f"スクリプト実行エラー: {e}")
        sys.exit(1)
