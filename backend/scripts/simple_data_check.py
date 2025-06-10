#!/usr/bin/env python3
"""
簡単なデータ確認スクリプト

実際のDBデータを確認して、GA戦略生成に使用可能なデータを把握します。
"""

import sys
import os
from datetime import datetime, timedelta, timezone
import logging

# プロジェクトルートをパスに追加
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from database.connection import SessionLocal
from database.repositories.ohlcv_repository import OHLCVRepository
from database.repositories.open_interest_repository import OpenInterestRepository
from database.repositories.funding_rate_repository import FundingRateRepository

# ログ設定
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


def simple_data_check():
    """シンプルなデータ確認"""
    print("🔍 シンプルデータ確認開始")
    print("=" * 60)
    
    try:
        db = SessionLocal()
        try:
            # リポジトリ初期化
            ohlcv_repo = OHLCVRepository(db)
            oi_repo = OpenInterestRepository(db)
            fr_repo = FundingRateRepository(db)
            
            # 1. OHLCVデータの確認
            print("\n📊 OHLCV データ確認:")
            print("-" * 30)
            
            symbols = ohlcv_repo.get_available_symbols()
            print(f"利用可能シンボル: {symbols}")
            
            # 各シンボルのデータ件数確認
            for symbol in symbols:
                try:
                    # 1日足データの件数確認
                    count_1d = ohlcv_repo.get_data_count(symbol, "1d")
                    count_1h = ohlcv_repo.get_data_count(symbol, "1h")
                    
                    print(f"  {symbol}: 1d({count_1d}件) 1h({count_1h}件)")
                    
                    # データがある場合は期間も確認
                    if count_1d > 0:
                        latest = ohlcv_repo.get_latest_timestamp(symbol, "1d")
                        oldest = ohlcv_repo.get_oldest_timestamp(symbol, "1d")
                        print(f"    期間: {oldest.strftime('%Y-%m-%d')} ～ {latest.strftime('%Y-%m-%d')}")
                        
                        # 最新データのサンプル
                        latest_data = ohlcv_repo.get_ohlcv_data(symbol, "1d", limit=1)
                        if latest_data:
                            data = latest_data[0]
                            print(f"    最新: Close={data.close:.2f} Volume={data.volume:.0f}")
                            
                except Exception as e:
                    print(f"  {symbol}: エラー - {e}")
            
            # 2. Open Interest データの確認
            print("\n🔢 Open Interest データ確認:")
            print("-" * 30)
            
            oi_available = []
            for symbol in symbols:
                try:
                    oi_data = oi_repo.get_open_interest_data(symbol, limit=1)
                    if oi_data:
                        oi_available.append(symbol)
                        print(f"  ✅ {symbol}: 最新OI = {oi_data[0].open_interest_value:,.0f}")
                    else:
                        print(f"  ❌ {symbol}: OIデータなし")
                except Exception as e:
                    print(f"  ❌ {symbol}: OIエラー - {e}")
            
            # 3. Funding Rate データの確認
            print("\n💰 Funding Rate データ確認:")
            print("-" * 30)
            
            fr_available = []
            for symbol in symbols:
                try:
                    fr_data = fr_repo.get_funding_rate_data(symbol, limit=1)
                    if fr_data:
                        fr_available.append(symbol)
                        fr_rate = fr_data[0].funding_rate
                        print(f"  ✅ {symbol}: 最新FR = {fr_rate:.6f} ({fr_rate*100:.4f}%)")
                    else:
                        print(f"  ❌ {symbol}: FRデータなし")
                except Exception as e:
                    print(f"  ❌ {symbol}: FRエラー - {e}")
            
            # 4. GA戦略生成に最適なシンボルの特定
            print("\n🎯 GA戦略生成推奨シンボル:")
            print("-" * 30)
            
            best_symbol = None
            best_score = 0
            
            for symbol in symbols:
                try:
                    # スコア計算
                    ohlcv_count = ohlcv_repo.get_data_count(symbol, "1d")
                    has_oi = symbol in oi_available
                    has_fr = symbol in fr_available
                    
                    score = 0
                    if ohlcv_count >= 30:  # 30日以上のデータ
                        score += ohlcv_count
                    if has_oi:
                        score += 100
                    if has_fr:
                        score += 100
                    
                    status = []
                    if ohlcv_count >= 30:
                        status.append(f"OHLCV({ohlcv_count})")
                    if has_oi:
                        status.append("OI")
                    if has_fr:
                        status.append("FR")
                    
                    print(f"  {symbol}: スコア={score} [{', '.join(status)}]")
                    
                    if score > best_score:
                        best_score = score
                        best_symbol = {
                            'symbol': symbol,
                            'ohlcv_count': ohlcv_count,
                            'has_oi': has_oi,
                            'has_fr': has_fr,
                            'score': score
                        }
                        
                except Exception as e:
                    print(f"  {symbol}: 評価エラー - {e}")
            
            # 5. 推奨設定の提案
            print(f"\n🚀 推奨GA設定:")
            print("-" * 30)
            
            if best_symbol and best_symbol['score'] > 30:
                print(f"推奨シンボル: {best_symbol['symbol']}")
                print(f"データ期間: {best_symbol['ohlcv_count']} 日分")
                print(f"OI利用: {'可能' if best_symbol['has_oi'] else '不可'}")
                print(f"FR利用: {'可能' if best_symbol['has_fr'] else '不可'}")
                
                # バックテスト期間の提案
                backtest_days = min(60, best_symbol['ohlcv_count'] - 10)
                print(f"推奨バックテスト期間: {backtest_days} 日")
                
                return {
                    'symbol': best_symbol['symbol'],
                    'timeframe': '1d',
                    'backtest_days': backtest_days,
                    'use_oi': best_symbol['has_oi'],
                    'use_fr': best_symbol['has_fr'],
                    'data_available': True
                }
            else:
                print("⚠️ 十分なデータが見つかりません")
                print("サンプルデータの生成を推奨します")
                return {
                    'data_available': False,
                    'need_sample_data': True
                }
            
        finally:
            db.close()
            
    except Exception as e:
        logger.error(f"データ確認エラー: {e}")
        import traceback
        traceback.print_exc()
        return {'data_available': False, 'error': str(e)}


def create_sample_data_if_needed():
    """必要に応じてサンプルデータを作成"""
    print("\n📝 サンプルデータ作成確認...")
    
    try:
        # サンプルデータ作成スクリプトを実行
        import subprocess
        result = subprocess.run([
            sys.executable, "scripts/create_sample_data.py"
        ], capture_output=True, text=True, cwd=os.path.dirname(os.path.dirname(__file__)))
        
        if result.returncode == 0:
            print("✅ サンプルデータ作成完了")
            return True
        else:
            print(f"❌ サンプルデータ作成失敗: {result.stderr}")
            return False
            
    except Exception as e:
        print(f"❌ サンプルデータ作成エラー: {e}")
        return False


if __name__ == "__main__":
    print("🔍 データベース簡単確認")
    print("=" * 60)
    
    # データ確認
    result = simple_data_check()
    
    print("\n" + "=" * 60)
    if result and result.get('data_available'):
        print("✅ データ確認完了 - GA戦略生成可能")
        print(f"推奨設定: {result}")
    elif result and result.get('need_sample_data'):
        print("⚠️ データ不足 - サンプルデータ作成を試行")
        if create_sample_data_if_needed():
            print("🔄 サンプルデータ作成後、再度確認してください")
        else:
            print("❌ サンプルデータ作成失敗")
    else:
        print("❌ データ確認失敗")
        
    print("\n次のステップ: python scripts/run_real_ga_strategy.py")
