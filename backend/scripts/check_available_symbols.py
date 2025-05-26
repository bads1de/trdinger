#!/usr/bin/env python3
"""
Bybitで利用可能なシンボルを確認するスクリプト

指定された通貨ペア（BTC、ETH、XRP、BNB、SOL）について、
スポットと先物の両方が利用可能かどうかを確認します。
"""

import asyncio
import ccxt
import logging
from typing import Dict, List, Set
from app.config.market_config import MarketDataConfig

# ログ設定
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class SymbolChecker:
    """Bybitで利用可能なシンボルを確認するクラス"""
    
    def __init__(self):
        """初期化"""
        self.exchange = ccxt.bybit(MarketDataConfig.BYBIT_CONFIG)
        self.target_currencies = ['BTC', 'ETH', 'XRP', 'BNB', 'SOL']
        
    async def check_available_symbols(self) -> Dict[str, Dict[str, List[str]]]:
        """
        利用可能なシンボルを確認
        
        Returns:
            通貨ごとの利用可能なスポット・先物ペアの辞書
        """
        logger.info("Bybitから利用可能なシンボルを取得中...")
        
        try:
            # 市場情報を取得
            markets = await asyncio.get_event_loop().run_in_executor(
                None, self.exchange.load_markets
            )
            
            logger.info(f"総市場数: {len(markets)}")
            
            # 結果を格納する辞書
            result = {}
            
            for currency in self.target_currencies:
                result[currency] = {
                    'spot': [],
                    'futures': []
                }
                
                # 各通貨について利用可能なペアを検索
                for symbol, market in markets.items():
                    if currency in symbol:
                        # スポット市場
                        if market['spot']:
                            if self._is_relevant_spot_pair(symbol, currency):
                                result[currency]['spot'].append(symbol)
                        
                        # 先物市場
                        if market['future'] or market['swap']:
                            if self._is_relevant_futures_pair(symbol, currency):
                                result[currency]['futures'].append(symbol)
            
            return result
            
        except Exception as e:
            logger.error(f"シンボル確認エラー: {e}")
            raise
    
    def _is_relevant_spot_pair(self, symbol: str, currency: str) -> bool:
        """
        関連するスポットペアかどうかを判定
        
        Args:
            symbol: シンボル名
            currency: 対象通貨
            
        Returns:
            関連するペアの場合True
        """
        # 主要なスポットペアのパターン
        relevant_patterns = [
            f"{currency}/USDT",
            f"{currency}/USD",
            f"{currency}/BTC",
            f"{currency}/ETH"
        ]
        
        return any(pattern in symbol for pattern in relevant_patterns)
    
    def _is_relevant_futures_pair(self, symbol: str, currency: str) -> bool:
        """
        関連する先物ペアかどうかを判定
        
        Args:
            symbol: シンボル名
            currency: 対象通貨
            
        Returns:
            関連するペアの場合True
        """
        # 主要な先物ペアのパターン
        relevant_patterns = [
            f"{currency}/USD:",  # 永続契約
            f"{currency}USD",    # 代替表記
            f"{currency}/USDT:", # USDT建て先物
        ]
        
        return any(pattern in symbol for pattern in relevant_patterns)
    
    def print_results(self, results: Dict[str, Dict[str, List[str]]]) -> None:
        """
        結果を見やすく表示
        
        Args:
            results: 確認結果
        """
        print("\n" + "="*60)
        print("Bybit 利用可能シンボル確認結果")
        print("="*60)
        
        for currency in self.target_currencies:
            print(f"\n【{currency}】")
            
            # スポット
            spot_pairs = results[currency]['spot']
            print(f"  スポット ({len(spot_pairs)}ペア):")
            if spot_pairs:
                for pair in sorted(spot_pairs):
                    print(f"    ✓ {pair}")
            else:
                print("    ❌ 利用可能なペアなし")
            
            # 先物
            futures_pairs = results[currency]['futures']
            print(f"  先物 ({len(futures_pairs)}ペア):")
            if futures_pairs:
                for pair in sorted(futures_pairs):
                    print(f"    ✓ {pair}")
            else:
                print("    ❌ 利用可能なペアなし")
    
    def get_recommended_symbols(self, results: Dict[str, Dict[str, List[str]]]) -> List[str]:
        """
        推奨シンボルリストを生成
        
        Args:
            results: 確認結果
            
        Returns:
            推奨シンボルのリスト
        """
        recommended = []
        
        for currency in self.target_currencies:
            # スポットの推奨ペア
            spot_pairs = results[currency]['spot']
            usdt_spot = f"{currency}/USDT"
            if usdt_spot in spot_pairs:
                recommended.append(usdt_spot)
            elif spot_pairs:
                # USDT ペアがない場合は最初のペアを選択
                recommended.append(spot_pairs[0])
            
            # 先物の推奨ペア
            futures_pairs = results[currency]['futures']
            usd_future = f"{currency}/USD:{currency}"
            usdt_future = f"{currency}/USDT:{currency}"
            
            if usd_future in futures_pairs:
                recommended.append(usd_future)
            elif usdt_future in futures_pairs:
                recommended.append(usdt_future)
            elif futures_pairs:
                # 上記がない場合は最初のペアを選択
                recommended.append(futures_pairs[0])
        
        return recommended


async def main():
    """メイン関数"""
    checker = SymbolChecker()
    
    try:
        # 利用可能なシンボルを確認
        results = await checker.check_available_symbols()
        
        # 結果を表示
        checker.print_results(results)
        
        # 推奨シンボルを表示
        recommended = checker.get_recommended_symbols(results)
        print(f"\n【推奨シンボル】")
        print("以下のシンボルを設定に追加することを推奨します:")
        for symbol in recommended:
            print(f"  {symbol}")
        
        print(f"\n推奨シンボル総数: {len(recommended)}")
        
    except Exception as e:
        logger.error(f"エラーが発生しました: {e}")
        return 1
    
    return 0


if __name__ == "__main__":
    exit_code = asyncio.run(main())
    exit(exit_code)
