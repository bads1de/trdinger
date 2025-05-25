/**
 * 通貨ペア一覧取得API
 *
 * システムで利用可能な通貨ペアの一覧を取得するAPIエンドポイントです。
 *
 * @author Trdinger Development Team
 * @version 1.0.0
 */

import { NextResponse } from 'next/server';
import { TradingPair } from '@/types/strategy';

/**
 * 利用可能な通貨ペアのマスターデータ
 */
const TRADING_PAIRS: TradingPair[] = [
  {
    symbol: 'BTC/USD',
    name: 'Bitcoin / US Dollar',
    base: 'BTC',
    quote: 'USD',
  },
  {
    symbol: 'ETH/USD',
    name: 'Ethereum / US Dollar',
    base: 'ETH',
    quote: 'USD',
  },
  {
    symbol: 'BNB/USD',
    name: 'Binance Coin / US Dollar',
    base: 'BNB',
    quote: 'USD',
  },
  {
    symbol: 'ADA/USD',
    name: 'Cardano / US Dollar',
    base: 'ADA',
    quote: 'USD',
  },
  {
    symbol: 'SOL/USD',
    name: 'Solana / US Dollar',
    base: 'SOL',
    quote: 'USD',
  },
  {
    symbol: 'MATIC/USD',
    name: 'Polygon / US Dollar',
    base: 'MATIC',
    quote: 'USD',
  },
  {
    symbol: 'DOT/USD',
    name: 'Polkadot / US Dollar',
    base: 'DOT',
    quote: 'USD',
  },
  {
    symbol: 'AVAX/USD',
    name: 'Avalanche / US Dollar',
    base: 'AVAX',
    quote: 'USD',
  },
];

/**
 * GET /api/data/symbols
 *
 * 利用可能な通貨ペアの一覧を取得します。
 */
export async function GET() {
  try {
    return NextResponse.json({
      success: true,
      data: TRADING_PAIRS,
      message: '通貨ペア一覧を取得しました',
      timestamp: new Date().toISOString(),
    });
  } catch (error) {
    console.error('通貨ペア一覧取得エラー:', error);
    
    return NextResponse.json(
      {
        success: false,
        message: 'サーバー内部エラーが発生しました',
        timestamp: new Date().toISOString(),
      },
      { status: 500 }
    );
  }
}
