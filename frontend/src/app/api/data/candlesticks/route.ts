/**
 * ローソク足データ取得API
 *
 * 仮想通貨のローソク足データ（OHLCV）を取得するAPIエンドポイントです。
 * 時間軸、通貨ペア、日付範囲を指定してデータを取得できます。
 *
 * @author Trdinger Development Team
 * @version 1.0.0
 */

import { NextRequest, NextResponse } from 'next/server';
import { CandlestickData, TimeFrame } from '@/types/strategy';

/**
 * 利用可能な時間軸の定義
 */
const VALID_TIMEFRAMES: TimeFrame[] = ['1m', '5m', '15m', '30m', '1h', '4h', '1d'];

/**
 * 利用可能な通貨ペア
 */
const VALID_SYMBOLS = ['BTC/USD', 'ETH/USD', 'BNB/USD', 'ADA/USD', 'SOL/USD'];

/**
 * モックデータ生成関数
 *
 * 実際のAPIでは外部データソースから取得しますが、
 * 開発段階ではモックデータを生成します。
 *
 * @param symbol 通貨ペア
 * @param timeframe 時間軸
 * @param limit データ件数
 * @returns ローソク足データの配列
 */
function generateMockCandlestickData(
  symbol: string,
  timeframe: TimeFrame,
  limit: number = 100
): CandlestickData[] {
  const data: CandlestickData[] = [];
  const now = new Date();
  
  // 時間軸に応じた間隔（ミリ秒）
  const intervals: Record<TimeFrame, number> = {
    '1m': 60 * 1000,
    '5m': 5 * 60 * 1000,
    '15m': 15 * 60 * 1000,
    '30m': 30 * 60 * 1000,
    '1h': 60 * 60 * 1000,
    '4h': 4 * 60 * 60 * 1000,
    '1d': 24 * 60 * 60 * 1000,
  };

  const interval = intervals[timeframe];
  
  // 基準価格（通貨ペアに応じて設定）
  const basePrices: Record<string, number> = {
    'BTC/USD': 45000,
    'ETH/USD': 3000,
    'BNB/USD': 300,
    'ADA/USD': 0.5,
    'SOL/USD': 100,
  };

  let basePrice = basePrices[symbol] || 1000;
  
  // 過去のデータから現在に向かって生成
  for (let i = limit - 1; i >= 0; i--) {
    const timestamp = new Date(now.getTime() - i * interval);
    
    // ランダムな価格変動を生成
    const volatility = 0.02; // 2%の変動率
    const change = (Math.random() - 0.5) * volatility;
    
    const open = basePrice;
    const close = open * (1 + change);
    const high = Math.max(open, close) * (1 + Math.random() * 0.01);
    const low = Math.min(open, close) * (1 - Math.random() * 0.01);
    const volume = Math.random() * 1000000 + 100000;

    data.push({
      timestamp: timestamp.toISOString(),
      open: Number(open.toFixed(2)),
      high: Number(high.toFixed(2)),
      low: Number(low.toFixed(2)),
      close: Number(close.toFixed(2)),
      volume: Number(volume.toFixed(0)),
    });

    basePrice = close; // 次のローソク足の基準価格を更新
  }

  return data;
}

/**
 * GET /api/data/candlesticks
 *
 * ローソク足データを取得します。
 *
 * クエリパラメータ:
 * - symbol: 通貨ペア (必須)
 * - timeframe: 時間軸 (必須)
 * - limit: 取得件数 (オプション、デフォルト: 100)
 * - start_date: 開始日時 (オプション)
 * - end_date: 終了日時 (オプション)
 */
export async function GET(request: NextRequest) {
  try {
    const { searchParams } = new URL(request.url);
    
    // パラメータの取得
    const symbol = searchParams.get('symbol');
    const timeframe = searchParams.get('timeframe') as TimeFrame;
    const limit = parseInt(searchParams.get('limit') || '100');
    const startDate = searchParams.get('start_date');
    const endDate = searchParams.get('end_date');

    // バリデーション
    if (!symbol) {
      return NextResponse.json(
        {
          success: false,
          message: 'symbol パラメータは必須です',
          timestamp: new Date().toISOString(),
        },
        { status: 400 }
      );
    }

    if (!timeframe) {
      return NextResponse.json(
        {
          success: false,
          message: 'timeframe パラメータは必須です',
          timestamp: new Date().toISOString(),
        },
        { status: 400 }
      );
    }

    if (!VALID_SYMBOLS.includes(symbol)) {
      return NextResponse.json(
        {
          success: false,
          message: `サポートされていない通貨ペアです。利用可能: ${VALID_SYMBOLS.join(', ')}`,
          timestamp: new Date().toISOString(),
        },
        { status: 400 }
      );
    }

    if (!VALID_TIMEFRAMES.includes(timeframe)) {
      return NextResponse.json(
        {
          success: false,
          message: `サポートされていない時間軸です。利用可能: ${VALID_TIMEFRAMES.join(', ')}`,
          timestamp: new Date().toISOString(),
        },
        { status: 400 }
      );
    }

    if (limit < 1 || limit > 1000) {
      return NextResponse.json(
        {
          success: false,
          message: 'limit は 1 から 1000 の間で指定してください',
          timestamp: new Date().toISOString(),
        },
        { status: 400 }
      );
    }

    // モックデータの生成
    const candlesticks = generateMockCandlestickData(symbol, timeframe, limit);

    // 日付フィルタリング（指定されている場合）
    let filteredData = candlesticks;
    if (startDate || endDate) {
      filteredData = candlesticks.filter((candle) => {
        const candleTime = new Date(candle.timestamp);
        if (startDate && candleTime < new Date(startDate)) return false;
        if (endDate && candleTime > new Date(endDate)) return false;
        return true;
      });
    }

    // レスポンスの返却
    return NextResponse.json({
      success: true,
      data: {
        symbol,
        timeframe,
        candlesticks: filteredData,
      },
      message: `${symbol} の ${timeframe} ローソク足データを取得しました`,
      timestamp: new Date().toISOString(),
    });

  } catch (error) {
    console.error('ローソク足データ取得エラー:', error);
    
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
