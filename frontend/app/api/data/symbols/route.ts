/**
 * 通貨ペア一覧取得API
 *
 * Bybit取引所でサポートされている通貨ペアの一覧を取得するAPIエンドポイントです。
 * CCXT ライブラリを使用してリアルタイムデータを提供します。
 *
 * @author Trdinger Development Team
 * @version 2.0.0
 */

import { NextResponse } from "next/server";
import { TradingPair } from "@/types/strategy";

/**
 * Bybitでサポートされている通貨ペアのマスターデータ
 */
const TRADING_PAIRS: TradingPair[] = [
  {
    symbol: "BTC/USD",
    name: "Bitcoin / US Dollar (Futures)",
    base: "BTC",
    quote: "USD",
  },
  {
    symbol: "BTCUSD",
    name: "Bitcoin / US Dollar (Alternative)",
    base: "BTC",
    quote: "USD",
  },
  {
    symbol: "BTC/USDT",
    name: "Bitcoin / Tether USD (Spot)",
    base: "BTC",
    quote: "USDT",
  },
  {
    symbol: "ETH/USD",
    name: "Ethereum / US Dollar (Futures)",
    base: "ETH",
    quote: "USD",
  },
  {
    symbol: "ETH/USDT",
    name: "Ethereum / Tether USD (Spot)",
    base: "ETH",
    quote: "USDT",
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
      message: "通貨ペア一覧を取得しました",
      timestamp: new Date().toISOString(),
    });
  } catch (error) {
    console.error("通貨ペア一覧取得エラー:", error);

    return NextResponse.json(
      {
        success: false,
        message: "サーバー内部エラーが発生しました",
        timestamp: new Date().toISOString(),
      },
      { status: 500 }
    );
  }
}
