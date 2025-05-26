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
 * 先物とイーサリアム関連銘柄を追加
 */
const TRADING_PAIRS: TradingPair[] = [
  // Bitcoin 関連
  {
    symbol: "BTC/USDT",
    name: "Bitcoin / Tether USD (スポット)",
    base: "BTC",
    quote: "USDT",
  },
  {
    symbol: "BTC/USD",
    name: "Bitcoin / US Dollar (先物・永続契約)",
    base: "BTC",
    quote: "USD",
  },
  {
    symbol: "BTCUSD",
    name: "Bitcoin / US Dollar (先物・代替表記)",
    base: "BTC",
    quote: "USD",
  },

  // Ethereum 関連
  {
    symbol: "ETH/USDT",
    name: "Ethereum / Tether USD (スポット)",
    base: "ETH",
    quote: "USDT",
  },
  {
    symbol: "ETH/USD",
    name: "Ethereum / US Dollar (先物・永続契約)",
    base: "ETH",
    quote: "USD",
  },
  {
    symbol: "ETHUSD",
    name: "Ethereum / US Dollar (先物・代替表記)",
    base: "ETH",
    quote: "USD",
  },
  {
    symbol: "ETH/BTC",
    name: "Ethereum / Bitcoin (スポットペア)",
    base: "ETH",
    quote: "BTC",
  },

  // その他主要アルトコイン
  {
    symbol: "BNB/USDT",
    name: "Binance Coin / Tether USD (スポット)",
    base: "BNB",
    quote: "USDT",
  },
  {
    symbol: "ADA/USDT",
    name: "Cardano / Tether USD (スポット)",
    base: "ADA",
    quote: "USDT",
  },
  {
    symbol: "SOL/USDT",
    name: "Solana / Tether USD (スポット)",
    base: "SOL",
    quote: "USDT",
  },
  {
    symbol: "XRP/USDT",
    name: "Ripple / Tether USD (スポット)",
    base: "XRP",
    quote: "USDT",
  },
  {
    symbol: "DOT/USDT",
    name: "Polkadot / Tether USD (スポット)",
    base: "DOT",
    quote: "USDT",
  },
  {
    symbol: "AVAX/USDT",
    name: "Avalanche / Tether USD (スポット)",
    base: "AVAX",
    quote: "USDT",
  },
  {
    symbol: "LTC/USDT",
    name: "Litecoin / Tether USD (スポット)",
    base: "LTC",
    quote: "USDT",
  },
  {
    symbol: "UNI/USDT",
    name: "Uniswap / Tether USD (スポット)",
    base: "UNI",
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
