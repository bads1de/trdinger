/**
 * アプリケーション全体で使用する定数定義
 *
 */

import { TradingPair, TimeFrameInfo } from "@/types/market-data";

/**
 * バックエンドAPIのベースURL
 */
export const BACKEND_API_URL =
  process.env.NEXT_PUBLIC_BACKEND_API_URL || "http://localhost:8000";

/**
 * サポートされている取引ペア（Bybit USDT無期限先物）
 *
 * バックエンドの SUPPORTED_MARKET_SYMBOLS（app/config/constants.py）と
 * 内容を一致させて運用する。
 */
export const SUPPORTED_TRADING_PAIRS: TradingPair[] = [
  {
    symbol: "BTC/USDT:USDT",
    name: "Bitcoin / USDT Perpetual",
    base: "BTC",
    quote: "USDT",
  },
  {
    symbol: "ETH/USDT:USDT",
    name: "Ethereum / USDT Perpetual",
    base: "ETH",
    quote: "USDT",
  },
];

/**
 * サポートされている時間軸
 */
export const SUPPORTED_TIMEFRAMES: TimeFrameInfo[] = [
  {
    value: "15m",
    label: "15分",
    description: "15分足データ",
  },
  {
    value: "30m",
    label: "30分",
    description: "30分足データ",
  },
  {
    value: "1h",
    label: "1時間",
    description: "1時間足データ",
  },
  {
    value: "4h",
    label: "4時間",
    description: "4時間足データ",
  },
  {
    value: "1d",
    label: "1日",
    description: "日足データ",
  },
];
