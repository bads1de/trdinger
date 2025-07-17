/**
 * アプリケーション全体で使用する定数定義
 *
 */

import { TradingPair, TimeFrameInfo } from "@/types/market-data";

/**
 * バックエンドAPIのベースURL
 */
export const BACKEND_API_URL = "http://127.0.0.1:8000";

/**
 * サポートされている取引ペア（BTC/USDT:USDT無期限先物のみ）
 *
 * 要件に従い、BTC/USDT:USDTの無期限先物のみをサポートします。
 */
export const SUPPORTED_TRADING_PAIRS: TradingPair[] = [
  // Bitcoin USDT無期限先物のみ
  {
    symbol: "BTC/USDT:USDT",
    name: "Bitcoin / USDT Perpetual",
    base: "BTC",
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

/**
 * デフォルトの取引ペア
 */
export const DEFAULT_TRADING_PAIR = "BTC/USDT:USDT";

/**
 * デフォルトの時間軸
 */
export const DEFAULT_TIMEFRAME = "1h";
