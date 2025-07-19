/**
 * データ収集ボタンの設定定義
 *
 * 各データ収集ボタンの設定を統一的に管理します。
 */

import { DataCollectionConfig } from "./DataCollectionButton";

/**
 * 全データ一括収集ボタンの設定
 */
export const allDataCollectionConfig: DataCollectionConfig = {
  endpoint: "/api/data-collection/all/bulk-collect",
  buttonText: "全データ取得",
  variant: "primary",
  confirmMessage: 
    "全データ（OHLCV・FR・OI）を一括取得します。\n\n" +
    "この処理には数分から十数分かかる場合があります。\n" +
    "テクニカル指標も自動計算されます。続行しますか？",
  loadingText: "収集中...",
};

/**
 * OHLCV収集ボタンの設定
 */
export const ohlcvCollectionConfig: DataCollectionConfig = {
  buttonText: "OHLCV収集",
  variant: "primary",
  useDataCollection: true,
  dataCollectionMethod: "ohlcv.collect",
  loadingText: "収集中...",
};

/**
 * Funding Rate収集ボタンの設定
 */
export const fundingRateCollectionConfig: DataCollectionConfig = {
  buttonText: "FR収集",
  variant: "success",
  useDataCollection: true,
  dataCollectionMethod: "fundingRate.collect",
  loadingText: "収集中...",
};

/**
 * Open Interest収集ボタンの設定
 */
export const openInterestCollectionConfig: DataCollectionConfig = {
  buttonText: "OI収集",
  variant: "warning",
  endpoint: "/api/open-interest/bulk-collect",
  confirmMessage: 
    "BTCの全期間OIデータを取得します。\n\n" +
    "この処理には数分かかる場合があります。続行しますか？",
  loadingText: "収集中...",
};

/**
 * Fear & Greed Index収集ボタンの設定
 */
export const fearGreedCollectionConfig: DataCollectionConfig = {
  endpoint: "/api/fear-greed/collect",
  buttonText: "FG収集",
  variant: "warning",
  loadingText: "収集中...",
};

/**
 * 外部市場データ収集ボタンの設定
 */
export const externalMarketCollectionConfig: DataCollectionConfig = {
  endpoint: "/api/external-market/collect",
  buttonText: "外部市場収集",
  variant: "secondary",
  loadingText: "収集中...",
};

/**
 * 単一シンボル Open Interest収集ボタンの設定を生成
 */
export const createSingleOpenInterestConfig = (symbol: string): DataCollectionConfig => ({
  endpoint: "/api/open-interest/collect",
  buttonText: `OI収集 (${symbol})`,
  variant: "warning",
  queryParams: {
    symbol: symbol,
    fetch_all: "true",
  },
  loadingText: "収集中...",
});

/**
 * 設定の型エクスポート
 */
export type { DataCollectionConfig };
