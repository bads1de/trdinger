/**
 * データ収集設定定数
 *
 * 各種データ収集ボタンの設定を定義します。
 * DataCollectionButtonコンポーネントで使用されます。
 *
 * @author Trdinger Development Team
 * @version 1.0.0
 */

import { DataCollectionConfig } from "@/components/common/DataCollectionButton";

/**
 * ファンディングレート収集設定（一括）
 */
export const FUNDING_RATE_BULK_CONFIG: DataCollectionConfig = {
  apiEndpoint: "/api/data/funding-rates/bulk",
  method: "POST",
  confirmMessage: 
    "BTC・ETHの全期間FRデータを取得します。\n\n" +
    "この処理には数分かかる場合があります。続行しますか？",
  buttonText: {
    idle: "BTC・ETHFR収集・保存",
    loading: "FR一括収集中...",
    success: "FR一括収集完了",
    error: "エラーが発生しました",
  },
  buttonIcon: {
    idle: <span className="text-blue-400">📊</span>,
  },
  description: "BTC・ETHの全期間FRデータを取得・保存します",
  successResetTime: 3000,
  errorResetTime: 5000,
};

/**
 * ファンディングレート収集設定（単一）
 */
export const createFundingRateSingleConfig = (symbol: string = "BTC/USDT"): DataCollectionConfig => ({
  apiEndpoint: `/api/data/funding-rates/collect?symbol=${encodeURIComponent(symbol)}&fetch_all=true`,
  method: "POST",
  buttonText: {
    idle: "FR収集・保存",
    loading: "FR収集中...",
    success: "FR収集完了",
    error: "エラーが発生しました",
  },
  buttonIcon: {
    idle: <span className="text-blue-400">📊</span>,
  },
  description: `${symbol}のFRデータを取得・保存します`,
  successResetTime: 3000,
  errorResetTime: 5000,
});

/**
 * オープンインタレスト収集設定（一括）
 */
export const OPEN_INTEREST_BULK_CONFIG: DataCollectionConfig = {
  apiEndpoint: "/api/data/open-interest/bulk-collect",
  method: "POST",
  confirmMessage:
    "BTC・ETHの全期間OIデータを取得します。\n\n" +
    "この処理には数分かかる場合があります。続行しますか？",
  buttonText: {
    idle: "📈 OI収集 (BTC・ETH)",
    loading: "一括収集中...",
    success: "✅ 完了",
    error: "❌ エラー",
  },
  description: "BTC・ETHの全期間OIデータを一括収集",
  successResetTime: 3000,
  errorResetTime: 5000,
};

/**
 * オープンインタレスト収集設定（単一）
 */
export const createOpenInterestSingleConfig = (symbol: string = "BTC/USDT"): DataCollectionConfig => ({
  apiEndpoint: `/api/data/open-interest/collect?symbol=${encodeURIComponent(symbol)}&fetch_all=true`,
  method: "POST",
  buttonText: {
    idle: `📈 OI収集 (${symbol})`,
    loading: "収集中...",
    success: "✅ 完了",
    error: "❌ エラー",
  },
  description: `${symbol}のOIデータを収集`,
  successResetTime: 3000,
  errorResetTime: 5000,
});

/**
 * OHLCV一括収集設定
 */
export const OHLCV_BULK_CONFIG: DataCollectionConfig = {
  apiEndpoint: "/api/data/ohlcv/bulk",
  method: "POST",
  confirmMessage:
    "全ての取引ペアと全ての時間軸でOHLCVデータを取得します。\n" +
    "この処理には時間がかかる場合があります。続行しますか？",
  buttonText: {
    idle: "全データ一括取得・保存",
    loading: "一括取得・保存中...",
    success: "一括取得・保存開始",
    error: "エラーが発生しました",
  },
  buttonIcon: {
    idle: (
      <svg className="w-5 h-5" fill="none" stroke="currentColor" viewBox="0 0 24 24">
        <path
          strokeLinecap="round"
          strokeLinejoin="round"
          strokeWidth={2}
          d="M4 16v1a3 3 0 003 3h10a3 3 0 003-3v-1m-4-8l-4-4m0 0L8 8m4-4v12"
        />
      </svg>
    ),
  },
  description: "全ての取引ペアと時間軸のOHLCVデータを一括取得・保存",
  successResetTime: 10000,
  errorResetTime: 10000,
};

/**
 * 全データ一括収集設定
 */
export const ALL_DATA_COLLECTION_CONFIG: DataCollectionConfig = {
  apiEndpoint: "/api/data/all/bulk-collect", // 注意: このエンドポイントは実装されていない可能性があります
  method: "POST",
  confirmMessage:
    "全データ（OHLCV・FR・OI）を一括取得します。\n\n" +
    "この処理には数分から十数分かかる場合があります。続行しますか？",
  buttonText: {
    idle: "全データ一括取得",
    loading: "全データ収集中...",
    success: "全データ収集完了",
    error: "収集エラー",
  },
  buttonIcon: {
    idle: (
      <svg className="w-4 h-4" fill="none" stroke="currentColor" viewBox="0 0 24 24">
        <path
          strokeLinecap="round"
          strokeLinejoin="round"
          strokeWidth={2}
          d="M4 7v10c0 2.21 1.79 4 4 4h8c2.21 0 4-1.79 4-4V7M4 7l8-4 8 4M4 7l8 4 8-4"
        />
      </svg>
    ),
  },
  description: "OHLCV・FR・OIの全データを一括収集",
  successResetTime: 10000,
  errorResetTime: 10000,
};

/**
 * 共通のボタンアイコン
 */
export const COMMON_ICONS = {
  loading: <div className="animate-spin rounded-full h-4 w-4 border-b-2 border-white" />,
  success: <span className="text-green-400">✅</span>,
  error: <span className="text-red-400">❌</span>,
  chart: <span className="text-blue-400">📊</span>,
  trend: <span className="text-green-400">📈</span>,
  download: (
    <svg className="w-4 h-4" fill="none" stroke="currentColor" viewBox="0 0 24 24">
      <path
        strokeLinecap="round"
        strokeLinejoin="round"
        strokeWidth={2}
        d="M12 10v6m0 0l-3-3m3 3l3-3m2 8H7a2 2 0 01-2-2V5a2 2 0 012-2h5.586a1 1 0 01.707.293l5.414 5.414a1 1 0 01.293.707V19a2 2 0 01-2 2z"
      />
    </svg>
  ),
};
