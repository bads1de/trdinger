/**
 * データテーブルのカラム定義設定
 *
 * 各データテーブルのカラム定義を統一的に管理します。
 * フォーマッター関数とスタイリングを含む完全なカラム設定を提供します。
 *
 */

import React from "react";
import { TableColumn } from "@/types/common";
import { FundingRateData } from "@/types/funding-rate";
import { PriceData } from "@/types/market-data";
import { OpenInterestData } from "@/types/open-interest";
import {
  ExternalMarketData,
  EXTERNAL_MARKET_SYMBOLS,
} from "@/hooks/useExternalMarketData";
import { formatDateTime } from "@/utils/formatters";
import {
  formatPrice,
  formatSymbol,
  formatFundingRate,
  formatCurrency,
  formatVolume,
  formatLargeNumber,
} from "@/utils/financialFormatters";
import { getFundingRateColor, getPriceChangeColor } from "@/utils/colorUtils";

/**
 * 数値をフォーマット
 */
const formatNumber = (value: number | null, decimals: number = 2): string => {
  if (value === null || value === undefined) return "-";
  return value.toLocaleString("ja-JP", {
    minimumFractionDigits: decimals,
    maximumFractionDigits: decimals,
  });
};

/**
 * 日時をフォーマット
 */
const formatDateTimeExternalMarket = (dateString: string): string => {
  try {
    const date = new Date(dateString);
    return date.toLocaleString("ja-JP", {
      year: "numeric",
      month: "2-digit",
      day: "2-digit",
      minute: "2-digit",
    });
  } catch {
    return dateString;
  }
};

interface EnrichedExternalMarketData extends ExternalMarketData {
  name: string;
}

/**
 * シンボル名を取得
 */
export const getSymbolName = (symbol: string): string => {
  return (
    EXTERNAL_MARKET_SYMBOLS[symbol as keyof typeof EXTERNAL_MARKET_SYMBOLS] ||
    symbol
  );
};

/**
 * 外部市場データテーブルのカラム定義
 */
export const externalMarketColumns: TableColumn<EnrichedExternalMarketData>[] =
  [
    {
      key: "symbol",
      header: "シンボル",
      sortable: true,
      formatter: (value: string, row: EnrichedExternalMarketData) => (
        <span className="font-semibold text-primary-400">
          {getSymbolName(row.symbol)}
        </span>
      ),
    },
    {
      key: "name",
      header: "名称",
      sortable: true,
      formatter: (value: string, row: EnrichedExternalMarketData) => (
        <span className="font-semibold text-primary-400">
          {getSymbolName(row.symbol)}
        </span>
      ),
    },
    {
      key: "open",
      header: "始値",
      formatter: (value: number) => formatNumber(value),
      sortable: true,
      cellClassName: "text-right font-mono",
    },
    {
      key: "high",
      header: "高値",
      formatter: (value: number) => formatNumber(value),
      sortable: true,
      cellClassName: "text-right font-mono",
    },
    {
      key: "low",
      header: "安値",
      formatter: (value: number) => formatNumber(value),
      sortable: true,
      cellClassName: "text-right font-mono",
    },
    {
      key: "close",
      header: "終値",
      formatter: (value: number) => formatNumber(value),
      sortable: true,
      cellClassName: "text-right font-mono",
    },
    {
      key: "volume",
      header: "出来高",
      formatter: (value: number) => formatNumber(value, 0),
      sortable: true,
      cellClassName: "text-right font-mono",
    },
    {
      key: "data_timestamp",
      header: "日時",
      formatter: (value: string) => formatDateTimeExternalMarket(value),
      sortable: true,
    },
  ];

/**
 * ファンディングレートデータテーブルのカラム定義
 */
export const fundingRateColumns: TableColumn<FundingRateData>[] = [
  {
    key: "symbol",
    header: "通貨ペア",
    width: "120px",
    sortable: true,
    formatter: (value: string) => (
      <span className="font-semibold text-primary-400">
        {formatSymbol(value)}
      </span>
    ),
  },
  {
    key: "funding_rate",
    header: "FR",
    width: "150px",
    sortable: true,
    formatter: (value: number) => (
      <span
        className={`font-mono text-sm font-semibold ${getFundingRateColor(
          value
        )}`}
      >
        {formatFundingRate(value)}
      </span>
    ),
    cellClassName: "text-right",
  },
  {
    key: "funding_timestamp",
    header: "ファンディング時刻",
    width: "180px",
    sortable: true,
    formatter: (value: string) => {
      const { date, time } = formatDateTime(value);
      if (time) {
        return (
          <span className="font-mono text-sm">
            {date}
            <br />
            {time}
          </span>
        );
      }
      return <span className="font-mono text-sm">{date}</span>;
    },
  },
  {
    key: "mark_price",
    header: "マーク価格",
    width: "120px",
    sortable: true,
    formatter: (value: number | null) => (
      <span className="font-mono text-sm text-blue-400">
        {formatPrice(value)}
      </span>
    ),
    cellClassName: "text-right",
  },
  {
    key: "index_price",
    header: "インデックス価格",
    width: "120px",
    sortable: true,
    formatter: (value: number | null) => (
      <span className="font-mono text-sm text-purple-400">
        {formatPrice(value)}
      </span>
    ),
    cellClassName: "text-right",
  },
  {
    key: "next_funding_timestamp",
    header: "次回ファンディング",
    width: "180px",
    sortable: true,
    formatter: (value: string | null) => {
      if (!value)
        return <span className="font-mono text-sm text-gray-400">-</span>;
      const { date, time } = formatDateTime(value);
      if (time) {
        return (
          <span className="font-mono text-sm text-gray-400">
            {date}
            <br />
            {time}
          </span>
        );
      }
      return <span className="font-mono text-sm text-gray-400">{date}</span>;
    },
  },
  {
    key: "timestamp",
    header: "取得時刻",
    width: "180px",
    sortable: true,
    formatter: (value: string) => {
      const { date, time } = formatDateTime(value);
      if (time) {
        return (
          <span className="font-mono text-xs text-gray-500">
            {date}
            <br />
            {time}
          </span>
        );
      }
      return <span className="font-mono text-xs text-gray-500">{date}</span>;
    },
  },
];

/**
 * OHLCVデータテーブルのカラム定義
 */
export const ohlcvColumns: TableColumn<PriceData>[] = [
  {
    key: "timestamp",
    header: "日時",
    width: "180px",
    sortable: true,
    formatter: (value: string) => {
      const { date, time } = formatDateTime(value);
      if (time) {
        return (
          <span className="font-mono text-sm">
            {date}
            <br />
            {time}
          </span>
        );
      }
      return <span className="font-mono text-sm">{date}</span>;
    },
  },
  {
    key: "open",
    header: "始値",
    width: "120px",
    sortable: true,
    formatter: (value: number) => (
      <span className="font-mono text-sm">{formatCurrency(value)}</span>
    ),
    cellClassName: "text-right",
  },
  {
    key: "high",
    header: "高値",
    width: "120px",
    sortable: true,
    formatter: (value: number) => (
      <span className="font-mono text-sm text-green-400">
        {formatCurrency(value)}
      </span>
    ),
    cellClassName: "text-right",
  },
  {
    key: "low",
    header: "安値",
    width: "120px",
    sortable: true,
    formatter: (value: number) => (
      <span className="font-mono text-sm text-red-400">
        {formatCurrency(value)}
      </span>
    ),
    cellClassName: "text-right",
  },
  {
    key: "close",
    header: "終値",
    width: "120px",
    sortable: true,
    formatter: (value: number, row: PriceData) => (
      <span
        className={`font-mono text-sm font-semibold ${getPriceChangeColor(
          row.open,
          value
        )}`}
      >
        {formatCurrency(value)}
      </span>
    ),
    cellClassName: "text-right",
  },
  {
    key: "volume",
    header: "出来高",
    width: "100px",
    sortable: true,
    formatter: (value: number) => (
      <span className="font-mono text-sm text-blue-400">
        {formatVolume(value)}
      </span>
    ),
    cellClassName: "text-right",
  },
];

/**
 * Fear & Greed Index 値に基づく色を取得
 *
 * 0-100のスケールで、低いほど恐怖（赤）、高いほど強欲（緑）を表す
 */
const getValueColor = (value: number): string => {
  // getScoreColorClassを使用するには0-1のスケールに変換する必要があるが、
  // Fear & Greedの色分けは特殊なので、カスタム実装を維持
  if (value <= 20) return "text-red-600 dark:text-red-400";
  if (value <= 40) return "text-orange-600 dark:text-orange-400";
  if (value <= 60) return "text-yellow-600 dark:text-yellow-400";
  if (value <= 80) return "text-green-600 dark:text-green-400";
  return "text-emerald-600 dark:text-emerald-400";
};

/**
 * Fear & Greed Index 分類に基づく背景色を取得
 */
const getClassificationBadge = (classification: string): string => {
  switch (classification) {
    case "Extreme Fear":
      return "bg-red-100 text-red-800 dark:bg-red-900 dark:text-red-200";
    case "Fear":
      return "bg-orange-100 text-orange-800 dark:bg-orange-900 dark:text-orange-200";
    case "Neutral":
      return "bg-yellow-100 text-yellow-800 dark:bg-yellow-900 dark:text-yellow-200";
    case "Greed":
      return "bg-green-100 text-green-800 dark:bg-green-900 dark:text-green-200";
    case "Extreme Greed":
      return "bg-emerald-100 text-emerald-800 dark:bg-emerald-900 dark:text-emerald-200";
    default:
      return "bg-gray-100 text-gray-800 dark:bg-gray-900 dark:text-gray-200";
  }
};

/**
 * Fear & Greed Index データテーブルのカラム定義
 */
export const fearGreedColumns: TableColumn<any>[] = [
  {
    key: "data_timestamp",
    header: "日付",
    formatter: (value: string) => formatDateTime(value).date,
    sortable: true,
  },
  {
    key: "value",
    header: "値",
    formatter: (value: number) => (
      <div className="flex items-center">
        <span className={`text-2xl font-bold ${getValueColor(value)}`}>
          {formatNumber(value, 0)}
        </span>
        <span className="ml-2 text-sm text-secondary-500 dark:text-secondary-400">
          / 100
        </span>
      </div>
    ),
    sortable: true,
  },
  {
    key: "value_classification",
    header: "分類",
    formatter: (value: string) => (
      <span
        className={`inline-flex px-2 py-1 text-xs font-semibold rounded-full ${getClassificationBadge(
          value
        )}`}
      >
        {value}
      </span>
    ),
    sortable: true,
  },
  {
    key: "timestamp",
    header: "取得時刻",
    formatter: (value: string) => formatDateTime(value).dateTime,
    sortable: true,
  },
];

/**
 * オープンインタレストデータテーブルのカラム定義
 */
export const openInterestColumns: TableColumn<OpenInterestData>[] = [
  {
    key: "symbol",
    header: "通貨ペア",
    width: "120px",
    sortable: true,
    formatter: (value: string) => (
      <span className="font-semibold text-primary-400">
        {formatSymbol(value)}
      </span>
    ),
  },
  {
    key: "open_interest_value",
    header: "OI値 (USD)",
    width: "180px",
    sortable: true,
    formatter: (value: number) => (
      <span className="font-mono text-sm font-semibold text-green-400">
        ${formatLargeNumber(value)}
      </span>
    ),
    cellClassName: "text-left",
  },
  {
    key: "data_timestamp",
    header: "データ時刻",
    width: "180px",
    sortable: true,
    formatter: (value: string) => {
      const { date, time } = formatDateTime(value);
      if (time) {
        return (
          <span className="font-mono text-sm text-gray-400">
            {date}
            <br />
            {time}
          </span>
        );
      }
      return <span className="font-mono text-sm text-gray-400">{date}</span>;
    },
  },
  {
    key: "timestamp",
    header: "取得時刻",
    width: "180px",
    sortable: true,
    formatter: (value: string) => {
      const { date, time } = formatDateTime(value);
      if (time) {
        return (
          <span className="font-mono text-xs text-gray-500">
            {date}
            <br />
            {time}
          </span>
        );
      }
      return <span className="font-mono text-xs text-gray-500">{date}</span>;
    },
  },
];
