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
