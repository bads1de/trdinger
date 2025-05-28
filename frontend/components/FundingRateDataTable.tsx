/**
 * ファンディングレートデータテーブルコンポーネント
 *
 * ファンディングレートデータを表形式で表示するコンポーネントです。
 * ソート、ページネーション、CSVエクスポート機能を提供します。
 *
 * @author Trdinger Development Team
 * @version 1.0.0
 */

"use client";

import React from "react";
import DataTable, { TableColumn } from "./DataTable";
import { FundingRateData } from "@/types/strategy";

/**
 * ファンディングレートデータテーブルのプロパティ
 */
interface FundingRateDataTableProps {
  /** ファンディングレートデータ */
  data: FundingRateData[];
  /** ローディング状態 */
  loading?: boolean;
  /** エラーメッセージ */
  error?: string;
  /** テーブルのクラス名 */
  className?: string;
}

/**
 * ファンディングレートをパーセント形式でフォーマットする関数
 */
const formatFundingRate = (rate: number): string => {
  const percentage = rate * 100;
  const sign = percentage >= 0 ? "+" : "";
  return `${sign}${percentage.toFixed(6)}%`;
};

/**
 * ファンディングレートの色を取得する関数
 */
const getFundingRateColor = (rate: number): string => {
  if (rate > 0) {
    return "text-red-400"; // 正のファンディングレート（ロングが支払い）
  } else if (rate < 0) {
    return "text-green-400"; // 負のファンディングレート（ショートが支払い）
  }
  return "text-gray-100"; // ゼロ
};

/**
 * 価格を通貨形式でフォーマットする関数
 */
const formatPrice = (value: number | null): string => {
  if (value === null || value === undefined) {
    return "-";
  }
  return new Intl.NumberFormat("ja-JP", {
    style: "currency",
    currency: "USD",
    minimumFractionDigits: 2,
    maximumFractionDigits: 8,
  }).format(value);
};

/**
 * 日時を読みやすい形式でフォーマットする関数
 */
const formatDateTime = (timestamp: string): string => {
  const date = new Date(timestamp);
  return new Intl.DateTimeFormat("ja-JP", {
    year: "numeric",
    month: "2-digit",
    day: "2-digit",
    hour: "2-digit",
    minute: "2-digit",
    second: "2-digit",
    timeZone: "Asia/Tokyo",
  }).format(date);
};

/**
 * シンボルを短縮表示する関数
 */
const formatSymbol = (symbol: string): string => {
  // "BTC/USDT:USDT" -> "BTC/USDT"
  return symbol.split(":")[0];
};

/**
 * ファンディングレートデータテーブルコンポーネント
 */
const FundingRateDataTable: React.FC<FundingRateDataTableProps> = ({
  data,
  loading = false,
  error,
  className = "",
}) => {
  // テーブルカラムの定義
  const columns: TableColumn<FundingRateData>[] = [
    {
      key: "symbol",
      header: "通貨ペア",
      width: "120px",
      sortable: true,
      formatter: (value: string) => (
        <span className="font-semibold text-primary-400">{formatSymbol(value)}</span>
      ),
    },
    {
      key: "funding_rate",
      header: "ファンディングレート",
      width: "150px",
      sortable: true,
      formatter: (value: number) => (
        <span className={`font-mono text-sm font-semibold ${getFundingRateColor(value)}`}>
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
      formatter: (value: string) => (
        <span className="font-mono text-sm">{formatDateTime(value)}</span>
      ),
    },
    {
      key: "mark_price",
      header: "マーク価格",
      width: "120px",
      sortable: true,
      formatter: (value: number | null) => (
        <span className="font-mono text-sm text-blue-400">{formatPrice(value)}</span>
      ),
      cellClassName: "text-right",
    },
    {
      key: "index_price",
      header: "インデックス価格",
      width: "120px",
      sortable: true,
      formatter: (value: number | null) => (
        <span className="font-mono text-sm text-purple-400">{formatPrice(value)}</span>
      ),
      cellClassName: "text-right",
    },
    {
      key: "next_funding_timestamp",
      header: "次回ファンディング",
      width: "180px",
      sortable: true,
      formatter: (value: string | null) => (
        <span className="font-mono text-sm text-gray-400">
          {value ? formatDateTime(value) : "-"}
        </span>
      ),
    },
    {
      key: "timestamp",
      header: "取得時刻",
      width: "180px",
      sortable: true,
      formatter: (value: string) => (
        <span className="font-mono text-xs text-gray-500">{formatDateTime(value)}</span>
      ),
    },
  ];

  return (
    <DataTable
      data={data}
      columns={columns}
      title="📊 ファンディングレートデータ"
      loading={loading}
      error={error}
      pageSize={50}
      enableExport={true}
      enableSearch={true}
      searchKeys={["symbol"]}
      className={className}
    />
  );
};

export default FundingRateDataTable;
