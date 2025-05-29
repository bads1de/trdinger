/**
 * オープンインタレストデータテーブルコンポーネント
 *
 * オープンインタレストデータを表形式で表示するコンポーネントです。
 * ソート、ページネーション、CSVエクスポート機能を提供します。
 *
 * @author Trdinger Development Team
 * @version 1.0.0
 */

"use client";

import React from "react";
import DataTable, { TableColumn } from "./DataTable";
import { OpenInterestData } from "@/types/strategy";

/**
 * オープンインタレストデータテーブルのプロパティ
 */
interface OpenInterestDataTableProps {
  /** オープンインタレストデータ */
  data: OpenInterestData[];
  /** ローディング状態 */
  loading?: boolean;
  /** エラーメッセージ */
  error?: string;
  /** テーブルのクラス名 */
  className?: string;
}

/**
 * 通貨ペアシンボルをフォーマットする関数
 */
const formatSymbol = (symbol: string): string => {
  // "BTC/USDT:USDT" -> "BTC/USDT"
  return symbol.replace(/:.*$/, "");
};

/**
 * 数値を通貨形式でフォーマットする関数
 */
const formatCurrency = (value: number): string => {
  return new Intl.NumberFormat("en-US", {
    style: "currency",
    currency: "USD",
    minimumFractionDigits: 0,
    maximumFractionDigits: 0,
  }).format(value);
};

/**
 * 数値をコンパクト形式でフォーマットする関数
 */
const formatCompactNumber = (value: number): string => {
  return new Intl.NumberFormat("en-US", {
    notation: "compact",
    maximumFractionDigits: 2,
  }).format(value);
};

/**
 * 日時をフォーマットする関数
 */
const formatDateTime = (dateString: string): string => {
  const date = new Date(dateString);
  return date.toLocaleString("ja-JP", {
    year: "numeric",
    month: "2-digit",
    day: "2-digit",
    hour: "2-digit",
    minute: "2-digit",
    second: "2-digit",
  });
};

/**
 * オープンインタレストデータテーブルコンポーネント
 */
const OpenInterestDataTable: React.FC<OpenInterestDataTableProps> = ({
  data,
  loading = false,
  error,
  className = "",
}) => {
  // テーブルカラムの定義
  const columns: TableColumn<OpenInterestData>[] = [
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
      key: "open_interest_value",
      header: "OI値 (USD)",
      width: "150px",
      sortable: true,
      formatter: (value: number) => (
        <span className="font-mono text-sm font-semibold text-green-400">
          {formatCurrency(value)}
        </span>
      ),
      cellClassName: "text-right",
    },
    {
      key: "open_interest_amount",
      header: "OI量",
      width: "120px",
      sortable: true,
      formatter: (value: number) => (
        <span className="font-mono text-sm text-blue-400">
          {formatCompactNumber(value)}
        </span>
      ),
      cellClassName: "text-right",
    },
    {
      key: "data_timestamp",
      header: "データ時刻",
      width: "180px",
      sortable: true,
      formatter: (value: string) => (
        <span className="font-mono text-sm text-gray-400">
          {formatDateTime(value)}
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
      title="📈 オープンインタレストデータ"
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

export default OpenInterestDataTable;
