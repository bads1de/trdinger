/**
 * FRデータテーブルコンポーネント
 *
 * FRデータを表形式で表示するコンポーネントです。
 * ソート、ページネーション、CSVエクスポート機能を提供します。
 *
 * @author Trdinger Development Team
 * @version 1.0.0
 */

"use client";

import React from "react";
import DataTable, { TableColumn } from "./DataTable";
import { FundingRateData } from "@/types/strategy";
import {
  formatDateTime,
  formatPrice,
  formatSymbol,
  formatFundingRate,
  getFundingRateColor,
} from "@/utils/formatters";

/**
 * FRデータテーブルのプロパティ
 */
interface FundingRateDataTableProps {
  /** FRデータ */
  data: FundingRateData[];
  /** ローディング状態 */
  loading?: boolean;
  /** エラーメッセージ */
  error?: string;
  /** テーブルのクラス名 */
  className?: string;
}

/**
 * FRデータテーブルコンポーネント
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
        <span className="font-mono text-xs text-gray-500">
          {formatDateTime(value)}
        </span>
      ),
    },
  ];

  return (
    <DataTable
      data={data}
      columns={columns}
      title="📊 FRデータ"
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
