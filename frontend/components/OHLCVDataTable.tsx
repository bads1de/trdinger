/**
 * OHLCVデータテーブルコンポーネント
 *
 * OHLCVデータを表形式で表示するコンポーネントです。
 * ソート、ページネーション、CSVエクスポート機能を提供します。
 *
 * @author Trdinger Development Team
 * @version 1.0.0
 */

"use client";

import React from "react";
import DataTable, { TableColumn } from "./DataTable";
import { PriceData } from "@/types/strategy";
import {
  formatDateTime,
  formatCurrency,
  formatVolume,
  getPriceChangeColor,
} from "@/utils/formatters";

/**
 * OHLCVデータテーブルのプロパティ
 */
interface OHLCVDataTableProps {
  /** OHLCVデータ */
  data: PriceData[];
  /** 通貨ペア */
  symbol: string;
  /** 時間軸 */
  timeframe: string;
  /** ローディング状態 */
  loading?: boolean;
  /** エラーメッセージ */
  error?: string;
  /** テーブルのクラス名 */
  className?: string;
}

/**
 * OHLCVデータテーブルコンポーネント
 */
const OHLCVDataTable: React.FC<OHLCVDataTableProps> = ({
  data,
  symbol,
  timeframe,
  loading = false,
  error,
  className = "",
}) => {
  // テーブルカラムの定義
  // 注意: 変動率カラムは計算機能未実装のため一時的に削除済み
  const columns: TableColumn<PriceData>[] = [
    {
      key: "timestamp",
      header: "日時",
      width: "180px",
      sortable: true,
      formatter: (value: string) => (
        <span className="font-mono text-sm">{formatDateTime(value)}</span>
      ),
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

  return (
    <DataTable
      data={data}
      columns={columns}
      title={`📊 ${symbol} - ${timeframe} OHLCVデータ`}
      loading={loading}
      error={error}
      pageSize={50}
      enableExport={true}
      enableSearch={false}
      className={className}
    />
  );
};

export default OHLCVDataTable;
