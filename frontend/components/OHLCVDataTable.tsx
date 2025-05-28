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
 * 数値を通貨形式でフォーマットする関数
 */
const formatCurrency = (value: number): string => {
  return new Intl.NumberFormat("ja-JP", {
    style: "currency",
    currency: "USD",
    minimumFractionDigits: 2,
    maximumFractionDigits: 8,
  }).format(value);
};

/**
 * 出来高を読みやすい形式でフォーマットする関数
 */
const formatVolume = (value: number): string => {
  if (value >= 1e9) {
    return `${(value / 1e9).toFixed(2)}B`;
  } else if (value >= 1e6) {
    return `${(value / 1e6).toFixed(2)}M`;
  } else if (value >= 1e3) {
    return `${(value / 1e3).toFixed(2)}K`;
  }
  return value.toFixed(2);
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
 * 価格変動の色を取得する関数
 */
const getPriceChangeColor = (open: number, close: number): string => {
  if (close > open) {
    return "text-green-400"; // 上昇（陽線）
  } else if (close < open) {
    return "text-red-400"; // 下降（陰線）
  }
  return "text-gray-100"; // 変化なし
};

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
