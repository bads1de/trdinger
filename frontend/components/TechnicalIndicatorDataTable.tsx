/**
 * テクニカル指標データテーブルコンポーネント
 *
 * テクニカル指標データを表形式で表示するコンポーネントです。
 * ソート、ページネーション、CSVエクスポート機能を提供します。
 *
 * @author Trdinger Development Team
 * @version 1.0.0
 */

"use client";

import React from "react";
import DataTable, { TableColumn } from "./DataTable";
import { TechnicalIndicatorData } from "@/types/strategy";
import {
  formatDateTime,
  formatSymbol,
} from "@/utils/formatters";

/**
 * テクニカル指標データテーブルのプロパティ
 */
interface TechnicalIndicatorDataTableProps {
  /** テクニカル指標データ */
  data: TechnicalIndicatorData[];
  /** ローディング状態 */
  loading?: boolean;
  /** エラーメッセージ */
  error?: string;
  /** テーブルのクラス名 */
  className?: string;
}

/**
 * テクニカル指標値のフォーマット
 */
const formatIndicatorValue = (value: number, indicatorType: string): string => {
  if (indicatorType === "RSI") {
    // RSIは0-100の範囲で小数点2桁
    return value.toFixed(2);
  } else {
    // SMA, EMAは価格なので適切な桁数で表示
    if (value >= 1000) {
      return value.toLocaleString("en-US", {
        minimumFractionDigits: 2,
        maximumFractionDigits: 2,
      });
    } else {
      return value.toFixed(4);
    }
  }
};

/**
 * 指標タイプの色を取得
 */
const getIndicatorTypeColor = (indicatorType: string): string => {
  switch (indicatorType) {
    case "SMA":
      return "text-blue-400";
    case "EMA":
      return "text-green-400";
    case "RSI":
      return "text-purple-400";
    case "MACD":
      return "text-orange-400";
    default:
      return "text-gray-400";
  }
};

/**
 * RSI値の色を取得（買われすぎ・売られすぎ判定）
 */
const getRSIColor = (value: number): string => {
  if (value >= 70) {
    return "text-red-400"; // 買われすぎ
  } else if (value <= 30) {
    return "text-green-400"; // 売られすぎ
  } else {
    return "text-gray-300"; // 中立
  }
};

/**
 * テクニカル指標データテーブルコンポーネント
 */
const TechnicalIndicatorDataTable: React.FC<TechnicalIndicatorDataTableProps> = ({
  data,
  loading = false,
  error,
  className = "",
}) => {
  // テーブルカラムの定義
  const columns: TableColumn<TechnicalIndicatorData>[] = [
    {
      key: "symbol",
      header: "通貨ペア",
      width: "100px",
      sortable: true,
      formatter: (value: string) => (
        <span className="font-semibold text-primary-400">
          {formatSymbol(value)}
        </span>
      ),
    },
    {
      key: "timeframe",
      header: "時間枠",
      width: "80px",
      sortable: true,
      formatter: (value: string) => (
        <span className="font-mono text-sm text-gray-300">
          {value}
        </span>
      ),
    },
    {
      key: "indicator_type",
      header: "指標",
      width: "80px",
      sortable: true,
      formatter: (value: string) => (
        <span className={`font-semibold text-sm ${getIndicatorTypeColor(value)}`}>
          {value}
        </span>
      ),
    },
    {
      key: "period",
      header: "期間",
      width: "60px",
      sortable: true,
      formatter: (value: number) => (
        <span className="font-mono text-sm text-gray-300">
          {value}
        </span>
      ),
      cellClassName: "text-center",
    },
    {
      key: "value",
      header: "値",
      width: "120px",
      sortable: true,
      formatter: (value: number, row: TechnicalIndicatorData) => {
        const formattedValue = formatIndicatorValue(value, row.indicator_type);
        const colorClass = row.indicator_type === "RSI" 
          ? getRSIColor(value) 
          : "text-yellow-400";
        
        return (
          <span className={`font-mono text-sm font-semibold ${colorClass}`}>
            {formattedValue}
          </span>
        );
      },
      cellClassName: "text-right",
    },
    {
      key: "signal_value",
      header: "シグナル",
      width: "100px",
      sortable: true,
      formatter: (value: number | null, row: TechnicalIndicatorData) => {
        if (value === null || value === undefined) {
          return <span className="text-gray-500">-</span>;
        }
        const formattedValue = formatIndicatorValue(value, row.indicator_type);
        return (
          <span className="font-mono text-sm text-cyan-400">
            {formattedValue}
          </span>
        );
      },
      cellClassName: "text-right",
    },
    {
      key: "histogram_value",
      header: "ヒストグラム",
      width: "100px",
      sortable: true,
      formatter: (value: number | null, row: TechnicalIndicatorData) => {
        if (value === null || value === undefined) {
          return <span className="text-gray-500">-</span>;
        }
        const formattedValue = formatIndicatorValue(value, row.indicator_type);
        const colorClass = value >= 0 ? "text-green-400" : "text-red-400";
        return (
          <span className={`font-mono text-sm ${colorClass}`}>
            {formattedValue}
          </span>
        );
      },
      cellClassName: "text-right",
    },
    {
      key: "timestamp",
      header: "データ時刻",
      width: "180px",
      sortable: true,
      formatter: (value: string) => (
        <span className="font-mono text-sm text-gray-400">
          {formatDateTime(value)}
        </span>
      ),
    },
  ];

  return (
    <DataTable
      data={data}
      columns={columns}
      title="📈 テクニカル指標データ"
      loading={loading}
      error={error}
      pageSize={50}
      enableExport={true}
      enableSearch={true}
      searchKeys={["symbol", "indicator_type", "timeframe"]}
      className={className}
    />
  );
};

export default TechnicalIndicatorDataTable;
