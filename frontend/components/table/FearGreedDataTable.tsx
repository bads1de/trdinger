/**
 * Fear & Greed Index データテーブルコンポーネント
 *
 * Fear & Greed Index データを表形式で表示します。
 * 共通のDataTableコンポーネントを使用して実装しています。
 */

"use client";

import React, { useMemo } from "react";
import { FearGreedIndexData } from "@/app/api/data/fear-greed/route";
import DataTable from "./DataTable";
import { TableColumn } from "@/types/common";

interface FearGreedDataTableProps {
  data: FearGreedIndexData[];
  loading: boolean;
  error: string | null;
}

/**
 * Fear & Greed Index 値に基づく色を取得
 */
const getValueColor = (value: number): string => {
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
 * 日時をフォーマット
 */
const formatDateTime = (dateString: string): string => {
  try {
    const date = new Date(dateString);
    return date.toLocaleString("ja-JP", {
      year: "numeric",
      month: "2-digit",
      day: "2-digit",
      hour: "2-digit",
      minute: "2-digit",
      timeZone: "Asia/Tokyo",
    });
  } catch {
    return dateString;
  }
};

/**
 * 日付のみをフォーマット
 */
const formatDate = (dateString: string): string => {
  try {
    const date = new Date(dateString);
    return date.toLocaleDateString("ja-JP", {
      year: "numeric",
      month: "2-digit",
      day: "2-digit",
      timeZone: "Asia/Tokyo",
    });
  } catch {
    return dateString;
  }
};

/**
 * Fear & Greed Index データテーブルコンポーネント
 */
const FearGreedDataTable: React.FC<FearGreedDataTableProps> = ({
  data,
  loading,
  error,
}) => {
  // テーブルカラム定義
  const columns = useMemo<TableColumn<FearGreedIndexData>[]>(
    () => [
      {
        key: "data_timestamp",
        header: "日付",
        formatter: (value) => formatDate(value as string),
        sortable: true,
      },
      {
        key: "value",
        header: "値",
        formatter: (value) => (
          <div className="flex items-center">
            <span
              className={`text-2xl font-bold ${getValueColor(value as number)}`}
            >
              {value}
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
        formatter: (value) => (
          <span
            className={`inline-flex px-2 py-1 text-xs font-semibold rounded-full ${getClassificationBadge(
              value as string
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
        formatter: (value) => formatDateTime(value as string),
        sortable: true,
      },
    ],
    []
  );

  // データがない場合の表示
  if (!loading && !error && (!data || data.length === 0)) {
    return (
      <div className="enterprise-card">
        <div className="p-6">
          <div className="text-center text-secondary-600 dark:text-secondary-400">
            <p className="text-lg font-medium mb-2">📊 データがありません</p>
            <p className="text-sm">
              Fear & Greed Index データを収集してください
            </p>
          </div>
        </div>
      </div>
    );
  }

  // 共通DataTableコンポーネントを使用
  return (
    <DataTable
      data={data}
      columns={columns}
      title="Fear & Greed Index データ"
      loading={loading}
      error={error || ""}
      enableExport={true}
      enableSearch={true}
      searchKeys={["value_classification"]}
      className="mb-4"
    />
  );
};

export default FearGreedDataTable;
