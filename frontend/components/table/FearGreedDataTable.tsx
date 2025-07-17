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
import { fearGreedColumns } from "@/components/common/tableColumns";

interface FearGreedDataTableProps {
  data: FearGreedIndexData[];
  loading: boolean;
  error: string | null;
}

const FearGreedDataTable: React.FC<FearGreedDataTableProps> = ({
  data,
  loading,
  error,
}) => {
  // テーブルカラム定義はtableColumns.tsxからインポート
  const columns = fearGreedColumns;

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
