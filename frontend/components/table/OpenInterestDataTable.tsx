/**
 * OIデータテーブルコンポーネント
 *
 * OIデータを表形式で表示するコンポーネントです。
 * ソート、ページネーション、CSVエクスポート機能を提供します。
 *
 */

"use client";

import React from "react";
import DataTable from "./DataTable";
import { OpenInterestData } from "@/types/strategy";
import { openInterestColumns } from "@/components/common/tableColumns";

/**
 * OIデータテーブルのプロパティ
 */
interface OpenInterestDataTableProps {
  /** OIデータ */
  data: OpenInterestData[];
  /** ローディング状態 */
  loading?: boolean;
  /** エラーメッセージ */
  error?: string;
  /** テーブルのクラス名 */
  className?: string;
}

/**
 * OIデータテーブルコンポーネント
 */
const OpenInterestDataTable: React.FC<OpenInterestDataTableProps> = ({
  data,
  loading = false,
  error,
  className = "",
}) => {
  return (
    <DataTable
      data={data}
      columns={openInterestColumns}
      title="📈 OIデータ"
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
