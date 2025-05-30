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
import DataTable from "./DataTable";
import { FundingRateData } from "@/types/strategy";
import { fundingRateColumns } from "@/components/common/tableColumns";

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
  return (
    <DataTable
      data={data}
      columns={fundingRateColumns}
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
