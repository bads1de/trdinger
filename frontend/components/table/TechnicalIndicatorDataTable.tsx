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
import DataTable from "./DataTable";
import { TechnicalIndicatorData } from "@/types/strategy";
import { technicalIndicatorColumns } from "@/components/common/tableColumns";

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
 * テクニカル指標データテーブルコンポーネント
 */
const TechnicalIndicatorDataTable: React.FC<
  TechnicalIndicatorDataTableProps
> = ({ data, loading = false, error, className = "" }) => {
  return (
    <DataTable
      data={data}
      columns={technicalIndicatorColumns}
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
