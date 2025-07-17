/**
 * 外部市場データテーブルコンポーネント
 *
 * 外部市場データ（SP500、NASDAQ、DXY、VIX）を表形式で表示します。
 * 共通のDataTableコンポーネントを使用して実装しています。
 */

"use client";

import React, { useMemo } from "react";
import {
  ExternalMarketData,
  EXTERNAL_MARKET_SYMBOLS,
} from "@/hooks/useExternalMarketData";
import DataTable from "./DataTable";
import { externalMarketColumns } from "@/components/common/tableColumns";
import { getSymbolName } from "@/components/common/tableColumns";

interface ExternalMarketDataTableProps {
  data: ExternalMarketData[];
  loading: boolean;
  error: string;
}

interface EnrichedExternalMarketData extends ExternalMarketData {
  name: string;
}

const ExternalMarketDataTable: React.FC<ExternalMarketDataTableProps> = ({
  data,
  loading,
  error,
}) => {
  const enrichedData = useMemo<EnrichedExternalMarketData[]>(
    () => data.map((row) => ({ ...row, name: getSymbolName(row.symbol) })),
    [data]
  );

  // テーブルカラム定義はtableColumns.tsxからインポート
  const columns = externalMarketColumns;

  // データがない場合の表示
  if (!loading && !error && (!enrichedData || enrichedData.length === 0)) {
    return (
      <div className="enterprise-card">
        <div className="p-8 text-center">
          <svg
            className="mx-auto h-12 w-12 text-gray-400"
            fill="none"
            stroke="currentColor"
            viewBox="0 0 24 24"
          >
            <path
              strokeLinecap="round"
              strokeLinejoin="round"
              strokeWidth={2}
              d="M9 19v-6a2 2 0 00-2-2H5a2 2 0 00-2 2v6a2 2 0 002 2h2a2 2 0 002-2zm0 0V9a2 2 0 012-2h2a2 2 0 012 2v10m-6 0a2 2 0 002 2h2a2 2 0 002-2m0 0V5a2 2 0 012-2h2a2 2 0 012 2v14a2 2 0 01-2 2h-2a2 2 0 01-2-2z"
            />
          </svg>
          <h3 className="mt-2 text-sm font-medium text-gray-900 dark:text-gray-100">
            外部市場データがありません
          </h3>
          <p className="mt-1 text-sm text-gray-500 dark:text-gray-400">
            「外部市場データ収集」ボタンをクリックしてデータを取得してください。
          </p>
          <p className="mt-1 text-xs text-gray-400 dark:text-gray-500">
            SP500、NASDAQ、DXY、VIXの日足データが取得されます。
          </p>
        </div>
      </div>
    );
  }

  // 共通DataTableコンポーネントを使用
  return (
    <DataTable
      data={enrichedData}
      columns={columns}
      title="外部市場データ"
      loading={loading}
      error={error}
      enableExport={true}
      enableSearch={true}
      searchKeys={["symbol", "name"]}
    />
  );
};

export default ExternalMarketDataTable;
