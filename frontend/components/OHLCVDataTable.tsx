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
import DataTable from "./DataTable";
import { PriceData } from "@/types/strategy";
import { ohlcvColumns } from "@/config/tableColumns";

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

  return (
    <DataTable
      data={data}
      columns={ohlcvColumns}
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
