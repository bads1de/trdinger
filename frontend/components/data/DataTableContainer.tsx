import React, { useMemo } from "react";
import DataTable from "@/components/table/DataTable";
import {
  ohlcvColumns,
  fundingRateColumns,
  openInterestColumns,
  fearGreedColumns,
  externalMarketColumns,
  getSymbolName,
} from "@/components/common/tableColumns";
import { PriceData, TimeFrame } from "@/types/market-data";
import { FundingRateData } from "@/types/funding-rate";
import { OpenInterestData } from "@/types/open-interest";
import { ExternalMarketData } from "@/hooks/useExternalMarketData";

/**
 * 外部市場データの拡張型（名前フィールド付き）
 */
interface EnrichedExternalMarketData extends ExternalMarketData {
  name: string;
}

interface DataTableContainerProps {
  selectedSymbol: string;
  selectedTimeFrame: TimeFrame;
  activeTab:
    | "ohlcv"
    | "funding"
    | "openinterest"
    | "feargreed"
    | "externalmarket";
  setActiveTab: (
    tab: "ohlcv" | "funding" | "openinterest" | "feargreed" | "externalmarket"
  ) => void;
  ohlcvData: PriceData[];
  loading: boolean;
  error: string;
  fundingRateData: FundingRateData[];
  fundingLoading: boolean;
  fundingError: string;
  openInterestData: OpenInterestData[];
  openInterestLoading: boolean;
  openInterestError: string;
  fearGreedData?: any[];
  fearGreedLoading?: boolean;
  fearGreedError?: string;
  externalMarketData?: ExternalMarketData[];
  externalMarketLoading?: boolean;
  externalMarketError?: string;
}

const DataTableContainer: React.FC<DataTableContainerProps> = ({
  selectedSymbol,
  selectedTimeFrame,
  activeTab,
  setActiveTab,
  ohlcvData,
  loading,
  error,
  fundingRateData,
  fundingLoading,
  fundingError,
  openInterestData,
  openInterestLoading,
  openInterestError,
  fearGreedData = [],
  fearGreedLoading = false,
  fearGreedError = "",
  externalMarketData = [],
  externalMarketLoading = false,
  externalMarketError = "",
}) => {
  // 外部市場データに名前フィールドを追加
  const enrichedExternalMarketData = useMemo<EnrichedExternalMarketData[]>(
    () =>
      externalMarketData.map((row) => ({
        ...row,
        name: getSymbolName(row.symbol),
      })),
    [externalMarketData]
  );

  // テーブル設定オブジェクト
  const tableConfigs = {
    ohlcv: {
      columns: ohlcvColumns,
      title: `📊 ${selectedSymbol} - ${selectedTimeFrame} OHLCVデータ`,
      pageSize: 50,
      enableExport: true,
      enableSearch: false,
      searchKeys: undefined as (keyof PriceData)[] | undefined,
    },
    funding: {
      columns: fundingRateColumns,
      title: "📊 FRデータ",
      pageSize: 50,
      enableExport: true,
      enableSearch: true,
      searchKeys: ["symbol"] as (keyof FundingRateData)[],
    },
    openinterest: {
      columns: openInterestColumns,
      title: "📈 OIデータ",
      pageSize: 50,
      enableExport: true,
      enableSearch: true,
      searchKeys: ["symbol"] as (keyof OpenInterestData)[],
    },
    feargreed: {
      columns: fearGreedColumns,
      title: "Fear & Greed Index データ",
      enableExport: true,
      enableSearch: true,
      searchKeys: ["value_classification"] as (keyof any)[],
    },
    externalmarket: {
      columns: externalMarketColumns,
      title: "外部市場データ",
      enableExport: true,
      enableSearch: true,
      searchKeys: ["symbol", "name"] as (keyof EnrichedExternalMarketData)[],
    },
  };
  return (
    <div className="enterprise-card animate-slide-up">
      <div className="p-6">
        {/* タブヘッダー */}
        <div className="flex items-center justify-between mb-6">
          <div className="flex items-center gap-4">
            <h2 className="text-xl font-semibold text-secondary-900 dark:text-secondary-100">
              📊 {selectedSymbol}
            </h2>
            <div className="flex bg-gray-800 rounded-lg p-1">
              <button
                onClick={() => setActiveTab("ohlcv")}
                className={`px-4 py-2 text-sm font-medium rounded-md transition-colors duration-200 ${
                  activeTab === "ohlcv"
                    ? "bg-primary-600 text-white"
                    : "text-gray-400 hover:text-gray-100"
                }`}
              >
                OHLCV
              </button>
              <button
                onClick={() => setActiveTab("funding")}
                className={`px-4 py-2 text-sm font-medium rounded-md transition-colors duration-200 ${
                  activeTab === "funding"
                    ? "bg-primary-600 text-white"
                    : "text-gray-400 hover:text-gray-100"
                }`}
              >
                FR
              </button>
              <button
                onClick={() => setActiveTab("openinterest")}
                className={`px-4 py-2 text-sm font-medium rounded-md transition-colors duration-200 ${
                  activeTab === "openinterest"
                    ? "bg-primary-600 text-white"
                    : "text-gray-400 hover:text-gray-100"
                }`}
              >
                OI
              </button>
              <button
                onClick={() => setActiveTab("feargreed")}
                className={`px-4 py-2 text-sm font-medium rounded-md transition-colors duration-200 ${
                  activeTab === "feargreed"
                    ? "bg-primary-600 text-white"
                    : "text-gray-400 hover:text-gray-100"
                }`}
              >
                F&G
              </button>
              <button
                onClick={() => setActiveTab("externalmarket")}
                className={`px-4 py-2 text-sm font-medium rounded-md transition-colors duration-200 ${
                  activeTab === "externalmarket"
                    ? "bg-primary-600 text-white"
                    : "text-gray-400 hover:text-gray-100"
                }`}
              >
                EM
              </button>
            </div>
          </div>

          {/* データ情報バッジ */}
          <div className="flex items-center gap-2">
            {activeTab === "ohlcv" && ohlcvData.length > 0 && !loading && (
              <>
                <span className="badge-primary">{ohlcvData.length}件</span>
                <span className="badge-success">
                  最新: ${ohlcvData[ohlcvData.length - 1]?.close.toFixed(2)}
                </span>
              </>
            )}
            {activeTab === "funding" &&
              fundingRateData.length > 0 &&
              !fundingLoading && (
                <>
                  <span className="badge-primary">
                    {fundingRateData.length}件
                  </span>
                  <span className="badge-info">
                    最新レート:{" "}
                    {(fundingRateData[0]?.funding_rate * 100).toFixed(4)}%
                  </span>
                </>
              )}
            {activeTab === "openinterest" &&
              openInterestData.length > 0 &&
              !openInterestLoading && (
                <>
                  <span className="badge-primary">
                    {openInterestData.length}件
                  </span>
                  <span className="badge-warning">
                    最新OI:{" "}
                    {new Intl.NumberFormat("en-US", {
                      style: "currency",
                      currency: "USD",
                      notation: "compact",
                      maximumFractionDigits: 1,
                    }).format(openInterestData[0]?.open_interest_value || 0)}
                  </span>
                </>
              )}
            {activeTab === "feargreed" &&
              fearGreedData.length > 0 &&
              !fearGreedLoading && (
                <>
                  <span className="badge-primary">
                    {fearGreedData.length}件
                  </span>
                  <span className="badge-info">
                    最新値: {fearGreedData[0]?.value}/100
                  </span>
                  <span
                    className={`badge-${
                      fearGreedData[0]?.value <= 20
                        ? "error"
                        : fearGreedData[0]?.value <= 40
                        ? "warning"
                        : fearGreedData[0]?.value <= 60
                        ? "info"
                        : fearGreedData[0]?.value <= 80
                        ? "success"
                        : "primary"
                    }`}
                  >
                    {fearGreedData[0]?.value_classification}
                  </span>
                </>
              )}
            {activeTab === "externalmarket" &&
              externalMarketData.length > 0 &&
              !externalMarketLoading && (
                <>
                  <span className="badge-primary">
                    {externalMarketData.length}件
                  </span>
                  <span className="badge-info">
                    {new Set(externalMarketData.map((d) => d.symbol)).size}
                    シンボル
                  </span>
                  {externalMarketData[0] && (
                    <span className="badge-success">
                      最新:{" "}
                      {new Date(
                        externalMarketData[0].data_timestamp
                      ).toLocaleDateString("ja-JP")}
                    </span>
                  )}
                </>
              )}
          </div>
        </div>

        {/* タブコンテンツ */}
        <div className="relative">
          {activeTab === "ohlcv" && (
            <DataTable
              data={ohlcvData}
              columns={tableConfigs.ohlcv.columns}
              title={tableConfigs.ohlcv.title}
              loading={loading}
              error={error}
              pageSize={tableConfigs.ohlcv.pageSize}
              enableExport={tableConfigs.ohlcv.enableExport}
              enableSearch={tableConfigs.ohlcv.enableSearch}
              searchKeys={tableConfigs.ohlcv.searchKeys}
            />
          )}
          {activeTab === "funding" && (
            <DataTable
              data={fundingRateData}
              columns={tableConfigs.funding.columns}
              title={tableConfigs.funding.title}
              loading={fundingLoading}
              error={fundingError}
              pageSize={tableConfigs.funding.pageSize}
              enableExport={tableConfigs.funding.enableExport}
              enableSearch={tableConfigs.funding.enableSearch}
              searchKeys={tableConfigs.funding.searchKeys}
            />
          )}
          {activeTab === "openinterest" && (
            <DataTable
              data={openInterestData}
              columns={tableConfigs.openinterest.columns}
              title={tableConfigs.openinterest.title}
              loading={openInterestLoading}
              error={openInterestError}
              pageSize={tableConfigs.openinterest.pageSize}
              enableExport={tableConfigs.openinterest.enableExport}
              enableSearch={tableConfigs.openinterest.enableSearch}
              searchKeys={tableConfigs.openinterest.searchKeys}
            />
          )}
          {activeTab === "feargreed" && (
            <>
              {!fearGreedLoading &&
              !fearGreedError &&
              (!fearGreedData || fearGreedData.length === 0) ? (
                <div className="enterprise-card">
                  <div className="p-6">
                    <div className="text-center text-secondary-600 dark:text-secondary-400">
                      <p className="text-lg font-medium mb-2">
                        📊 データがありません
                      </p>
                      <p className="text-sm">
                        Fear & Greed Index データを収集してください
                      </p>
                    </div>
                  </div>
                </div>
              ) : (
                <DataTable
                  data={fearGreedData}
                  columns={tableConfigs.feargreed.columns}
                  title={tableConfigs.feargreed.title}
                  loading={fearGreedLoading}
                  error={fearGreedError || ""}
                  enableExport={tableConfigs.feargreed.enableExport}
                  enableSearch={tableConfigs.feargreed.enableSearch}
                  searchKeys={tableConfigs.feargreed.searchKeys}
                  className="mb-4"
                />
              )}
            </>
          )}
          {activeTab === "externalmarket" && (
            <>
              {!externalMarketLoading &&
              !externalMarketError &&
              (!externalMarketData || externalMarketData.length === 0) ? (
                <div className="enterprise-card">
                  <div className="p-6">
                    <div className="text-center text-secondary-600 dark:text-secondary-400">
                      <p className="text-lg font-medium mb-2">
                        📊 データがありません
                      </p>
                      <p className="text-sm">
                        外部市場データを収集してください
                      </p>
                      <p className="mt-1 text-xs text-gray-400 dark:text-gray-500">
                        SP500、NASDAQ、DXY、VIXの日足データが取得されます。
                      </p>
                    </div>
                  </div>
                </div>
              ) : (
                <DataTable
                  data={enrichedExternalMarketData}
                  columns={tableConfigs.externalmarket.columns}
                  title={tableConfigs.externalmarket.title}
                  loading={externalMarketLoading}
                  error={externalMarketError || ""}
                  enableExport={tableConfigs.externalmarket.enableExport}
                  enableSearch={tableConfigs.externalmarket.enableSearch}
                  searchKeys={tableConfigs.externalmarket.searchKeys}
                />
              )}
            </>
          )}
        </div>
      </div>
    </div>
  );
};

export default DataTableContainer;
