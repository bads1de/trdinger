import React, { useMemo } from "react";
import DataTable from "@/components/table/DataTable";
import {
  ohlcvColumns,
  fundingRateColumns,
  openInterestColumns,
  fearGreedColumns,
} from "@/components/common/tableColumns";
import { PriceData, TimeFrame } from "@/types/market-data";
import { FundingRateData } from "@/types/funding-rate";
import { OpenInterestData } from "@/types/open-interest";

interface DataTableContainerProps {
  selectedSymbol: string;
  selectedTimeFrame: TimeFrame;
  activeTab: "ohlcv" | "funding" | "openinterest" | "feargreed";

  setActiveTab: (
    tab: "ohlcv" | "funding" | "openinterest" | "feargreed"
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
}) => {
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
        </div>
      </div>
    </div>
  );
};

export default DataTableContainer;
