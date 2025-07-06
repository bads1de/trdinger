import React from "react";
import OHLCVDataTable from "@/components/table/OHLCVDataTable";
import FundingRateDataTable from "@/components/table/FundingRateDataTable";
import OpenInterestDataTable from "@/components/table/OpenInterestDataTable";
import {
  PriceData,
  FundingRateData,
  OpenInterestData,
  TimeFrame,
} from "@/types/strategy";

interface DataTableContainerProps {
  selectedSymbol: string;
  selectedTimeFrame: TimeFrame;
  activeTab: "ohlcv" | "funding" | "openinterest";
  setActiveTab: (tab: "ohlcv" | "funding" | "openinterest") => void;
  ohlcvData: PriceData[];
  loading: boolean;
  error: string;
  fundingRateData: FundingRateData[];
  fundingLoading: boolean;
  fundingError: string;
  openInterestData: OpenInterestData[];
  openInterestLoading: boolean;
  openInterestError: string;
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
}) => {
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
          </div>
        </div>

        {/* タブコンテンツ */}
        <div className="relative">
          {activeTab === "ohlcv" && (
            <OHLCVDataTable
              data={ohlcvData}
              symbol={selectedSymbol}
              timeframe={selectedTimeFrame}
              loading={loading}
              error={error}
            />
          )}
          {activeTab === "funding" && (
            <FundingRateDataTable
              data={fundingRateData}
              loading={fundingLoading}
              error={fundingError}
            />
          )}
          {activeTab === "openinterest" && (
            <OpenInterestDataTable
              data={openInterestData}
              loading={openInterestLoading}
              error={openInterestError}
            />
          )}
        </div>
      </div>
    </div>
  );
};

export default DataTableContainer;
