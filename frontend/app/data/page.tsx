/**
 * データページコンポーネント
 *
 * OHLCVデータとFRデータを表形式で表示するページです。
 * リアルタイムでデータを取得・表示します。
 *
 */

"use client";

import React, { useState, useCallback } from "react";
import DataHeader from "@/components/data/DataHeader";
import DataControls from "@/components/data/DataControls";
import DataTableContainer from "@/components/data/DataTableContainer";
import { useOhlcvData } from "@/hooks/useOhlcvData";
import { useFundingRateData } from "@/hooks/useFundingRateData";
import { useOpenInterestData } from "@/hooks/useOpenInterestData";
import { useFearGreedData } from "@/hooks/useFearGreedData";
import { TimeFrame } from "@/types/market-data";
import { SUPPORTED_TRADING_PAIRS } from "@/constants";

// 新規フック
import { useDataStatus } from "@/hooks/useDataStatus";
import { useMessages, DefaultMessageDurations } from "@/hooks/useMessages";
import { useIncrementalUpdateHandler } from "@/hooks/useIncrementalUpdateHandler";
import { useCollectionMessageHandlers } from "@/hooks/useCollectionMessageHandlers";

/**
 * データページコンポーネント
 */
const DataPage: React.FC = () => {
  // 状態管理
  const [selectedSymbol, setSelectedSymbol] = useState<string>("BTC/USDT:USDT");
  const [selectedTimeFrame, setSelectedTimeFrame] = useState<TimeFrame>("1h");
  const [activeTab, setActiveTab] = useState<
    "ohlcv" | "funding" | "openinterest" | "feargreed"
  >("ohlcv");

  // メッセージとキー定義
  const MESSAGE_DURATION = DefaultMessageDurations;
  const MESSAGE_KEYS = {
    BULK_COLLECTION: "bulkCollection",
    FUNDING_RATE_COLLECTION: "fundingRateCollection",
    OPEN_INTEREST_COLLECTION: "openInterestCollection",
    FEAR_GREED_COLLECTION: "fearGreedCollection",
    ALL_DATA_COLLECTION: "allDataCollection",
    INCREMENTAL_UPDATE: "incrementalUpdate",
    EXTERNAL_MARKET_COLLECTION: "externalMarketCollection",
  } as const;

  const { messages, setMessage } = useMessages({
    defaultDurations: MESSAGE_DURATION,
  });

  // データステータス
  const { dataStatus, dataStatusLoading, fetchDataStatus } = useDataStatus();

  // データ取得
  const symbols = SUPPORTED_TRADING_PAIRS;
  const {
    data: ohlcvData,
    loading: ohlcvLoading,
    error: ohlcvError,
    refetch: fetchOHLCVData,
  } = useOhlcvData(selectedSymbol, selectedTimeFrame);

  const {
    data: fundingRateData,
    loading: fundingLoading,
    error: fundingError,
    refetch: fetchFundingRateData,
  } = useFundingRateData(selectedSymbol);

  const {
    data: openInterestData,
    loading: openInterestLoading,
    error: openInterestError,
    refetch: fetchOpenInterestData,
  } = useOpenInterestData(selectedSymbol);

  const {
    data: fearGreedData,
    loading: fearGreedLoading,
    error: fearGreedError,
    fetchLatestData: fetchFearGreedData,
  } = useFearGreedData();

  // リフレッシュ
  const handleRefresh = useCallback(() => {
    if (activeTab === "ohlcv") {
      fetchOHLCVData();
    } else if (activeTab === "funding") {
      fetchFundingRateData();
    } else if (activeTab === "openinterest") {
      fetchOpenInterestData();
    }
  }, [activeTab, fetchOHLCVData, fetchFundingRateData, fetchOpenInterestData]);

  // 一括差分更新
  const {
    handleBulkIncrementalUpdate,
    bulkIncrementalUpdateLoading,
    bulkIncrementalUpdateError,
  } = useIncrementalUpdateHandler({
    setMessage,
    fetchOHLCVData,
    fetchDataStatus,
    MESSAGE_KEYS,
    MESSAGE_DURATION,
  });

  // 収集メッセージ/ハンドラ
  const {
    handleBulkCollectionStart,
    handleBulkCollectionError,
    handleFundingRateCollectionStart,
    handleFundingRateCollectionError,
    handleOpenInterestCollectionStart,
    handleOpenInterestCollectionError,
    handleFearGreedCollectionStart,
    handleFearGreedCollectionError,
    handleAllDataCollectionStart,
    handleAllDataCollectionError,
  } = useCollectionMessageHandlers({
    setMessage,
    fetchFearGreedData,
    fetchDataStatus,
    fetchOHLCVData,
    fetchFundingRateData,
    fetchOpenInterestData,
    MESSAGE_KEYS,
    MESSAGE_DURATION,
  });

  return (
    <div className="min-h-screen  from-gray-900 animate-fade-in">
      <DataHeader
        loading={
          ohlcvLoading ||
          fundingLoading ||
          openInterestLoading ||
          dataStatusLoading
        }
        error={
          ohlcvError ||
          fundingError ||
          openInterestError ||
          bulkIncrementalUpdateError ||
          ""
        }
        updating={false}
        bulkUpdating={bulkIncrementalUpdateLoading}
        handleRefresh={handleRefresh}
        handleBulkIncrementalUpdate={() =>
          handleBulkIncrementalUpdate(selectedSymbol, selectedTimeFrame)
        }
      />

      {/* メインコンテンツエリア */}
      <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8 py-8 space-y-8">
        {/* エラー表示 */}
        {(() => {
          const errors = [
            ohlcvError,
            fundingError,
            openInterestError,
            bulkIncrementalUpdateError,
          ].filter(Boolean);

          if (errors.length === 0) return null;

          return (
            <div className="enterprise-card border-error-200 dark:border-error-800 bg-error-50 dark:bg-error-900/20 animate-slide-down">
              <div className="p-4">
                <div className="flex items-center">
                  <svg
                    className="w-5 h-5 text-error-500 mr-3"
                    fill="none"
                    stroke="currentColor"
                    viewBox="0 0 24 24"
                  >
                    <path
                      strokeLinecap="round"
                      strokeLinejoin="round"
                      strokeWidth={2}
                      d="M12 8v4m0 4h.01M21 12a9 9 0 11-18 0 9 9 0 0118 0z"
                    />
                  </svg>
                  <h3 className="text-sm font-medium text-error-800 dark:text-error-200">
                    データ取得エラー
                  </h3>
                </div>
                <p className="mt-2 text-sm text-error-700 dark:text-error-300">
                  {errors[0] as string}
                </p>
              </div>
            </div>
          );
        })()}

        <DataControls
          dataStatus={dataStatus}
          symbols={symbols}
          selectedSymbol={selectedSymbol}
          handleSymbolChange={setSelectedSymbol}
          symbolsLoading={false}
          loading={ohlcvLoading || fundingLoading || openInterestLoading}
          selectedTimeFrame={selectedTimeFrame}
          handleTimeFrameChange={setSelectedTimeFrame}
          updating={bulkIncrementalUpdateLoading}
          handleAllDataCollectionStart={handleAllDataCollectionStart}
          handleAllDataCollectionError={handleAllDataCollectionError}
          handleBulkCollectionStart={handleBulkCollectionStart}
          handleBulkCollectionError={handleBulkCollectionError}
          handleFundingRateCollectionStart={handleFundingRateCollectionStart}
          handleFundingRateCollectionError={handleFundingRateCollectionError}
          handleOpenInterestCollectionStart={handleOpenInterestCollectionStart}
          handleOpenInterestCollectionError={handleOpenInterestCollectionError}
          handleFearGreedCollectionStart={handleFearGreedCollectionStart}
          handleFearGreedCollectionError={handleFearGreedCollectionError}
          bulkCollectionMessage={messages[MESSAGE_KEYS.BULK_COLLECTION] || ""}
          fundingRateCollectionMessage={
            messages[MESSAGE_KEYS.FUNDING_RATE_COLLECTION] || ""
          }
          openInterestCollectionMessage={
            messages[MESSAGE_KEYS.OPEN_INTEREST_COLLECTION] || ""
          }
          fearGreedCollectionMessage={
            messages[MESSAGE_KEYS.FEAR_GREED_COLLECTION] || ""
          }
          externalMarketCollectionMessage={
            messages[MESSAGE_KEYS.EXTERNAL_MARKET_COLLECTION] || ""
          }
          allDataCollectionMessage={
            messages[MESSAGE_KEYS.ALL_DATA_COLLECTION] || ""
          }
          incrementalUpdateMessage={
            messages[MESSAGE_KEYS.INCREMENTAL_UPDATE] || ""
          }
        />

        <DataTableContainer
          selectedSymbol={selectedSymbol}
          selectedTimeFrame={selectedTimeFrame}
          activeTab={activeTab}
          setActiveTab={setActiveTab}
          ohlcvData={ohlcvData}
          loading={ohlcvLoading}
          error={ohlcvError || ""}
          fundingRateData={fundingRateData}
          fundingLoading={fundingLoading}
          fundingError={fundingError || ""}
          openInterestData={openInterestData}
          openInterestLoading={openInterestLoading}
          openInterestError={openInterestError || ""}
          fearGreedData={fearGreedData}
          fearGreedLoading={fearGreedLoading}
          fearGreedError={fearGreedError || ""}
        />
      </div>
    </div>
  );
};

export default DataPage;
