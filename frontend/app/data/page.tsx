/**
 * データページコンポーネント
 *
 * OHLCVデータとFRデータを表形式で表示するページです。
 * リアルタイムでデータを取得・表示します。
 *
 */

"use client";

import React, { useState, useEffect, useCallback } from "react";
import DataHeader from "@/components/data/DataHeader";
import DataControls from "@/components/data/DataControls";
import DataTableContainer from "@/components/data/DataTableContainer";
import { useOhlcvData } from "@/hooks/useOhlcvData";
import { useFundingRateData } from "@/hooks/useFundingRateData";
import { useOpenInterestData } from "@/hooks/useOpenInterestData";
import {
  useFearGreedData,
  FearGreedCollectionResult,
} from "@/hooks/useFearGreedData";
import { useBulkIncrementalUpdate } from "@/hooks/useBulkIncrementalUpdate";
import { useApiCall } from "@/hooks/useApiCall";
import { TimeFrame } from "@/types/market-data";
import {
  BulkOHLCVCollectionResult,
  AllDataCollectionResult,
} from "@/types/data-collection";
import {
  BulkFundingRateCollectionResult,
  FundingRateCollectionResult,
} from "@/types/funding-rate";
import {
  OpenInterestCollectionResult,
  BulkOpenInterestCollectionResult,
} from "@/types/open-interest";
import { BACKEND_API_URL, SUPPORTED_TRADING_PAIRS } from "@/constants";

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

  const [dataStatus, setDataStatus] = useState<any>(null);
  const [messages, setMessages] = useState<Record<string, string>>({});

  // 定数定義
  const MESSAGE_DURATION = {
    SHORT: 10000,
    MEDIUM: 15000,
    LONG: 20000,
  } as const;

  const MESSAGE_KEYS = {
    BULK_COLLECTION: "bulkCollection",
    FUNDING_RATE_COLLECTION: "fundingRateCollection",
    OPEN_INTEREST_COLLECTION: "openInterestCollection",
    FEAR_GREED_COLLECTION: "fearGreedCollection",
    ALL_DATA_COLLECTION: "allDataCollection",
    INCREMENTAL_UPDATE: "incrementalUpdate",
    EXTERNAL_MARKET_COLLECTION: "externalMarketCollection",
  } as const;

  type MessageKey = (typeof MESSAGE_KEYS)[keyof typeof MESSAGE_KEYS];

  const setMessage = useCallback(
    (
      key: MessageKey,
      message: string,
      duration: number = MESSAGE_DURATION.SHORT
    ) => {
      setMessages((prev) => ({ ...prev, [key]: message }));
      if (duration > 0) {
        setTimeout(() => {
          setMessages((prev) => {
            const newMessages = { ...prev };
            delete newMessages[key];
            return newMessages;
          });
        }, duration);
      }
    },
    []
  );

  // カスタムフックを使用してデータ取得
  const symbols = SUPPORTED_TRADING_PAIRS;
  const {
    bulkUpdate: updateBulkIncrementalData,
    loading: bulkIncrementalUpdateLoading,
    error: bulkIncrementalUpdateError,
  } = useBulkIncrementalUpdate();
  const { execute: fetchDataStatusApi, loading: dataStatusLoading } =
    useApiCall();

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
    status: fearGreedStatus,
    fetchLatestData: fetchFearGreedData,
  } = useFearGreedData();

  /**
   * 通貨ペア変更ハンドラ
   */
  const handleSymbolChange = (symbol: string) => {
    setSelectedSymbol(symbol);
  };

  /**
   * 時間軸変更ハンドラ
   */
  const handleTimeFrameChange = (timeFrame: TimeFrame) => {
    setSelectedTimeFrame(timeFrame);
  };

  /**
   * データ更新ハンドラ
   */
  const handleRefresh = () => {
    if (activeTab === "ohlcv") {
      fetchOHLCVData();
    } else if (activeTab === "funding") {
      fetchFundingRateData();
    } else if (activeTab === "openinterest") {
      fetchOpenInterestData();
    }
  };

  /**
   * 一括差分データ更新
   */
  const handleBulkIncrementalUpdate = async () => {
    setMessage(MESSAGE_KEYS.INCREMENTAL_UPDATE, "");
    await updateBulkIncrementalData(selectedSymbol, selectedTimeFrame, {
      onSuccess: async (result) => {
        const totalSavedCount = result.data.total_saved_count || 0;
        const ohlcvCount = result.data.data.ohlcv.saved_count || 0;
        const frCount = result.data.data.funding_rate.saved_count || 0;
        const oiCount = result.data.data.open_interest.saved_count || 0;

        // 時間足別の詳細情報を取得
        let timeframeDetails = "";
        if (result.data.data.ohlcv.timeframe_results) {
          const tfResults = Object.entries(
            result.data.data.ohlcv.timeframe_results
          )
            .map(([tf, res]) => `${tf}:${res.saved_count}`)
            .join(", ");
          timeframeDetails = ` [${tfResults}]`;
        }

        setMessage(
          MESSAGE_KEYS.INCREMENTAL_UPDATE,
          `✅ 一括差分更新完了！ ${selectedSymbol} - ` +
            `総計${totalSavedCount}件 (OHLCV:${ohlcvCount}${timeframeDetails}, FR:${frCount}, OI:${oiCount})`,
          MESSAGE_DURATION.MEDIUM
        );

        // 現在選択されている時間足のデータを再取得
        await fetchOHLCVData();
        fetchDataStatus();
      },
      onError: (errorMessage) => {
        setMessage(
          MESSAGE_KEYS.INCREMENTAL_UPDATE,
          `❌ ${errorMessage}`,
          MESSAGE_DURATION.SHORT
        );
        console.error("一括差分更新エラー:", errorMessage);
      },
    });
  };

  /**
   * データ収集状況を取得（詳細版）
   */
  const fetchDataStatus = useCallback(() => {
    const url = `${BACKEND_API_URL}/api/data-reset/status`;
    fetchDataStatusApi(url, {
      onSuccess: (result) => {
        if (result) {
          setDataStatus(result);
        }
      },
      onError: (err) => {
        console.error("データ状況取得エラー:", err);
      },
    });
  }, [fetchDataStatusApi]);

  // 汎用メッセージハンドラ
  const createMessageHandler = (
    key: MessageKey,
    duration: number = MESSAGE_DURATION.SHORT
  ) => ({
    onStart: (message: string) => setMessage(key, message, duration),
    onError: (errorMessage: string) =>
      setMessage(key, `❌ ${errorMessage}`, duration),
  });

  // データ収集メッセージ生成関数
  const generateCollectionMessage = (type: string, result: any): string => {
    switch (type) {
      case "bulk":
        return `🚀 ${result.message} (${result.total_tasks}タスク)`;
      case "funding":
        if ("total_symbols" in result) {
          return `🚀 ${result.message} (${result.successful_symbols}/${result.total_symbols}シンボル成功)`;
        }
        return `🚀 ${result.symbol}のFRデータ収集完了 (${result.saved_count}件保存)`;
      case "openinterest":
        if ("total_symbols" in result) {
          return `🚀 ${result.message} (${result.successful_symbols}/${result.total_symbols}シンボル成功)`;
        }
        return `🚀 ${result.symbol}のOIデータ収集完了 (${result.saved_count}件保存)`;
      case "feargreed":
        return result.success
          ? `🚀 Fear & Greed Index収集完了 (取得:${result.fetched_count}件, 挿入:${result.inserted_count}件)`
          : `❌ ${result.message}`;
      case "alldata":
        if (result.ohlcv_result?.status === "completed") {
          const ohlcvCount = result.ohlcv_result?.total_tasks || 0;
          const fundingCount =
            result.funding_rate_result?.total_saved_records || 0;
          const openInterestCount =
            result.open_interest_result?.total_saved_records || 0;
          return `🚀 全データ収集完了！ OHLCV:${ohlcvCount}タスク, FR:${fundingCount}件, OI:${openInterestCount}件, TI:自動計算済み`;
        }
        return `🔄 ${result.ohlcv_result?.message || "処理中..."} (実行中...)`;
      default:
        return `🚀 ${result.message || "処理完了"}`;
    }
  };

  // 各種ハンドラを簡潔に定義
  const handleBulkCollectionStart = (result: BulkOHLCVCollectionResult) => {
    setMessage(
      MESSAGE_KEYS.BULK_COLLECTION,
      generateCollectionMessage("bulk", result)
    );
    fetchDataStatus();
  };

  const handleBulkCollectionError = (errorMessage: string) => {
    createMessageHandler(MESSAGE_KEYS.BULK_COLLECTION).onError(errorMessage);
  };

  const handleFundingRateCollectionStart = (
    result: BulkFundingRateCollectionResult | FundingRateCollectionResult
  ) => {
    setMessage(
      MESSAGE_KEYS.FUNDING_RATE_COLLECTION,
      generateCollectionMessage("funding", result)
    );
  };

  const handleFundingRateCollectionError = (errorMessage: string) => {
    createMessageHandler(MESSAGE_KEYS.FUNDING_RATE_COLLECTION).onError(
      errorMessage
    );
  };

  const handleOpenInterestCollectionStart = (
    result: BulkOpenInterestCollectionResult | OpenInterestCollectionResult
  ) => {
    setMessage(
      MESSAGE_KEYS.OPEN_INTEREST_COLLECTION,
      generateCollectionMessage("openinterest", result)
    );
  };

  const handleOpenInterestCollectionError = (errorMessage: string) => {
    createMessageHandler(MESSAGE_KEYS.OPEN_INTEREST_COLLECTION).onError(
      errorMessage
    );
  };

  const handleFearGreedCollectionStart = (
    result: FearGreedCollectionResult
  ) => {
    setMessage(
      MESSAGE_KEYS.FEAR_GREED_COLLECTION,
      generateCollectionMessage("feargreed", result)
    );
    if (result.success) {
      fetchFearGreedData();
    }
    fetchDataStatus();
  };

  const handleFearGreedCollectionError = (errorMessage: string) => {
    createMessageHandler(MESSAGE_KEYS.FEAR_GREED_COLLECTION).onError(
      errorMessage
    );
  };

  const handleAllDataCollectionStart = (result: AllDataCollectionResult) => {
    setMessage(
      MESSAGE_KEYS.ALL_DATA_COLLECTION,
      generateCollectionMessage("alldata", result),
      MESSAGE_DURATION.MEDIUM
    );
    fetchDataStatus();

    setTimeout(() => {
      fetchOHLCVData();
      fetchFundingRateData();
      fetchOpenInterestData();
    }, 3000);
  };

  const handleAllDataCollectionError = (errorMessage: string) => {
    createMessageHandler(
      MESSAGE_KEYS.ALL_DATA_COLLECTION,
      MESSAGE_DURATION.MEDIUM
    ).onError(errorMessage);
  };

  // コンポーネント初期化時にデータステータスをフェッチ
  useEffect(() => {
    fetchDataStatus();
  }, [fetchDataStatus]);

  return (
    <div className="min-h-screen  from-gray-900 animate-fade-in">
      <DataHeader
        loading={ohlcvLoading || fundingLoading || openInterestLoading}
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
        handleBulkIncrementalUpdate={handleBulkIncrementalUpdate}
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
                  {errors[0]}
                </p>
              </div>
            </div>
          );
        })()}

        <DataControls
          dataStatus={dataStatus}
          symbols={symbols}
          selectedSymbol={selectedSymbol}
          handleSymbolChange={handleSymbolChange}
          symbolsLoading={false}
          loading={ohlcvLoading || fundingLoading || openInterestLoading}
          selectedTimeFrame={selectedTimeFrame}
          handleTimeFrameChange={handleTimeFrameChange}
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
