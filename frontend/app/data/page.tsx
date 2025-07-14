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
import { useApiCall } from "@/hooks/useApiCall";
import {
  TimeFrame,
  TradingPair,
  BulkOHLCVCollectionResult,
  BulkFundingRateCollectionResult,
  FundingRateCollectionResult,
  OpenInterestCollectionResult,
  BulkOpenInterestCollectionResult,
  AllDataCollectionResult,
} from "@/types/strategy";
import { BACKEND_API_URL } from "@/constants";
import { useSymbols } from "@/hooks/useSymbols";

/**
 * データページコンポーネント
 */
const DataPage: React.FC = () => {
  // 状態管理
  const [selectedSymbol, setSelectedSymbol] = useState<string>("BTC/USDT:USDT");
  const [selectedTimeFrame, setSelectedTimeFrame] = useState<TimeFrame>("1h");
  const [activeTab, setActiveTab] = useState<
    "ohlcv" | "funding" | "openinterest"
  >("ohlcv");

  const [dataStatus, setDataStatus] = useState<any>(null);
  const [bulkCollectionMessage, setBulkCollectionMessage] =
    useState<string>("");
  const [fundingRateCollectionMessage, setFundingRateCollectionMessage] =
    useState<string>("");
  const [openInterestCollectionMessage, setOpenInterestCollectionMessage] =
    useState<string>("");
  const [allDataCollectionMessage, setAllDataCollectionMessage] =
    useState<string>("");
  const [incrementalUpdateMessage, setIncrementalUpdateMessage] =
    useState<string>("");

  // カスタムフックを使用してデータ取得
  const { symbols } = useSymbols();
  const { execute: updateIncrementalData, loading: incrementalUpdateLoading } =
    useApiCall();
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
   * 差分データ更新
   */
  const handleIncrementalUpdate = async () => {
    setIncrementalUpdateMessage("");

    await updateIncrementalData(
      `${BACKEND_API_URL}/api/data-collection/update?symbol=${selectedSymbol}&timeframe=${selectedTimeFrame}`,
      {
        method: "POST",
        onSuccess: async (result) => {
          // 成功メッセージを表示
          const savedCount = result.saved_count || 0;
          setIncrementalUpdateMessage(
            `✅ 差分更新完了！ ${selectedSymbol} ${selectedTimeFrame} - ${savedCount}件のデータを更新しました`
          );

          // 更新後に全てのデータを再取得
          await Promise.all([
            fetchOHLCVData(),
            fetchFundingRateData(),
            fetchOpenInterestData(),
          ]);

          // データ状況も更新
          fetchDataStatus();

          // 10秒後にメッセージをクリア
          setTimeout(() => setIncrementalUpdateMessage(""), 10000);
        },
        onError: (errorMessage) => {
          setIncrementalUpdateMessage(`❌ ${errorMessage}`);
          console.error("差分更新エラー:", errorMessage);
          // 10秒後にメッセージをクリア
          setTimeout(() => setIncrementalUpdateMessage(""), 10000);
        },
      }
    );
  };

  /**
   * データ収集状況を取得
   */
  const fetchDataStatus = useCallback(() => {
    const url = `${BACKEND_API_URL}/api/data-collection/status/${selectedSymbol}/${selectedTimeFrame}`;
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
  }, [selectedSymbol, selectedTimeFrame, fetchDataStatusApi]);

  /**
   * 一括OHLCVデータ収集開始時のコールバック
   */
  const handleBulkCollectionStart = (result: BulkOHLCVCollectionResult) => {
    setBulkCollectionMessage(
      `🚀 ${result.message} (${result.total_tasks}タスク)`
    );
    // データ状況を更新
    fetchDataStatus();
    // 10秒後にメッセージをクリア
    setTimeout(() => setBulkCollectionMessage(""), 10000);
  };

  /**
   * 一括OHLCVデータ収集エラー時のコールバック
   */
  const handleBulkCollectionError = (errorMessage: string) => {
    setBulkCollectionMessage(`❌ ${errorMessage}`);
    // 10秒後にメッセージをクリア
    setTimeout(() => setBulkCollectionMessage(""), 10000);
  };

  /**
   * FRデータ収集開始時のコールバック
   */
  const handleFundingRateCollectionStart = (
    result: BulkFundingRateCollectionResult | FundingRateCollectionResult
  ) => {
    if ("total_symbols" in result) {
      // BulkFundingRateCollectionResult
      const bulkResult = result as BulkFundingRateCollectionResult;
      setFundingRateCollectionMessage(
        `🚀 ${bulkResult.message} (${bulkResult.successful_symbols}/${bulkResult.total_symbols}シンボル成功)`
      );
    } else {
      // FundingRateCollectionResult
      const singleResult = result as FundingRateCollectionResult;
      setFundingRateCollectionMessage(
        `🚀 ${singleResult.symbol}のFRデータ収集完了 (${singleResult.saved_count}件保存)`
      );
    }
    // 10秒後にメッセージをクリア
    setTimeout(() => setFundingRateCollectionMessage(""), 10000);
  };

  /**
   * FRデータ収集エラー時のコールバック
   */
  const handleFundingRateCollectionError = (errorMessage: string) => {
    setFundingRateCollectionMessage(`❌ ${errorMessage}`);
    // 10秒後にメッセージをクリア
    setTimeout(() => setFundingRateCollectionMessage(""), 10000);
  };

  /**
   * OIデータ収集開始時のコールバック
   */
  const handleOpenInterestCollectionStart = (
    result: BulkOpenInterestCollectionResult | OpenInterestCollectionResult
  ) => {
    if ("total_symbols" in result) {
      // BulkOpenInterestCollectionResult
      const bulkResult = result as BulkOpenInterestCollectionResult;
      setOpenInterestCollectionMessage(
        `🚀 ${bulkResult.message} (${bulkResult.successful_symbols}/${bulkResult.total_symbols}シンボル成功)`
      );
    } else {
      // OpenInterestCollectionResult
      const singleResult = result as OpenInterestCollectionResult;
      setOpenInterestCollectionMessage(
        `🚀 ${singleResult.symbol}のOIデータ収集完了 (${singleResult.saved_count}件保存)`
      );
    }
    // 10秒後にメッセージをクリア
    setTimeout(() => setOpenInterestCollectionMessage(""), 10000);
  };

  /**
   * OIデータ収集エラー時のコールバック
   */
  const handleOpenInterestCollectionError = (errorMessage: string) => {
    setOpenInterestCollectionMessage(`❌ ${errorMessage}`);
    // 10秒後にメッセージをクリア
    setTimeout(() => setOpenInterestCollectionMessage(""), 10000);
  };

  /**
   * 全データ一括収集開始時のコールバック
   */
  const handleAllDataCollectionStart = (result: AllDataCollectionResult) => {
    if (result.ohlcv_result.status === "completed") {
      const ohlcvCount = result.ohlcv_result?.total_tasks || 0;
      const fundingCount = result.funding_rate_result?.total_saved_records || 0;
      const openInterestCount =
        result.open_interest_result?.total_saved_records || 0;

      setAllDataCollectionMessage(
        `🚀 全データ収集完了！ OHLCV:${ohlcvCount}タスク, FR:${fundingCount}件, OI:${openInterestCount}件, TI:自動計算済み`
      );
    } else {
      setAllDataCollectionMessage(
        `🔄 ${result.ohlcv_result.message} (実行中...)`
      );
    }

    // データ状況を更新
    fetchDataStatus();

    // 全データ収集完了後に全てのデータを再取得
    setTimeout(() => {
      fetchOHLCVData();
      fetchFundingRateData();
      fetchOpenInterestData();
    }, 3000);

    // 15秒後にメッセージをクリア
    setTimeout(() => setAllDataCollectionMessage(""), 15000);
  };

  /**
   * 全データ一括収集エラー時のコールバック
   */
  const handleAllDataCollectionError = (errorMessage: string) => {
    setAllDataCollectionMessage(`❌ ${errorMessage}`);
    // 15秒後にメッセージをクリア
    setTimeout(() => setAllDataCollectionMessage(""), 15000);
  };

  // 選択が変更されたときにデータステータスをフェッチ
  useEffect(() => {
    if (selectedSymbol && selectedTimeFrame) {
      fetchDataStatus();
    }
  }, [selectedSymbol, selectedTimeFrame, fetchDataStatus]);

  return (
    <div className="min-h-screen bg-secondary-50 dark:bg-secondary-950 animate-fade-in">
      <DataHeader
        loading={ohlcvLoading || fundingLoading || openInterestLoading}
        error={ohlcvError || fundingError || openInterestError || ""}
        updating={incrementalUpdateLoading}
        handleRefresh={handleRefresh}
        handleIncrementalUpdate={handleIncrementalUpdate}
      />

      {/* メインコンテンツエリア */}
      <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8 py-8 space-y-8">
        {/* エラー表示 */}
        {(ohlcvError || fundingError || openInterestError) && (
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
                {ohlcvError || fundingError || openInterestError}
              </p>
            </div>
          </div>
        )}

        <DataControls
          dataStatus={dataStatus}
          symbols={symbols}
          selectedSymbol={selectedSymbol}
          handleSymbolChange={handleSymbolChange}
          symbolsLoading={false}
          loading={ohlcvLoading || fundingLoading || openInterestLoading}
          selectedTimeFrame={selectedTimeFrame}
          handleTimeFrameChange={handleTimeFrameChange}
          updating={incrementalUpdateLoading}
          handleAllDataCollectionStart={handleAllDataCollectionStart}
          handleAllDataCollectionError={handleAllDataCollectionError}
          handleBulkCollectionStart={handleBulkCollectionStart}
          handleBulkCollectionError={handleBulkCollectionError}
          handleFundingRateCollectionStart={handleFundingRateCollectionStart}
          handleFundingRateCollectionError={handleFundingRateCollectionError}
          handleOpenInterestCollectionStart={handleOpenInterestCollectionStart}
          handleOpenInterestCollectionError={handleOpenInterestCollectionError}
          bulkCollectionMessage={bulkCollectionMessage}
          fundingRateCollectionMessage={fundingRateCollectionMessage}
          openInterestCollectionMessage={openInterestCollectionMessage}
          allDataCollectionMessage={allDataCollectionMessage}
          incrementalUpdateMessage={incrementalUpdateMessage}
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
        />
      </div>
    </div>
  );
};

export default DataPage;
