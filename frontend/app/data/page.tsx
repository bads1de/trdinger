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
    "ohlcv" | "funding" | "openinterest" | "feargreed"
  >("ohlcv");

  const [dataStatus, setDataStatus] = useState<any>(null);
  const [bulkCollectionMessage, setBulkCollectionMessage] =
    useState<string>("");
  const [fundingRateCollectionMessage, setFundingRateCollectionMessage] =
    useState<string>("");
  const [openInterestCollectionMessage, setOpenInterestCollectionMessage] =
    useState<string>("");
  const [fearGreedCollectionMessage, setFearGreedCollectionMessage] =
    useState<string>("");
  const [allDataCollectionMessage, setAllDataCollectionMessage] =
    useState<string>("");
  const [incrementalUpdateMessage, setIncrementalUpdateMessage] =
    useState<string>("");

  // カスタムフックを使用してデータ取得
  const { symbols } = useSymbols();
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
    setIncrementalUpdateMessage("");
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
        } else {
          console.warn("時間足別結果が見つかりません");
        }

        setIncrementalUpdateMessage(
          `✅ 一括差分更新完了！ ${selectedSymbol} - ` +
            `総計${totalSavedCount}件 (OHLCV:${ohlcvCount}${timeframeDetails}, FR:${frCount}, OI:${oiCount})`
        );

        // 現在選択されている時間足のデータを再取得
        await fetchOHLCVData();
        fetchDataStatus();
        setTimeout(() => setIncrementalUpdateMessage(""), 15000);
      },
      onError: (errorMessage) => {
        setIncrementalUpdateMessage(`❌ ${errorMessage}`);
        console.error("一括差分更新エラー:", errorMessage);
        setTimeout(() => setIncrementalUpdateMessage(""), 10000);
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
   * Fear & Greed Index データ収集開始時のコールバック
   */
  const handleFearGreedCollectionStart = (
    result: FearGreedCollectionResult
  ) => {
    if (result.success) {
      setFearGreedCollectionMessage(
        `🚀 Fear & Greed Index収集完了 (取得:${result.fetched_count}件, 挿入:${result.inserted_count}件)`
      );
    } else {
      setFearGreedCollectionMessage(`❌ ${result.message}`);
    }
    // データ状況を更新
    fetchDataStatus();
    // 10秒後にメッセージをクリア
    setTimeout(() => setFearGreedCollectionMessage(""), 10000);
  };

  /**
   * Fear & Greed Index データ収集エラー時のコールバック
   */
  const handleFearGreedCollectionError = (errorMessage: string) => {
    setFearGreedCollectionMessage(`❌ ${errorMessage}`);
    // 10秒後にメッセージをクリア
    setTimeout(() => setFearGreedCollectionMessage(""), 10000);
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
        {(ohlcvError ||
          fundingError ||
          openInterestError ||
          bulkIncrementalUpdateError) && (
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
                {ohlcvError ||
                  fundingError ||
                  openInterestError ||
                  bulkIncrementalUpdateError}
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
          bulkCollectionMessage={bulkCollectionMessage}
          fundingRateCollectionMessage={fundingRateCollectionMessage}
          openInterestCollectionMessage={openInterestCollectionMessage}
          fearGreedCollectionMessage={fearGreedCollectionMessage}
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
          fearGreedData={fearGreedData}
          fearGreedLoading={fearGreedLoading}
          fearGreedError={fearGreedError || ""}
        />
      </div>
    </div>
  );
};

export default DataPage;
