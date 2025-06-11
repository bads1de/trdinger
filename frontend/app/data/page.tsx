/**
 * データページコンポーネント
 *
 * OHLCVデータとFRデータを表形式で表示するページです。
 * リアルタイムでデータを取得・表示します。
 *
 * @author Trdinger Development Team
 * @version 2.0.0
 */

"use client";

import React, { useState, useEffect } from "react";
import DataHeader from "./components/DataHeader";
import DataControls from "./components/DataControls";

import DataTableContainer from "./components/DataTableContainer";

import {
  PriceData,
  FundingRateData,
  OpenInterestData,
  TimeFrame,
  TradingPair,
  OHLCVResponse,
  FundingRateResponse,
  OpenInterestResponse,
  BulkOHLCVCollectionResult,
  BulkFundingRateCollectionResult,
  FundingRateCollectionResult,
  OpenInterestCollectionResult,
  BulkOpenInterestCollectionResult,
  AllDataCollectionResult,
} from "@/types/strategy";
import { BACKEND_API_URL } from "@/constants";

/**
 * データページコンポーネント
 */
const DataPage: React.FC = () => {
  // 状態管理
  const [symbols, setSymbols] = useState<TradingPair[]>([]);
  const [selectedSymbol, setSelectedSymbol] = useState<string>("BTC/USDT");
  const [selectedTimeFrame, setSelectedTimeFrame] = useState<TimeFrame>("1d");
  const [ohlcvData, setOhlcvData] = useState<PriceData[]>([]);
  const [fundingRateData, setFundingRateData] = useState<FundingRateData[]>([]);
  const [openInterestData, setOpenInterestData] = useState<OpenInterestData[]>(
    []
  );
  const [activeTab, setActiveTab] = useState<
    "ohlcv" | "funding" | "openinterest"
  >("ohlcv");
  const [loading, setLoading] = useState<boolean>(false);
  const [fundingLoading, setFundingLoading] = useState<boolean>(false);
  const [openInterestLoading, setOpenInterestLoading] =
    useState<boolean>(false);
  const [error, setError] = useState<string>("");
  const [fundingError, setFundingError] = useState<string>("");
  const [openInterestError, setOpenInterestError] = useState<string>("");
  const [symbolsLoading, setSymbolsLoading] = useState<boolean>(true);
  const [updating, setUpdating] = useState<boolean>(false);
  const [dataStatus, setDataStatus] = useState<any>(null);
  const [bulkCollectionMessage, setBulkCollectionMessage] =
    useState<string>("");
  const [fundingRateCollectionMessage, setFundingRateCollectionMessage] =
    useState<string>("");
  const [openInterestCollectionMessage, setOpenInterestCollectionMessage] =
    useState<string>("");
  const [allDataCollectionMessage, setAllDataCollectionMessage] =
    useState<string>("");

  /**
   * 通貨ペア一覧を取得
   */
  const fetchSymbols = async () => {
    try {
      setSymbolsLoading(true);
      const response = await fetch("/api/data/symbols");
      const result = await response.json();

      if (result.success) {
        setSymbols(result.data);
      } else {
        setError("通貨ペア一覧の取得に失敗しました");
      }
    } catch (err) {
      setError("通貨ペア一覧の取得中にエラーが発生しました");
      console.error("通貨ペア取得エラー:", err);
    } finally {
      setSymbolsLoading(false);
    }
  };

  /**
   * OHLCVデータを取得
   */
  const fetchOHLCVData = async () => {
    try {
      setLoading(true);
      setError("");

      const params = new URLSearchParams({
        symbol: selectedSymbol,
        timeframe: selectedTimeFrame,
        limit: "100",
      });

      const response = await fetch(`/api/data/candlesticks?${params}`);
      const result: OHLCVResponse = await response.json();

      if (result.success) {
        setOhlcvData(result.data.ohlcv);
      } else {
        setError(result.message || "データの取得に失敗しました");
      }
    } catch (err) {
      setError("データの取得中にエラーが発生しました");
      console.error("OHLCVデータ取得エラー:", err);
    } finally {
      setLoading(false);
    }
  };

  /**
   * FRデータを取得
   */
  const fetchFundingRateData = async () => {
    try {
      setFundingLoading(true);
      setFundingError("");

      const params = new URLSearchParams({
        symbol: selectedSymbol,
        limit: "100",
      });

      const response = await fetch(`/api/data/funding-rates?${params}`);
      const result: FundingRateResponse = await response.json();

      if (result.success) {
        setFundingRateData(result.data.funding_rates);
      } else {
        setFundingError(result.message || "FRデータの取得に失敗しました");
      }
    } catch (err) {
      setFundingError("FRデータの取得中にエラーが発生しました");
      console.error("FRデータ取得エラー:", err);
    } finally {
      setFundingLoading(false);
    }
  };

  /**
   * OIデータを取得
   */
  const fetchOpenInterestData = async () => {
    try {
      setOpenInterestLoading(true);
      setOpenInterestError("");

      const params = new URLSearchParams({
        symbol: selectedSymbol,
        limit: "100",
      });

      const response = await fetch(`/api/data/open-interest?${params}`);
      const result: OpenInterestResponse = await response.json();

      if (result.success) {
        setOpenInterestData(result.data.open_interest);
      } else {
        setOpenInterestError(result.message || "OIデータの取得に失敗しました");
      }
    } catch (err) {
      setOpenInterestError("OIデータの取得中にエラーが発生しました");
      console.error("OIデータ取得エラー:", err);
    } finally {
      setOpenInterestLoading(false);
    }
  };

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
    try {
      setUpdating(true);
      setError("");

      const response = await fetch(
        `${BACKEND_API_URL}/api/data-collection/update?symbol=${selectedSymbol}&timeframe=${selectedTimeFrame}`,
        {
          method: "POST",
        }
      );

      const result = await response.json();

      if (result.success) {
        // 更新後にデータを再取得
        await fetchOHLCVData();
      } else {
        setError(result.message || "差分更新に失敗しました");
      }
    } catch (err) {
      setError("差分更新中にエラーが発生しました");
      console.error("差分更新エラー:", err);
    } finally {
      setUpdating(false);
    }
  };

  /**
   * データ収集状況を取得
   */
  const fetchDataStatus = async () => {
    try {
      const url = `${BACKEND_API_URL}/api/data-collection/status/${selectedSymbol}/${selectedTimeFrame}`;
      const response = await fetch(url);
      const result = await response.json();

      if (result.success) {
        setDataStatus(result);
      }
    } catch (err) {
      console.error("データ状況取得エラー:", err);
    }
  };

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
    if (result.status === "completed") {
      const ohlcvCount = result.ohlcv_result?.total_tasks || 0;
      const fundingCount = result.funding_rate_result?.total_saved_records || 0;
      const openInterestCount =
        result.open_interest_result?.total_saved_records || 0;

      setAllDataCollectionMessage(
        `🚀 全データ収集完了！ OHLCV:${ohlcvCount}タスク, FR:${fundingCount}件, OI:${openInterestCount}件, TI:自動計算済み`
      );
    } else {
      setAllDataCollectionMessage(`🔄 ${result.message} (実行中...)`);
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

  // 初期データ取得
  useEffect(() => {
    fetchSymbols();
  }, []);

  // 通貨ペアまたは時間軸変更時にデータを再取得
  useEffect(() => {
    if (selectedSymbol && selectedTimeFrame) {
      fetchOHLCVData();
      fetchFundingRateData();
      fetchOpenInterestData();
      fetchDataStatus();
    }
  }, [selectedSymbol, selectedTimeFrame]);

  return (
    <div className="min-h-screen bg-secondary-50 dark:bg-secondary-950 animate-fade-in">
      <DataHeader
        loading={loading}
        error={error}
        updating={updating}
        handleRefresh={handleRefresh}
        handleIncrementalUpdate={handleIncrementalUpdate}
      />

      {/* メインコンテンツエリア */}
      <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8 py-8 space-y-8">
        {/* エラー表示 */}
        {error && (
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
                {error}
              </p>
            </div>
          </div>
        )}

        <DataControls
          dataStatus={dataStatus}
          symbols={symbols}
          selectedSymbol={selectedSymbol}
          handleSymbolChange={handleSymbolChange}
          symbolsLoading={symbolsLoading}
          loading={loading}
          selectedTimeFrame={selectedTimeFrame}
          handleTimeFrameChange={handleTimeFrameChange}
          updating={updating}
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
        />

        <DataTableContainer
          selectedSymbol={selectedSymbol}
          selectedTimeFrame={selectedTimeFrame}
          activeTab={activeTab}
          setActiveTab={setActiveTab}
          ohlcvData={ohlcvData}
          loading={loading}
          error={error}
          fundingRateData={fundingRateData}
          fundingLoading={fundingLoading}
          fundingError={fundingError}
          openInterestData={openInterestData}
          openInterestLoading={openInterestLoading}
          openInterestError={openInterestError}
        />
      </div>
    </div>
  );
};

export default DataPage;
