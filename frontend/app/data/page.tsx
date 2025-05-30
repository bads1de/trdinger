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
import OHLCVDataTable from "@/components/OHLCVDataTable";
import FundingRateDataTable from "@/components/FundingRateDataTable";
import OpenInterestDataTable from "@/components/OpenInterestDataTable";
import TechnicalIndicatorDataTable from "@/components/TechnicalIndicatorDataTable";
import OpenInterestCollectionButton from "@/components/common/OpenInterestCollectionButton";
import AllDataCollectionButton from "@/components/common/AllDataCollectionButton";
import TechnicalIndicatorCalculationButton from "@/components/common/TechnicalIndicatorCalculationButton";
import SymbolSelector from "@/components/common/SymbolSelector";
import TimeFrameSelector from "@/components/common/TimeFrameSelector";

import {
  PriceData,
  FundingRateData,
  OpenInterestData,
  TechnicalIndicatorData,
  TimeFrame,
  TradingPair,
  OHLCVResponse,
  FundingRateResponse,
  OpenInterestResponse,
  TechnicalIndicatorResponse,
  BulkOHLCVCollectionResult,
  BulkFundingRateCollectionResult,
  FundingRateCollectionResult,
  OpenInterestCollectionResult,
  BulkOpenInterestCollectionResult,
  BulkTechnicalIndicatorCalculationResult,
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
  const [technicalIndicatorData, setTechnicalIndicatorData] = useState<
    TechnicalIndicatorData[]
  >([]);
  const [activeTab, setActiveTab] = useState<
    "ohlcv" | "funding" | "openinterest" | "technical"
  >("ohlcv");
  const [loading, setLoading] = useState<boolean>(false);
  const [fundingLoading, setFundingLoading] = useState<boolean>(false);
  const [openInterestLoading, setOpenInterestLoading] =
    useState<boolean>(false);
  const [technicalIndicatorLoading, setTechnicalIndicatorLoading] =
    useState<boolean>(false);
  const [error, setError] = useState<string>("");
  const [fundingError, setFundingError] = useState<string>("");
  const [openInterestError, setOpenInterestError] = useState<string>("");
  const [technicalIndicatorError, setTechnicalIndicatorError] =
    useState<string>("");
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
  const [technicalIndicatorCalculationMessage, setTechnicalIndicatorCalculationMessage] =
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
   * テクニカル指標データを取得
   */
  const fetchTechnicalIndicatorData = async () => {
    try {
      setTechnicalIndicatorLoading(true);
      setTechnicalIndicatorError("");

      const params = new URLSearchParams({
        symbol: selectedSymbol,
        timeframe: selectedTimeFrame,
        limit: "100",
      });

      const response = await fetch(`/api/data/technical-indicators?${params}`);
      const result: TechnicalIndicatorResponse = await response.json();

      if (result.success) {
        setTechnicalIndicatorData(result.data.technical_indicators);
      } else {
        setTechnicalIndicatorError(
          result.message || "テクニカル指標データの取得に失敗しました"
        );
      }
    } catch (err) {
      setTechnicalIndicatorError(
        "テクニカル指標データの取得中にエラーが発生しました"
      );
      console.error("テクニカル指標データ取得エラー:", err);
    } finally {
      setTechnicalIndicatorLoading(false);
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
    } else if (activeTab === "technical") {
      fetchTechnicalIndicatorData();
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
      setAllDataCollectionMessage(
        `🔄 ${result.message} (実行中...)`
      );
    }

    // データ状況を更新
    fetchDataStatus();

    // 全データ収集完了後に全てのデータを再取得
    setTimeout(() => {
      fetchOHLCVData();
      fetchFundingRateData();
      fetchOpenInterestData();
      fetchTechnicalIndicatorData();
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

  /**
   * TI一括計算開始時のコールバック
   */
  const handleTechnicalIndicatorCalculationStart = (result: BulkTechnicalIndicatorCalculationResult) => {
    setTechnicalIndicatorCalculationMessage(
      `🚀 ${result.symbol} ${result.timeframe}のTI一括計算完了 (${result.total_calculated}件計算完了)`
    );
    // データ状況を更新
    fetchDataStatus();
    // 計算完了後にテクニカル指標データを再取得
    setTimeout(() => {
      fetchTechnicalIndicatorData();
    }, 2000);
    // 10秒後にメッセージをクリア
    setTimeout(() => setTechnicalIndicatorCalculationMessage(""), 10000);
  };

  /**
   * TI一括計算エラー時のコールバック
   */
  const handleTechnicalIndicatorCalculationError = (errorMessage: string) => {
    setTechnicalIndicatorCalculationMessage(`❌ ${errorMessage}`);
    // 10秒後にメッセージをクリア
    setTimeout(() => setTechnicalIndicatorCalculationMessage(""), 10000);
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
      fetchTechnicalIndicatorData();
      fetchDataStatus();
    }
  }, [selectedSymbol, selectedTimeFrame]);

  return (
    <div className="min-h-screen bg-secondary-50 dark:bg-secondary-950 animate-fade-in">
      {/* エンタープライズヘッダー */}
      <div className="enterprise-card border-0 rounded-none border-b border-secondary-200 dark:border-secondary-700 shadow-enterprise-sm">
        <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8 py-6">
          <div className="flex flex-col sm:flex-row sm:items-center sm:justify-between gap-4">
            <div className="animate-slide-up">
              <h1 className="text-3xl font-bold text-gradient">
                📊 データテーブル
              </h1>
              <p className="mt-2 text-base text-secondary-600 dark:text-secondary-400">
                エンタープライズレベルの仮想通貨データ分析・表示
              </p>
              <div className="mt-2 flex items-center gap-2">
                <span className="badge-primary">リアルタイム</span>
                <span className="badge-success">高精度データ</span>
              </div>
            </div>

            <div className="flex items-center gap-3 animate-slide-up">
              {/* ステータスインジケーター */}
              <div className="flex items-center gap-2">
                <div
                  className={`w-2 h-2 rounded-full ${
                    loading
                      ? "bg-warning-500 animate-pulse"
                      : error
                      ? "bg-error-500"
                      : "bg-success-500"
                  }`}
                ></div>
                <span className="text-sm text-secondary-600 dark:text-secondary-400">
                  {loading ? "更新中" : error ? "エラー" : "接続中"}
                </span>
              </div>

              <div className="flex gap-2">
                <button
                  onClick={handleRefresh}
                  disabled={loading || updating}
                  className="btn-primary group"
                >
                  <svg
                    className={`w-4 h-4 mr-2 transition-transform duration-200 ${
                      loading ? "animate-spin" : "group-hover:rotate-180"
                    }`}
                    fill="none"
                    stroke="currentColor"
                    viewBox="0 0 24 24"
                  >
                    <path
                      strokeLinecap="round"
                      strokeLinejoin="round"
                      strokeWidth={2}
                      d="M4 4v5h.582m15.356 2A8.001 8.001 0 004.582 9m0 0H9m11 11v-5h-.581m0 0a8.003 8.003 0 01-15.357-2m15.357 2H15"
                    />
                  </svg>
                  {loading ? "更新中..." : "データ更新"}
                </button>

                <button
                  onClick={handleIncrementalUpdate}
                  disabled={loading || updating}
                  className="btn-secondary group"
                >
                  <svg
                    className={`w-4 h-4 mr-2 transition-transform duration-200 ${
                      updating ? "animate-spin" : "group-hover:scale-110"
                    }`}
                    fill="none"
                    stroke="currentColor"
                    viewBox="0 0 24 24"
                  >
                    <path
                      strokeLinecap="round"
                      strokeLinejoin="round"
                      strokeWidth={2}
                      d="M12 4v16m8-8H4"
                    />
                  </svg>
                  {updating ? "差分更新中..." : "差分更新"}
                </button>
              </div>
            </div>
          </div>
        </div>
      </div>

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

        {/* データ状況表示 */}
        {dataStatus && (
          <div className="enterprise-card animate-slide-up">
            <div className="p-6">
              <div className="flex items-center justify-between mb-4">
                <h2 className="text-xl font-semibold text-secondary-900 dark:text-secondary-100">
                  📊 データベース状況
                </h2>
                <span className="badge-primary">
                  {dataStatus.data_count?.toLocaleString()}件
                </span>
              </div>
              <div className="grid grid-cols-1 md:grid-cols-3 gap-4 text-sm">
                <div className="flex justify-between">
                  <span className="text-secondary-600 dark:text-secondary-400">
                    データ件数:
                  </span>
                  <span className="font-medium text-secondary-900 dark:text-secondary-100">
                    {dataStatus.data_count?.toLocaleString()}件
                  </span>
                </div>
                <div className="flex justify-between">
                  <span className="text-secondary-600 dark:text-secondary-400">
                    最新データ:
                  </span>
                  <span className="font-medium text-secondary-900 dark:text-secondary-100">
                    {dataStatus.latest_timestamp
                      ? new Date(dataStatus.latest_timestamp).toLocaleString(
                          "ja-JP"
                        )
                      : "なし"}
                  </span>
                </div>
                <div className="flex justify-between">
                  <span className="text-secondary-600 dark:text-secondary-400">
                    最古データ:
                  </span>
                  <span className="font-medium text-secondary-900 dark:text-secondary-100">
                    {dataStatus.oldest_timestamp
                      ? new Date(dataStatus.oldest_timestamp).toLocaleString(
                          "ja-JP"
                        )
                      : "なし"}
                  </span>
                </div>
              </div>
            </div>
          </div>
        )}

        {/* コンパクトデータ設定 */}
        <div className="enterprise-card animate-slide-up">
          <div className="p-6">
            {/* セクションヘッダー */}
            <div className="flex items-center justify-between mb-6">
              <h2 className="text-lg font-semibold text-secondary-900 dark:text-secondary-100">
                📈 データ設定
              </h2>
            </div>

            {/* 設定コントロール */}
            <div className="space-y-6">
              {/* 上段：基本設定 */}
              <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
                {/* 通貨ペア選択 */}
                <SymbolSelector
                  symbols={symbols}
                  selectedSymbol={selectedSymbol}
                  onSymbolChange={handleSymbolChange}
                  loading={symbolsLoading}
                  disabled={loading}
                  mode="compact"
                  showCategories={false}
                  enableSearch={false}
                />

                {/* 時間軸選択 */}
                <TimeFrameSelector
                  selectedTimeFrame={selectedTimeFrame}
                  onTimeFrameChange={handleTimeFrameChange}
                  disabled={loading}
                  mode="compact"
                />
              </div>

              {/* 下段：データ収集ボタン */}
              <div className="space-y-3">
                <label className="block text-sm font-medium text-secondary-600 dark:text-secondary-400">
                  データ収集
                </label>
                <div className="grid grid-cols-1 sm:grid-cols-2 lg:grid-cols-5 gap-3">
                  {/* 全データ一括収集ボタン */}
                  <AllDataCollectionButton
                    onCollectionStart={handleAllDataCollectionStart}
                    onCollectionError={handleAllDataCollectionError}
                    disabled={loading || updating}
                    className="h-10 text-sm"
                  />

                  {/* OHLCV収集ボタン（CompactDataCollectionButtonsから分離） */}
                  <button
                    onClick={async () => {
                      if (
                        !confirm(
                          "全ペア・全時間軸でOHLCVデータを収集しますか？"
                        )
                      )
                        return;

                      try {
                        const response = await fetch("/api/data/ohlcv/bulk", {
                          method: "POST",
                          headers: { "Content-Type": "application/json" },
                        });
                        const result = await response.json();

                        if (response.ok && result.success) {
                          handleBulkCollectionStart?.(result);
                        } else {
                          handleBulkCollectionError?.(
                            result.message || "OHLCV収集に失敗しました"
                          );
                        }
                      } catch (error) {
                        handleBulkCollectionError?.(
                          "OHLCV収集中にエラーが発生しました"
                        );
                      }
                    }}
                    disabled={loading || updating}
                    className="h-10 px-4 text-sm font-medium rounded-lg bg-blue-600 text-white hover:bg-blue-700 disabled:bg-gray-700 disabled:text-gray-400 disabled:cursor-not-allowed transition-all duration-200 focus:outline-none focus:ring-2 focus:ring-blue-500 focus:ring-offset-1"
                  >
                    <span>OHLCV収集</span>
                  </button>

                  {/* FR収集ボタン（CompactDataCollectionButtonsから分離） */}
                  <button
                    onClick={async () => {
                      if (!confirm("FRデータを収集しますか？")) return;

                      try {
                        const response = await fetch(
                          "/api/data/funding-rates/bulk",
                          {
                            method: "POST",
                            headers: { "Content-Type": "application/json" },
                          }
                        );
                        const result = await response.json();

                        if (response.ok && result.success) {
                          handleFundingRateCollectionStart?.(result);
                        } else {
                          handleFundingRateCollectionError?.(
                            result.message || "FR収集に失敗しました"
                          );
                        }
                      } catch (error) {
                        handleFundingRateCollectionError?.(
                          "FR収集中にエラーが発生しました"
                        );
                      }
                    }}
                    disabled={loading || updating}
                    className="h-10 px-4 text-sm font-medium rounded-lg bg-green-600 text-white hover:bg-green-700 disabled:bg-gray-700 disabled:text-gray-400 disabled:cursor-not-allowed transition-all duration-200 focus:outline-none focus:ring-2 focus:ring-green-500 focus:ring-offset-1"
                  >
                    <span>FR収集</span>
                  </button>

                  {/* OI収集ボタン */}
                  <OpenInterestCollectionButton
                    mode="bulk"
                    onCollectionStart={handleOpenInterestCollectionStart}
                    onCollectionError={handleOpenInterestCollectionError}
                    disabled={loading || updating}
                    className="h-10 text-sm"
                  />

                  {/* TI一括計算ボタン */}
                  <TechnicalIndicatorCalculationButton
                    mode="bulk"
                    symbol={selectedSymbol}
                    timeframe={selectedTimeFrame}
                    onCalculationStart={handleTechnicalIndicatorCalculationStart}
                    onCalculationError={handleTechnicalIndicatorCalculationError}
                    disabled={loading || updating}
                    className="h-10 text-sm"
                  />
                </div>
              </div>
            </div>

            {/* ステータスメッセージ */}
            {(bulkCollectionMessage ||
              fundingRateCollectionMessage ||
              openInterestCollectionMessage ||
              allDataCollectionMessage ||
              technicalIndicatorCalculationMessage) && (
              <div className="mt-6 pt-4 border-t border-secondary-200 dark:border-secondary-700">
                <div className="space-y-2">
                  {allDataCollectionMessage && (
                    <div className="text-sm text-secondary-600 dark:text-secondary-400 font-medium">
                      {allDataCollectionMessage}
                    </div>
                  )}
                  {bulkCollectionMessage && (
                    <div className="text-sm text-secondary-600 dark:text-secondary-400">
                      {bulkCollectionMessage}
                    </div>
                  )}
                  {fundingRateCollectionMessage && (
                    <div className="text-sm text-secondary-600 dark:text-secondary-400">
                      {fundingRateCollectionMessage}
                    </div>
                  )}
                  {openInterestCollectionMessage && (
                    <div className="text-sm text-secondary-600 dark:text-secondary-400">
                      {openInterestCollectionMessage}
                    </div>
                  )}
                  {technicalIndicatorCalculationMessage && (
                    <div className="text-sm text-secondary-600 dark:text-secondary-400">
                      {technicalIndicatorCalculationMessage}
                    </div>
                  )}
                </div>
              </div>
            )}
          </div>
        </div>

        {/* データ表示エリア */}
        <div className="enterprise-card animate-slide-up">
          <div className="p-6">
            {/* タブヘッダー */}
            <div className="flex items-center justify-between mb-6">
              <div className="flex items-center gap-4">
                <h2 className="text-xl font-semibold text-secondary-900 dark:text-secondary-100">
                  📊 {selectedSymbol} - データテーブル
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
                    onClick={() => setActiveTab("technical")}
                    className={`px-4 py-2 text-sm font-medium rounded-md transition-colors duration-200 ${
                      activeTab === "technical"
                        ? "bg-primary-600 text-white"
                        : "text-gray-400 hover:text-gray-100"
                    }`}
                  >
                    TI
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
                        }).format(
                          openInterestData[0]?.open_interest_value || 0
                        )}
                      </span>
                    </>
                  )}
                {activeTab === "technical" &&
                  technicalIndicatorData.length > 0 &&
                  !technicalIndicatorLoading && (
                    <>
                      <span className="badge-primary">
                        {technicalIndicatorData.length}件
                      </span>
                      <span className="badge-info">
                        指標数:{" "}
                        {
                          new Set(
                            technicalIndicatorData.map(
                              (item) => `${item.indicator_type}(${item.period})`
                            )
                          ).size
                        }
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
              {activeTab === "technical" && (
                <TechnicalIndicatorDataTable
                  data={technicalIndicatorData}
                  loading={technicalIndicatorLoading}
                  error={technicalIndicatorError}
                />
              )}
            </div>
          </div>
        </div>
      </div>
    </div>
  );
};

export default DataPage;
