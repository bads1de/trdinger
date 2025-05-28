/**
 * データページコンポーネント
 *
 * OHLCVデータとファンディングレートデータを表形式で表示するページです。
 * リアルタイムでデータを取得・表示します。
 *
 * @author Trdinger Development Team
 * @version 2.0.0
 */

"use client";

import React, { useState, useEffect } from "react";
import OHLCVDataTable from "@/components/OHLCVDataTable";
import FundingRateDataTable from "@/components/FundingRateDataTable";
import CompactSymbolSelector from "@/components/CompactSymbolSelector";
import CompactTimeFrameSelector from "@/components/CompactTimeFrameSelector";
import CompactDataCollectionButtons from "@/components/CompactDataCollectionButtons";
import {
  PriceData,
  FundingRateData,
  TimeFrame,
  TradingPair,
  OHLCVResponse,
  FundingRateResponse,
  BulkOHLCVCollectionResult,
  BulkFundingRateCollectionResult,
  FundingRateCollectionResult,
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
  const [activeTab, setActiveTab] = useState<"ohlcv" | "funding">("ohlcv");
  const [loading, setLoading] = useState<boolean>(false);
  const [fundingLoading, setFundingLoading] = useState<boolean>(false);
  const [error, setError] = useState<string>("");
  const [fundingError, setFundingError] = useState<string>("");
  const [symbolsLoading, setSymbolsLoading] = useState<boolean>(true);
  const [updating, setUpdating] = useState<boolean>(false);
  const [dataStatus, setDataStatus] = useState<any>(null);
  const [bulkCollectionMessage, setBulkCollectionMessage] =
    useState<string>("");
  const [fundingRateCollectionMessage, setFundingRateCollectionMessage] =
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
   * ファンディングレートデータを取得
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
        setFundingError(
          result.message || "ファンディングレートデータの取得に失敗しました"
        );
      }
    } catch (err) {
      setFundingError(
        "ファンディングレートデータの取得中にエラーが発生しました"
      );
      console.error("ファンディングレートデータ取得エラー:", err);
    } finally {
      setFundingLoading(false);
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
    } else {
      fetchFundingRateData();
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
        console.log(`差分更新完了: ${result.saved_count}件`);
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
      console.log("Requesting data status from:", url); // ★ログ追加
      const response = await fetch(url);
      const result = await response.json();

      if (result.success) {
        setDataStatus(result);
      }
    } catch (err) {
      console.error("データ状況取得エラー詳細:", err); // ★エラーオブジェクト全体をログに出力
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
   * ファンディングレートデータ収集開始時のコールバック
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
        `🚀 ${singleResult.symbol}のファンディングレートデータ収集完了 (${singleResult.saved_count}件保存)`
      );
    }
    // 10秒後にメッセージをクリア
    setTimeout(() => setFundingRateCollectionMessage(""), 10000);
  };

  /**
   * ファンディングレートデータ収集エラー時のコールバック
   */
  const handleFundingRateCollectionError = (errorMessage: string) => {
    setFundingRateCollectionMessage(`❌ ${errorMessage}`);
    // 10秒後にメッセージをクリア
    setTimeout(() => setFundingRateCollectionMessage(""), 10000);
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
          <div className="p-4">
            <div className="flex flex-col lg:flex-row lg:items-center lg:justify-between gap-4">
              <div className="flex items-center gap-2">
                <h2 className="text-lg font-semibold text-secondary-900 dark:text-secondary-100">
                  📈 データ設定
                </h2>
              </div>

              <div className="flex flex-col sm:flex-row items-start sm:items-center gap-4">
                {/* 通貨ペア選択 */}
                <div className="flex items-center gap-2">
                  <span className="text-sm text-secondary-600 dark:text-secondary-400 whitespace-nowrap">
                    通貨ペア:
                  </span>
                  <CompactSymbolSelector
                    symbols={symbols}
                    selectedSymbol={selectedSymbol}
                    onSymbolChange={handleSymbolChange}
                    loading={symbolsLoading}
                    disabled={loading}
                  />
                </div>

                {/* 時間軸選択 */}
                <div className="flex items-center gap-2">
                  <span className="text-sm text-secondary-600 dark:text-secondary-400 whitespace-nowrap">
                    時間軸:
                  </span>
                  <CompactTimeFrameSelector
                    selectedTimeFrame={selectedTimeFrame}
                    onTimeFrameChange={handleTimeFrameChange}
                    disabled={loading}
                  />
                </div>

                {/* データ収集ボタン */}
                <div className="flex items-center gap-2">
                  <span className="text-sm text-secondary-600 dark:text-secondary-400 whitespace-nowrap">
                    データ収集:
                  </span>
                  <CompactDataCollectionButtons
                    onBulkCollectionStart={handleBulkCollectionStart}
                    onBulkCollectionError={handleBulkCollectionError}
                    onFundingRateCollectionStart={handleFundingRateCollectionStart}
                    onFundingRateCollectionError={handleFundingRateCollectionError}
                    disabled={loading || updating}
                  />
                </div>
              </div>
            </div>

            {/* ステータスメッセージ */}
            {(bulkCollectionMessage || fundingRateCollectionMessage) && (
              <div className="mt-3 pt-3 border-t border-secondary-200 dark:border-secondary-700">
                {bulkCollectionMessage && (
                  <div className="text-sm text-secondary-600 dark:text-secondary-400 mb-1">
                    {bulkCollectionMessage}
                  </div>
                )}
                {fundingRateCollectionMessage && (
                  <div className="text-sm text-secondary-600 dark:text-secondary-400">
                    {fundingRateCollectionMessage}
                  </div>
                )}
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
                <div className="flex bg-gray-800 dark:bg-gray-800 rounded-lg p-1">
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
                    ファンディングレート
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
            </div>
          </div>
        </div>
      </div>
    </div>
  );
};

export default DataPage;
