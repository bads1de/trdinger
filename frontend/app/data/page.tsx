/**
 * データページコンポーネント
 *
 * ローソク足チャートを表示し、通貨ペアと時間軸を選択できるページです。
 * リアルタイムでチャートデータを取得・表示します。
 *
 * @author Trdinger Development Team
 * @version 1.0.0
 */

"use client";

import React, { useState, useEffect } from "react";
import CandlestickChart from "@/components/CandlestickChart";
import TimeFrameSelector from "@/components/TimeFrameSelector";
import SymbolSelector from "@/components/SymbolSelector";
import {
  CandlestickData,
  TimeFrame,
  TradingPair,
  CandlestickResponse,
} from "@/types/strategy";

/**
 * データページコンポーネント
 */
const DataPage: React.FC = () => {
  // 状態管理
  const [symbols, setSymbols] = useState<TradingPair[]>([]);
  const [selectedSymbol, setSelectedSymbol] = useState<string>("BTC/USDT");
  const [selectedTimeFrame, setSelectedTimeFrame] = useState<TimeFrame>("1d");
  const [candlestickData, setCandlestickData] = useState<CandlestickData[]>([]);
  const [loading, setLoading] = useState<boolean>(false);
  const [error, setError] = useState<string>("");
  const [symbolsLoading, setSymbolsLoading] = useState<boolean>(true);
  const [updating, setUpdating] = useState<boolean>(false);
  const [dataStatus, setDataStatus] = useState<any>(null);

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
   * ローソク足データを取得
   */
  const fetchCandlestickData = async () => {
    try {
      setLoading(true);
      setError("");

      const params = new URLSearchParams({
        symbol: selectedSymbol,
        timeframe: selectedTimeFrame,
        limit: "100",
      });

      const response = await fetch(`/api/data/candlesticks?${params}`);
      const result: CandlestickResponse = await response.json();

      if (result.success) {
        setCandlestickData(result.data.candlesticks);
      } else {
        setError(result.message || "データの取得に失敗しました");
      }
    } catch (err) {
      setError("データの取得中にエラーが発生しました");
      console.error("ローソク足データ取得エラー:", err);
    } finally {
      setLoading(false);
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
    fetchCandlestickData();
  };

  /**
   * 差分データ更新
   */
  const handleIncrementalUpdate = async () => {
    try {
      setUpdating(true);
      setError("");

      const response = await fetch(
        `http://127.0.0.1:8000/api/v1/data-collection/update?symbol=${selectedSymbol}&timeframe=${selectedTimeFrame}`, // ポート番号を8001から8000に変更
        {
          method: "POST",
        }
      );

      const result = await response.json();

      if (result.success) {
        // 更新後にデータを再取得
        await fetchCandlestickData();
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
      const url = `http://127.0.0.1:8000/api/v1/data-collection/status/${selectedSymbol}/${selectedTimeFrame}`; // ポート番号を8001から8000に変更
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

  // 初期データ取得
  useEffect(() => {
    fetchSymbols();
  }, []);

  // 通貨ペアまたは時間軸変更時にデータを再取得
  useEffect(() => {
    if (selectedSymbol && selectedTimeFrame) {
      fetchCandlestickData();
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
                📊 チャートデータ
              </h1>
              <p className="mt-2 text-base text-secondary-600 dark:text-secondary-400">
                エンタープライズレベルの仮想通貨ローソク足チャート分析
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

        {/* コントロールパネル */}
        <div className="enterprise-card animate-slide-up">
          <div className="p-6">
            <div className="flex items-center justify-between mb-6">
              <h2 className="text-xl font-semibold text-secondary-900 dark:text-secondary-100">
                📈 チャート設定
              </h2>
              <span className="text-sm text-secondary-500 dark:text-secondary-400">
                表示パラメータを調整
              </span>
            </div>

            <div className="grid grid-cols-1 lg:grid-cols-2 gap-6">
              {/* 通貨ペア選択 */}
              <div className="space-y-2">
                <SymbolSelector
                  symbols={symbols}
                  selectedSymbol={selectedSymbol}
                  onSymbolChange={handleSymbolChange}
                  loading={symbolsLoading}
                  disabled={loading}
                />
              </div>

              {/* 時間軸選択 */}
              <div className="space-y-2">
                <TimeFrameSelector
                  selectedTimeFrame={selectedTimeFrame}
                  onTimeFrameChange={handleTimeFrameChange}
                  disabled={loading}
                />
              </div>
            </div>
          </div>
        </div>

        {/* チャート表示エリア */}
        <div className="enterprise-card animate-slide-up">
          <div className="p-6">
            <div className="flex items-center justify-between mb-6">
              <div>
                <h2 className="text-xl font-semibold text-secondary-900 dark:text-secondary-100">
                  📊 {selectedSymbol} - {selectedTimeFrame}足チャート
                </h2>
                <p className="text-sm text-secondary-600 dark:text-secondary-400 mt-1">
                  {candlestickData.length > 0 &&
                    !loading &&
                    `${candlestickData.length}件のデータポイントを表示中`}
                </p>
              </div>

              {/* チャート情報バッジ */}
              {candlestickData.length > 0 && !loading && (
                <div className="flex items-center gap-2">
                  <span className="badge-primary">
                    {candlestickData.length}件
                  </span>
                  <span className="badge-success">
                    最新: $
                    {candlestickData[candlestickData.length - 1]?.close.toFixed(
                      2
                    )}
                  </span>
                </div>
              )}
            </div>

            <div className="relative">
              <CandlestickChart
                data={candlestickData}
                height={600}
                loading={loading}
                error={error}
              />

              {/* ローディングオーバーレイ */}
              {loading && (
                <div className="absolute inset-0 glass-effect rounded-enterprise-lg flex items-center justify-center">
                  <div className="text-center">
                    <div className="animate-spin rounded-full h-12 w-12 border-b-2 border-primary-600 mx-auto mb-4"></div>
                    <p className="text-sm font-medium text-secondary-700 dark:text-secondary-300">
                      チャートデータを読み込み中...
                    </p>
                  </div>
                </div>
              )}
            </div>
          </div>
        </div>

        {/* データ統計情報 */}
        {candlestickData.length > 0 && !loading && !error && (
          <div className="enterprise-card animate-slide-up">
            <div className="p-6">
              <div className="flex items-center justify-between mb-6">
                <h3 className="text-xl font-semibold text-secondary-900 dark:text-secondary-100">
                  📈 データ統計
                </h3>
                <span className="text-sm text-secondary-500 dark:text-secondary-400">
                  期間:{" "}
                  {new Date(candlestickData[0]?.timestamp).toLocaleDateString(
                    "ja-JP"
                  )}{" "}
                  -{" "}
                  {new Date(
                    candlestickData[candlestickData.length - 1]?.timestamp
                  ).toLocaleDateString("ja-JP")}
                </span>
              </div>

              <div className="grid grid-cols-2 md:grid-cols-4 gap-6">
                <div className="text-center p-4 bg-secondary-50 dark:bg-secondary-800/50 rounded-enterprise border border-secondary-200 dark:border-secondary-700">
                  <div className="text-2xl font-bold text-primary-600 dark:text-primary-400">
                    $
                    {candlestickData[candlestickData.length - 1]?.close.toFixed(
                      2
                    )}
                  </div>
                  <div className="text-sm text-secondary-600 dark:text-secondary-400 mt-1">
                    最新価格
                  </div>
                </div>

                <div className="text-center p-4 bg-secondary-50 dark:bg-secondary-800/50 rounded-enterprise border border-secondary-200 dark:border-secondary-700">
                  <div className="text-2xl font-bold text-success-600 dark:text-success-400">
                    $
                    {Math.max(...candlestickData.map((d) => d.high)).toFixed(2)}
                  </div>
                  <div className="text-sm text-secondary-600 dark:text-secondary-400 mt-1">
                    期間最高値
                  </div>
                </div>

                <div className="text-center p-4 bg-secondary-50 dark:bg-secondary-800/50 rounded-enterprise border border-secondary-200 dark:border-secondary-700">
                  <div className="text-2xl font-bold text-error-600 dark:text-error-400">
                    ${Math.min(...candlestickData.map((d) => d.low)).toFixed(2)}
                  </div>
                  <div className="text-sm text-secondary-600 dark:text-secondary-400 mt-1">
                    期間最安値
                  </div>
                </div>

                <div className="text-center p-4 bg-secondary-50 dark:bg-secondary-800/50 rounded-enterprise border border-secondary-200 dark:border-secondary-700">
                  <div className="text-2xl font-bold text-accent-600 dark:text-accent-400">
                    {(
                      ((candlestickData[candlestickData.length - 1]?.close -
                        candlestickData[0]?.open) /
                        candlestickData[0]?.open) *
                        100 || 0
                    ).toFixed(2)}
                    %
                  </div>
                  <div className="text-sm text-secondary-600 dark:text-secondary-400 mt-1">
                    期間変動率
                  </div>
                </div>
              </div>

              {/* 追加統計情報 */}
              <div className="mt-6 pt-6 border-t border-secondary-200 dark:border-secondary-700">
                <div className="grid grid-cols-1 md:grid-cols-3 gap-4 text-sm">
                  <div className="flex justify-between">
                    <span className="text-secondary-600 dark:text-secondary-400">
                      平均価格:
                    </span>
                    <span className="font-medium text-secondary-900 dark:text-secondary-100">
                      $
                      {(
                        candlestickData.reduce((sum, d) => sum + d.close, 0) /
                        candlestickData.length
                      ).toFixed(2)}
                    </span>
                  </div>
                  <div className="flex justify-between">
                    <span className="text-secondary-600 dark:text-secondary-400">
                      データポイント:
                    </span>
                    <span className="font-medium text-secondary-900 dark:text-secondary-100">
                      {candlestickData.length.toLocaleString()}件
                    </span>
                  </div>
                  <div className="flex justify-between">
                    <span className="text-secondary-600 dark:text-secondary-400">
                      最終更新:
                    </span>
                    <span className="font-medium text-secondary-900 dark:text-secondary-100">
                      {new Date().toLocaleTimeString("ja-JP")}
                    </span>
                  </div>
                </div>
              </div>
            </div>
          </div>
        )}
      </div>
    </div>
  );
};

export default DataPage;
