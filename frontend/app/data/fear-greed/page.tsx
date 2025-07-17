/**
 * Fear & Greed Index データページ
 *
 * Fear & Greed Index データの表示・収集・管理を行うページです。
 */

"use client";

import React, { useState } from "react";
import { useFearGreedData, FearGreedCollectionResult } from "@/hooks/useFearGreedData";
import FearGreedDataTable from "@/components/data/FearGreedDataTable";
import FearGreedCollectionButton from "@/components/button/FearGreedCollectionButton";
import FearGreedStatusCard from "@/components/data/FearGreedStatusCard";

const FearGreedPage: React.FC = () => {
  const {
    data,
    loading,
    error,
    status,
    fetchData,
    fetchLatestData,
    fetchStatus,
  } = useFearGreedData();

  const [collectionMessage, setCollectionMessage] = useState<string>("");
  const [collectionError, setCollectionError] = useState<string>("");
  const [isRefreshing, setIsRefreshing] = useState<boolean>(false);

  /**
   * データ収集開始時の処理
   */
  const handleCollectionStart = (result: FearGreedCollectionResult) => {
    if (result.success) {
      setCollectionMessage(
        `✅ ${result.message} (取得: ${result.fetched_count}件, 挿入: ${result.inserted_count}件)`
      );
      setCollectionError("");
      
      // 状態を更新
      fetchStatus();
    } else {
      setCollectionError(`❌ ${result.message}`);
      setCollectionMessage("");
    }
  };

  /**
   * データ収集エラー時の処理
   */
  const handleCollectionError = (errorMessage: string) => {
    setCollectionError(`❌ ${errorMessage}`);
    setCollectionMessage("");
  };

  /**
   * データを手動で更新
   */
  const handleRefreshData = async () => {
    setIsRefreshing(true);
    try {
      await fetchLatestData(100); // 最新100件を取得
      await fetchStatus();
      setCollectionMessage("✅ データを更新しました");
      setCollectionError("");
    } catch (err) {
      setCollectionError("❌ データ更新に失敗しました");
      setCollectionMessage("");
    } finally {
      setIsRefreshing(false);
    }
  };

  /**
   * 全期間データを取得
   */
  const handleFetchAllData = async () => {
    setIsRefreshing(true);
    try {
      await fetchData(1000); // 最大1000件を取得
      setCollectionMessage("✅ 全期間データを取得しました");
      setCollectionError("");
    } catch (err) {
      setCollectionError("❌ 全期間データ取得に失敗しました");
      setCollectionMessage("");
    } finally {
      setIsRefreshing(false);
    }
  };

  return (
    <div className="container mx-auto px-4 py-8">
      {/* ページヘッダー */}
      <div className="mb-8">
        <h1 className="text-3xl font-bold text-secondary-900 dark:text-secondary-100 mb-2">
          😨 Fear & Greed Index
        </h1>
        <p className="text-secondary-600 dark:text-secondary-400">
          Alternative.me APIから取得したセンチメント指標データの管理
        </p>
      </div>

      {/* 状態とコントロール */}
      <div className="grid grid-cols-1 lg:grid-cols-3 gap-6 mb-8">
        {/* 状態カード */}
        <div className="lg:col-span-1">
          <FearGreedStatusCard status={status} loading={loading} />
        </div>

        {/* データ収集コントロール */}
        <div className="lg:col-span-2">
          <div className="enterprise-card">
            <div className="p-6">
              <h3 className="text-xl font-semibold text-secondary-900 dark:text-secondary-100 mb-4">
                📥 データ収集・管理
              </h3>
              
              <FearGreedCollectionButton
                onCollectionStart={handleCollectionStart}
                onCollectionError={handleCollectionError}
                disabled={loading || isRefreshing}
              />

              {/* 手動更新ボタン */}
              <div className="mt-4 pt-4 border-t border-secondary-200 dark:border-secondary-700">
                <div className="grid grid-cols-1 md:grid-cols-2 gap-3">
                  <button
                    onClick={handleRefreshData}
                    disabled={loading || isRefreshing}
                    className="btn-outline text-sm py-2 px-4 disabled:opacity-50 disabled:cursor-not-allowed"
                  >
                    {isRefreshing ? (
                      <div className="flex items-center justify-center">
                        <div className="animate-spin rounded-full h-4 w-4 border-b-2 border-current mr-2"></div>
                        更新中...
                      </div>
                    ) : (
                      "📊 表示データ更新"
                    )}
                  </button>

                  <button
                    onClick={handleFetchAllData}
                    disabled={loading || isRefreshing}
                    className="btn-outline text-sm py-2 px-4 disabled:opacity-50 disabled:cursor-not-allowed"
                  >
                    {isRefreshing ? (
                      <div className="flex items-center justify-center">
                        <div className="animate-spin rounded-full h-4 w-4 border-b-2 border-current mr-2"></div>
                        取得中...
                      </div>
                    ) : (
                      "📈 全期間データ表示"
                    )}
                  </button>
                </div>
              </div>
            </div>
          </div>
        </div>
      </div>

      {/* メッセージ表示 */}
      {(collectionMessage || collectionError) && (
        <div className="mb-6">
          {collectionMessage && (
            <div className="enterprise-card bg-green-50 dark:bg-green-900/20 border-green-200 dark:border-green-800">
              <div className="p-4">
                <p className="text-green-800 dark:text-green-200 text-sm">
                  {collectionMessage}
                </p>
              </div>
            </div>
          )}
          {collectionError && (
            <div className="enterprise-card bg-red-50 dark:bg-red-900/20 border-red-200 dark:border-red-800">
              <div className="p-4">
                <p className="text-red-800 dark:text-red-200 text-sm">
                  {collectionError}
                </p>
              </div>
            </div>
          )}
        </div>
      )}

      {/* データテーブル */}
      <FearGreedDataTable
        data={data}
        loading={loading}
        error={error}
      />

      {/* フッター情報 */}
      <div className="mt-8 text-center text-sm text-secondary-500 dark:text-secondary-400">
        <p>
          データソース: <a 
            href="https://alternative.me/crypto/fear-and-greed-index/" 
            target="_blank" 
            rel="noopener noreferrer"
            className="text-primary-600 dark:text-primary-400 hover:underline"
          >
            Alternative.me Fear & Greed Index
          </a>
        </p>
        <p className="mt-1">
          Fear & Greed Index は暗号通貨市場のセンチメントを0-100の数値で表現します
        </p>
      </div>
    </div>
  );
};

export default FearGreedPage;
