/**
 * コンパクトデータ収集ボタンコンポーネント
 *
 * データ設定セクション用のコンパクトなデータ収集ボタン群
 */

"use client";

import React, { useState } from "react";
import {
  BulkOHLCVCollectionResult,
  BulkFundingRateCollectionResult,
  FundingRateCollectionResult,
} from "@/types/strategy";

interface CompactDataCollectionButtonsProps {
  onBulkCollectionStart?: (result: BulkOHLCVCollectionResult) => void;
  onBulkCollectionError?: (error: string) => void;
  onFundingRateCollectionStart?: (
    result: BulkFundingRateCollectionResult | FundingRateCollectionResult
  ) => void;
  onFundingRateCollectionError?: (error: string) => void;
  disabled?: boolean;
}

const CompactDataCollectionButtons: React.FC<
  CompactDataCollectionButtonsProps
> = ({
  onBulkCollectionStart,
  onBulkCollectionError,
  onFundingRateCollectionStart,
  onFundingRateCollectionError,
  disabled = false,
}) => {
  const [ohlcvLoading, setOhlcvLoading] = useState(false);
  const [fundingLoading, setFundingLoading] = useState(false);

  const handleOHLCVCollection = async () => {
    if (!confirm("全ペア・全時間軸でOHLCVデータを収集しますか？")) return;

    try {
      setOhlcvLoading(true);
      const response = await fetch("/api/data/ohlcv/bulk", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
      });

      const result: BulkOHLCVCollectionResult = await response.json();

      if (response.ok && result.success) {
        onBulkCollectionStart?.(result);
      } else {
        onBulkCollectionError?.(result.message || "OHLCV収集に失敗しました");
      }
    } catch (error) {
      onBulkCollectionError?.("OHLCV収集中にエラーが発生しました");
    } finally {
      setOhlcvLoading(false);
    }
  };

  const handleFundingRateCollection = async () => {
    if (!confirm("FRデータを収集しますか？")) return;

    try {
      setFundingLoading(true);
      const response = await fetch("/api/data/funding-rates/bulk", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
      });

      const result = await response.json();

      if (response.ok && result.success) {
        onFundingRateCollectionStart?.(result);
      } else {
        onFundingRateCollectionError?.(
          result.message || "FR収集に失敗しました"
        );
      }
    } catch (error) {
      onFundingRateCollectionError?.("FR収集中にエラーが発生しました");
    } finally {
      setFundingLoading(false);
    }
  };

  return (
    <div className="flex gap-2">
      {/* OHLCV一括収集ボタン */}
      <button
        onClick={handleOHLCVCollection}
        disabled={disabled || ohlcvLoading || fundingLoading}
        className={`
          flex items-center gap-2 px-4 py-2 text-sm font-medium rounded-lg
          transition-all duration-200 min-w-[120px] h-[36px]
          ${
            disabled || ohlcvLoading || fundingLoading
              ? "bg-gray-700 text-gray-400 cursor-not-allowed"
              : "bg-blue-600 text-white hover:bg-blue-700 cursor-pointer"
          }
          focus:outline-none focus:ring-2 focus:ring-blue-500 focus:ring-offset-1
        `}
      >
        {ohlcvLoading ? (
          <>
            <div className="animate-spin rounded-full h-4 w-4 border-b-2 border-white"></div>
            <span>収集中...</span>
          </>
        ) : (
          <>
            <svg
              className="w-4 h-4"
              fill="none"
              stroke="currentColor"
              viewBox="0 0 24 24"
            >
              <path
                strokeLinecap="round"
                strokeLinejoin="round"
                strokeWidth={2}
                d="M7 16a4 4 0 01-.88-7.903A5 5 0 1115.9 6L16 6a5 5 0 011 9.9M9 19l3 3m0 0l3-3m-3 3V10"
              />
            </svg>
            <span>OHLCV収集</span>
          </>
        )}
      </button>

      {/* FR収集ボタン */}
      <button
        onClick={handleFundingRateCollection}
        disabled={disabled || ohlcvLoading || fundingLoading}
        className={`
          flex items-center gap-2 px-4 py-2 text-sm font-medium rounded-lg
          transition-all duration-200 min-w-[140px] h-[36px]
          ${
            disabled || ohlcvLoading || fundingLoading
              ? "bg-gray-700 text-gray-400 cursor-not-allowed"
              : "bg-green-600 text-white hover:bg-green-700 cursor-pointer"
          }
          focus:outline-none focus:ring-2 focus:ring-green-500 focus:ring-offset-1
        `}
      >
        {fundingLoading ? (
          <>
            <div className="animate-spin rounded-full h-4 w-4 border-b-2 border-white"></div>
            <span>収集中...</span>
          </>
        ) : (
          <>
            <svg
              className="w-4 h-4"
              fill="none"
              stroke="currentColor"
              viewBox="0 0 24 24"
            >
              <path
                strokeLinecap="round"
                strokeLinejoin="round"
                strokeWidth={2}
                d="M9 19v-6a2 2 0 00-2-2H5a2 2 0 00-2 2v6a2 2 0 002 2h2a2 2 0 002-2zm0 0V9a2 2 0 012-2h2a2 2 0 012 2v10m-6 0a2 2 0 002 2h2a2 2 0 002-2m0 0V5a2 2 0 012-2h2a2 2 0 012 2v4a2 2 0 01-2 2H9a2 2 0 01-2-2z"
              />
            </svg>
            <span>ファンディング収集</span>
          </>
        )}
      </button>
    </div>
  );
};

export default CompactDataCollectionButtons;
