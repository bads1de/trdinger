/**
 * 全データ一括収集ボタンコンポーネント
 *
 * OHLCV、FR、OIの
 * 全データタイプを一括で収集するUIコンポーネントです。
 * ダークモード対応、状態管理、エラーハンドリング、進行状況表示を含みます。
 *
 * @author Trdinger Development Team
 * @version 1.0.0
 */

"use client";

import React, { useState } from "react";
import {
  AllDataCollectionResult,
  BulkOHLCVCollectionResult,
  BulkFundingRateCollectionResult,
  BulkOpenInterestCollectionResult,
} from "@/types/strategy";

/**
 * 全データ一括収集ボタンコンポーネントのプロパティ
 */
interface AllDataCollectionButtonProps {
  /** データ収集開始時のコールバック */
  onCollectionStart?: (result: AllDataCollectionResult) => void;
  /** データ収集エラー時のコールバック */
  onCollectionError?: (error: string) => void;
  /** ボタンの無効化フラグ */
  disabled?: boolean;
  /** カスタムクラス名 */
  className?: string;
}

/**
 * 収集ステップを表す列挙型
 */
type CollectionStep =
  | "idle"
  | "ohlcv"
  | "funding_rate"
  | "open_interest"
  | "completed"
  | "error";

/**
 * 全データ一括収集ボタンコンポーネント
 */
const AllDataCollectionButton: React.FC<AllDataCollectionButtonProps> = ({
  onCollectionStart,
  onCollectionError,
  disabled = false,
  className = "",
}) => {
  const [currentStep, setCurrentStep] = useState<CollectionStep>("idle");
  const [collectionResult, setCollectionResult] =
    useState<AllDataCollectionResult | null>(null);
  const [stepResults, setStepResults] = useState<{
    ohlcv?: BulkOHLCVCollectionResult;
    funding_rate?: BulkFundingRateCollectionResult;
    open_interest?: BulkOpenInterestCollectionResult;
  }>({});

  /**
   * 全データを一括収集
   */
  const handleCollectAllData = async () => {
    // 確認ダイアログ
    const confirmed = window.confirm(
      `全データ（OHLCV・FR・OI）を一括取得します。\n\n` +
        "この処理には数分から十数分かかる場合があります。続行しますか？"
    );

    if (!confirmed) {
      return;
    }

    const startTime = new Date().toISOString();

    try {
      setCurrentStep("ohlcv");
      setCollectionResult(null);
      setStepResults({});

      const result: AllDataCollectionResult = {
        success: false,
        message: "全データ一括収集を開始しました",
        status: "started",
        total_steps: 3,
        completed_steps: 0,
        current_step: "ohlcv",
        started_at: startTime,
      };

      setCollectionResult(result);
      onCollectionStart?.(result);

      // ステップ1: OHLCV一括収集
      setCurrentStep("ohlcv");
      result.current_step = "ohlcv";
      result.status = "in_progress";
      setCollectionResult({ ...result });

      const ohlcvResponse = await fetch("/api/data/ohlcv/bulk", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
      });

      const ohlcvResult: BulkOHLCVCollectionResult = await ohlcvResponse.json();

      if (!ohlcvResponse.ok || !ohlcvResult.success) {
        throw new Error(
          `OHLCV収集に失敗: ${ohlcvResult.message || "Unknown error"}`
        );
      }

      setStepResults((prev) => ({ ...prev, ohlcv: ohlcvResult }));
      result.completed_steps = 1;
      result.ohlcv_result = ohlcvResult;
      setCollectionResult({ ...result });

      // ステップ2: FR一括収集
      setCurrentStep("funding_rate");
      result.current_step = "funding_rate";
      setCollectionResult({ ...result });

      const fundingResponse = await fetch("/api/data/funding-rates/bulk", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
      });

      const fundingResult: BulkFundingRateCollectionResult =
        await fundingResponse.json();

      if (!fundingResponse.ok || !fundingResult.success) {
        throw new Error(
          `FR収集に失敗: ${fundingResult.message || "Unknown error"}`
        );
      }

      setStepResults((prev) => ({ ...prev, funding_rate: fundingResult }));
      result.completed_steps = 2;
      result.funding_rate_result = fundingResult;
      setCollectionResult({ ...result });

      // ステップ3: OI一括収集
      setCurrentStep("open_interest");
      result.current_step = "open_interest";
      setCollectionResult({ ...result });

      const openInterestResponse = await fetch(
        "/api/data/open-interest/bulk-collect",
        {
          method: "POST",
          headers: { "Content-Type": "application/json" },
        }
      );

      const openInterestResult: BulkOpenInterestCollectionResult =
        await openInterestResponse.json();

      if (!openInterestResponse.ok || !openInterestResult.success) {
        throw new Error(
          `OI収集に失敗: ${openInterestResult.message || "Unknown error"}`
        );
      }

      setStepResults((prev) => ({
        ...prev,
        open_interest: openInterestResult,
      }));
      result.completed_steps = 3;
      result.open_interest_result = openInterestResult;
      result.status = "completed";
      result.completed_at = new Date().toISOString();
      result.success = true;
      result.message = "全データの一括収集が完了しました";

      setCurrentStep("completed");
      setCollectionResult(result);
      onCollectionStart?.(result);

      // 10秒後にアイドル状態に戻す
      setTimeout(() => {
        setCurrentStep("idle");
      }, 10000);
    } catch (error) {
      console.error("全データ一括収集エラー:", error);
      setCurrentStep("error");

      const errorMessage =
        error instanceof Error
          ? error.message
          : "全データ一括収集中にエラーが発生しました";

      const errorResult: AllDataCollectionResult = {
        success: false,
        message: errorMessage,
        status: "error",
        total_steps: 3,
        completed_steps: collectionResult?.completed_steps || 0,
        started_at: startTime,
        completed_at: new Date().toISOString(),
        error_details: errorMessage,
        ...stepResults,
      };

      setCollectionResult(errorResult);
      onCollectionError?.(errorMessage);

      // 10秒後にアイドル状態に戻す
      setTimeout(() => {
        setCurrentStep("idle");
      }, 10000);
    }
  };

  /**
   * ボタンが無効かどうかを判定
   */
  const isButtonDisabled =
    disabled ||
    (currentStep !== "idle" &&
      currentStep !== "completed" &&
      currentStep !== "error");

  /**
   * ボタンのスタイルクラスを取得
   */
  const getButtonClasses = () => {
    const baseClasses =
      "flex items-center gap-2 px-4 py-2 text-sm font-medium rounded-lg transition-all duration-200 min-w-[160px] h-[36px]";

    if (isButtonDisabled) {
      return `${baseClasses} bg-gray-700 text-gray-400 cursor-not-allowed`;
    }

    switch (currentStep) {
      case "completed":
        return `${baseClasses} bg-green-600 text-white hover:bg-green-700`;
      case "error":
        return `${baseClasses} bg-red-600 text-white hover:bg-red-700`;
      default:
        return `${baseClasses} bg-purple-600 text-white hover:bg-purple-700 cursor-pointer`;
    }
  };

  /**
   * ボタンのアイコンを取得
   */
  const getButtonIcon = () => {
    if (
      currentStep === "idle" ||
      currentStep === "completed" ||
      currentStep === "error"
    ) {
      return (
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
            d="M4 7v10c0 2.21 1.79 4 4 4h8c2.21 0 4-1.79 4-4V7M4 7l8-4 8 4M4 7l8 4 8-4"
          />
        </svg>
      );
    }

    return (
      <svg
        className="w-4 h-4 animate-spin"
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
    );
  };

  /**
   * ボタンのテキストを取得
   */
  const getButtonText = () => {
    switch (currentStep) {
      case "ohlcv":
        return "OHLCV収集中... (1/3)";
      case "funding_rate":
        return "ファンディング収集中... (2/3)";
      case "open_interest":
        return "OI収集中... (3/3)";
      case "completed":
        return "全データ収集完了";
      case "error":
        return "収集エラー";
      default:
        return "全データ一括取得";
    }
  };

  return (
    <button
      onClick={handleCollectAllData}
      disabled={isButtonDisabled}
      className={`${getButtonClasses()} ${className}`}
      title="OHLCV・FR・OIの全データを一括収集"
    >
      {getButtonIcon()}
      <span>{getButtonText()}</span>
    </button>
  );
};

export default AllDataCollectionButton;
