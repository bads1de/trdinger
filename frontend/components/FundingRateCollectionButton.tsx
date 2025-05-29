/**
 * FRデータ収集ボタンコンポーネント
 *
 * FRデータを取得し、データベースに保存するためのUIコンポーネントです。
 * ダークモード対応、状態管理、エラーハンドリング、確認ダイアログを含みます。
 *
 * @author Trdinger Development Team
 * @version 1.0.0
 */

"use client";

import React, { useState } from "react";
import {
  FundingRateCollectionResult,
  BulkFundingRateCollectionResult,
} from "@/types/strategy";

/**
 * FR収集ボタンコンポーネントのプロパティ
 */
interface FundingRateCollectionButtonProps {
  /** データ収集開始時のコールバック */
  onCollectionStart?: (
    result: FundingRateCollectionResult | BulkFundingRateCollectionResult
  ) => void;
  /** データ収集エラー時のコールバック */
  onCollectionError?: (error: string) => void;
  /** ボタンの無効化フラグ */
  disabled?: boolean;
  /** カスタムクラス名 */
  className?: string;
  /** 収集モード */
  mode?: "single" | "bulk";
  /** 単一収集時のシンボル */
  symbol?: string;
}

/**
 * ボタンの状態を表す列挙型
 */
type ButtonState = "idle" | "loading" | "success" | "error";

/**
 * FRデータ収集ボタンコンポーネント
 */
const FundingRateCollectionButton: React.FC<
  FundingRateCollectionButtonProps
> = ({
  onCollectionStart,
  onCollectionError,
  disabled = false,
  className = "",
  mode = "bulk",
  symbol = "BTC/USDT",
}) => {
  const [buttonState, setButtonState] = useState<ButtonState>("idle");
  const [lastResult, setLastResult] = useState<
    FundingRateCollectionResult | BulkFundingRateCollectionResult | null
  >(null);

  /**
   * FRデータを収集
   */
  const handleCollectData = async () => {
    if (mode === "bulk") {
      // 確認ダイアログ
      const confirmed = window.confirm(
        `BTC・ETHの全期間FRデータを取得します。\n\n` +
          "この処理には数分かかる場合があります。続行しますか？"
      );

      if (!confirmed) {
        return;
      }
    }

    try {
      setButtonState("loading");
      setLastResult(null);

      let apiUrl: string;
      let requestOptions: RequestInit;

      if (mode === "bulk") {
        // 一括収集
        apiUrl = "/api/data/funding-rates/bulk";
        requestOptions = {
          method: "POST",
          headers: {
            "Content-Type": "application/json",
          },
        };
      } else {
        // 単一収集（全期間取得）
        apiUrl = `/api/data/funding-rates/collect?symbol=${encodeURIComponent(
          symbol
        )}&fetch_all=true`;
        requestOptions = {
          method: "POST",
          headers: {
            "Content-Type": "application/json",
          },
        };
      }

      const response = await fetch(apiUrl, requestOptions);
      const result = await response.json();

      if (result.success) {
        setButtonState("success");
        setLastResult(result.data);
        onCollectionStart?.(result.data);

        // 3秒後にアイドル状態に戻す
        setTimeout(() => {
          setButtonState("idle");
        }, 3000);
      } else {
        throw new Error(result.message || "FRデータの収集に失敗しました");
      }
    } catch (error) {
      console.error("FRデータ収集エラー:", error);
      setButtonState("error");
      const errorMessage =
        error instanceof Error
          ? error.message
          : "FRデータの収集中にエラーが発生しました";
      onCollectionError?.(errorMessage);

      // 5秒後にアイドル状態に戻す
      setTimeout(() => {
        setButtonState("idle");
      }, 5000);
    }
  };

  /**
   * ボタンが無効かどうかを判定
   */
  const isButtonDisabled = disabled || buttonState === "loading";

  /**
   * ボタンのテキストを取得
   */
  const getButtonText = () => {
    switch (buttonState) {
      case "loading":
        return mode === "bulk" ? "FR一括収集中..." : "FR収集中...";
      case "success":
        return mode === "bulk" ? "FR一括収集完了" : "FR収集完了";
      case "error":
        return "エラーが発生しました";
      default:
        return mode === "bulk" ? "BTC・ETHFR収集・保存" : "FR収集・保存";
    }
  };

  /**
   * ボタンのアイコンを取得
   */
  const getButtonIcon = () => {
    switch (buttonState) {
      case "loading":
        return (
          <div
            className="animate-spin rounded-full h-4 w-4 border-b-2 border-white"
            data-testid="loading-spinner"
          />
        );
      case "success":
        return (
          <span className="text-green-400" data-testid="success-icon">
            ✅
          </span>
        );
      case "error":
        return (
          <span className="text-red-400" data-testid="error-icon">
            ❌
          </span>
        );
      default:
        return <span className="text-blue-400">📊</span>;
    }
  };

  /**
   * ボタンのCSSクラスを取得
   */
  const getButtonClasses = () => {
    const baseClasses = `
      flex items-center justify-center gap-2 px-4 py-2 rounded-lg font-medium
      transition-all duration-200 focus:outline-none focus:ring-2 focus:ring-offset-2
      focus:ring-offset-gray-900 min-w-[200px]
    `;

    switch (buttonState) {
      case "loading":
        return `${baseClasses} bg-blue-600 text-white cursor-not-allowed opacity-75`;
      case "success":
        return `${baseClasses} bg-green-600 text-white`;
      case "error":
        return `${baseClasses} bg-red-600 text-white`;
      default:
        return `${baseClasses} ${
          isButtonDisabled
            ? "bg-gray-600 text-gray-400 cursor-not-allowed"
            : "bg-primary-600 hover:bg-primary-700 text-white dark:bg-primary-600 dark:hover:bg-primary-700 focus:ring-primary-500"
        }`;
    }
  };

  /**
   * 結果表示を取得
   */
  const getResultDisplay = () => {
    if (!lastResult || buttonState !== "success") return null;

    if (mode === "bulk" && "total_symbols" in lastResult) {
      const bulkResult = lastResult as BulkFundingRateCollectionResult;
      return (
        <div className="text-sm text-secondary-600 dark:text-secondary-400">
          {bulkResult.successful_symbols}/{bulkResult.total_symbols}シンボルで
          合計{bulkResult.total_saved_records}件のデータを保存しました
        </div>
      );
    } else if ("saved_count" in lastResult) {
      const singleResult = lastResult as FundingRateCollectionResult;
      return (
        <div className="text-sm text-secondary-600 dark:text-secondary-400">
          {singleResult.saved_count}件のFRデータを保存しました
        </div>
      );
    }

    return null;
  };

  return (
    <div className={`flex flex-col gap-3 ${className}`}>
      {/* メインボタン */}
      <button
        onClick={handleCollectData}
        disabled={isButtonDisabled}
        className={getButtonClasses()}
        title={
          mode === "bulk"
            ? `BTC・ETHの全期間FRデータを収集・保存`
            : `${symbol}のFRデータを収集・保存`
        }
      >
        {getButtonIcon()}
        <span>{getButtonText()}</span>
      </button>

      {/* 説明テキスト */}
      <div className="text-xs text-secondary-500 dark:text-secondary-500">
        {mode === "bulk" ? (
          <>
            BTC・ETHの全期間FRデータを
            <br />
            取得・保存します
          </>
        ) : (
          <>
            {symbol}のFRデータを
            <br />
            取得・保存します
          </>
        )}
      </div>

      {/* 結果表示 */}
      {getResultDisplay()}
    </div>
  );
};

export default FundingRateCollectionButton;
