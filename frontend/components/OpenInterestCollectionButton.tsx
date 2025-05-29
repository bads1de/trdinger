/**
 * OIデータ収集ボタンコンポーネント
 *
 * OIデータを取得し、データベースに保存するためのUIコンポーネントです。
 * ダークモード対応、状態管理、エラーハンドリング、確認ダイアログを含みます。
 *
 * @author Trdinger Development Team
 * @version 1.0.0
 */

"use client";

import React, { useState } from "react";
import {
  OpenInterestCollectionResult,
  BulkOpenInterestCollectionResult,
} from "@/types/strategy";

/**
 * OI収集ボタンコンポーネントのプロパティ
 */
interface OpenInterestCollectionButtonProps {
  /** データ収集開始時のコールバック */
  onCollectionStart?: (
    result: OpenInterestCollectionResult | BulkOpenInterestCollectionResult
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
 * OIデータ収集ボタンコンポーネント
 */
const OpenInterestCollectionButton: React.FC<
  OpenInterestCollectionButtonProps
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
    OpenInterestCollectionResult | BulkOpenInterestCollectionResult | null
  >(null);

  /**
   * OIデータを収集
   */
  const handleCollectData = async () => {
    if (mode === "bulk") {
      // 確認ダイアログ
      const confirmed = window.confirm(
        `BTC・ETHの全期間OIデータを取得します。\n\n` +
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
      if (mode === "bulk") {
        apiUrl = "/api/data/open-interest/bulk-collect";
      } else {
        apiUrl = `/api/data/open-interest/collect?symbol=${encodeURIComponent(
          symbol
        )}&fetch_all=true`;
      }

      console.log(`OIデータ収集開始: ${apiUrl}`);

      const response = await fetch(apiUrl, {
        method: "POST",
        headers: {
          "Content-Type": "application/json",
        },
      });

      const result = await response.json();

      if (result.success) {
        setButtonState("success");
        setLastResult(result.data);
        onCollectionStart?.(result.data);

        // 成功メッセージを表示
        if (mode === "bulk") {
          const bulkResult = result.data as BulkOpenInterestCollectionResult;
          alert(
            `✅ 一括収集完了\n\n` +
              `成功: ${bulkResult.successful_symbols}/${bulkResult.total_symbols}シンボル\n` +
              `保存: ${bulkResult.total_saved_records}件`
          );
        } else {
          const singleResult = result.data as OpenInterestCollectionResult;
          alert(
            `✅ 収集完了\n\n` +
              `シンボル: ${singleResult.symbol}\n` +
              `保存: ${singleResult.saved_count}件`
          );
        }

        // 3秒後に状態をリセット
        setTimeout(() => {
          setButtonState("idle");
        }, 3000);
      } else {
        throw new Error(result.message || "データ収集に失敗しました");
      }
    } catch (error) {
      console.error("OIデータ収集エラー:", error);
      setButtonState("error");
      const errorMessage =
        error instanceof Error ? error.message : "不明なエラーが発生しました";
      onCollectionError?.(errorMessage);

      // エラーメッセージを表示
      alert(`❌ 収集エラー\n\n${errorMessage}`);

      // 5秒後に状態をリセット
      setTimeout(() => {
        setButtonState("idle");
      }, 5000);
    }
  };

  /**
   * ボタンのスタイルを取得
   */
  const getButtonStyle = () => {
    const baseStyle =
      "px-6 py-3 rounded-lg font-semibold text-sm transition-all duration-200 flex items-center gap-2 min-w-[200px] justify-center";

    switch (buttonState) {
      case "loading":
        return `${baseStyle} bg-blue-600 text-white cursor-not-allowed`;
      case "success":
        return `${baseStyle} bg-green-600 text-white`;
      case "error":
        return `${baseStyle} bg-red-600 text-white`;
      default:
        return `${baseStyle} bg-blue-500 hover:bg-blue-600 text-white hover:shadow-lg disabled:bg-gray-600 disabled:cursor-not-allowed`;
    }
  };

  /**
   * ボタンのテキストを取得
   */
  const getButtonText = () => {
    switch (buttonState) {
      case "loading":
        return mode === "bulk" ? "一括収集中..." : "収集中...";
      case "success":
        return "✅ 完了";
      case "error":
        return "❌ エラー";
      default:
        return mode === "bulk"
          ? "📈 OI一括収集 (BTC・ETH)"
          : `📈 OI収集 (${symbol})`;
    }
  };

  /**
   * ローディングアイコンを取得
   */
  const getLoadingIcon = () => {
    if (buttonState === "loading") {
      return (
        <div className="animate-spin rounded-full h-4 w-4 border-b-2 border-white"></div>
      );
    }
    return null;
  };

  return (
    <button
      onClick={handleCollectData}
      disabled={disabled || buttonState === "loading"}
      className={`${getButtonStyle()} ${className}`}
      title={
        mode === "bulk"
          ? "BTC・ETHの全期間OIデータを一括収集"
          : `${symbol}のOIデータを収集`
      }
    >
      {getLoadingIcon()}
      {getButtonText()}
    </button>
  );
};

export default OpenInterestCollectionButton;
