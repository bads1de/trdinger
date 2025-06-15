/**
 * データリセットボタンコンポーネント
 *
 * OHLCV、ファンディングレート、オープンインタレストデータのリセット機能を提供します。
 *
 */

"use client";

import React, { useState } from "react";
import { useApiCall } from "@/hooks/useApiCall";
import ApiButton from "./ApiButton";

/**
 * データリセットの種類
 */
export type ResetType =
  | "all"
  | "ohlcv"
  | "funding-rates"
  | "open-interest"
  | "symbol";

/**
 * データリセット結果の型
 */
export interface DataResetResult {
  success: boolean;
  deleted_counts?: {
    ohlcv: number;
    funding_rates: number;
    open_interest: number;
  };
  deleted_count?: number;
  total_deleted?: number;
  data_type?: string;
  symbol?: string;
  message: string;
  errors?: string[];
  timestamp: string;
}

/**
 * データリセットボタンコンポーネントのプロパティ
 */
interface DataResetButtonProps {
  /** リセットの種類 */
  resetType: ResetType;
  /** 特定シンボルリセット時のシンボル名 */
  symbol?: string;
  /** リセット完了時のコールバック */
  onResetComplete?: (result: DataResetResult) => void;
  /** リセットエラー時のコールバック */
  onResetError?: (error: string) => void;
  /** ボタンの無効化フラグ */
  disabled?: boolean;
  /** カスタムクラス名 */
  className?: string;
  /** ボタンサイズ */
  size?: "xs" | "sm" | "md" | "lg" | "xl";
  /** ボタンバリアント */
  variant?:
    | "primary"
    | "secondary"
    | "success"
    | "warning"
    | "error"
    | "outline"
    | "ghost";
}

/**
 * リセット種類に応じた設定
 */
const RESET_CONFIGS = {
  all: {
    label: "全データリセット",
    endpoint: "/api/data-reset/all",
    confirmMessage:
      "⚠️ 全てのデータ（OHLCV・ファンディングレート・オープンインタレスト）を削除します。\n\n" +
      "この操作は取り消すことができません。\n" +
      "本当に実行しますか？",
    variant: "error" as const,
    icon: "🗑️",
  },
  ohlcv: {
    label: "OHLCVリセット",
    endpoint: "/api/data-reset/ohlcv",
    confirmMessage:
      "⚠️ 全てのOHLCVデータを削除します。\n\n" +
      "この操作は取り消すことができません。\n" +
      "本当に実行しますか？",
    variant: "warning" as const,
    icon: "📊",
  },
  "funding-rates": {
    label: "FRリセット",
    endpoint: "/api/data-reset/funding-rates",
    confirmMessage:
      "⚠️ 全てのファンディングレートデータを削除します。\n\n" +
      "この操作は取り消すことができません。\n" +
      "本当に実行しますか？",
    variant: "warning" as const,
    icon: "💰",
  },
  "open-interest": {
    label: "OIリセット",
    endpoint: "/api/data-reset/open-interest",
    confirmMessage:
      "⚠️ 全てのオープンインタレストデータを削除します。\n\n" +
      "この操作は取り消すことができません。\n" +
      "本当に実行しますか？",
    variant: "warning" as const,
    icon: "📈",
  },
  symbol: {
    label: "シンボル別リセット",
    endpoint: "/api/data-reset/symbol",
    confirmMessage:
      "⚠️ 指定されたシンボルの全データを削除します。\n\n" +
      "この操作は取り消すことができません。\n" +
      "本当に実行しますか？",
    variant: "warning" as const,
    icon: "🎯",
  },
};

/**
 * データリセットボタンコンポーネント
 */
const DataResetButton: React.FC<DataResetButtonProps> = ({
  resetType,
  symbol,
  onResetComplete,
  onResetError,
  disabled = false,
  className = "",
  size = "sm",
  variant,
}) => {
  const apiCall = useApiCall<DataResetResult>();
  const [isProcessing, setIsProcessing] = useState(false);

  const config = RESET_CONFIGS[resetType];
  const buttonVariant = variant || config.variant;

  const handleReset = async () => {
    if (isProcessing) return;

    try {
      setIsProcessing(true);

      // エンドポイントURLを構築
      let endpoint = config.endpoint;
      let confirmMessage = config.confirmMessage;

      if (resetType === "symbol" && symbol) {
        endpoint = `${endpoint}/${encodeURIComponent(symbol)}`;
        confirmMessage = confirmMessage.replace(
          "指定されたシンボル",
          `シンボル「${symbol}」`
        );
      }

      // 確認メッセージをカスタマイズ
      if (resetType === "symbol" && symbol) {
        confirmMessage =
          `⚠️ シンボル「${symbol}」の全データ（OHLCV・ファンディングレート・オープンインタレスト）を削除します。\n\n` +
          "この操作は取り消すことができません。\n" +
          "本当に実行しますか？";
      }

      const result = await apiCall.execute(endpoint, {
        method: "DELETE",
        confirmMessage,
        onSuccess: (data: DataResetResult) => {
          console.log("データリセット完了:", data);
          onResetComplete?.(data);
        },
        onError: (error: string) => {
          console.error("データリセットエラー:", error);
          onResetError?.(error);
        },
      });

      if (result) {
        console.log("データリセット結果:", result);
      }
    } catch (error) {
      console.error("データリセット処理エラー:", error);
      const errorMessage =
        error instanceof Error
          ? error.message
          : "データリセット中にエラーが発生しました";
      onResetError?.(errorMessage);
    } finally {
      setIsProcessing(false);
    }
  };

  // ボタンラベルを動的に生成
  const getButtonLabel = () => {
    if (resetType === "symbol" && symbol) {
      return `${symbol} リセット`;
    }
    return config.label;
  };

  return (
    <ApiButton
      onClick={handleReset}
      loading={apiCall.loading || isProcessing}
      disabled={disabled || isProcessing}
      variant={buttonVariant}
      size={size}
      loadingText="削除中..."
      className={className}
      icon={<span>{config.icon}</span>}
    >
      {getButtonLabel()}
    </ApiButton>
  );
};

export default DataResetButton;
