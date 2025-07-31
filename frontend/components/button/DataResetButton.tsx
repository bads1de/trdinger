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
import { RESET_CONFIGS } from "@/constants/data-reset-constants";

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
  size?: "sm" | "md" | "lg";
  /** ボタンバリアント */
  variant?: "primary" | "secondary" | "success" | "warning" | "danger";
}

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
  const { execute, loading: isLoading } = useApiCall<DataResetResult>();

  const config = RESET_CONFIGS[resetType];
  const buttonVariant = variant || config.variant;

  const handleReset = async () => {
    if (isLoading) return;

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

    await execute(endpoint, {
      method: "DELETE",
      confirmMessage,
      onSuccess: (data) => {
        onResetComplete?.(data);
      },
      onError: (error) => {
        console.error("データリセットエラー:", error);
        onResetError?.(error || "データリセット中にエラーが発生しました");
      },
    });
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
      loading={isLoading}
      disabled={disabled || isLoading}
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
