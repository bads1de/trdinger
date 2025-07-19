/**
 * OIデータ収集ボタンコンポーネント
 *
 * 共通のDataCollectionButtonコンポーネントを使用したOIデータ収集ボタンです。
 *
 */

"use client";

import React from "react";
import { usePostRequest } from "@/hooks/usePostRequest";
import ApiButton from "./ApiButton";

/**
 * OI収集ボタンコンポーネントのプロパティ
 */
interface OpenInterestCollectionButtonProps {
  onCollectionStart?: (result: any) => void;
  onCollectionError?: (error: string) => void;
  disabled?: boolean;
  className?: string;
  mode?: "single" | "bulk";
  symbol?: string;
}

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
  const { sendPostRequest, isLoading } = usePostRequest();

  const handleClick = async () => {
    const endpoint =
      mode === "bulk"
        ? "/api/open-interest/bulk-collect"
        : `/api/open-interest/collect?symbol=${encodeURIComponent(
            symbol
          )}&fetch_all=true`;

    const confirmMessage =
      mode === "bulk"
        ? "BTCの全期間OIデータを取得します。\n\nこの処理には数分かかる場合があります。続行しますか？"
        : undefined;

    const executeRequest = async () => {
      const { success, data, error } = await sendPostRequest(endpoint);
      if (success) {
        onCollectionStart?.(data);
      } else {
        onCollectionError?.(error || "データ収集に失敗しました");
      }
    };

    if (confirmMessage) {
      if (window.confirm(confirmMessage)) {
        await executeRequest();
      }
    } else {
      await executeRequest();
    }
  };

  return (
    <ApiButton
      onClick={handleClick}
      loading={isLoading}
      disabled={disabled}
      variant="warning"
      size="sm"
      loadingText="収集中..."
      className={className}
    >
      {mode === "bulk" ? "OI収集" : `OI収集 (${symbol})`}
    </ApiButton>
  );
};

export default OpenInterestCollectionButton;
