/**
 * OIデータ収集ボタンコンポーネント（共通化版）
 *
 * 共通のDataCollectionButtonコンポーネントを使用したOIデータ収集ボタンです。
 *
 * @author Trdinger Development Team
 * @version 2.0.0
 */

"use client";

import React from "react";
import { useApiCall } from "@/hooks/useApiCall";
import ApiButton from "./ApiButton";

/**
 * OI収集ボタンコンポーネントのプロパティ
 */
interface OpenInterestCollectionButtonProps {
  /** データ収集開始時のコールバック */
  onCollectionStart?: (result: any) => void;
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
 * OIデータ収集ボタンコンポーネント（共通化版）
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
  const apiCall = useApiCall();

  const handleClick = async () => {
    const endpoint = mode === "bulk"
      ? "/api/data/open-interest/bulk-collect"
      : `/api/data/open-interest/collect?symbol=${encodeURIComponent(symbol)}&fetch_all=true`;

    await apiCall.execute(endpoint, {
      method: "POST",
      confirmMessage: mode === "bulk"
        ? "BTC・ETHの全期間OIデータを取得します。\n\nこの処理には数分かかる場合があります。続行しますか？"
        : undefined,
      onSuccess: onCollectionStart,
      onError: onCollectionError,
    });
  };

  return (
    <ApiButton
      onClick={handleClick}
      loading={apiCall.loading}
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
