/**
 * 全データ一括収集ボタンコンポーネント（共通化版）
 *
 * 共通のDataCollectionButtonコンポーネントを使用した全データ収集ボタンです。
 *
 * @author Trdinger Development Team
 * @version 2.0.0
 */

"use client";

import React from "react";
import { useApiCall } from "@/hooks/useApiCall";
import ApiButton from "./ApiButton";

/**
 * 全データ一括収集ボタンコンポーネントのプロパティ
 */
interface AllDataCollectionButtonProps {
  /** データ収集開始時のコールバック */
  onCollectionStart?: (result: any) => void;
  /** データ収集エラー時のコールバック */
  onCollectionError?: (error: string) => void;
  /** ボタンの無効化フラグ */
  disabled?: boolean;
  /** カスタムクラス名 */
  className?: string;
}

/**
 * 全データ一括収集ボタンコンポーネント（共通化版）
 */
const AllDataCollectionButton: React.FC<AllDataCollectionButtonProps> = ({
  onCollectionStart,
  onCollectionError,
  disabled = false,
  className = "",
}) => {
  const apiCall = useApiCall();

  const handleClick = async () => {
    await apiCall.execute("/api/data/all/bulk-collect", {
      method: "POST",
      confirmMessage:
        "全データ（OHLCV・FR・OI・TI）を一括取得します。\n\n" +
        "この処理には数分から十数分かかる場合があります。\n" +
        "テクニカル指標も自動計算されます。続行しますか？",
      onSuccess: onCollectionStart,
      onError: onCollectionError,
    });
  };

  return (
    <ApiButton
      onClick={handleClick}
      loading={apiCall.loading}
      disabled={disabled}
      variant="primary"
      size="sm"
      loadingText="収集中..."
      className={className}
      icon={
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
      }
    >
      全データ取得
    </ApiButton>
  );
};

export default AllDataCollectionButton;
