/**
 * 全データ一括収集ボタンコンポーネント（共通化版）
 *
 * 共通のDataCollectionButtonコンポーネントを使用した全データ収集ボタンです。
 *
 */

"use client";

import React from "react";
import { usePostRequest } from "@/hooks/usePostRequest";
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
  const { sendPostRequest, isLoading } = usePostRequest();

  const handleClick = async () => {
    const confirmMessage =
      "全データ（OHLCV・FR・OI）を一括取得します。\n\n" +
      "この処理には数分から十数分かかる場合があります。\n" +
      "テクニカル指標も自動計算されます。続行しますか？";

    if (window.confirm(confirmMessage)) {
      const { success, data, error } = await sendPostRequest(
        "/api/data-collection/all/bulk-collect"
      );
      if (success) {
        onCollectionStart?.(data);
      } else {
        onCollectionError?.(error || "データ収集に失敗しました");
      }
    }
  };

  return (
    <ApiButton
      onClick={handleClick}
      loading={isLoading}
      disabled={disabled}
      variant="primary"
      size="sm"
      loadingText="収集中..."
      className={className}
    >
      全データ取得
    </ApiButton>
  );
};

export default AllDataCollectionButton;
