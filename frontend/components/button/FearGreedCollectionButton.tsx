/**
 * Fear & Greed Index データ収集ボタンコンポーネント
 *
 * Fear & Greed Index データの収集機能を提供します。
 * 他のデータ収集ボタンと統一されたデザインパターンを使用します。
 */

"use client";

import React from "react";
import { usePostRequest } from "@/hooks/usePostRequest";
import ApiButton from "./ApiButton";
import { FearGreedCollectionResult } from "@/hooks/useFearGreedData";

interface FearGreedCollectionButtonProps {
  onCollectionStart?: (result: FearGreedCollectionResult) => void;
  onCollectionError?: (errorMessage: string) => void;
  disabled?: boolean;
  className?: string;
}

const FearGreedCollectionButton: React.FC<FearGreedCollectionButtonProps> = ({
  onCollectionStart,
  onCollectionError,
  disabled = false,
  className = "",
}) => {
  const { sendPostRequest, isLoading } =
    usePostRequest<FearGreedCollectionResult>();

  const handleClick = async () => {
    const { success, data, error } = await sendPostRequest(
      "/api/fear-greed/collect"
    );
    if (success && data) {
      onCollectionStart?.(data);
    } else {
      onCollectionError?.(error || "データ収集に失敗しました");
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
      FG収集
    </ApiButton>
  );
};

export default FearGreedCollectionButton;
