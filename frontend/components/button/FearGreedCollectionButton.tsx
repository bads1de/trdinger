/**
 * Fear & Greed Index データ収集ボタンコンポーネント
 *
 * Fear & Greed Index データの収集機能を提供します。
 * 他のデータ収集ボタンと統一されたデザインパターンを使用します。
 */

"use client";

import React from "react";
import { useApiCall } from "@/hooks/useApiCall";
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
  const apiCall = useApiCall();

  const handleClick = async () => {
    await apiCall.execute("/api/data/fear-greed/collect-incremental", {
      method: "POST",
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
      FG収集
    </ApiButton>
  );
};

export default FearGreedCollectionButton;
