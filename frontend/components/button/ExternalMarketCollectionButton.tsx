/**
 * 外部市場データ収集ボタンコンポーネント
 *
 * 外部市場データ（SP500、NASDAQ、DXY、VIX）の収集を実行するボタンです。
 */

import React from "react";
import { usePostRequest } from "@/hooks/usePostRequest";
import ApiButton from "./ApiButton";
import { ExternalMarketCollectionResult } from "@/hooks/useExternalMarketData";

interface ExternalMarketCollectionButtonProps {
  onCollectionStart: (result: ExternalMarketCollectionResult) => void;
  onCollectionError: (errorMessage: string) => void;
  disabled?: boolean;
  className?: string;
}

/**
 * 外部市場データ収集ボタンコンポーネント
 */
const ExternalMarketCollectionButton: React.FC<
  ExternalMarketCollectionButtonProps
> = ({
  onCollectionStart,
  onCollectionError,
  disabled = false,
  className = "",
}) => {
  const { sendPostRequest, isLoading } =
    usePostRequest<ExternalMarketCollectionResult>();

  const handleClick = async () => {
    const { success, data, error } = await sendPostRequest(
      "/api/external-market/collect"
    );
    if (success && data) {
      onCollectionStart(data);
    } else {
      onCollectionError(error || "データ収集に失敗しました");
    }
  };

  const title = "外部市場データ（SP500、NASDAQ、DXY、VIX）を収集します";

  return (
    <ApiButton
      onClick={handleClick}
      loading={isLoading}
      disabled={disabled}
      variant="secondary"
      size="sm"
      loadingText="収集中..."
      className={className}
      title={title}
    >
      外部市場収集
    </ApiButton>
  );
};

export default ExternalMarketCollectionButton;
