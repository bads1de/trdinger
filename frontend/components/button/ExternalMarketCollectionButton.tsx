/**
 * 外部市場データ収集ボタンコンポーネント
 *
 * 外部市場データ（SP500、NASDAQ、DXY、VIX）の収集を実行するボタンです。
 */

import React from "react";
import { useApiCall } from "@/hooks/useApiCall";
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
> = ({ onCollectionStart, onCollectionError, disabled = false, className = "" }) => {
  const apiCall = useApiCall<ExternalMarketCollectionResult>();

  const handleClick = async () => {
    await apiCall.execute("/api/data/external-market/collect", {
      method: "POST",
      onSuccess: onCollectionStart,
      onError: onCollectionError,
    });
  };

  const buttonText = "外部市場収集";
  const title = "外部市場データ（SP500、NASDAQ、DXY、VIX）を収集します";

  return (
    <ApiButton
      onClick={handleClick}
      loading={apiCall.loading}
      disabled={disabled}
      variant="secondary"
      size="sm"
      loadingText="収集中..."
      className={className}
      title={title}
    >
      {buttonText}
    </ApiButton>
  );
};

export default ExternalMarketCollectionButton;
