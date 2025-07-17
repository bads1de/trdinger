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
  /** 収集モード: 'incremental' = 差分収集, 'historical' = 履歴収集 */
  mode?: "incremental" | "historical";
  /** 履歴収集時の開始日（YYYY-MM-DD形式、デフォルト: 2020-01-01） */
  startDate?: string;
  /** 履歴収集時の終了日（YYYY-MM-DD形式、デフォルト: 今日） */
  endDate?: string;
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
  mode = "incremental",
  startDate = "2020-01-01",
  endDate,
}) => {
  const apiCall = useApiCall<ExternalMarketCollectionResult>();

  const handleClick = async () => {
    if (mode === "historical") {
      // 履歴データ収集
      const finalEndDate = endDate || new Date().toISOString().split("T")[0];
      const url = new URL(
        "/api/data/external-market/collect-historical",
        window.location.origin
      );
      url.searchParams.set("start_date", startDate);
      url.searchParams.set("end_date", finalEndDate);

      await apiCall.execute(url.toString(), {
        method: "POST",
        onSuccess: onCollectionStart,
        onError: onCollectionError,
      });
    } else {
      // 差分収集
      await apiCall.execute("/api/data/external-market/collect-incremental", {
        method: "POST",
        onSuccess: onCollectionStart,
        onError: onCollectionError,
      });
    }
  };

  // ボタンテキストとタイトルを生成
  const getButtonText = () => {
    if (mode === "historical") {
      const finalEndDate = endDate || "今日";
      return `履歴データ収集 (${startDate} ～ ${finalEndDate})`;
    }
    return "外部市場収集";
  };

  const getTitle = () => {
    if (mode === "historical") {
      const finalEndDate = endDate || "今日";
      return `外部市場データ（SP500、NASDAQ、DXY、VIX）の履歴データを収集します\n期間: ${startDate} ～ ${finalEndDate}`;
    }
    return "外部市場データ（SP500、NASDAQ、DXY、VIX）を収集します";
  };

  return (
    <ApiButton
      onClick={handleClick}
      loading={apiCall.loading}
      disabled={disabled}
      variant={mode === "historical" ? "primary" : "secondary"}
      size="sm"
      loadingText={mode === "historical" ? "履歴収集中..." : "収集中..."}
      className={className}
      title={getTitle()}
    >
      {getButtonText()}
    </ApiButton>
  );
};

export default ExternalMarketCollectionButton;
