/**
 * テクニカル指標計算ボタンコンポーネント（共通化版）
 *
 * 共通のDataCollectionButtonコンポーネントを使用したテクニカル指標計算ボタンです。
 *
 * @author Trdinger Development Team
 * @version 1.0.0
 */

"use client";

import React from "react";
import { useApiCall } from "@/hooks/useApiCall";
import ApiButton from "../common/ApiButton";

/**
 * テクニカル指標計算ボタンコンポーネントのプロパティ
 */
interface TechnicalIndicatorCalculationButtonProps {
  /** データ計算開始時のコールバック */
  onCalculationStart?: (result: any) => void;
  /** データ計算エラー時のコールバック */
  onCalculationError?: (error: string) => void;
  /** ボタンの無効化フラグ */
  disabled?: boolean;
  /** カスタムクラス名 */
  className?: string;
  /** 計算モード */
  mode?: "single" | "bulk";
  /** 単一計算時のシンボル */
  symbol?: string;
  /** 単一計算時の時間枠 */
  timeframe?: string;
  /** 単一計算時の指標タイプ */
  indicatorType?: string;
  /** 単一計算時の期間 */
  period?: number;
}

/**
 * テクニカル指標計算ボタンコンポーネント（共通化版）
 */
const TechnicalIndicatorCalculationButton: React.FC<
  TechnicalIndicatorCalculationButtonProps
> = ({
  onCalculationStart,
  onCalculationError,
  disabled = false,
  className = "",
  mode = "bulk",
  symbol = "BTC/USDT",
  timeframe = "1h",
  indicatorType = "SMA",
  period = 20,
}) => {
  const apiCall = useApiCall();

  const handleClick = async () => {
    const endpoint =
      mode === "bulk"
        ? `/api/data/technical-indicators/bulk-calculate?symbol=${encodeURIComponent(
            symbol
          )}&timeframe=${encodeURIComponent(timeframe)}`
        : `/api/data/technical-indicators/calculate?symbol=${encodeURIComponent(
            symbol
          )}&timeframe=${encodeURIComponent(
            timeframe
          )}&indicator_type=${encodeURIComponent(
            indicatorType
          )}&period=${period}`;

    await apiCall.execute(endpoint, {
      method: "POST",
      confirmMessage:
        mode === "bulk"
          ? `${symbol} ${timeframe}のデフォルトテクニカル指標を一括計算します。\n\nこの処理には時間がかかる場合があります。続行しますか？`
          : `${symbol} ${timeframe}の${indicatorType}(${period})を計算します。続行しますか？`,
      onSuccess: onCalculationStart,
      onError: onCalculationError,
    });
  };

  return (
    <ApiButton
      onClick={handleClick}
      loading={apiCall.loading}
      disabled={disabled}
      variant="secondary"
      size="sm"
      loadingText="計算中..."
      className={className}
    >
      {mode === "bulk" ? "TI一括計算" : `${indicatorType}(${period})計算`}
    </ApiButton>
  );
};

export default TechnicalIndicatorCalculationButton;
