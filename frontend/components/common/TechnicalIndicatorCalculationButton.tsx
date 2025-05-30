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
import DataCollectionButton from "./DataCollectionButton";
import type { DataCollectionConfig } from "./DataCollectionButton";

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
  // 設定を作成
  const config: DataCollectionConfig = {
    apiEndpoint:
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
          )}&period=${period}`,
    method: "POST",
    confirmMessage:
      mode === "bulk"
        ? `${symbol} ${timeframe}のデフォルトテクニカル指標を一括計算します。\n\nこの処理には時間がかかる場合があります。続行しますか？`
        : `${symbol} ${timeframe}の${indicatorType}(${period})を計算します。続行しますか？`,
    buttonText: {
      idle:
        mode === "bulk"
          ? "TI一括計算"
          : `${indicatorType}(${period})計算`,
      loading: "計算中...",
      success: "✅ 完了",
      error: "❌ エラー",
    },
    buttonIcon: {
      idle: <span className="text-purple-400">📈</span>,
    },
    description:
      mode === "bulk"
        ? `${symbol} ${timeframe}のデフォルトテクニカル指標を一括計算・保存します`
        : `${symbol} ${timeframe}の${indicatorType}(${period})を計算・保存します`,
    successResetTime: 3000,
    errorResetTime: 5000,
  };

  return (
    <DataCollectionButton
      config={config}
      onCollectionStart={onCalculationStart}
      onCollectionError={onCalculationError}
      disabled={disabled}
      className={className}
    />
  );
};

export default TechnicalIndicatorCalculationButton;
