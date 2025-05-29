/**
 * FRデータ収集ボタンコンポーネント（共通化版）
 *
 * 共通のDataCollectionButtonコンポーネントを使用したFRデータ収集ボタンです。
 *
 * @author Trdinger Development Team
 * @version 2.0.0
 */

"use client";

import React from "react";
import DataCollectionButton from "./DataCollectionButton";
import type { DataCollectionConfig } from "./DataCollectionButton";

/**
 * FR収集ボタンコンポーネントのプロパティ
 */
interface FundingRateCollectionButtonProps {
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
 * FRデータ収集ボタンコンポーネント（共通化版）
 */
const FundingRateCollectionButton: React.FC<FundingRateCollectionButtonProps> = ({
  onCollectionStart,
  onCollectionError,
  disabled = false,
  className = "",
  mode = "bulk",
  symbol = "BTC/USDT",
}) => {
  // 設定を作成
  const config: DataCollectionConfig = {
    apiEndpoint: mode === "bulk" 
      ? "/api/data/funding-rates/bulk"
      : `/api/data/funding-rates/collect?symbol=${encodeURIComponent(symbol)}&fetch_all=true`,
    method: "POST",
    confirmMessage: mode === "bulk"
      ? "BTC・ETHの全期間FRデータを取得します。\n\nこの処理には数分かかる場合があります。続行しますか？"
      : undefined,
    buttonText: {
      idle: mode === "bulk" ? "BTC・ETHFR収集・保存" : "FR収集・保存",
      loading: "FR収集中...",
      success: "FR収集完了",
      error: "エラーが発生しました",
    },
    buttonIcon: {
      idle: <span className="text-blue-400">📊</span>,
    },
    description: mode === "bulk" 
      ? "BTC・ETHの全期間FRデータを取得・保存します"
      : `${symbol}のFRデータを取得・保存します`,
    successResetTime: 3000,
    errorResetTime: 5000,
  };

  return (
    <DataCollectionButton
      config={config}
      onCollectionStart={onCollectionStart}
      onCollectionError={onCollectionError}
      disabled={disabled}
      className={className}
    />
  );
};

export default FundingRateCollectionButton;
