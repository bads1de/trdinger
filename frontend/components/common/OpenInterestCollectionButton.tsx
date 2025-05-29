/**
 * OIデータ収集ボタンコンポーネント（共通化版）
 *
 * 共通のDataCollectionButtonコンポーネントを使用したOIデータ収集ボタンです。
 *
 * @author Trdinger Development Team
 * @version 2.0.0
 */

"use client";

import React from "react";
import DataCollectionButton from "./DataCollectionButton";
import type { DataCollectionConfig } from "./DataCollectionButton";

/**
 * OI収集ボタンコンポーネントのプロパティ
 */
interface OpenInterestCollectionButtonProps {
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
 * OIデータ収集ボタンコンポーネント（共通化版）
 */
const OpenInterestCollectionButton: React.FC<
  OpenInterestCollectionButtonProps
> = ({
  onCollectionStart,
  onCollectionError,
  disabled = false,
  className = "",
  mode = "bulk",
  symbol = "BTC/USDT",
}) => {
  // 設定を作成
  const config: DataCollectionConfig = {
    apiEndpoint:
      mode === "bulk"
        ? "/api/data/open-interest/bulk-collect"
        : `/api/data/open-interest/collect?symbol=${encodeURIComponent(
            symbol
          )}&fetch_all=true`,
    method: "POST",
    confirmMessage:
      mode === "bulk"
        ? "BTC・ETHの全期間OIデータを取得します。\n\nこの処理には数分かかる場合があります。続行しますか？"
        : undefined,
    buttonText: {
      idle: mode === "bulk" ? "OI収集" : `OI収集 (${symbol})`,
      loading: "収集中...",
      success: "✅ 完了",
      error: "❌ エラー",
    },
    description:
      mode === "bulk"
        ? "BTC・ETHの全期間OIデータを一括収集"
        : `${symbol}のOIデータを収集`,
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

export default OpenInterestCollectionButton;
