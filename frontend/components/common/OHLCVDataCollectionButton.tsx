/**
 * OHLCVデータ収集ボタンコンポーネント（共通化版）
 *
 * 共通のDataCollectionButtonコンポーネントを使用したOHLCVデータ収集ボタンです。
 *
 * @author Trdinger Development Team
 * @version 2.0.0
 */

"use client";

import React from "react";
import DataCollectionButton from "./DataCollectionButton";
import type { DataCollectionConfig } from "./DataCollectionButton";

/**
 * OHLCV収集ボタンコンポーネントのプロパティ
 */
interface OHLCVDataCollectionButtonProps {
  /** データ収集開始時のコールバック */
  onBulkCollectionStart?: (result: any) => void;
  /** データ収集エラー時のコールバック */
  onCollectionError?: (error: string) => void;
  /** ボタンの無効化フラグ */
  disabled?: boolean;
  /** カスタムクラス名 */
  className?: string;
}

/**
 * OHLCVデータ収集ボタンコンポーネント（共通化版）
 */
const OHLCVDataCollectionButton: React.FC<OHLCVDataCollectionButtonProps> = ({
  onBulkCollectionStart,
  onCollectionError,
  disabled = false,
  className = "",
}) => {
  // 設定を作成
  const config: DataCollectionConfig = {
    apiEndpoint: "/api/data/ohlcv/bulk",
    method: "POST",
    confirmMessage: 
      "全ての取引ペアと全ての時間軸でOHLCVデータを取得します。\n" +
      "この処理には時間がかかる場合があります。続行しますか？",
    buttonText: {
      idle: "全データ一括取得・保存",
      loading: "一括取得・保存中...",
      success: "一括取得・保存開始",
      error: "エラーが発生しました",
    },
    buttonIcon: {
      idle: (
        <svg className="w-5 h-5" fill="none" stroke="currentColor" viewBox="0 0 24 24">
          <path
            strokeLinecap="round"
            strokeLinejoin="round"
            strokeWidth={2}
            d="M4 16v1a3 3 0 003 3h10a3 3 0 003-3v-1m-4-8l-4-4m0 0L8 8m4-4v12"
          />
        </svg>
      ),
    },
    description: "全ての取引ペアと時間軸のOHLCVデータを一括取得・保存",
    successResetTime: 10000,
    errorResetTime: 10000,
  };

  return (
    <DataCollectionButton
      config={config}
      onCollectionStart={onBulkCollectionStart}
      onCollectionError={onCollectionError}
      disabled={disabled}
      className={className}
    />
  );
};

export default OHLCVDataCollectionButton;
