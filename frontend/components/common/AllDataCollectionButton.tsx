/**
 * 全データ一括収集ボタンコンポーネント（共通化版）
 *
 * 共通のDataCollectionButtonコンポーネントを使用した全データ収集ボタンです。
 *
 * @author Trdinger Development Team
 * @version 2.0.0
 */

"use client";

import React from "react";
import DataCollectionButton from "./DataCollectionButton";
import type { DataCollectionConfig } from "./DataCollectionButton";

/**
 * 全データ一括収集ボタンコンポーネントのプロパティ
 */
interface AllDataCollectionButtonProps {
  /** データ収集開始時のコールバック */
  onCollectionStart?: (result: any) => void;
  /** データ収集エラー時のコールバック */
  onCollectionError?: (error: string) => void;
  /** ボタンの無効化フラグ */
  disabled?: boolean;
  /** カスタムクラス名 */
  className?: string;
}

/**
 * 全データ一括収集ボタンコンポーネント（共通化版）
 */
const AllDataCollectionButton: React.FC<AllDataCollectionButtonProps> = ({
  onCollectionStart,
  onCollectionError,
  disabled = false,
  className = "",
}) => {
  // 設定を作成
  const config: DataCollectionConfig = {
    apiEndpoint: "/api/data/all/bulk-collect",
    method: "POST",
    confirmMessage:
      "全データ（OHLCV・FR・OI）を一括取得します。\n\n" +
      "この処理には数分から十数分かかる場合があります。続行しますか？",
    buttonText: {
      idle: "全データ取得",
      loading: "収集中...",
      success: "✅ 完了",
      error: "❌ エラー",
    },
    buttonIcon: {
      idle: (
        <svg
          className="w-4 h-4"
          fill="none"
          stroke="currentColor"
          viewBox="0 0 24 24"
        >
          <path
            strokeLinecap="round"
            strokeLinejoin="round"
            strokeWidth={2}
            d="M4 7v10c0 2.21 1.79 4 4 4h8c2.21 0 4-1.79 4-4V7M4 7l8-4 8 4M4 7l8 4 8-4"
          />
        </svg>
      ),
    },
    description: "OHLCV・FR・OIの全データを一括収集",
    successResetTime: 10000,
    errorResetTime: 10000,
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

export default AllDataCollectionButton;
