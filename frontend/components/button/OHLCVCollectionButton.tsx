/**
 * OHLCV収集ボタンコンポーネント
 *
 * OHLCV（Open, High, Low, Close, Volume）データの収集を行う専用ボタンです。
 *
 */

"use client";

import React from "react";
import { useDataCollection } from "@/hooks/useDataCollection";
import ApiButton from "./ApiButton";
import { BulkOHLCVCollectionResult } from "@/types/data-collection";

/**
 * OHLCV収集ボタンコンポーネントのプロパティ
 */
interface OHLCVCollectionButtonProps {
  /** データ収集開始時のコールバック */
  onCollectionStart?: (result: BulkOHLCVCollectionResult) => void;
  /** データ収集エラー時のコールバック */
  onCollectionError?: (error: string) => void;
  /** ボタンの無効化フラグ */
  disabled?: boolean;
  /** カスタムクラス名 */
  className?: string;
}

/**
 * OHLCV収集ボタンコンポーネント
 */
const OHLCVCollectionButton: React.FC<OHLCVCollectionButtonProps> = ({
  onCollectionStart,
  onCollectionError,
  disabled = false,
  className = "",
}) => {
  const dataCollection = useDataCollection();

  const handleClick = async () => {
    await dataCollection.ohlcv.collect(onCollectionStart, onCollectionError);
  };

  return (
    <ApiButton
      onClick={handleClick}
      loading={dataCollection.ohlcv.loading}
      disabled={disabled || dataCollection.isAnyLoading}
      variant="primary"
      size="sm"
      loadingText="収集中..."
      className={className}
    >
      OHLCV収集
    </ApiButton>
  );
};

export default OHLCVCollectionButton;
