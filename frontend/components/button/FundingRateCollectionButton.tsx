/**
 * Funding Rate収集ボタンコンポーネント
 *
 * Funding Rate（資金調達率）データの収集を行う専用ボタンです。
 *
 */

"use client";

import React from "react";
import { useDataCollection } from "@/hooks/useDataCollection";
import ApiButton from "./ApiButton";
import {
  BulkFundingRateCollectionResult,
  FundingRateCollectionResult,
} from "@/types/strategy";

/**
 * Funding Rate収集ボタンコンポーネントのプロパティ
 */
interface FundingRateCollectionButtonProps {
  /** データ収集開始時のコールバック */
  onCollectionStart?: (
    result: BulkFundingRateCollectionResult | FundingRateCollectionResult
  ) => void;
  /** データ収集エラー時のコールバック */
  onCollectionError?: (error: string) => void;
  /** ボタンの無効化フラグ */
  disabled?: boolean;
  /** カスタムクラス名 */
  className?: string;
}

/**
 * Funding Rate収集ボタンコンポーネント
 */
const FundingRateCollectionButton: React.FC<
  FundingRateCollectionButtonProps
> = ({
  onCollectionStart,
  onCollectionError,
  disabled = false,
  className = "",
}) => {
  const dataCollection = useDataCollection();

  const handleClick = async () => {
    await dataCollection.fundingRate.collect(
      onCollectionStart,
      onCollectionError
    );
  };

  return (
    <ApiButton
      onClick={handleClick}
      loading={dataCollection.fundingRate.loading}
      disabled={disabled || dataCollection.isAnyLoading}
      variant="success"
      size="sm"
      loadingText="収集中..."
      className={className}
    >
      FR収集
    </ApiButton>
  );
};

export default FundingRateCollectionButton;
