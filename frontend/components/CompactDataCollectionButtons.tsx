/**
 * コンパクトデータ収集ボタンコンポーネント
 *
 * データ設定セクション用のコンパクトなデータ収集ボタン群
 */

"use client";

import React from "react";
import {
  BulkOHLCVCollectionResult,
  BulkFundingRateCollectionResult,
  FundingRateCollectionResult,
} from "@/types/strategy";
import { useDataCollection } from "@/hooks/useDataCollection";
import ApiButton from "./common/ApiButton";
import { DownloadIcon, ChartIcon } from "./common/Icons";

interface CompactDataCollectionButtonsProps {
  onBulkCollectionStart?: (result: BulkOHLCVCollectionResult) => void;
  onBulkCollectionError?: (error: string) => void;
  onFundingRateCollectionStart?: (
    result: BulkFundingRateCollectionResult | FundingRateCollectionResult
  ) => void;
  onFundingRateCollectionError?: (error: string) => void;
  disabled?: boolean;
}

const CompactDataCollectionButtons: React.FC<
  CompactDataCollectionButtonsProps
> = ({
  onBulkCollectionStart,
  onBulkCollectionError,
  onFundingRateCollectionStart,
  onFundingRateCollectionError,
  disabled = false,
}) => {
  const dataCollection = useDataCollection();

  const handleOHLCVCollection = async () => {
    await dataCollection.ohlcv.collect(
      onBulkCollectionStart,
      onBulkCollectionError
    );
  };

  const handleFundingRateCollection = async () => {
    await dataCollection.fundingRate.collect(
      onFundingRateCollectionStart,
      onFundingRateCollectionError
    );
  };

  return (
    <div className="flex gap-2">
      {/* OHLCV一括収集ボタン */}
      <ApiButton
        onClick={handleOHLCVCollection}
        loading={dataCollection.ohlcv.loading}
        disabled={disabled || dataCollection.isAnyLoading}
        variant="primary"
        size="sm"
        loadingText="収集中..."
        className="bg-blue-600 hover:bg-blue-700 focus:ring-blue-500"
        icon={<DownloadIcon />}
      >
        OHLCV収集
      </ApiButton>

      {/* FR収集ボタン */}
      <ApiButton
        onClick={handleFundingRateCollection}
        loading={dataCollection.fundingRate.loading}
        disabled={disabled || dataCollection.isAnyLoading}
        variant="success"
        size="sm"
        loadingText="収集中..."
        className="bg-green-600 hover:bg-green-700 focus:ring-green-500"
        icon={<ChartIcon />}
      >
        FR収集
      </ApiButton>
    </div>
  );
};

export default CompactDataCollectionButtons;
