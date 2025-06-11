import React from "react";
import SymbolSelector from "@/components/common/SymbolSelector";
import TimeFrameSelector from "@/components/common/TimeFrameSelector";
import AllDataCollectionButton from "@/components/button/AllDataCollectionButton";
import OHLCVCollectionButton from "@/components/button/OHLCVCollectionButton";
import FundingRateCollectionButton from "@/components/button/FundingRateCollectionButton";

import {
  TradingPair,
  TimeFrame,
  AllDataCollectionResult,
  BulkOHLCVCollectionResult,
  BulkFundingRateCollectionResult,
  FundingRateCollectionResult,
  OpenInterestCollectionResult,
  BulkOpenInterestCollectionResult,
} from "@/types/strategy";
import OpenInterestCollectionButton from "@/components/button/OpenInterestCollectionButton";

interface DataControlsProps {
  symbols: TradingPair[];
  selectedSymbol: string;
  handleSymbolChange: (symbol: string) => void;
  symbolsLoading: boolean;
  loading: boolean;
  selectedTimeFrame: TimeFrame;
  handleTimeFrameChange: (timeFrame: TimeFrame) => void;
  updating: boolean;
  handleAllDataCollectionStart: (result: AllDataCollectionResult) => void;
  handleAllDataCollectionError: (errorMessage: string) => void;
  handleBulkCollectionStart: (result: BulkOHLCVCollectionResult) => void;
  handleBulkCollectionError: (errorMessage: string) => void;
  handleFundingRateCollectionStart: (
    result: BulkFundingRateCollectionResult | FundingRateCollectionResult
  ) => void;
  handleFundingRateCollectionError: (errorMessage: string) => void;
  handleOpenInterestCollectionStart: (
    result: BulkOpenInterestCollectionResult | OpenInterestCollectionResult
  ) => void;
  handleOpenInterestCollectionError: (errorMessage: string) => void;
  bulkCollectionMessage: string;
  fundingRateCollectionMessage: string;
  openInterestCollectionMessage: string;
  allDataCollectionMessage: string;
  dataStatus: any; // TODO: より具体的な型を指定する
}

const DataControls: React.FC<DataControlsProps> = ({
  symbols,
  selectedSymbol,
  handleSymbolChange,
  symbolsLoading,
  loading,
  selectedTimeFrame,
  handleTimeFrameChange,
  updating,
  handleAllDataCollectionStart,
  handleAllDataCollectionError,
  handleBulkCollectionStart,
  handleBulkCollectionError,
  handleFundingRateCollectionStart,
  handleFundingRateCollectionError,
  handleOpenInterestCollectionStart,
  handleOpenInterestCollectionError,
  bulkCollectionMessage,
  fundingRateCollectionMessage,
  openInterestCollectionMessage,
  allDataCollectionMessage,
  dataStatus,
}) => {
  return (
    <div className="enterprise-card animate-slide-up">
      <div className="p-6">
        {/* データベース状況セクション */}
        {dataStatus && (
          <div className="mb-6">
            <div className="flex items-center justify-between mb-4">
              <h2 className="text-xl font-semibold text-secondary-900 dark:text-secondary-100">
                📊 データベース状況
              </h2>
              <span className="badge-primary">
                {dataStatus.data_count?.toLocaleString()}件
              </span>
            </div>
            <div className="grid grid-cols-1 md:grid-cols-3 gap-4 text-sm">
              <div className="flex justify-between">
                <span className="text-secondary-600 dark:text-secondary-400">
                  データ件数:
                </span>
                <span className="font-medium text-secondary-900 dark:text-secondary-100">
                  {dataStatus.data_count?.toLocaleString()}件
                </span>
              </div>
              <div className="flex justify-between">
                <span className="text-secondary-600 dark:text-secondary-400">
                  最新データ:
                </span>
                <span className="font-medium text-secondary-900 dark:text-secondary-100">
                  {dataStatus.latest_timestamp
                    ? new Date(dataStatus.latest_timestamp).toLocaleString(
                        "ja-JP"
                      )
                    : "なし"}
                </span>
              </div>
              <div className="flex justify-between">
                <span className="text-secondary-600 dark:text-secondary-400">
                  最古データ:
                </span>
                <span className="font-medium text-secondary-900 dark:text-secondary-100">
                  {dataStatus.oldest_timestamp
                    ? new Date(dataStatus.oldest_timestamp).toLocaleString(
                        "ja-JP"
                      )
                    : "なし"}
                </span>
              </div>
            </div>
          </div>
        )}

        {/* データ設定セクション */}
        <div className="flex items-center justify-between mb-6">
          <h2 className="text-lg font-semibold text-secondary-900 dark:text-secondary-100">
            📈 データ設定
          </h2>
        </div>

        {/* 設定コントロール */}
        <div className="space-y-6">
          {/* 上段：基本設定 */}
          <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
            {/* 通貨ペア選択 */}
            <SymbolSelector
              symbols={symbols}
              selectedSymbol={selectedSymbol}
              onSymbolChange={handleSymbolChange}
              loading={symbolsLoading}
              disabled={loading}
              mode="compact"
              showCategories={false}
              enableSearch={false}
            />

            {/* 時間軸選択 */}
            <TimeFrameSelector
              selectedTimeFrame={selectedTimeFrame}
              onTimeFrameChange={handleTimeFrameChange}
              disabled={loading}
              mode="compact"
            />
          </div>

          {/* 下段：データ収集ボタン */}
          <div className="space-y-3">
            <label className="block text-sm font-medium text-secondary-600 dark:text-secondary-400">
              データ収集
            </label>
            <div className="grid grid-cols-1 sm:grid-cols-2 lg:grid-cols-5 gap-3">
              {/* 全データ一括収集ボタン */}
              <AllDataCollectionButton
                onCollectionStart={handleAllDataCollectionStart}
                onCollectionError={handleAllDataCollectionError}
                disabled={loading || updating}
                className="h-10 text-sm"
              />

              {/* OHLCV収集ボタン */}
              <OHLCVCollectionButton
                onCollectionStart={handleBulkCollectionStart}
                onCollectionError={handleBulkCollectionError}
                disabled={loading || updating}
                className="h-10"
              />

              {/* FR収集ボタン */}
              <FundingRateCollectionButton
                onCollectionStart={handleFundingRateCollectionStart}
                onCollectionError={handleFundingRateCollectionError}
                disabled={loading || updating}
                className="h-10"
              />

              {/* OI収集ボタン */}
              <OpenInterestCollectionButton
                mode="bulk"
                onCollectionStart={handleOpenInterestCollectionStart}
                onCollectionError={handleOpenInterestCollectionError}
                disabled={loading || updating}
                className="h-10 bg-green-600 hover:bg-green-700 focus:ring-green-500"
              />
            </div>
          </div>
        </div>

        {/* ステータスメッセージ */}
        {(bulkCollectionMessage ||
          fundingRateCollectionMessage ||
          openInterestCollectionMessage ||
          allDataCollectionMessage) && (
          <div className="mt-6 pt-4 border-t border-secondary-200 dark:border-secondary-700">
            <div className="space-y-2">
              {allDataCollectionMessage && (
                <div className="text-sm text-secondary-600 dark:text-secondary-400 font-medium">
                  {allDataCollectionMessage}
                </div>
              )}
              {bulkCollectionMessage && (
                <div className="text-sm text-secondary-600 dark:text-secondary-400">
                  {bulkCollectionMessage}
                </div>
              )}
              {fundingRateCollectionMessage && (
                <div className="text-sm text-secondary-600 dark:text-secondary-400">
                  {fundingRateCollectionMessage}
                </div>
              )}
              {openInterestCollectionMessage && (
                <div className="text-sm text-secondary-600 dark:text-secondary-400">
                  {openInterestCollectionMessage}
                </div>
              )}
            </div>
          </div>
        )}
      </div>
    </div>
  );
};

export default DataControls;
