import React from "react";
import SymbolSelector from "@/components/common/SymbolSelector";
import TimeFrameSelector from "@/components/common/TimeFrameSelector";
import AllDataCollectionButton from "@/components/common/AllDataCollectionButton";
import OpenInterestCollectionButton from "@/components/common/OpenInterestCollectionButton";
import TechnicalIndicatorCalculationButton from "@/components/common/TechnicalIndicatorCalculationButton";
import { useDataCollection } from "@/hooks/useDataCollection";
import ApiButton from "@/components/common/ApiButton";
import { DownloadIcon, ChartIcon } from "@/components/common/Icons";
import {
  TradingPair,
  TimeFrame,
  AllDataCollectionResult,
  BulkOHLCVCollectionResult,
  BulkFundingRateCollectionResult,
  FundingRateCollectionResult,
  OpenInterestCollectionResult,
  BulkOpenInterestCollectionResult,
  BulkTechnicalIndicatorCalculationResult,
} from "@/types/strategy";

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
  handleTechnicalIndicatorCalculationStart: (
    result: BulkTechnicalIndicatorCalculationResult
  ) => void;
  handleTechnicalIndicatorCalculationError: (errorMessage: string) => void;
  bulkCollectionMessage: string;
  fundingRateCollectionMessage: string;
  openInterestCollectionMessage: string;
  allDataCollectionMessage: string;
  technicalIndicatorCalculationMessage: string;
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
  handleTechnicalIndicatorCalculationStart,
  handleTechnicalIndicatorCalculationError,
  bulkCollectionMessage,
  fundingRateCollectionMessage,
  openInterestCollectionMessage,
  allDataCollectionMessage,
  technicalIndicatorCalculationMessage,
}) => {
  const dataCollection = useDataCollection();
  return (
    <div className="enterprise-card animate-slide-up">
      <div className="p-6">
        {/* セクションヘッダー */}
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
              <ApiButton
                onClick={async () => {
                  await dataCollection.ohlcv.collect(
                    handleBulkCollectionStart,
                    handleBulkCollectionError
                  );
                }}
                loading={dataCollection.ohlcv.loading}
                disabled={loading || updating || dataCollection.isAnyLoading}
                variant="primary"
                size="sm"
                loadingText="収集中..."
                className="h-10 bg-blue-600 hover:bg-blue-700 focus:ring-blue-500"
                fixedWidth={false}
              >
                OHLCV収集
              </ApiButton>

              {/* FR収集ボタン */}
              <ApiButton
                onClick={async () => {
                  await dataCollection.fundingRate.collect(
                    handleFundingRateCollectionStart,
                    handleFundingRateCollectionError
                  );
                }}
                loading={dataCollection.fundingRate.loading}
                disabled={loading || updating || dataCollection.isAnyLoading}
                variant="success"
                size="sm"
                loadingText="収集中..."
                className="h-10 bg-green-600 hover:bg-green-700 focus:ring-green-500"
                fixedWidth={false}
              >
                FR収集
              </ApiButton>

              {/* OI収集ボタン */}
              <OpenInterestCollectionButton
                mode="bulk"
                onCollectionStart={handleOpenInterestCollectionStart}
                onCollectionError={handleOpenInterestCollectionError}
                disabled={loading || updating}
                className="h-10 bg-green-600 hover:bg-green-700 focus:ring-green-500"
              />

              {/* TI一括計算ボタン */}
              <TechnicalIndicatorCalculationButton
                mode="bulk"
                symbol={selectedSymbol}
                timeframe={selectedTimeFrame}
                onCalculationStart={handleTechnicalIndicatorCalculationStart}
                onCalculationError={handleTechnicalIndicatorCalculationError}
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
          allDataCollectionMessage ||
          technicalIndicatorCalculationMessage) && (
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
              {technicalIndicatorCalculationMessage && (
                <div className="text-sm text-secondary-600 dark:text-secondary-400">
                  {technicalIndicatorCalculationMessage}
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
