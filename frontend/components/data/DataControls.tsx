import React, { useState } from "react";
import SymbolSelector from "@/components/common/SymbolSelector";
import TimeFrameSelector from "@/components/common/TimeFrameSelector";
import DataCollectionButton from "@/components/button/DataCollectionButton";
import {
  allDataCollectionConfig,
  ohlcvCollectionConfig,
  fundingRateCollectionConfig,
  openInterestCollectionConfig,
  fearGreedCollectionConfig,
  externalMarketCollectionConfig,
} from "@/components/button/dataCollectionConfigs";
import DataResetPanel from "@/components/common/DataResetPanel";

import { TradingPair, TimeFrame } from "@/types/market-data";
import {
  AllDataCollectionResult,
  BulkOHLCVCollectionResult,
} from "@/types/data-collection";
import {
  BulkFundingRateCollectionResult,
  FundingRateCollectionResult,
} from "@/types/funding-rate";
import {
  BulkOpenInterestCollectionResult,
  OpenInterestCollectionResult,
} from "@/types/open-interest";
import { FearGreedCollectionResult } from "@/hooks/useFearGreedData";

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
  handleFearGreedCollectionStart: (result: FearGreedCollectionResult) => void;
  handleFearGreedCollectionError: (errorMessage: string) => void;
  bulkCollectionMessage: string;
  fundingRateCollectionMessage: string;
  openInterestCollectionMessage: string;
  fearGreedCollectionMessage: string;
  externalMarketCollectionMessage: string;
  allDataCollectionMessage: string;
  incrementalUpdateMessage: string;
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
  handleFearGreedCollectionStart,
  handleFearGreedCollectionError,
  bulkCollectionMessage,
  fundingRateCollectionMessage,
  openInterestCollectionMessage,
  fearGreedCollectionMessage,
  externalMarketCollectionMessage,
  allDataCollectionMessage,
  incrementalUpdateMessage,
  dataStatus,
}) => {
  const [showResetPanel, setShowResetPanel] = useState(false);
  return (
    <div className="enterprise-card animate-slide-up">
      <div className="p-6">
        {/* データベース状況セクション */}
        {dataStatus && dataStatus.data && (
          <div className="mb-6">
            <div className="flex items-center justify-between mb-4">
              <h2 className="text-xl font-semibold text-secondary-900 dark:text-secondary-100">
                📊 データベース状況
              </h2>
              <span className="badge-primary">
                {dataStatus.data.total_records?.toLocaleString()}件
              </span>
            </div>

            {/* 総計表示 */}
            <div className="grid grid-cols-1 md:grid-cols-3 gap-4 text-sm mb-4">
              <div className="flex justify-between">
                <span className="text-secondary-600 dark:text-secondary-400">
                  OHLCV:
                </span>
                <span className="font-medium text-secondary-900 dark:text-secondary-100">
                  {dataStatus.data.data_counts?.ohlcv?.toLocaleString()}件
                </span>
              </div>
              <div className="flex justify-between">
                <span className="text-secondary-600 dark:text-secondary-400">
                  ファンディングレート:
                </span>
                <span className="font-medium text-secondary-900 dark:text-secondary-100">
                  {dataStatus.data.data_counts?.funding_rates?.toLocaleString()}
                  件
                </span>
              </div>
              <div className="flex justify-between">
                <span className="text-secondary-600 dark:text-secondary-400">
                  オープンインタレスト:
                </span>
                <span className="font-medium text-secondary-900 dark:text-secondary-100">
                  {dataStatus.data.data_counts?.open_interest?.toLocaleString()}
                  件
                </span>
              </div>
              <div className="flex justify-between">
                <span className="text-secondary-600 dark:text-secondary-400">
                  Fear & Greed Index:
                </span>
                <span className="font-medium text-secondary-900 dark:text-secondary-100">
                  {dataStatus.data.data_counts?.fear_greed_index?.toLocaleString() ||
                    0}
                  件
                </span>
              </div>
            </div>

            {/* OHLCV詳細（時間足別） */}
            {dataStatus.data.details?.ohlcv?.timeframes && (
              <div className="mb-4">
                <h3 className="text-lg font-medium text-secondary-800 dark:text-secondary-200 mb-2">
                  詳細
                </h3>
                <div className="grid grid-cols-1 md:grid-cols-5 gap-2 text-xs">
                  {Object.entries(dataStatus.data.details.ohlcv.timeframes).map(
                    ([tf, details]: [string, any]) => (
                      <div
                        key={tf}
                        className="bg-secondary-100 dark:bg-secondary-800 p-2 rounded"
                      >
                        <div className="font-medium text-secondary-900 dark:text-secondary-100">
                          {tf}
                        </div>
                        <div className="text-secondary-600 dark:text-secondary-400">
                          {details.count?.toLocaleString()}件
                        </div>
                        {details.latest_timestamp && (
                          <div className="text-xs text-secondary-500 dark:text-secondary-500">
                            最新:{" "}
                            {new Date(
                              details.latest_timestamp
                            ).toLocaleDateString("ja-JP")}
                          </div>
                        )}
                      </div>
                    )
                  )}
                </div>
              </div>
            )}

            {/* FR・OI・FG詳細 */}
            <div className="grid grid-cols-1 md:grid-cols-3 gap-4 text-xs">
              {/* ファンディングレート詳細 */}
              {dataStatus.data.details?.funding_rates && (
                <div className="bg-secondary-100 dark:bg-secondary-800 p-3 rounded">
                  <h4 className="font-medium text-secondary-900 dark:text-secondary-100 mb-2">
                    ファンディングレート詳細
                  </h4>
                  <div className="space-y-1">
                    <div className="flex justify-between">
                      <span className="text-secondary-600 dark:text-secondary-400">
                        件数:
                      </span>
                      <span>
                        {dataStatus.data.details.funding_rates.count?.toLocaleString()}
                        件
                      </span>
                    </div>
                    {dataStatus.data.details.funding_rates.latest_timestamp && (
                      <div className="flex justify-between">
                        <span className="text-secondary-600 dark:text-secondary-400">
                          最新:
                        </span>
                        <span>
                          {new Date(
                            dataStatus.data.details.funding_rates.latest_timestamp
                          ).toLocaleDateString("ja-JP")}
                        </span>
                      </div>
                    )}
                    {dataStatus.data.details.funding_rates.oldest_timestamp && (
                      <div className="flex justify-between">
                        <span className="text-secondary-600 dark:text-secondary-400">
                          最古:
                        </span>
                        <span>
                          {new Date(
                            dataStatus.data.details.funding_rates.oldest_timestamp
                          ).toLocaleDateString("ja-JP")}
                        </span>
                      </div>
                    )}
                  </div>
                </div>
              )}

              {/* オープンインタレスト詳細 */}
              {dataStatus.data.details?.open_interest && (
                <div className="bg-secondary-100 dark:bg-secondary-800 p-3 rounded">
                  <h4 className="font-medium text-secondary-900 dark:text-secondary-100 mb-2">
                    オープンインタレスト詳細
                  </h4>
                  <div className="space-y-1">
                    <div className="flex justify-between">
                      <span className="text-secondary-600 dark:text-secondary-400">
                        件数:
                      </span>
                      <span>
                        {dataStatus.data.details.open_interest.count?.toLocaleString()}
                        件
                      </span>
                    </div>
                    {dataStatus.data.details.open_interest.latest_timestamp && (
                      <div className="flex justify-between">
                        <span className="text-secondary-600 dark:text-secondary-400">
                          最新:
                        </span>
                        <span>
                          {new Date(
                            dataStatus.data.details.open_interest.latest_timestamp
                          ).toLocaleDateString("ja-JP")}
                        </span>
                      </div>
                    )}
                    {dataStatus.data.details.open_interest.oldest_timestamp && (
                      <div className="flex justify-between">
                        <span className="text-secondary-600 dark:text-secondary-400">
                          最古:
                        </span>
                        <span>
                          {new Date(
                            dataStatus.data.details.open_interest.oldest_timestamp
                          ).toLocaleDateString("ja-JP")}
                        </span>
                      </div>
                    )}
                  </div>
                </div>
              )}

              {/* Fear & Greed Index詳細 */}
              {dataStatus.data.details?.fear_greed_index && (
                <div className="bg-secondary-100 dark:bg-secondary-800 p-3 rounded">
                  <h4 className="font-medium text-secondary-900 dark:text-secondary-100 mb-2">
                    Fear & Greed Index詳細
                  </h4>
                  <div className="space-y-1">
                    <div className="flex justify-between">
                      <span className="text-secondary-600 dark:text-secondary-400">
                        件数:
                      </span>
                      <span>
                        {dataStatus.data.details.fear_greed_index.count?.toLocaleString()}
                        件
                      </span>
                    </div>
                    {dataStatus.data.details.fear_greed_index
                      .latest_timestamp && (
                      <div className="flex justify-between">
                        <span className="text-secondary-600 dark:text-secondary-400">
                          最新:
                        </span>
                        <span>
                          {new Date(
                            dataStatus.data.details.fear_greed_index.latest_timestamp
                          ).toLocaleDateString("ja-JP")}
                        </span>
                      </div>
                    )}
                    {dataStatus.data.details.fear_greed_index
                      .oldest_timestamp && (
                      <div className="flex justify-between">
                        <span className="text-secondary-600 dark:text-secondary-400">
                          最古:
                        </span>
                        <span>
                          {new Date(
                            dataStatus.data.details.fear_greed_index.oldest_timestamp
                          ).toLocaleDateString("ja-JP")}
                        </span>
                      </div>
                    )}
                  </div>
                </div>
              )}
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
            <div className="flex items-center justify-between">
              <label className="block text-sm font-medium text-secondary-600 dark:text-secondary-400">
                データ収集
              </label>
              <button
                onClick={() => setShowResetPanel(!showResetPanel)}
                className="text-sm text-warning-600 hover:text-warning-800 dark:text-warning-400 dark:hover:text-warning-200 font-medium"
              >
                {showResetPanel
                  ? "🔼 リセット機能を閉じる"
                  : "🗑️ データリセット"}
              </button>
            </div>
            <div className="space-y-3">
              {/* 上段：主要データ収集ボタン */}
              <div className="grid grid-cols-1 sm:grid-cols-2 lg:grid-cols-5 gap-3">
                {/* 全データ一括収集ボタン */}
                <DataCollectionButton
                  config={allDataCollectionConfig}
                  onCollectionStart={handleAllDataCollectionStart}
                  onCollectionError={handleAllDataCollectionError}
                  disabled={loading || updating}
                  className="h-10 text-sm"
                />

                {/* OHLCV収集ボタン */}
                <DataCollectionButton
                  config={ohlcvCollectionConfig}
                  onCollectionStart={handleBulkCollectionStart}
                  onCollectionError={handleBulkCollectionError}
                  disabled={loading || updating}
                  className="h-10"
                />

                {/* FR収集ボタン */}
                <DataCollectionButton
                  config={fundingRateCollectionConfig}
                  onCollectionStart={handleFundingRateCollectionStart}
                  onCollectionError={handleFundingRateCollectionError}
                  disabled={loading || updating}
                  className="h-10"
                />

                {/* OI収集ボタン */}
                <DataCollectionButton
                  config={openInterestCollectionConfig}
                  onCollectionStart={handleOpenInterestCollectionStart}
                  onCollectionError={handleOpenInterestCollectionError}
                  disabled={loading || updating}
                  className="h-10 bg-green-600 hover:bg-green-700 focus:ring-green-500"
                />

                {/* FG収集ボタン */}
                <DataCollectionButton
                  config={fearGreedCollectionConfig}
                  onCollectionStart={handleFearGreedCollectionStart}
                  onCollectionError={handleFearGreedCollectionError}
                  disabled={loading || updating}
                  className="h-10"
                />
              </div>
            </div>
          </div>
        </div>

        {/* ステータスメッセージ */}
        {(bulkCollectionMessage ||
          fundingRateCollectionMessage ||
          openInterestCollectionMessage ||
          fearGreedCollectionMessage ||
          externalMarketCollectionMessage ||
          allDataCollectionMessage ||
          incrementalUpdateMessage) && (
          <div className="mt-6 pt-4 border-t border-secondary-200 dark:border-secondary-700">
            <div className="space-y-2">
              {allDataCollectionMessage && (
                <div className="text-sm text-secondary-600 dark:text-secondary-400 font-medium">
                  {allDataCollectionMessage}
                </div>
              )}
              {incrementalUpdateMessage && (
                <div className="text-sm text-secondary-600 dark:text-secondary-400 font-medium">
                  {incrementalUpdateMessage}
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
              {fearGreedCollectionMessage && (
                <div className="text-sm text-secondary-600 dark:text-secondary-400">
                  {fearGreedCollectionMessage}
                </div>
              )}
              {externalMarketCollectionMessage && (
                <div className="text-sm text-secondary-600 dark:text-secondary-400">
                  {externalMarketCollectionMessage}
                </div>
              )}
            </div>
          </div>
        )}
      </div>

      {/* データリセットパネル */}
      {showResetPanel && (
        <div className="mt-4">
          <DataResetPanel
            selectedSymbol={selectedSymbol}
            isVisible={showResetPanel}
            onClose={() => setShowResetPanel(false)}
          />
        </div>
      )}
    </div>
  );
};

export default DataControls;
