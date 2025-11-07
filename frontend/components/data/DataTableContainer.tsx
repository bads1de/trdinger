import React, { useMemo } from "react";
import DataTable from "@/components/table/DataTable";
import {
  ohlcvColumns,
  fundingRateColumns,
  openInterestColumns,
} from "@/components/common";
import { Tabs, TabsList, TabsTrigger, TabsContent } from "@/components/ui/tabs";
import { PriceData, TimeFrame } from "@/types/market-data";
import { FundingRateData } from "@/types/funding-rate";
import { OpenInterestData } from "@/types/open-interest";

interface DataTableContainerProps {
  selectedSymbol: string;
  selectedTimeFrame: TimeFrame;
  activeTab: string;
  setActiveTab: (value: string) => void;
  ohlcvData: PriceData[];
  loading: boolean;
  error: string;
  fundingRateData: FundingRateData[];
  fundingLoading: boolean;
  fundingError: string;
  openInterestData: OpenInterestData[];
  openInterestLoading: boolean;
  openInterestError: string;
}

const DataTableContainer: React.FC<DataTableContainerProps> = ({
  selectedSymbol,
  selectedTimeFrame,
  activeTab,
  setActiveTab,
  ohlcvData,
  loading,
  error,
  fundingRateData,
  fundingLoading,
  fundingError,
  openInterestData,
  openInterestLoading,
  openInterestError,
}) => {
  // テーブル設定オブジェクト
  const tableConfigs = {
    ohlcv: {
      columns: ohlcvColumns,
      title: `📊 ${selectedSymbol} - ${selectedTimeFrame} OHLCVデータ`,
      pageSize: 50,
      enableExport: true,
      enableSearch: false,
      searchKeys: undefined as (keyof PriceData)[] | undefined,
    },
    funding: {
      columns: fundingRateColumns,
      title: "📊 FRデータ",
      pageSize: 50,
      enableExport: true,
      enableSearch: true,
      searchKeys: ["symbol"] as (keyof FundingRateData)[],
    },
    openinterest: {
      columns: openInterestColumns,
      title: "📈 OIデータ",
      pageSize: 50,
      enableExport: true,
      enableSearch: true,
      searchKeys: ["symbol"] as (keyof OpenInterestData)[],
    },
  };

  return (
    <div className="enterprise-card animate-slide-up">
      <div className="p-6">
        {/* タブヘッダー */}
        <Tabs
          value={activeTab}
          onValueChange={(value) =>
            setActiveTab(value)
          }
          className="w-full"
        >
          <div className="flex items-center justify-between mb-6">
            <div className="flex items-center gap-4">
              <h2 className="text-xl font-semibold text-secondary-900 dark:text-secondary-100">
                📊 {selectedSymbol}
              </h2>
              <TabsList>
                <TabsTrigger value="ohlcv">OHLCV</TabsTrigger>
                <TabsTrigger value="funding">FR</TabsTrigger>
                <TabsTrigger value="openinterest">OI</TabsTrigger>
              </TabsList>
            </div>

            {/* データ情報バッジ */}
            <div className="flex items-center gap-2">
              {activeTab === "ohlcv" && ohlcvData.length > 0 && !loading && (
                <>
                  <span className="badge-primary">{ohlcvData.length}件</span>
                  <span className="badge-success">
                    最新: ${ohlcvData[ohlcvData.length - 1]?.close.toFixed(2)}
                  </span>
                </>
              )}
              {activeTab === "funding" &&
                fundingRateData.length > 0 &&
                !fundingLoading && (
                  <>
                    <span className="badge-primary">
                      {fundingRateData.length}件
                    </span>
                    <span className="badge-info">
                      最新レート:{" "}
                      {(fundingRateData[0]?.funding_rate * 100).toFixed(4)}%
                    </span>
                  </>
                )}
              {activeTab === "openinterest" &&
                openInterestData.length > 0 &&
                !openInterestLoading && (
                  <>
                    <span className="badge-primary">
                      {openInterestData.length}件
                    </span>
                    <span className="badge-warning">
                      最新OI:{" "}
                      {new Intl.NumberFormat("en-US", {
                        style: "currency",
                        currency: "USD",
                        notation: "compact",
                        maximumFractionDigits: 1,
                      }).format(openInterestData[0]?.open_interest_value || 0)}
                    </span>
                  </>
                )}
            </div>
          </div>

          {/* タブコンテンツ */}
          <TabsContent value="ohlcv">
            <DataTable
              data={ohlcvData}
              columns={tableConfigs.ohlcv.columns}
              title={tableConfigs.ohlcv.title}
              loading={loading}
              error={error}
              pageSize={tableConfigs.ohlcv.pageSize}
              enableExport={tableConfigs.ohlcv.enableExport}
              enableSearch={tableConfigs.ohlcv.enableSearch}
              searchKeys={tableConfigs.ohlcv.searchKeys}
            />
          </TabsContent>
          <TabsContent value="funding">
            <DataTable
              data={fundingRateData}
              columns={tableConfigs.funding.columns}
              title={tableConfigs.funding.title}
              loading={fundingLoading}
              error={fundingError}
              pageSize={tableConfigs.funding.pageSize}
              enableExport={tableConfigs.funding.enableExport}
              enableSearch={tableConfigs.funding.enableSearch}
              searchKeys={tableConfigs.funding.searchKeys}
            />
          </TabsContent>
          <TabsContent value="openinterest">
            <DataTable
              data={openInterestData}
              columns={tableConfigs.openinterest.columns}
              title={tableConfigs.openinterest.title}
              loading={openInterestLoading}
              error={openInterestError}
              pageSize={tableConfigs.openinterest.pageSize}
              enableExport={tableConfigs.openinterest.enableExport}
              enableSearch={tableConfigs.openinterest.enableSearch}
              searchKeys={tableConfigs.openinterest.searchKeys}
            />
          </TabsContent>
        </Tabs>
      </div>
    </div>
  );
};

export default DataTableContainer;
