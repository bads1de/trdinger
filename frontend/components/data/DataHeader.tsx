import React from "react";

interface DataHeaderProps {
  loading: boolean;
  error: string;
  updating: boolean;
  bulkUpdating?: boolean;
  handleRefresh: () => void;
  handleBulkIncrementalUpdate: () => void;
}

const DataHeader: React.FC<DataHeaderProps> = ({
  loading,
  error,
  updating,
  bulkUpdating = false,
  handleRefresh,
  handleBulkIncrementalUpdate,
}) => {
  return (
    <div className="enterprise-card border-0 rounded-none border-b border-secondary-200 dark:border-secondary-700 shadow-enterprise-sm">
      <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8 py-6">
        <div className="flex flex-col sm:flex-row sm:items-center sm:justify-between gap-4">
          <div className="animate-slide-up">
            <h1 className="text-3xl font-bold text-gradient">
              📊 データテーブル
            </h1>
            <p className="mt-2 text-base text-secondary-600 dark:text-secondary-400">
              エンタープライズレベルの仮想通貨データ分析・表示
            </p>
            <div className="mt-2 flex items-center gap-2">
              <span className="badge-primary">リアルタイム</span>
              <span className="badge-success">高精度データ</span>
            </div>
          </div>

          <div className="flex items-center gap-3 animate-slide-up">
            {/* ステータスインジケーター */}
            <div className="flex items-center gap-2">
              <div
                className={`w-2 h-2 rounded-full ${
                  loading
                    ? "bg-warning-500 animate-pulse"
                    : error
                    ? "bg-error-500"
                    : "bg-success-500"
                }`}
              ></div>
              <span className="text-sm text-secondary-600 dark:text-secondary-400">
                {loading ? "更新中" : error ? "エラー" : "接続中"}
              </span>
            </div>

            <div className="flex gap-2">
              <button
                onClick={handleRefresh}
                disabled={loading || updating}
                className="btn-primary group"
              >
                <svg
                  className={`w-4 h-4 mr-2 transition-transform duration-200 ${
                    loading ? "animate-spin" : "group-hover:rotate-180"
                  }`}
                  fill="none"
                  stroke="currentColor"
                  viewBox="0 0 24 24"
                >
                  <path
                    strokeLinecap="round"
                    strokeLinejoin="round"
                    strokeWidth={2}
                    d="M4 4v5h.582m15.356 2A8.001 8.001 0 004.582 9m0 0H9m11 11v-5h-.581m0 0a8.003 8.003 0 01-15.357-2m15.357 2H15"
                  />
                </svg>
                {loading ? "更新中..." : "データ更新"}
              </button>

              <button
                onClick={handleBulkIncrementalUpdate}
                disabled={loading || updating || bulkUpdating}
                className="btn-primary group"
              >
                <svg
                  className={`w-4 h-4 mr-2 transition-transform duration-200 ${
                    bulkUpdating ? "animate-spin" : "group-hover:scale-110"
                  }`}
                  fill="none"
                  stroke="currentColor"
                  viewBox="0 0 24 24"
                >
                  <path
                    strokeLinecap="round"
                    strokeLinejoin="round"
                    strokeWidth={2}
                    d="M4 4v5h.582m15.356 2A8.001 8.001 0 004.582 9m0 0H9m11 11v-5h-.581m0 0a8.003 8.003 0 01-15.357-2m15.357 2H15"
                  />
                </svg>
                {bulkUpdating ? "差分更新中..." : "差分更新"}
              </button>
            </div>
          </div>
        </div>
      </div>
    </div>
  );
};

export default DataHeader;
