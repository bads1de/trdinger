import React from "react";

interface DataStatusPanelProps {
  dataStatus: any; // TODO: より具体的な型を指定する
}

const DataStatusPanel: React.FC<DataStatusPanelProps> = ({ dataStatus }) => {
  if (!dataStatus) {
    return null;
  }

  return (
    <div className="enterprise-card animate-slide-up">
      <div className="p-6">
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
                ? new Date(dataStatus.latest_timestamp).toLocaleString("ja-JP")
                : "なし"}
            </span>
          </div>
          <div className="flex justify-between">
            <span className="text-secondary-600 dark:text-secondary-400">
              最古データ:
            </span>
            <span className="font-medium text-secondary-900 dark:text-secondary-100">
              {dataStatus.oldest_timestamp
                ? new Date(dataStatus.oldest_timestamp).toLocaleString("ja-JP")
                : "なし"}
            </span>
          </div>
        </div>
      </div>
    </div>
  );
};

export default DataStatusPanel;
