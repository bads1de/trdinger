/**
 * GA実験進捗表示コンポーネント
 *
 * 実行中のGA実験の進捗状況（プログレスバー、世代カウンター、ステータス）を表示します。
 */

import { ExperimentProgress } from "@/hooks/useAutoStrategy";

interface ExperimentProgressCardProps {
  progress: ExperimentProgress;
  onStop?: (experimentId: string) => void;
}

/** 実験進捗カード */
export const ExperimentProgressCard: React.FC<ExperimentProgressCardProps> = ({
  progress,
  onStop,
}) => {
  const progressPercent =
    progress.total_generations && progress.total_generations > 0
      ? Math.round(
          ((progress.current_generation ?? 0) / progress.total_generations) *
            100,
        )
      : 0;

  const statusLabel = getStatusLabel(progress.status);
  const statusColor = getStatusColor(progress.status);

  return (
    <div className="rounded-lg border border-gray-200 bg-white p-4 shadow-sm dark:border-gray-700 dark:bg-gray-800">
      <div className="mb-3 flex items-center justify-between">
        <div className="flex items-center gap-2">
          <span
            className={`inline-flex items-center rounded-full px-2.5 py-0.5 text-xs font-medium ${statusColor}`}
          >
            {statusLabel}
          </span>
          <span className="text-sm font-medium text-gray-900 dark:text-gray-100">
            {progress.name || progress.experiment_id || "実験"}
          </span>
        </div>
        {onStop && progress.status === "running" && (
          <button
            onClick={() => onStop(progress.experiment_id!)}
            className="rounded-md bg-red-50 px-3 py-1.5 text-xs font-medium text-red-700 hover:bg-red-100 dark:bg-red-900/20 dark:text-red-400 dark:hover:bg-red-900/30"
          >
            停止
          </button>
        )}
      </div>

      {/* プログレスバー */}
      <div className="mb-2">
        <div className="flex items-center justify-between text-xs text-gray-500 dark:text-gray-400">
          <span>
            世代 {progress.current_generation ?? 0} /{" "}
            {progress.total_generations ?? "?"}
          </span>
          <span>{progressPercent}%</span>
        </div>
        <div className="mt-1 h-2 w-full overflow-hidden rounded-full bg-gray-200 dark:bg-gray-700">
          <div
            className="h-full rounded-full bg-blue-600 transition-all duration-500"
            style={{ width: `${progressPercent}%` }}
          />
        </div>
      </div>

      {/* 追加情報 */}
      {progress.best_fitness != null && (
        <div className="text-xs text-gray-500 dark:text-gray-400">
          最高フィットネス:{" "}
          <span className="font-medium text-gray-700 dark:text-gray-300">
            {progress.best_fitness.toFixed(4)}
          </span>
        </div>
      )}
    </div>
  );
};

/** 実験進捗リスト */
interface ExperimentProgressListProps {
  experiments: Map<string, ExperimentProgress>;
  onStop?: (experimentId: string) => void;
}

export const ExperimentProgressList: React.FC<ExperimentProgressListProps> = ({
  experiments,
  onStop,
}) => {
  if (experiments.size === 0) {
    return null;
  }

  return (
    <div className="space-y-3">
      <h3 className="text-sm font-semibold text-gray-700 dark:text-gray-300">
        実行中の実験 ({experiments.size}件)
      </h3>
      {Array.from(experiments.values()).map((exp) => (
        <ExperimentProgressCard
          key={exp.experiment_id ?? exp.id}
          progress={exp}
          onStop={onStop}
        />
      ))}
    </div>
  );
};

function getStatusLabel(status: string | null): string {
  switch (status) {
    case "running":
      return "実行中";
    case "completed":
      return "完了";
    case "failed":
      return "失敗";
    case "stopped":
      return "停止";
    default:
      return "不明";
  }
}

function getStatusColor(status: string | null): string {
  switch (status) {
    case "running":
      return "bg-blue-100 text-blue-800 dark:bg-blue-900/30 dark:text-blue-300";
    case "completed":
      return "bg-green-100 text-green-800 dark:bg-green-900/30 dark:text-green-300";
    case "failed":
      return "bg-red-100 text-red-800 dark:bg-red-900/30 dark:text-red-300";
    case "stopped":
      return "bg-gray-100 text-gray-800 dark:bg-gray-700 dark:text-gray-300";
    default:
      return "bg-gray-100 text-gray-600 dark:bg-gray-700 dark:text-gray-400";
  }
}
