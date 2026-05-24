/**
 * MLモデル共通ユーティリティ関数
 *
 * 複数フックで重複していたバッジバリアント関数などを集約し、DRY原則に従います。
 */

/**
 * スコアに応じたバッジバリアントを取得
 *
 * useModelInfo.getAccuracyBadgeVariant と useModelPerformance.getScoreBadgeVariant は
 * 全く同じロジックのため統合しました。
 *
 * @param score - スコア（0-1）
 * @returns バッジバリアント（"success"、"warning"、"destructive"、"outline"）
 *
 * @example
 * ```ts
 * getScoreBadgeVariant(0.85) // "success"
 * getScoreBadgeVariant(0.75) // "warning"
 * getScoreBadgeVariant(0.5)  // "destructive"
 * getScoreBadgeVariant()     // "outline"
 * ```
 */
export const getScoreBadgeVariant = (score?: number): "success" | "warning" | "destructive" | "outline" => {
  if (!score) return "outline";
  if (score >= 0.8) return "success";
  if (score >= 0.7) return "warning";
  return "destructive";
};

/**
 * モデルタイプに応じたバッジバリアントを取得
 *
 * @param modelType - モデルタイプ
 * @returns バッジバリアント（"default"または"outline"）
 *
 * @example
 * ```ts
 * getModelTypeBadgeVariant("lightgbm") // "default"
 * getModelTypeBadgeVariant("xgboost")  // "outline"
 * ```
 */
export const getModelTypeBadgeVariant = (modelType?: string): "default" | "outline" => {
  switch (modelType?.toLowerCase()) {
    case "lightgbm":
      return "default";
    case "xgboost":
      return "outline";
    default:
      return "outline";
  }
};

/**
 * モデルの状態に応じたバッジバリアントを取得
 *
 * @param isTraining - トレーニング中かどうか
 * @param isModelLoaded - モデルがロードされているかどうか
 * @returns バッジバリアント
 */
export const getStatusBadgeVariant = (
  isTraining?: boolean,
  isModelLoaded?: boolean
): "default" | "success" | "outline" => {
  if (isTraining) return "default";
  if (isModelLoaded) return "success";
  return "outline";
};
