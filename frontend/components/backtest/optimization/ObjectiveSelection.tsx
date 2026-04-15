import React from "react";
import { AVAILABLE_OBJECTIVES } from "@/constants/backtest";
import { GAConfig } from "@/types/optimization";

interface ObjectiveSelectionProps {
  gaConfig: GAConfig["ga_config"];
  onGAConfigChange: (newConfig: Partial<GAConfig["ga_config"]>) => void;
}

export const ObjectiveSelection: React.FC<ObjectiveSelectionProps> = ({
  gaConfig,
  onGAConfigChange,
}) => {
  /**
   * 目的関数の選択・解除時のハンドラ
   * @param objectiveName 目的関数名
   * @param checked チェック状態（true: 選択, false: 解除）
   */
  const handleObjectiveChange = (objectiveName: string, checked: boolean) => {
    // 現在選択されている目的関数と重みを取得（未設定時は空配列）
    const currentObjectives = gaConfig.objectives || [];
    const currentWeights = gaConfig.objective_weights || [];

    if (checked) {
      // 追加処理: 目的関数が未選択の場合のみ追加
      const objective = AVAILABLE_OBJECTIVES.find(
        (obj) => obj.name === objectiveName
      );

      if (objective && !currentObjectives.includes(objectiveName)) {
        // 新しい目的関数リストと重みリストを作成
        const newObjectives = [...currentObjectives, objectiveName];
        const newWeights = [...currentWeights, objective.weight];

        // 親コンポーネントに変更を通知
        onGAConfigChange({
          objectives: newObjectives,
          objective_weights: newWeights,
        });
      }
    } else {
      // 削除処理: 指定の目的関数が存在する場合のみ削除
      const index = currentObjectives.indexOf(objectiveName);

      if (index > -1) {
        // 指定インデックスを除外した新しいリストを作成
        const newObjectives = currentObjectives.filter((_, i) => i !== index);
        const newWeights = currentWeights.filter((_, i) => i !== index);

        // 親コンポーネントに変更を通知
        onGAConfigChange({
          objectives: newObjectives,
          objective_weights: newWeights,
        });
      }
    }
  };

  return (
    <div className="border-t border-indigo-500/20 pt-3 mt-3">
      <div className="space-y-3">
        <h5 className="text-sm font-medium text-indigo-200">最適化目的</h5>

        <div className="grid grid-cols-2 gap-3">
          {AVAILABLE_OBJECTIVES.map((objective) => (
            <div key={objective.name}>
              <label className="flex items-center space-x-2">
                <input
                  type="checkbox"
                  checked={gaConfig.objectives?.includes(objective.name) ?? false}
                  onChange={(e) =>
                    handleObjectiveChange(objective.name, e.target.checked)
                  }
                  className="rounded border-indigo-500 text-indigo-600 focus:ring-indigo-500"
                />
                <span className="text-sm text-indigo-200">
                  {objective.display_name}
                </span>
                <span
                  className={`text-xs ${
                    objective.weight > 0 ? "text-green-400" : "text-red-400"
                  }`}
                >
                  {objective.weight > 0 ? "最大化" : "最小化"}
                </span>
              </label>
              <div className="text-xs text-indigo-300/60 ml-6">
                {objective.description}
              </div>
            </div>
          ))}
        </div>

        {gaConfig.objectives && gaConfig.objectives.length > 0 && (
          <div className="mt-3 p-2 bg-indigo-800/30 rounded border border-indigo-500/30">
            <h6 className="text-xs font-medium text-indigo-200 mb-2">
              選択された目的 ({gaConfig.objectives.length}個)
            </h6>
            <div className="space-y-1">
              {gaConfig.objectives.map((objective, index) => (
                <div
                  key={objective}
                  className="flex justify-between items-center text-xs"
                >
                  <span className="text-indigo-200">
                    {AVAILABLE_OBJECTIVES.find(
                      (obj) => obj.name === objective,
                    )?.display_name || objective}
                  </span>
                  <span className="text-indigo-300">
                    重み:{" "}
                    {gaConfig.objective_weights?.[index]?.toFixed(1) || "1.0"}
                  </span>
                </div>
              ))}
            </div>
          </div>
        )}
      </div>
    </div>
  );
};
