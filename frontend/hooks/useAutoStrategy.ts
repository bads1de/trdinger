import { useState } from "react";
import { useApiCall } from "@/hooks/useApiCall";
import { GAConfig } from "@/types/optimization";

export const useAutoStrategy = (loadResults: () => void) => {
  const [showAutoStrategyModal, setShowAutoStrategyModal] = useState(false);

  const { execute: runAutoStrategy, loading: autoStrategyLoading } =
    useApiCall();

  /**
   * オートストラテジー実行
   */
  const handleAutoStrategy = async (config: GAConfig) => {
    // GAConfigをAPIリクエスト形式に変換
    const requestBody = {
      experiment_name: config.experiment_name,
      base_config: config.base_config,
      ga_config: config.ga_config,
    };

    const response = await runAutoStrategy("/api/auto-strategy/generate", {
      method: "POST",
      body: requestBody,
      onSuccess: (data) => {
        setShowAutoStrategyModal(false);
        const isMultiObjective = config.ga_config.enable_multi_objective;
        const message = isMultiObjective
          ? `🚀 多目的最適化GA戦略生成を開始しました！\n\n実験ID: ${data.experiment_id}\n\n複数の目的を同時に最適化します。\n生成完了後、オートストラテジーページで結果を確認できます。\n数分お待ちください。`
          : `🚀 戦略生成を開始しました！\n\n実験ID: ${data.experiment_id}\n\n生成完了後、結果一覧に自動的に表示されます。\n数分お待ちください。`;

        alert(message);
        // 結果一覧を更新（GA完了後に結果が表示される）
        loadResults();
      },
      onError: (error) => {
        console.error("Auto strategy generation failed:", error);
      },
    });
  };

  /**
   * オートストラテジーモーダルを開く
   */
  const openAutoStrategyModal = () => {
    setShowAutoStrategyModal(true);
  };

  return {
    showAutoStrategyModal,
    autoStrategyLoading,
    handleAutoStrategy,
    openAutoStrategyModal,
    setShowAutoStrategyModal,
  };
};
