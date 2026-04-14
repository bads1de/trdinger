/**
 * オートストラテジー用カスタムフック
 *
 * 遺伝的アルゴリズム（GA）を使用した自動戦略生成機能を提供します。
 * 戦略生成の実行、モーダル表示、設定検証などの機能を統合的に管理します。
 */

import { useState, useEffect, useCallback, useRef } from "react";
import { useApiCall } from "@/hooks/useApiCall";
import { GAConfig } from "@/types/optimization";
import { BACKEND_API_URL } from "@/constants";

/** 実験進捗情報の型 */
export interface ExperimentProgress {
  id: number;
  experiment_id: string | null;
  name: string | null;
  status: string | null;
  progress: number | null;
  current_generation: number | null;
  total_generations: number | null;
  best_fitness: number | null;
  created_at: string | null;
  completed_at: string | null;
}

/**
 * オートストラテジーフック
 *
 * 遺伝的アルゴリズム（GA）を使用した自動戦略生成機能を提供します。
 * 戦略生成の実行、モーダル表示、設定検証などの機能を統合的に管理します。
 *
 * @example
 * ```tsx
 * const {
 *   showAutoStrategyModal,
 *   autoStrategyLoading,
 *   handleAutoStrategy,
 *   openAutoStrategyModal
 * } = useAutoStrategy(loadResults);
 *
 * // 戦略生成を実行
 * handleAutoStrategy(gaConfig);
 *
 * // モーダルを開く
 * openAutoStrategyModal();
 * ```
 *
 * @param {() => void} loadResults - 戦略生成完了後に結果一覧を更新する関数
 * @returns {{
 *   showAutoStrategyModal: boolean,
 *   autoStrategyLoading: boolean,
 *   handleAutoStrategy: (config: GAConfig) => Promise<void>,
 *   openAutoStrategyModal: () => void,
 *   setShowAutoStrategyModal: (show: boolean) => void
 * }} オートストラテジー関連の状態と操作関数
 */
export const useAutoStrategy = (loadResults: () => void) => {
  const [showAutoStrategyModal, setShowAutoStrategyModal] = useState(false);
  const [runningExperiments, setRunningExperiments] = useState<
    Map<string, ExperimentProgress>
  >(new Map());

  const { execute: runAutoStrategy, loading: autoStrategyLoading } =
    useApiCall();

  // 実行中実験のポーリング（5秒間隔）
  const pollingIntervalRef = useRef<NodeJS.Timeout | null>(null);

  const fetchExperimentProgress = useCallback(
    async (experimentId: string) => {
      try {
        const response = await fetch(
          `${BACKEND_API_URL}/api/auto-strategy/experiments/${experimentId}`,
        );
        if (!response.ok) {
          if (response.status === 404) {
            // 実験が見つからない場合はポーリング対象から外す
            setRunningExperiments((prev) => {
              const next = new Map(prev);
              next.delete(experimentId);
              return next;
            });
          }
          return;
        }
        const data: ExperimentProgress = await response.json();
        setRunningExperiments((prev) => {
          const next = new Map(prev);
          next.set(experimentId, data);
          return next;
        });
        // 完了または失敗したらポーリング停止
        if (data.status === "completed" || data.status === "failed") {
          setRunningExperiments((prev) => {
            const next = new Map(prev);
            next.delete(experimentId);
            return next;
          });
          loadResults();
        }
      } catch (error) {
        console.error("進捗取得エラー:", error);
      }
    },
    [loadResults],
  );

  // ポーリングタイマーの管理
  useEffect(() => {
    if (runningExperiments.size === 0) {
      if (pollingIntervalRef.current) {
        clearInterval(pollingIntervalRef.current);
        pollingIntervalRef.current = null;
      }
      return;
    }

    if (!pollingIntervalRef.current) {
      pollingIntervalRef.current = setInterval(() => {
        runningExperiments.forEach((_, id) => {
          fetchExperimentProgress(id);
        });
      }, 5000);
    }

    return () => {
      if (pollingIntervalRef.current) {
        clearInterval(pollingIntervalRef.current);
        pollingIntervalRef.current = null;
      }
    };
  }, [runningExperiments.size, fetchExperimentProgress]);

  /** 実験をポーリング対象に追加 */
  const pollExperimentProgress = useCallback(
    (experimentId: string) => {
      setRunningExperiments((prev) => {
        const next = new Map(prev);
        next.set(experimentId, {
          id: 0,
          experiment_id: experimentId,
          name: null,
          status: "running",
          progress: 0,
          current_generation: 0,
          total_generations: null,
          best_fitness: null,
          created_at: new Date().toISOString(),
          completed_at: null,
        });
        return next;
      });
    },
    [],
  );

  /**
   * オートストラテジー実行
   */
  const handleAutoStrategy = async (config: GAConfig) => {
    // リクエストボディのバリデーション
    if (!config.experiment_name || !config.base_config || !config.ga_config) {
      const errorMessage =
        "必須フィールドが不足しています: experiment_name, base_config, or ga_config";
      alert(errorMessage);
      console.error(errorMessage);
      return;
    }

    // base_configの必須フィールドをチェック
    const requiredBaseConfigFields = [
      "symbol",
      "timeframe",
      "start_date",
      "end_date",
      "initial_capital",
      "commission_rate",
    ];

    for (const field of requiredBaseConfigFields) {
      if (!(field in config.base_config)) {
        const errorMessage = `base_configに必須フィールドがありません: ${field}`;
        alert(errorMessage);
        console.error(errorMessage);
        return;
      }
    }

    // ga_configの必須フィールドをチェック
    const requiredGAConfigFields = [
      "population_size",
      "generations",
      "crossover_rate",
      "mutation_rate",
      "elite_size",
    ];

    for (const field of requiredGAConfigFields) {
      if (
        config.ga_config[field as keyof typeof config.ga_config] ===
          undefined ||
        config.ga_config[field as keyof typeof config.ga_config] === null
      ) {
        const errorMessage = `ga_configに必須フィールドがありません: ${field}`;
        alert(errorMessage);
        console.error(errorMessage);
        return;
      }
    }

    // フロントエンド側でUUIDを生成
    const experimentId = crypto.randomUUID();

    // GAConfigをAPIリクエスト形式に変換
    const requestBody = {
      experiment_id: experimentId, // UUIDを追加
      experiment_name: config.experiment_name,
      base_config: config.base_config,
      ga_config: config.ga_config,
    };

    await runAutoStrategy(`${BACKEND_API_URL}/api/auto-strategy/generate`, {
      method: "POST",
      body: requestBody,
      onSuccess: (data) => {
        setShowAutoStrategyModal(false);
        const isMultiObjective = config.ga_config.enable_multi_objective;
        // フロントエンドで生成したIDを使用
        const message = isMultiObjective
          ? `🚀 多目的最適化GA戦略生成を開始しました！\n\n実験ID: ${experimentId}\n\n複数の目的を同時に最適化します。\n生成完了後、オートストラテジーページで結果を確認できます。\n数分お待ちください。`
          : `🚀 戦略生成を開始しました！\n\n実験ID: ${experimentId}\n\n生成完了後、結果一覧に自動的に表示されます。\n数分お待ちください。`;

        alert(message);
        // 進捗ポーリングを開始
        pollExperimentProgress(experimentId);
      },
      onError: (error) => {
        alert(`オートストラテジーの生成に失敗しました: ${error}`);
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
    /** オートストラテジーモーダルの表示状態 */
    showAutoStrategyModal,
    /** 戦略生成実行中のローディング状態 */
    autoStrategyLoading,
    /** 戦略生成を実行する関数 */
    handleAutoStrategy,
    /** オートストラテジーモーダルを開く関数 */
    openAutoStrategyModal,
    /** オートストラテジーモーダルの表示状態を設定する関数 */
    setShowAutoStrategyModal,
    /** 実行中実験の進捗情報 */
    runningExperiments,
    /** 実験の進捗ポーリングを開始する関数 */
    pollExperimentProgress,
  };
};
