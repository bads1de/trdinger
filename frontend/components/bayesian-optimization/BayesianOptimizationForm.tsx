"use client";

import React, { useState } from "react";
import ActionButton from "@/components/common/ActionButton";
import { Card } from "@/components/ui/card";
import { InputField } from "@/components/common/InputField";
import { SelectField } from "@/components/common/SelectField";
import InfoModal from "@/components/common/InfoModal";
import { Info } from "lucide-react";
import { ML_INFO_MESSAGES } from "@/constants/info";
import CollapsibleJson from "@/components/common/CollapsibleJson";
import { useBayesianOptimizationForm } from "@/hooks/useBayesianOptimizationForm";
import {
  BayesianOptimizationConfig,
  ParameterSpace,
} from "@/types/bayesian-optimization";
import { BacktestConfig } from "@/types/backtest";

interface BayesianOptimizationFormProps {
  onMLOptimization: (config: BayesianOptimizationConfig) => void;
  isLoading?: boolean;
  currentBacktestConfig?: BacktestConfig | null;
}

const BayesianOptimizationForm: React.FC<BayesianOptimizationFormProps> = ({
  onMLOptimization,
  isLoading = false,
  currentBacktestConfig = null,
}) => {
  const [optimizationType, setOptimizationType] = useState<"ml">("ml");
  const [modelType, setModelType] = useState("lightgbm");
  const [nCalls, setNCalls] = useState(50);
  const [useDefaultParams, setUseDefaultParams] = useState(true);
  const [acquisitionFunction, setAcquisitionFunction] = useState("EI");
  const [nInitialPoints, setNInitialPoints] = useState(10);
  const [randomState, setRandomState] = useState(42);
  const [isInfoModalOpen, setIsInfoModalOpen] = useState(false);
  const [modalContent, setModalContent] = useState<{
    title: string;
    content: React.ReactNode;
  }>({
    title: "",
    content: null,
  });

  // プロファイル保存関連の状態
  const [saveAsProfile, setSaveAsProfile] = useState(false);
  const [profileName, setProfileName] = useState("");
  const [profileDescription, setProfileDescription] = useState("");

  const {
    customParameterSpace,
    setCustomParameterSpace,
    defaultParameterSpace,
  } = useBayesianOptimizationForm(modelType);

  const openInfoModal = (title: string, content: React.ReactNode) => {
    setModalContent({ title, content });
    setIsInfoModalOpen(true);
  };

  const handleSubmit = (e: React.FormEvent) => {
    e.preventDefault();

    // プロファイル保存時のバリデーション
    if (saveAsProfile && !profileName.trim()) {
      alert("プロファイル保存を有効にする場合、プロファイル名は必須です");
      return;
    }

    let parameterSpace: Record<string, ParameterSpace> | undefined;

    if (!useDefaultParams) {
      try {
        parameterSpace = JSON.parse(customParameterSpace);
      } catch (error) {
        alert("パラメータ空間のJSONが無効です");
        return;
      }
    }

    const config: BayesianOptimizationConfig = {
      optimization_type: optimizationType,
      n_calls: nCalls,
      parameter_space: parameterSpace,
      optimization_config: {
        acq_func: acquisitionFunction,
        n_initial_points: nInitialPoints,
        random_state: randomState,
      },
      // プロファイル保存設定を追加
      save_as_profile: saveAsProfile,
      profile_name:
        saveAsProfile && profileName.trim() ? profileName.trim() : undefined,
      profile_description:
        saveAsProfile && profileDescription.trim()
          ? profileDescription.trim()
          : undefined,
    };

    config.model_type = modelType;
    onMLOptimization(config);
  };

  return (
    <>
      <form
        onSubmit={handleSubmit}
        className="grid grid-cols-1 md:grid-cols-2 gap-6"
      >
        <div className="space-y-6">
          {/* 最適化タイプ選択 */}
          <Card className="p-4">
            <SelectField
              label="モデルタイプ"
              value={modelType}
              onChange={setModelType}
              options={[{ value: "lightgbm", label: "LightGBM" }]}
            />
          </Card>

          {/* 最適化パラメータ */}
          <Card className="p-4">
            <h3 className="text-lg font-medium mb-4">最適化パラメータ</h3>
            <div className="grid grid-cols-1 sm:grid-cols-2 gap-4 mb-4">
              <InputField
                label="試行回数"
                type="number"
                value={nCalls}
                onChange={setNCalls}
                min={10}
                max={200}
                labelAddon={
                  <Info
                    className="h-5 w-5 text-gray-400 cursor-pointer"
                    onClick={() =>
                      openInfoModal(
                        "試行回数 (n_calls)",
                        ML_INFO_MESSAGES.nCalls
                      )
                    }
                  />
                }
              />
              <InputField
                label="初期ランダム試行数"
                type="number"
                value={nInitialPoints}
                onChange={setNInitialPoints}
                min={5}
                max={50}
                labelAddon={
                  <Info
                    className="h-5 w-5 text-gray-400 cursor-pointer"
                    onClick={() =>
                      openInfoModal(
                        "初期ランダム試行数 (n_initial_points)",
                        ML_INFO_MESSAGES.nInitialPoints
                      )
                    }
                  />
                }
              />
            </div>
            <div className="grid grid-cols-1 sm:grid-cols-2 gap-4">
              <SelectField
                label="獲得関数 (acq_func)"
                value={acquisitionFunction}
                onChange={setAcquisitionFunction}
                options={[
                  { value: "EI", label: "Expected Improvement (EI)" },
                  { value: "PI", label: "Probability of Improvement (PI)" },
                  { value: "UCB", label: "Upper Confidence Bound (UCB)" },
                ]}
                labelAddon={
                  <Info
                    className="h-5 w-5 text-gray-400 cursor-pointer"
                    onClick={() =>
                      openInfoModal(
                        "獲得関数 (Acquisition Function)",
                        <>
                          <p>
                            次にどのパラメータを試すべきかを決定するための関数です。
                          </p>
                          <ul className="list-disc pl-5 mt-2 space-y-1">
                            <li>
                              <strong>EI (Expected Improvement):</strong>{" "}
                              {ML_INFO_MESSAGES.acquisitionFunctionEI}
                            </li>
                            <li>
                              <strong>PI (Probability of Improvement):</strong>{" "}
                              {ML_INFO_MESSAGES.acquisitionFunctionPI}
                            </li>
                            <li>
                              <strong>UCB (Upper Confidence Bound):</strong>{" "}
                              {ML_INFO_MESSAGES.acquisitionFunctionUCB}
                            </li>
                          </ul>
                        </>
                      )
                    }
                  />
                }
              />
              <InputField
                label="乱数シード (random_state)"
                type="number"
                value={randomState}
                onChange={setRandomState}
                labelAddon={
                  <Info
                    className="h-5 w-5 text-gray-400 cursor-pointer"
                    onClick={() =>
                      openInfoModal(
                        "乱数シード (random_state)",
                        ML_INFO_MESSAGES.randomState
                      )
                    }
                  />
                }
              />
            </div>
          </Card>
        </div>

        <div className="space-y-6">
          {/* パラメータ空間設定 */}
          <Card className="p-4 flex flex-col">
            <h3 className="text-lg font-medium mb-4">パラメータ空間</h3>
            <div className="mb-4">
              <div className="flex items-center space-x-2">
                <input
                  type="checkbox"
                  id="useDefaultParams"
                  checked={useDefaultParams}
                  onChange={(e) => setUseDefaultParams(e.target.checked)}
                  className="h-4 w-4 rounded border-gray-300 text-indigo-600 focus:ring-indigo-500"
                />
                <label
                  htmlFor="useDefaultParams"
                  className="text-sm font-medium text-gray-300"
                >
                  デフォルトパラメータ空間を使用
                </label>
              </div>
            </div>
            <div className="flex-grow flex flex-col">
              {!useDefaultParams && (
                <div className="flex-grow flex flex-col">
                  <label
                    htmlFor="customParameterSpace"
                    className="text-sm font-medium mb-2 block"
                  >
                    カスタムパラメータ空間 (JSON)
                  </label>
                  <textarea
                    id="customParameterSpace"
                    value={customParameterSpace}
                    onChange={(e: React.ChangeEvent<HTMLTextAreaElement>) =>
                      setCustomParameterSpace(e.target.value)
                    }
                    placeholder="パラメータ空間をJSON形式で入力してください"
                    className="font-mono text-sm w-full flex-grow p-3 bg-gray-800 border border-secondary-700 text-white rounded-md focus:ring-2 focus:ring-blue-500 focus:border-transparent"
                    rows={15}
                  />
                </div>
              )}
              {useDefaultParams && defaultParameterSpace && (
                <div className="flex-grow flex flex-col">
                  <CollapsibleJson
                    data={defaultParameterSpace}
                    title="デフォルトパラメータ空間 (プレビュー)"
                    defaultExpanded={true}
                    theme="dark"
                  />
                </div>
              )}
            </div>
          </Card>
        </div>

        {/* プロファイル保存設定 */}
        <div className="md:col-span-2">
          <Card className="p-4">
            <h3 className="text-lg font-semibold mb-4 text-white">
              プロファイル保存設定
            </h3>

            <div className="space-y-4">
              {/* プロファイル保存チェックボックス */}
              <div className="flex items-center space-x-2">
                <input
                  type="checkbox"
                  id="save-as-profile"
                  checked={saveAsProfile}
                  onChange={(e) => setSaveAsProfile(e.target.checked)}
                  className="w-4 h-4 text-blue-600 bg-gray-100 border-gray-300 rounded focus:ring-blue-500 dark:focus:ring-blue-600 dark:ring-offset-gray-800 focus:ring-2 dark:bg-gray-700 dark:border-gray-600"
                />
                <label
                  htmlFor="save-as-profile"
                  className="text-sm font-medium text-white"
                >
                  最適化結果をプロファイルとして保存
                </label>
              </div>

              {/* プロファイル名入力 */}
              {saveAsProfile && (
                <>
                  <InputField
                    label="プロファイル名 *"
                    type="text"
                    value={profileName}
                    onChange={setProfileName}
                    placeholder="プロファイル名を入力してください"
                    required={saveAsProfile}
                    className="text-sm"
                  />

                  <div className="space-y-2">
                    <label className="block text-sm font-medium text-white">
                      プロファイル説明（オプション）
                    </label>
                    <textarea
                      value={profileDescription}
                      onChange={(e) => setProfileDescription(e.target.value)}
                      placeholder="プロファイルの説明を入力してください"
                      rows={3}
                      className="w-full px-3 py-2 text-sm border border-gray-300 rounded-md shadow-sm focus:outline-none focus:ring-blue-500 focus:border-blue-500 dark:bg-gray-700 dark:border-gray-600 dark:text-white dark:placeholder-gray-400"
                    />
                  </div>
                </>
              )}
            </div>
          </Card>
        </div>

        {/* 実行ボタン */}
        <div className="md:col-span-2 flex justify-end">
          <ActionButton
            type="submit"
            disabled={isLoading}
            className="px-6 py-2"
          >
            {isLoading ? "最適化実行中..." : "ベイジアン最適化を実行"}
          </ActionButton>
        </div>
      </form>
      <InfoModal
        isOpen={isInfoModalOpen}
        onClose={() => setIsInfoModalOpen(false)}
        title={modalContent.title}
      >
        {modalContent.content}
      </InfoModal>
    </>
  );
};

export default BayesianOptimizationForm;
