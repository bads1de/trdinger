"use client";

import React, { useState, useEffect } from "react";
import ActionButton from "@/components/common/ActionButton";
import { Card } from "@/components/ui/card";
import { InputField } from "@/components/common/InputField";
import { SelectField } from "@/components/common/SelectField";
import {
  BayesianOptimizationConfig,
  ParameterSpace,
  DefaultParameterSpaceResponse,
} from "@/types/bayesian-optimization";
import { BacktestConfig } from "@/types/backtest";

interface BayesianOptimizationFormProps {
  onGAOptimization: (config: BayesianOptimizationConfig) => void;
  onMLOptimization: (config: BayesianOptimizationConfig) => void;
  isLoading?: boolean;
  currentBacktestConfig?: BacktestConfig | null;
}

const BayesianOptimizationForm: React.FC<BayesianOptimizationFormProps> = ({
  onGAOptimization,
  onMLOptimization,
  isLoading = false,
  currentBacktestConfig = null,
}) => {
  const [optimizationType, setOptimizationType] = useState<"ga" | "ml">("ga");
  const [experimentName, setExperimentName] = useState("");
  const [modelType, setModelType] = useState("lightgbm");
  const [nCalls, setNCalls] = useState(50);
  const [useDefaultParams, setUseDefaultParams] = useState(true);
  const [customParameterSpace, setCustomParameterSpace] = useState("");
  const [defaultParameterSpace, setDefaultParameterSpace] = useState<Record<string, ParameterSpace> | null>(null);
  const [acquisitionFunction, setAcquisitionFunction] = useState("EI");
  const [nInitialPoints, setNInitialPoints] = useState(10);
  const [randomState, setRandomState] = useState(42);

  // デフォルトパラメータ空間を取得
  useEffect(() => {
    const fetchDefaultParameterSpace = async () => {
      try {
        const type = optimizationType === "ga" ? "ga" : modelType;
        const response = await fetch(`/api/bayesian-optimization/parameter-spaces/${type}`);
        const data: DefaultParameterSpaceResponse = await response.json();
        
        if (data.success && data.parameter_space) {
          setDefaultParameterSpace(data.parameter_space);
          setCustomParameterSpace(JSON.stringify(data.parameter_space, null, 2));
        }
      } catch (error) {
        console.error("デフォルトパラメータ空間の取得に失敗:", error);
      }
    };

    fetchDefaultParameterSpace();
  }, [optimizationType, modelType]);

  const handleSubmit = (e: React.FormEvent) => {
    e.preventDefault();

    if (!currentBacktestConfig && optimizationType === "ga") {
      alert("GAパラメータ最適化にはバックテスト設定が必要です");
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
    };

    if (optimizationType === "ga") {
      config.experiment_name = experimentName;
      config.base_config = currentBacktestConfig;
      onGAOptimization(config);
    } else {
      config.model_type = modelType;
      onMLOptimization(config);
    }
  };

  return (
    <form onSubmit={handleSubmit} className="space-y-6">
      {/* 最適化タイプ選択 */}
      <Card className="p-4">
        <label className="text-sm font-medium mb-3 block">最適化タイプ</label>
        <div className="flex space-x-4">
          <label className="flex items-center">
            <input
              type="radio"
              value="ga"
              checked={optimizationType === "ga"}
              onChange={(e) => setOptimizationType(e.target.value as "ga" | "ml")}
              className="mr-2"
            />
            GAパラメータ最適化
          </label>
          <label className="flex items-center">
            <input
              type="radio"
              value="ml"
              checked={optimizationType === "ml"}
              onChange={(e) => setOptimizationType(e.target.value as "ga" | "ml")}
              className="mr-2"
            />
            MLハイパーパラメータ最適化
          </label>
        </div>
      </Card>

      {/* GA最適化設定 */}
      {optimizationType === "ga" && (
        <Card className="p-4">
          <InputField
            label="実験名"
            value={experimentName}
            onChange={setExperimentName}
            placeholder="ベイジアン最適化実験"
            required
          />
        </Card>
      )}

      {/* ML最適化設定 */}
      {optimizationType === "ml" && (
        <Card className="p-4">
          <SelectField
            label="モデルタイプ"
            value={modelType}
            onChange={setModelType}
            options={[
              { value: "lightgbm", label: "LightGBM" },
              { value: "xgboost", label: "XGBoost" },
              { value: "random_forest", label: "Random Forest" },
            ]}
          />
        </Card>
      )}

      {/* 最適化パラメータ */}
      <Card className="p-4">
        <h3 className="text-lg font-medium mb-4">最適化パラメータ</h3>
        
        <div className="grid grid-cols-1 md:grid-cols-3 gap-4 mb-4">
          <div>
            <InputField
              label="試行回数"
              type="number"
              value={nCalls}
              onChange={setNCalls}
              min={10}
              max={200}
            />
          </div>
          
          <div>
            <SelectField
              label="獲得関数"
              value={acquisitionFunction}
              onChange={setAcquisitionFunction}
              options={[
                { value: "EI", label: "Expected Improvement" },
                { value: "PI", label: "Probability of Improvement" },
                { value: "UCB", label: "Upper Confidence Bound" },
              ]}
            />
          </div>
          
          <div>
            <InputField
              label="初期ランダム試行数"
              type="number"
              value={nInitialPoints}
              onChange={setNInitialPoints}
              min={5}
              max={50}
            />
          </div>
        </div>

        <div className="mb-4">
          <InputField
            label="乱数シード"
            type="number"
            value={randomState}
            onChange={setRandomState}
          />
        </div>
      </Card>

      {/* パラメータ空間設定 */}
      <Card className="p-4">
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
            <label htmlFor="useDefaultParams" className="text-sm font-medium text-gray-300">
              デフォルトパラメータ空間を使用
            </label>
          </div>
        </div>

        {!useDefaultParams && (
          <div>
            <label htmlFor="customParameterSpace" className="text-sm font-medium mb-2 block">
              カスタムパラメータ空間 (JSON)
            </label>
            <textarea
              id="customParameterSpace"
              value={customParameterSpace}
              onChange={(e: React.ChangeEvent<HTMLTextAreaElement>) => setCustomParameterSpace(e.target.value)}
              rows={10}
              placeholder="パラメータ空間をJSON形式で入力してください"
              className="font-mono text-sm w-full p-3 bg-gray-800 border border-secondary-700 text-white rounded-md focus:ring-2 focus:ring-blue-500 focus:border-transparent"
            />
          </div>
        )}

        {useDefaultParams && defaultParameterSpace && (
          <div>
            <label className="text-sm font-medium mb-2 block">
              デフォルトパラメータ空間 (プレビュー)
            </label>
            <pre className="bg-gray-100 p-3 rounded text-sm overflow-auto max-h-40">
              {JSON.stringify(defaultParameterSpace, null, 2)}
            </pre>
          </div>
        )}
      </Card>

      {/* 実行ボタン */}
      <div className="flex justify-end">
        <ActionButton
          type="submit"
          disabled={isLoading}
          className="px-6 py-2"
        >
          {isLoading ? "最適化実行中..." : "ベイジアン最適化を実行"}
        </ActionButton>
      </div>
    </form>
  );
};

export default BayesianOptimizationForm;
