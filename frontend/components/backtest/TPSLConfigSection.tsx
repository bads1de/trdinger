/**
 * TP/SL自動決定設定セクション
 *
 * 複雑なTP/SL範囲設定を簡素化されたプリセット選択に置き換える
 * 新しいコンポーネントです。
 */

import React, { useState } from "react";
import {
  GAConfig,
  TPSLStrategy,
  VolatilitySensitivity,
} from "../../types/optimization";
import {
  PRESET_OPTIONS,
  TPSL_STRATEGY_OPTIONS,
  VOLATILITY_SENSITIVITY_OPTIONS,
  getPresetConfig,
  convertPresetToGAConfig,
  getPresetDescription,
  validateCustomTPSLConfig,
} from "../../utils/tpslPresets";
import { InputField } from "../common/InputField";

interface TPSLConfigSectionProps {
  config: GAConfig;
  setConfig: React.Dispatch<React.SetStateAction<GAConfig>>;
}

const TPSLConfigSection: React.FC<TPSLConfigSectionProps> = ({
  config,
  setConfig,
}) => {
  const [selectedPreset, setSelectedPreset] = useState<string>("balanced");
  const [showAdvanced, setShowAdvanced] = useState<boolean>(false);
  const [showPresetDescription, setShowPresetDescription] =
    useState<boolean>(false);

  // プリセット選択の処理
  const handlePresetChange = (presetName: string) => {
    setSelectedPreset(presetName);

    if (presetName === "custom") {
      // カスタム設定の場合は詳細設定を表示
      setShowAdvanced(true);
      setConfig((prev) => ({
        ...prev,
        ga_config: {
          ...prev.ga_config,
          enable_advanced_tpsl: true,
          tpsl_strategy: "auto_optimal",
          max_risk_per_trade: 0.03,
          preferred_risk_reward_ratio: 2.0,
          volatility_sensitivity: "medium",
        },
      }));
    } else {
      // プリセット設定を適用
      const preset = getPresetConfig(presetName);
      if (preset) {
        const presetConfig = convertPresetToGAConfig(preset);
        setConfig((prev) => ({
          ...prev,
          ga_config: {
            ...prev.ga_config,
            ...presetConfig,
            // 従来の範囲設定も更新（後方互換性のため）
            stop_loss_range: [
              preset.max_risk_per_trade * 0.5,
              preset.max_risk_per_trade * 1.5,
            ],
            take_profit_range: [
              preset.max_risk_per_trade *
                preset.preferred_risk_reward_ratio *
                0.8,
              preset.max_risk_per_trade *
                preset.preferred_risk_reward_ratio *
                1.5,
            ],
          },
        }));
      }
      setShowAdvanced(false);
    }
  };

  // カスタム設定の更新
  const updateCustomConfig = (field: string, value: any) => {
    setConfig((prev) => ({
      ...prev,
      ga_config: {
        ...prev.ga_config,
        [field]: value,
      },
    }));
  };

  // バリデーション
  const validateCurrentConfig = () => {
    if (selectedPreset === "custom") {
      const customConfig = {
        strategy: config.ga_config.tpsl_strategy as TPSLStrategy,
        max_risk_per_trade: config.ga_config.max_risk_per_trade || 0.03,
        preferred_risk_reward_ratio:
          config.ga_config.preferred_risk_reward_ratio || 2.0,
        volatility_sensitivity:
          (config.ga_config.volatility_sensitivity as VolatilitySensitivity) ||
          "medium",
      };
      return validateCustomTPSLConfig(customConfig);
    }
    return [];
  };

  const validationErrors = validateCurrentConfig();

  return (
    <div className="space-y-6">
      {/* プリセット選択 */}
      <div>
        <label className="block text-sm font-medium text-gray-700 mb-2">
          TP/SL自動決定設定
        </label>
        <select
          value={selectedPreset}
          onChange={(e) => handlePresetChange(e.target.value)}
          className="w-full px-3 py-2 border border-gray-300 rounded-md focus:outline-none focus:ring-2 focus:ring-blue-500 text-black bg-white"
        >
          {PRESET_OPTIONS.map((option) => (
            <option key={option.value} value={option.value}>
              {option.label}
            </option>
          ))}
        </select>

        {/* プリセット説明の表示/非表示ボタン */}
        {selectedPreset !== "custom" && (
          <button
            type="button"
            onClick={() => setShowPresetDescription(!showPresetDescription)}
            className="mt-2 text-sm text-blue-600 hover:text-blue-800"
          >
            {showPresetDescription ? "説明を隠す" : "設定詳細を表示"}
          </button>
        )}

        {/* プリセット説明 */}
        {showPresetDescription && selectedPreset !== "custom" && (
          <div className="mt-3 p-4 bg-blue-50 rounded-md">
            <pre className="text-sm text-gray-700 whitespace-pre-wrap">
              {getPresetDescription(selectedPreset)}
            </pre>
          </div>
        )}
      </div>

      {/* カスタム設定（詳細設定） */}
      {selectedPreset === "custom" && (
        <div className="space-y-4 p-4 bg-gray-50 rounded-md">
          <h4 className="font-medium text-gray-900">カスタム設定</h4>

          {/* TP/SL戦略選択 */}
          <div>
            <label className="block text-sm font-medium text-gray-700 mb-1">
              TP/SL決定戦略
            </label>
            <select
              value={config.ga_config.tpsl_strategy || "auto_optimal"}
              onChange={(e) =>
                updateCustomConfig("tpsl_strategy", e.target.value)
              }
              className="w-full px-3 py-2 border border-gray-300 rounded-md focus:outline-none focus:ring-2 focus:ring-blue-500"
            >
              {TPSL_STRATEGY_OPTIONS.map((option) => (
                <option key={option.value} value={option.value}>
                  {option.label}
                </option>
              ))}
            </select>
          </div>

          {/* 最大リスク設定 */}
          <InputField
            label="1取引あたりの最大リスク (%)"
            type="number"
            value={(config.ga_config.max_risk_per_trade || 0.03) * 100}
            onChange={(value: number) =>
              updateCustomConfig("max_risk_per_trade", value / 100)
            }
            min={0.5}
            max={10}
            step={0.1}
            required
          />

          {/* リスクリワード比設定 */}
          <InputField
            label="希望するリスクリワード比 (1:X)"
            type="number"
            value={config.ga_config.preferred_risk_reward_ratio || 2.0}
            onChange={(value: number) =>
              updateCustomConfig("preferred_risk_reward_ratio", value)
            }
            min={1.0}
            max={5.0}
            step={0.1}
            required
          />

          {/* ボラティリティ感度設定 */}
          <div>
            <label className="block text-sm font-medium text-gray-700 mb-1">
              ボラティリティ感度
            </label>
            <select
              value={config.ga_config.volatility_sensitivity || "medium"}
              onChange={(e) =>
                updateCustomConfig("volatility_sensitivity", e.target.value)
              }
              className="w-full px-3 py-2 border border-gray-300 rounded-md focus:outline-none focus:ring-2 focus:ring-blue-500"
            >
              {VOLATILITY_SENSITIVITY_OPTIONS.map((option) => (
                <option key={option.value} value={option.value}>
                  {option.label}
                </option>
              ))}
            </select>
          </div>

          {/* バリデーションエラー表示 */}
          {validationErrors.length > 0 && (
            <div className="p-3 bg-red-50 border border-red-200 rounded-md">
              <h5 className="text-sm font-medium text-red-800 mb-1">
                設定エラー:
              </h5>
              <ul className="text-sm text-red-700 list-disc list-inside">
                {validationErrors.map((error, index) => (
                  <li key={index}>{error}</li>
                ))}
              </ul>
            </div>
          )}
        </div>
      )}

      {/* 高度な設定（統計的・ボラティリティベース） */}
      {(selectedPreset === "custom" ||
        selectedPreset === "statistical" ||
        selectedPreset === "volatility_adaptive") && (
        <div>
          <button
            type="button"
            onClick={() => setShowAdvanced(!showAdvanced)}
            className="text-sm text-gray-600 hover:text-gray-800"
          >
            {showAdvanced ? "▼ 高度な設定を隠す" : "▶ 高度な設定を表示"}
          </button>

          {showAdvanced && (
            <div className="mt-3 space-y-4 p-4 bg-gray-50 rounded-md">
              <h4 className="font-medium text-gray-900">高度な設定</h4>

              {/* 統計的設定 */}
              {(config.ga_config.tpsl_strategy === "statistical" ||
                selectedPreset === "statistical") && (
                <div className="space-y-3">
                  <h5 className="text-sm font-medium text-gray-800">
                    統計的最適化設定
                  </h5>

                  <InputField
                    label="学習期間 (日)"
                    type="number"
                    value={config.ga_config.statistical_lookback_days || 365}
                    onChange={(value: number) =>
                      updateCustomConfig("statistical_lookback_days", value)
                    }
                    min={30}
                    max={1095}
                    step={1}
                  />

                  <InputField
                    label="最小サンプル数"
                    type="number"
                    value={config.ga_config.statistical_min_samples || 50}
                    onChange={(value: number) =>
                      updateCustomConfig("statistical_min_samples", value)
                    }
                    min={10}
                    max={500}
                    step={1}
                  />
                </div>
              )}

              {/* ボラティリティベース設定 */}
              {(config.ga_config.tpsl_strategy === "volatility_adaptive" ||
                selectedPreset === "volatility_adaptive") && (
                <div className="space-y-3">
                  <h5 className="text-sm font-medium text-gray-800">
                    ボラティリティベース設定
                  </h5>

                  <InputField
                    label="ATR計算期間"
                    type="number"
                    value={config.ga_config.atr_period || 14}
                    onChange={(value: number) =>
                      updateCustomConfig("atr_period", value)
                    }
                    min={5}
                    max={50}
                    step={1}
                  />

                  <div className="grid grid-cols-2 gap-3">
                    <InputField
                      label="SL用ATR倍率"
                      type="number"
                      value={config.ga_config.atr_multiplier_sl || 2.0}
                      onChange={(value: number) =>
                        updateCustomConfig("atr_multiplier_sl", value)
                      }
                      min={0.5}
                      max={5.0}
                      step={0.1}
                    />

                    <InputField
                      label="TP用ATR倍率"
                      type="number"
                      value={config.ga_config.atr_multiplier_tp || 3.0}
                      onChange={(value: number) =>
                        updateCustomConfig("atr_multiplier_tp", value)
                      }
                      min={1.0}
                      max={10.0}
                      step={0.1}
                    />
                  </div>
                </div>
              )}
            </div>
          )}
        </div>
      )}

      {/* 設定概要表示 */}
      <div className="p-3 bg-green-50 border border-green-200 rounded-md">
        <h5 className="text-sm font-medium text-green-800 mb-1">
          現在の設定概要:
        </h5>
        <div className="text-sm text-green-700">
          <p>• 戦略: {config.ga_config.tpsl_strategy || "auto_optimal"}</p>
          <p>
            • 最大リスク:{" "}
            {((config.ga_config.max_risk_per_trade || 0.03) * 100).toFixed(1)}%
          </p>
          <p>
            • リスクリワード比: 1:
            {config.ga_config.preferred_risk_reward_ratio || 2.0}
          </p>
          <p>
            • ボラティリティ感度:{" "}
            {config.ga_config.volatility_sensitivity || "medium"}
          </p>
        </div>
      </div>
    </div>
  );
};

export default TPSLConfigSection;
