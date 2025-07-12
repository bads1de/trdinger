"use client";

import React, { useState, useEffect } from "react";
import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card";
import ActionButton from "@/components/common/ActionButton";
import { InputField } from "@/components/common/InputField";
import { Alert, AlertDescription } from "@/components/ui/alert";
import ErrorDisplay from "@/components/common/ErrorDisplay";
import LoadingSpinner from "@/components/common/LoadingSpinner";
import {
  Settings,
  Save,
  RotateCcw,
  Database,
  Clock,
  Brain,
  Trash2,
} from "lucide-react";

interface MLConfig {
  data_processing: {
    max_ohlcv_rows: number;
    max_feature_rows: number;
    feature_calculation_timeout: number;
    model_training_timeout: number;
  };
  model: {
    model_save_path: string;
    max_model_versions: number;
    model_retention_days: number;
  };
  lightgbm: {
    learning_rate: number;
    num_leaves: number;
    feature_fraction: number;
    bagging_fraction: number;
    num_boost_round: number;
    early_stopping_rounds: number;
  };
  training: {
    train_test_split: number;
    prediction_horizon: number;
    threshold_up: number;
    threshold_down: number;
    min_training_samples: number;
  };
  prediction: {
    default_up_prob: number;
    default_down_prob: number;
    default_range_prob: number;
  };
}

/**
 * ML設定コンポーネント
 *
 * ML関連の設定を変更・管理するコンポーネント
 */
export default function MLSettings() {
  const [config, setConfig] = useState<MLConfig | null>(null);
  const [isLoading, setIsLoading] = useState(true);
  const [isSaving, setIsSaving] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [successMessage, setSuccessMessage] = useState<string | null>(null);

  useEffect(() => {
    fetchConfig();
  }, []);

  const fetchConfig = async () => {
    try {
      setIsLoading(true);
      const response = await fetch("/api/ml/config");
      if (!response.ok) {
        throw new Error("設定の取得に失敗しました");
      }
      const data = await response.json();
      setConfig(data);
      setError(null);
    } catch (err) {
      setError(err instanceof Error ? err.message : "エラーが発生しました");
    } finally {
      setIsLoading(false);
    }
  };

  const saveConfig = async () => {
    if (!config) return;

    try {
      setIsSaving(true);
      setError(null);
      setSuccessMessage(null);

      const response = await fetch("/api/ml/config", {
        method: "PUT",
        headers: {
          "Content-Type": "application/json",
        },
        body: JSON.stringify(config),
      });

      if (!response.ok) {
        throw new Error("設定の保存に失敗しました");
      }

      setSuccessMessage("設定が正常に保存されました");
      setTimeout(() => setSuccessMessage(null), 3000);
    } catch (err) {
      setError(err instanceof Error ? err.message : "保存エラーが発生しました");
    } finally {
      setIsSaving(false);
    }
  };

  const resetToDefaults = async () => {
    if (!confirm("設定をデフォルト値にリセットしますか？")) {
      return;
    }

    try {
      const response = await fetch("/api/ml/config/reset", {
        method: "POST",
      });

      if (!response.ok) {
        throw new Error("設定のリセットに失敗しました");
      }

      await fetchConfig();
      setSuccessMessage("設定がデフォルト値にリセットされました");
      setTimeout(() => setSuccessMessage(null), 3000);
    } catch (err) {
      setError(
        err instanceof Error ? err.message : "リセットエラーが発生しました"
      );
    }
  };

  const cleanupOldModels = async () => {
    if (
      !confirm("古いモデルファイルを削除しますか？この操作は取り消せません。")
    ) {
      return;
    }

    try {
      const response = await fetch("/api/ml/models/cleanup", {
        method: "POST",
      });

      if (!response.ok) {
        throw new Error("モデルクリーンアップに失敗しました");
      }

      setSuccessMessage("古いモデルファイルが削除されました");
      setTimeout(() => setSuccessMessage(null), 3000);
    } catch (err) {
      setError(
        err instanceof Error
          ? err.message
          : "クリーンアップエラーが発生しました"
      );
    }
  };

  const updateConfig = (section: keyof MLConfig, key: string, value: any) => {
    if (!config) return;

    setConfig((prev) => ({
      ...prev!,
      [section]: {
        ...prev![section],
        [key]: value,
      },
    }));
  };

  if (isLoading) {
    return <LoadingSpinner text="設定を読み込んでいます..." />;
  }

  if (error) {
    return <ErrorDisplay message={error} />;
  }

  if (!config) {
    return (
      <div className="text-center p-8 text-gray-500">
        <Settings className="h-12 w-12 mx-auto mb-4 text-gray-300" />
        <p>設定を読み込めませんでした</p>
      </div>
    );
  }

  return (
    <div className="space-y-6">
      {/* 成功メッセージ */}
      {successMessage && (
        <Alert className="border-green-200 bg-green-50">
          <AlertDescription className="text-green-800">
            {successMessage}
          </AlertDescription>
        </Alert>
      )}

      {/* データ処理設定 */}
      <Card>
        <CardHeader>
          <CardTitle className="flex items-center space-x-2">
            <Database className="h-5 w-5" />
            <span>データ処理設定</span>
          </CardTitle>
        </CardHeader>
        <CardContent className="space-y-4">
          <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
            <div>
              <InputField
                label="最大OHLCV行数"
                type="number"
                value={config.data_processing.max_ohlcv_rows}
                onChange={(value) =>
                  updateConfig("data_processing", "max_ohlcv_rows", value)
                }
              />
            </div>
            <div>
              <InputField
                label="最大特徴量行数"
                type="number"
                value={config.data_processing.max_feature_rows}
                onChange={(value) =>
                  updateConfig("data_processing", "max_feature_rows", value)
                }
              />
            </div>
            <div>
              <InputField
                label="特徴量計算タイムアウト（秒）"
                type="number"
                value={config.data_processing.feature_calculation_timeout}
                onChange={(value) =>
                  updateConfig(
                    "data_processing",
                    "feature_calculation_timeout",
                    value
                  )
                }
              />
            </div>
            <div>
              <InputField
                label="モデル学習タイムアウト（秒）"
                type="number"
                value={config.data_processing.model_training_timeout}
                onChange={(value) =>
                  updateConfig(
                    "data_processing",
                    "model_training_timeout",
                    value
                  )
                }
              />
            </div>
          </div>
        </CardContent>
      </Card>

      {/* LightGBM設定 */}
      <Card>
        <CardHeader>
          <CardTitle className="flex items-center space-x-2">
            <Brain className="h-5 w-5" />
            <span>LightGBM設定</span>
          </CardTitle>
        </CardHeader>
        <CardContent className="space-y-4">
          <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
            <div>
              <InputField
                label="学習率"
                type="number"
                step={0.01}
                value={config.lightgbm.learning_rate}
                onChange={(value) =>
                  updateConfig("lightgbm", "learning_rate", value)
                }
              />
            </div>
            <div>
              <InputField
                label="葉の数"
                type="number"
                value={config.lightgbm.num_leaves}
                onChange={(value) =>
                  updateConfig("lightgbm", "num_leaves", value)
                }
              />
            </div>
            <div>
              <InputField
                label="特徴量サンプリング率"
                type="number"
                step={0.1}
                min={0.1}
                max={1.0}
                value={config.lightgbm.feature_fraction}
                onChange={(value) =>
                  updateConfig("lightgbm", "feature_fraction", value)
                }
              />
            </div>
            <div>
              <InputField
                label="バギング率"
                type="number"
                step={0.1}
                min={0.1}
                max={1.0}
                value={config.lightgbm.bagging_fraction}
                onChange={(value) =>
                  updateConfig("lightgbm", "bagging_fraction", value)
                }
              />
            </div>
          </div>
        </CardContent>
      </Card>

      {/* 学習設定 */}
      <Card>
        <CardHeader>
          <CardTitle className="flex items-center space-x-2">
            <Clock className="h-5 w-5" />
            <span>学習設定</span>
          </CardTitle>
        </CardHeader>
        <CardContent className="space-y-4">
          <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
            <div>
              <InputField
                label="学習/テスト分割比率"
                type="number"
                step={0.1}
                min={0.1}
                max={0.9}
                value={config.training.train_test_split}
                onChange={(value) =>
                  updateConfig("training", "train_test_split", value)
                }
              />
            </div>
            <div>
              <InputField
                label="予測期間（時間）"
                type="number"
                value={config.training.prediction_horizon}
                onChange={(value) =>
                  updateConfig("training", "prediction_horizon", value)
                }
              />
            </div>
            <div>
              <InputField
                label="上昇判定閾値"
                type="number"
                step={0.01}
                value={config.training.threshold_up}
                onChange={(value) =>
                  updateConfig("training", "threshold_up", value)
                }
              />
            </div>
            <div>
              <InputField
                label="下落判定閾値"
                type="number"
                step={0.01}
                value={config.training.threshold_down}
                onChange={(value) =>
                  updateConfig("training", "threshold_down", value)
                }
              />
            </div>
          </div>
        </CardContent>
      </Card>

      {/* アクションボタン */}
      <div className="flex flex-wrap gap-4">
        <ActionButton
          onClick={saveConfig}
          disabled={isSaving}
          variant="primary"
          icon={<Save className="h-4 w-4" />}
          loading={isSaving}
          loadingText="保存中..."
        >
          設定を保存
        </ActionButton>

        <ActionButton
          onClick={resetToDefaults}
          variant="secondary"
          icon={<RotateCcw className="h-4 w-4" />}
        >
          デフォルトにリセット
        </ActionButton>

        <ActionButton
          onClick={cleanupOldModels}
          variant="warning"
          icon={<Trash2 className="h-4 w-4" />}
        >
          古いモデルを削除
        </ActionButton>
      </div>
    </div>
  );
}
