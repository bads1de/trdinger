import React, { useEffect } from "react";
import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card";
import { Button } from "@/components/ui/button";
import { Badge } from "@/components/ui/badge";
import { Separator } from "@/components/ui/separator";
import LoadingSpinner from "@/components/common/LoadingSpinner";
import ErrorDisplay from "@/components/common/ErrorDisplay";
import { useModelManagement, ModelInfo } from "@/hooks/useModelManagement";
import {
  Database,
  Download,
  RefreshCw,
  CheckCircle,
  Clock,
  BarChart3,
  HardDrive,
  Calendar,
} from "lucide-react";

interface ModelManagementProps {
  className?: string;
}

const ModelCard: React.FC<{
  model: ModelInfo;
  isCurrentModel: boolean;
  onLoad: (modelName: string) => void;
  loading: boolean;
}> = ({ model, isCurrentModel, onLoad, loading }) => {
  const formatDate = (dateString: string) => {
    return new Date(dateString).toLocaleString("ja-JP", {
      year: "numeric",
      month: "2-digit",
      day: "2-digit",
      hour: "2-digit",
      minute: "2-digit",
    });
  };

  const getModelTypeColor = (modelType: string) => {
    if (!modelType) {
      return "bg-gray-500/20 text-gray-400 border-gray-500/30";
    }
    switch (modelType.toLowerCase()) {
      case "lightgbm":
        return "bg-green-500/20 text-green-400 border-green-500/30";
      case "xgboost":
        return "bg-blue-500/20 text-blue-400 border-blue-500/30";
      case "catboost":
        return "bg-purple-500/20 text-purple-400 border-purple-500/30";
      case "tabnet":
        return "bg-orange-500/20 text-orange-400 border-orange-500/30";
      default:
        return "bg-gray-500/20 text-gray-400 border-gray-500/30";
    }
  };

  const getTrainerTypeColor = (trainerType: string) => {
    if (!trainerType) {
      return "bg-gray-500/20 text-gray-400 border-gray-500/30";
    }
    switch (trainerType.toLowerCase()) {
      case "single_model":
        return "bg-cyan-500/20 text-cyan-400 border-cyan-500/30";
      case "ensembletrainer":
        return "bg-yellow-500/20 text-yellow-400 border-yellow-500/30";
      default:
        return "bg-gray-500/20 text-gray-400 border-gray-500/30";
    }
  };

  return (
    <Card
      className={`bg-gray-900/50 border-gray-800 transition-all duration-300 hover:border-cyan-500/60 ${
        isCurrentModel ? "border-cyan-500 bg-cyan-500/5" : ""
      }`}
    >
      <CardHeader className="pb-3">
        <div className="flex items-center justify-between">
          <CardTitle className="text-lg font-semibold text-white flex items-center gap-2">
            <Database className="h-5 w-5 text-cyan-400" />
            {model.name}
            {isCurrentModel && (
              <Badge className="bg-cyan-500/20 text-cyan-400 border-cyan-500/30">
                <CheckCircle className="h-3 w-3 mr-1" />
                読み込み中
              </Badge>
            )}
          </CardTitle>
          <Button
            variant="outline"
            size="sm"
            onClick={() => onLoad(model.name)}
            disabled={loading || isCurrentModel}
            className="border-gray-700 hover:border-cyan-500"
          >
            <Download className="h-4 w-4 mr-1" />
            {isCurrentModel ? "読み込み済み" : "読み込み"}
          </Button>
        </div>
      </CardHeader>

      <CardContent className="space-y-3">
        {/* モデル情報 */}
        <div className="flex flex-wrap gap-2">
          <Badge className={getModelTypeColor(model.model_type)}>
            {model.model_type.toUpperCase()}
          </Badge>
          <Badge className={getTrainerTypeColor(model.trainer_type)}>
            {model.trainer_type === "single_model" ? "単一モデル" : "アンサンブル"}
          </Badge>
        </div>

        {/* 統計情報 */}
        <div className="grid grid-cols-2 gap-4 text-sm">
          <div className="flex items-center gap-2 text-gray-300">
            <BarChart3 className="h-4 w-4 text-cyan-400" />
            <span>特徴量: {model.feature_count}個</span>
          </div>
          <div className="flex items-center gap-2 text-gray-300">
            <HardDrive className="h-4 w-4 text-cyan-400" />
            <span>サイズ: {model.size_mb.toFixed(1)}MB</span>
          </div>
        </div>

        {/* 特徴量重要度情報 */}
        {model.has_feature_importance && (
          <div className="flex items-center gap-2 text-sm text-green-400">
            <BarChart3 className="h-4 w-4" />
            <span>特徴量重要度: {model.feature_importance_count}個</span>
          </div>
        )}

        <Separator className="bg-gray-700" />

        {/* 更新日時 */}
        <div className="flex items-center gap-2 text-xs text-gray-400">
          <Calendar className="h-3 w-3" />
          <span>更新: {formatDate(model.modified_at)}</span>
        </div>
      </CardContent>
    </Card>
  );
};

export default function ModelManagement({ className = "" }: ModelManagementProps) {
  const {
    models,
    currentModel,
    loading,
    error,
    loadModel,
    refreshModels,
  } = useModelManagement();

  useEffect(() => {
    refreshModels();
  }, [refreshModels]);

  if (error) {
    return (
      <Card className={`bg-gray-900/50 border-gray-800 ${className}`}>
        <CardHeader>
          <CardTitle className="text-white flex items-center gap-2">
            <Database className="h-5 w-5 text-cyan-400" />
            モデル管理
          </CardTitle>
        </CardHeader>
        <CardContent>
          <ErrorDisplay message={error} />
        </CardContent>
      </Card>
    );
  }

  return (
    <Card className={`bg-gray-900/50 border-gray-800 ${className}`}>
      <CardHeader>
        <div className="flex items-center justify-between">
          <CardTitle className="text-white flex items-center gap-2">
            <Database className="h-5 w-5 text-cyan-400" />
            モデル管理
            <Badge variant="outline" className="border-gray-700 text-gray-300">
              {models.length}個
            </Badge>
          </CardTitle>
          <Button
            variant="outline"
            size="sm"
            onClick={refreshModels}
            disabled={loading}
            className="border-gray-700 hover:border-cyan-500"
          >
            <RefreshCw className={`h-4 w-4 ${loading ? "animate-spin" : ""}`} />
          </Button>
        </div>
      </CardHeader>

      <CardContent>
        {/* 現在のモデル情報 */}
        {currentModel && currentModel.loaded && (
          <div className="mb-6 p-4 bg-cyan-500/10 border border-cyan-500/30 rounded-lg">
            <h3 className="text-sm font-medium text-cyan-400 mb-2 flex items-center gap-2">
              <CheckCircle className="h-4 w-4" />
              現在読み込まれているモデル
            </h3>
            <div className="flex flex-wrap gap-2 text-sm">
              <Badge className="bg-cyan-500/20 text-cyan-400 border-cyan-500/30">
                {currentModel.trainer_type}
              </Badge>
              <Badge className="bg-gray-500/20 text-gray-400 border-gray-500/30">
                {currentModel.model_type}
              </Badge>
              {currentModel.has_feature_importance && (
                <Badge className="bg-green-500/20 text-green-400 border-green-500/30">
                  特徴量重要度: {currentModel.feature_importance_count}個
                </Badge>
              )}
            </div>
          </div>
        )}

        {loading && models.length === 0 ? (
          <div className="flex justify-center py-8">
            <LoadingSpinner />
          </div>
        ) : models.length === 0 ? (
          <div className="text-center py-8 text-gray-400">
            <Database className="h-12 w-12 mx-auto mb-4 opacity-50" />
            <p>利用可能なモデルがありません</p>
          </div>
        ) : (
          <div className="space-y-4">
            {models.map((model) => (
              <ModelCard
                key={model.path}
                model={model}
                isCurrentModel={
                  currentModel?.loaded &&
                  currentModel.trainer_type === model.trainer_type &&
                  currentModel.model_type === model.model_type
                }
                onLoad={loadModel}
                loading={loading}
              />
            ))}
          </div>
        )}
      </CardContent>
    </Card>
  );
}
