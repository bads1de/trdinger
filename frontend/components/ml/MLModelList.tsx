"use client";

import React, { useState, useEffect } from "react";
import { formatDateTime, formatFileSize } from "@/utils/formatters";
import { Card, CardContent } from "@/components/ui/card";
import ActionButton from "@/components/common/ActionButton";
import { Badge } from "@/components/ui/badge";
import ErrorDisplay from "@/components/common/ErrorDisplay";
import LoadingSpinner from "@/components/common/LoadingSpinner";
import { useMLModels, MLModel } from "@/hooks/useMLModels";
import {
  Trash2,
  Eye,
  Calendar,
  Database,
  TrendingUp,
  MoreVertical,
  Archive,
  RefreshCw,
} from "lucide-react";
import {
  DropdownMenu,
  DropdownMenuContent,
  DropdownMenuItem,
  DropdownMenuTrigger,
} from "@/components/ui/dropdown-menu";

interface MLModelListProps {
  limit?: number;
  showActions?: boolean;
}

/**
 * MLモデル一覧コンポーネント
 *
 * 学習済みモデルの一覧表示、詳細確認、削除、バックアップなどの機能を提供
 */
export default function MLModelList({
  limit,
  showActions = true,
}: MLModelListProps) {
  const [selectedModel, setSelectedModel] = useState<string | null>(null);
  const { models, isLoading, error, fetchModels, deleteModel, deleteAllModels } =
    useMLModels(limit);

  useEffect(() => {
    fetchModels();
  }, [fetchModels]);

  const getModelStatusBadge = (model: MLModel) => {
    if (model.is_active) {
      return (
        <Badge
          variant="default"
          className="bg-green-800/20 text-green-400 border-green-700/40"
        >
          アクティブ
        </Badge>
      );
    }
    return <Badge variant="secondary">待機中</Badge>;
  };

  if (isLoading) {
    return <LoadingSpinner text="モデル一覧を読み込んでいます..." />;
  }

  if (error) {
    return <ErrorDisplay message={error} />;
  }

  if (models.length === 0) {
    return (
      <div className="text-center p-8 text-muted-foreground">
        <Database className="h-12 w-12 mx-auto mb-4 text-gray-600" />
        <p>学習済みモデルがありません</p>
        <p className="text-sm">新しいモデルをトレーニングしてください</p>
      </div>
    );
  }

  return (
    <div className="space-y-4">
      {/* 全削除ボタン */}
      {showActions && models.length > 0 && (
        <div className="flex justify-end">
          <ActionButton
            variant="danger"
            size="sm"
            onClick={deleteAllModels}
            loading={isLoading}
            icon={<Trash2 className="h-4 w-4" />}
          >
            すべてのモデルを削除
          </ActionButton>
        </div>
      )}

      {models.map((model) => (
        <Card
          key={model.id}
          className="bg-gray-900/50 border border-gray-800 transition-all duration-300 hover:border-primary/60 hover:shadow-[0_0_15px_2px_rgba(0,192,255,0.3)]"
        >
          <CardContent className="p-4">
            <div className="flex items-center justify-between">
              <div className="flex-1">
                <div className="flex items-center space-x-3 mb-2">
                  <h3 className="font-medium text-foreground">{model.name}</h3>
                  {getModelStatusBadge(model)}
                  {model.model_type && (
                    <Badge variant="outline">{model.model_type}</Badge>
                  )}
                </div>

                <div className="grid grid-cols-2 md:grid-cols-4 gap-4 text-sm text-muted-foreground">
                  <div className="flex items-center space-x-1">
                    <Calendar className="h-4 w-4" />
                    <span>{formatDateTime(model.modified_at).dateTime}</span>
                  </div>

                  <div className="flex items-center space-x-1">
                    <Database className="h-4 w-4" />
                    <span>{formatFileSize(model.size_mb)}</span>
                  </div>

                  {model.accuracy && model.accuracy > 0 && (
                    <div className="flex items-center space-x-1">
                      <TrendingUp className="h-4 w-4" />
                      <span>精度: {(model.accuracy * 100).toFixed(1)}%</span>
                    </div>
                  )}

                  {model.f1_score && model.f1_score > 0 && (
                    <div className="flex items-center space-x-1">
                      <TrendingUp className="h-4 w-4" />
                      <span>F1: {(model.f1_score * 100).toFixed(1)}%</span>
                    </div>
                  )}

                  {model.feature_count && model.feature_count > 0 && (
                    <div className="flex items-center space-x-1">
                      <span>特徴量: {model.feature_count}個</span>
                    </div>
                  )}

                  {model.training_samples && model.training_samples > 0 && (
                    <div className="flex items-center space-x-1">
                      <span>
                        学習: {model.training_samples.toLocaleString()}件
                      </span>
                    </div>
                  )}
                </div>

                <div className="mt-2 text-xs text-gray-500">
                  パス: {model.path}
                </div>
              </div>

              {showActions && (
                <div className="flex items-center space-x-2">
                  <ActionButton
                    variant="secondary"
                    size="sm"
                    onClick={() => setSelectedModel(model.id)}
                    icon={<Eye className="h-4 w-4" />}
                  >
                    詳細
                  </ActionButton>

                  <DropdownMenu>
                    <DropdownMenuTrigger>
                      <ActionButton
                        variant="secondary"
                        size="sm"
                        icon={<MoreVertical className="h-4 w-4" />}
                      >
                        操作
                      </ActionButton>
                    </DropdownMenuTrigger>
                    <DropdownMenuContent align="end">
                      <DropdownMenuItem
                        onClick={() => deleteModel(model.id)}
                        className="text-red-500 hover:text-red-400"
                      >
                        <Trash2 className="h-4 w-4 mr-2" />
                        削除
                      </DropdownMenuItem>
                    </DropdownMenuContent>
                  </DropdownMenu>
                </div>
              )}
            </div>
          </CardContent>
        </Card>
      ))}

      {!limit && models.length > 0 && (
        <div className="text-center pt-4">
          <ActionButton
            variant="secondary"
            onClick={fetchModels}
            icon={<RefreshCw className="h-4 w-4" />}
          >
            更新
          </ActionButton>
        </div>
      )}
    </div>
  );
}
