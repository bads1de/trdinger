"use client";

import React, { useState, useEffect } from "react";
import { Card, CardContent } from "@/components/ui/card";
import ActionButton from "@/components/common/ActionButton";
import { Badge } from "@/components/ui/badge";
import { Alert, AlertDescription } from "@/components/ui/alert";
import {
  Download,
  Trash2,
  Eye,
  Calendar,
  Database,
  TrendingUp,
  MoreVertical,
  Archive,
} from "lucide-react";
import {
  DropdownMenu,
  DropdownMenuContent,
  DropdownMenuItem,
  DropdownMenuTrigger,
} from "@/components/ui/dropdown-menu";

interface MLModel {
  id: string;
  name: string;
  path: string;
  size_mb: number;
  modified_at: string;
  directory: string;
  accuracy?: number;
  feature_count?: number;
  model_type?: string;
  is_active?: boolean;
}

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
  const [models, setModels] = useState<MLModel[]>([]);
  const [isLoading, setIsLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);
  const [selectedModel, setSelectedModel] = useState<string | null>(null);

  useEffect(() => {
    fetchModels();
  }, []);

  const fetchModels = async () => {
    try {
      setIsLoading(true);
      const response = await fetch("/api/ml/models");
      if (!response.ok) {
        throw new Error("モデル一覧の取得に失敗しました");
      }
      const data = await response.json();
      let modelList = data.models || [];

      if (limit) {
        modelList = modelList.slice(0, limit);
      }

      setModels(modelList);
      setError(null);
    } catch (err) {
      setError(err instanceof Error ? err.message : "エラーが発生しました");
    } finally {
      setIsLoading(false);
    }
  };

  const handleDeleteModel = async (modelId: string) => {
    if (!confirm("このモデルを削除しますか？この操作は取り消せません。")) {
      return;
    }

    try {
      const response = await fetch(`/api/ml/models/${modelId}`, {
        method: "DELETE",
      });

      if (!response.ok) {
        throw new Error("モデルの削除に失敗しました");
      }

      await fetchModels(); // リストを更新
    } catch (err) {
      setError(err instanceof Error ? err.message : "削除エラーが発生しました");
    }
  };

  const handleBackupModel = async (modelId: string) => {
    try {
      const response = await fetch(`/api/ml/models/${modelId}/backup`, {
        method: "POST",
      });

      if (!response.ok) {
        throw new Error("モデルのバックアップに失敗しました");
      }

      alert("モデルのバックアップが完了しました");
    } catch (err) {
      setError(
        err instanceof Error ? err.message : "バックアップエラーが発生しました"
      );
    }
  };

  const formatFileSize = (sizeInMB: number): string => {
    if (sizeInMB < 1) {
      return `${(sizeInMB * 1024).toFixed(1)} KB`;
    }
    return `${sizeInMB.toFixed(1)} MB`;
  };

  const formatDate = (dateString: string): string => {
    return new Date(dateString).toLocaleString("ja-JP");
  };

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
    return (
      <div className="flex items-center justify-center p-8">
        <div className="animate-spin rounded-full h-6 w-6 border-b-2 border-primary"></div>
        <span className="ml-2 text-muted-foreground">読み込み中...</span>
      </div>
    );
  }

  if (error) {
    return (
      <Alert variant="destructive">
        <AlertDescription>{error}</AlertDescription>
      </Alert>
    );
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
                    <span>{formatDate(model.modified_at)}</span>
                  </div>

                  <div className="flex items-center space-x-1">
                    <Database className="h-4 w-4" />
                    <span>{formatFileSize(model.size_mb)}</span>
                  </div>

                  {model.accuracy && (
                    <div className="flex items-center space-x-1">
                      <TrendingUp className="h-4 w-4" />
                      <span>精度: {(model.accuracy * 100).toFixed(1)}%</span>
                    </div>
                  )}

                  {model.feature_count && (
                    <div className="flex items-center space-x-1">
                      <span>特徴量: {model.feature_count}個</span>
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
                        onClick={() => handleBackupModel(model.id)}
                      >
                        <Archive className="h-4 w-4 mr-2" />
                        バックアップ
                      </DropdownMenuItem>
                      <DropdownMenuItem
                        onClick={() => handleDeleteModel(model.id)}
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
            icon={<Download className="h-4 w-4" />}
          >
            更新
          </ActionButton>
        </div>
      )}
    </div>
  );
}
