"use client";

import React, { useState } from "react";
import { Card } from "@/components/ui/card";
import { Badge } from "@/components/ui/badge";
import ActionButton from "@/components/common/ActionButton";
import ProfileDeleteDialog from "./ProfileDeleteDialog";
import { useOptimizationProfiles } from "@/hooks/useOptimizationProfiles";
import { OptimizationProfile } from "@/types/bayesian-optimization";
import {
  RefreshCw,
  Trash2,
  Eye,
  Settings,
  Calendar,
  Target,
  TrendingUp,
  AlertCircle,
} from "lucide-react";

interface ProfileManagementProps {
  modelType?: string;
  className?: string;
}

const ProfileManagement: React.FC<ProfileManagementProps> = ({
  modelType,
  className = "",
}) => {
  const {
    profiles,
    isLoading,
    error,
    fetchProfiles,
    deleteProfile,
    deleteLoading,
    deleteError,
  } = useOptimizationProfiles(modelType);

  const [selectedProfile, setSelectedProfile] = useState<OptimizationProfile | null>(null);
  const [showDeleteDialog, setShowDeleteDialog] = useState(false);
  const [expandedProfile, setExpandedProfile] = useState<number | null>(null);

  const handleDeleteClick = (profile: OptimizationProfile) => {
    setSelectedProfile(profile);
    setShowDeleteDialog(true);
  };

  const handleDeleteConfirm = async () => {
    if (!selectedProfile) return;

    const success = await deleteProfile(selectedProfile.id);
    if (success) {
      setShowDeleteDialog(false);
      setSelectedProfile(null);
    }
  };

  const handleDeleteCancel = () => {
    setShowDeleteDialog(false);
    setSelectedProfile(null);
  };

  const toggleExpanded = (profileId: number) => {
    setExpandedProfile(expandedProfile === profileId ? null : profileId);
  };

  const formatDate = (dateString: string) => {
    return new Date(dateString).toLocaleString("ja-JP", {
      year: "numeric",
      month: "short",
      day: "numeric",
      hour: "2-digit",
      minute: "2-digit",
    });
  };

  const formatScore = (score: number) => {
    return score.toFixed(4);
  };

  return (
    <div className={`space-y-6 ${className}`}>
      {/* ヘッダー */}
      <div className="flex items-center justify-between">
        <div>
          <h2 className="text-2xl font-bold">最適化プロファイル管理</h2>
          {modelType && (
            <p className="text-sm text-gray-600 dark:text-gray-400 mt-1">
              モデルタイプ: {modelType}
            </p>
          )}
        </div>
        <ActionButton
          variant="secondary"
          onClick={fetchProfiles}
          disabled={isLoading}
          loading={isLoading}
          loadingText="読み込み中..."
          icon={<RefreshCw className="h-4 w-4" />}
        >
          更新
        </ActionButton>
      </div>

      {/* エラー表示 */}
      {(error || deleteError) && (
        <div className="bg-red-50 dark:bg-red-900/20 border border-red-200 dark:border-red-800 rounded-lg p-4">
          <div className="flex items-center space-x-2">
            <AlertCircle className="h-5 w-5 text-red-600 dark:text-red-400" />
            <span className="text-red-800 dark:text-red-200">
              {error || deleteError}
            </span>
          </div>
        </div>
      )}

      {/* プロファイル統計 */}
      <div className="grid grid-cols-1 md:grid-cols-3 gap-4">
        <Card className="p-4">
          <div className="flex items-center space-x-2">
            <Settings className="h-5 w-5 text-blue-600 dark:text-blue-400" />
            <div>
              <p className="text-sm text-gray-600 dark:text-gray-400">
                総プロファイル数
              </p>
              <p className="text-2xl font-bold">{profiles.length}</p>
            </div>
          </div>
        </Card>
        <Card className="p-4">
          <div className="flex items-center space-x-2">
            <Target className="h-5 w-5 text-green-600 dark:text-green-400" />
            <div>
              <p className="text-sm text-gray-600 dark:text-gray-400">
                アクティブ
              </p>
              <p className="text-2xl font-bold">
                {profiles.filter((p) => p.is_active).length}
              </p>
            </div>
          </div>
        </Card>
        <Card className="p-4">
          <div className="flex items-center space-x-2">
            <TrendingUp className="h-5 w-5 text-purple-600 dark:text-purple-400" />
            <div>
              <p className="text-sm text-gray-600 dark:text-gray-400">
                デフォルト
              </p>
              <p className="text-2xl font-bold">
                {profiles.filter((p) => p.is_default).length}
              </p>
            </div>
          </div>
        </Card>
      </div>

      {/* プロファイル一覧 */}
      <div className="space-y-4">
        {profiles.length === 0 ? (
          <Card className="p-8 text-center">
            <Settings className="h-12 w-12 text-gray-400 mx-auto mb-4" />
            <h3 className="text-lg font-medium text-gray-600 dark:text-gray-400 mb-2">
              プロファイルがありません
            </h3>
            <p className="text-sm text-gray-500 dark:text-gray-500">
              ベイジアン最適化を実行してプロファイルを作成してください。
            </p>
          </Card>
        ) : (
          profiles.map((profile) => (
            <Card key={profile.id} className="p-6">
              <div className="flex items-start justify-between">
                <div className="flex-1">
                  <div className="flex items-center space-x-3 mb-2">
                    <h3 className="text-lg font-semibold">
                      {profile.profile_name}
                    </h3>
                    <div className="flex items-center space-x-2">
                      {profile.is_default && (
                        <Badge variant="secondary">デフォルト</Badge>
                      )}
                      {!profile.is_active && (
                        <Badge variant="destructive">非アクティブ</Badge>
                      )}
                    </div>
                  </div>

                  <div className="grid grid-cols-1 md:grid-cols-3 gap-4 text-sm text-gray-600 dark:text-gray-400">
                    <div className="flex items-center space-x-2">
                      <Calendar className="h-4 w-4" />
                      <span>{formatDate(profile.created_at)}</span>
                    </div>
                    {profile.model_type && (
                      <div className="flex items-center space-x-2">
                        <Target className="h-4 w-4" />
                        <span>{profile.model_type}</span>
                      </div>
                    )}
                    {profile.optimization_result && (
                      <div className="flex items-center space-x-2">
                        <TrendingUp className="h-4 w-4" />
                        <span>
                          スコア: {formatScore(profile.optimization_result.best_score)}
                        </span>
                      </div>
                    )}
                  </div>

                  {profile.description && (
                    <p className="text-sm text-gray-600 dark:text-gray-400 mt-2">
                      {profile.description}
                    </p>
                  )}
                </div>

                <div className="flex items-center space-x-2 ml-4">
                  <ActionButton
                    variant="secondary"
                    size="sm"
                    onClick={() => toggleExpanded(profile.id)}
                    icon={<Eye className="h-4 w-4" />}
                  >
                    {expandedProfile === profile.id ? "閉じる" : "詳細"}
                  </ActionButton>
                  <ActionButton
                    variant="danger"
                    size="sm"
                    onClick={() => handleDeleteClick(profile)}
                    disabled={deleteLoading}
                    icon={<Trash2 className="h-4 w-4" />}
                  >
                    削除
                  </ActionButton>
                </div>
              </div>

              {/* 詳細表示 */}
              {expandedProfile === profile.id && profile.optimization_result && (
                <div className="mt-6 pt-6 border-t border-gray-200 dark:border-gray-700">
                  <h4 className="text-md font-semibold mb-4">最適化結果詳細</h4>
                  <div className="grid grid-cols-1 md:grid-cols-2 gap-6">
                    <div>
                      <h5 className="text-sm font-medium mb-2">基本情報</h5>
                      <div className="space-y-2 text-sm">
                        <div className="flex justify-between">
                          <span>ベストスコア:</span>
                          <span className="font-mono text-green-600 dark:text-green-400">
                            {formatScore(profile.optimization_result.best_score)}
                          </span>
                        </div>
                        <div className="flex justify-between">
                          <span>評価回数:</span>
                          <span>{profile.optimization_result.total_evaluations}</span>
                        </div>
                        <div className="flex justify-between">
                          <span>最適化時間:</span>
                          <span>{profile.optimization_result.optimization_time.toFixed(2)}秒</span>
                        </div>
                      </div>
                    </div>
                    <div>
                      <h5 className="text-sm font-medium mb-2">最適パラメータ</h5>
                      <div className="space-y-1 text-sm max-h-32 overflow-y-auto">
                        {Object.entries(profile.optimization_result.best_params).map(
                          ([key, value]) => (
                            <div key={key} className="flex justify-between">
                              <span className="truncate">{key}:</span>
                              <span className="font-mono text-blue-600 dark:text-blue-400 ml-2">
                                {typeof value === "number" ? value.toFixed(4) : String(value)}
                              </span>
                            </div>
                          )
                        )}
                      </div>
                    </div>
                  </div>
                </div>
              )}
            </Card>
          ))
        )}
      </div>

      {/* 削除確認ダイアログ */}
      <ProfileDeleteDialog
        isOpen={showDeleteDialog}
        onClose={handleDeleteCancel}
        onConfirm={handleDeleteConfirm}
        profile={selectedProfile}
        isLoading={deleteLoading}
      />
    </div>
  );
};

export default ProfileManagement;
