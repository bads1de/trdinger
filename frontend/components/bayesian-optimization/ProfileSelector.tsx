"use client";

import React, { useState } from "react";
import {
  Select,
  SelectContent,
  SelectItem,
  SelectTrigger,
  SelectValue,
} from "@/components/ui/select";
import ActionButton from "@/components/common/ActionButton";
import { Badge } from "@/components/ui/badge";
import { Card } from "@/components/ui/card";
import { useOptimizationProfiles } from "@/hooks/useOptimizationProfiles";
import { OptimizationProfile } from "@/types/bayesian-optimization";
import ProfileDeleteDialog from "./ProfileDeleteDialog";
import { Loader2, RefreshCw, Settings, Trash2 } from "lucide-react";

interface ProfileSelectorProps {
  selectedProfileId?: number;
  onProfileSelect: (profile: OptimizationProfile | null) => void;
  modelType?: string;
  className?: string;
  showManagement?: boolean;
  onManagementClick?: () => void;
}

const ProfileSelector: React.FC<ProfileSelectorProps> = ({
  selectedProfileId,
  onProfileSelect,
  modelType,
  className = "",
  showManagement = false,
  onManagementClick,
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

  const [selectedProfileForDelete, setSelectedProfileForDelete] =
    useState<OptimizationProfile | null>(null);
  const [showDeleteDialog, setShowDeleteDialog] = useState(false);

  const handleProfileChange = (profileId: string) => {
    if (profileId === "none") {
      onProfileSelect(null);
      return;
    }

    const profile = profiles.find((p) => p.id.toString() === profileId);
    if (profile) {
      onProfileSelect(profile);
    }
  };

  const selectedProfile = profiles.find((p) => p.id === selectedProfileId);

  const handleDeleteClick = (
    profile: OptimizationProfile,
    event: React.MouseEvent
  ) => {
    event.stopPropagation();
    setSelectedProfileForDelete(profile);
    setShowDeleteDialog(true);
  };

  const handleDeleteConfirm = async () => {
    if (!selectedProfileForDelete) return;

    const success = await deleteProfile(selectedProfileForDelete.id);
    if (success) {
      // 削除されたプロファイルが選択されていた場合、選択を解除
      if (selectedProfileId === selectedProfileForDelete.id) {
        onProfileSelect(null);
      }
      setShowDeleteDialog(false);
      setSelectedProfileForDelete(null);
    }
  };

  const handleDeleteCancel = () => {
    setShowDeleteDialog(false);
    setSelectedProfileForDelete(null);
  };

  return (
    <div className={`space-y-3 ${className}`}>
      <div className="flex items-center justify-between">
        <label className="text-sm font-medium">
          最適化プロファイル
          {modelType && (
            <span className="text-xs text-gray-500 ml-1">({modelType})</span>
          )}
        </label>
        <div className="flex items-center space-x-2">
          {showManagement && onManagementClick && (
            <ActionButton
              variant="secondary"
              size="sm"
              onClick={onManagementClick}
              icon={<Settings className="h-4 w-4" />}
            >
              管理
            </ActionButton>
          )}
          <ActionButton
            variant="secondary"
            size="sm"
            onClick={fetchProfiles}
            disabled={isLoading}
          >
            <RefreshCw
              className={`h-4 w-4 ${isLoading ? "animate-spin" : ""}`}
            />
          </ActionButton>
        </div>
      </div>

      {(error || deleteError) && (
        <div className="text-sm text-red-600 dark:text-red-400">
          {error || deleteError}
        </div>
      )}

      <Select
        value={selectedProfileId?.toString() || "none"}
        onValueChange={handleProfileChange}
        disabled={isLoading}
      >
        <SelectTrigger>
          <SelectValue placeholder="プロファイルを選択してください">
            {isLoading ? (
              <div className="flex items-center">
                <Loader2 className="h-4 w-4 animate-spin mr-2" />
                読み込み中...
              </div>
            ) : selectedProfile ? (
              <div className="flex items-center space-x-2 truncate">
                <span className="truncate">{selectedProfile.profile_name}</span>
                {selectedProfile.is_default && (
                  <Badge variant="secondary" className="text-xs flex-shrink-0">
                    デフォルト
                  </Badge>
                )}
              </div>
            ) : (
              "プロファイルを選択してください"
            )}
          </SelectValue>
        </SelectTrigger>
        <SelectContent className="min-w-[350px]">
          <SelectItem value="none">
            <span className="truncate">プロファイルを使用しない</span>
          </SelectItem>
          {profiles.map((profile) => (
            <SelectItem key={profile.id} value={profile.id.toString()}>
              <div className="flex items-center justify-between w-full">
                <span className="truncate">{profile.profile_name}</span>
                <div className="flex items-center space-x-1 ml-2 flex-shrink-0">
                  {profile.is_default && (
                    <Badge variant="secondary" className="text-xs">
                      デフォルト
                    </Badge>
                  )}
                  {profile.optimization_result && (
                    <span className="text-xs text-gray-500">
                      スコア:{" "}
                      {profile.optimization_result.best_score.toFixed(3)}
                    </span>
                  )}
                  {showManagement && (
                    <ActionButton
                      variant="danger"
                      size="sm"
                      onClick={(e) => handleDeleteClick(profile, e)}
                      disabled={deleteLoading}
                      className="h-6 w-6 p-0"
                      icon={<Trash2 className="h-3 w-3" />}
                    />
                  )}
                </div>
              </div>
            </SelectItem>
          ))}
        </SelectContent>
      </Select>

      {selectedProfile && (
        <Card className="p-3 bg-blue-50 dark:bg-blue-900/20 border-blue-200 dark:border-blue-800">
          <div className="space-y-2">
            <div className="flex items-center justify-between">
              <span className="font-medium text-sm">
                {selectedProfile.profile_name}
              </span>
              {selectedProfile.is_default && (
                <Badge variant="secondary" className="text-xs">
                  デフォルト
                </Badge>
              )}
            </div>

            {selectedProfile.description && (
              <p className="text-xs text-gray-600 dark:text-gray-400">
                {selectedProfile.description}
              </p>
            )}

            {selectedProfile.optimization_result && (
              <div className="grid grid-cols-2 gap-2 text-xs">
                <div>
                  <span className="text-gray-500">ベストスコア:</span>
                  <span className="ml-1 font-mono">
                    {selectedProfile.optimization_result.best_score.toFixed(4)}
                  </span>
                </div>
                <div>
                  <span className="text-gray-500">評価回数:</span>
                  <span className="ml-1 font-mono">
                    {selectedProfile.optimization_result.total_evaluations}
                  </span>
                </div>
              </div>
            )}
          </div>
        </Card>
      )}

      {/* 削除確認ダイアログ */}
      <ProfileDeleteDialog
        isOpen={showDeleteDialog}
        onClose={handleDeleteCancel}
        onConfirm={handleDeleteConfirm}
        profile={selectedProfileForDelete}
        isLoading={deleteLoading}
      />
    </div>
  );
};

export default ProfileSelector;
