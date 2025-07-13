"use client";

import React, { useState } from "react";
import { Dialog, DialogContent, DialogHeader, DialogTitle } from "@/components/ui/dialog";
import ActionButton from "@/components/common/ActionButton";
import { Input } from "@/components/ui/input";
import { Textarea } from "@/components/ui/textarea";
import { Label } from "@/components/ui/label";
import { Checkbox } from "@/components/ui/checkbox";
import { BayesianOptimizationResult } from "@/types/bayesian-optimization";

interface ProfileSaveDialogProps {
  isOpen: boolean;
  onClose: () => void;
  onSave: (profileData: {
    name: string;
    description?: string;
    isDefault?: boolean;
  }) => void;
  optimizationResult?: BayesianOptimizationResult;
}

const ProfileSaveDialog: React.FC<ProfileSaveDialogProps> = ({
  isOpen,
  onClose,
  onSave,
  optimizationResult,
}) => {
  const [profileName, setProfileName] = useState("");
  const [description, setDescription] = useState("");
  const [isDefault, setIsDefault] = useState(false);
  const [isLoading, setIsLoading] = useState(false);

  const handleSave = async () => {
    if (!profileName.trim()) {
      return;
    }

    setIsLoading(true);
    try {
      await onSave({
        name: profileName.trim(),
        description: description.trim() || undefined,
        isDefault,
      });
      
      // リセット
      setProfileName("");
      setDescription("");
      setIsDefault(false);
      onClose();
    } catch (error) {
      console.error("プロファイル保存エラー:", error);
    } finally {
      setIsLoading(false);
    }
  };

  const handleClose = () => {
    setProfileName("");
    setDescription("");
    setIsDefault(false);
    onClose();
  };

  return (
    <Dialog open={isOpen} onOpenChange={handleClose}>
      <DialogContent className="sm:max-w-[425px]">
        <DialogHeader>
          <DialogTitle>最適化結果をプロファイルとして保存</DialogTitle>
        </DialogHeader>
        
        <div className="space-y-4">
          {optimizationResult && (
            <div className="bg-gray-50 dark:bg-gray-800 p-3 rounded-lg">
              <div className="text-sm text-gray-600 dark:text-gray-400">
                最適化結果
              </div>
              <div className="font-medium">
                ベストスコア: {optimizationResult.best_score.toFixed(4)}
              </div>
              <div className="text-sm text-gray-600 dark:text-gray-400">
                モデルタイプ: {optimizationResult.model_type || "未指定"}
              </div>
            </div>
          )}
          
          <div className="space-y-2">
            <Label htmlFor="profile-name">プロファイル名 *</Label>
            <Input
              id="profile-name"
              value={profileName}
              onChange={(e) => setProfileName(e.target.value)}
              placeholder="プロファイル名を入力してください"
              disabled={isLoading}
            />
          </div>
          
          <div className="space-y-2">
            <Label htmlFor="profile-description">説明（オプション）</Label>
            <Textarea
              id="profile-description"
              value={description}
              onChange={(e) => setDescription(e.target.value)}
              placeholder="プロファイルの説明を入力してください"
              rows={3}
              disabled={isLoading}
            />
          </div>
          
          <div className="flex items-center space-x-2">
            <Checkbox
              id="is-default"
              checked={isDefault}
              onCheckedChange={(checked) => setIsDefault(checked as boolean)}
              disabled={isLoading}
            />
            <Label htmlFor="is-default" className="text-sm">
              デフォルトプロファイルとして設定
            </Label>
          </div>
          
          <div className="flex justify-end space-x-2 pt-4">
            <ActionButton
              variant="secondary"
              onClick={handleClose}
              disabled={isLoading}
            >
              キャンセル
            </ActionButton>
            <ActionButton
              variant="primary"
              onClick={handleSave}
              disabled={!profileName.trim() || isLoading}
              loading={isLoading}
              loadingText="保存中..."
            >
              保存
            </ActionButton>
          </div>
        </div>
      </DialogContent>
    </Dialog>
  );
};

export default ProfileSaveDialog;
