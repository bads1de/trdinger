"use client";

import React, { useState } from "react";
import {
  Dialog,
  DialogContent,
  DialogHeader,
  DialogTitle,
  DialogDescription,
} from "@/components/ui/dialog";
import ActionButton from "@/components/common/ActionButton";
import { Input } from "@/components/ui/input";
import { Label } from "@/components/ui/label";
import { OptimizationProfile } from "@/types/bayesian-optimization";
import { AlertTriangle, Trash2 } from "lucide-react";

interface ProfileDeleteDialogProps {
  isOpen: boolean;
  onClose: () => void;
  onConfirm: () => Promise<void>;
  profile: OptimizationProfile | null;
  isLoading?: boolean;
}

const ProfileDeleteDialog: React.FC<ProfileDeleteDialogProps> = ({
  isOpen,
  onClose,
  onConfirm,
  profile,
  isLoading = false,
}) => {
  const [confirmText, setConfirmText] = useState("");
  const [isDeleting, setIsDeleting] = useState(false);

  const expectedConfirmText = "削除";
  const isConfirmValid = confirmText === expectedConfirmText;

  const handleConfirm = async () => {
    if (!isConfirmValid || !profile) return;

    setIsDeleting(true);
    try {
      await onConfirm();
      handleClose();
    } catch (error) {
      console.error("削除エラー:", error);
    } finally {
      setIsDeleting(false);
    }
  };

  const handleClose = () => {
    setConfirmText("");
    onClose();
  };

  if (!profile) return null;

  return (
    <Dialog open={isOpen} onOpenChange={handleClose}>
      <DialogContent className="sm:max-w-[500px]">
        <DialogHeader>
          <DialogTitle className="flex items-center space-x-2 text-red-600 dark:text-red-400">
            <AlertTriangle className="h-5 w-5" />
            <span>プロファイル削除の確認</span>
          </DialogTitle>
          <DialogDescription className="text-base">
            この操作は取り消すことができません。削除されたプロファイルは復元できません。
          </DialogDescription>
        </DialogHeader>

        <div className="space-y-6">
          {/* プロファイル情報 */}
          <div className="bg-red-50 dark:bg-red-900/20 border border-red-200 dark:border-red-800 rounded-lg p-4">
            <h4 className="font-semibold text-red-800 dark:text-red-200 mb-2">
              削除対象プロファイル
            </h4>
            <div className="space-y-2 text-sm">
              <div>
                <span className="font-medium">名前:</span>{" "}
                <span className="text-red-700 dark:text-red-300">
                  {profile.profile_name}
                </span>
              </div>
              {profile.description && (
                <div>
                  <span className="font-medium">説明:</span>{" "}
                  <span className="text-gray-600 dark:text-gray-400">
                    {profile.description}
                  </span>
                </div>
              )}
              <div>
                <span className="font-medium">モデルタイプ:</span>{" "}
                <span className="text-gray-600 dark:text-gray-400">
                  {profile.model_type || "未指定"}
                </span>
              </div>
              {profile.optimization_result && (
                <div>
                  <span className="font-medium">ベストスコア:</span>{" "}
                  <span className="text-gray-600 dark:text-gray-400">
                    {profile.optimization_result.best_score.toFixed(4)}
                  </span>
                </div>
              )}
              <div>
                <span className="font-medium">作成日:</span>{" "}
                <span className="text-gray-600 dark:text-gray-400">
                  {new Date(profile.created_at).toLocaleString("ja-JP")}
                </span>
              </div>
            </div>
          </div>

          {/* 警告メッセージ */}
          <div className="bg-yellow-50 dark:bg-yellow-900/20 border border-yellow-200 dark:border-yellow-800 rounded-lg p-4">
            <div className="flex items-start space-x-2">
              <AlertTriangle className="h-5 w-5 text-yellow-600 dark:text-yellow-400 flex-shrink-0 mt-0.5" />
              <div className="text-sm text-yellow-800 dark:text-yellow-200">
                <p className="font-semibold mb-1">削除による影響:</p>
                <ul className="list-disc list-inside space-y-1">
                  <li>このプロファイルを使用している設定は無効になります</li>
                  <li>最適化履歴とパラメータ情報が失われます</li>
                  <li>関連するバックテスト結果への参照が無効になる可能性があります</li>
                </ul>
              </div>
            </div>
          </div>

          {/* 確認入力 */}
          <div className="space-y-2">
            <Label htmlFor="confirm-text" className="text-sm font-medium">
              削除を確認するため、「<span className="font-bold text-red-600 dark:text-red-400">{expectedConfirmText}</span>」と入力してください:
            </Label>
            <Input
              id="confirm-text"
              type="text"
              value={confirmText}
              onChange={(e) => setConfirmText(e.target.value)}
              placeholder={expectedConfirmText}
              className="border-red-300 dark:border-red-700 focus:border-red-500 dark:focus:border-red-500"
              disabled={isDeleting || isLoading}
            />
          </div>

          {/* アクションボタン */}
          <div className="flex justify-end space-x-3 pt-4">
            <ActionButton
              variant="secondary"
              onClick={handleClose}
              disabled={isDeleting || isLoading}
            >
              キャンセル
            </ActionButton>
            <ActionButton
              variant="danger"
              onClick={handleConfirm}
              disabled={!isConfirmValid || isDeleting || isLoading}
              loading={isDeleting || isLoading}
              loadingText="削除中..."
              icon={<Trash2 className="h-4 w-4" />}
            >
              削除する
            </ActionButton>
          </div>
        </div>
      </DialogContent>
    </Dialog>
  );
};

export default ProfileDeleteDialog;
