"use client";

import React from "react";
import {
  AlertDialog,
  AlertDialogAction,
  AlertDialogCancel,
  AlertDialogContent,
  AlertDialogDescription,
  AlertDialogFooter,
  AlertDialogHeader,
  AlertDialogTitle,
} from "@/components/ui/alert-dialog";
import { Button } from "@/components/ui/button";
import { AlertTriangle, Square, Zap } from "lucide-react";

interface ConfirmDialogProps {
  open: boolean;
  onOpenChange: (open: boolean) => void;
  title: string;
  description: string;
  confirmText?: string;
  cancelText?: string;
  variant?: "default" | "destructive" | "warning";
  onConfirm: () => void;
  onCancel?: () => void;
  showForceOption?: boolean;
  onForceConfirm?: () => void;
  forceText?: string;
}

/**
 * 確認ダイアログコンポーネント
 * 
 * 重要な操作の前に確認を求めるダイアログ
 */
export default function ConfirmDialog({
  open,
  onOpenChange,
  title,
  description,
  confirmText = "確認",
  cancelText = "キャンセル",
  variant = "default",
  onConfirm,
  onCancel,
  showForceOption = false,
  onForceConfirm,
  forceText = "強制実行",
}: ConfirmDialogProps) {
  const handleConfirm = () => {
    onConfirm();
    onOpenChange(false);
  };

  const handleForceConfirm = () => {
    if (onForceConfirm) {
      onForceConfirm();
    }
    onOpenChange(false);
  };

  const handleCancel = () => {
    if (onCancel) {
      onCancel();
    }
    onOpenChange(false);
  };

  const getIcon = () => {
    switch (variant) {
      case "destructive":
        return <AlertTriangle className="h-6 w-6 text-red-600" />;
      case "warning":
        return <AlertTriangle className="h-6 w-6 text-yellow-600" />;
      default:
        return <Square className="h-6 w-6 text-blue-600" />;
    }
  };

  const getConfirmButtonVariant = () => {
    switch (variant) {
      case "destructive":
        return "destructive";
      case "warning":
        return "default";
      default:
        return "default";
    }
  };

  return (
    <AlertDialog open={open} onOpenChange={onOpenChange}>
      <AlertDialogContent className="sm:max-w-[425px]">
        <AlertDialogHeader>
          <AlertDialogTitle className="flex items-center space-x-2">
            {getIcon()}
            <span>{title}</span>
          </AlertDialogTitle>
          <AlertDialogDescription className="text-left">
            {description}
          </AlertDialogDescription>
        </AlertDialogHeader>
        <AlertDialogFooter className="flex-col sm:flex-row gap-2">
          <AlertDialogCancel onClick={handleCancel}>
            {cancelText}
          </AlertDialogCancel>
          
          <div className="flex gap-2">
            <AlertDialogAction
              onClick={handleConfirm}
              className={
                variant === "destructive"
                  ? "bg-red-600 hover:bg-red-700"
                  : variant === "warning"
                  ? "bg-yellow-600 hover:bg-yellow-700"
                  : ""
              }
            >
              {confirmText}
            </AlertDialogAction>
            
            {showForceOption && onForceConfirm && (
              <Button
                onClick={handleForceConfirm}
                variant="destructive"
                size="sm"
                className="bg-red-700 hover:bg-red-800"
              >
                <Zap className="h-4 w-4 mr-1" />
                {forceText}
              </Button>
            )}
          </div>
        </AlertDialogFooter>
      </AlertDialogContent>
    </AlertDialog>
  );
}

/**
 * MLトレーニング停止確認ダイアログ
 */
interface StopTrainingDialogProps {
  open: boolean;
  onOpenChange: (open: boolean) => void;
  onConfirm: () => void;
  onForceConfirm: () => void;
  isTraining: boolean;
  processId?: string;
}

export function StopTrainingDialog({
  open,
  onOpenChange,
  onConfirm,
  onForceConfirm,
  isTraining,
  processId,
}: StopTrainingDialogProps) {
  return (
    <ConfirmDialog
      open={open}
      onOpenChange={onOpenChange}
      title="トレーニング停止の確認"
      description={
        isTraining
          ? `現在実行中のMLトレーニングを停止しますか？${
              processId ? `\n\nプロセスID: ${processId}` : ""
            }\n\n通常停止では現在の処理が完了してから停止します。強制停止では即座に停止しますが、データが失われる可能性があります。`
          : "実行中のトレーニングがありません。"
      }
      confirmText="通常停止"
      cancelText="キャンセル"
      variant="warning"
      onConfirm={onConfirm}
      showForceOption={isTraining}
      onForceConfirm={onForceConfirm}
      forceText="強制停止"
    />
  );
}
