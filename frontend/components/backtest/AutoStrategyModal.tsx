/**
 * オートストラテジーモーダルコンポーネント
 *
 * バックテストページからオートストラテジー機能を使用するためのモーダル
 */

"use client";

import React from "react";
import { X } from "lucide-react";
import GAConfigForm from "./GAConfigForm";
import { GAConfig } from "@/types/optimization";

interface AutoStrategyModalProps {
  isOpen: boolean;
  onClose: () => void;
  onSubmit: (config: GAConfig) => void;
  isLoading?: boolean;
  currentBacktestConfig?: any;
}

export default function AutoStrategyModal({
  isOpen,
  onClose,
  onSubmit,
  isLoading = false,
  currentBacktestConfig,
}: AutoStrategyModalProps) {
  if (!isOpen) return null;

  const handleSubmit = (config: GAConfig) => {
    onSubmit(config);
  };

  const handleBackdropClick = (e: React.MouseEvent) => {
    if (e.target === e.currentTarget) {
      onClose();
    }
  };

  return (
    <div
      className="fixed inset-0 bg-black bg-opacity-50 flex items-center justify-center z-50 p-4"
      onClick={handleBackdropClick}
    >
      <div className="bg-secondary-950 rounded-lg shadow-xl max-w-4xl w-full max-h-[90vh] overflow-hidden border border-secondary-700">
        {/* ヘッダー */}
        <div className="flex items-center justify-between p-6 border-b border-secondary-700">
          <div>
            <h2 className="text-2xl font-bold text-secondary-100">
              🚀 オートストラテジー生成
            </h2>
            <p className="text-sm text-secondary-400 mt-1">
              遺伝的アルゴリズムを使用して最適な取引戦略を自動生成します
            </p>
          </div>
          <button
            onClick={onClose}
            disabled={isLoading}
            className="p-2 hover:bg-secondary-700 rounded-lg transition-colors disabled:opacity-50"
          >
            <X className="w-6 h-6 text-secondary-400" />
          </button>
        </div>

        {/* コンテンツ */}
        <div className="p-6 overflow-y-auto max-h-[calc(90vh-120px)]">
          <GAConfigForm
            onSubmit={handleSubmit}
            isLoading={isLoading}
            currentBacktestConfig={currentBacktestConfig}
          />
        </div>

        {/* フッター */}
        <div className="flex items-center justify-between p-6 border-t border-secondary-700 bg-secondary-900">
          <div className="text-sm text-secondary-400">
            💡 ヒント: 初期設定のまま実行することをお勧めします
          </div>
          <button
            onClick={onClose}
            disabled={isLoading}
            className="px-4 py-2 text-secondary-400 hover:text-secondary-200 transition-colors disabled:opacity-50"
          >
            キャンセル
          </button>
        </div>
      </div>
    </div>
  );
}
