"use client";

import React, { useState, useEffect } from "react";
import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card";
import { Brain, Database, TrendingUp, Settings } from "lucide-react";
import TabButton from "@/components/common/TabButton";
import ErrorDisplay from "@/components/common/ErrorDisplay";
import LoadingSpinner from "@/components/common/LoadingSpinner";

import MLModelList from "@/components/ml/MLModelList";
import MLTraining from "@/components/ml/MLTraining";
import MLModelStatus from "@/components/ml/MLModelStatus";
import MLSettings from "@/components/ml/MLSettings";

/**
 * ML管理専用ページ
 *
 * MLモデルの管理、トレーニング、状態確認などを行う専用ページ
 */
export default function MLManagementPage() {
  const [activeTab, setActiveTab] = useState("overview");
  const [isLoading, setIsLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);

  useEffect(() => {
    // ページ初期化
    const initializePage = async () => {
      try {
        setIsLoading(true);
        // 必要な初期データの読み込み
        await new Promise((resolve) => setTimeout(resolve, 500));
        setIsLoading(false);
      } catch (err) {
        setError("ページの初期化に失敗しました");
        setIsLoading(false);
      }
    };

    initializePage();
  }, []);

  if (isLoading) {
    return (
      <div className="container mx-auto p-6">
        <LoadingSpinner
          text="ページの初期データを読み込んでいます..."
          size="lg"
        />
      </div>
    );
  }

  if (error) {
    return (
      <div className="container mx-auto p-6">
        <ErrorDisplay message={error} />
      </div>
    );
  }

  return (
    <div className="container mx-auto p-6 space-y-6">
      {/* ページヘッダー */}
      <div className="flex items-center space-x-3">
        <Brain className="h-8 w-8 text-primary" />
        <div>
          <h1 className="text-3xl font-bold text-foreground">ML管理</h1>
          <p className="text-muted-foreground">
            機械学習モデルの管理とトレーニング
          </p>
        </div>
      </div>

      {/* メインコンテンツ */}
      <div className="space-y-6">
        {/* タブナビゲーション */}
        <div className="flex space-x-2 border-b border-border pb-2">
          <TabButton
            label="概要"
            isActive={activeTab === "overview"}
            onClick={() => setActiveTab("overview")}
            icon={<TrendingUp className="h-4 w-4" />}
          />
          <TabButton
            label="モデル一覧"
            isActive={activeTab === "models"}
            onClick={() => setActiveTab("models")}
            icon={<Database className="h-4 w-4" />}
          />
          <TabButton
            label="トレーニング"
            isActive={activeTab === "training"}
            onClick={() => setActiveTab("training")}
            icon={<Brain className="h-4 w-4" />}
          />
          <TabButton
            label="設定"
            isActive={activeTab === "settings"}
            onClick={() => setActiveTab("settings")}
            icon={<Settings className="h-4 w-4" />}
          />
        </div>

        {/* 概要タブ */}
        {activeTab === "overview" && (
          <div className="space-y-6">
            <div className="grid grid-cols-1 gap-6">
              <Card>
                <CardHeader>
                  <CardTitle className="flex items-center space-x-2">
                    <Database className="h-5 w-5" />
                    <span>最近のモデル</span>
                  </CardTitle>
                </CardHeader>
                <CardContent>
                  <MLModelList limit={5} showActions={false} />
                </CardContent>
              </Card>
            </div>
          </div>
        )}

        {/* モデル一覧タブ */}
        {activeTab === "models" && (
          <div className="space-y-6">
            <Card>
              <CardHeader>
                <CardTitle>学習済みモデル一覧</CardTitle>
              </CardHeader>
              <CardContent>
                <MLModelList />
              </CardContent>
            </Card>
          </div>
        )}

        {/* トレーニングタブ */}
        {activeTab === "training" && (
          <div className="space-y-6">
            <Card>
              <CardHeader>
                <CardTitle>モデルトレーニング</CardTitle>
              </CardHeader>
              <CardContent>
                <MLTraining />
              </CardContent>
            </Card>
          </div>
        )}

        {/* 設定タブ */}
        {activeTab === "settings" && (
          <div className="space-y-6">
            <Card>
              <CardHeader>
                <CardTitle>ML設定</CardTitle>
              </CardHeader>
              <CardContent>
                <MLSettings />
              </CardContent>
            </Card>
          </div>
        )}
      </div>
    </div>
  );
}
