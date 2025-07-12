"use client";

import React, { useState, useEffect } from "react";
import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card";
import { Alert, AlertDescription } from "@/components/ui/alert";
import { Brain, Database, TrendingUp, Settings } from "lucide-react";
import TabButton from "@/components/common/TabButton";

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
        await new Promise((resolve) => setTimeout(resolve, 500)); // 仮の読み込み時間
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
        <div className="flex items-center justify-center h-64">
          <div className="animate-spin rounded-full h-8 w-8 border-b-2 border-primary"></div>
          <span className="ml-2 text-muted-foreground">読み込み中...</span>
        </div>
      </div>
    );
  }

  if (error) {
    return (
      <div className="container mx-auto p-6">
        <Alert variant="destructive">
          <AlertDescription>{error}</AlertDescription>
        </Alert>
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
            <div className="grid grid-cols-1 lg:grid-cols-2 gap-6">
              <Card>
                <CardHeader>
                  <CardTitle className="flex items-center space-x-2">
                    <TrendingUp className="h-5 w-5" />
                    <span>モデル状態</span>
                  </CardTitle>
                </CardHeader>
                <CardContent>
                  <MLModelStatus />
                </CardContent>
              </Card>

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

            {/* クイックアクション */}
            <Card>
              <CardHeader>
                <CardTitle>クイックアクション</CardTitle>
              </CardHeader>
              <CardContent>
                <div className="grid grid-cols-1 md:grid-cols-3 gap-4">
                  <div
                    onClick={() => setActiveTab("training")}
                    className="p-4 border border-border rounded-lg hover:bg-muted/50 transition-all text-left cursor-pointer hover:shadow-[0_0_15px_#3b82f6] hover:border-blue-400"
                  >
                    <Brain className="h-6 w-6 text-blue-500 mb-2" />
                    <h3 className="font-medium text-foreground">
                      新しいモデルを学習
                    </h3>
                    <p className="text-sm text-muted-foreground">
                      最新データでモデルをトレーニング
                    </p>
                  </div>

                  <div
                    onClick={() => setActiveTab("models")}
                    className="p-4 border border-border rounded-lg hover:bg-muted/50 transition-all text-left cursor-pointer hover:shadow-[0_0_15px_#22c55e] hover:border-green-400"
                  >
                    <Database className="h-6 w-6 text-green-500 mb-2" />
                    <h3 className="font-medium text-foreground">モデル管理</h3>
                    <p className="text-sm text-muted-foreground">
                      既存モデルの確認・管理
                    </p>
                  </div>

                  <div
                    onClick={() => setActiveTab("settings")}
                    className="p-4 border border-border rounded-lg hover:bg-muted/50 transition-all text-left cursor-pointer hover:shadow-[0_0_15px_#a855f7] hover:border-purple-400"
                  >
                    <Settings className="h-6 w-6 text-purple-500 mb-2" />
                    <h3 className="font-medium text-foreground">設定</h3>
                    <p className="text-sm text-muted-foreground">
                      ML関連の設定を変更
                    </p>
                  </div>
                </div>
              </CardContent>
            </Card>
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
