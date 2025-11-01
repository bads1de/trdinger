import React from "react";
import { Metadata } from "next";
import MLOverviewDashboard from "@/components/ml/MLOverviewDashboard";

export const metadata: Metadata = {
  title: "ML Management | Trdinger",
  description: "機械学習モデルの管理とトレーニング",
};

export default function MLPage() {
  return (
    <div className="flex flex-col gap-6 p-6 bg-gradient-to-br from-background to-sidebar-accent/10 min-h-screen">
      {/* ページヘッダー */}
      <div className="flex items-center justify-between">
        <div className="flex items-center gap-3">
          <div className="p-2 bg-gradient-to-br from-cyan-500 to-blue-600 rounded-lg shadow-lg">
            <svg
              xmlns="http://www.w3.org/2000/svg"
              width="24"
              height="24"
              viewBox="0 0 24 24"
              fill="none"
              stroke="currentColor"
              strokeWidth="2"
              strokeLinecap="round"
              strokeLinejoin="round"
              className="text-white"
            >
              <path d="M12 2v20M2 12h20" />
            </svg>
          </div>
          <div>
            <h1 className="text-3xl font-bold bg-gradient-to-r from-cyan-600 to-blue-600 bg-clip-text text-transparent">
              ML Management
            </h1>
            <p className="text-sm text-muted-foreground">
              機械学習モデルの管理、トレーニング、監視
            </p>
          </div>
        </div>
      </div>

      {/* メインコンテンツ */}
      <MLOverviewDashboard />
    </div>
  );
}
