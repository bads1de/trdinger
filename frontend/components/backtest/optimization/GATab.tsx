/**
 * GATab.tsx
 *
 * 遺伝的アルゴリズム設定タブのコンポーネント
 */
"use client";

import React from "react";
import GAConfigForm from "../GAConfigForm";
import { GAConfig } from "@/types/optimization";
import { BacktestConfig } from "@/types/backtest";

interface GATabProps {
  baseConfig: BacktestConfig;
  selectedStrategy: string;
  onRun: (config: GAConfig) => void;
  isLoading: boolean;
}

export default function GATab({
  baseConfig,
  selectedStrategy,
  onRun,
  isLoading,
}: GATabProps) {
  return (
    <GAConfigForm
      onSubmit={onRun}
      isLoading={isLoading}
      currentBacktestConfig={baseConfig}
    />
  );
}