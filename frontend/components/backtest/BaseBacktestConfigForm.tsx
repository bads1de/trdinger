"use client";

import React from "react";
import { InputField } from "@/components/common/InputField";
import { SelectField } from "@/components/common/SelectField";
import { BacktestConfig } from "@/types/backtest";
import { TIME_FRAME_OPTIONS } from "@/constants/backtest";

interface BaseBacktestConfigFormProps {
  config: BacktestConfig;
  onConfigChange: (updates: Partial<BacktestConfig>) => void;
  isOptimization?: boolean;
}

export const BaseBacktestConfigForm: React.FC<BaseBacktestConfigFormProps> = ({
  config,
  onConfigChange,
  isOptimization = false,
}) => {
  const symbolOptions = [{ value: "BTC/USDT", label: "BTC/USDT" }];

  return (
    <div className="space-y-4">
      {/* 戦略名 */}
      {isOptimization && (
        <InputField
          label="戦略名"
          value={config.strategy_name}
          onChange={(value) => onConfigChange({ strategy_name: value })}
          required
        />
      )}

      {/* 取引ペアと時間軸 */}
      <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
        <SelectField
          label="取引ペア"
          value={config.symbol}
          onChange={(value) => onConfigChange({ symbol: value })}
          options={symbolOptions}
          required
        />
        <SelectField
          label="時間軸"
          value={config.timeframe}
          onChange={(value) => onConfigChange({ timeframe: value })}
          options={TIME_FRAME_OPTIONS}
          required
        />
      </div>

      {/* 期間設定 */}
      <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
        <InputField
          label="開始日"
          type="date"
          value={config.start_date}
          onChange={(value) => onConfigChange({ start_date: value })}
          required
        />
        <InputField
          label="終了日"
          type="date"
          value={config.end_date}
          onChange={(value) => onConfigChange({ end_date: value })}
          required
        />
      </div>

      {/* 資金設定 */}
      <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
        <InputField
          label="初期資金 (USDT)"
          type="number"
          value={config.initial_capital}
          onChange={(value) => onConfigChange({ initial_capital: value })}
          min={1000}
          step={1000}
          required
        />
        <InputField
          label="手数料率 (%)"
          type="number"
          value={config.commission_rate * 100}
          onChange={(value) => onConfigChange({ commission_rate: value / 100 })}
          min={0}
          max={100}
          step={0.001}
          required
        />
      </div>
    </div>
  );
};
