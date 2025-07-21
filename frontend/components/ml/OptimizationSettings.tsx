"use client";

import React from "react";
import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card";
import { Button } from "@/components/ui/button";
import { Switch } from "@/components/ui/switch";
import { Label } from "@/components/ui/label";
import { Zap, Info } from "lucide-react";

export interface OptimizationSettingsConfig {
  enabled: boolean;
  method: "optuna";
  n_calls: number;
  parameter_space: Record<string, any>;
}

interface OptimizationSettingsProps {
  settings: OptimizationSettingsConfig;
  onChange: (settings: OptimizationSettingsConfig) => void;
}

export default function OptimizationSettings({
  settings,
  onChange,
}: OptimizationSettingsProps) {
  return (
    <Card>
      <CardHeader>
        <CardTitle className="flex items-center gap-2">
          <Zap className="h-5 w-5" />
          ハイパーパラメータ最適化設定
        </CardTitle>
        <p className="text-sm text-muted-foreground">
          Optunaによる高効率な自動最適化
        </p>
      </CardHeader>
      <CardContent className="space-y-6">
        {/* 最適化有効/無効 */}
        <div className="flex items-center space-x-2">
          <Switch
            id="optimization-enabled"
            checked={settings.enabled}
            onCheckedChange={(enabled) => onChange({ ...settings, enabled })}
          />
          <Label htmlFor="optimization-enabled">
            ハイパーパラメータ自動最適化を有効にする
          </Label>
        </div>

        {settings.enabled && (
          <div className="space-y-4">
            {/* 試行回数 */}
            <div className="space-y-2">
              <Label>最適化試行回数</Label>
              <div className="grid grid-cols-3 gap-2">
                <Button
                  variant={settings.n_calls === 20 ? "default" : "outline"}
                  onClick={() => onChange({ ...settings, n_calls: 20 })}
                >
                  高速 (20回)
                </Button>
                <Button
                  variant={settings.n_calls === 50 ? "default" : "outline"}
                  onClick={() => onChange({ ...settings, n_calls: 50 })}
                >
                  標準 (50回)
                </Button>
                <Button
                  variant={settings.n_calls === 100 ? "default" : "outline"}
                  onClick={() => onChange({ ...settings, n_calls: 100 })}
                >
                  高精度 (100回)
                </Button>
              </div>
            </div>

            {/* 情報表示 */}
            <div className="bg-blue-50 dark:bg-blue-900/20 p-4 rounded-lg">
              <div className="flex items-center gap-2 mb-2">
                <Info className="h-4 w-4 text-blue-500" />
                <span className="text-sm font-medium text-blue-700 dark:text-blue-300">
                  Optuna最適化について
                </span>
              </div>
              <ul className="text-xs text-blue-600 dark:text-blue-400 space-y-1">
                <li>• TPEサンプラーによる効率的な探索</li>
                <li>• MedianPrunerによる早期停止</li>
                <li>• 予想時間: {Math.ceil(settings.n_calls * 0.2)}分</li>
              </ul>
            </div>
          </div>
        )}
      </CardContent>
    </Card>
  );
}
