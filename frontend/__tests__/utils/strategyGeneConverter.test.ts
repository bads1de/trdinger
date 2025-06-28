/**
 * StrategyGene変換ロジックのテスト
 *
 * TDDアプローチでStrategyGeneからBacktestConfigへの変換ロジックをテストします。
 */

import {
  convertToBacktestConfig,
  generateStrategyGene,
} from "@/utils/strategyGeneConverter";
import { SelectedIndicator, Condition } from "@/hooks/useStrategyBuilder";

describe("StrategyGene変換ロジック", () => {
  // テスト用のモックデータ
  const mockSelectedIndicators: SelectedIndicator[] = [
    {
      name: "SMA",
      type: "SMA",
      params: { period: 20 },
      parameters: { period: 20 },
      enabled: true,
    },
    {
      name: "RSI",
      type: "RSI",
      params: { period: 14 },
      parameters: { period: 14 },
      enabled: true,
    },
    {
      name: "MACD",
      type: "MACD",
      params: { fast_period: 12, slow_period: 26, signal_period: 9 },
      parameters: { fast_period: 12, slow_period: 26, signal_period: 9 },
      enabled: false, // 無効化された指標
    },
  ];

  const mockEntryConditions: Condition[] = [
    {
      type: "threshold",
      indicator1: "RSI",
      operator: "<",
      value: 30,
    },
    {
      type: "crossover",
      indicator1: "SMA",
      indicator2: "close",
      operator: ">",
    },
  ];

  const mockExitConditions: Condition[] = [
    {
      type: "threshold",
      indicator1: "RSI",
      operator: ">",
      value: 70,
    },
  ];

  const mockBacktestSettings = {
    symbol: "BTC/USDT",
    timeframe: "1h",
    start_date: "2024-01-01",
    end_date: "2024-12-31",
    initial_capital: 100000,
    commission_rate: 0.00055,
  };

  describe("generateStrategyGene", () => {
    test("有効な指標のみがStrategyGeneに含まれる", () => {
      const strategyGene = generateStrategyGene(
        mockSelectedIndicators,
        mockEntryConditions,
        mockExitConditions
      );

      // 有効な指標のみが含まれることを確認
      expect(strategyGene.indicators).toHaveLength(2);
      expect(strategyGene.indicators[0].type).toBe("SMA");
      expect(strategyGene.indicators[1].type).toBe("RSI");

      // 無効化された指標は含まれないことを確認
      const indicatorTypes = strategyGene.indicators.map((ind) => ind.type);
      expect(indicatorTypes).not.toContain("MACD");
    });

    test("指標が正しい形式で変換される", () => {
      const strategyGene = generateStrategyGene(
        mockSelectedIndicators,
        mockEntryConditions,
        mockExitConditions
      );

      const smaIndicator = strategyGene.indicators.find(
        (ind) => ind.type === "SMA"
      );
      expect(smaIndicator).toEqual({
        type: "SMA",
        parameters: { period: 20 },
        enabled: true,
        json_config: {
          indicator_name: "SMA",
          parameters: { period: 20 },
        },
      });
    });

    test("エントリー条件が正しく変換される", () => {
      const strategyGene = generateStrategyGene(
        mockSelectedIndicators,
        mockEntryConditions,
        mockExitConditions
      );

      expect(strategyGene.entry_conditions).toHaveLength(2);

      // threshold条件の変換
      const thresholdCondition = strategyGene.entry_conditions[0];
      expect(thresholdCondition).toEqual({
        type: "threshold",
        operator: "<",
        indicator: "RSI",
        value: 30,
      });

      // crossover条件の変換
      const crossoverCondition = strategyGene.entry_conditions[1];
      expect(crossoverCondition).toEqual({
        type: "crossover",
        operator: ">",
        indicator1: "SMA",
        indicator2: "close",
      });
    });

    test("イグジット条件が正しく変換される", () => {
      const strategyGene = generateStrategyGene(
        mockSelectedIndicators,
        mockEntryConditions,
        mockExitConditions
      );

      expect(strategyGene.exit_conditions).toHaveLength(1);
      expect(strategyGene.exit_conditions[0]).toEqual({
        type: "threshold",
        operator: ">",
        indicator: "RSI",
        value: 70,
      });
    });

    test("リスク管理設定がデフォルト値で設定される", () => {
      const strategyGene = generateStrategyGene(
        mockSelectedIndicators,
        mockEntryConditions,
        mockExitConditions
      );

      expect(strategyGene.risk_management).toEqual({
        stop_loss_pct: 0.02,
        take_profit_pct: 0.05,
        position_sizing: "fixed",
      });
    });

    test("メタデータが正しく設定される", () => {
      const strategyGene = generateStrategyGene(
        mockSelectedIndicators,
        mockEntryConditions,
        mockExitConditions
      );

      expect(strategyGene.metadata).toEqual({
        created_by: "strategy_builder",
        version: "1.0",
        created_at: expect.any(String),
      });

      // ISO日付形式であることを確認
      expect(new Date(strategyGene.metadata.created_at).toISOString()).toBe(
        strategyGene.metadata.created_at
      );
    });

    test("一意のIDが生成される", async () => {
      const strategyGene1 = generateStrategyGene(
        mockSelectedIndicators,
        mockEntryConditions,
        mockExitConditions
      );

      // 少し待ってから2つ目を生成
      await new Promise((resolve) => setTimeout(resolve, 1));

      const strategyGene2 = generateStrategyGene(
        mockSelectedIndicators,
        mockEntryConditions,
        mockExitConditions
      );

      expect(strategyGene1.id).not.toBe(strategyGene2.id);
      expect(strategyGene1.id).toMatch(/^user_strategy_\d+$/);
    });
  });

  describe("convertToBacktestConfig", () => {
    test("StrategyGeneとバックテスト設定が正しくBacktestConfigに変換される", () => {
      const strategyGene = generateStrategyGene(
        mockSelectedIndicators,
        mockEntryConditions,
        mockExitConditions
      );

      const backtestConfig = convertToBacktestConfig(
        strategyGene,
        mockBacktestSettings
      );

      expect(backtestConfig).toEqual({
        strategy_name: expect.stringMatching(/^CUSTOM_\d+$/),
        symbol: "BTC/USDT",
        timeframe: "1h",
        start_date: "2024-01-01",
        end_date: "2024-12-31",
        initial_capital: 100000,
        commission_rate: 0.00055,
        strategy_config: {
          strategy_type: "USER_CUSTOM",
          parameters: {
            strategy_gene: strategyGene,
          },
        },
      });
    });

    test("戦略名が一意に生成される", async () => {
      const strategyGene = generateStrategyGene(
        mockSelectedIndicators,
        mockEntryConditions,
        mockExitConditions
      );

      const config1 = convertToBacktestConfig(
        strategyGene,
        mockBacktestSettings
      );

      // 少し待ってから2つ目を生成
      await new Promise((resolve) => setTimeout(resolve, 1));

      const config2 = convertToBacktestConfig(
        strategyGene,
        mockBacktestSettings
      );

      expect(config1.strategy_name).not.toBe(config2.strategy_name);
    });

    test("異なるバックテスト設定が正しく反映される", () => {
      const strategyGene = generateStrategyGene(
        mockSelectedIndicators,
        mockEntryConditions,
        mockExitConditions
      );

      const customSettings = {
        symbol: "ETH/USDT",
        timeframe: "4h",
        start_date: "2023-01-01",
        end_date: "2023-12-31",
        initial_capital: 50000,
        commission_rate: 0.001,
      };

      const backtestConfig = convertToBacktestConfig(
        strategyGene,
        customSettings
      );

      expect(backtestConfig.symbol).toBe("ETH/USDT");
      expect(backtestConfig.timeframe).toBe("4h");
      expect(backtestConfig.start_date).toBe("2023-01-01");
      expect(backtestConfig.end_date).toBe("2023-12-31");
      expect(backtestConfig.initial_capital).toBe(50000);
      expect(backtestConfig.commission_rate).toBe(0.001);
    });
  });

  describe("エッジケース", () => {
    test("指標が空の場合でも正常に動作する", () => {
      const strategyGene = generateStrategyGene([], [], []);

      expect(strategyGene.indicators).toHaveLength(0);
      expect(strategyGene.entry_conditions).toHaveLength(0);
      expect(strategyGene.exit_conditions).toHaveLength(0);
    });

    test("すべての指標が無効化されている場合", () => {
      const disabledIndicators = mockSelectedIndicators.map((ind) => ({
        ...ind,
        enabled: false,
      }));

      const strategyGene = generateStrategyGene(
        disabledIndicators,
        mockEntryConditions,
        mockExitConditions
      );

      expect(strategyGene.indicators).toHaveLength(0);
    });

    test("comparison条件の変換", () => {
      const comparisonConditions: Condition[] = [
        {
          type: "comparison",
          indicator1: "SMA",
          indicator2: "EMA",
          operator: ">",
        },
      ];

      const strategyGene = generateStrategyGene(
        mockSelectedIndicators,
        comparisonConditions,
        []
      );

      expect(strategyGene.entry_conditions[0]).toEqual({
        type: "comparison",
        operator: ">",
        indicator1: "SMA",
        indicator2: "EMA",
      });
    });

    test("未知の条件タイプでも基本プロパティが設定される", () => {
      const unknownConditions: Condition[] = [
        {
          type: "unknown_type" as any,
          operator: "=",
          indicator1: "SMA",
        },
      ];

      const strategyGene = generateStrategyGene(
        mockSelectedIndicators,
        unknownConditions,
        []
      );

      expect(strategyGene.entry_conditions[0]).toEqual({
        type: "unknown_type",
        operator: "=",
      });
    });
  });
});
