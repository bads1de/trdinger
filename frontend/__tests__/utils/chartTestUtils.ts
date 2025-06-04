/**
 * チャートテスト用ユーティリティ
 *
 * チャートコンポーネントのテストで使用するモックデータとヘルパー関数
 */

import {
  BacktestResult,
  EquityPoint,
  Trade,
  PerformanceMetrics,
  ChartEquityPoint,
  ChartTradePoint,
} from "@/types/backtest";

/**
 * モック用の資産曲線データを生成
 */
export const generateMockEquityCurve = (
  points: number = 100,
  startEquity: number = 100000,
  volatility: number = 0.02
): EquityPoint[] => {
  const data: EquityPoint[] = [];
  let currentEquity = startEquity;
  const startDate = new Date("2024-01-01");

  for (let i = 0; i < points; i++) {
    const date = new Date(startDate);
    date.setDate(date.getDate() + i);

    // ランダムウォークで資産額を変動
    const change = (Math.random() - 0.5) * 2 * volatility;
    currentEquity *= 1 + change;

    data.push({
      timestamp: date.toISOString(),
      equity: Math.round(currentEquity * 100) / 100,
    });
  }

  return data;
};

/**
 * モック用の取引履歴データを生成
 */
export const generateMockTradeHistory = (
  trades: number = 50,
  winRate: number = 0.6
): Trade[] => {
  const data: Trade[] = [];
  const startDate = new Date("2024-01-01");

  for (let i = 0; i < trades; i++) {
    const entryDate = new Date(startDate);
    entryDate.setDate(entryDate.getDate() + i * 2);

    const exitDate = new Date(entryDate);
    exitDate.setDate(exitDate.getDate() + 1);

    const isWin = Math.random() < winRate;
    const size = Math.random() > 0.5 ? 1 : -1; // ロングまたはショート
    const entryPrice = 50000 + Math.random() * 10000;
    const returnPct = isWin
      ? Math.random() * 0.05 + 0.01 // 1-6%の利益
      : -(Math.random() * 0.03 + 0.005); // 0.5-3.5%の損失

    const exitPrice = entryPrice * (1 + returnPct);
    const pnl = (exitPrice - entryPrice) * size;

    data.push({
      size,
      entry_price: Math.round(entryPrice * 100) / 100,
      exit_price: Math.round(exitPrice * 100) / 100,
      pnl: Math.round(pnl * 100) / 100,
      return_pct: returnPct,
      entry_time: entryDate.toISOString(),
      exit_time: exitDate.toISOString(),
    });
  }

  return data;
};

/**
 * モック用のパフォーマンス指標を生成
 */
export const generateMockPerformanceMetrics = (): PerformanceMetrics => {
  return {
    total_return: 0.25, // 25%
    sharpe_ratio: 1.8,
    max_drawdown: -0.15, // -15%
    win_rate: 0.65, // 65%
    profit_factor: 1.8,
    total_trades: 50,
    winning_trades: 33,
    losing_trades: 17,
    avg_win: 1250.5,
    avg_loss: -680.25,
    equity_final: 125000,
    buy_hold_return: 0.2, // 20%
    exposure_time: 0.85, // 85%
    sortino_ratio: 2.1,
    calmar_ratio: 1.67,
  };
};

/**
 * 完全なモックバックテスト結果を生成
 */
export const generateMockBacktestResult = (
  equityPoints: number = 100,
  tradeCount: number = 50
): BacktestResult => {
  const equityCurve = generateMockEquityCurve(equityPoints);
  const tradeHistory = generateMockTradeHistory(tradeCount);
  const performanceMetrics = generateMockPerformanceMetrics();

  return {
    id: "test-result-1",
    strategy_name: "Test Strategy",
    symbol: "BTC/USDT",
    timeframe: "1h",
    start_date: "2024-01-01",
    end_date: "2024-12-31",
    initial_capital: 100000,
    commission_rate: 0.00055,
    performance_metrics: performanceMetrics,
    equity_curve: equityCurve,
    trade_history: tradeHistory,
    created_at: new Date().toISOString(),
  };
};

/**
 * チャート表示用のモックデータを生成
 */
export const generateMockChartEquityData = (
  points: number = 100
): ChartEquityPoint[] => {
  const equityCurve = generateMockEquityCurve(points);
  return equityCurve.map((point, index) => ({
    date: new Date(point.timestamp).getTime(),
    equity: point.equity,
    drawdown: Math.random() * 10, // 0-10%のドローダウン
    formattedDate: new Date(point.timestamp).toLocaleDateString("ja-JP"),
  }));
};

/**
 * チャート表示用のモック取引データを生成
 */
export const generateMockChartTradeData = (
  trades: number = 50
): ChartTradePoint[] => {
  const tradeHistory = generateMockTradeHistory(trades);
  return tradeHistory.map((trade) => ({
    entryDate: new Date(trade.entry_time).getTime(),
    exitDate: new Date(trade.exit_time).getTime(),
    pnl: trade.pnl,
    returnPct: trade.return_pct * 100,
    size: Math.abs(trade.size),
    type: trade.size > 0 ? "long" : "short",
    isWin: trade.pnl > 0,
  }));
};

/**
 * 空のデータセットを生成（エラーケーステスト用）
 */
export const generateEmptyBacktestResult = (): BacktestResult => {
  return {
    strategy_name: "Empty Strategy",
    symbol: "BTC/USDT",
    timeframe: "1h",
    start_date: "2024-01-01",
    end_date: "2024-12-31",
    initial_capital: 100000,
    commission_rate: 0.00055,
    performance_metrics: {
      total_return: null,
      sharpe_ratio: null,
      max_drawdown: null,
      win_rate: null,
      profit_factor: null,
      total_trades: null,
      winning_trades: null,
      losing_trades: null,
      avg_win: null,
      avg_loss: null,
    },
    equity_curve: [],
    trade_history: [],
  };
};

/**
 * エラー状態のモックデータ
 */
export const generateErrorBacktestResult = (): BacktestResult => {
  const result = generateMockBacktestResult();
  // 意図的に不正なデータを含める
  result.equity_curve = undefined;
  result.trade_history = undefined;
  return result;
};

/**
 * テスト用のアサーション関数
 */
export const chartTestAssertions = {
  /**
   * チャートコンテナが正しくレンダリングされているかチェック
   */
  expectChartContainerToRender: (container: HTMLElement) => {
    expect(container.querySelector(".bg-gray-800\\/30")).toBeInTheDocument();
  },

  /**
   * ローディング状態が正しく表示されているかチェック
   */
  expectLoadingState: (container: HTMLElement) => {
    expect(container.querySelector(".animate-pulse")).toBeInTheDocument();
    expect(container.textContent).toContain("チャートを読み込み中");
  },

  /**
   * エラー状態が正しく表示されているかチェック
   */
  expectErrorState: (container: HTMLElement, errorMessage?: string) => {
    expect(container.textContent).toContain("エラーが発生しました");
    if (errorMessage) {
      expect(container.textContent).toContain(errorMessage);
    }
  },

  /**
   * 空データ状態が正しく表示されているかチェック
   */
  expectEmptyState: (container: HTMLElement) => {
    expect(container.textContent).toContain("データがありません");
  },

  /**
   * チャートタイトルが正しく表示されているかチェック
   */
  expectChartTitle: (container: HTMLElement, title: string) => {
    expect(container.textContent).toContain(title);
  },
};

/**
 * Recharts コンポーネントのモック
 */
export const mockRechartsComponents = () => {
  // ResponsiveContainer のモック
  jest.mock("recharts", () => ({
    ResponsiveContainer: ({ children }: { children: any }) => ({
      type: "div",
      props: { "data-testid": "responsive-container", children },
    }),
    LineChart: ({ children }: { children: any }) => ({
      type: "div",
      props: { "data-testid": "line-chart", children },
    }),
    AreaChart: ({ children }: { children: any }) => ({
      type: "div",
      props: { "data-testid": "area-chart", children },
    }),
    ScatterChart: ({ children }: { children: any }) => ({
      type: "div",
      props: { "data-testid": "scatter-chart", children },
    }),
    BarChart: ({ children }: { children: any }) => ({
      type: "div",
      props: { "data-testid": "bar-chart", children },
    }),
    Line: () => ({ type: "div", props: { "data-testid": "line" } }),
    Area: () => ({ type: "div", props: { "data-testid": "area" } }),
    Scatter: () => ({ type: "div", props: { "data-testid": "scatter" } }),
    Bar: () => ({ type: "div", props: { "data-testid": "bar" } }),
    XAxis: () => ({ type: "div", props: { "data-testid": "x-axis" } }),
    YAxis: () => ({ type: "div", props: { "data-testid": "y-axis" } }),
    CartesianGrid: () => ({
      type: "div",
      props: { "data-testid": "cartesian-grid" },
    }),
    Tooltip: () => ({ type: "div", props: { "data-testid": "tooltip" } }),
    Legend: () => ({ type: "div", props: { "data-testid": "legend" } }),
    ReferenceLine: () => ({
      type: "div",
      props: { "data-testid": "reference-line" },
    }),
  }));
};
