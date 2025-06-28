/**
 * 既存コンポーネント統合テスト
 *
 * TDDアプローチでPerformanceMetricsコンポーネント等の既存コンポーネント統合をテストします。
 */

import { render, screen, fireEvent, waitFor } from "@testing-library/react";
import "@testing-library/jest-dom";
import StrategyBacktestResults from "@/components/strategy-builder/StrategyBacktestResults";
import { BacktestResult } from "@/types/backtest";

// TradeHistoryTableとChartModalのモック
jest.mock("@/components/backtest/TradeHistoryTable", () => {
  return function MockTradeHistoryTable({ tradeHistory }: any) {
    return (
      <div data-testid="trade-history-table">
        <div>取引履歴テーブル</div>
        {tradeHistory.map((trade: any, index: number) => (
          <div key={index} data-testid={`trade-${index}`}>
            エントリー: {trade.entry_price}, イグジット: {trade.exit_price},
            損益: {trade.pnl}
          </div>
        ))}
      </div>
    );
  };
});

jest.mock("@/components/backtest/charts/ChartModal", () => {
  return function MockChartModal({ isOpen, onClose, result }: any) {
    if (!isOpen) return null;
    return (
      <div data-testid="chart-modal">
        <div>バックテスト結果チャート</div>
        <div>戦略: {result.strategy_name}</div>
        <button onClick={onClose}>閉じる</button>
      </div>
    );
  };
});

describe("既存コンポーネント統合", () => {
  // テスト用のモックデータ
  const mockBacktestResult: BacktestResult = {
    id: "test-result-1",
    strategy_name: "INTEGRATION_TEST_STRATEGY",
    symbol: "BTC/USDT",
    timeframe: "1h",
    start_date: "2024-01-01",
    end_date: "2024-12-31",
    initial_capital: 100000,
    commission_rate: 0.00055,
    performance_metrics: {
      total_return: 25.8,
      sharpe_ratio: 1.5,
      max_drawdown: -12.4,
      win_rate: 72.3,
      profit_factor: 2.1,
      total_trades: 38,
      winning_trades: 27,
      losing_trades: 11,
      avg_win: 3.2,
      avg_loss: -1.8,
      largest_win: 12.5,
      largest_loss: -4.2,
      avg_trade_duration: 18.7,
      buy_hold_return: 18.9,
    },
    equity_curve: [
      { timestamp: "2024-01-01T00:00:00Z", equity: 100000 },
      { timestamp: "2024-06-01T00:00:00Z", equity: 112000 },
      { timestamp: "2024-12-31T23:59:59Z", equity: 125800 },
    ],
    trade_history: [
      {
        size: 1.2,
        entry_price: 42000,
        exit_price: 44500,
        pnl: 3000,
        return_pct: 0.059,
        entry_time: "2024-02-15T09:30:00Z",
        exit_time: "2024-02-18T14:15:00Z",
      },
      {
        size: 0.8,
        entry_price: 48000,
        exit_price: 46800,
        pnl: -960,
        return_pct: -0.025,
        entry_time: "2024-03-10T11:00:00Z",
        exit_time: "2024-03-12T16:30:00Z",
      },
      {
        size: 1.5,
        entry_price: 45000,
        exit_price: 50000,
        pnl: 7500,
        return_pct: 0.111,
        entry_time: "2024-04-05T08:45:00Z",
        exit_time: "2024-04-08T13:20:00Z",
      },
    ],
    created_at: "2024-12-31T23:59:59Z",
  };

  const mockOnSave = jest.fn();
  const mockOnNewBacktest = jest.fn();

  beforeEach(() => {
    jest.clearAllMocks();
  });

  test("TradeHistoryTableコンポーネントが正しく統合される", () => {
    render(
      <StrategyBacktestResults
        result={mockBacktestResult}
        onSave={mockOnSave}
        onNewBacktest={mockOnNewBacktest}
      />
    );

    // 取引履歴タブをクリック
    const tradesTab = screen.getByRole("tab", { name: "取引履歴" });
    fireEvent.click(tradesTab);

    // TradeHistoryTableが表示されることを確認
    expect(screen.getByTestId("trade-history-table")).toBeInTheDocument();
    expect(screen.getByText("取引履歴テーブル")).toBeInTheDocument();

    // 取引データが正しく渡されることを確認
    expect(screen.getByTestId("trade-0")).toBeInTheDocument();
    expect(screen.getByTestId("trade-1")).toBeInTheDocument();
    expect(screen.getByTestId("trade-2")).toBeInTheDocument();

    // 取引データの内容を確認
    expect(
      screen.getByText(/エントリー: 42000.*イグジット: 44500.*損益: 3000/)
    ).toBeInTheDocument();
    expect(
      screen.getByText(/エントリー: 48000.*イグジット: 46800.*損益: -960/)
    ).toBeInTheDocument();
    expect(
      screen.getByText(/エントリー: 45000.*イグジット: 50000.*損益: 7500/)
    ).toBeInTheDocument();
  });

  test("ChartModalコンポーネントが正しく統合される", () => {
    render(
      <StrategyBacktestResults
        result={mockBacktestResult}
        onSave={mockOnSave}
        onNewBacktest={mockOnNewBacktest}
      />
    );

    // チャート表示ボタンをクリック
    const chartButton = screen.getByText("チャート表示");
    fireEvent.click(chartButton);

    // ChartModalが表示されることを確認
    expect(screen.getByTestId("chart-modal")).toBeInTheDocument();
    expect(screen.getByText("バックテスト結果チャート")).toBeInTheDocument();

    // モーダル内の戦略名を確認（testidを使用）
    const chartModal = screen.getByTestId("chart-modal");
    expect(chartModal).toHaveTextContent("戦略: INTEGRATION_TEST_STRATEGY");

    // モーダルを閉じる
    const closeButton = screen.getByText("閉じる");
    fireEvent.click(closeButton);

    // モーダルが閉じられることを確認
    expect(screen.queryByTestId("chart-modal")).not.toBeInTheDocument();
  });

  test("パフォーマンス指標が正確に計算・表示される", () => {
    render(
      <StrategyBacktestResults
        result={mockBacktestResult}
        onSave={mockOnSave}
        onNewBacktest={mockOnNewBacktest}
      />
    );

    // 主要指標の表示を確認
    expect(screen.getAllByText("25.80%")[0]).toBeInTheDocument(); // 総リターン
    expect(screen.getByText("1.50")).toBeInTheDocument(); // シャープレシオ
    expect(screen.getByText("-12.40%")).toBeInTheDocument(); // 最大ドローダウン
    expect(screen.getByText("72.30%")).toBeInTheDocument(); // 勝率
    expect(screen.getByText("2.10")).toBeInTheDocument(); // プロフィットファクター
    expect(screen.getByText("38")).toBeInTheDocument(); // 総取引数

    // 詳細統計の表示を確認
    expect(screen.getByText("27")).toBeInTheDocument(); // 勝ち取引数
    expect(screen.getByText("11")).toBeInTheDocument(); // 負け取引数
    expect(screen.getByText("3.20%")).toBeInTheDocument(); // 平均勝ち
    expect(screen.getByText("-1.80%")).toBeInTheDocument(); // 平均負け
    expect(screen.getByText("12.50%")).toBeInTheDocument(); // 最大勝ち
    expect(screen.getByText("-4.20%")).toBeInTheDocument(); // 最大負け
  });

  test("Buy & Hold比較が正しく表示される", () => {
    render(
      <StrategyBacktestResults
        result={mockBacktestResult}
        onSave={mockOnSave}
        onNewBacktest={mockOnNewBacktest}
      />
    );

    // Buy & Hold比較セクションの表示を確認
    expect(screen.getByText("Buy & Hold比較")).toBeInTheDocument();
    expect(screen.getByText("戦略リターン")).toBeInTheDocument();
    expect(screen.getByText("Buy & Hold")).toBeInTheDocument();
    expect(screen.getByText("18.90%")).toBeInTheDocument(); // Buy & Hold リターン

    // 戦略がBuy & Holdを上回っていることを確認
    const strategyReturnElements = screen.getAllByText("25.80%");
    const buyHoldReturnElement = screen.getByText("18.90%");
    expect(strategyReturnElements.length).toBeGreaterThan(0);
    expect(buyHoldReturnElement).toBeInTheDocument();
  });

  test("エラー状態での既存コンポーネント統合", () => {
    // 不完全なデータでのテスト
    const incompleteResult: BacktestResult = {
      ...mockBacktestResult,
      trade_history: [],
      performance_metrics: {
        ...mockBacktestResult.performance_metrics,
        total_return: NaN,
        sharpe_ratio: undefined as any,
        max_drawdown: null as any,
      },
    };

    render(
      <StrategyBacktestResults
        result={incompleteResult}
        onSave={mockOnSave}
        onNewBacktest={mockOnNewBacktest}
      />
    );

    // エラー値が適切にフォーマットされることを確認
    expect(screen.getAllByText("0.00%")[0]).toBeInTheDocument(); // NaN -> 0.00%
    expect(screen.getByText("N/A")).toBeInTheDocument(); // undefined -> N/A

    // 取引履歴タブで空の状態が表示されることを確認
    const tradesTab = screen.getByRole("tab", { name: "取引履歴" });
    fireEvent.click(tradesTab);

    expect(screen.getByText("取引履歴がありません")).toBeInTheDocument();
  });

  test("大量データでのパフォーマンス", () => {
    // 大量の取引履歴を持つ結果
    const largeTradeHistory = Array.from({ length: 1000 }, (_, i) => ({
      size: 1.0,
      entry_price: 40000 + i * 10,
      exit_price: 40000 + i * 10 + (i % 2 === 0 ? 500 : -300),
      pnl: i % 2 === 0 ? 500 : -300,
      return_pct: i % 2 === 0 ? 0.0125 : -0.0075,
      entry_time: `2024-01-${String((i % 30) + 1).padStart(2, "0")}T10:00:00Z`,
      exit_time: `2024-01-${String((i % 30) + 1).padStart(2, "0")}T16:00:00Z`,
    }));

    const largeDataResult: BacktestResult = {
      ...mockBacktestResult,
      trade_history: largeTradeHistory,
      performance_metrics: {
        ...mockBacktestResult.performance_metrics,
        total_trades: 1000,
        winning_trades: 500,
        losing_trades: 500,
      },
    };

    const startTime = performance.now();

    render(
      <StrategyBacktestResults
        result={largeDataResult}
        onSave={mockOnSave}
        onNewBacktest={mockOnNewBacktest}
      />
    );

    // 取引履歴タブをクリック
    const tradesTab = screen.getByRole("tab", { name: "取引履歴" });
    fireEvent.click(tradesTab);

    const endTime = performance.now();
    const renderTime = endTime - startTime;

    // レンダリング時間が合理的な範囲内であることを確認（1秒以内）
    expect(renderTime).toBeLessThan(1000);

    // 大量データが正しく表示されることを確認
    // 概要タブに戻って統計を確認
    const overviewTab = screen.getByRole("tab", { name: "概要" });
    fireEvent.click(overviewTab);

    expect(screen.getByText("1000")).toBeInTheDocument(); // 総取引数
    expect(screen.getAllByText("500")[0]).toBeInTheDocument(); // 勝ち取引数

    // 再度取引履歴タブに戻る
    fireEvent.click(tradesTab);
    expect(screen.getByTestId("trade-history-table")).toBeInTheDocument();
  });

  test("レスポンシブデザインでの表示", () => {
    // モバイルサイズでのテスト
    Object.defineProperty(window, "innerWidth", {
      writable: true,
      configurable: true,
      value: 375,
    });

    render(
      <StrategyBacktestResults
        result={mockBacktestResult}
        onSave={mockOnSave}
        onNewBacktest={mockOnNewBacktest}
      />
    );

    // メトリックカードが適切に表示されることを確認
    expect(screen.getByText("パフォーマンス指標")).toBeInTheDocument();
    expect(screen.getByText("詳細統計")).toBeInTheDocument();

    // ボタンが適切に配置されることを確認
    expect(screen.getByText("チャート表示")).toBeInTheDocument();
    expect(screen.getByText("結果を保存")).toBeInTheDocument();
    expect(screen.getByText("新しいバックテスト")).toBeInTheDocument();
  });

  test("アクセシビリティ要件の確認", () => {
    render(
      <StrategyBacktestResults
        result={mockBacktestResult}
        onSave={mockOnSave}
        onNewBacktest={mockOnNewBacktest}
      />
    );

    // タブのrole属性が正しく設定されていることを確認
    expect(screen.getByRole("tab", { name: "概要" })).toBeInTheDocument();
    expect(screen.getByRole("tab", { name: "取引履歴" })).toBeInTheDocument();

    // ボタンが適切にラベル付けされていることを確認
    expect(
      screen.getByRole("button", { name: "チャート表示" })
    ).toBeInTheDocument();
    expect(
      screen.getByRole("button", { name: "結果を保存" })
    ).toBeInTheDocument();
    expect(
      screen.getByRole("button", { name: "新しいバックテスト" })
    ).toBeInTheDocument();

    // タブナビゲーションがキーボードで操作可能であることを確認
    const overviewTab = screen.getByRole("tab", { name: "概要" });
    const tradesTab = screen.getByRole("tab", { name: "取引履歴" });

    expect(overviewTab).toHaveAttribute("aria-selected", "true");
    expect(tradesTab).toHaveAttribute("aria-selected", "false");

    // タブをクリックして状態が変わることを確認
    fireEvent.click(tradesTab);

    expect(overviewTab).toHaveAttribute("aria-selected", "false");
    expect(tradesTab).toHaveAttribute("aria-selected", "true");
  });

  test("国際化対応の確認", () => {
    render(
      <StrategyBacktestResults
        result={mockBacktestResult}
        onSave={mockOnSave}
        onNewBacktest={mockOnNewBacktest}
      />
    );

    // 日本語のラベルが正しく表示されることを確認
    expect(screen.getByText("バックテスト結果")).toBeInTheDocument();
    expect(screen.getByText("パフォーマンス指標")).toBeInTheDocument();
    expect(screen.getByText("詳細統計")).toBeInTheDocument();
    expect(screen.getByText("Buy & Hold比較")).toBeInTheDocument();

    // 通貨フォーマットが日本円で表示されることを確認
    expect(screen.getByText("￥100,000")).toBeInTheDocument();

    // 日付フォーマットが日本語形式で表示されることを確認
    expect(
      screen.getByText(/2024\/01\/01.*-.*2024\/12\/31/)
    ).toBeInTheDocument();
  });
});
