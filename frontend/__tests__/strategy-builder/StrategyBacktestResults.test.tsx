/**
 * StrategyBacktestResultsコンポーネントのテスト
 *
 * TDDアプローチでバックテスト結果表示コンポーネントをテストします。
 */

import { render, screen, fireEvent, waitFor } from "@testing-library/react";
import "@testing-library/jest-dom";
import StrategyBacktestResults from "@/components/strategy-builder/StrategyBacktestResults";
import { BacktestResult } from "@/types/backtest";

// モックデータ
const mockBacktestResult: BacktestResult = {
  id: "test-result-1",
  strategy_name: "CUSTOM_1234567890",
  symbol: "BTC/USDT",
  timeframe: "1h",
  start_date: "2024-01-01",
  end_date: "2024-12-31",
  initial_capital: 100000,
  commission_rate: 0.00055,
  performance_metrics: {
    total_return: 15.5,
    sharpe_ratio: 1.2,
    max_drawdown: -8.3,
    win_rate: 65.5,
    profit_factor: 1.8,
    total_trades: 45,
    winning_trades: 29,
    losing_trades: 16,
    avg_win: 2.5,
    avg_loss: -1.2,
    largest_win: 8.5,
    largest_loss: -3.2,
    avg_trade_duration: 24.5,
    buy_hold_return: 12.3,
  },
  equity_curve: [
    { timestamp: "2024-01-01T00:00:00Z", equity: 100000 },
    { timestamp: "2024-06-01T00:00:00Z", equity: 105000 },
    { timestamp: "2024-12-31T23:59:59Z", equity: 115500 },
  ],
  trade_history: [
    {
      size: 1.0,
      entry_price: 45000,
      exit_price: 46000,
      pnl: 1000,
      return_pct: 0.022,
      entry_time: "2024-01-15T10:00:00Z",
      exit_time: "2024-01-16T14:30:00Z",
    },
    {
      size: 1.5,
      entry_price: 47000,
      exit_price: 46500,
      pnl: -750,
      return_pct: -0.011,
      entry_time: "2024-02-10T09:15:00Z",
      exit_time: "2024-02-11T16:45:00Z",
    },
  ],
  created_at: "2024-12-31T23:59:59Z",
};

const mockOnSave = jest.fn();
const mockOnNewBacktest = jest.fn();

describe("StrategyBacktestResultsコンポーネント", () => {
  beforeEach(() => {
    jest.clearAllMocks();
  });

  test("コンポーネントが正しくレンダリングされる", () => {
    render(
      <StrategyBacktestResults
        result={mockBacktestResult}
        onSave={mockOnSave}
        onNewBacktest={mockOnNewBacktest}
      />
    );

    // タイトルが表示されることを確認
    expect(screen.getByText("バックテスト結果")).toBeInTheDocument();

    // 戦略名が表示されることを確認
    expect(screen.getByText(/戦略:.*CUSTOM_1234567890/)).toBeInTheDocument();
  });

  test("パフォーマンス指標が正しく表示される", () => {
    render(
      <StrategyBacktestResults
        result={mockBacktestResult}
        onSave={mockOnSave}
        onNewBacktest={mockOnNewBacktest}
      />
    );

    // 主要な指標が表示されることを確認
    expect(screen.getAllByText("15.50%")[0]).toBeInTheDocument(); // 総リターン
    expect(screen.getByText("1.20")).toBeInTheDocument(); // シャープレシオ
    expect(screen.getByText("-8.30%")).toBeInTheDocument(); // 最大ドローダウン
    expect(screen.getByText("65.50%")).toBeInTheDocument(); // 勝率
    expect(screen.getByText("45")).toBeInTheDocument(); // 総取引数
  });

  test("タブナビゲーションが機能する", () => {
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

    // 取引履歴が表示されることを確認
    expect(screen.getByText("取引履歴詳細")).toBeInTheDocument();
  });

  test("取引履歴が正しく表示される", () => {
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

    // 取引データが表示されることを確認
    expect(screen.getByText("￥45,000")).toBeInTheDocument(); // エントリー価格
    expect(screen.getByText("￥46,000")).toBeInTheDocument(); // イグジット価格
    expect(screen.getByText("￥1,000")).toBeInTheDocument(); // 損益
  });

  test("チャート表示ボタンが機能する", () => {
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

    // チャートモーダルが表示されることを確認
    expect(screen.getByText("バックテスト結果チャート")).toBeInTheDocument();
  });

  test("結果保存ボタンが機能する", async () => {
    render(
      <StrategyBacktestResults
        result={mockBacktestResult}
        onSave={mockOnSave}
        onNewBacktest={mockOnNewBacktest}
      />
    );

    // 結果保存ボタンをクリック
    const saveButton = screen.getByText("結果を保存");
    fireEvent.click(saveButton);

    // onSaveコールバックが呼ばれることを確認
    await waitFor(() => {
      expect(mockOnSave).toHaveBeenCalledWith(mockBacktestResult);
    });
  });

  test("新しいバックテストボタンが機能する", () => {
    render(
      <StrategyBacktestResults
        result={mockBacktestResult}
        onSave={mockOnSave}
        onNewBacktest={mockOnNewBacktest}
      />
    );

    // 新しいバックテストボタンをクリック
    const newBacktestButton = screen.getByText("新しいバックテスト");
    fireEvent.click(newBacktestButton);

    // onNewBacktestコールバックが呼ばれることを確認
    expect(mockOnNewBacktest).toHaveBeenCalled();
  });

  test("戦略設定情報が表示される", () => {
    render(
      <StrategyBacktestResults
        result={mockBacktestResult}
        onSave={mockOnSave}
        onNewBacktest={mockOnNewBacktest}
      />
    );

    // 設定情報が表示されることを確認
    expect(screen.getByText("BTC/USDT")).toBeInTheDocument();
    expect(screen.getByText("1h")).toBeInTheDocument();
    expect(screen.getByText("￥100,000")).toBeInTheDocument();
  });

  test("Buy & Hold比較が表示される", () => {
    render(
      <StrategyBacktestResults
        result={mockBacktestResult}
        onSave={mockOnSave}
        onNewBacktest={mockOnNewBacktest}
      />
    );

    // Buy & Hold比較が表示されることを確認
    expect(screen.getByText("12.30%")).toBeInTheDocument(); // Buy & Hold リターン
  });

  test("取引履歴が空の場合の表示", () => {
    const resultWithoutTrades = {
      ...mockBacktestResult,
      trade_history: [],
    };

    render(
      <StrategyBacktestResults
        result={resultWithoutTrades}
        onSave={mockOnSave}
        onNewBacktest={mockOnNewBacktest}
      />
    );

    // 取引履歴タブをクリック
    const tradesTab = screen.getByRole("tab", { name: "取引履歴" });
    fireEvent.click(tradesTab);

    // 取引履歴がない旨のメッセージが表示されることを確認
    expect(screen.getByText("取引履歴がありません")).toBeInTheDocument();
  });

  test("パフォーマンス指標の色分けが正しく適用される", () => {
    render(
      <StrategyBacktestResults
        result={mockBacktestResult}
        onSave={mockOnSave}
        onNewBacktest={mockOnNewBacktest}
      />
    );

    // 負の最大ドローダウンが赤色で表示されることを確認
    const maxDrawdownElement = screen.getByText("-8.30%");
    expect(maxDrawdownElement).toHaveClass("text-red-400");

    // 正の総リターンが緑色で表示されることを確認（最初の要素を取得）
    const totalReturnElements = screen.getAllByText("15.50%");
    expect(totalReturnElements[0]).toHaveClass("text-green-400");
  });

  test("詳細な取引統計が表示される", () => {
    render(
      <StrategyBacktestResults
        result={mockBacktestResult}
        onSave={mockOnSave}
        onNewBacktest={mockOnNewBacktest}
      />
    );

    // 詳細統計が表示されることを確認
    expect(screen.getByText("29")).toBeInTheDocument(); // 勝ち取引数
    expect(screen.getByText("16")).toBeInTheDocument(); // 負け取引数
    expect(screen.getByText("1.80")).toBeInTheDocument(); // プロフィットファクター
  });

  test("日付フォーマットが正しく表示される", () => {
    render(
      <StrategyBacktestResults
        result={mockBacktestResult}
        onSave={mockOnSave}
        onNewBacktest={mockOnNewBacktest}
      />
    );

    // 日付が日本語形式で表示されることを確認（期間表示内で）
    expect(
      screen.getByText(/2024\/01\/01.*-.*2024\/12\/31/)
    ).toBeInTheDocument();
  });
});
