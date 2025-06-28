/**
 * バックテスト結果保存機能のテスト
 *
 * TDDアプローチで結果保存・履歴管理機能をテストします。
 */

import { render, screen, fireEvent, waitFor } from "@testing-library/react";
import "@testing-library/jest-dom";
import BacktestTab from "@/components/strategy-builder/BacktestTab";
import { SelectedIndicator, Condition } from "@/hooks/useStrategyBuilder";
import { BacktestResult } from "@/types/backtest";

// fetchのモック
global.fetch = jest.fn();

// localStorageのモック
const localStorageMock = {
  getItem: jest.fn(),
  setItem: jest.fn(),
  removeItem: jest.fn(),
  clear: jest.fn(),
};
Object.defineProperty(window, "localStorage", {
  value: localStorageMock,
});

describe("バックテスト結果保存機能", () => {
  // テスト用のモックデータ
  const mockSelectedIndicators: SelectedIndicator[] = [
    {
      name: "SMA",
      type: "SMA",
      params: { period: 20 },
      parameters: { period: 20 },
      enabled: true,
    },
  ];

  const mockEntryConditions: Condition[] = [
    {
      type: "threshold",
      indicator1: "RSI",
      operator: "<",
      value: 30,
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
    ],
    created_at: "2024-12-31T23:59:59Z",
  };

  const mockOnBacktestRun = jest.fn();
  const mockOnResultSave = jest.fn();

  beforeEach(() => {
    jest.clearAllMocks();
    localStorageMock.getItem.mockReturnValue(null);
    (fetch as jest.Mock).mockResolvedValue({
      ok: true,
      json: async () => ({
        success: true,
        result: mockBacktestResult,
      }),
    });
  });

  test("バックテスト結果が正常に保存される", async () => {
    // バックテスト実行をモック
    mockOnBacktestRun.mockImplementation(async (config) => {
      // 結果表示状態をシミュレート
      return mockBacktestResult;
    });

    render(
      <BacktestTab
        selectedIndicators={mockSelectedIndicators}
        entryConditions={mockEntryConditions}
        exitConditions={mockExitConditions}
        onBacktestRun={mockOnBacktestRun}
        onResultSave={mockOnResultSave}
      />
    );

    // バックテスト実行
    const runButton = screen.getByText("バックテスト実行");
    fireEvent.click(runButton);

    // 結果が表示されるまで待機
    await waitFor(() => {
      expect(screen.getByText("バックテスト結果")).toBeInTheDocument();
    });

    // 結果保存ボタンをクリック
    const saveButton = screen.getByText("結果を保存");
    fireEvent.click(saveButton);

    // onResultSaveが呼ばれることを確認
    await waitFor(() => {
      expect(mockOnResultSave).toHaveBeenCalledWith(mockBacktestResult);
    });
  });

  test("保存された結果がlocalStorageに格納される", async () => {
    const mockSaveToStorage = jest.fn();

    // useStrategyBacktestResultsフックをモック
    jest.doMock("@/hooks/useStrategyBacktestResults", () => ({
      useStrategyBacktestResults: () => ({
        savedResults: [],
        saveResult: mockSaveToStorage,
        deleteResult: jest.fn(),
        loadResults: jest.fn(),
      }),
    }));

    render(
      <BacktestTab
        selectedIndicators={mockSelectedIndicators}
        entryConditions={mockEntryConditions}
        exitConditions={mockExitConditions}
        onBacktestRun={mockOnBacktestRun}
        onResultSave={mockOnResultSave}
      />
    );

    // バックテスト実行
    const runButton = screen.getByText("バックテスト実行");
    fireEvent.click(runButton);

    await waitFor(() => {
      expect(screen.getByText("バックテスト結果")).toBeInTheDocument();
    });

    // 結果保存
    const saveButton = screen.getByText("結果を保存");
    fireEvent.click(saveButton);

    // localStorageに保存されることを確認
    await waitFor(() => {
      expect(localStorageMock.setItem).toHaveBeenCalledWith(
        "strategy_backtest_results",
        expect.stringContaining("CUSTOM_1234567890")
      );
    });
  });

  test("保存済み結果の履歴が表示される", async () => {
    // 保存済み結果をモック
    const savedResults = [
      {
        ...mockBacktestResult,
        id: "saved-result-1",
        strategy_name: "SAVED_STRATEGY_1",
        created_at: "2024-12-30T12:00:00Z",
      },
      {
        ...mockBacktestResult,
        id: "saved-result-2",
        strategy_name: "SAVED_STRATEGY_2",
        created_at: "2024-12-29T15:30:00Z",
      },
    ];

    localStorageMock.getItem.mockReturnValue(JSON.stringify(savedResults));

    render(
      <BacktestTab
        selectedIndicators={mockSelectedIndicators}
        entryConditions={mockEntryConditions}
        exitConditions={mockExitConditions}
        onBacktestRun={mockOnBacktestRun}
        onResultSave={mockOnResultSave}
      />
    );

    // 履歴タブをクリック
    const historyTab = screen.getByText("履歴");
    fireEvent.click(historyTab);

    // 保存済み結果が表示されることを確認
    expect(screen.getByText("SAVED_STRATEGY_1")).toBeInTheDocument();
    expect(screen.getByText("SAVED_STRATEGY_2")).toBeInTheDocument();
  });

  test("保存済み結果を削除できる", async () => {
    const savedResults = [
      {
        ...mockBacktestResult,
        id: "saved-result-1",
        strategy_name: "SAVED_STRATEGY_1",
      },
    ];

    localStorageMock.getItem.mockReturnValue(JSON.stringify(savedResults));

    render(
      <BacktestTab
        selectedIndicators={mockSelectedIndicators}
        entryConditions={mockEntryConditions}
        exitConditions={mockExitConditions}
        onBacktestRun={mockOnBacktestRun}
        onResultSave={mockOnResultSave}
      />
    );

    // 履歴タブをクリック
    const historyTab = screen.getByText("履歴");
    fireEvent.click(historyTab);

    // 削除ボタンをクリック
    const deleteButton = screen.getByTitle("削除");
    fireEvent.click(deleteButton);

    // localStorageから削除されることを確認
    await waitFor(() => {
      expect(localStorageMock.setItem).toHaveBeenCalledWith(
        "strategy_backtest_results",
        "[]"
      );
    });
  });

  test("保存済み結果を再表示できる", async () => {
    const savedResults = [
      {
        ...mockBacktestResult,
        id: "saved-result-1",
        strategy_name: "SAVED_STRATEGY_1",
      },
    ];

    localStorageMock.getItem.mockReturnValue(JSON.stringify(savedResults));

    render(
      <BacktestTab
        selectedIndicators={mockSelectedIndicators}
        entryConditions={mockEntryConditions}
        exitConditions={mockExitConditions}
        onBacktestRun={mockOnBacktestRun}
        onResultSave={mockOnResultSave}
      />
    );

    // 履歴タブをクリック
    const historyTab = screen.getByText("履歴");
    fireEvent.click(historyTab);

    // 結果をクリックして表示
    const resultRow = screen.getByText("SAVED_STRATEGY_1");
    fireEvent.click(resultRow);

    // 結果詳細が表示されることを確認
    expect(screen.getByText("バックテスト結果")).toBeInTheDocument();
    expect(screen.getByText(/戦略:.*SAVED_STRATEGY_1/)).toBeInTheDocument();
  });

  test("結果保存時にエラーハンドリングが機能する", async () => {
    // localStorageエラーをシミュレート
    localStorageMock.setItem.mockImplementation(() => {
      throw new Error("Storage quota exceeded");
    });

    mockOnBacktestRun.mockResolvedValue(mockBacktestResult);

    render(
      <BacktestTab
        selectedIndicators={mockSelectedIndicators}
        entryConditions={mockEntryConditions}
        exitConditions={mockExitConditions}
        onBacktestRun={mockOnBacktestRun}
        onResultSave={mockOnResultSave}
      />
    );

    // バックテスト実行
    const runButton = screen.getByText("バックテスト実行");
    fireEvent.click(runButton);

    await waitFor(() => {
      expect(screen.getByText("バックテスト結果")).toBeInTheDocument();
    });

    // 結果保存を試行
    const saveButton = screen.getByText("結果を保存");
    fireEvent.click(saveButton);

    // エラーメッセージが表示されることを確認
    await waitFor(() => {
      expect(screen.getByText(/保存に失敗しました/)).toBeInTheDocument();
    });
  });

  test("最大保存数制限が機能する", async () => {
    // 既に最大数の結果が保存されている状態をシミュレート
    const maxResults = Array.from({ length: 50 }, (_, i) => ({
      ...mockBacktestResult,
      id: `saved-result-${i}`,
      strategy_name: `STRATEGY_${i}`,
      created_at: new Date(Date.now() - i * 1000).toISOString(),
    }));

    localStorageMock.getItem.mockReturnValue(JSON.stringify(maxResults));

    render(
      <BacktestTab
        selectedIndicators={mockSelectedIndicators}
        entryConditions={mockEntryConditions}
        exitConditions={mockExitConditions}
        onBacktestRun={mockOnBacktestRun}
        onResultSave={mockOnResultSave}
      />
    );

    // バックテスト実行
    const runButton = screen.getByText("バックテスト実行");
    fireEvent.click(runButton);

    await waitFor(() => {
      expect(screen.getByText("バックテスト結果")).toBeInTheDocument();
    });

    // 結果保存
    const saveButton = screen.getByText("結果を保存");
    fireEvent.click(saveButton);

    // 古い結果が削除されて新しい結果が保存されることを確認
    await waitFor(() => {
      expect(localStorageMock.setItem).toHaveBeenCalledWith(
        "strategy_backtest_results",
        expect.stringContaining("CUSTOM_1234567890")
      );
    });

    // 保存された配列の長さが最大数以下であることを確認
    const savedData = JSON.parse(localStorageMock.setItem.mock.calls[0][1]);
    expect(savedData.length).toBeLessThanOrEqual(50);
  });

  test("結果のエクスポート機能が動作する", async () => {
    // Blob と URL.createObjectURL をモック
    global.Blob = jest.fn().mockImplementation((content, options) => ({
      content,
      options,
    }));
    global.URL.createObjectURL = jest.fn().mockReturnValue("mock-url");
    global.URL.revokeObjectURL = jest.fn();

    // download属性を持つリンクのクリックをモック
    const mockClick = jest.fn();
    const mockLink = {
      href: "",
      download: "",
      click: mockClick,
      style: {},
    };
    jest.spyOn(document, "createElement").mockReturnValue(mockLink as any);

    render(
      <BacktestTab
        selectedIndicators={mockSelectedIndicators}
        entryConditions={mockEntryConditions}
        exitConditions={mockExitConditions}
        onBacktestRun={mockOnBacktestRun}
        onResultSave={mockOnResultSave}
      />
    );

    // 保存済み結果をモック
    const savedResults = [
      {
        ...mockBacktestResult,
        id: "saved-result-1",
        strategy_name: "SAVED_STRATEGY_1",
        saved_at: "2024-12-30T12:00:00Z",
      },
    ];

    localStorageMock.getItem.mockReturnValue(JSON.stringify(savedResults));

    // 履歴タブをクリック
    const historyTab = screen.getByText("履歴");
    fireEvent.click(historyTab);

    // エクスポートボタンをクリック
    const exportButton = screen.getByText("エクスポート");
    fireEvent.click(exportButton);

    // ファイルダウンロードが実行されることを確認
    expect(global.Blob).toHaveBeenCalledWith(
      [expect.stringContaining("SAVED_STRATEGY_1")],
      { type: "application/json" }
    );
    expect(mockClick).toHaveBeenCalled();
  });
});
