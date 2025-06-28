/**
 * BacktestTabコンポーネントのテスト
 *
 * TDDアプローチでBacktestTabコンポーネントの機能をテストします。
 */

import { render, screen, fireEvent, waitFor } from "@testing-library/react";
import "@testing-library/jest-dom";
import BacktestTab from "@/components/strategy-builder/BacktestTab";
import { SelectedIndicator, Condition } from "@/hooks/useStrategyBuilder";

// モックデータ
const mockSelectedIndicators: SelectedIndicator[] = [
  {
    name: "SMA",
    params: { period: 20 },
    enabled: true,
  },
  {
    name: "RSI",
    params: { period: 14 },
    enabled: true,
  },
];

const mockEntryConditions: Condition[] = [
  {
    type: "indicator_comparison",
    indicator1: "SMA",
    operator: ">",
    value: 100,
  },
];

const mockExitConditions: Condition[] = [
  {
    type: "indicator_comparison",
    indicator1: "RSI",
    operator: "<",
    value: 30,
  },
];

// モック関数
const mockOnBacktestRun = jest.fn();
const mockOnResultSave = jest.fn();

describe("BacktestTabコンポーネント", () => {
  beforeEach(() => {
    jest.clearAllMocks();
  });

  test("コンポーネントが正しくレンダリングされる", () => {
    render(
      <BacktestTab
        selectedIndicators={mockSelectedIndicators}
        entryConditions={mockEntryConditions}
        exitConditions={mockExitConditions}
        onBacktestRun={mockOnBacktestRun}
        onResultSave={mockOnResultSave}
      />
    );

    // バックテストタブのタイトルが表示されることを確認（最初のh3要素）
    expect(screen.getAllByText("バックテスト設定")[0]).toBeInTheDocument();
  });

  test("バックテスト設定フォームが表示される", () => {
    render(
      <BacktestTab
        selectedIndicators={mockSelectedIndicators}
        entryConditions={mockEntryConditions}
        exitConditions={mockExitConditions}
        onBacktestRun={mockOnBacktestRun}
        onResultSave={mockOnResultSave}
      />
    );

    // 必要な設定項目が表示されることを確認
    expect(screen.getByLabelText(/銘柄/)).toBeInTheDocument();
    expect(screen.getByLabelText(/時間軸/)).toBeInTheDocument();
    expect(screen.getByLabelText(/開始日/)).toBeInTheDocument();
    expect(screen.getByLabelText(/終了日/)).toBeInTheDocument();
    expect(screen.getByLabelText(/初期資金/)).toBeInTheDocument();
    expect(screen.getByLabelText(/手数料率/)).toBeInTheDocument();
  });

  test("バックテスト実行ボタンが表示される", () => {
    render(
      <BacktestTab
        selectedIndicators={mockSelectedIndicators}
        entryConditions={mockEntryConditions}
        exitConditions={mockExitConditions}
        onBacktestRun={mockOnBacktestRun}
        onResultSave={mockOnResultSave}
      />
    );

    const runButton = screen.getByText("バックテスト実行");
    expect(runButton).toBeInTheDocument();
    expect(runButton).not.toBeDisabled();
  });

  test("戦略サマリーが表示される", () => {
    render(
      <BacktestTab
        selectedIndicators={mockSelectedIndicators}
        entryConditions={mockEntryConditions}
        exitConditions={mockExitConditions}
        onBacktestRun={mockOnBacktestRun}
        onResultSave={mockOnResultSave}
      />
    );

    // 選択された指標が表示されることを確認
    expect(screen.getByText("SMA")).toBeInTheDocument();
    expect(screen.getByText("RSI")).toBeInTheDocument();

    // 条件数が表示されることを確認
    expect(screen.getByText(/エントリー条件.*1件/)).toBeInTheDocument();
    expect(screen.getByText(/イグジット条件.*1件/)).toBeInTheDocument();
  });

  test("バックテスト設定の初期値が正しく設定される", () => {
    render(
      <BacktestTab
        selectedIndicators={mockSelectedIndicators}
        entryConditions={mockEntryConditions}
        exitConditions={mockExitConditions}
        onBacktestRun={mockOnBacktestRun}
        onResultSave={mockOnResultSave}
      />
    );

    // デフォルト値が設定されていることを確認
    const symbolSelect = screen.getByLabelText(/銘柄/) as HTMLSelectElement;
    const timeframeSelect = screen.getByLabelText(
      /時間軸/
    ) as HTMLSelectElement;

    expect(symbolSelect.value).toBe("BTC/USDT");
    expect(timeframeSelect.value).toBe("1h");
    expect(screen.getByDisplayValue("100000")).toBeInTheDocument();
    expect(screen.getByDisplayValue("0.00055")).toBeInTheDocument();
  });

  test("設定値を変更できる", async () => {
    render(
      <BacktestTab
        selectedIndicators={mockSelectedIndicators}
        entryConditions={mockEntryConditions}
        exitConditions={mockExitConditions}
        onBacktestRun={mockOnBacktestRun}
        onResultSave={mockOnResultSave}
      />
    );

    // 銘柄を変更
    const symbolSelect = screen.getByLabelText(/銘柄/);
    fireEvent.change(symbolSelect, { target: { value: "ETH/USDT" } });
    expect(screen.getByDisplayValue("ETH/USDT")).toBeInTheDocument();

    // 初期資金を変更
    const capitalInput = screen.getByLabelText(/初期資金/);
    fireEvent.change(capitalInput, { target: { value: "50000" } });
    expect(screen.getByDisplayValue("50000")).toBeInTheDocument();
  });

  test("バックテスト実行ボタンをクリックすると適切な設定でコールバックが呼ばれる", async () => {
    render(
      <BacktestTab
        selectedIndicators={mockSelectedIndicators}
        entryConditions={mockEntryConditions}
        exitConditions={mockExitConditions}
        onBacktestRun={mockOnBacktestRun}
        onResultSave={mockOnResultSave}
      />
    );

    // バックテスト実行ボタンをクリック
    const runButton = screen.getByText("バックテスト実行");
    fireEvent.click(runButton);

    await waitFor(() => {
      expect(mockOnBacktestRun).toHaveBeenCalledWith(
        expect.objectContaining({
          strategy_name: expect.any(String),
          symbol: "BTC/USDT",
          timeframe: "1h",
          start_date: expect.any(String),
          end_date: expect.any(String),
          initial_capital: 100000,
          commission_rate: 0.00055,
          strategy_config: expect.objectContaining({
            strategy_type: "USER_CUSTOM",
            parameters: expect.objectContaining({
              strategy_gene: expect.any(Object),
            }),
          }),
        })
      );
    });
  });

  test("バリデーションエラーが表示される", async () => {
    render(
      <BacktestTab
        selectedIndicators={mockSelectedIndicators}
        entryConditions={mockEntryConditions}
        exitConditions={mockExitConditions}
        onBacktestRun={mockOnBacktestRun}
        onResultSave={mockOnResultSave}
      />
    );

    // 無効な初期資金を設定
    const capitalInput = screen.getByLabelText(/初期資金/);
    fireEvent.change(capitalInput, { target: { value: "-1000" } });

    // バックテスト実行ボタンをクリック
    const runButton = screen.getByText("バックテスト実行");
    fireEvent.click(runButton);

    // エラーメッセージが表示されることを確認
    await waitFor(() => {
      expect(
        screen.getByText(/初期資金は正の値である必要があります/)
      ).toBeInTheDocument();
    });

    // コールバックが呼ばれないことを確認
    expect(mockOnBacktestRun).not.toHaveBeenCalled();
  });

  test("ローディング状態が表示される", async () => {
    // ローディング状態をシミュレート
    const slowMockOnBacktestRun = jest
      .fn()
      .mockImplementation(
        () => new Promise((resolve) => setTimeout(resolve, 100))
      );

    render(
      <BacktestTab
        selectedIndicators={mockSelectedIndicators}
        entryConditions={mockEntryConditions}
        exitConditions={mockExitConditions}
        onBacktestRun={slowMockOnBacktestRun}
        onResultSave={mockOnResultSave}
      />
    );

    // バックテスト実行ボタンをクリック
    const runButton = screen.getByText("バックテスト実行");
    fireEvent.click(runButton);

    // ローディング状態が表示されることを確認
    expect(screen.getByText("実行中...")).toBeInTheDocument();
    expect(runButton).toBeDisabled();

    // ローディングが完了することを確認
    await waitFor(() => {
      expect(screen.queryByText("実行中...")).not.toBeInTheDocument();
      expect(runButton).not.toBeDisabled();
    });
  });

  test("指標や条件が未設定の場合は警告が表示される", () => {
    render(
      <BacktestTab
        selectedIndicators={[]}
        entryConditions={[]}
        exitConditions={[]}
        onBacktestRun={mockOnBacktestRun}
        onResultSave={mockOnResultSave}
      />
    );

    // 警告メッセージが表示されることを確認
    expect(screen.getByText(/戦略が設定されていません/)).toBeInTheDocument();

    // バックテスト実行ボタンが無効化されることを確認
    const runButton = screen.getByText("バックテスト実行");
    expect(runButton).toBeDisabled();
  });
});
