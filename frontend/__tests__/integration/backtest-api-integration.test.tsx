/**
 * バックテストAPI統合テスト
 * 
 * TDDアプローチでストラテジービルダーとバックテストAPIの統合をテストします。
 */

import { render, screen, fireEvent, waitFor } from '@testing-library/react';
import '@testing-library/jest-dom';
import BacktestTab from '@/components/strategy-builder/BacktestTab';
import { SelectedIndicator, Condition } from '@/hooks/useStrategyBuilder';

// fetchのモック
global.fetch = jest.fn();

describe('バックテストAPI統合', () => {
  // テスト用のモックデータ
  const mockSelectedIndicators: SelectedIndicator[] = [
    {
      name: 'SMA',
      type: 'SMA',
      params: { period: 20 },
      parameters: { period: 20 },
      enabled: true,
    },
    {
      name: 'RSI',
      type: 'RSI',
      params: { period: 14 },
      parameters: { period: 14 },
      enabled: true,
    },
  ];

  const mockEntryConditions: Condition[] = [
    {
      type: 'threshold',
      indicator1: 'RSI',
      operator: '<',
      value: 30,
    },
  ];

  const mockExitConditions: Condition[] = [
    {
      type: 'threshold',
      indicator1: 'RSI',
      operator: '>',
      value: 70,
    },
  ];

  const mockOnBacktestRun = jest.fn();
  const mockOnResultSave = jest.fn();

  beforeEach(() => {
    jest.clearAllMocks();
    (fetch as jest.Mock).mockResolvedValue({
      ok: true,
      json: async () => ({
        success: true,
        result: {
          id: 'test-result-1',
          strategy_name: 'CUSTOM_123456789',
          performance_metrics: {
            total_return: 15.5,
            sharpe_ratio: 1.2,
            max_drawdown: -8.3,
          },
        },
      }),
    });
  });

  test('バックテスト実行時に正しいAPI呼び出しが行われる', async () => {
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
    const runButton = screen.getByText('バックテスト実行');
    fireEvent.click(runButton);

    await waitFor(() => {
      expect(mockOnBacktestRun).toHaveBeenCalledWith(
        expect.objectContaining({
          strategy_name: expect.stringMatching(/^CUSTOM_\d+$/),
          symbol: 'BTC/USDT',
          timeframe: '1h',
          start_date: '2024-01-01',
          end_date: '2024-12-31',
          initial_capital: 100000,
          commission_rate: 0.00055,
          strategy_config: expect.objectContaining({
            strategy_type: 'USER_CUSTOM',
            parameters: expect.objectContaining({
              strategy_gene: expect.objectContaining({
                id: expect.stringMatching(/^user_strategy_\d+$/),
                indicators: expect.arrayContaining([
                  expect.objectContaining({
                    type: 'SMA',
                    parameters: { period: 20 },
                    enabled: true,
                  }),
                  expect.objectContaining({
                    type: 'RSI',
                    parameters: { period: 14 },
                    enabled: true,
                  }),
                ]),
                entry_conditions: expect.arrayContaining([
                  expect.objectContaining({
                    type: 'threshold',
                    operator: '<',
                    indicator: 'RSI',
                    value: 30,
                  }),
                ]),
                exit_conditions: expect.arrayContaining([
                  expect.objectContaining({
                    type: 'threshold',
                    operator: '>',
                    indicator: 'RSI',
                    value: 70,
                  }),
                ]),
              }),
            }),
          }),
        })
      );
    });
  });

  test('カスタム設定でのバックテスト実行', async () => {
    render(
      <BacktestTab
        selectedIndicators={mockSelectedIndicators}
        entryConditions={mockEntryConditions}
        exitConditions={mockExitConditions}
        onBacktestRun={mockOnBacktestRun}
        onResultSave={mockOnResultSave}
      />
    );

    // 設定を変更
    const symbolSelect = screen.getByLabelText(/銘柄/);
    fireEvent.change(symbolSelect, { target: { value: 'ETH/USDT' } });

    const timeframeSelect = screen.getByLabelText(/時間軸/);
    fireEvent.change(timeframeSelect, { target: { value: '4h' } });

    const capitalInput = screen.getByLabelText(/初期資金/);
    fireEvent.change(capitalInput, { target: { value: '50000' } });

    // バックテスト実行
    const runButton = screen.getByText('バックテスト実行');
    fireEvent.click(runButton);

    await waitFor(() => {
      expect(mockOnBacktestRun).toHaveBeenCalledWith(
        expect.objectContaining({
          symbol: 'ETH/USDT',
          timeframe: '4h',
          initial_capital: 50000,
        })
      );
    });
  });

  test('複数の指標と条件を含む複雑な戦略のバックテスト', async () => {
    const complexIndicators: SelectedIndicator[] = [
      {
        name: 'SMA',
        type: 'SMA',
        params: { period: 20 },
        parameters: { period: 20 },
        enabled: true,
      },
      {
        name: 'EMA',
        type: 'EMA',
        params: { period: 12 },
        parameters: { period: 12 },
        enabled: true,
      },
      {
        name: 'RSI',
        type: 'RSI',
        params: { period: 14 },
        parameters: { period: 14 },
        enabled: true,
      },
      {
        name: 'MACD',
        type: 'MACD',
        params: { fast_period: 12, slow_period: 26, signal_period: 9 },
        parameters: { fast_period: 12, slow_period: 26, signal_period: 9 },
        enabled: false, // 無効化
      },
    ];

    const complexEntryConditions: Condition[] = [
      {
        type: 'threshold',
        indicator1: 'RSI',
        operator: '<',
        value: 30,
      },
      {
        type: 'crossover',
        indicator1: 'EMA',
        indicator2: 'SMA',
        operator: '>',
      },
    ];

    const complexExitConditions: Condition[] = [
      {
        type: 'threshold',
        indicator1: 'RSI',
        operator: '>',
        value: 70,
      },
      {
        type: 'crossover',
        indicator1: 'SMA',
        indicator2: 'EMA',
        operator: '>',
      },
    ];

    render(
      <BacktestTab
        selectedIndicators={complexIndicators}
        entryConditions={complexEntryConditions}
        exitConditions={complexExitConditions}
        onBacktestRun={mockOnBacktestRun}
        onResultSave={mockOnResultSave}
      />
    );

    // バックテスト実行
    const runButton = screen.getByText('バックテスト実行');
    fireEvent.click(runButton);

    await waitFor(() => {
      expect(mockOnBacktestRun).toHaveBeenCalledWith(
        expect.objectContaining({
          strategy_config: expect.objectContaining({
            parameters: expect.objectContaining({
              strategy_gene: expect.objectContaining({
                // 有効な指標のみが含まれることを確認
                indicators: expect.arrayContaining([
                  expect.objectContaining({ type: 'SMA', enabled: true }),
                  expect.objectContaining({ type: 'EMA', enabled: true }),
                  expect.objectContaining({ type: 'RSI', enabled: true }),
                ]),
                // 複数のエントリー条件
                entry_conditions: expect.arrayContaining([
                  expect.objectContaining({
                    type: 'threshold',
                    indicator: 'RSI',
                    value: 30,
                  }),
                  expect.objectContaining({
                    type: 'crossover',
                    indicator1: 'EMA',
                    indicator2: 'SMA',
                  }),
                ]),
                // 複数のイグジット条件
                exit_conditions: expect.arrayContaining([
                  expect.objectContaining({
                    type: 'threshold',
                    indicator: 'RSI',
                    value: 70,
                  }),
                  expect.objectContaining({
                    type: 'crossover',
                    indicator1: 'SMA',
                    indicator2: 'EMA',
                  }),
                ]),
              }),
            }),
          }),
        })
      );
    });

    // 無効化された指標が含まれていないことを確認
    const callArgs = mockOnBacktestRun.mock.calls[0][0];
    const indicators = callArgs.strategy_config.parameters.strategy_gene.indicators;
    const indicatorTypes = indicators.map((ind: any) => ind.type);
    expect(indicatorTypes).not.toContain('MACD');
  });

  test('バリデーションエラーがある場合はAPI呼び出しが行われない', async () => {
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
    fireEvent.change(capitalInput, { target: { value: '-1000' } });

    // バックテスト実行を試行
    const runButton = screen.getByText('バックテスト実行');
    fireEvent.click(runButton);

    // エラーメッセージが表示されることを確認
    await waitFor(() => {
      expect(screen.getByText(/初期資金は正の値である必要があります/)).toBeInTheDocument();
    });

    // API呼び出しが行われないことを確認
    expect(mockOnBacktestRun).not.toHaveBeenCalled();
  });

  test('戦略が未設定の場合はバックテスト実行ボタンが無効化される', () => {
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
    const runButton = screen.getByText('バックテスト実行');
    expect(runButton).toBeDisabled();
  });

  test('StrategyGeneの構造が正しく生成される', async () => {
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
    const runButton = screen.getByText('バックテスト実行');
    fireEvent.click(runButton);

    await waitFor(() => {
      expect(mockOnBacktestRun).toHaveBeenCalled();
    });

    const callArgs = mockOnBacktestRun.mock.calls[0][0];
    const strategyGene = callArgs.strategy_config.parameters.strategy_gene;

    // StrategyGeneの必須フィールドが存在することを確認
    expect(strategyGene).toHaveProperty('id');
    expect(strategyGene).toHaveProperty('indicators');
    expect(strategyGene).toHaveProperty('entry_conditions');
    expect(strategyGene).toHaveProperty('exit_conditions');
    expect(strategyGene).toHaveProperty('risk_management');
    expect(strategyGene).toHaveProperty('metadata');

    // リスク管理設定のデフォルト値を確認
    expect(strategyGene.risk_management).toEqual({
      stop_loss_pct: 0.02,
      take_profit_pct: 0.05,
      position_sizing: 'fixed',
    });

    // メタデータの構造を確認
    expect(strategyGene.metadata).toEqual({
      created_by: 'strategy_builder',
      version: '1.0',
      created_at: expect.any(String),
    });
  });
});
