/**
 * useStrategyBuilderフック バックテスト機能拡張のテスト
 * 
 * TDDアプローチでuseStrategyBuilderフックのバックテスト機能をテストします。
 */

import { renderHook, act } from '@testing-library/react';
import { useStrategyBuilder } from '@/hooks/useStrategyBuilder';

// APIモック
global.fetch = jest.fn();

describe('useStrategyBuilder バックテスト機能', () => {
  beforeEach(() => {
    jest.clearAllMocks();
    (fetch as jest.Mock).mockResolvedValue({
      ok: true,
      json: async () => ({ success: true, data: {} }),
    });
  });

  test('初期状態でバックテスト関連の状態が正しく設定される', () => {
    const { result } = renderHook(() => useStrategyBuilder());

    // バックテスト関連の初期状態を確認
    expect(result.current.backtestConfig).toBeNull();
    expect(result.current.backtestResult).toBeNull();
    expect(result.current.backtestLoading).toBe(false);
    expect(result.current.backtestError).toBeNull();
  });

  test('updateBacktestConfigでバックテスト設定を更新できる', () => {
    const { result } = renderHook(() => useStrategyBuilder());

    const testConfig = {
      strategy_name: 'TEST_STRATEGY',
      symbol: 'BTC/USDT',
      timeframe: '1h',
      start_date: '2024-01-01',
      end_date: '2024-12-31',
      initial_capital: 100000,
      commission_rate: 0.00055,
      strategy_config: {
        strategy_type: 'USER_CUSTOM',
        parameters: {
          strategy_gene: {
            indicators: [],
            entry_conditions: [],
            exit_conditions: [],
          },
        },
      },
    };

    act(() => {
      result.current.updateBacktestConfig(testConfig);
    });

    expect(result.current.backtestConfig).toEqual(testConfig);
  });

  test('runBacktestでバックテストを実行できる', async () => {
    const { result } = renderHook(() => useStrategyBuilder());

    const mockBacktestResult = {
      id: 'test-result-1',
      strategy_name: 'TEST_STRATEGY',
      symbol: 'BTC/USDT',
      performance_metrics: {
        total_return: 15.5,
        sharpe_ratio: 1.2,
        max_drawdown: -8.3,
      },
    };

    (fetch as jest.Mock).mockResolvedValueOnce({
      ok: true,
      json: async () => ({ success: true, result: mockBacktestResult }),
    });

    const testConfig = {
      strategy_name: 'TEST_STRATEGY',
      symbol: 'BTC/USDT',
      timeframe: '1h',
      start_date: '2024-01-01',
      end_date: '2024-12-31',
      initial_capital: 100000,
      commission_rate: 0.00055,
      strategy_config: {
        strategy_type: 'USER_CUSTOM',
        parameters: {
          strategy_gene: {
            indicators: [],
            entry_conditions: [],
            exit_conditions: [],
          },
        },
      },
    };

    await act(async () => {
      await result.current.runBacktest(testConfig);
    });

    // API呼び出しが正しく行われることを確認
    expect(fetch).toHaveBeenCalledWith('/api/backtest/run', {
      method: 'POST',
      headers: {
        'Content-Type': 'application/json',
      },
      body: JSON.stringify(testConfig),
    });

    // 結果が正しく設定されることを確認
    expect(result.current.backtestResult).toEqual(mockBacktestResult);
    expect(result.current.backtestLoading).toBe(false);
    expect(result.current.backtestError).toBeNull();
  });

  test('runBacktest実行中はローディング状態になる', async () => {
    const { result } = renderHook(() => useStrategyBuilder());

    let resolvePromise: (value: any) => void;
    const mockPromise = new Promise((resolve) => {
      resolvePromise = resolve;
    });

    (fetch as jest.Mock).mockReturnValueOnce(mockPromise);

    const testConfig = {
      strategy_name: 'TEST_STRATEGY',
      symbol: 'BTC/USDT',
      timeframe: '1h',
      start_date: '2024-01-01',
      end_date: '2024-12-31',
      initial_capital: 100000,
      commission_rate: 0.00055,
      strategy_config: {
        strategy_type: 'USER_CUSTOM',
        parameters: {
          strategy_gene: {
            indicators: [],
            entry_conditions: [],
            exit_conditions: [],
          },
        },
      },
    };

    // バックテスト実行開始
    act(() => {
      result.current.runBacktest(testConfig);
    });

    // ローディング状態を確認
    expect(result.current.backtestLoading).toBe(true);

    // プロミスを解決
    act(() => {
      resolvePromise!({
        ok: true,
        json: async () => ({ success: true, result: {} }),
      });
    });

    // ローディング状態が解除されることを確認
    await act(async () => {
      await new Promise(resolve => setTimeout(resolve, 0));
    });

    expect(result.current.backtestLoading).toBe(false);
  });

  test('runBacktestでエラーが発生した場合の処理', async () => {
    const { result } = renderHook(() => useStrategyBuilder());

    (fetch as jest.Mock).mockRejectedValueOnce(new Error('Network error'));

    const testConfig = {
      strategy_name: 'TEST_STRATEGY',
      symbol: 'BTC/USDT',
      timeframe: '1h',
      start_date: '2024-01-01',
      end_date: '2024-12-31',
      initial_capital: 100000,
      commission_rate: 0.00055,
      strategy_config: {
        strategy_type: 'USER_CUSTOM',
        parameters: {
          strategy_gene: {
            indicators: [],
            entry_conditions: [],
            exit_conditions: [],
          },
        },
      },
    };

    await act(async () => {
      await result.current.runBacktest(testConfig);
    });

    // エラー状態が正しく設定されることを確認
    expect(result.current.backtestError).toBe('バックテスト実行中にエラーが発生しました');
    expect(result.current.backtestLoading).toBe(false);
    expect(result.current.backtestResult).toBeNull();
  });

  test('saveBacktestResultで結果を保存できる', async () => {
    const { result } = renderHook(() => useStrategyBuilder());

    const mockResult = {
      id: 'test-result-1',
      strategy_name: 'TEST_STRATEGY',
      performance_metrics: {
        total_return: 15.5,
      },
    };

    (fetch as jest.Mock).mockResolvedValueOnce({
      ok: true,
      json: async () => ({ success: true }),
    });

    await act(async () => {
      await result.current.saveBacktestResult(mockResult);
    });

    // API呼び出しが正しく行われることを確認
    expect(fetch).toHaveBeenCalledWith('/api/backtest/results', {
      method: 'POST',
      headers: {
        'Content-Type': 'application/json',
      },
      body: JSON.stringify(mockResult),
    });
  });

  test('clearBacktestResultでバックテスト結果をクリアできる', () => {
    const { result } = renderHook(() => useStrategyBuilder());

    // 初期状態を設定
    act(() => {
      result.current.updateBacktestConfig({
        strategy_name: 'TEST',
        symbol: 'BTC/USDT',
        timeframe: '1h',
        start_date: '2024-01-01',
        end_date: '2024-12-31',
        initial_capital: 100000,
        commission_rate: 0.00055,
        strategy_config: {
          strategy_type: 'USER_CUSTOM',
          parameters: { strategy_gene: {} },
        },
      });
    });

    // 結果をクリア
    act(() => {
      result.current.clearBacktestResult();
    });

    // 状態がクリアされることを確認
    expect(result.current.backtestConfig).toBeNull();
    expect(result.current.backtestResult).toBeNull();
    expect(result.current.backtestError).toBeNull();
  });

  test('バックテストステップへの進行可否判定が正しく動作する', () => {
    const { result } = renderHook(() => useStrategyBuilder());

    // 初期状態では進行不可
    expect(result.current.canProceedToStep('backtest')).toBe(false);

    // 指標と条件、戦略名を設定
    act(() => {
      result.current.updateSelectedIndicators([
        { name: 'SMA', params: { period: 20 }, enabled: true },
      ]);
      result.current.updateEntryConditions([
        { type: 'indicator_comparison', indicator1: 'SMA', operator: '>', value: 100 },
      ]);
      result.current.updateStrategyName('Test Strategy');
    });

    // 条件が満たされれば進行可能
    expect(result.current.canProceedToStep('backtest')).toBe(true);
  });
});
