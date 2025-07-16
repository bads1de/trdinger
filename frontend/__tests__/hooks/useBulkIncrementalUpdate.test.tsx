/**
 * useBulkIncrementalUpdateフックのテスト
 */

import { renderHook, act } from '@testing-library/react';
import '@testing-library/jest-dom';
import { jest } from '@jest/globals';

import { useBulkIncrementalUpdate } from '../../hooks/useBulkIncrementalUpdate';

// fetchのモック
global.fetch = jest.fn();

describe('useBulkIncrementalUpdate', () => {
  beforeEach(() => {
    (fetch as jest.MockedFunction<typeof fetch>).mockClear();
  });

  afterEach(() => {
    jest.clearAllMocks();
  });

  test('初期状態が正しく設定される', () => {
    const { result } = renderHook(() => useBulkIncrementalUpdate());

    expect(result.current.loading).toBe(false);
    expect(result.current.error).toBe(null);
    expect(typeof result.current.bulkUpdate).toBe('function');
    expect(typeof result.current.reset).toBe('function');
  });

  test('一括差分更新が成功する', async () => {
    const mockResponse = {
      success: true,
      data: {
        success: true,
        total_saved_count: 15,
        data: {
          ohlcv: {
            symbol: 'BTC/USDT:USDT',
            timeframe: 'all',
            saved_count: 10,
            success: true,
            timeframe_results: {
              '15m': { symbol: 'BTC/USDT:USDT', timeframe: '15m', saved_count: 2, success: true },
              '30m': { symbol: 'BTC/USDT:USDT', timeframe: '30m', saved_count: 2, success: true },
              '1h': { symbol: 'BTC/USDT:USDT', timeframe: '1h', saved_count: 2, success: true },
              '4h': { symbol: 'BTC/USDT:USDT', timeframe: '4h', saved_count: 2, success: true },
              '1d': { symbol: 'BTC/USDT:USDT', timeframe: '1d', saved_count: 2, success: true },
            }
          },
          funding_rate: {
            symbol: 'BTC/USDT:USDT',
            saved_count: 3,
            success: true
          },
          open_interest: {
            symbol: 'BTC/USDT:USDT',
            saved_count: 2,
            success: true
          }
        }
      },
      message: '一括差分データ更新が完了しました',
      timestamp: new Date().toISOString(),
    };

    (fetch as jest.MockedFunction<typeof fetch>).mockResolvedValueOnce({
      ok: true,
      json: async () => mockResponse,
    } as Response);

    const onSuccess = jest.fn();
    const onError = jest.fn();

    const { result } = renderHook(() => useBulkIncrementalUpdate());

    await act(async () => {
      await result.current.bulkUpdate('BTC/USDT:USDT', '1h', {
        onSuccess,
        onError,
      });
    });

    expect(fetch).toHaveBeenCalledWith(
      '/api/data/bulk-incremental-update?symbol=BTC%2FUSDT%3AUSDT&timeframe=1h',
      expect.objectContaining({
        method: 'POST',
      })
    );

    expect(onSuccess).toHaveBeenCalledWith(mockResponse);
    expect(onError).not.toHaveBeenCalled();
  });

  test('一括差分更新がエラーになる', async () => {
    const mockErrorResponse = {
      success: false,
      message: 'バックエンドAPIエラー',
    };

    (fetch as jest.MockedFunction<typeof fetch>).mockResolvedValueOnce({
      ok: false,
      status: 500,
      json: async () => mockErrorResponse,
    } as Response);

    const onSuccess = jest.fn();
    const onError = jest.fn();

    const { result } = renderHook(() => useBulkIncrementalUpdate());

    await act(async () => {
      await result.current.bulkUpdate('BTC/USDT:USDT', '1h', {
        onSuccess,
        onError,
      });
    });

    expect(onSuccess).not.toHaveBeenCalled();
    expect(onError).toHaveBeenCalled();
  });

  test('ネットワークエラーが発生する', async () => {
    (fetch as jest.MockedFunction<typeof fetch>).mockRejectedValueOnce(
      new Error('Network error')
    );

    const onSuccess = jest.fn();
    const onError = jest.fn();

    const { result } = renderHook(() => useBulkIncrementalUpdate());

    await act(async () => {
      await result.current.bulkUpdate('BTC/USDT:USDT', '1h', {
        onSuccess,
        onError,
      });
    });

    expect(onSuccess).not.toHaveBeenCalled();
    expect(onError).toHaveBeenCalled();
  });

  test('resetが正しく動作する', async () => {
    const { result } = renderHook(() => useBulkIncrementalUpdate());

    act(() => {
      result.current.reset();
    });

    expect(result.current.loading).toBe(false);
    expect(result.current.error).toBe(null);
  });

  test('URLエンコーディングが正しく行われる', async () => {
    const mockResponse = {
      success: true,
      data: { total_saved_count: 0 },
      message: 'テスト',
      timestamp: new Date().toISOString(),
    };

    (fetch as jest.MockedFunction<typeof fetch>).mockResolvedValueOnce({
      ok: true,
      json: async () => mockResponse,
    } as Response);

    const { result } = renderHook(() => useBulkIncrementalUpdate());

    await act(async () => {
      await result.current.bulkUpdate('ETH/USDT:USDT', '4h');
    });

    expect(fetch).toHaveBeenCalledWith(
      '/api/data/bulk-incremental-update?symbol=ETH%2FUSDT%3AUSDT&timeframe=4h',
      expect.objectContaining({
        method: 'POST',
      })
    );
  });
});
