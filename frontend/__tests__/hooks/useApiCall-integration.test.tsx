/**
 * useApiCall統合テスト
 * 
 * 修正されたフックが正しく動作することを確認するテスト
 */

import { renderHook, act } from '@testing-library/react';
import { useOhlcvData } from '@/hooks/useOhlcvData';
import { useFundingRateData } from '@/hooks/useFundingRateData';
import { useOpenInterestData } from '@/hooks/useOpenInterestData';

// fetchのモック
global.fetch = jest.fn();

describe('useApiCall統合テスト', () => {
  beforeEach(() => {
    jest.clearAllMocks();
    (fetch as jest.Mock).mockResolvedValue({
      ok: true,
      text: () => Promise.resolve(JSON.stringify({
        success: true,
        data: { ohlcv: [], funding_rates: [], open_interest: [] }
      }))
    });
  });

  describe('useOhlcvData', () => {
    test('useApiCallを使用してデータを取得する', async () => {
      const { result } = renderHook(() => 
        useOhlcvData('BTC/USDT:USDT', '1h', 100)
      );

      // 初期状態の確認
      expect(result.current.loading).toBe(true);
      expect(result.current.data).toEqual([]);

      // API呼び出しが完了するまで待機
      await act(async () => {
        await new Promise(resolve => setTimeout(resolve, 100));
      });

      // fetchが正しいパラメータで呼び出されたことを確認
      expect(fetch).toHaveBeenCalledWith(
        '/api/data/candlesticks?symbol=BTC%2FUSDT%3AUSDT&timeframe=1h&limit=100',
        expect.objectContaining({
          method: 'GET',
          headers: { 'Content-Type': 'application/json' }
        })
      );
    });
  });

  describe('useFundingRateData', () => {
    test('useApiCallを使用してデータを取得する', async () => {
      const { result } = renderHook(() => 
        useFundingRateData('BTC/USDT:USDT', 100)
      );

      // 初期状態の確認
      expect(result.current.loading).toBe(true);
      expect(result.current.data).toEqual([]);

      // API呼び出しが完了するまで待機
      await act(async () => {
        await new Promise(resolve => setTimeout(resolve, 100));
      });

      // fetchが正しいパラメータで呼び出されたことを確認
      expect(fetch).toHaveBeenCalledWith(
        '/api/data/funding-rates?symbol=BTC%2FUSDT%3AUSDT&limit=100',
        expect.objectContaining({
          method: 'GET',
          headers: { 'Content-Type': 'application/json' }
        })
      );
    });
  });

  describe('useOpenInterestData', () => {
    test('useApiCallを使用してデータを取得する', async () => {
      const { result } = renderHook(() => 
        useOpenInterestData('BTC/USDT:USDT', 100)
      );

      // 初期状態の確認
      expect(result.current.loading).toBe(true);
      expect(result.current.data).toEqual([]);

      // API呼び出しが完了するまで待機
      await act(async () => {
        await new Promise(resolve => setTimeout(resolve, 100));
      });

      // fetchが正しいパラメータで呼び出されたことを確認
      expect(fetch).toHaveBeenCalledWith(
        '/api/data/open-interest?symbol=BTC%2FUSDT%3AUSDT&limit=100',
        expect.objectContaining({
          method: 'GET',
          headers: { 'Content-Type': 'application/json' }
        })
      );
    });
  });

  describe('エラーハンドリング', () => {
    test('APIエラー時に適切にエラーを処理する', async () => {
      (fetch as jest.Mock).mockResolvedValue({
        ok: false,
        status: 500,
        text: () => Promise.resolve(JSON.stringify({
          success: false,
          message: 'Internal Server Error'
        }))
      });

      const { result } = renderHook(() => 
        useOhlcvData('BTC/USDT:USDT', '1h', 100)
      );

      // API呼び出しが完了するまで待機
      await act(async () => {
        await new Promise(resolve => setTimeout(resolve, 100));
      });

      // エラー状態の確認
      expect(result.current.loading).toBe(false);
      expect(result.current.error).toBeTruthy();
    });

    test('ネットワークエラー時に適切にエラーを処理する', async () => {
      (fetch as jest.Mock).mockRejectedValue(new Error('Network Error'));

      const { result } = renderHook(() => 
        useOhlcvData('BTC/USDT:USDT', '1h', 100)
      );

      // API呼び出しが完了するまで待機
      await act(async () => {
        await new Promise(resolve => setTimeout(resolve, 100));
      });

      // エラー状態の確認
      expect(result.current.loading).toBe(false);
      expect(result.current.error).toBeTruthy();
    });
  });
});
