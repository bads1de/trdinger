/**
 * useApiCall POST統合テスト
 * 
 * POSTメソッドでuseApiCallを使用するコンポーネントが正しく動作することを確認するテスト
 */

import { renderHook, act } from '@testing-library/react';
import { useApiCall } from '@/hooks/useApiCall';

// fetchのモック
global.fetch = jest.fn();

describe('useApiCall POST統合テスト', () => {
  beforeEach(() => {
    jest.clearAllMocks();
    (fetch as jest.Mock).mockResolvedValue({
      ok: true,
      text: () => Promise.resolve(JSON.stringify({
        success: true,
        message: "操作が完了しました"
      }))
    });
  });

  describe('MLトレーニング開始', () => {
    test('POSTリクエストでMLトレーニングを開始する', async () => {
      const { result } = renderHook(() => useApiCall());

      const config = {
        symbol: "BTC/USDT:USDT",
        timeframe: "1h",
        start_date: "2020-03-05",
        end_date: "2024-12-31",
        save_model: true,
        train_test_split: 0.8,
        random_state: 42,
        use_profile: false,
      };

      await act(async () => {
        await result.current.execute("/api/ml/training/start", {
          method: "POST",
          body: config,
        });
      });

      // fetchが正しいパラメータで呼び出されたことを確認
      expect(fetch).toHaveBeenCalledWith(
        '/api/ml/training/start',
        expect.objectContaining({
          method: 'POST',
          headers: { 'Content-Type': 'application/json' },
          body: JSON.stringify(config)
        })
      );
    });
  });

  describe('MLトレーニング停止', () => {
    test('POSTリクエストでMLトレーニングを停止する', async () => {
      const { result } = renderHook(() => useApiCall());

      await act(async () => {
        await result.current.execute("/api/ml/training/stop", {
          method: "POST",
        });
      });

      // fetchが正しいパラメータで呼び出されたことを確認
      expect(fetch).toHaveBeenCalledWith(
        '/api/ml/training/stop',
        expect.objectContaining({
          method: 'POST',
          headers: { 'Content-Type': 'application/json' }
        })
      );
    });
  });

  describe('ベイジアン最適化', () => {
    test('POSTリクエストでベイジアン最適化を実行する', async () => {
      const { result } = renderHook(() => useApiCall());

      const config = {
        optimization_type: "ml",
        model_type: "lightgbm",
        n_calls: 50,
        optimization_config: {
          acq_func: "EI",
          n_initial_points: 10,
          random_state: 42,
        },
        save_as_profile: false,
      };

      await act(async () => {
        await result.current.execute("/api/bayesian-optimization/ml-hyperparameters", {
          method: "POST",
          body: config,
        });
      });

      // fetchが正しいパラメータで呼び出されたことを確認
      expect(fetch).toHaveBeenCalledWith(
        '/api/bayesian-optimization/ml-hyperparameters',
        expect.objectContaining({
          method: 'POST',
          headers: { 'Content-Type': 'application/json' },
          body: JSON.stringify(config)
        })
      );
    });
  });

  describe('エラーハンドリング', () => {
    test('POSTリクエストでAPIエラー時に適切にエラーを処理する', async () => {
      (fetch as jest.Mock).mockResolvedValue({
        ok: false,
        status: 400,
        text: () => Promise.resolve(JSON.stringify({
          success: false,
          message: 'Bad Request'
        }))
      });

      const { result } = renderHook(() => useApiCall());

      await act(async () => {
        await result.current.execute("/api/ml/training/start", {
          method: "POST",
          body: { invalid: "data" },
        });
      });

      // エラー状態の確認
      expect(result.current.loading).toBe(false);
      expect(result.current.error).toBeTruthy();
    });

    test('POSTリクエストでネットワークエラー時に適切にエラーを処理する', async () => {
      (fetch as jest.Mock).mockRejectedValue(new Error('Network Error'));

      const { result } = renderHook(() => useApiCall());

      await act(async () => {
        await result.current.execute("/api/ml/training/start", {
          method: "POST",
          body: {},
        });
      });

      // エラー状態の確認
      expect(result.current.loading).toBe(false);
      expect(result.current.error).toBeTruthy();
    });
  });

  describe('成功時コールバック', () => {
    test('成功時にonSuccessコールバックが呼び出される', async () => {
      const onSuccess = jest.fn();
      const onError = jest.fn();
      
      const { result } = renderHook(() => useApiCall());

      await act(async () => {
        await result.current.execute("/api/ml/training/start", {
          method: "POST",
          body: {},
          onSuccess,
          onError,
        });
      });

      expect(onSuccess).toHaveBeenCalledWith({
        success: true,
        message: "操作が完了しました"
      });
      expect(onError).not.toHaveBeenCalled();
    });
  });
});
