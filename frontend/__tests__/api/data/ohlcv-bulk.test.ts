/**
 * Bulk OHLCV API Route Handler のテスト
 * 
 * TDDアプローチ: 失敗するテストから開始
 */

import { NextRequest } from 'next/server';
import { POST } from '@/app/api/data/ohlcv/bulk/route';

// fetch をモック
global.fetch = jest.fn();

// NextRequest のモック
const createMockRequest = () => {
  return {
    json: async () => ({}),
    headers: new Map([['Content-Type', 'application/json']]),
  } as any;
};

describe('/api/data/ohlcv/bulk', () => {
  beforeEach(() => {
    jest.clearAllMocks();
  });

  describe('POST リクエスト', () => {
    it('正常なリクエストで成功レスポンスを返す', async () => {
      // バックエンドAPIのモックレスポンス
      (fetch as jest.Mock).mockResolvedValueOnce({
        ok: true,
        json: () => Promise.resolve({
          success: true,
          message: '全データの一括収集を開始しました',
          status: 'started',
          total_tasks: 84,
          started_at: '2024-01-01T00:00:00Z'
        })
      });

      const request = createMockRequest();
      const response = await POST(request);
      const data = await response.json();

      expect(response.status).toBe(200);
      expect(data.success).toBe(true);
      expect(data.total_tasks).toBe(84);
      expect(data.status).toBe('started');
    });

    it('バックエンドAPIエラー時に適切なエラーレスポンスを返す', async () => {
      (fetch as jest.Mock).mockResolvedValueOnce({
        ok: false,
        status: 500,
        json: () => Promise.resolve({
          detail: 'Internal server error'
        })
      });

      const request = createMockRequest();
      const response = await POST(request);
      const data = await response.json();

      expect(response.status).toBe(500);
      expect(data.success).toBe(false);
      expect(data.message).toContain('バックエンドAPI');
    });

    it('ネットワークエラー時に適切なエラーレスポンスを返す', async () => {
      (fetch as jest.Mock).mockRejectedValueOnce(new Error('Network error'));

      const request = createMockRequest();
      const response = await POST(request);
      const data = await response.json();

      expect(response.status).toBe(500);
      expect(data.success).toBe(false);
      expect(data.message).toContain('ネットワークエラー');
    });

    it('正しいバックエンドAPIエンドポイントを呼び出す', async () => {
      (fetch as jest.Mock).mockResolvedValueOnce({
        ok: true,
        json: () => Promise.resolve({
          success: true,
          message: 'Success',
          status: 'started',
          total_tasks: 84
        })
      });

      const request = createMockRequest();
      await POST(request);

      expect(fetch).toHaveBeenCalledWith(
        'http://127.0.0.1:8000/api/data-collection/bulk-historical',
        {
          method: 'POST',
          headers: {
            'Content-Type': 'application/json',
          }
        }
      );
    });

    it('処理中ステータスの適切なレスポンス', async () => {
      (fetch as jest.Mock).mockResolvedValueOnce({
        ok: true,
        json: () => Promise.resolve({
          success: true,
          message: '一括データ収集が進行中です',
          status: 'in_progress',
          total_tasks: 84,
          completed_tasks: 42,
          successful_tasks: 40,
          failed_tasks: 2
        })
      });

      const request = createMockRequest();
      const response = await POST(request);
      const data = await response.json();

      expect(response.status).toBe(200);
      expect(data.success).toBe(true);
      expect(data.status).toBe('in_progress');
      expect(data.completed_tasks).toBe(42);
      expect(data.successful_tasks).toBe(40);
      expect(data.failed_tasks).toBe(2);
    });

    it('完了ステータスの適切なレスポンス', async () => {
      (fetch as jest.Mock).mockResolvedValueOnce({
        ok: true,
        json: () => Promise.resolve({
          success: true,
          message: '全データの一括収集が完了しました',
          status: 'completed',
          total_tasks: 84,
          completed_tasks: 84,
          successful_tasks: 82,
          failed_tasks: 2,
          task_results: [
            {
              symbol: 'BTC/USDT',
              timeframe: '1h',
              success: true,
              message: 'データ収集完了',
              saved_count: 1000
            },
            {
              symbol: 'ETH/USDT',
              timeframe: '1h',
              success: false,
              message: 'データ収集失敗'
            }
          ]
        })
      });

      const request = createMockRequest();
      const response = await POST(request);
      const data = await response.json();

      expect(response.status).toBe(200);
      expect(data.success).toBe(true);
      expect(data.status).toBe('completed');
      expect(data.task_results).toHaveLength(2);
      expect(data.task_results[0].symbol).toBe('BTC/USDT');
      expect(data.task_results[0].success).toBe(true);
      expect(data.task_results[1].success).toBe(false);
    });
  });
});
