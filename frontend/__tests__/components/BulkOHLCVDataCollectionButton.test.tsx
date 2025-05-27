/**
 * OHLCVDataCollectionButton 統合コンポーネントのテスト
 *
 * 個別取得と一括取得の両方のモードをテスト
 */

import React from 'react';
import { render, screen, fireEvent, waitFor } from '@testing-library/react';
import '@testing-library/jest-dom';
import OHLCVDataCollectionButton from '@/components/BulkOHLCVDataCollectionButton';

// fetch をモック
global.fetch = jest.fn();

describe('OHLCVDataCollectionButton', () => {
  beforeEach(() => {
    jest.clearAllMocks();
  });

  describe('初期状態', () => {
    it('ボタンが正しくレンダリングされる', () => {
      render(<OHLCVDataCollectionButton />);

      expect(screen.getByRole('button')).toBeInTheDocument();
      expect(screen.getByText('全データ一括取得・保存')).toBeInTheDocument();
    });

    it('ボタンが有効状態である', () => {
      render(<OHLCVDataCollectionButton />);

      expect(screen.getByRole('button')).not.toBeDisabled();
    });

    it('ダークモード対応のクラスが適用されている', () => {
      render(<OHLCVDataCollectionButton />);

      const button = screen.getByRole('button');
      expect(button).toHaveClass('dark:bg-primary-600');
    });

    it('説明テキストが表示されている', () => {
      render(<OHLCVDataCollectionButton />);

      expect(screen.getByText(/全ての取引ペア/)).toBeInTheDocument();
      expect(screen.getByText(/全ての時間軸/)).toBeInTheDocument();
    });
  });

  describe('ローディング状態', () => {
    it('ボタンクリック時にローディング状態になる', async () => {
      // window.confirm をモック
      const mockConfirm = jest.spyOn(window, 'confirm').mockReturnValue(true);

      (fetch as jest.Mock).mockImplementation(() =>
        new Promise(resolve => setTimeout(() => resolve({
          ok: true,
          json: () => Promise.resolve({
            success: true,
            message: 'Bulk collection started',
            status: 'started',
            total_tasks: 84
          })
        }), 100))
      );

      render(<OHLCVDataCollectionButton />);

      const button = screen.getByRole('button');
      fireEvent.click(button);

      expect(button).toBeDisabled();
      expect(screen.getByText('一括取得・保存中...')).toBeInTheDocument();
      expect(screen.getByTestId('loading-spinner')).toBeInTheDocument();

      mockConfirm.mockRestore();
    });
  });

  describe('成功状態', () => {
    it('API成功時に成功メッセージが表示される', async () => {
      const mockConfirm = jest.spyOn(window, 'confirm').mockReturnValue(true);

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

      render(<OHLCVDataCollectionButton />);

      const button = screen.getByRole('button');
      fireEvent.click(button);

      await waitFor(() => {
        expect(screen.getByText('一括取得・保存開始')).toBeInTheDocument();
        expect(screen.getByTestId('success-icon')).toBeInTheDocument();
        expect(screen.getByText('84個のタスクを開始しました')).toBeInTheDocument();
      });

      mockConfirm.mockRestore();
    });
  });

  describe('エラー状態', () => {
    it('API失敗時にエラーメッセージが表示される', async () => {
      (fetch as jest.Mock).mockResolvedValueOnce({
        ok: false,
        json: () => Promise.resolve({
          success: false,
          message: '一括データ収集に失敗しました'
        })
      });

      render(<OHLCVDataCollectionButton />);

      const button = screen.getByRole('button');
      fireEvent.click(button);

      await waitFor(() => {
        expect(screen.getByText('エラーが発生しました')).toBeInTheDocument();
        expect(screen.getByTestId('error-icon')).toBeInTheDocument();
      });
    });

    it('ネットワークエラー時にエラーメッセージが表示される', async () => {
      (fetch as jest.Mock).mockRejectedValueOnce(new Error('Network error'));

      render(<OHLCVDataCollectionButton />);

      const button = screen.getByRole('button');
      fireEvent.click(button);

      await waitFor(() => {
        expect(screen.getByText('エラーが発生しました')).toBeInTheDocument();
      });
    });
  });

  describe('API呼び出し', () => {
    it('正しいエンドポイントでAPI呼び出しが行われる', async () => {
      (fetch as jest.Mock).mockResolvedValueOnce({
        ok: true,
        json: () => Promise.resolve({
          success: true,
          message: 'Success',
          status: 'started',
          total_tasks: 84
        })
      });

      render(<OHLCVDataCollectionButton />);

      const button = screen.getByRole('button');
      fireEvent.click(button);

      await waitFor(() => {
        expect(fetch).toHaveBeenCalledWith('/api/data/ohlcv/bulk', {
          method: 'POST',
          headers: {
            'Content-Type': 'application/json',
          }
        });
      });
    });
  });

  describe('コールバック', () => {
    it('成功時にonCollectionStartが呼ばれる', async () => {
      const mockOnStart = jest.fn();
      const mockResult = {
        success: true,
        message: 'Success',
        status: 'started' as const,
        total_tasks: 84,
        started_at: '2024-01-01T00:00:00Z'
      };

      (fetch as jest.Mock).mockResolvedValueOnce({
        ok: true,
        json: () => Promise.resolve(mockResult)
      });

      render(<OHLCVDataCollectionButton onBulkCollectionStart={mockOnStart} />);

      const button = screen.getByRole('button');
      fireEvent.click(button);

      await waitFor(() => {
        expect(mockOnStart).toHaveBeenCalledWith(mockResult);
      });
    });

    it('エラー時にonCollectionErrorが呼ばれる', async () => {
      const mockOnError = jest.fn();

      (fetch as jest.Mock).mockRejectedValueOnce(new Error('Test error'));

      render(<OHLCVDataCollectionButton onCollectionError={mockOnError} />);

      const button = screen.getByRole('button');
      fireEvent.click(button);

      await waitFor(() => {
        expect(mockOnError).toHaveBeenCalledWith('一括データ収集中にエラーが発生しました');
      });
    });
  });

  describe('確認ダイアログ', () => {
    it('ボタンクリック時に確認ダイアログが表示される', () => {
      // window.confirm をモック
      const mockConfirm = jest.spyOn(window, 'confirm').mockReturnValue(false);

      render(<OHLCVDataCollectionButton />);

      const button = screen.getByRole('button');
      fireEvent.click(button);

      expect(mockConfirm).toHaveBeenCalledWith(
        '全ての取引ペア（12個）と全ての時間軸（7個）でOHLCVデータを取得します。\n' +
        '合計84個のタスクが実行されます。\n\n' +
        'この処理には時間がかかる場合があります。続行しますか？'
      );

      mockConfirm.mockRestore();
    });

    it('確認ダイアログでキャンセルした場合はAPI呼び出しが行われない', () => {
      const mockConfirm = jest.spyOn(window, 'confirm').mockReturnValue(false);

      render(<OHLCVDataCollectionButton />);

      const button = screen.getByRole('button');
      fireEvent.click(button);

      expect(fetch).not.toHaveBeenCalled();

      mockConfirm.mockRestore();
    });
  });
});
