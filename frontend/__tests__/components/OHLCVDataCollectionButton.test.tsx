/**
 * OHLCVDataCollectionButton コンポーネントのテスト
 * 
 * TDDアプローチ: 失敗するテストから開始
 */

import React from 'react';
import { render, screen, fireEvent, waitFor } from '@testing-library/react';
import '@testing-library/jest-dom';
import OHLCVDataCollectionButton from '@/components/OHLCVDataCollectionButton';

// fetch をモック
global.fetch = jest.fn();

describe('OHLCVDataCollectionButton', () => {
  beforeEach(() => {
    jest.clearAllMocks();
  });

  describe('初期状態', () => {
    it('ボタンが正しくレンダリングされる', () => {
      render(
        <OHLCVDataCollectionButton 
          selectedSymbol="BTC/USDT" 
          timeframe="1h" 
        />
      );
      
      expect(screen.getByRole('button')).toBeInTheDocument();
      expect(screen.getByText('OHLCVデータ取得・保存')).toBeInTheDocument();
    });

    it('ボタンが有効状態である', () => {
      render(
        <OHLCVDataCollectionButton 
          selectedSymbol="BTC/USDT" 
          timeframe="1h" 
        />
      );
      
      expect(screen.getByRole('button')).not.toBeDisabled();
    });

    it('ダークモード対応のクラスが適用されている', () => {
      render(
        <OHLCVDataCollectionButton 
          selectedSymbol="BTC/USDT" 
          timeframe="1h" 
        />
      );
      
      const button = screen.getByRole('button');
      expect(button).toHaveClass('dark:bg-primary-600');
    });
  });

  describe('ローディング状態', () => {
    it('ボタンクリック時にローディング状態になる', async () => {
      (fetch as jest.Mock).mockImplementation(() => 
        new Promise(resolve => setTimeout(() => resolve({
          ok: true,
          json: () => Promise.resolve({ success: true, message: 'Success' })
        }), 100))
      );

      render(
        <OHLCVDataCollectionButton 
          selectedSymbol="BTC/USDT" 
          timeframe="1h" 
        />
      );
      
      const button = screen.getByRole('button');
      fireEvent.click(button);
      
      expect(button).toBeDisabled();
      expect(screen.getByText('取得・保存中...')).toBeInTheDocument();
      expect(screen.getByTestId('loading-spinner')).toBeInTheDocument();
    });
  });

  describe('成功状態', () => {
    it('API成功時に成功メッセージが表示される', async () => {
      (fetch as jest.Mock).mockResolvedValueOnce({
        ok: true,
        json: () => Promise.resolve({
          success: true,
          message: 'データ収集が完了しました',
          saved_count: 100
        })
      });

      render(
        <OHLCVDataCollectionButton 
          selectedSymbol="BTC/USDT" 
          timeframe="1h" 
        />
      );
      
      const button = screen.getByRole('button');
      fireEvent.click(button);
      
      await waitFor(() => {
        expect(screen.getByText('取得・保存完了')).toBeInTheDocument();
        expect(screen.getByTestId('success-icon')).toBeInTheDocument();
      });
    });
  });

  describe('エラー状態', () => {
    it('API失敗時にエラーメッセージが表示される', async () => {
      (fetch as jest.Mock).mockResolvedValueOnce({
        ok: false,
        json: () => Promise.resolve({
          success: false,
          message: 'データ収集に失敗しました'
        })
      });

      render(
        <OHLCVDataCollectionButton 
          selectedSymbol="BTC/USDT" 
          timeframe="1h" 
        />
      );
      
      const button = screen.getByRole('button');
      fireEvent.click(button);
      
      await waitFor(() => {
        expect(screen.getByText('エラーが発生しました')).toBeInTheDocument();
        expect(screen.getByTestId('error-icon')).toBeInTheDocument();
      });
    });

    it('ネットワークエラー時にエラーメッセージが表示される', async () => {
      (fetch as jest.Mock).mockRejectedValueOnce(new Error('Network error'));

      render(
        <OHLCVDataCollectionButton 
          selectedSymbol="BTC/USDT" 
          timeframe="1h" 
        />
      );
      
      const button = screen.getByRole('button');
      fireEvent.click(button);
      
      await waitFor(() => {
        expect(screen.getByText('エラーが発生しました')).toBeInTheDocument();
      });
    });
  });

  describe('API呼び出し', () => {
    it('正しいパラメータでAPI呼び出しが行われる', async () => {
      (fetch as jest.Mock).mockResolvedValueOnce({
        ok: true,
        json: () => Promise.resolve({ success: true, message: 'Success' })
      });

      render(
        <OHLCVDataCollectionButton 
          selectedSymbol="ETH/USDT" 
          timeframe="4h" 
        />
      );
      
      const button = screen.getByRole('button');
      fireEvent.click(button);
      
      await waitFor(() => {
        expect(fetch).toHaveBeenCalledWith('/api/data/ohlcv', {
          method: 'POST',
          headers: {
            'Content-Type': 'application/json',
          },
          body: JSON.stringify({
            symbol: 'ETH/USDT',
            timeframe: '4h'
          })
        });
      });
    });
  });

  describe('コールバック', () => {
    it('成功時にonCollectionCompleteが呼ばれる', async () => {
      const mockOnComplete = jest.fn();
      const mockResult = { success: true, message: 'Success', saved_count: 50 };
      
      (fetch as jest.Mock).mockResolvedValueOnce({
        ok: true,
        json: () => Promise.resolve(mockResult)
      });

      render(
        <OHLCVDataCollectionButton 
          selectedSymbol="BTC/USDT" 
          timeframe="1h"
          onCollectionComplete={mockOnComplete}
        />
      );
      
      const button = screen.getByRole('button');
      fireEvent.click(button);
      
      await waitFor(() => {
        expect(mockOnComplete).toHaveBeenCalledWith(mockResult);
      });
    });

    it('エラー時にonCollectionErrorが呼ばれる', async () => {
      const mockOnError = jest.fn();
      
      (fetch as jest.Mock).mockRejectedValueOnce(new Error('Test error'));

      render(
        <OHLCVDataCollectionButton 
          selectedSymbol="BTC/USDT" 
          timeframe="1h"
          onCollectionError={mockOnError}
        />
      );
      
      const button = screen.getByRole('button');
      fireEvent.click(button);
      
      await waitFor(() => {
        expect(mockOnError).toHaveBeenCalledWith('データ収集中にエラーが発生しました');
      });
    });
  });
});
