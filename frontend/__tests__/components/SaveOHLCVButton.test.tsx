/**
 * SaveOHLCVButtonコンポーネントのテスト
 * 
 * TDDアプローチでOHLCVデータ保存ボタンの機能をテストします。
 */

import React from 'react';
import { render, screen, fireEvent, waitFor } from '@testing-library/react';
import '@testing-library/jest-dom';
import SaveOHLCVButton from '@/components/SaveOHLCVButton';

// fetch APIのモック
global.fetch = jest.fn();

describe('SaveOHLCVButton', () => {
  const defaultProps = {
    symbol: 'BTC/USD',
    timeframe: '1h' as const,
    limit: 100,
  };

  beforeEach(() => {
    // 各テスト前にfetchモックをリセット
    (fetch as jest.Mock).mockClear();
  });

  afterEach(() => {
    jest.clearAllMocks();
  });

  test('ボタンが正しく表示される', () => {
    render(<SaveOHLCVButton {...defaultProps} />);
    
    const button = screen.getByRole('button', { name: /OHLCVデータ保存/i });
    expect(button).toBeInTheDocument();
    expect(button).not.toBeDisabled();
  });

  test('ボタンに正しいアイコンとテキストが表示される', () => {
    render(<SaveOHLCVButton {...defaultProps} />);
    
    // アイコンの確認（SVGまたはアイコンクラス）
    const button = screen.getByRole('button', { name: /OHLCVデータ保存/i });
    expect(button).toHaveTextContent('OHLCVデータ保存');
    
    // 保存アイコンが含まれていることを確認
    const icon = button.querySelector('svg');
    expect(icon).toBeInTheDocument();
  });

  test('保存対象の情報が表示される', () => {
    render(<SaveOHLCVButton {...defaultProps} />);
    
    // 銘柄と時間軸の情報が表示されることを確認
    expect(screen.getByText(/BTC\/USD/)).toBeInTheDocument();
    expect(screen.getByText(/1h/)).toBeInTheDocument();
  });

  test('ボタンクリック時に保存APIが呼ばれる', async () => {
    const mockResponse = {
      success: true,
      records_saved: 5,
      symbol: 'BTC/USD',
      timeframe: '1h',
      message: 'データを5件保存しました',
    };

    (fetch as jest.Mock).mockResolvedValueOnce({
      ok: true,
      json: async () => mockResponse,
    });

    render(<SaveOHLCVButton {...defaultProps} />);
    
    const button = screen.getByRole('button', { name: /OHLCVデータ保存/i });
    fireEvent.click(button);

    // APIが正しいパラメータで呼ばれることを確認
    await waitFor(() => {
      expect(fetch).toHaveBeenCalledWith('/api/v1/market-data/save-ohlcv', {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
        },
        body: JSON.stringify({
          symbol: 'BTC/USD',
          timeframe: '1h',
          limit: 100,
        }),
      });
    });
  });

  test('保存中はローディング状態を表示する', async () => {
    // 遅延するPromiseを作成
    let resolvePromise: (value: any) => void;
    const delayedPromise = new Promise((resolve) => {
      resolvePromise = resolve;
    });

    (fetch as jest.Mock).mockReturnValueOnce(delayedPromise);

    render(<SaveOHLCVButton {...defaultProps} />);
    
    const button = screen.getByRole('button', { name: /OHLCVデータ保存/i });
    fireEvent.click(button);

    // ローディング状態の確認
    await waitFor(() => {
      expect(button).toBeDisabled();
      expect(screen.getByText(/保存中/i)).toBeInTheDocument();
    });

    // ローディングアニメーションの確認
    const loadingSpinner = screen.getByTestId('loading-spinner');
    expect(loadingSpinner).toBeInTheDocument();

    // Promiseを解決してローディング状態を終了
    resolvePromise!({
      ok: true,
      json: async () => ({ success: true, records_saved: 3 }),
    });

    await waitFor(() => {
      expect(button).not.toBeDisabled();
      expect(screen.queryByText(/保存中/i)).not.toBeInTheDocument();
    });
  });

  test('保存成功時に成功メッセージを表示する', async () => {
    const mockResponse = {
      success: true,
      records_saved: 3,
      symbol: 'BTC/USD',
      timeframe: '1h',
      message: 'BTC/USD の 1h OHLCVデータを 3件保存しました',
    };

    (fetch as jest.Mock).mockResolvedValueOnce({
      ok: true,
      json: async () => mockResponse,
    });

    render(<SaveOHLCVButton {...defaultProps} />);
    
    const button = screen.getByRole('button', { name: /OHLCVデータ保存/i });
    fireEvent.click(button);

    // 成功メッセージの確認
    await waitFor(() => {
      expect(screen.getByText(/3件保存しました/i)).toBeInTheDocument();
    });

    // 成功アイコンの確認
    const successIcon = screen.getByTestId('success-icon');
    expect(successIcon).toBeInTheDocument();
  });

  test('重複データの場合に適切なメッセージを表示する', async () => {
    const mockResponse = {
      success: true,
      records_saved: 0,
      symbol: 'BTC/USD',
      timeframe: '1h',
      message: 'BTC/USD の 1h データは既に存在するため、新規保存はありませんでした',
    };

    (fetch as jest.Mock).mockResolvedValueOnce({
      ok: true,
      json: async () => mockResponse,
    });

    render(<SaveOHLCVButton {...defaultProps} />);
    
    const button = screen.getByRole('button', { name: /OHLCVデータ保存/i });
    fireEvent.click(button);

    // 重複メッセージの確認
    await waitFor(() => {
      expect(screen.getByText(/既に存在/i)).toBeInTheDocument();
    });

    // 情報アイコンの確認
    const infoIcon = screen.getByTestId('info-icon');
    expect(infoIcon).toBeInTheDocument();
  });

  test('保存失敗時にエラーメッセージを表示する', async () => {
    const mockErrorResponse = {
      success: false,
      message: '無効なシンボルです: INVALID',
    };

    (fetch as jest.Mock).mockResolvedValueOnce({
      ok: false,
      status: 400,
      json: async () => ({ detail: mockErrorResponse }),
    });

    render(<SaveOHLCVButton {...defaultProps} symbol="INVALID" />);
    
    const button = screen.getByRole('button', { name: /OHLCVデータ保存/i });
    fireEvent.click(button);

    // エラーメッセージの確認
    await waitFor(() => {
      expect(screen.getByText(/無効なシンボル/i)).toBeInTheDocument();
    });

    // エラーアイコンの確認
    const errorIcon = screen.getByTestId('error-icon');
    expect(errorIcon).toBeInTheDocument();
  });

  test('ネットワークエラー時に適切なエラーメッセージを表示する', async () => {
    (fetch as jest.Mock).mockRejectedValueOnce(new Error('Network error'));

    render(<SaveOHLCVButton {...defaultProps} />);
    
    const button = screen.getByRole('button', { name: /OHLCVデータ保存/i });
    fireEvent.click(button);

    // ネットワークエラーメッセージの確認
    await waitFor(() => {
      expect(screen.getByText(/ネットワークエラー/i)).toBeInTheDocument();
    });
  });

  test('メッセージが一定時間後に自動で消える', async () => {
    jest.useFakeTimers();

    const mockResponse = {
      success: true,
      records_saved: 2,
      message: 'データを保存しました',
    };

    (fetch as jest.Mock).mockResolvedValueOnce({
      ok: true,
      json: async () => mockResponse,
    });

    render(<SaveOHLCVButton {...defaultProps} />);
    
    const button = screen.getByRole('button', { name: /OHLCVデータ保存/i });
    fireEvent.click(button);

    // メッセージが表示されることを確認
    await waitFor(() => {
      expect(screen.getByText(/データを保存しました/i)).toBeInTheDocument();
    });

    // 5秒後にメッセージが消えることを確認
    jest.advanceTimersByTime(5000);

    await waitFor(() => {
      expect(screen.queryByText(/データを保存しました/i)).not.toBeInTheDocument();
    });

    jest.useRealTimers();
  });

  test('disabled状態の時はクリックできない', () => {
    render(<SaveOHLCVButton {...defaultProps} disabled={true} />);
    
    const button = screen.getByRole('button', { name: /OHLCVデータ保存/i });
    expect(button).toBeDisabled();

    fireEvent.click(button);

    // APIが呼ばれないことを確認
    expect(fetch).not.toHaveBeenCalled();
  });

  test('カスタムクラス名が適用される', () => {
    const customClassName = 'custom-save-button';
    render(<SaveOHLCVButton {...defaultProps} className={customClassName} />);
    
    const button = screen.getByRole('button', { name: /OHLCVデータ保存/i });
    expect(button).toHaveClass(customClassName);
  });
});
