/**
 * 差分更新機能のテスト
 *
 * フロントエンドの差分更新ボタンとAPIとの連携をテストします。
 */

import React from "react";
import { render, screen, fireEvent, waitFor } from "@testing-library/react";
import "@testing-library/jest-dom";
import { jest } from "@jest/globals";

// テスト対象のコンポーネントをインポート
import DataPage from "../app/data/page";
import DataHeader from "../components/data/DataHeader";

// fetchのモック
global.fetch = jest.fn();

describe("差分更新機能のテスト", () => {
  beforeEach(() => {
    // 各テストの前にfetchモックをリセット
    (fetch as jest.MockedFunction<typeof fetch>).mockClear();
  });

  afterEach(() => {
    // テスト後のクリーンアップ
    jest.clearAllMocks();
  });

  describe("DataHeaderコンポーネント", () => {
    const defaultProps = {
      loading: false,
      error: "",
      updating: false,
      bulkUpdating: false,
      handleRefresh: jest.fn(),
      handleBulkIncrementalUpdate: jest.fn(),
    };

    test("一括差分更新ボタンが正しく表示される", () => {
      render(<DataHeader {...defaultProps} />);

      const bulkIncrementalButton = screen.getByText("一括差分更新");
      expect(bulkIncrementalButton).toBeInTheDocument();
      expect(bulkIncrementalButton).not.toBeDisabled();
    });

    test("bulkUpdating状態の時にボタンが無効化される", () => {
      render(<DataHeader {...defaultProps} bulkUpdating={true} />);

      const bulkIncrementalButton = screen.getByText("一括差分更新中...");
      expect(bulkIncrementalButton).toBeInTheDocument();
      expect(bulkIncrementalButton).toBeDisabled();
    });

    test("loading状態の時にボタンが無効化される", () => {
      render(<DataHeader {...defaultProps} loading={true} />);

      const bulkIncrementalButton = screen.getByText("一括差分更新");
      expect(bulkIncrementalButton).toBeDisabled();
    });

    test("一括差分更新ボタンクリック時にハンドラが呼び出される", () => {
      const mockHandler = jest.fn();
      render(
        <DataHeader
          {...defaultProps}
          handleBulkIncrementalUpdate={mockHandler}
        />
      );

      const bulkIncrementalButton = screen.getByText("一括差分更新");
      fireEvent.click(bulkIncrementalButton);

      expect(mockHandler).toHaveBeenCalledTimes(1);
    });
  });

  describe("差分更新API連携", () => {
    test("一括差分更新成功時の処理", async () => {
      // 一括差分更新の成功レスポンスのモック
      const mockBulkSuccessResponse = {
        success: true,
        message: "一括差分データ更新が完了しました",
        data: {
          success: true,
          total_saved_count: 15,
          data: {
            ohlcv: {
              symbol: "BTC/USDT:USDT",
              timeframe: "1h",
              saved_count: 10,
              success: true,
            },
            funding_rate: {
              symbol: "BTC/USDT:USDT",
              saved_count: 3,
              success: true,
            },
            open_interest: {
              symbol: "BTC/USDT:USDT",
              saved_count: 2,
              success: true,
            },
          },
        },
      };

      // fetchのモック設定
      (fetch as jest.MockedFunction<typeof fetch>)
        .mockResolvedValueOnce({
          ok: true,
          json: async () => mockBulkSuccessResponse,
        } as Response)
        .mockResolvedValue({
          ok: true,
          json: async () => ({ success: true, data: { ohlcv: [] } }),
        } as Response);

      // DataPageコンポーネントをレンダリング
      render(<DataPage />);

      // 一括差分更新ボタンを見つけてクリック
      await waitFor(() => {
        const bulkIncrementalButton = screen.getByText("一括差分更新");
        expect(bulkIncrementalButton).toBeInTheDocument();
      });

      const bulkIncrementalButton = screen.getByText("一括差分更新");
      fireEvent.click(bulkIncrementalButton);

      // 一括差分更新API呼び出しが正しく行われることを確認
      await waitFor(() => {
        const calls = (fetch as jest.MockedFunction<typeof fetch>).mock.calls;
        expect(calls[0][0]).toContain("/api/data/bulk-incremental-update");
      });

      // 成功メッセージが表示されることを確認
      await waitFor(() => {
        expect(screen.getByText(/一括差分更新完了/)).toBeInTheDocument();
        expect(screen.getByText(/総計15件/)).toBeInTheDocument();
        expect(screen.getByText(/OHLCV:10/)).toBeInTheDocument();
      });
    });

    test("差分更新失敗時の処理", async () => {
      // 失敗レスポンスのモック
      const mockErrorResponse = {
        success: false,
        message: "データベース接続エラー",
      };

      // fetchのモック設定
      (fetch as jest.MockedFunction<typeof fetch>).mockResolvedValueOnce({
        ok: true,
        json: async () => mockErrorResponse,
      } as Response);

      render(<DataPage />);

      // 一括差分更新ボタンをクリック
      await waitFor(() => {
        const bulkIncrementalButton = screen.getByText("一括差分更新");
        expect(bulkIncrementalButton).toBeInTheDocument();
      });

      const bulkIncrementalButton = screen.getByText("一括差分更新");
      fireEvent.click(bulkIncrementalButton);

      // エラーメッセージが表示されることを確認
      await waitFor(() => {
        const errorMessage = screen.getByText(/データベース接続エラー/);
        expect(errorMessage).toBeInTheDocument();
      });
    });

    test("ネットワークエラー時の処理", async () => {
      // ネットワークエラーのモック
      (fetch as jest.MockedFunction<typeof fetch>).mockRejectedValueOnce(
        new Error("Network Error")
      );

      render(<DataPage />);

      // 一括差分更新ボタンをクリック
      await waitFor(() => {
        const bulkIncrementalButton = screen.getByText("一括差分更新");
        expect(bulkIncrementalButton).toBeInTheDocument();
      });

      const bulkIncrementalButton = screen.getByText("一括差分更新");
      fireEvent.click(bulkIncrementalButton);

      // エラーメッセージが表示されることを確認
      await waitFor(() => {
        const errorMessage =
          screen.getByText(/差分更新中にエラーが発生しました/);
        expect(errorMessage).toBeInTheDocument();
      });
    });

    test("新しいデータがない場合の処理", async () => {
      // 新しいデータなしのレスポンス
      const mockNoDataResponse = {
        success: true,
        message: "新しいデータはありません",
        saved_count: 0,
      };

      (fetch as jest.MockedFunction<typeof fetch>)
        .mockResolvedValueOnce({
          ok: true,
          json: async () => mockNoDataResponse,
        } as Response)
        .mockResolvedValue({
          ok: true,
          json: async () => ({ success: true, data: { ohlcv: [] } }),
        } as Response);

      render(<DataPage />);

      await waitFor(() => {
        const bulkIncrementalButton = screen.getByText("一括差分更新");
        expect(bulkIncrementalButton).toBeInTheDocument();
      });

      const bulkIncrementalButton = screen.getByText("一括差分更新");
      fireEvent.click(bulkIncrementalButton);

      // 適切なメッセージが表示されることを確認
      await waitFor(() => {
        const message = screen.getByText(/一括差分更新完了！.*0件/);
        expect(message).toBeInTheDocument();
      });
    });
  });

  describe("メッセージ表示の確認", () => {
    test("成功メッセージが10秒後に自動的にクリアされる", async () => {
      jest.useFakeTimers();

      const mockSuccessResponse = {
        success: true,
        symbol: "BTC/USDT:USDT",
        timeframe: "1h",
        saved_count: 2,
      };

      (fetch as jest.MockedFunction<typeof fetch>).mockResolvedValue({
        ok: true,
        json: async () => mockSuccessResponse,
      } as Response);

      render(<DataPage />);

      await waitFor(() => {
        const bulkIncrementalButton = screen.getByText("一括差分更新");
        fireEvent.click(bulkIncrementalButton);
      });

      // 成功メッセージが表示されることを確認
      await waitFor(() => {
        const successMessage = screen.getByText(/一括差分更新完了！/);
        expect(successMessage).toBeInTheDocument();
      });

      // 15秒経過をシミュレート（一括差分更新は15秒でクリア）
      jest.advanceTimersByTime(15000);

      // メッセージがクリアされることを確認
      await waitFor(() => {
        const successMessage = screen.queryByText(/一括差分更新完了！/);
        expect(successMessage).not.toBeInTheDocument();
      });

      jest.useRealTimers();
    });
  });

  describe("一括差分更新API連携", () => {
    test("一括差分更新成功時の処理", async () => {
      // 成功レスポンスのモック
      const mockBulkSuccessResponse = {
        success: true,
        data: {
          success: true,
          total_saved_count: 15,
          data: {
            ohlcv: {
              symbol: "BTC/USDT:USDT",
              timeframe: "all",
              saved_count: 10,
              success: true,
              timeframe_results: {
                "15m": {
                  symbol: "BTC/USDT:USDT",
                  timeframe: "15m",
                  saved_count: 2,
                  success: true,
                },
                "30m": {
                  symbol: "BTC/USDT:USDT",
                  timeframe: "30m",
                  saved_count: 2,
                  success: true,
                },
                "1h": {
                  symbol: "BTC/USDT:USDT",
                  timeframe: "1h",
                  saved_count: 2,
                  success: true,
                },
                "4h": {
                  symbol: "BTC/USDT:USDT",
                  timeframe: "4h",
                  saved_count: 2,
                  success: true,
                },
                "1d": {
                  symbol: "BTC/USDT:USDT",
                  timeframe: "1d",
                  saved_count: 2,
                  success: true,
                },
              },
            },
            funding_rate: {
              symbol: "BTC/USDT:USDT",
              saved_count: 3,
              success: true,
            },
            open_interest: {
              symbol: "BTC/USDT:USDT",
              saved_count: 2,
              success: true,
            },
          },
        },
      };

      // fetchのモック設定
      (fetch as jest.MockedFunction<typeof fetch>)
        .mockResolvedValueOnce({
          ok: true,
          json: async () => mockBulkSuccessResponse,
        } as Response)
        .mockResolvedValue({
          ok: true,
          json: async () => ({ success: true, data: { ohlcv: [] } }),
        } as Response);

      // DataPageコンポーネントをレンダリング
      render(<DataPage />);

      // 一括差分更新ボタンを見つけてクリック
      await waitFor(() => {
        const bulkIncrementalButton = screen.getByText("一括差分更新");
        expect(bulkIncrementalButton).toBeInTheDocument();
      });

      const bulkIncrementalButton = screen.getByText("一括差分更新");
      fireEvent.click(bulkIncrementalButton);

      // 成功メッセージが表示されることを確認
      await waitFor(() => {
        expect(screen.getByText(/一括差分更新完了/)).toBeInTheDocument();
        expect(screen.getByText(/総計15件/)).toBeInTheDocument();
        expect(screen.getByText(/OHLCV:10/)).toBeInTheDocument();
      });

      // 一括差分更新API + OHLCVデータ再取得 + ステータス取得が呼び出されることを確認
      await waitFor(() => {
        expect(fetch).toHaveBeenCalledTimes(3); // 1 + 1 + 1
      });

      // 一括差分更新APIエンドポイントが呼び出されることを確認
      await waitFor(() => {
        const calls = (fetch as jest.MockedFunction<typeof fetch>).mock.calls;
        expect(calls[0][0]).toContain("/api/data/bulk-incremental-update");
      });
    });

    test("一括差分更新エラー時の処理", async () => {
      // エラーレスポンスのモック
      const mockErrorResponse = {
        success: false,
        message: "バックエンドAPIエラー: 500",
      };

      (fetch as jest.MockedFunction<typeof fetch>).mockResolvedValueOnce({
        ok: false,
        status: 500,
        json: async () => mockErrorResponse,
      } as Response);

      render(<DataPage />);

      await waitFor(() => {
        const bulkIncrementalButton = screen.getByText("一括差分更新");
        expect(bulkIncrementalButton).toBeInTheDocument();
      });

      const bulkIncrementalButton = screen.getByText("一括差分更新");
      fireEvent.click(bulkIncrementalButton);

      // エラーメッセージが表示されることを確認
      await waitFor(() => {
        expect(screen.getByText(/❌/)).toBeInTheDocument();
      });
    });
  });
});
