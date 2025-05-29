/**
 * FundingRateCollectionButton コンポーネントのテスト
 *
 * FR収集ボタンの動作をテストします。
 */

import React from "react";
import { render, screen, fireEvent, waitFor } from "@testing-library/react";
import "@testing-library/jest-dom";
import FundingRateCollectionButton from "@/components/FundingRateCollectionButton";

// fetch をモック
global.fetch = jest.fn();

describe("FundingRateCollectionButton", () => {
  beforeEach(() => {
    jest.clearAllMocks();
  });

  describe("初期状態", () => {
    it("一括収集ボタンが正しくレンダリングされる", () => {
      render(<FundingRateCollectionButton mode="bulk" />);

      expect(screen.getByRole("button")).toBeInTheDocument();
      expect(screen.getByText("BTC・ETHFR収集・保存")).toBeInTheDocument();
    });

    it("単一収集ボタンが正しくレンダリングされる", () => {
      render(<FundingRateCollectionButton mode="single" symbol="BTC/USDT" />);

      expect(screen.getByRole("button")).toBeInTheDocument();
      expect(screen.getByText("FR収集・保存")).toBeInTheDocument();
    });

    it("ボタンが有効状態である", () => {
      render(<FundingRateCollectionButton />);

      expect(screen.getByRole("button")).not.toBeDisabled();
    });

    it("ダークモード対応のクラスが適用されている", () => {
      render(<FundingRateCollectionButton />);

      const button = screen.getByRole("button");
      expect(button).toHaveClass("dark:bg-primary-600");
    });

    it("説明テキストが表示されている", () => {
      render(<FundingRateCollectionButton mode="bulk" />);

      expect(screen.getByText(/BTCのFRデータ/)).toBeInTheDocument();
      expect(screen.getByText(/取得・保存します/)).toBeInTheDocument();
    });
  });

  describe("一括収集モード", () => {
    it("確認ダイアログが表示される", () => {
      // window.confirm をモック
      const mockConfirm = jest.spyOn(window, "confirm").mockReturnValue(false);

      render(<FundingRateCollectionButton mode="bulk" />);

      const button = screen.getByRole("button");
      fireEvent.click(button);

      expect(mockConfirm).toHaveBeenCalledWith(
        expect.stringContaining("BTCのFRデータを取得します")
      );

      mockConfirm.mockRestore();
    });

    it("確認ダイアログでキャンセルした場合はAPI呼び出しが行われない", () => {
      const mockConfirm = jest.spyOn(window, "confirm").mockReturnValue(false);

      render(<FundingRateCollectionButton mode="bulk" />);

      const button = screen.getByRole("button");
      fireEvent.click(button);

      expect(fetch).not.toHaveBeenCalled();

      mockConfirm.mockRestore();
    });

    it("API成功時に成功メッセージが表示される", async () => {
      const mockConfirm = jest.spyOn(window, "confirm").mockReturnValue(true);

      (fetch as jest.Mock).mockResolvedValueOnce({
        ok: true,
        json: () =>
          Promise.resolve({
            success: true,
            data: {
              success: true,
              message: "FR一括収集完了",
              total_symbols: 10,
              successful_symbols: 8,
              failed_symbols: 2,
              total_saved_records: 800,
              results: [],
              failures: [],
            },
          }),
      });

      render(<FundingRateCollectionButton mode="bulk" />);

      const button = screen.getByRole("button");
      fireEvent.click(button);

      await waitFor(() => {
        expect(screen.getByText("FR一括収集完了")).toBeInTheDocument();
        expect(screen.getByTestId("success-icon")).toBeInTheDocument();
      });

      mockConfirm.mockRestore();
    });
  });

  describe("単一収集モード", () => {
    it("確認ダイアログなしでAPI呼び出しが行われる", async () => {
      (fetch as jest.Mock).mockResolvedValueOnce({
        ok: true,
        json: () =>
          Promise.resolve({
            success: true,
            data: {
              symbol: "BTC/USDT",
              fetched_count: 100,
              saved_count: 95,
              success: true,
            },
          }),
      });

      render(<FundingRateCollectionButton mode="single" symbol="BTC/USDT" />);

      const button = screen.getByRole("button");
      fireEvent.click(button);

      await waitFor(() => {
        expect(fetch).toHaveBeenCalledWith(
          "/api/data/funding-rates/collect?symbol=BTC%2FUSDT&fetch_all=true",
          expect.objectContaining({
            method: "POST",
            headers: {
              "Content-Type": "application/json",
            },
          })
        );
      });
    });
  });

  describe("ローディング状態", () => {
    it("ボタンクリック時にローディング状態になる", async () => {
      const mockConfirm = jest.spyOn(window, "confirm").mockReturnValue(true);

      (fetch as jest.Mock).mockImplementation(
        () =>
          new Promise((resolve) =>
            setTimeout(
              () =>
                resolve({
                  ok: true,
                  json: () =>
                    Promise.resolve({
                      success: true,
                      data: {
                        success: true,
                        message: "FR収集完了",
                        total_symbols: 10,
                        successful_symbols: 10,
                        failed_symbols: 0,
                        total_saved_records: 1000,
                        results: [],
                        failures: [],
                      },
                    }),
                }),
              100
            )
          )
      );

      render(<FundingRateCollectionButton mode="bulk" />);

      const button = screen.getByRole("button");
      fireEvent.click(button);

      expect(button).toBeDisabled();
      expect(screen.getByText("FR一括収集中...")).toBeInTheDocument();
      expect(screen.getByTestId("loading-spinner")).toBeInTheDocument();

      mockConfirm.mockRestore();
    });
  });

  describe("エラー状態", () => {
    it("API失敗時にエラーメッセージが表示される", async () => {
      const mockConfirm = jest.spyOn(window, "confirm").mockReturnValue(true);

      (fetch as jest.Mock).mockResolvedValueOnce({
        ok: false,
        status: 500,
        statusText: "Internal Server Error",
        json: () =>
          Promise.resolve({
            success: false,
            message: "サーバーエラーが発生しました",
          }),
      });

      const mockOnError = jest.fn();

      render(
        <FundingRateCollectionButton
          mode="bulk"
          onCollectionError={mockOnError}
        />
      );

      const button = screen.getByRole("button");
      fireEvent.click(button);

      await waitFor(() => {
        expect(screen.getByText("エラーが発生しました")).toBeInTheDocument();
        expect(screen.getByTestId("error-icon")).toBeInTheDocument();
        expect(mockOnError).toHaveBeenCalled();
      });

      mockConfirm.mockRestore();
    });

    it("ネットワークエラー時にエラーメッセージが表示される", async () => {
      const mockConfirm = jest.spyOn(window, "confirm").mockReturnValue(true);

      (fetch as jest.Mock).mockRejectedValueOnce(new Error("Network error"));

      const mockOnError = jest.fn();

      render(
        <FundingRateCollectionButton
          mode="bulk"
          onCollectionError={mockOnError}
        />
      );

      const button = screen.getByRole("button");
      fireEvent.click(button);

      await waitFor(() => {
        expect(screen.getByText("エラーが発生しました")).toBeInTheDocument();
        expect(mockOnError).toHaveBeenCalledWith("Network error");
      });

      mockConfirm.mockRestore();
    });
  });

  describe("無効化状態", () => {
    it("disabled=trueの場合ボタンが無効化される", () => {
      render(<FundingRateCollectionButton disabled={true} />);

      const button = screen.getByRole("button");
      expect(button).toBeDisabled();
    });

    it("無効化されたボタンはクリックできない", () => {
      render(<FundingRateCollectionButton disabled={true} />);

      const button = screen.getByRole("button");
      fireEvent.click(button);

      expect(fetch).not.toHaveBeenCalled();
    });
  });

  describe("コールバック", () => {
    it("収集開始時にコールバックが呼ばれる", async () => {
      const mockConfirm = jest.spyOn(window, "confirm").mockReturnValue(true);
      const mockOnStart = jest.fn();

      (fetch as jest.Mock).mockResolvedValueOnce({
        ok: true,
        json: () =>
          Promise.resolve({
            success: true,
            data: {
              success: true,
              message: "FR収集完了",
              total_symbols: 10,
              successful_symbols: 10,
              failed_symbols: 0,
              total_saved_records: 1000,
              results: [],
              failures: [],
            },
          }),
      });

      render(
        <FundingRateCollectionButton
          mode="bulk"
          onCollectionStart={mockOnStart}
        />
      );

      const button = screen.getByRole("button");
      fireEvent.click(button);

      await waitFor(() => {
        expect(mockOnStart).toHaveBeenCalledWith(
          expect.objectContaining({
            success: true,
            total_symbols: 10,
            successful_symbols: 10,
          })
        );
      });

      mockConfirm.mockRestore();
    });
  });
});
