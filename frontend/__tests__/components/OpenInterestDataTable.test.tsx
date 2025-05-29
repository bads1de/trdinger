/**
 * オープンインタレストデータテーブルコンポーネント テスト
 *
 * OpenInterestDataTableコンポーネントのテストケースです。
 * 表示、ソート、ページネーション、CSVエクスポート機能をテストします。
 *
 * @author Trdinger Development Team
 * @version 1.0.0
 */

import React from "react";
import { render, screen, fireEvent, waitFor } from "@testing-library/react";
import "@testing-library/jest-dom";
import OpenInterestDataTable from "@/components/OpenInterestDataTable";
import { OpenInterestData } from "@/types/strategy";

// モックデータ
const mockOpenInterestData: OpenInterestData[] = [
  {
    symbol: "BTC/USDT:USDT",
    open_interest_value: 15000000000,
    open_interest_amount: 500000,
    data_timestamp: "2024-01-15T12:00:00Z",
    timestamp: "2024-01-15T12:01:00Z",
  },
  {
    symbol: "ETH/USDT:USDT",
    open_interest_value: 8000000000,
    open_interest_amount: 3000000,
    data_timestamp: "2024-01-15T12:00:00Z",
    timestamp: "2024-01-15T12:01:00Z",
  },
  {
    symbol: "BTC/USDT:USDT",
    open_interest_value: 14800000000,
    open_interest_amount: 495000,
    data_timestamp: "2024-01-15T11:00:00Z",
    timestamp: "2024-01-15T11:01:00Z",
  },
  {
    symbol: "ETH/USDT:USDT",
    open_interest_value: 7900000000,
    open_interest_amount: 2950000,
    data_timestamp: "2024-01-15T11:00:00Z",
    timestamp: "2024-01-15T11:01:00Z",
  },
  {
    symbol: "BTC/USDT:USDT",
    open_interest_value: 15200000000,
    open_interest_amount: 505000,
    data_timestamp: "2024-01-15T10:00:00Z",
    timestamp: "2024-01-15T10:01:00Z",
  },
  {
    symbol: "ETH/USDT:USDT",
    open_interest_value: 8100000000,
    open_interest_amount: 3050000,
    data_timestamp: "2024-01-15T10:00:00Z",
    timestamp: "2024-01-15T10:01:00Z",
  },
  {
    symbol: "BTC/USDT:USDT",
    open_interest_value: 14900000000,
    open_interest_amount: 498000,
    data_timestamp: "2024-01-15T09:00:00Z",
    timestamp: "2024-01-15T09:01:00Z",
  },
  {
    symbol: "ETH/USDT:USDT",
    open_interest_value: 7950000000,
    open_interest_amount: 2975000,
    data_timestamp: "2024-01-15T09:00:00Z",
    timestamp: "2024-01-15T09:01:00Z",
  },
  {
    symbol: "BTC/USDT:USDT",
    open_interest_value: 15100000000,
    open_interest_amount: 502000,
    data_timestamp: "2024-01-15T08:00:00Z",
    timestamp: "2024-01-15T08:01:00Z",
  },
  {
    symbol: "ETH/USDT:USDT",
    open_interest_value: 8050000000,
    open_interest_amount: 3025000,
    data_timestamp: "2024-01-15T08:00:00Z",
    timestamp: "2024-01-15T08:01:00Z",
  },
];

// デフォルトプロパティ
const defaultProps = {
  data: mockOpenInterestData,
  loading: false,
  error: "",
};

describe("OpenInterestDataTable", () => {
  describe("基本表示テスト", () => {
    test("オープンインタレストデータが正しく表示される", () => {
      render(<OpenInterestDataTable {...defaultProps} />);

      // タイトルの確認
      expect(screen.getByText("📈 オープンインタレストデータ")).toBeInTheDocument();

      // テーブルヘッダーの確認
      expect(screen.getByText("通貨ペア")).toBeInTheDocument();
      expect(screen.getByText("OI値 (USD)")).toBeInTheDocument();
      expect(screen.getByText("OI量")).toBeInTheDocument();
      expect(screen.getByText("データ時刻")).toBeInTheDocument();
      expect(screen.getByText("取得時刻")).toBeInTheDocument();

      // データ行の確認
      expect(screen.getAllByRole("row")).toHaveLength(11); // ヘッダー + 10データ行
    });

    test("通貨ペアシンボルが正しくフォーマットされる", () => {
      render(<OpenInterestDataTable {...defaultProps} />);

      // "BTC/USDT:USDT" -> "BTC/USDT" にフォーマットされることを確認
      expect(screen.getAllByText("BTC/USDT")).toHaveLength(5);
      expect(screen.getAllByText("ETH/USDT")).toHaveLength(5);
    });

    test("オープンインタレスト値が通貨形式で表示される", () => {
      render(<OpenInterestDataTable {...defaultProps} />);

      // 通貨形式での表示を確認
      expect(screen.getByText("$15,000,000,000")).toBeInTheDocument();
      expect(screen.getByText("$8,000,000,000")).toBeInTheDocument();
    });

    test("オープンインタレスト量がコンパクト形式で表示される", () => {
      render(<OpenInterestDataTable {...defaultProps} />);

      // コンパクト形式での表示を確認
      expect(screen.getByText("500K")).toBeInTheDocument();
      expect(screen.getByText("3M")).toBeInTheDocument();
    });
  });

  describe("状態表示テスト", () => {
    test("ローディング状態が正しく表示される", () => {
      render(<OpenInterestDataTable {...defaultProps} loading={true} />);

      // ローディング表示の確認
      expect(screen.getByText("データを読み込み中...")).toBeInTheDocument();
    });

    test("エラー状態が正しく表示される", () => {
      const errorMessage = "データの取得に失敗しました";
      render(<OpenInterestDataTable {...defaultProps} error={errorMessage} />);

      // エラーメッセージの確認
      expect(screen.getByText(errorMessage)).toBeInTheDocument();
    });

    test("データが空の場合の表示", () => {
      render(<OpenInterestDataTable {...defaultProps} data={[]} />);

      // 空データメッセージの確認
      expect(screen.getByText("データがありません")).toBeInTheDocument();
    });
  });

  describe("ソート機能テスト", () => {
    test("通貨ペアでソートできる", async () => {
      render(<OpenInterestDataTable {...defaultProps} />);

      const symbolHeader = screen.getByText("通貨ペア");
      fireEvent.click(symbolHeader);

      await waitFor(() => {
        const rows = screen.getAllByRole("row");
        // ヘッダーを除く最初のデータ行を確認
        expect(rows[1]).toHaveTextContent("BTC/USDT");
      });
    });

    test("OI値でソートできる", async () => {
      render(<OpenInterestDataTable {...defaultProps} />);

      const valueHeader = screen.getByText("OI値 (USD)");
      fireEvent.click(valueHeader);

      await waitFor(() => {
        // ソート後の順序を確認
        const rows = screen.getAllByRole("row");
        expect(rows[1]).toHaveTextContent("$15,200,000,000");
      });
    });
  });

  describe("検索機能テスト", () => {
    test("通貨ペアで検索できる", async () => {
      render(<OpenInterestDataTable {...defaultProps} />);

      const searchInput = screen.getByPlaceholderText("検索...");
      fireEvent.change(searchInput, { target: { value: "BTC" } });

      await waitFor(() => {
        const rows = screen.getAllByRole("row");
        // ヘッダー + BTCのデータ行のみ表示されることを確認
        expect(rows).toHaveLength(6); // ヘッダー + 5行のBTCデータ
      });
    });
  });

  describe("エクスポート機能テスト", () => {
    test("CSVエクスポートボタンが表示される", () => {
      render(<OpenInterestDataTable {...defaultProps} />);

      const exportButton = screen.getByText("CSV出力");
      expect(exportButton).toBeInTheDocument();
    });

    test("CSVエクスポートボタンをクリックできる", () => {
      render(<OpenInterestDataTable {...defaultProps} />);

      const exportButton = screen.getByText("CSV出力");
      fireEvent.click(exportButton);

      // エクスポート処理が実行されることを確認（実際のファイルダウンロードはテストしない）
      expect(exportButton).toBeInTheDocument();
    });
  });

  describe("ページネーション機能テスト", () => {
    test("ページネーションが正しく動作する", async () => {
      // 大量のデータでテスト
      const largeData = Array.from({ length: 100 }, (_, i) => ({
        symbol: `TEST${i}/USDT:USDT`,
        open_interest_value: 1000000000 + i * 1000000,
        open_interest_amount: 100000 + i * 1000,
        data_timestamp: `2024-01-15T${String(i % 24).padStart(2, "0")}:00:00Z`,
        timestamp: `2024-01-15T${String(i % 24).padStart(2, "0")}:01:00Z`,
      }));

      render(<OpenInterestDataTable {...defaultProps} data={largeData} />);

      // 最初のページに50件表示されることを確認
      const rows = screen.getAllByRole("row");
      expect(rows).toHaveLength(51); // ヘッダー + 50データ行

      // ページネーションボタンの確認
      expect(screen.getByText("次へ")).toBeInTheDocument();
    });
  });
});
