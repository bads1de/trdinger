/**
 * OHLCVデータテーブルコンポーネント テスト
 *
 * OHLCVDataTableコンポーネントのテストケースです。
 * 表示、ソート、ページネーション、CSVエクスポート機能をテストします。
 *
 */

import React from "react";
import { render, screen, fireEvent, waitFor } from "@testing-library/react";
import "@testing-library/jest-dom";
import OHLCVDataTable from "@/components/table/OHLCVDataTable";
import { PriceData } from "@/types/market-data";

// モックデータの作成
const createMockOHLCVData = (count: number): PriceData[] => {
  return Array.from({ length: count }, (_, index) => ({
    timestamp: new Date(
      Date.now() - (count - index) * 24 * 60 * 60 * 1000
    ).toISOString(),
    open: 50000 + Math.random() * 1000,
    high: 51000 + Math.random() * 1000,
    low: 49000 + Math.random() * 1000,
    close: 50500 + Math.random() * 1000,
    volume: 1000 + Math.random() * 500,
  }));
};

// URL.createObjectURLのモック
Object.defineProperty(global, "URL", {
  value: {
    createObjectURL: jest.fn(() => "mock-url"),
    revokeObjectURL: jest.fn(),
  },
});

// document.createElementのモック
const mockClick = jest.fn();
Object.defineProperty(document, "createElement", {
  value: jest.fn(() => ({
    setAttribute: jest.fn(),
    click: mockClick,
    style: {},
  })),
});

Object.defineProperty(document.body, "appendChild", {
  value: jest.fn(),
});

Object.defineProperty(document.body, "removeChild", {
  value: jest.fn(),
});

describe("OHLCVDataTable", () => {
  const defaultProps = {
    data: createMockOHLCVData(10),
    symbol: "BTC/USDT",
    timeframe: "1d",
    loading: false,
    error: "",
  };

  beforeEach(() => {
    jest.clearAllMocks();
  });

  describe("基本表示テスト", () => {
    test("OHLCVデータが正しく表示される", () => {
      render(<OHLCVDataTable {...defaultProps} />);

      // タイトルの確認
      expect(
        screen.getByText("📊 BTC/USDT - 1d足 OHLCVデータ")
      ).toBeInTheDocument();

      // テーブルヘッダーの確認
      expect(screen.getByText("日時")).toBeInTheDocument();
      expect(screen.getByText("始値")).toBeInTheDocument();
      expect(screen.getByText("高値")).toBeInTheDocument();
      expect(screen.getByText("安値")).toBeInTheDocument();
      expect(screen.getByText("終値")).toBeInTheDocument();
      expect(screen.getByText("出来高")).toBeInTheDocument();
      expect(screen.getByText("変動率")).toBeInTheDocument();

      // データ行の確認
      expect(screen.getAllByRole("row")).toHaveLength(11); // ヘッダー + 10データ行
    });

    test("ローディング状態が正しく表示される", () => {
      render(<OHLCVDataTable {...defaultProps} loading={true} />);

      expect(screen.getByText("データを読み込み中...")).toBeInTheDocument();
    });

    test("エラー状態が正しく表示される", () => {
      render(<OHLCVDataTable {...defaultProps} error="テストエラー" />);

      expect(
        screen.getByText("📊 データの読み込みに失敗しました")
      ).toBeInTheDocument();
      expect(screen.getByText("テストエラー")).toBeInTheDocument();
    });

    test("データが空の場合の表示", () => {
      render(<OHLCVDataTable {...defaultProps} data={[]} />);

      expect(screen.getByText("データがありません")).toBeInTheDocument();
    });
  });

  describe("ソート機能テスト", () => {
    test("日時カラムでソートできる", async () => {
      render(<OHLCVDataTable {...defaultProps} />);

      const timestampHeader = screen.getByText("日時");
      fireEvent.click(timestampHeader);

      // ソートアイコンが表示されることを確認
      await waitFor(() => {
        expect(timestampHeader.closest("th")).toHaveClass("cursor-pointer");
      });
    });

    test("価格カラムでソートできる", async () => {
      render(<OHLCVDataTable {...defaultProps} />);

      const openHeader = screen.getByText("始値");
      fireEvent.click(openHeader);

      await waitFor(() => {
        expect(openHeader.closest("th")).toHaveClass("cursor-pointer");
      });
    });
  });

  describe("CSVエクスポート機能テスト", () => {
    test("CSV出力ボタンが表示される", () => {
      render(<OHLCVDataTable {...defaultProps} />);

      const exportButton = screen.getByText("CSV出力");
      expect(exportButton).toBeInTheDocument();
      expect(exportButton).not.toBeDisabled();
    });

    test("CSV出力ボタンをクリックするとダウンロードが実行される", () => {
      render(<OHLCVDataTable {...defaultProps} />);

      const exportButton = screen.getByText("CSV出力");
      fireEvent.click(exportButton);

      expect(mockClick).toHaveBeenCalled();
    });

    test("データが空の場合はCSV出力ボタンが無効になる", () => {
      render(<OHLCVDataTable {...defaultProps} data={[]} />);

      const exportButton = screen.getByText("CSV出力");
      expect(exportButton).toBeDisabled();
    });

    test("ローディング中はCSV出力ボタンが無効になる", () => {
      render(<OHLCVDataTable {...defaultProps} loading={true} />);

      const exportButton = screen.getByText("CSV出力");
      expect(exportButton).toBeDisabled();
    });
  });

  describe("データフォーマット表示テスト", () => {
    test("価格が通貨形式で表示される", () => {
      const testData = createMockOHLCVData(1);
      testData[0].open = 50000.12;
      testData[0].close = 50500.34;

      render(<OHLCVDataTable {...defaultProps} data={testData} />);

      // 通貨形式（$記号付き）で表示されることを確認
      expect(screen.getByText(/\$50,000\.12/)).toBeInTheDocument();
      expect(screen.getByText(/\$50,500\.34/)).toBeInTheDocument();
    });

    test("出来高が読みやすい形式で表示される", () => {
      const testData = createMockOHLCVData(1);
      testData[0].volume = 1500000; // 1.5M

      render(<OHLCVDataTable {...defaultProps} data={testData} />);

      expect(screen.getByText("1.50M")).toBeInTheDocument();
    });

    test("変動率が正しく計算・表示される", () => {
      const testData = createMockOHLCVData(1);
      testData[0].open = 50000;
      testData[0].close = 51000; // +2%

      render(<OHLCVDataTable {...defaultProps} data={testData} />);

      expect(screen.getByText("+2.00%")).toBeInTheDocument();
    });
  });

  describe("ページネーション機能テスト", () => {
    test("大量データでページネーションが表示される", () => {
      const largeData = createMockOHLCVData(100);
      render(<OHLCVDataTable {...defaultProps} data={largeData} />);

      // ページネーション情報の確認
      expect(screen.getByText(/1 - 50 \/ 100件/)).toBeInTheDocument();
      expect(screen.getByText("次へ")).toBeInTheDocument();
      expect(screen.getByText("前へ")).toBeInTheDocument();
    });

    test("次ページボタンが機能する", async () => {
      const largeData = createMockOHLCVData(100);
      render(<OHLCVDataTable {...defaultProps} data={largeData} />);

      const nextButton = screen.getByText("次へ");
      fireEvent.click(nextButton);

      await waitFor(() => {
        expect(screen.getByText(/51 - 100 \/ 100件/)).toBeInTheDocument();
      });
    });
  });

  describe("レスポンシブ表示テスト", () => {
    test("テーブルが横スクロール可能", () => {
      render(<OHLCVDataTable {...defaultProps} />);

      const tableContainer = screen.getByRole("table").closest("div");
      expect(tableContainer).toHaveClass("overflow-x-auto");
    });
  });
});
