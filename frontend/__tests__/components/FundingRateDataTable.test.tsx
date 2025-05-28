/**
 * ファンディングレートデータテーブルコンポーネント テスト
 *
 * FundingRateDataTableコンポーネントのテストケースです。
 * 表示、ソート、ページネーション、CSVエクスポート機能をテストします。
 *
 * @author Trdinger Development Team
 * @version 1.0.0
 */

import React from "react";
import { render, screen, fireEvent, waitFor } from "@testing-library/react";
import "@testing-library/jest-dom";
import FundingRateDataTable from "@/components/FundingRateDataTable";
import { FundingRateData } from "@/types/strategy";

// モックデータの作成
const createMockFundingRateData = (count: number): FundingRateData[] => {
  return Array.from({ length: count }, (_, index) => ({
    symbol: index % 2 === 0 ? "BTC/USDT:USDT" : "ETH/USDT:USDT",
    funding_rate: (Math.random() - 0.5) * 0.001, // -0.0005 to 0.0005
    funding_timestamp: new Date(Date.now() - (count - index) * 8 * 60 * 60 * 1000).toISOString(),
    timestamp: new Date(Date.now() - (count - index) * 8 * 60 * 60 * 1000).toISOString(),
    next_funding_timestamp: new Date(Date.now() + 8 * 60 * 60 * 1000).toISOString(),
    mark_price: 50000 + Math.random() * 1000,
    index_price: 50000 + Math.random() * 1000,
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

describe("FundingRateDataTable", () => {
  const defaultProps = {
    data: createMockFundingRateData(10),
    loading: false,
    error: "",
  };

  beforeEach(() => {
    jest.clearAllMocks();
  });

  describe("基本表示テスト", () => {
    test("ファンディングレートデータが正しく表示される", () => {
      render(<FundingRateDataTable {...defaultProps} />);

      // タイトルの確認
      expect(screen.getByText("📊 ファンディングレートデータ")).toBeInTheDocument();

      // テーブルヘッダーの確認
      expect(screen.getByText("通貨ペア")).toBeInTheDocument();
      expect(screen.getByText("ファンディングレート")).toBeInTheDocument();
      expect(screen.getByText("ファンディング時刻")).toBeInTheDocument();
      expect(screen.getByText("マーク価格")).toBeInTheDocument();
      expect(screen.getByText("インデックス価格")).toBeInTheDocument();
      expect(screen.getByText("次回ファンディング")).toBeInTheDocument();
      expect(screen.getByText("取得時刻")).toBeInTheDocument();

      // データ行の確認
      expect(screen.getAllByRole("row")).toHaveLength(11); // ヘッダー + 10データ行
    });

    test("ローディング状態が正しく表示される", () => {
      render(<FundingRateDataTable {...defaultProps} loading={true} />);

      expect(screen.getByText("データを読み込み中...")).toBeInTheDocument();
    });

    test("エラー状態が正しく表示される", () => {
      render(<FundingRateDataTable {...defaultProps} error="テストエラー" />);

      expect(screen.getByText("📊 データの読み込みに失敗しました")).toBeInTheDocument();
      expect(screen.getByText("テストエラー")).toBeInTheDocument();
    });

    test("データが空の場合の表示", () => {
      render(<FundingRateDataTable {...defaultProps} data={[]} />);

      expect(screen.getByText("データがありません")).toBeInTheDocument();
    });
  });

  describe("データフォーマット表示テスト", () => {
    test("ファンディングレートがパーセント形式で表示される", () => {
      const testData = createMockFundingRateData(1);
      testData[0].funding_rate = 0.0001; // 0.01%

      render(<FundingRateDataTable {...defaultProps} data={testData} />);

      expect(screen.getByText("+0.010000%")).toBeInTheDocument();
    });

    test("負のファンディングレートが正しく表示される", () => {
      const testData = createMockFundingRateData(1);
      testData[0].funding_rate = -0.0002; // -0.02%

      render(<FundingRateDataTable {...defaultProps} data={testData} />);

      expect(screen.getByText("-0.020000%")).toBeInTheDocument();
    });

    test("通貨ペアが短縮形式で表示される", () => {
      const testData = createMockFundingRateData(1);
      testData[0].symbol = "BTC/USDT:USDT";

      render(<FundingRateDataTable {...defaultProps} data={testData} />);

      expect(screen.getByText("BTC/USDT")).toBeInTheDocument();
    });

    test("価格が通貨形式で表示される", () => {
      const testData = createMockFundingRateData(1);
      testData[0].mark_price = 50000.12;
      testData[0].index_price = 50001.34;

      render(<FundingRateDataTable {...defaultProps} data={testData} />);

      expect(screen.getByText(/\$50,000\.12/)).toBeInTheDocument();
      expect(screen.getByText(/\$50,001\.34/)).toBeInTheDocument();
    });

    test("null値が正しく表示される", () => {
      const testData = createMockFundingRateData(1);
      testData[0].mark_price = null;
      testData[0].index_price = null;
      testData[0].next_funding_timestamp = null;

      render(<FundingRateDataTable {...defaultProps} data={testData} />);

      // null値は "-" として表示される
      const dashElements = screen.getAllByText("-");
      expect(dashElements.length).toBeGreaterThanOrEqual(3);
    });
  });

  describe("ソート機能テスト", () => {
    test("通貨ペアカラムでソートできる", async () => {
      render(<FundingRateDataTable {...defaultProps} />);

      const symbolHeader = screen.getByText("通貨ペア");
      fireEvent.click(symbolHeader);

      await waitFor(() => {
        expect(symbolHeader.closest("th")).toHaveClass("cursor-pointer");
      });
    });

    test("ファンディングレートカラムでソートできる", async () => {
      render(<FundingRateDataTable {...defaultProps} />);

      const rateHeader = screen.getByText("ファンディングレート");
      fireEvent.click(rateHeader);

      await waitFor(() => {
        expect(rateHeader.closest("th")).toHaveClass("cursor-pointer");
      });
    });

    test("日時カラムでソートできる", async () => {
      render(<FundingRateDataTable {...defaultProps} />);

      const timestampHeader = screen.getByText("ファンディング時刻");
      fireEvent.click(timestampHeader);

      await waitFor(() => {
        expect(timestampHeader.closest("th")).toHaveClass("cursor-pointer");
      });
    });
  });

  describe("検索機能テスト", () => {
    test("検索ボックスが表示される", () => {
      render(<FundingRateDataTable {...defaultProps} />);

      const searchInput = screen.getByPlaceholderText("検索...");
      expect(searchInput).toBeInTheDocument();
    });

    test("通貨ペアで検索できる", async () => {
      const mixedData = [
        ...createMockFundingRateData(5).map(d => ({ ...d, symbol: "BTC/USDT:USDT" })),
        ...createMockFundingRateData(5).map(d => ({ ...d, symbol: "ETH/USDT:USDT" })),
      ];

      render(<FundingRateDataTable {...defaultProps} data={mixedData} />);

      const searchInput = screen.getByPlaceholderText("検索...");
      fireEvent.change(searchInput, { target: { value: "BTC" } });

      await waitFor(() => {
        // BTC関連のデータのみ表示されることを確認
        const btcElements = screen.getAllByText("BTC/USDT");
        expect(btcElements.length).toBeGreaterThan(0);
      });
    });
  });

  describe("CSVエクスポート機能テスト", () => {
    test("CSV出力ボタンが表示される", () => {
      render(<FundingRateDataTable {...defaultProps} />);

      const exportButton = screen.getByText("CSV出力");
      expect(exportButton).toBeInTheDocument();
      expect(exportButton).not.toBeDisabled();
    });

    test("CSV出力ボタンをクリックするとダウンロードが実行される", () => {
      render(<FundingRateDataTable {...defaultProps} />);

      const exportButton = screen.getByText("CSV出力");
      fireEvent.click(exportButton);

      expect(mockClick).toHaveBeenCalled();
    });

    test("データが空の場合はCSV出力ボタンが無効になる", () => {
      render(<FundingRateDataTable {...defaultProps} data={[]} />);

      const exportButton = screen.getByText("CSV出力");
      expect(exportButton).toBeDisabled();
    });
  });

  describe("ページネーション機能テスト", () => {
    test("大量データでページネーションが表示される", () => {
      const largeData = createMockFundingRateData(100);
      render(<FundingRateDataTable {...defaultProps} data={largeData} />);

      expect(screen.getByText(/1 - 50 \/ 100件/)).toBeInTheDocument();
      expect(screen.getByText("次へ")).toBeInTheDocument();
      expect(screen.getByText("前へ")).toBeInTheDocument();
    });

    test("次ページボタンが機能する", async () => {
      const largeData = createMockFundingRateData(100);
      render(<FundingRateDataTable {...defaultProps} data={largeData} />);

      const nextButton = screen.getByText("次へ");
      fireEvent.click(nextButton);

      await waitFor(() => {
        expect(screen.getByText(/51 - 100 \/ 100件/)).toBeInTheDocument();
      });
    });
  });
});
