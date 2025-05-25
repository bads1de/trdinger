/**
 * ローソク足チャートコンポーネント テスト
 *
 * CandlestickChart コンポーネントのテストケースです。
 * レンダリング、プロパティ、状態管理をテストします。
 *
 * @author Trdinger Development Team
 * @version 1.0.0
 */

import React from "react";
import { render, screen } from "@testing-library/react";
import "@testing-library/jest-dom";
import CandlestickChart from "@/components/CandlestickChart";
import { CandlestickData } from "@/types/strategy";

// モックデータの作成
const createMockCandlestickData = (count: number = 5): CandlestickData[] => {
  const data: CandlestickData[] = [];
  const basePrice = 45000;
  const now = new Date();

  for (let i = 0; i < count; i++) {
    const timestamp = new Date(
      now.getTime() - (count - 1 - i) * 24 * 60 * 60 * 1000
    );
    const open = basePrice + (Math.random() - 0.5) * 1000;
    const close = open + (Math.random() - 0.5) * 500;
    const high = Math.max(open, close) + Math.random() * 200;
    const low = Math.min(open, close) - Math.random() * 200;
    const volume = Math.random() * 1000000 + 100000;

    data.push({
      timestamp: timestamp.toISOString(),
      open: Number(open.toFixed(2)),
      high: Number(high.toFixed(2)),
      low: Number(low.toFixed(2)),
      close: Number(close.toFixed(2)),
      volume: Number(volume.toFixed(0)),
    });
  }

  return data;
};

// Rechartsのモック
jest.mock("recharts", () => ({
  ResponsiveContainer: ({ children }: { children: React.ReactNode }) => (
    <div data-testid="responsive-container">{children}</div>
  ),
  LineChart: ({ children }: { children: React.ReactNode }) => (
    <div data-testid="line-chart">{children}</div>
  ),
  Line: () => <div data-testid="line" />,
  XAxis: () => <div data-testid="x-axis" />,
  YAxis: () => <div data-testid="y-axis" />,
  CartesianGrid: () => <div data-testid="cartesian-grid" />,
  Tooltip: () => <div data-testid="tooltip" />,
}));

describe("CandlestickChart", () => {
  describe("正常系テスト", () => {
    test("有効なデータでチャートが正常にレンダリングされる", () => {
      const mockData = createMockCandlestickData(10);

      render(<CandlestickChart data={mockData} />);

      expect(screen.getByTestId("responsive-container")).toBeInTheDocument();
      expect(screen.getByTestId("line-chart")).toBeInTheDocument();
      expect(screen.getAllByTestId("line")).toHaveLength(3); // close, high, low
      expect(screen.getByTestId("x-axis")).toBeInTheDocument();
      expect(screen.getByTestId("y-axis")).toBeInTheDocument();
      expect(screen.getByTestId("cartesian-grid")).toBeInTheDocument();
      expect(screen.getByTestId("tooltip")).toBeInTheDocument();
    });

    test("カスタムの高さが適用される", () => {
      const mockData = createMockCandlestickData(5);
      const customHeight = 600;

      const { container } = render(
        <CandlestickChart data={mockData} height={customHeight} />
      );

      const responsiveContainer = screen.getByTestId("responsive-container");
      expect(responsiveContainer).toBeInTheDocument();
    });

    test("空のデータ配列でも正常にレンダリングされる", () => {
      render(<CandlestickChart data={[]} />);

      expect(
        screen.getByText("表示するデータがありません")
      ).toBeInTheDocument();
    });
  });

  describe("ローディング状態テスト", () => {
    test("ローディング状態が正しく表示される", () => {
      const mockData = createMockCandlestickData(5);

      render(<CandlestickChart data={mockData} loading={true} />);

      expect(
        screen.getByText("チャートデータを読み込み中...")
      ).toBeInTheDocument();
      expect(screen.queryByTestId("line-chart")).not.toBeInTheDocument();
    });

    test("ローディング状態でスピナーが表示される", () => {
      const mockData = createMockCandlestickData(5);

      const { container } = render(
        <CandlestickChart data={mockData} loading={true} />
      );

      const spinner = container.querySelector(".animate-spin");
      expect(spinner).toBeInTheDocument();
    });
  });

  describe("エラー状態テスト", () => {
    test("エラーメッセージが正しく表示される", () => {
      const mockData = createMockCandlestickData(5);
      const errorMessage = "データの取得に失敗しました";

      render(<CandlestickChart data={mockData} error={errorMessage} />);

      expect(
        screen.getByText("チャートの読み込みに失敗しました")
      ).toBeInTheDocument();
      expect(screen.getByText(errorMessage)).toBeInTheDocument();
      expect(screen.queryByTestId("line-chart")).not.toBeInTheDocument();
    });

    test("エラー状態でチャートが表示されない", () => {
      const mockData = createMockCandlestickData(5);

      render(<CandlestickChart data={mockData} error="エラーが発生しました" />);

      expect(
        screen.queryByTestId("responsive-container")
      ).not.toBeInTheDocument();
      expect(screen.queryByTestId("line-chart")).not.toBeInTheDocument();
    });
  });

  describe("データ検証テスト", () => {
    test("undefinedデータで空状態が表示される", () => {
      render(<CandlestickChart data={undefined as any} />);

      expect(
        screen.getByText("表示するデータがありません")
      ).toBeInTheDocument();
    });

    test("nullデータで空状態が表示される", () => {
      render(<CandlestickChart data={null as any} />);

      expect(
        screen.getByText("表示するデータがありません")
      ).toBeInTheDocument();
    });
  });

  describe("プロパティテスト", () => {
    test("デフォルトの高さ（400px）が適用される", () => {
      const mockData = createMockCandlestickData(5);

      render(<CandlestickChart data={mockData} />);

      // ResponsiveContainerが存在することを確認
      expect(screen.getByTestId("responsive-container")).toBeInTheDocument();
    });

    test("loadingとerrorが同時に指定された場合、errorが優先される", () => {
      const mockData = createMockCandlestickData(5);
      const errorMessage = "エラーメッセージ";

      render(
        <CandlestickChart data={mockData} loading={true} error={errorMessage} />
      );

      // エラー状態が優先されることを確認
      expect(
        screen.getByText(/チャートの読み込みに失敗しました/)
      ).toBeInTheDocument();
      expect(screen.getByText(errorMessage)).toBeInTheDocument();
      expect(
        screen.queryByText(/チャートデータを読み込み中/)
      ).not.toBeInTheDocument();
    });
  });

  describe("アクセシビリティテスト", () => {
    test("適切なARIAラベルが設定されている", () => {
      const mockData = createMockCandlestickData(5);

      const { container } = render(<CandlestickChart data={mockData} />);

      // チャートコンテナが存在することを確認
      expect(container.querySelector(".w-full")).toBeInTheDocument();
    });

    test("エラー状態で適切なスタイルが適用される", () => {
      const mockData = createMockCandlestickData(5);

      const { container } = render(
        <CandlestickChart data={mockData} error="エラー" />
      );

      const errorContainer = container.querySelector(".bg-red-50");
      expect(errorContainer).toBeInTheDocument();
    });
  });
});
