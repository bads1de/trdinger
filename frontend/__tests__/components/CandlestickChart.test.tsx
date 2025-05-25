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

// ApexChartsのモック
jest.mock("react-apexcharts", () => {
  return {
    __esModule: true,
    default: ({ options, series, type, height }: any) => (
      <div
        data-testid="apex-chart"
        data-chart-type={type}
        data-chart-height={height}
        data-chart-options={JSON.stringify(options)}
        data-chart-series={JSON.stringify(series)}
      >
        ApexChart Mock
      </div>
    ),
  };
});

// Next.js dynamic importのモック
jest.mock("next/dynamic", () => {
  return (importFunc: any) => {
    const Component = ({ options, series, type, height }: any) => (
      <div
        data-testid="apex-chart"
        data-chart-type={type}
        data-chart-height={height}
        data-chart-options={JSON.stringify(options)}
        data-chart-series={JSON.stringify(series)}
      >
        ApexChart Mock
      </div>
    );
    Component.displayName = "MockedDynamicChart";
    return Component;
  };
});

describe("CandlestickChart", () => {
  describe("ApexCharts実装テスト", () => {
    test("ApexChartsのcandlestickチャートが正常にレンダリングされる", () => {
      const mockData = createMockCandlestickData(10);

      render(<CandlestickChart data={mockData} />);

      const apexChart = screen.getByTestId("apex-chart");
      expect(apexChart).toBeInTheDocument();
      expect(apexChart).toHaveAttribute("data-chart-type", "candlestick");
    });

    test("ローソク足の色が正しく設定される（陽線=白、陰線=グレー）", () => {
      const mockData = createMockCandlestickData(5);

      render(<CandlestickChart data={mockData} />);

      const apexChart = screen.getByTestId("apex-chart");
      const options = JSON.parse(
        apexChart.getAttribute("data-chart-options") || "{}"
      );

      expect(options.plotOptions?.candlestick?.colors?.upward).toBe("#ffffff");
      expect(options.plotOptions?.candlestick?.colors?.downward).toBe(
        "#808080"
      );
    });

    test("OHLCデータが正しい形式でApexChartsに渡される", () => {
      const mockData = createMockCandlestickData(3);

      render(<CandlestickChart data={mockData} />);

      const apexChart = screen.getByTestId("apex-chart");
      const series = JSON.parse(
        apexChart.getAttribute("data-chart-series") || "[]"
      );

      expect(series).toHaveLength(1);
      expect(series[0].data).toHaveLength(3);

      // 各データポイントがOHLC形式であることを確認
      series[0].data.forEach((point: any) => {
        expect(point).toHaveProperty("x");
        expect(point).toHaveProperty("y");
        expect(point.y).toHaveLength(4); // [open, high, low, close]
      });
    });

    test("カスタムの高さが適用される", () => {
      const mockData = createMockCandlestickData(5);
      const customHeight = 600;

      render(<CandlestickChart data={mockData} height={customHeight} />);

      const apexChart = screen.getByTestId("apex-chart");
      expect(apexChart).toHaveAttribute(
        "data-chart-height",
        customHeight.toString()
      );
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
      expect(screen.queryByTestId("apex-chart")).not.toBeInTheDocument();
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
      expect(screen.queryByTestId("apex-chart")).not.toBeInTheDocument();
    });

    test("エラー状態でチャートが表示されない", () => {
      const mockData = createMockCandlestickData(5);

      render(<CandlestickChart data={mockData} error="エラーが発生しました" />);

      expect(screen.queryByTestId("apex-chart")).not.toBeInTheDocument();
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

      // ApexChartが存在することを確認
      expect(screen.getByTestId("apex-chart")).toBeInTheDocument();
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
