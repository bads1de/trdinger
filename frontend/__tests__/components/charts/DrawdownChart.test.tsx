/**
 * ドローダウンチャートコンポーネントのテスト
 */

import React from "react";
import { render, screen } from "@testing-library/react";
import "@testing-library/jest-dom";
import DrawdownChart from "@/components/backtest/charts/DrawdownChart";
import { generateMockChartEquityData } from "../../utils/chartTestUtils";

// Recharts のモック
jest.mock("recharts", () => ({
  ResponsiveContainer: ({ children }: { children: React.ReactNode }) => (
    <div data-testid="responsive-container">{children}</div>
  ),
  AreaChart: ({ children }: { children: React.ReactNode }) => (
    <div data-testid="area-chart">{children}</div>
  ),
  Area: ({ dataKey }: { dataKey: string }) => (
    <div data-testid={`area-${dataKey}`} />
  ),
  XAxis: ({ dataKey }: { dataKey: string }) => (
    <div data-testid={`x-axis-${dataKey}`} />
  ),
  YAxis: () => <div data-testid="y-axis" />,
  CartesianGrid: () => <div data-testid="cartesian-grid" />,
  Tooltip: () => <div data-testid="tooltip" />,
  ReferenceLine: ({ y }: { y: number }) => (
    <div data-testid="reference-line" data-y={y} />
  ),
}));

describe("DrawdownChart", () => {
  const mockData = generateMockChartEquityData(50).map((point) => ({
    ...point,
    drawdown: Math.random() * 15, // 0-15%のドローダウン
  }));

  beforeEach(() => {
    jest.clearAllMocks();
  });

  it("正常なデータでチャートをレンダリングする", () => {
    render(<DrawdownChart data={mockData} title="ドローダウン" />);

    // チャートコンテナが表示される
    expect(screen.getByTestId("responsive-container")).toBeInTheDocument();
    expect(screen.getByTestId("area-chart")).toBeInTheDocument();

    // ドローダウンエリアが表示される
    expect(screen.getByTestId("area-drawdown")).toBeInTheDocument();

    // 軸が表示される
    expect(screen.getByTestId("x-axis-date")).toBeInTheDocument();
    expect(screen.getByTestId("y-axis")).toBeInTheDocument();

    // グリッドとツールチップが表示される
    expect(screen.getByTestId("cartesian-grid")).toBeInTheDocument();
    expect(screen.getByTestId("tooltip")).toBeInTheDocument();
  });

  it("最大ドローダウンの参照線を表示する", () => {
    const maxDrawdown = 12.5;

    render(
      <DrawdownChart
        data={mockData}
        maxDrawdown={maxDrawdown}
        showMaxDrawdown={true}
        title="ドローダウン"
      />
    );

    // 最大ドローダウンの参照線が表示される（負の値として）
    const referenceLines = screen.getAllByTestId("reference-line");
    const maxDrawdownLine = referenceLines.find(
      (line) => line.getAttribute("data-y") === (-maxDrawdown).toString()
    );
    expect(maxDrawdownLine).toBeInTheDocument();
  });

  it("空のデータでエラー状態を表示する", () => {
    render(<DrawdownChart data={[]} title="ドローダウン" />);

    // エラー状態が表示される
    expect(screen.getByText("データがありません")).toBeInTheDocument();
  });

  it("ローディング状態を表示する", () => {
    render(
      <DrawdownChart data={mockData} loading={true} title="ドローダウン" />
    );

    // ローディング状態が表示される
    expect(screen.getByText("チャートを読み込み中...")).toBeInTheDocument();
  });

  it("エラー状態を表示する", () => {
    const errorMessage = "ドローダウンデータの読み込みに失敗しました";

    render(
      <DrawdownChart
        data={mockData}
        error={errorMessage}
        title="ドローダウン"
      />
    );

    // エラー状態が表示される
    expect(screen.getByText("エラーが発生しました")).toBeInTheDocument();
    expect(screen.getByText(errorMessage)).toBeInTheDocument();
  });

  it("カスタムの高さとクラス名を適用する", () => {
    const customHeight = 350;
    const customClassName = "custom-drawdown-chart";

    render(
      <DrawdownChart
        data={mockData}
        height={customHeight}
        className={customClassName}
        title="ドローダウン"
      />
    );

    // カスタムクラスが適用される
    const container = screen
      .getByTestId("responsive-container")
      .closest(".bg-gray-800\\/30");
    expect(container).toHaveClass(customClassName);
  });

  it("サブタイトルを表示する", () => {
    const subtitle = "最大ドローダウン期間の可視化";

    render(
      <DrawdownChart data={mockData} title="ドローダウン" subtitle={subtitle} />
    );

    expect(screen.getByText(subtitle)).toBeInTheDocument();
  });

  it("アクションボタンを表示する", () => {
    const actions = <button data-testid="analyze-button">分析</button>;

    render(
      <DrawdownChart data={mockData} title="ドローダウン" actions={actions} />
    );

    expect(screen.getByTestId("analyze-button")).toBeInTheDocument();
  });

  it("大量データをサンプリングして表示する", () => {
    const largeData = generateMockChartEquityData(2000).map((point) => ({
      ...point,
      drawdown: Math.random() * 20,
    }));

    render(
      <DrawdownChart
        data={largeData}
        maxDataPoints={800}
        title="ドローダウン"
      />
    );

    // チャートが正常にレンダリングされる
    expect(screen.getByTestId("area-chart")).toBeInTheDocument();
  });

  it("ドローダウンが0の場合も正しく表示する", () => {
    const noDrawdownData = mockData.map((point) => ({
      ...point,
      drawdown: 0,
    }));

    render(<DrawdownChart data={noDrawdownData} title="ドローダウン" />);

    // チャートが正常にレンダリングされる
    expect(screen.getByTestId("area-chart")).toBeInTheDocument();
    expect(screen.getByTestId("area-drawdown")).toBeInTheDocument();
  });
});
