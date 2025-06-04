/**
 * 取引散布図チャートコンポーネントのテスト
 */

import React from "react";
import { render, screen } from "@testing-library/react";
import "@testing-library/jest-dom";
import TradeScatterChart from "@/components/backtest/charts/TradeScatterChart";
import { generateMockChartTradeData } from "../../utils/chartTestUtils";

// Recharts のモック
jest.mock("recharts", () => ({
  ResponsiveContainer: ({ children }: { children: React.ReactNode }) => (
    <div data-testid="responsive-container">{children}</div>
  ),
  ScatterChart: ({ children }: { children: React.ReactNode }) => (
    <div data-testid="scatter-chart">{children}</div>
  ),
  Scatter: ({ name }: { name: string }) => (
    <div data-testid="scatter" data-name={name} />
  ),
  XAxis: ({ dataKey }: { dataKey: string }) => (
    <div data-testid={`x-axis-${dataKey}`} />
  ),
  YAxis: ({ dataKey }: { dataKey: string }) => (
    <div data-testid={`y-axis-${dataKey}`} />
  ),
  CartesianGrid: () => <div data-testid="cartesian-grid" />,
  Tooltip: () => <div data-testid="tooltip" />,
  Legend: () => <div data-testid="legend" />,
  ReferenceLine: ({ y }: { y: number }) => (
    <div data-testid="reference-line" data-y={y} />
  ),
}));

describe("TradeScatterChart", () => {
  const mockData = generateMockChartTradeData(30);

  beforeEach(() => {
    jest.clearAllMocks();
  });

  it("正常なデータでチャートをレンダリングする", () => {
    render(<TradeScatterChart data={mockData} title="取引散布図" />);

    // チャートコンテナが表示される
    expect(screen.getByTestId("responsive-container")).toBeInTheDocument();
    expect(screen.getByTestId("scatter-chart")).toBeInTheDocument();

    // 取引の散布図が表示される
    expect(screen.getByTestId("scatter")).toBeInTheDocument();

    // 軸が表示される
    expect(screen.getByTestId("x-axis-entryDate")).toBeInTheDocument();
    expect(screen.getByTestId("y-axis-returnPct")).toBeInTheDocument();

    // グリッド、ツールチップ、凡例が表示される
    expect(screen.getByTestId("cartesian-grid")).toBeInTheDocument();
    expect(screen.getByTestId("tooltip")).toBeInTheDocument();
    expect(screen.getByTestId("legend")).toBeInTheDocument();
  });

  it("ゼロラインの参照線を表示する", () => {
    render(
      <TradeScatterChart
        data={mockData}
        showZeroLine={true}
        title="取引散布図"
      />
    );

    // ゼロラインの参照線が表示される
    const referenceLine = screen.getByTestId("reference-line");
    expect(referenceLine).toBeInTheDocument();
    expect(referenceLine).toHaveAttribute("data-y", "0");
  });

  it("取引タイプ別に分けて表示する", () => {
    render(
      <TradeScatterChart
        data={mockData}
        separateByType={true}
        title="取引散布図"
      />
    );

    // ロング取引とショート取引の散布図が表示される
    const scatterElements = screen.getAllByTestId("scatter");
    expect(scatterElements.length).toBeGreaterThanOrEqual(1);
  });

  it("空のデータでエラー状態を表示する", () => {
    render(<TradeScatterChart data={[]} title="取引散布図" />);

    // エラー状態が表示される
    expect(screen.getByText("データがありません")).toBeInTheDocument();
  });

  it("ローディング状態を表示する", () => {
    render(
      <TradeScatterChart data={mockData} loading={true} title="取引散布図" />
    );

    // ローディング状態が表示される
    expect(screen.getByText("チャートを読み込み中...")).toBeInTheDocument();
  });

  it("エラー状態を表示する", () => {
    const errorMessage = "取引データの読み込みに失敗しました";

    render(
      <TradeScatterChart
        data={mockData}
        error={errorMessage}
        title="取引散布図"
      />
    );

    // エラー状態が表示される
    expect(screen.getByText("エラーが発生しました")).toBeInTheDocument();
    expect(screen.getByText(errorMessage)).toBeInTheDocument();
  });

  it("カスタムの高さとクラス名を適用する", () => {
    const customHeight = 450;
    const customClassName = "custom-scatter-chart";

    render(
      <TradeScatterChart
        data={mockData}
        height={customHeight}
        className={customClassName}
        title="取引散布図"
      />
    );

    // カスタムクラスが適用される
    const container = screen
      .getByTestId("responsive-container")
      .closest(".bg-gray-800\\/30");
    expect(container).toHaveClass(customClassName);
  });

  it("サブタイトルを表示する", () => {
    const subtitle = "利益/損失の分布と取引パフォーマンス";

    render(
      <TradeScatterChart
        data={mockData}
        title="取引散布図"
        subtitle={subtitle}
      />
    );

    expect(screen.getByText(subtitle)).toBeInTheDocument();
  });

  it("アクションボタンを表示する", () => {
    const actions = <button data-testid="filter-button">フィルター</button>;

    render(
      <TradeScatterChart data={mockData} title="取引散布図" actions={actions} />
    );

    expect(screen.getByTestId("filter-button")).toBeInTheDocument();
  });

  it("大量データをサンプリングして表示する", () => {
    const largeData = generateMockChartTradeData(1500);

    render(
      <TradeScatterChart
        data={largeData}
        maxDataPoints={500}
        title="取引散布図"
      />
    );

    // チャートが正常にレンダリングされる
    expect(screen.getByTestId("scatter-chart")).toBeInTheDocument();
  });

  it("勝ち取引のみのデータでも正しく表示する", () => {
    const winOnlyData = mockData.filter((trade) => trade.isWin);

    render(<TradeScatterChart data={winOnlyData} title="取引散布図" />);

    // チャートが正常にレンダリングされる
    expect(screen.getByTestId("scatter-chart")).toBeInTheDocument();
    expect(screen.getByTestId("scatter")).toBeInTheDocument();
  });

  it("負け取引のみのデータでも正しく表示する", () => {
    const lossOnlyData = mockData.filter((trade) => !trade.isWin);

    render(<TradeScatterChart data={lossOnlyData} title="取引散布図" />);

    // チャートが正常にレンダリングされる
    expect(screen.getByTestId("scatter-chart")).toBeInTheDocument();
    expect(screen.getByTestId("scatter")).toBeInTheDocument();
  });
});
