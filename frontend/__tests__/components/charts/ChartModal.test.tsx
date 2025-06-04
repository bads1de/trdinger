/**
 * チャートモーダルコンポーネントのテスト
 */

import React from "react";
import { render, screen, fireEvent } from "@testing-library/react";
import "@testing-library/jest-dom";
import ChartModal from "@/components/backtest/charts/ChartModal";
import { generateMockBacktestResult } from "../../utils/chartTestUtils";

// Recharts のモック
jest.mock("recharts", () => ({
  ResponsiveContainer: ({ children }: { children: React.ReactNode }) => (
    <div data-testid="responsive-container">{children}</div>
  ),
  LineChart: ({ children }: { children: React.ReactNode }) => (
    <div data-testid="line-chart">{children}</div>
  ),
  AreaChart: ({ children }: { children: React.ReactNode }) => (
    <div data-testid="area-chart">{children}</div>
  ),
  ScatterChart: ({ children }: { children: React.ReactNode }) => (
    <div data-testid="scatter-chart">{children}</div>
  ),
  Line: () => <div data-testid="line" />,
  Area: () => <div data-testid="area" />,
  Scatter: () => <div data-testid="scatter" />,
  XAxis: () => <div data-testid="x-axis" />,
  YAxis: () => <div data-testid="y-axis" />,
  CartesianGrid: () => <div data-testid="cartesian-grid" />,
  Tooltip: () => <div data-testid="tooltip" />,
  Legend: () => <div data-testid="legend" />,
  ReferenceLine: () => <div data-testid="reference-line" />,
}));

describe("ChartModal", () => {
  const mockResult = generateMockBacktestResult();
  const mockOnClose = jest.fn();

  beforeEach(() => {
    jest.clearAllMocks();
  });

  it("モーダルが開いている時に表示される", () => {
    render(
      <ChartModal isOpen={true} onClose={mockOnClose} result={mockResult} />
    );

    // モーダルオーバーレイが表示される
    expect(screen.getByTestId("modal-overlay")).toBeInTheDocument();

    // モーダルコンテンツが表示される
    expect(screen.getByTestId("modal-content")).toBeInTheDocument();

    // タイトルが表示される
    expect(screen.getByText("チャート分析")).toBeInTheDocument();
  });

  it("モーダルが閉じている時に表示されない", () => {
    render(
      <ChartModal isOpen={false} onClose={mockOnClose} result={mockResult} />
    );

    // モーダルが表示されない
    expect(screen.queryByTestId("modal-overlay")).not.toBeInTheDocument();
  });

  it("閉じるボタンをクリックするとonCloseが呼ばれる", () => {
    render(
      <ChartModal isOpen={true} onClose={mockOnClose} result={mockResult} />
    );

    // 閉じるボタンをクリック
    const closeButton = screen.getByTestId("close-button");
    fireEvent.click(closeButton);

    expect(mockOnClose).toHaveBeenCalledTimes(1);
  });

  it("オーバーレイをクリックするとonCloseが呼ばれる", () => {
    render(
      <ChartModal isOpen={true} onClose={mockOnClose} result={mockResult} />
    );

    // オーバーレイをクリック
    const overlay = screen.getByTestId("modal-overlay");
    fireEvent.click(overlay);

    expect(mockOnClose).toHaveBeenCalledTimes(1);
  });

  it("モーダルコンテンツをクリックしてもonCloseが呼ばれない", () => {
    render(
      <ChartModal isOpen={true} onClose={mockOnClose} result={mockResult} />
    );

    // モーダルコンテンツをクリック
    const content = screen.getByTestId("modal-content");
    fireEvent.click(content);

    expect(mockOnClose).not.toHaveBeenCalled();
  });

  it("Escapeキーを押すとonCloseが呼ばれる", () => {
    render(
      <ChartModal isOpen={true} onClose={mockOnClose} result={mockResult} />
    );

    // Escapeキーを押す
    fireEvent.keyDown(document, { key: "Escape", code: "Escape" });

    expect(mockOnClose).toHaveBeenCalledTimes(1);
  });

  it("タブナビゲーションが表示される", () => {
    render(
      <ChartModal isOpen={true} onClose={mockOnClose} result={mockResult} />
    );

    // タブボタンが表示される（title属性で識別）
    expect(screen.getByTitle("時系列での資産推移")).toBeInTheDocument();
    expect(screen.getByTitle("最大下落期間の分析")).toBeInTheDocument();
    expect(screen.getByTitle("利益/損失の分布")).toBeInTheDocument();
  });

  it("タブをクリックすると表示が切り替わる", () => {
    render(
      <ChartModal isOpen={true} onClose={mockOnClose} result={mockResult} />
    );

    // 初期状態では資産曲線タブがアクティブ
    const equityTab = screen.getByTitle("時系列での資産推移");
    expect(equityTab).toHaveClass("bg-blue-600");

    // ドローダウンタブをクリック
    const drawdownTab = screen.getByTitle("最大下落期間の分析");
    fireEvent.click(drawdownTab);

    // ドローダウンタブがアクティブになる
    expect(drawdownTab).toHaveClass("bg-blue-600");
    expect(equityTab).not.toHaveClass("bg-blue-600");
  });

  it("資産曲線チャートが表示される", () => {
    render(
      <ChartModal isOpen={true} onClose={mockOnClose} result={mockResult} />
    );

    // 資産曲線チャートが表示される
    expect(screen.getByTestId("line-chart")).toBeInTheDocument();
  });

  it("ドローダウンタブでドローダウンチャートが表示される", () => {
    render(
      <ChartModal isOpen={true} onClose={mockOnClose} result={mockResult} />
    );

    // ドローダウンタブをクリック
    const drawdownTab = screen.getByTitle("最大下落期間の分析");
    fireEvent.click(drawdownTab);

    // ドローダウンチャートが表示される
    expect(screen.getByTestId("area-chart")).toBeInTheDocument();
  });

  it("取引分析タブで取引散布図が表示される", () => {
    render(
      <ChartModal isOpen={true} onClose={mockOnClose} result={mockResult} />
    );

    // 取引分析タブをクリック
    const tradeTab = screen.getByTitle("利益/損失の分布");
    fireEvent.click(tradeTab);

    // 取引散布図が表示される
    expect(screen.getByTestId("scatter-chart")).toBeInTheDocument();
  });

  it("データがない場合にエラー状態を表示する", () => {
    const emptyResult = {
      ...mockResult,
      equity_curve: [],
      trade_history: [],
    };

    render(
      <ChartModal isOpen={true} onClose={mockOnClose} result={emptyResult} />
    );

    // エラー状態が表示される
    expect(screen.getByText("データがありません")).toBeInTheDocument();
  });

  it("レスポンシブ対応でモバイル表示になる", () => {
    // モバイルサイズに変更
    Object.defineProperty(window, "innerWidth", {
      writable: true,
      configurable: true,
      value: 375,
    });

    render(
      <ChartModal isOpen={true} onClose={mockOnClose} result={mockResult} />
    );

    // モーダルが全画面表示になる
    const modalContent = screen.getByTestId("modal-content");
    expect(modalContent).toHaveClass("w-full", "h-full");
  });
});
