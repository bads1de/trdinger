import React from "react";
import { render, screen, waitFor } from "@testing-library/react";
import "@testing-library/jest-dom";
import { LongShortRatioChart } from "@/components/data/LongShortRatioChart";
import { LongShortRatioData } from "@/types/long-short-ratio";

const mockData: LongShortRatioData[] = [
  {
    symbol: "BTC/USDT",
    period: "1h",
    buy_ratio: 0.6,
    sell_ratio: 0.4,
    timestamp: "2024-01-01T00:00:00Z",
    ls_ratio: 1.5,
  },
  {
    symbol: "BTC/USDT",
    period: "1h",
    buy_ratio: 0.55,
    sell_ratio: 0.45,
    timestamp: "2024-01-01T01:00:00Z",
    ls_ratio: 1.222,
  },
];

describe("LongShortRatioChart", () => {
  const defaultProps = {
    data: mockData,
    loading: false,
    collecting: false,
    onRefresh: jest.fn(),
    onCollect: jest.fn(),
    period: "1h",
    symbol: "BTC/USDT",
  };

  beforeEach(() => {
    jest.clearAllMocks();
  });

  it("コンポーネントが正常にレンダリングされること", () => {
    render(<LongShortRatioChart {...defaultProps} />);
    expect(screen.getByText(/Long\/Short Ratio/i)).toBeInTheDocument();
    expect(screen.getByText(/BTC\/USDT/)).toBeInTheDocument();
    expect(screen.getByText(/1h/)).toBeInTheDocument();
  });

  it("データが空の場合「No data available」と表示されること", () => {
    render(<LongShortRatioChart {...defaultProps} data={[]} />);
    expect(screen.getByText("No data available")).toBeInTheDocument();
  });

  it("リフレッシュボタンがクリックされたらonRefreshが呼ばれること", async () => {
    const onRefresh = jest.fn();
    render(<LongShortRatioChart {...defaultProps} onRefresh={onRefresh} />);

    // RefreshCwアイコンを持つボタンをクリック
    const refreshButton = screen.getAllByRole("button")[0];
    refreshButton.click();

    expect(onRefresh).toHaveBeenCalled();
  });

  it("収集ボタンがクリックされたらonCollectが呼ばれること", async () => {
    const onCollect = jest.fn();
    render(<LongShortRatioChart {...defaultProps} onCollect={onCollect} />);

    // Collect Latestボタンをクリック
    const collectButton = screen.getByText("Collect Latest");
    collectButton.click();

    expect(onCollect).toHaveBeenCalledWith("incremental");
  });

  it("loading時にリフレッシュボタンが無効化されること", () => {
    render(<LongShortRatioChart {...defaultProps} loading={true} />);
    const refreshButton = screen.getAllByRole("button")[0];
    expect(refreshButton).toBeDisabled();
  });

  it("collecting時に収集ボタンが無効化されること", () => {
    render(<LongShortRatioChart {...defaultProps} collecting={true} />);
    const collectButton = screen.getByRole("button", {
      name: /Collect Latest/i,
    });
    expect(collectButton).toBeDisabled();
  });
});
