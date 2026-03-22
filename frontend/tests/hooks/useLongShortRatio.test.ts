import { renderHook, act, waitFor } from "@testing-library/react";
import { useLongShortRatio } from "@/hooks/useLongShortRatio";

// モック設定
jest.mock("@/hooks/useDataFetching", () => ({
  useParameterizedDataFetching: jest.fn(() => ({
    data: [
      {
        symbol: "BTC/USDT",
        period: "1h",
        buy_ratio: 0.6,
        sell_ratio: 0.4,
        timestamp: "2024-01-01T00:00:00Z",
        ls_ratio: 1.5,
      },
    ],
    loading: false,
    error: null,
    params: { symbol: "BTC/USDT", period: "1h", limit: 100 },
    setParams: jest.fn(),
    refetch: jest.fn(),
  })),
}));

jest.mock("@/hooks/useApiCall", () => ({
  useApiCall: jest.fn(() => ({
    execute: jest.fn().mockResolvedValue({ message: "success" }),
    loading: false,
  })),
}));

describe("useLongShortRatio", () => {
  beforeEach(() => {
    jest.clearAllMocks();
  });

  it("フックが正常に初期化されること", () => {
    const { result } = renderHook(() =>
      useLongShortRatio("BTC/USDT", "1h", 100)
    );

    expect(result.current.data).toHaveLength(1);
    expect(result.current.loading).toBe(false);
    expect(result.current.collecting).toBe(false);
    expect(result.current.error).toBeNull();
    expect(result.current.limit).toBe(100);
    expect(result.current.period).toBe("1h");
  });

  it("setLimitが正しく動作すること", () => {
    const mockSetParams = jest.fn();
    jest
      .requireMock("@/hooks/useDataFetching")
      .useParameterizedDataFetching.mockReturnValue({
        data: [],
        loading: false,
        error: null,
        params: { symbol: "BTC/USDT", period: "1h", limit: 100 },
        setParams: mockSetParams,
        refetch: jest.fn(),
      });

    const { result } = renderHook(() =>
      useLongShortRatio("BTC/USDT", "1h", 100)
    );

    act(() => {
      result.current.setLimit(200);
    });

    expect(mockSetParams).toHaveBeenCalledWith({ limit: 200 });
  });

  it("setPeriodが正しく動作すること", () => {
    const mockSetParams = jest.fn();
    jest
      .requireMock("@/hooks/useDataFetching")
      .useParameterizedDataFetching.mockReturnValue({
        data: [],
        loading: false,
        error: null,
        params: { symbol: "BTC/USDT", period: "1h", limit: 100 },
        setParams: mockSetParams,
        refetch: jest.fn(),
      });

    const { result } = renderHook(() =>
      useLongShortRatio("BTC/USDT", "1h", 100)
    );

    act(() => {
      result.current.setPeriod("5min");
    });

    expect(mockSetParams).toHaveBeenCalledWith({ period: "5min" });
  });

  it("collectDataがAPIを呼び出すこと", async () => {
    const mockExecute = jest.fn().mockResolvedValue({ message: "success" });
    jest.requireMock("@/hooks/useApiCall").useApiCall.mockReturnValue({
      execute: mockExecute,
      loading: false,
    });

    const { result } = renderHook(() =>
      useLongShortRatio("BTC/USDT", "1h", 100)
    );

    await act(async () => {
      await result.current.collectData("incremental");
    });

    expect(mockExecute).toHaveBeenCalledWith(
      expect.stringContaining("/api/long-short-ratio/collect"),
      expect.objectContaining({ method: "POST" })
    );
  });

  it("refetchが呼び出し可能であること", () => {
    const mockRefetch = jest.fn();
    jest
      .requireMock("@/hooks/useDataFetching")
      .useParameterizedDataFetching.mockReturnValue({
        data: [],
        loading: false,
        error: null,
        params: { symbol: "BTC/USDT", period: "1h", limit: 100 },
        setParams: jest.fn(),
        refetch: mockRefetch,
      });

    const { result } = renderHook(() =>
      useLongShortRatio("BTC/USDT", "1h", 100)
    );

    act(() => {
      result.current.refetch();
    });

    expect(mockRefetch).toHaveBeenCalled();
  });
});
