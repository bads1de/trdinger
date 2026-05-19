import { renderHook, act } from "@testing-library/react";
import { useOhlcvData } from "@/hooks/useOhlcvData";

jest.mock("@/hooks/useDataFetching", () => ({
  useParameterizedDataFetching: jest.fn(),
}));

import { useParameterizedDataFetching } from "@/hooks/useDataFetching";

const mockUseParameterizedDataFetching = useParameterizedDataFetching as jest.MockedFunction<typeof useParameterizedDataFetching>;

describe("useOhlcvData", () => {
  const mockRefetch = jest.fn();
  const mockSetParams = jest.fn();

  beforeEach(() => {
    jest.clearAllMocks();
    mockUseParameterizedDataFetching.mockReturnValue({
      data: [],
      loading: false,
      error: null,
      refetch: mockRefetch,
      params: { symbol: "BTC/USDT:USDT", timeframe: "1h", limit: 100 },
      setParams: mockSetParams,
      reset: jest.fn(),
      setData: jest.fn(),
    });
  });

  it("初期状態が正しいこと", () => {
    const { result } = renderHook(() => useOhlcvData("BTC/USDT:USDT", "1h", 100));

    expect(result.current.data).toEqual([]);
    expect(result.current.loading).toBe(false);
    expect(result.current.error).toBeNull();
    expect(result.current.limit).toBe(100);
    expect(typeof result.current.refetch).toBe("function");
    expect(typeof result.current.setLimit).toBe("function");
  });

  it("データを再取得すること", () => {
    const { result } = renderHook(() => useOhlcvData("BTC/USDT:USDT", "1h", 100));

    act(() => {
      result.current.refetch();
    });

    expect(mockRefetch).toHaveBeenCalled();
  });

  it("setLimitで取得数を変更すること", () => {
    const { result } = renderHook(() => useOhlcvData("BTC/USDT:USDT", "1h", 100));

    act(() => {
      result.current.setLimit(200);
    });

    expect(mockSetParams).toHaveBeenCalledWith({ limit: 200 });
  });

  it("正しいエンドポイントとパラメータを渡すこと", () => {
    renderHook(() => useOhlcvData("BTC/USDT:USDT", "1h", 100));

    expect(mockUseParameterizedDataFetching).toHaveBeenCalledWith(
      "/api/market-data/ohlcv",
      { symbol: "BTC/USDT:USDT", timeframe: "1h", limit: 100 },
      expect.objectContaining({
        dependencies: ["BTC/USDT:USDT", "1h"],
        errorMessage: "OHLCVデータの取得に失敗しました",
      })
    );
  });
});
