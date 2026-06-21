import { renderHook, act } from "@testing-library/react";
import { useOpenInterestData } from "@/hooks/useOpenInterestData";

jest.mock("@/hooks/useDataFetching", () => ({
  useParameterizedDataFetching: jest.fn(),
}));

import { useParameterizedDataFetching } from "@/hooks/useDataFetching";

const mockUseParamDataFetching = useParameterizedDataFetching as jest.MockedFunction<
  typeof useParameterizedDataFetching
>;

describe("useOpenInterestData", () => {
  const mockSetParams = jest.fn();
  const mockRefetch = jest.fn();

  beforeEach(() => {
    jest.clearAllMocks();
    mockUseParamDataFetching.mockReturnValue({
      data: [],
      loading: false,
      error: null,
      params: { symbol: "BTC/USDT:USDT", limit: 100 },
      setParams: mockSetParams,
      refetch: mockRefetch,
    } as any);
  });

  it("指定されたシンボルで初期化されること", () => {
    renderHook(() => useOpenInterestData("BTC/USDT:USDT", 100));

    expect(mockUseParamDataFetching).toHaveBeenCalledWith(
      "/api/open-interest/",
      { symbol: "BTC/USDT:USDT", limit: 100 },
      expect.objectContaining({
        dataPath: "data.open_interest",
        dependencies: ["BTC/USDT:USDT"],
      })
    );
  });

  it("データが正しく返されること", () => {
    const mockData = [
      { symbol: "BTC/USDT:USDT", open_interest: 100000, timestamp: "2023-01-01" },
    ];
    mockUseParamDataFetching.mockReturnValue({
      data: mockData,
      loading: false,
      error: null,
      params: { symbol: "BTC/USDT:USDT", limit: 100 },
      setParams: mockSetParams,
      refetch: mockRefetch,
    } as any);

    const { result } = renderHook(() => useOpenInterestData("BTC/USDT:USDT"));
    expect(result.current.data).toEqual(mockData);
  });

  it("loading状態が正しく返されること", () => {
    mockUseParamDataFetching.mockReturnValue({
      data: [],
      loading: true,
      error: null,
      params: { symbol: "BTC/USDT:USDT", limit: 100 },
      setParams: mockSetParams,
      refetch: mockRefetch,
    } as any);

    const { result } = renderHook(() => useOpenInterestData("BTC/USDT:USDT"));
    expect(result.current.loading).toBe(true);
  });

  it("error状態が正しく返されること", () => {
    mockUseParamDataFetching.mockReturnValue({
      data: [],
      loading: false,
      error: "データ取得エラー",
      params: { symbol: "BTC/USDT:USDT", limit: 100 },
      setParams: mockSetParams,
      refetch: mockRefetch,
    } as any);

    const { result } = renderHook(() => useOpenInterestData("BTC/USDT:USDT"));
    expect(result.current.error).toBe("データ取得エラー");
  });

  it("setLimitでlimitが更新されること", () => {
    const { result } = renderHook(() => useOpenInterestData("BTC/USDT:USDT", 100));

    act(() => {
      result.current.setLimit(200);
    });

    expect(mockSetParams).toHaveBeenCalledWith({ limit: 200 });
  });

  it("refetchが正しく呼び出せること", () => {
    const { result } = renderHook(() => useOpenInterestData("BTC/USDT:USDT"));

    act(() => {
      result.current.refetch();
    });

    expect(mockRefetch).toHaveBeenCalledTimes(1);
  });

  it("limitの現在値がparamsから取得されること", () => {
    mockUseParamDataFetching.mockReturnValue({
      data: [],
      loading: false,
      error: null,
      params: { symbol: "ETH/USDT:USDT", limit: 50 },
      setParams: mockSetParams,
      refetch: mockRefetch,
    } as any);

    const { result } = renderHook(() => useOpenInterestData("ETH/USDT:USDT", 50));
    expect(result.current.limit).toBe(50);
  });

  it("異なるinitialLimitで初期化されること", () => {
    renderHook(() => useOpenInterestData("BTC/USDT:USDT", 500));

    expect(mockUseParamDataFetching).toHaveBeenCalledWith(
      "/api/open-interest/",
      { symbol: "BTC/USDT:USDT", limit: 500 },
      expect.anything()
    );
  });
});
