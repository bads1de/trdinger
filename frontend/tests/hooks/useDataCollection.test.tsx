import { renderHook, act } from "@testing-library/react";
import { useDataCollection } from "@/hooks/useDataCollection";
import { useApiCall } from "@/hooks/useApiCall";

// useApiCallをモック
jest.mock("@/hooks/useApiCall", () => ({
  useApiCall: jest.fn(),
}));

describe("useDataCollection", () => {
  const mockExecute = jest.fn();

  beforeEach(() => {
    jest.clearAllMocks();
    (useApiCall as jest.Mock).mockReturnValue({
      execute: mockExecute,
      loading: false,
      error: null,
    });
  });

  it("初期状態が正しいこと", () => {
    const { result } = renderHook(() => useDataCollection());
    expect(result.current.isAnyLoading).toBe(false);
    expect(result.current.hasAnyError).toBe(false);
    expect(result.current.ohlcv.loading).toBe(false);
    expect(result.current.fundingRate.loading).toBe(false);
    expect(result.current.openInterest.loading).toBe(false);
  });

  it("collectOHLCVDataが正しいパラメータでAPIを呼ぶこと", async () => {
    const { result } = renderHook(() => useDataCollection());
    const onSuccess = jest.fn();
    const onError = jest.fn();

    // モックの実装：executeが呼ばれたらonSuccessを発火させる
    mockExecute.mockImplementation(async (url, options) => {
      if (options && options.onSuccess) {
        options.onSuccess("mock data");
      }
    });

    await act(async () => {
      await result.current.ohlcv.collect(onSuccess, onError);
    });

    expect(mockExecute).toHaveBeenCalledWith(
      "/api/data-collection/bulk-historical",
      expect.objectContaining({
        method: "POST",
        confirmMessage: expect.stringContaining("OHLCV"),
      })
    );
    expect(onSuccess).toHaveBeenCalledWith("mock data");
  });

  it("collectFundingRateDataが正しいパラメータでAPIを呼ぶこと", async () => {
    const { result } = renderHook(() => useDataCollection());

    await act(async () => {
      await result.current.fundingRate.collect();
    });

    expect(mockExecute).toHaveBeenCalledWith(
      "/api/funding-rates/bulk-collect",
      expect.objectContaining({
        method: "POST",
        confirmMessage: expect.stringContaining("FR"),
      })
    );
  });

  it("collectOpenInterestDataが正しいパラメータでAPIを呼ぶこと", async () => {
    const { result } = renderHook(() => useDataCollection());

    await act(async () => {
      await result.current.openInterest.collect();
    });

    expect(mockExecute).toHaveBeenCalledWith(
      "/api/open-interest/bulk-collect",
      expect.objectContaining({
        method: "POST",
        confirmMessage: expect.stringContaining("OI"),
      })
    );
  });

  it("APIエラー時にonErrorコールバックが呼ばれること", async () => {
    const { result } = renderHook(() => useDataCollection());
    const onError = jest.fn();

    // エラーケースのモック
    mockExecute.mockImplementation(async (url, options) => {
      if (options && options.onError) {
        options.onError("API Error");
      }
    });

    await act(async () => {
      await result.current.ohlcv.collect(undefined, onError);
    });

    expect(onError).toHaveBeenCalledWith("API Error");
  });
});
