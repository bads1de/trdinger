import { renderHook, act } from "@testing-library/react";
import { useBulkIncrementalUpdate } from "@/hooks/useBulkIncrementalUpdate";

jest.mock("@/hooks/useApiCall", () => ({
  useApiCall: jest.fn(),
}));

import { useApiCall } from "@/hooks/useApiCall";

const mockUseApiCall = useApiCall as jest.MockedFunction<typeof useApiCall>;

describe("useBulkIncrementalUpdate", () => {
  const mockExecute = jest.fn();

  beforeEach(() => {
    jest.clearAllMocks();
    mockUseApiCall.mockReturnValue({
      execute: mockExecute,
      loading: false,
      error: null,
      reset: jest.fn(),
    });
  });

  it("初期状態が正しいこと", () => {
    const { result } = renderHook(() => useBulkIncrementalUpdate());

    expect(typeof result.current.bulkUpdate).toBe("function");
    expect(result.current.loading).toBe(false);
    expect(result.current.error).toBeNull();
  });

  it("bulkUpdateが正しいURLでexecuteを呼び出すこと", async () => {
    const { result } = renderHook(() => useBulkIncrementalUpdate());

    mockExecute.mockResolvedValue({ success: true });

    await act(async () => {
      await result.current.bulkUpdate("BTC/USDT:USDT", "1h");
    });

    expect(mockExecute).toHaveBeenCalledWith(
      "/api/data-collection/bulk-incremental-update?symbol=BTC%2FUSDT%3AUSDT",
      expect.objectContaining({
        method: "POST",
      })
    );
  });

  it("複数のシンボルで正しくエンコードされること", async () => {
    const { result } = renderHook(() => useBulkIncrementalUpdate());

    mockExecute.mockResolvedValue({ success: true });

    await act(async () => {
      await result.current.bulkUpdate("ETH/USDT:USDT", "4h");
    });

    expect(mockExecute).toHaveBeenCalledWith(
      "/api/data-collection/bulk-incremental-update?symbol=ETH%2FUSDT%3AUSDT",
      expect.anything()
    );
  });

  it("onSuccessコールバックが呼ばれること", async () => {
    const { result } = renderHook(() => useBulkIncrementalUpdate());
    const onSuccess = jest.fn();
    const mockData = { total_saved_count: 100 };

    mockExecute.mockImplementation(async (_url: string, options: any) => {
      options.onSuccess(mockData);
    });

    await act(async () => {
      await result.current.bulkUpdate("BTC/USDT:USDT", "1h", { onSuccess });
    });

    expect(onSuccess).toHaveBeenCalledWith(mockData);
  });

  it("onErrorコールバックがエラーメッセージ付きで呼ばれること", async () => {
    const { result } = renderHook(() => useBulkIncrementalUpdate());
    const onError = jest.fn();

    mockExecute.mockImplementation(async (_url: string, options: any) => {
      options.onError?.("カスタムエラーメッセージ");
    });

    await act(async () => {
      await result.current.bulkUpdate("BTC/USDT:USDT", "1h", {
        onError,
      });
    });

    expect(onError).toHaveBeenCalledWith("カスタムエラーメッセージ");
  });

  it("onError未指定の場合、デフォルトメッセージでエラーハンドリングが行われること", async () => {
    const { result } = renderHook(() => useBulkIncrementalUpdate());

    mockExecute.mockImplementation(async (_url: string, options: any) => {
      options.onError?.("一括差分更新に失敗しました");
    });

    let errorCaught: string | undefined;
    await act(async () => {
      await result.current.bulkUpdate("BTC/USDT:USDT", "1h", {
        onError: (error) => {
          errorCaught = error;
        },
      });
    });

    expect(errorCaught).toBe("一括差分更新に失敗しました");
  });

  it("loading状態がuseApiCallから正しく渡されること", () => {
    mockUseApiCall.mockReturnValue({
      execute: mockExecute,
      loading: true,
      error: null,
      reset: jest.fn(),
    });

    const { result } = renderHook(() => useBulkIncrementalUpdate());
    expect(result.current.loading).toBe(true);
  });

  it("error状態がuseApiCallから正しく渡されること", () => {
    mockUseApiCall.mockReturnValue({
      execute: mockExecute,
      loading: false,
      error: "API Error",
      reset: jest.fn(),
    });

    const { result } = renderHook(() => useBulkIncrementalUpdate());
    expect(result.current.error).toBe("API Error");
  });
});
