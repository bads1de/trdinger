import { renderHook, act } from "@testing-library/react";
import { useDataReset } from "@/hooks/useDataReset";

jest.mock("@/hooks/useDataFetching", () => ({
  useDataFetching: jest.fn(),
}));

jest.mock("@/hooks/useApiCall", () => ({
  useApiCall: jest.fn(),
}));

import { useDataFetching } from "@/hooks/useDataFetching";

const mockUseDataFetching = useDataFetching as jest.MockedFunction<typeof useDataFetching>;

describe("useDataReset", () => {
  const mockRefetch = jest.fn();

  beforeEach(() => {
    jest.clearAllMocks();
    jest.useFakeTimers();
    mockUseDataFetching.mockReturnValue({
      data: [],
      loading: false,
      error: null,
      refetch: mockRefetch,
      params: {},
      setParams: jest.fn(),
      reset: jest.fn(),
      setData: jest.fn(),
    });
  });

  afterEach(() => {
    jest.useRealTimers();
  });

  it("初期状態が正しいこと", () => {
    const { result } = renderHook(() => useDataReset(true));

    expect(result.current.dataStatus).toBeNull();
    expect(result.current.resetMessage).toBe("");
    expect(result.current.isLoading).toBe(false);
    expect(result.current.error).toBeNull();
    expect(typeof result.current.handleResetComplete).toBe("function");
    expect(typeof result.current.handleResetError).toBe("function");
  });

  it("isVisibleがfalseの場合、自動取得を無効にすること", () => {
    renderHook(() => useDataReset(false));

    expect(mockUseDataFetching).toHaveBeenCalledWith(
      expect.objectContaining({
        disableAutoFetch: true,
      })
    );
  });

  it("isVisibleがtrueの場合、自動取得を有効にすること", () => {
    renderHook(() => useDataReset(true));

    expect(mockUseDataFetching).toHaveBeenCalledWith(
      expect.objectContaining({
        disableAutoFetch: false,
      })
    );
  });

  it("リセット成功時にメッセージを表示すること", () => {
    const { result } = renderHook(() => useDataReset(true));

    act(() => {
      result.current.handleResetComplete({
        success: true,
        message: "リセット完了",
        total_deleted: 1000,
        timestamp: new Date().toISOString(),
      });
    });

    expect(result.current.resetMessage).toContain("リセット完了");
    expect(result.current.resetMessage).toContain("1,000件削除");
  });

  it("リセット失敗時にエラーメッセージを表示すること", () => {
    const { result } = renderHook(() => useDataReset(true));

    act(() => {
      result.current.handleResetComplete({
        success: false,
        message: "リセット失敗",
        timestamp: new Date().toISOString(),
      });
    });

    expect(result.current.resetMessage).toContain("リセット失敗");
  });

  it("handleResetErrorでエラーメッセージを表示すること", () => {
    const { result } = renderHook(() => useDataReset(true));

    act(() => {
      result.current.handleResetError("エラーが発生しました");
    });

    expect(result.current.resetMessage).toContain("エラーが発生しました");
  });

  it("リセット完了後にデータステータスを再取得すること", () => {
    const { result } = renderHook(() => useDataReset(true));

    act(() => {
      result.current.handleResetComplete({
        success: true,
        message: "完了",
        timestamp: new Date().toISOString(),
      });
    });

    act(() => {
      jest.advanceTimersByTime(1500);
    });

    expect(mockRefetch).toHaveBeenCalled();
  });
});
