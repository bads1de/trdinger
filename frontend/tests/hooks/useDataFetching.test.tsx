import { renderHook, act, waitFor } from "@testing-library/react";
import { useDataFetching } from "@/hooks/useDataFetching";

// useApiCallをモック化
jest.mock("@/hooks/useApiCall", () => ({
  useApiCall: jest.fn(),
}));

import { useApiCall } from "@/hooks/useApiCall";

describe("useDataFetching", () => {
  const mockExecute = jest.fn();
  const mockReset = jest.fn();

  beforeEach(() => {
    jest.clearAllMocks();
    (useApiCall as jest.Mock).mockReturnValue({
      loading: false,
      error: null,
      execute: mockExecute,
      reset: mockReset,
    });
  });

  it("初期状態が正しいこと", () => {
    const { result } = renderHook(() =>
      useDataFetching({ endpoint: "/test-endpoint" })
    );

    expect(result.current.data).toEqual([]);
    expect(result.current.params).toEqual({});
    expect(result.current.loading).toBe(false);
    expect(result.current.error).toBeNull();
  });

  it("自動フェッチが行われること", async () => {
    mockExecute.mockImplementation(async (_, options) => {
      options.onSuccess({ success: true, data: ["item1", "item2"] });
      return { success: true, data: ["item1", "item2"] };
    });

    const { result } = renderHook(() =>
      useDataFetching({ endpoint: "/test-endpoint" })
    );

    await waitFor(() => {
      expect(mockExecute).toHaveBeenCalledTimes(1);
    });

    // 初回レンダリング後にfetchDataが呼ばれることを確認
    expect(mockExecute).toHaveBeenCalledWith(
      "/test-endpoint",
      expect.objectContaining({ method: "GET" })
    );

    expect(result.current.data).toEqual(["item1", "item2"]);
  });

  it("パラメータが変更されたときに再フェッチが行われること", async () => {
    mockExecute.mockImplementation(async (_, options) => {
      options.onSuccess({ success: true, data: [] });
      return { success: true, data: [] };
    });

    const { result } = renderHook(() =>
      useDataFetching({
        endpoint: "/test-endpoint",
        initialParams: { page: 1 },
      })
    );

    // 初回のフェッチ
    await waitFor(() => {
      expect(mockExecute).toHaveBeenCalledTimes(1);
    });

    // パラメータ変更
    act(() => {
      result.current.setParams({ page: 2 });
    });

    // 2回目のフェッチを待機
    await waitFor(() => {
      expect(mockExecute).toHaveBeenCalledTimes(2);
    });

    expect(mockExecute).toHaveBeenLastCalledWith(
      expect.stringContaining("page=2"),
      expect.anything()
    );
  });

  it("disableAutoFetchがtrueの場合、自動フェッチが行われないこと", () => {
    const { result } = renderHook(() =>
      useDataFetching({
        endpoint: "/test-endpoint",
        disableAutoFetch: true,
      })
    );

    expect(mockExecute).not.toHaveBeenCalled();

    // 手動でrefetch
    act(() => {
      result.current.refetch();
    });

    expect(mockExecute).toHaveBeenCalledTimes(1);
  });

  it("依存関係が不足している場合、フェッチが行われないこと", () => {
    const { result } = renderHook(() =>
      useDataFetching({
        endpoint: "/test-endpoint",
        dependencies: [null], // 依存関係がnull
      })
    );

    expect(mockExecute).not.toHaveBeenCalled();
  });

  it("transformオプションが正しく機能すること", async () => {
    mockExecute.mockImplementation(async (_, options) => {
      const rawData = { items: ["transformed"] };
      options.onSuccess(rawData);
      return rawData;
    });

    const transform = jest.fn((data) => data.items);

    const { result } = renderHook(() =>
      useDataFetching({
        endpoint: "/test-endpoint",
        transform,
      })
    );

    await waitFor(() => {
      expect(result.current.data).toEqual(["transformed"]);
    });

    expect(transform).toHaveBeenCalledWith({ items: ["transformed"] });
  });

  it("dataPathオプションが正しく機能すること", async () => {
    mockExecute.mockImplementation(async (_, options) => {
      const rawData = { result: { list: ["nested"] } };
      options.onSuccess(rawData);
      return rawData;
    });

    const { result } = renderHook(() =>
      useDataFetching({
        endpoint: "/test-endpoint",
        dataPath: "result.list",
      })
    );

    await waitFor(() => {
      expect(result.current.data).toEqual(["nested"]);
    });
  });

  it("reset関数で状態が初期化されること", async () => {
    const { result } = renderHook(() =>
      useDataFetching({
        endpoint: "/test-endpoint",
        initialParams: { page: 1 },
      })
    );

    act(() => {
      result.current.setData(["some", "data"]);
      result.current.setParams({ page: 2 });
    });

    act(() => {
      result.current.reset();
    });

    expect(result.current.data).toEqual([]);
    expect(result.current.params).toEqual({ page: 1 });
    expect(mockReset).toHaveBeenCalled();
  });
});
