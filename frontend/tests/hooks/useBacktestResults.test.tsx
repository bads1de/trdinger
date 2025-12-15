import { renderHook, act, waitFor } from "@testing-library/react";
import { useBacktestResults } from "@/hooks/useBacktestResults";
import { useParameterizedDataFetching } from "@/hooks/useDataFetching";
import { useApiCall } from "@/hooks/useApiCall";

// モック化
jest.mock("@/hooks/useDataFetching", () => ({
  useParameterizedDataFetching: jest.fn(),
}));

jest.mock("@/hooks/useApiCall", () => ({
  useApiCall: jest.fn(),
}));

describe("useBacktestResults", () => {
  const mockRefetch = jest.fn();
  const mockExecute = jest.fn();

  beforeEach(() => {
    jest.clearAllMocks();

    // useParameterizedDataFetchingのデフォルトモック
    (useParameterizedDataFetching as jest.Mock).mockReturnValue({
      data: [],
      loading: false,
      error: null,
      refetch: mockRefetch,
    });

    // useApiCallのデフォルトモック
    // 常に有効なオブジェクトを返すようにする
    (useApiCall as jest.Mock).mockReturnValue({
      execute: mockExecute,
      loading: false,
      error: null,
      reset: jest.fn(),
    });
  });

  it("初期状態が正しいこと", () => {
    const { result } = renderHook(() => useBacktestResults());

    expect(result.current.results).toEqual([]);
    expect(result.current.total).toBe(0);
    expect(result.current.selectedResult).toBeNull();
    expect(result.current.resultsLoading).toBe(false);
  });

  it("結果の選択機能が動作すること", () => {
    const { result } = renderHook(() => useBacktestResults());
    const mockResult = { id: "1", strategy_name: "Test Strategy" } as any;

    act(() => {
      result.current.handleResultSelect(mockResult);
    });

    expect(result.current.selectedResult).toEqual(mockResult);

    act(() => {
      result.current.setSelectedResult(null);
    });

    expect(result.current.selectedResult).toBeNull();
  });

  it("結果の削除機能が動作すること", async () => {
    const { result } = renderHook(() => useBacktestResults());
    const mockResult = { id: "1", strategy_name: "Test Strategy" } as any;

    mockExecute.mockImplementation(async (_, options) => {
      options.onSuccess();
      return {};
    });

    await act(async () => {
      await result.current.handleDeleteResult(mockResult);
    });

    expect(mockExecute).toHaveBeenCalledWith(
      `/api/backtest/results/1/`,
      expect.objectContaining({
        method: "DELETE",
        confirmMessage: expect.stringContaining(mockResult.strategy_name),
      })
    );

    // 削除成功後に再取得が呼ばれること
    expect(mockRefetch).toHaveBeenCalled();
  });

  it("全削除機能が動作すること", async () => {
    const { result } = renderHook(() => useBacktestResults());

    mockExecute.mockImplementation(async (_, options) => {
      options.onSuccess();
      return {};
    });

    await act(async () => {
      await result.current.handleDeleteAllResults();
    });

    expect(mockExecute).toHaveBeenCalledWith(
      "/api/backtest/results-all",
      expect.objectContaining({
        method: "DELETE",
      })
    );

    // 削除成功後に再取得が呼ばれること
    expect(mockRefetch).toHaveBeenCalled();
    // 選択状態が解除されること
    expect(result.current.selectedResult).toBeNull();
  });
});
