import { renderHook, act } from "@testing-library/react";
import { useModelInfo } from "@/hooks/useModelInfo";

jest.mock("@/hooks/useDataFetching", () => ({
  useDataFetching: jest.fn(),
}));

import { useDataFetching } from "@/hooks/useDataFetching";

const mockUseDataFetching = useDataFetching as jest.MockedFunction<typeof useDataFetching>;

describe("useModelInfo", () => {
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
    const { result } = renderHook(() => useModelInfo());

    expect(result.current.modelStatus).toBeNull();
    expect(result.current.loading).toBe(false);
    expect(result.current.error).toBeNull();
    expect(typeof result.current.loadModelStatus).toBe("function");
    expect(typeof result.current.getModelTypeBadgeVariant).toBe("function");
    expect(typeof result.current.getAccuracyBadgeVariant).toBe("function");
  });

  it("モデル状態を再取得すること", () => {
    const { result } = renderHook(() => useModelInfo());

    act(() => {
      result.current.loadModelStatus();
    });

    expect(mockRefetch).toHaveBeenCalled();
  });

  it("autoRefreshIntervalを設定すると自動更新が有効になること", () => {
    renderHook(() => useModelInfo(5));

    act(() => {
      jest.advanceTimersByTime(5000);
    });

    expect(mockRefetch).toHaveBeenCalled();
  });

  it("getModelTypeBadgeVariantがlightgbmにdefaultを返すこと", () => {
    const { result } = renderHook(() => useModelInfo());

    expect(result.current.getModelTypeBadgeVariant("lightgbm")).toBe("default");
  });

  it("getModelTypeBadgeVariantがxgboostにoutlineを返すこと", () => {
    const { result } = renderHook(() => useModelInfo());

    expect(result.current.getModelTypeBadgeVariant("xgboost")).toBe("outline");
  });

  it("getModelTypeBadgeVariantが未知のタイプにoutlineを返すこと", () => {
    const { result } = renderHook(() => useModelInfo());

    expect(result.current.getModelTypeBadgeVariant("unknown")).toBe("outline");
  });

  it("getModelTypeBadgeVariantがundefinedにoutlineを返すこと", () => {
    const { result } = renderHook(() => useModelInfo());

    expect(result.current.getModelTypeBadgeVariant(undefined)).toBe("outline");
  });

  it("getAccuracyBadgeVariantが0.8以上にsuccessを返すこと", () => {
    const { result } = renderHook(() => useModelInfo());

    expect(result.current.getAccuracyBadgeVariant(0.85)).toBe("success");
  });

  it("getAccuracyBadgeVariantが0.7以上0.8未満にwarningを返すこと", () => {
    const { result } = renderHook(() => useModelInfo());

    expect(result.current.getAccuracyBadgeVariant(0.75)).toBe("warning");
  });

  it("getAccuracyBadgeVariantが0.7未満にdestructiveを返すこと", () => {
    const { result } = renderHook(() => useModelInfo());

    expect(result.current.getAccuracyBadgeVariant(0.5)).toBe("destructive");
  });

  it("getAccuracyBadgeVariantがundefinedにoutlineを返すこと", () => {
    const { result } = renderHook(() => useModelInfo());

    expect(result.current.getAccuracyBadgeVariant(undefined)).toBe("outline");
  });
});
