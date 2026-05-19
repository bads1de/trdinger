import { renderHook } from "@testing-library/react";
import { useModelPerformance } from "@/hooks/useModelPerformance";

jest.mock("@/hooks/useDataFetching", () => ({
  useDataFetching: jest.fn(),
}));

import { useDataFetching } from "@/hooks/useDataFetching";

const mockUseDataFetching = useDataFetching as jest.MockedFunction<typeof useDataFetching>;

describe("useModelPerformance", () => {
  const mockRefetch = jest.fn();

  beforeEach(() => {
    jest.clearAllMocks();
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

  it("初期状態が正しいこと", () => {
    const { result } = renderHook(() => useModelPerformance());

    expect(result.current.modelStatus).toBeNull();
    expect(result.current.loading).toBe(false);
    expect(result.current.error).toBeNull();
    expect(typeof result.current.loadModelStatus).toBe("function");
    expect(typeof result.current.getScoreBadgeVariant).toBe("function");
    expect(typeof result.current.getStatusBadgeVariant).toBe("function");
  });

  it("getScoreBadgeVariantが0.8以上にsuccessを返すこと", () => {
    const { result } = renderHook(() => useModelPerformance());

    expect(result.current.getScoreBadgeVariant(0.85)).toBe("success");
  });

  it("getScoreBadgeVariantが0.7以上0.8未満にwarningを返すこと", () => {
    const { result } = renderHook(() => useModelPerformance());

    expect(result.current.getScoreBadgeVariant(0.75)).toBe("warning");
  });

  it("getScoreBadgeVariantが0.7未満にdestructiveを返すこと", () => {
    const { result } = renderHook(() => useModelPerformance());

    expect(result.current.getScoreBadgeVariant(0.5)).toBe("destructive");
  });

  it("getScoreBadgeVariantがundefinedにoutlineを返すこと", () => {
    const { result } = renderHook(() => useModelPerformance());

    expect(result.current.getScoreBadgeVariant(undefined)).toBe("outline");
  });

  it("getStatusBadgeVariantがトレーニング中にdefaultを返すこと", () => {
    mockUseDataFetching.mockReturnValue({
      data: [{ is_model_loaded: true, is_trained: true, is_training: true, feature_count: 10 }],
      loading: false,
      error: null,
      refetch: mockRefetch,
      params: {},
      setParams: jest.fn(),
      reset: jest.fn(),
      setData: jest.fn(),
    });

    const { result } = renderHook(() => useModelPerformance());

    expect(result.current.getStatusBadgeVariant()).toBe("default");
  });

  it("getStatusBadgeVariantがモデル読み込み済みにsuccessを返すこと", () => {
    mockUseDataFetching.mockReturnValue({
      data: [{ is_model_loaded: true, is_trained: true, is_training: false, feature_count: 10 }],
      loading: false,
      error: null,
      refetch: mockRefetch,
      params: {},
      setParams: jest.fn(),
      reset: jest.fn(),
      setData: jest.fn(),
    });

    const { result } = renderHook(() => useModelPerformance());

    expect(result.current.getStatusBadgeVariant()).toBe("success");
  });

  it("getStatusBadgeVariantがモデル未読み込みにoutlineを返すこと", () => {
    const { result } = renderHook(() => useModelPerformance());

    expect(result.current.getStatusBadgeVariant()).toBe("outline");
  });
});
