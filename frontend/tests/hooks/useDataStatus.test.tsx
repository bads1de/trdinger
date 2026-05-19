import { renderHook, act } from "@testing-library/react";
import { useDataStatus } from "@/hooks/useDataStatus";

jest.mock("@/hooks/useApiCall", () => ({
  useApiCall: jest.fn(),
}));

import { useApiCall } from "@/hooks/useApiCall";

const mockUseApiCall = useApiCall as jest.MockedFunction<typeof useApiCall>;

describe("useDataStatus", () => {
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
    const { result } = renderHook(() => useDataStatus());

    expect(result.current.dataStatus).toBeNull();
    expect(result.current.dataStatusLoading).toBe(false);
    expect(result.current.dataStatusError).toBeNull();
    expect(typeof result.current.fetchDataStatus).toBe("function");
  });

  it("マウント時にデータステータスを取得すること", () => {
    renderHook(() => useDataStatus());

    expect(mockExecute).toHaveBeenCalledWith(
      expect.stringContaining("/api/data-reset/status"),
      expect.objectContaining({
        onSuccess: expect.any(Function),
        onError: expect.any(Function),
      })
    );
  });

  it("fetchDataStatusを呼び出すとデータを再取得すること", () => {
    const { result } = renderHook(() => useDataStatus());

    act(() => {
      result.current.fetchDataStatus();
    });

    expect(mockExecute).toHaveBeenCalled();
  });
});
