import { renderHook, act } from "@testing-library/react";
import { useIncrementalUpdateHandler } from "@/hooks/useIncrementalUpdateHandler";

jest.mock("@/hooks/useBulkIncrementalUpdate", () => ({
  useBulkIncrementalUpdate: jest.fn(),
}));

import { useBulkIncrementalUpdate } from "@/hooks/useBulkIncrementalUpdate";

const mockUseBulkIncrementalUpdate = useBulkIncrementalUpdate as jest.MockedFunction<
  typeof useBulkIncrementalUpdate
>;

describe("useIncrementalUpdateHandler", () => {
  const mockBulkUpdate = jest.fn();
  const MESSAGE_KEYS = {
    INCREMENTAL_UPDATE: "incremental_update",
  };
  const MESSAGE_DURATION = {
    SHORT: 2000,
    MEDIUM: 5000,
    LONG: 10000,
  };

  const createDefaultDeps = (overrides = {}) => ({
    setMessage: jest.fn(),
    fetchOHLCVData: jest.fn(),
    fetchDataStatus: jest.fn(),
    MESSAGE_KEYS,
    MESSAGE_DURATION,
    ...overrides,
  });

  beforeEach(() => {
    jest.clearAllMocks();
    mockUseBulkIncrementalUpdate.mockReturnValue({
      bulkUpdate: mockBulkUpdate,
      loading: false,
      error: null,
    } as any);
  });

  it("初期状態が正しいこと", () => {
    const deps = createDefaultDeps();
    const { result } = renderHook(() => useIncrementalUpdateHandler(deps));

    expect(typeof result.current.handleBulkIncrementalUpdate).toBe("function");
    expect(result.current.bulkIncrementalUpdateLoading).toBe(false);
    expect(result.current.bulkIncrementalUpdateError).toBeNull();
  });

  it("handleBulkIncrementalUpdateが成功時に正しいメッセージを表示すること", async () => {
    const setMessage = jest.fn();
    const fetchOHLCVData = jest.fn().mockResolvedValue(undefined);
    const fetchDataStatus = jest.fn();
    const deps = createDefaultDeps({
      setMessage,
      fetchOHLCVData,
      fetchDataStatus,
    });

    const { result } = renderHook(() => useIncrementalUpdateHandler(deps));

    mockBulkUpdate.mockImplementation(
      async (_symbol: string, _tf: string, options: any) => {
        options.onSuccess({
          data: {
            total_saved_count: 150,
            data: {
              ohlcv: { saved_count: 100, timeframe_results: null },
              funding_rate: { saved_count: 30 },
              open_interest: { saved_count: 20 },
            },
          },
        });
      }
    );

    await act(async () => {
      await result.current.handleBulkIncrementalUpdate(
        "BTC/USDT:USDT",
        "1h"
      );
    });

    expect(setMessage).toHaveBeenNthCalledWith(
      1,
      MESSAGE_KEYS.INCREMENTAL_UPDATE,
      ""
    );

    expect(setMessage).toHaveBeenNthCalledWith(
      2,
      MESSAGE_KEYS.INCREMENTAL_UPDATE,
      "一括差分更新完了！ BTC/USDT:USDT - 総計150件 (OHLCV:100, FR:30, OI:20)",
      MESSAGE_DURATION.MEDIUM,
      "success"
    );

    expect(fetchOHLCVData).toHaveBeenCalledTimes(1);
    expect(fetchDataStatus).toHaveBeenCalledTimes(1);
  });

  it("handleBulkIncrementalUpdateが成功時にtimeframe_resultsを含めてメッセージ表示すること", async () => {
    const setMessage = jest.fn();
    const deps = createDefaultDeps({ setMessage });

    const { result } = renderHook(() => useIncrementalUpdateHandler(deps));

    mockBulkUpdate.mockImplementation(
      async (_symbol: string, _tf: string, options: any) => {
        options.onSuccess({
          data: {
            total_saved_count: 200,
            data: {
              ohlcv: {
                saved_count: 150,
                timeframe_results: {
                  "1h": { saved_count: 80 },
                  "4h": { saved_count: 40 },
                  "1d": { saved_count: 30 },
                },
              },
              funding_rate: { saved_count: 30 },
              open_interest: { saved_count: 20 },
            },
          },
        });
      }
    );

    await act(async () => {
      await result.current.handleBulkIncrementalUpdate(
        "ETH/USDT:USDT",
        "4h"
      );
    });

    expect(setMessage).toHaveBeenNthCalledWith(
      2,
      MESSAGE_KEYS.INCREMENTAL_UPDATE,
      expect.stringContaining("[1h:80, 4h:40, 1d:30]"),
      expect.anything(),
      expect.anything()
    );
  });

  it("handleBulkIncrementalUpdateがエラー時に正しくメッセージを表示すること", async () => {
    const setMessage = jest.fn();
    const deps = createDefaultDeps({ setMessage });

    const { result } = renderHook(() => useIncrementalUpdateHandler(deps));

    mockBulkUpdate.mockImplementation(
      async (_symbol: string, _tf: string, options: any) => {
        options.onError("エラーが発生しました");
      }
    );

    await act(async () => {
      await result.current.handleBulkIncrementalUpdate(
        "BTC/USDT:USDT",
        "1h"
      );
    });

    expect(setMessage).toHaveBeenNthCalledWith(
      2,
      MESSAGE_KEYS.INCREMENTAL_UPDATE,
      "エラーが発生しました",
      MESSAGE_DURATION.SHORT,
      "error"
    );
  });

  it("loading状態がuseBulkIncrementalUpdateから正しく渡されること", () => {
    mockUseBulkIncrementalUpdate.mockReturnValue({
      bulkUpdate: mockBulkUpdate,
      loading: true,
      error: null,
    } as any);

    const deps = createDefaultDeps();
    const { result } = renderHook(() => useIncrementalUpdateHandler(deps));
    expect(result.current.bulkIncrementalUpdateLoading).toBe(true);
  });

  it("error状態がuseBulkIncrementalUpdateから正しく渡されること", () => {
    mockUseBulkIncrementalUpdate.mockReturnValue({
      bulkUpdate: mockBulkUpdate,
      loading: false,
      error: "API Error",
    } as any);

    const deps = createDefaultDeps();
    const { result } = renderHook(() => useIncrementalUpdateHandler(deps));
    expect(result.current.bulkIncrementalUpdateError).toBe("API Error");
  });
});
