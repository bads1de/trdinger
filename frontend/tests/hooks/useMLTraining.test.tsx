import { renderHook, act } from "@testing-library/react";
import { useMLTraining } from "@/hooks/useMLTraining";
import { useApiCall } from "@/hooks/useApiCall";

// useApiCallをモック化
jest.mock("@/hooks/useApiCall", () => ({
  useApiCall: jest.fn(),
}));

describe("useMLTraining", () => {
  const mockExecute = jest.fn();
  const mockReset = jest.fn();

  beforeEach(() => {
    jest.clearAllMocks();
    jest.useFakeTimers();

    // useApiCallが常に有効なモックAPIオブジェクトを返すように設定
    (useApiCall as jest.Mock).mockReturnValue({
      execute: mockExecute,
      loading: false,
      error: null,
      reset: mockReset,
    });
  });

  afterEach(() => {
    jest.useRealTimers();
  });

  it("初期設定が正しいこと", () => {
    const { result } = renderHook(() => useMLTraining());

    expect(result.current.config.symbol).toBe("BTC/USDT:USDT");
    expect(result.current.trainingStatus.is_training).toBe(false);
    expect(result.current.availableModels.length).toBeGreaterThan(0); // 定数から読み込まれるはず
  });

  it("設定の更新ができること", () => {
    const { result } = renderHook(() => useMLTraining());
    const newConfig = { ...result.current.config, timeframe: "4h" };

    act(() => {
      result.current.setConfig(newConfig);
    });

    expect(result.current.config.timeframe).toBe("4h");
  });

  it("トレーニング開始が正しく動作すること", async () => {
    const { result } = renderHook(() => useMLTraining());

    mockExecute.mockImplementation(async (_, options) => {
      options?.onSuccess?.();
      return {};
    });

    await act(async () => {
      await result.current.startTraining();
    });

    expect(mockExecute).toHaveBeenCalledWith(
      "/api/ml-training/train",
      expect.objectContaining({
        method: "POST",
        body: expect.anything(),
      })
    );

    expect(result.current.trainingStatus.is_training).toBe(true);
    expect(result.current.trainingStatus.status).toBe("starting");
  });

  it("トレーニング停止が正しく動作すること", async () => {
    const { result } = renderHook(() => useMLTraining());

    mockExecute.mockImplementation(async (_, options) => {
      options?.onSuccess?.();
      return {};
    });

    await act(async () => {
      await result.current.stopTraining();
    });

    expect(mockExecute).toHaveBeenCalledWith(
      "/api/ml-training/stop",
      expect.objectContaining({
        method: "POST",
      })
    );

    expect(result.current.trainingStatus.is_training).toBe(false);
    expect(result.current.trainingStatus.status).toBe("stopped");
  });

  it("トレーニング中のポーリングが動作すること", async () => {
    const { result } = renderHook(() => useMLTraining());

    // 1. トレーニングを開始して is_training = true にする
    mockExecute.mockImplementation(async (url, options) => {
      if (url === "/api/ml-training/train") {
        options?.onSuccess?.();
      } else if (url === "/api/ml-training/training/status") {
        options?.onSuccess?.({
          is_training: true,
          status: "running",
          progress: 50,
          message: "Training...",
        });
      }
      return {};
    });

    await act(async () => {
      await result.current.startTraining();
    });

    // 2. タイマーを進める
    act(() => {
      jest.advanceTimersByTime(2000);
    });

    // 3. ポーリングによる status call を確認
    expect(mockExecute).toHaveBeenCalledWith(
      "/api/ml-training/training/status",
      expect.anything()
    );

    // ステータスが更新されているか確認
    expect(result.current.trainingStatus.progress).toBe(50);
  });
});
