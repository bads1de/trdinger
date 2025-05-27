/**
 * OHLCV API Route Handler のテスト
 *
 * TDDアプローチ: 失敗するテストから開始
 */

import { POST } from "@/app/api/data/ohlcv/route";

// fetch をモック
global.fetch = jest.fn();

// NextRequest のモック
const createMockRequest = (
  body: string,
  headers: Record<string, string> = {}
) => {
  return {
    json: async () => {
      if (body === "invalid json") {
        throw new Error("Invalid JSON");
      }
      return JSON.parse(body);
    },
    headers: new Map(Object.entries(headers)),
  } as any;
};

describe("/api/data/ohlcv", () => {
  beforeEach(() => {
    jest.clearAllMocks();
  });

  describe("POST リクエスト", () => {
    it("正常なリクエストで成功レスポンスを返す", async () => {
      // バックエンドAPIのモックレスポンス
      (fetch as jest.Mock).mockResolvedValueOnce({
        ok: true,
        json: () =>
          Promise.resolve({
            success: true,
            message: "BTC/USDT 1h の履歴データ収集を開始しました",
            status: "started",
          }),
      });

      const request = createMockRequest(
        JSON.stringify({
          symbol: "BTC/USDT",
          timeframe: "1h",
        }),
        {
          "Content-Type": "application/json",
        }
      );

      const response = await POST(request);
      const data = await response.json();

      expect(response.status).toBe(200);
      expect(data.success).toBe(true);
      expect(data.message).toContain("BTC/USDT");
    });

    it("必須パラメータが不足している場合400エラーを返す", async () => {
      const request = createMockRequest(
        JSON.stringify({
          symbol: "BTC/USDT",
          // timeframe が不足
        }),
        {
          "Content-Type": "application/json",
        }
      );

      const response = await POST(request);
      const data = await response.json();

      expect(response.status).toBe(400);
      expect(data.success).toBe(false);
      expect(data.message).toContain("timeframe");
    });

    it("無効なJSONの場合400エラーを返す", async () => {
      const request = createMockRequest("invalid json", {
        "Content-Type": "application/json",
      });

      const response = await POST(request);
      const data = await response.json();

      expect(response.status).toBe(400);
      expect(data.success).toBe(false);
      expect(data.message).toContain("JSON");
    });

    it("バックエンドAPIエラー時に適切なエラーレスポンスを返す", async () => {
      (fetch as jest.Mock).mockResolvedValueOnce({
        ok: false,
        status: 500,
        json: () =>
          Promise.resolve({
            detail: "Internal server error",
          }),
      });

      const request = createMockRequest(
        JSON.stringify({
          symbol: "BTC/USDT",
          timeframe: "1h",
        }),
        {
          "Content-Type": "application/json",
        }
      );

      const response = await POST(request);
      const data = await response.json();

      expect(response.status).toBe(500);
      expect(data.success).toBe(false);
      expect(data.message).toContain("バックエンドAPI");
    });

    it("ネットワークエラー時に適切なエラーレスポンスを返す", async () => {
      (fetch as jest.Mock).mockRejectedValueOnce(new Error("Network error"));

      const request = createMockRequest(
        JSON.stringify({
          symbol: "BTC/USDT",
          timeframe: "1h",
        }),
        {
          "Content-Type": "application/json",
        }
      );

      const response = await POST(request);
      const data = await response.json();

      expect(response.status).toBe(500);
      expect(data.success).toBe(false);
      expect(data.message).toContain("ネットワークエラー");
    });

    it("正しいバックエンドAPIエンドポイントを呼び出す", async () => {
      (fetch as jest.Mock).mockResolvedValueOnce({
        ok: true,
        json: () =>
          Promise.resolve({
            success: true,
            message: "Success",
          }),
      });

      const request = createMockRequest(
        JSON.stringify({
          symbol: "ETH/USDT",
          timeframe: "4h",
        }),
        {
          "Content-Type": "application/json",
        }
      );

      await POST(request);

      expect(fetch).toHaveBeenCalledWith(
        "http://127.0.0.1:8000/api/data-collection/historical?symbol=ETH%2FUSDT&timeframe=4h",
        {
          method: "POST",
          headers: {
            "Content-Type": "application/json",
          },
        }
      );
    });

    it("データが既に存在する場合の適切なレスポンス", async () => {
      (fetch as jest.Mock).mockResolvedValueOnce({
        ok: true,
        json: () =>
          Promise.resolve({
            success: true,
            message:
              "BTC/USDT 1h のデータは既に存在します。新規収集は行いません。",
            status: "exists",
          }),
      });

      const request = createMockRequest(
        JSON.stringify({
          symbol: "BTC/USDT",
          timeframe: "1h",
        }),
        {
          "Content-Type": "application/json",
        }
      );

      const response = await POST(request);
      const data = await response.json();

      expect(response.status).toBe(200);
      expect(data.success).toBe(true);
      expect(data.status).toBe("exists");
      expect(data.message).toContain("既に存在");
    });
  });
});
