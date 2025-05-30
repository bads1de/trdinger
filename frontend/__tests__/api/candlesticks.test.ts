/**
 * OHLCVデータAPI テスト
 *
 * /api/data/candlesticks エンドポイントのテストケースです。
 * 正常系・異常系の両方をテストします。
 *
 * @author Trdinger Development Team
 * @version 2.0.0
 */



import { GET } from "@/app/api/data/candlesticks/route";

// データベース関数をモック
jest.mock("@/lib/database/ohlcv", () => ({
  fetchDatabaseOHLCVData: jest.fn(),
}));

import { fetchDatabaseOHLCVData } from "@/lib/database/ohlcv";

// モックのNextRequestを作成するヘルパー関数
function createMockRequest(searchParams: Record<string, string>) {
  const url = new URL("http://localhost:3000/api/data/candlesticks");
  Object.entries(searchParams).forEach(([key, value]) => {
    url.searchParams.set(key, value);
  });

  return {
    url: url.toString(),
    nextUrl: {
      searchParams: url.searchParams,
    },
  } as any;
}

describe("/api/data/candlesticks", () => {
  beforeEach(() => {
    // モック関数をリセット
    jest.clearAllMocks();

    // デフォルトのモックデータを設定
    (fetchDatabaseOHLCVData as jest.Mock).mockResolvedValue([
      {
        timestamp: "2024-01-01T00:00:00Z",
        open: 50000,
        high: 51000,
        low: 49000,
        close: 50500,
        volume: 1000000,
      },
    ]);
  });

  describe("正常系テスト", () => {
    test("有効なパラメータでOHLCVデータを取得できる", async () => {
      const request = createMockRequest({
        symbol: "BTC/USD",
        timeframe: "1d",
        limit: "50",
      });

      const response = await GET(request);
      const data = await response.json();

      expect(response.status).toBe(200);
      expect(data.success).toBe(true);
      expect(data.data).toBeDefined();
      expect(data.data.symbol).toBe("BTC/USD");
      expect(data.data.timeframe).toBe("1d");
      expect(data.data.ohlcv).toBeInstanceOf(Array);
      expect(data.data.ohlcv).toHaveLength(50);
      expect(data.timestamp).toBeDefined();
    });

    test("デフォルトのlimit値（100）でデータを取得できる", async () => {
      const request = createMockRequest({
        symbol: "ETH/USD",
        timeframe: "1h",
      });

      const response = await GET(request);
      const data = await response.json();

      expect(response.status).toBe(200);
      expect(data.success).toBe(true);
      expect(data.data.ohlcv).toHaveLength(100);
    });

    test("各OHLCVデータが正しい形式を持つ", async () => {
      const request = createMockRequest({
        symbol: "BTC/USD",
        timeframe: "1d",
        limit: "5",
      });

      const response = await GET(request);
      const data = await response.json();

      expect(response.status).toBe(200);

      const ohlcv = data.data.ohlcv[0];
      expect(ohlcv).toHaveProperty("timestamp");
      expect(ohlcv).toHaveProperty("open");
      expect(ohlcv).toHaveProperty("high");
      expect(ohlcv).toHaveProperty("low");
      expect(ohlcv).toHaveProperty("close");
      expect(ohlcv).toHaveProperty("volume");

      // 数値型の検証
      expect(typeof ohlcv.open).toBe("number");
      expect(typeof ohlcv.high).toBe("number");
      expect(typeof ohlcv.low).toBe("number");
      expect(typeof ohlcv.close).toBe("number");
      expect(typeof ohlcv.volume).toBe("number");

      // 日時形式の検証
      expect(new Date(ohlcv.timestamp)).toBeInstanceOf(Date);
      expect(new Date(ohlcv.timestamp).getTime()).not.toBeNaN();
    });

    test("すべての利用可能な時間軸でデータを取得できる", async () => {
      const timeframes = ["15m", "30m", "1h", "4h", "1d"];

      for (const timeframe of timeframes) {
        const request = createMockRequest({
          symbol: "BTC/USD",
          timeframe,
          limit: "10",
        });

        const response = await GET(request);
        const data = await response.json();

        expect(response.status).toBe(200);
        expect(data.success).toBe(true);
        expect(data.data.timeframe).toBe(timeframe);
      }
    });
  });

  describe("異常系テスト", () => {
    test("symbolパラメータが未指定の場合はエラーを返す", async () => {
      const request = createMockRequest({
        timeframe: "1d",
      });

      const response = await GET(request);
      const data = await response.json();

      expect(response.status).toBe(400);
      expect(data.success).toBe(false);
      expect(data.message).toContain("symbol パラメータは必須です");
    });

    test("timeframeパラメータが未指定の場合はエラーを返す", async () => {
      const request = createMockRequest({
        symbol: "BTC/USD",
      });

      const response = await GET(request);
      const data = await response.json();

      expect(response.status).toBe(400);
      expect(data.success).toBe(false);
      expect(data.message).toContain("timeframe パラメータは必須です");
    });

    test("サポートされていない通貨ペアの場合はエラーを返す", async () => {
      const request = createMockRequest({
        symbol: "INVALID/PAIR",
        timeframe: "1d",
      });

      const response = await GET(request);
      const data = await response.json();

      expect(response.status).toBe(400);
      expect(data.success).toBe(false);
      expect(data.message).toContain("サポートされていない通貨ペアです");
    });

    test("サポートされていない時間軸の場合はエラーを返す", async () => {
      const request = createMockRequest({
        symbol: "BTC/USD",
        timeframe: "invalid",
      });

      const response = await GET(request);
      const data = await response.json();

      expect(response.status).toBe(400);
      expect(data.success).toBe(false);
      expect(data.message).toContain("サポートされていない時間軸です");
    });

    test("limitが範囲外の場合はエラーを返す", async () => {
      // limit が 0 の場合
      let request = createMockRequest({
        symbol: "BTC/USD",
        timeframe: "1d",
        limit: "0",
      });

      let response = await GET(request);
      let data = await response.json();

      expect(response.status).toBe(400);
      expect(data.success).toBe(false);
      expect(data.message).toContain(
        "limit は 1 から 1000 の間で指定してください"
      );

      // limit が 1001 の場合
      request = createMockRequest({
        symbol: "BTC/USD",
        timeframe: "1d",
        limit: "1001",
      });

      response = await GET(request);
      data = await response.json();

      expect(response.status).toBe(400);
      expect(data.success).toBe(false);
      expect(data.message).toContain(
        "limit は 1 から 1000 の間で指定してください"
      );
    });
  });

  describe("データ整合性テスト", () => {
    test("OHLCVデータの価格関係が正しい（high >= low, high >= open, high >= close, low <= open, low <= close）", async () => {
      const request = createMockRequest({
        symbol: "BTC/USD",
        timeframe: "1d",
        limit: "20",
      });

      const response = await GET(request);
      const data = await response.json();

      expect(response.status).toBe(200);

      data.data.ohlcv.forEach((ohlcv: any) => {
        expect(ohlcv.high).toBeGreaterThanOrEqual(ohlcv.low);
        expect(ohlcv.high).toBeGreaterThanOrEqual(ohlcv.open);
        expect(ohlcv.high).toBeGreaterThanOrEqual(ohlcv.close);
        expect(ohlcv.low).toBeLessThanOrEqual(ohlcv.open);
        expect(ohlcv.low).toBeLessThanOrEqual(ohlcv.close);
      });
    });

    test("タイムスタンプが時系列順になっている", async () => {
      const request = createMockRequest({
        symbol: "BTC/USD",
        timeframe: "1h",
        limit: "10",
      });

      const response = await GET(request);
      const data = await response.json();

      expect(response.status).toBe(200);

      const timestamps = data.data.ohlcv.map((ohlcv: any) =>
        new Date(ohlcv.timestamp).getTime()
      );

      for (let i = 1; i < timestamps.length; i++) {
        expect(timestamps[i]).toBeGreaterThan(timestamps[i - 1]);
      }
    });
  });
});
