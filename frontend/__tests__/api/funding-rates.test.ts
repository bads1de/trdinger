/**
 * FRデータAPI テスト
 *
 * /api/data/funding-rates エンドポイントのテストケースです。
 * 正常系・異常系の両方をテストします。
 *
 * @author Trdinger Development Team
 * @version 1.0.0
 */

import { GET } from "@/app/api/data/funding-rates/route";

// データベース関数をモック
jest.mock("@/lib/database/funding-rates", () => ({
  fetchDatabaseFundingRateData: jest.fn(),
}));

// import { fetchDatabaseFundingRateData } from "@/lib/database/funding-rates"; // 削除済み

// モックのNextRequestを作成するヘルパー関数
function createMockRequest(searchParams: Record<string, string>) {
  const url = new URL("http://localhost:3000/api/data/funding-rates");
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

describe("/api/data/funding-rates", () => {
  beforeEach(() => {
    // モック関数をリセット
    jest.clearAllMocks();

    // デフォルトのモックデータを設定（削除済み）
    // (fetchDatabaseFundingRateData as jest.Mock).mockResolvedValue([
    //   {
    //     symbol: "BTC/USDT",
    //     funding_rate: 0.0001,
    //     funding_timestamp: "2024-01-01T00:00:00Z",
    //     mark_price: 50000,
    //     index_price: 50001,
    //     next_funding_timestamp: "2024-01-01T08:00:00Z",
    //   },
    // ]);
  });

  describe("正常系テスト", () => {
    test("有効なパラメータでFRデータを取得できる", async () => {
      const request = createMockRequest({
        symbol: "BTC/USDT",
        limit: "50",
      });

      const response = await GET(request);
      const data = await response.json();

      expect(response.status).toBe(200);
      expect(data.success).toBe(true);
      expect(data.data).toBeDefined();
      expect(data.data.symbol).toBe("BTC/USDT");
      expect(data.data.funding_rates).toBeInstanceOf(Array);
      expect(data.data.funding_rates).toHaveLength(50);
      expect(data.data.count).toBe(50);
    });

    test("デフォルトのlimit値（100）でデータを取得できる", async () => {
      const request = createMockRequest({
        symbol: "ETH/USDT",
      });

      const response = await GET(request);
      const data = await response.json();

      expect(response.status).toBe(200);
      expect(data.success).toBe(true);
      expect(data.data.funding_rates).toHaveLength(100);
    });

    test("各FRデータが正しい形式を持つ", async () => {
      const request = createMockRequest({
        symbol: "BTC/USDT",
        limit: "5",
      });

      const response = await GET(request);
      const data = await response.json();

      expect(response.status).toBe(200);

      const fundingRate = data.data.funding_rates[0];
      expect(fundingRate).toHaveProperty("symbol");
      expect(fundingRate).toHaveProperty("funding_rate");
      expect(fundingRate).toHaveProperty("funding_timestamp");
      expect(fundingRate).toHaveProperty("timestamp");

      // 数値型の検証
      expect(typeof fundingRate.funding_rate).toBe("number");

      // 日時形式の検証
      expect(new Date(fundingRate.funding_timestamp)).toBeInstanceOf(Date);
      expect(new Date(fundingRate.funding_timestamp).getTime()).not.toBeNaN();
      expect(new Date(fundingRate.timestamp)).toBeInstanceOf(Date);
      expect(new Date(fundingRate.timestamp).getTime()).not.toBeNaN();
    });

    test("日付範囲指定でデータを取得できる", async () => {
      const startDate = "2024-01-01T00:00:00Z";
      const endDate = "2024-01-02T00:00:00Z";

      const request = createMockRequest({
        symbol: "BTC/USDT",
        start_date: startDate,
        end_date: endDate,
        limit: "10",
      });

      const response = await GET(request);
      const data = await response.json();

      expect(response.status).toBe(200);
      expect(data.success).toBe(true);
      expect(data.data.funding_rates).toBeInstanceOf(Array);
    });
  });

  describe("異常系テスト", () => {
    test("limitが範囲外の場合はエラーを返す", async () => {
      // limit が 0 の場合
      let request = createMockRequest({
        symbol: "BTC/USDT",
        limit: "0",
      });

      let response = await GET(request);
      let data = await response.json();

      expect(response.status).toBe(400);
      expect(data.success).toBe(false);
      expect(data.message).toContain(
        "limitパラメータは1から1000の間で指定してください"
      );

      // limit が 1001 の場合
      request = createMockRequest({
        symbol: "BTC/USDT",
        limit: "1001",
      });

      response = await GET(request);
      data = await response.json();

      expect(response.status).toBe(400);
      expect(data.success).toBe(false);
      expect(data.message).toContain(
        "limitパラメータは1から1000の間で指定してください"
      );
    });

    test("データが見つからない場合は404エラーを返す", async () => {
      const request = createMockRequest({
        symbol: "NONEXISTENT/USDT",
        limit: "10",
      });

      const response = await GET(request);
      const data = await response.json();

      expect(response.status).toBe(404);
      expect(data.success).toBe(false);
      expect(data.message).toContain("データが見つかりません");
    });
  });

  describe("データ整合性テスト", () => {
    test("FRが数値範囲内にある", async () => {
      const request = createMockRequest({
        symbol: "BTC/USDT",
        limit: "20",
      });

      const response = await GET(request);
      const data = await response.json();

      expect(response.status).toBe(200);

      data.data.funding_rates.forEach((rate: any) => {
        // FRは通常 -1% から +1% の範囲内
        expect(rate.funding_rate).toBeGreaterThanOrEqual(-0.01);
        expect(rate.funding_rate).toBeLessThanOrEqual(0.01);
      });
    });

    test("タイムスタンプが時系列順になっている", async () => {
      const request = createMockRequest({
        symbol: "BTC/USDT",
        limit: "10",
      });

      const response = await GET(request);
      const data = await response.json();

      expect(response.status).toBe(200);

      const timestamps = data.data.funding_rates.map((rate: any) =>
        new Date(rate.funding_timestamp).getTime()
      );

      for (let i = 1; i < timestamps.length; i++) {
        expect(timestamps[i]).toBeGreaterThan(timestamps[i - 1]);
      }
    });
  });
});
