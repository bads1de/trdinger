/**
 * 通貨ペア一覧API テスト
 *
 * /api/data/symbols エンドポイントのテストケースです。
 * 通貨ペア一覧の取得機能をテストします。
 *
 * @author Trdinger Development Team
 * @version 1.0.0
 */

import { GET } from "@/app/api/data/symbols/route";

describe("/api/data/symbols", () => {
  describe("正常系テスト", () => {
    test("通貨ペア一覧を正常に取得できる", async () => {
      const response = await GET();
      const data = await response.json();

      expect(response.status).toBe(200);
      expect(data.success).toBe(true);
      expect(data.data).toBeInstanceOf(Array);
      expect(data.data.length).toBeGreaterThan(0);
      expect(data.message).toBeDefined();
      expect(data.timestamp).toBeDefined();
    });

    test("各通貨ペアが正しい形式を持つ", async () => {
      const response = await GET();
      const data = await response.json();

      expect(response.status).toBe(200);

      data.data.forEach((pair: any) => {
        expect(pair).toHaveProperty("symbol");
        expect(pair).toHaveProperty("name");
        expect(pair).toHaveProperty("base");
        expect(pair).toHaveProperty("quote");

        // 文字列型の検証
        expect(typeof pair.symbol).toBe("string");
        expect(typeof pair.name).toBe("string");
        expect(typeof pair.base).toBe("string");
        expect(typeof pair.quote).toBe("string");

        // シンボル形式の検証（BASE/QUOTE形式またはBASEQUOTE形式）
        expect(pair.symbol).toMatch(/^[A-Z]+(\/[A-Z]+|[A-Z]+)$/);

        // ベース通貨とクォート通貨がシンボルと一致することを確認
        if (pair.symbol.includes("/")) {
          const [expectedBase, expectedQuote] = pair.symbol.split("/");
          expect(pair.base).toBe(expectedBase);
          expect(pair.quote).toBe(expectedQuote);
        } else {
          // BTCUSDのような形式の場合
          expect(pair.symbol).toBe(pair.base + pair.quote);
        }
      });
    });

    test("主要な通貨ペアが含まれている", async () => {
      const response = await GET();
      const data = await response.json();

      expect(response.status).toBe(200);

      const symbols = data.data.map((pair: any) => pair.symbol);

      // 主要な通貨ペアが含まれていることを確認
      expect(symbols).toContain("BTC/USD");
      expect(symbols).toContain("ETH/USD");
      expect(symbols).toContain("BTC/USDT");
      expect(symbols).toContain("ETH/USDT");
      expect(symbols).toContain("ETH/BTC");
      expect(symbols).toContain("BTCUSD");
      expect(symbols).toContain("ETHUSD");
    });

    test("重複する通貨ペアがない", async () => {
      const response = await GET();
      const data = await response.json();

      expect(response.status).toBe(200);

      const symbols = data.data.map((pair: any) => pair.symbol);
      const uniqueSymbols = [...new Set(symbols)];

      expect(symbols.length).toBe(uniqueSymbols.length);
    });

    test("レスポンス形式が正しい", async () => {
      const response = await GET();
      const data = await response.json();

      expect(response.status).toBe(200);
      expect(data).toHaveProperty("success");
      expect(data).toHaveProperty("data");
      expect(data).toHaveProperty("message");
      expect(data).toHaveProperty("timestamp");

      expect(data.success).toBe(true);
      expect(typeof data.message).toBe("string");
      expect(typeof data.timestamp).toBe("string");

      // タイムスタンプがISO形式であることを確認
      expect(new Date(data.timestamp)).toBeInstanceOf(Date);
      expect(new Date(data.timestamp).getTime()).not.toBeNaN();
    });
  });

  describe("データ整合性テスト", () => {
    test("サポートされているクォート通貨が正しい", async () => {
      const response = await GET();
      const data = await response.json();

      expect(response.status).toBe(200);

      const validQuotes = ["USD", "USDT", "BTC"];
      data.data.forEach((pair: any) => {
        expect(validQuotes).toContain(pair.quote);
      });
    });

    test("通貨ペア名が適切な形式である", async () => {
      const response = await GET();
      const data = await response.json();

      expect(response.status).toBe(200);

      data.data.forEach((pair: any) => {
        // 名前が "Base Currency / Quote Currency" 形式であることを確認
        expect(pair.name).toMatch(/.+ \/ .+/);
        // USD, USDT, BTCのいずれかが含まれていることを確認
        expect(pair.name).toMatch(/(US Dollar|Tether USD|Bitcoin)/);
      });
    });

    test("ベース通貨が有効な仮想通貨シンボルである", async () => {
      const response = await GET();
      const data = await response.json();

      expect(response.status).toBe(200);

      const validBaseCurrencies = [
        "BTC",
        "ETH",
        "BNB",
        "ADA",
        "SOL",
        "XRP",
        "DOT",
        "AVAX",
        "LTC",
        "UNI",
      ];

      data.data.forEach((pair: any) => {
        expect(validBaseCurrencies).toContain(pair.base);
      });
    });
  });

  describe("パフォーマンステスト", () => {
    test("レスポンス時間が適切である", async () => {
      const startTime = Date.now();
      const response = await GET();
      const endTime = Date.now();

      const responseTime = endTime - startTime;

      expect(response.status).toBe(200);
      // レスポンス時間が1秒以内であることを確認
      expect(responseTime).toBeLessThan(1000);
    });
  });
});
