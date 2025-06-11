/**
 * BTC/USDT:USDT限定APIエンドポイントのテスト
 * 
 * TDD: APIがBTC/USDT:USDTの無期限先物のみをサポートすることをテスト
 */

import { NextRequest } from 'next/server';

// テスト用のモック関数
const mockFetch = jest.fn();
global.fetch = mockFetch;

// 環境変数のモック
process.env.BACKEND_API_URL = 'http://127.0.0.1:8000';

describe('BTC/USDT:USDT限定APIエンドポイントテスト', () => {
  
  beforeEach(() => {
    mockFetch.mockClear();
  });

  describe('OHLCVデータ収集API', () => {
    
    test('デフォルトシンボルがBTC/USDT:USDTであること', async () => {
      // このテストは最初は失敗する（現在のデフォルトはBTC/USDTのため）
      mockFetch.mockResolvedValueOnce({
        ok: true,
        json: async () => ({ success: true, data: [] })
      });

      const { POST } = await import('@/app/api/data/ohlcv/collect/route');
      
      // シンボルを指定しないリクエスト
      const request = new NextRequest('http://localhost:3000/api/data/ohlcv/collect', {
        method: 'POST'
      });

      await POST(request);

      // バックエンドAPIが正しいデフォルトシンボルで呼ばれることを確認
      expect(mockFetch).toHaveBeenCalledWith(
        expect.stringContaining('symbol=BTC%2FUSDT%3AUSDT'),
        expect.any(Object)
      );
    });

    test('BTC/USDT:USDTシンボルが正しくエンコードされること', async () => {
      mockFetch.mockResolvedValueOnce({
        ok: true,
        json: async () => ({ success: true, data: [] })
      });

      const { POST } = await import('@/app/api/data/ohlcv/collect/route');
      
      const request = new NextRequest(
        'http://localhost:3000/api/data/ohlcv/collect?symbol=BTC/USDT:USDT', 
        { method: 'POST' }
      );

      await POST(request);

      // URLエンコードされたシンボルでバックエンドAPIが呼ばれることを確認
      expect(mockFetch).toHaveBeenCalledWith(
        expect.stringContaining('symbol=BTC%2FUSDT%3AUSDT'),
        expect.any(Object)
      );
    });

    test('サポートされていないシンボルでエラーが返されること', async () => {
      // このテストは実装後に通るようになる
      const { POST } = await import('@/app/api/data/ohlcv/collect/route');
      
      const request = new NextRequest(
        'http://localhost:3000/api/data/ohlcv/collect?symbol=ETH/USDT', 
        { method: 'POST' }
      );

      const response = await POST(request);
      const data = await response.json();

      expect(response.status).toBe(400);
      expect(data.success).toBe(false);
      expect(data.message).toContain('サポートされていないシンボル');
    });
  });

  describe('Funding Rate収集API', () => {
    
    test('デフォルトシンボルがBTC/USDT:USDTであること', async () => {
      mockFetch.mockResolvedValueOnce({
        ok: true,
        json: async () => ({ success: true, data: [] })
      });

      const { POST } = await import('@/app/api/data/funding-rates/collect/route');
      
      const request = new NextRequest('http://localhost:3000/api/data/funding-rates/collect', {
        method: 'POST'
      });

      await POST(request);

      expect(mockFetch).toHaveBeenCalledWith(
        expect.stringContaining('symbol=BTC%2FUSDT%3AUSDT'),
        expect.any(Object)
      );
    });

    test('BTC現物シンボルでエラーが返されること', async () => {
      const { POST } = await import('@/app/api/data/funding-rates/collect/route');
      
      const request = new NextRequest(
        'http://localhost:3000/api/data/funding-rates/collect?symbol=BTC/USDT', 
        { method: 'POST' }
      );

      const response = await POST(request);
      const data = await response.json();

      expect(response.status).toBe(400);
      expect(data.success).toBe(false);
      expect(data.message).toContain('サポートされていないシンボル');
    });
  });

  describe('Open Interest収集API', () => {
    
    test('デフォルトシンボルがBTC/USDT:USDTであること', async () => {
      mockFetch.mockResolvedValueOnce({
        ok: true,
        json: async () => ({ success: true, data: [] })
      });

      const { POST } = await import('@/app/api/data/open-interest/collect/route');
      
      const request = new NextRequest('http://localhost:3000/api/data/open-interest/collect', {
        method: 'POST'
      });

      await POST(request);

      expect(mockFetch).toHaveBeenCalledWith(
        expect.stringContaining('symbol=BTC%2FUSDT%3AUSDT'),
        expect.any(Object)
      );
    });

    test('BTCUSD無期限先物でエラーが返されること', async () => {
      const { POST } = await import('@/app/api/data/open-interest/collect/route');
      
      const request = new NextRequest(
        'http://localhost:3000/api/data/open-interest/collect?symbol=BTCUSD', 
        { method: 'POST' }
      );

      const response = await POST(request);
      const data = await response.json();

      expect(response.status).toBe(400);
      expect(data.success).toBe(false);
      expect(data.message).toContain('サポートされていないシンボル');
    });
  });

  describe('一括収集API', () => {
    
    test('一括収集がBTC/USDT:USDTのみを対象とすること', async () => {
      mockFetch.mockResolvedValueOnce({
        ok: true,
        json: async () => ({ 
          success: true, 
          data: {
            collected_symbols: ['BTC/USDT:USDT'],
            collected_timeframes: ['15m', '30m', '1h', '4h', '1d']
          }
        })
      });

      const { POST } = await import('@/app/api/data/all/bulk-collect/route');
      
      const request = new NextRequest('http://localhost:3000/api/data/all/bulk-collect', {
        method: 'POST'
      });

      const response = await POST(request);
      const data = await response.json();

      expect(response.status).toBe(200);
      expect(data.success).toBe(true);
      
      // バックエンドAPIが呼ばれることを確認
      expect(mockFetch).toHaveBeenCalledWith(
        expect.stringContaining('/api/data-collection/all/bulk-collect'),
        expect.objectContaining({
          method: 'POST',
          headers: expect.objectContaining({
            'Content-Type': 'application/json'
          })
        })
      );
    });
  });

  describe('データ取得API', () => {
    
    test('candlesticksエンドポイントでBTC/USDT:USDTが取得できること', async () => {
      mockFetch.mockResolvedValueOnce({
        ok: true,
        json: async () => ({ 
          success: true, 
          data: {
            symbol: 'BTC/USDT:USDT',
            timeframe: '1h',
            ohlcv: []
          }
        })
      });

      const { GET } = await import('@/app/api/data/candlesticks/route');
      
      const request = new NextRequest(
        'http://localhost:3000/api/data/candlesticks?symbol=BTC/USDT:USDT&timeframe=1h&limit=100', 
        { method: 'GET' }
      );

      const response = await GET(request);
      const data = await response.json();

      expect(response.status).toBe(200);
      expect(data.success).toBe(true);
      expect(data.data.symbol).toBe('BTC/USDT:USDT');
    });

    test('サポートされていない時間足でエラーが返されること', async () => {
      const { GET } = await import('@/app/api/data/candlesticks/route');
      
      const request = new NextRequest(
        'http://localhost:3000/api/data/candlesticks?symbol=BTC/USDT:USDT&timeframe=1m&limit=100', 
        { method: 'GET' }
      );

      const response = await GET(request);
      const data = await response.json();

      expect(response.status).toBe(400);
      expect(data.success).toBe(false);
      expect(data.message).toContain('サポートされていない時間足');
    });
  });

  describe('シンボル一覧API', () => {
    
    test('シンボル一覧がBTC/USDT:USDTのみを返すこと', async () => {
      mockFetch.mockResolvedValueOnce({
        ok: true,
        json: async () => ({ 
          success: true, 
          data: {
            symbols: [
              {
                symbol: 'BTC/USDT:USDT',
                name: 'Bitcoin / USDT Perpetual',
                base: 'BTC',
                quote: 'USDT'
              }
            ]
          }
        })
      });

      const { GET } = await import('@/app/api/data/symbols/route');
      
      const request = new NextRequest('http://localhost:3000/api/data/symbols', {
        method: 'GET'
      });

      const response = await GET(request);
      const data = await response.json();

      expect(response.status).toBe(200);
      expect(data.success).toBe(true);
      expect(data.data.symbols).toHaveLength(1);
      expect(data.data.symbols[0].symbol).toBe('BTC/USDT:USDT');
    });
  });
});

describe('APIバリデーション関数テスト', () => {
  
  test('validateSymbol関数がBTC/USDT:USDTのみを受け入れること', () => {
    // この関数は実装後にテストする
    const validateSymbol = (symbol: string): boolean => {
      const supportedSymbols = ['BTC/USDT:USDT'];
      return supportedSymbols.includes(symbol);
    };

    expect(validateSymbol('BTC/USDT:USDT')).toBe(true);
    expect(validateSymbol('BTC/USDT')).toBe(false);
    expect(validateSymbol('BTCUSD')).toBe(false);
    expect(validateSymbol('ETH/USDT:USDT')).toBe(false);
  });

  test('validateTimeframe関数が要求された時間足のみを受け入れること', () => {
    const validateTimeframe = (timeframe: string): boolean => {
      const supportedTimeframes = ['15m', '30m', '1h', '4h', '1d'];
      return supportedTimeframes.includes(timeframe);
    };

    expect(validateTimeframe('15m')).toBe(true);
    expect(validateTimeframe('30m')).toBe(true);
    expect(validateTimeframe('1h')).toBe(true);
    expect(validateTimeframe('4h')).toBe(true);
    expect(validateTimeframe('1d')).toBe(true);
    
    expect(validateTimeframe('1m')).toBe(false);
    expect(validateTimeframe('5m')).toBe(false);
    expect(validateTimeframe('12h')).toBe(false);
    expect(validateTimeframe('1w')).toBe(false);
  });
});
