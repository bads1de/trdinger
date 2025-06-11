/**
 * BTC/USDT:USDT限定設定のテスト
 * 
 * TDD: フロントエンドがBTC/USDT:USDTの無期限先物のみをサポートすることをテスト
 */

import { 
  SUPPORTED_TRADING_PAIRS, 
  SUPPORTED_TIMEFRAMES, 
  DEFAULT_TRADING_PAIR,
  DEFAULT_TIMEFRAME 
} from '@/constants';

describe('BTC/USDT:USDT限定設定テスト', () => {
  
  test('サポートされる取引ペアがBTC/USDT:USDTのみであること', () => {
    // このテストは最初は失敗する（現在は複数のシンボルをサポートしているため）
    expect(SUPPORTED_TRADING_PAIRS).toHaveLength(1);
    expect(SUPPORTED_TRADING_PAIRS[0].symbol).toBe('BTC/USDT:USDT');
    expect(SUPPORTED_TRADING_PAIRS[0].name).toContain('Perpetual');
  });

  test('デフォルト取引ペアがBTC/USDT:USDTであること', () => {
    // このテストは最初は失敗する可能性がある
    expect(DEFAULT_TRADING_PAIR).toBe('BTC/USDT:USDT');
  });

  test('サポートされる時間足に要求された時間足が含まれること', () => {
    const requiredTimeframes = ['1d', '4h', '1h', '30m', '15m'];
    
    for (const timeframe of requiredTimeframes) {
      const found = SUPPORTED_TIMEFRAMES.find(tf => tf.value === timeframe);
      expect(found).toBeDefined();
      expect(found?.value).toBe(timeframe);
    }
  });

  test('サポートされる時間足が要求された5つのみであること', () => {
    const expectedTimeframes = ['15m', '30m', '1h', '4h', '1d'];
    
    expect(SUPPORTED_TIMEFRAMES).toHaveLength(5);
    
    for (const expectedTf of expectedTimeframes) {
      const found = SUPPORTED_TIMEFRAMES.find(tf => tf.value === expectedTf);
      expect(found).toBeDefined();
    }
  });

  test('1分足がサポートされていないこと', () => {
    // 1分足は要求されていないため、サポートされていないはず
    const oneMinute = SUPPORTED_TIMEFRAMES.find(tf => tf.value === '1m');
    expect(oneMinute).toBeUndefined();
  });

  test('BTC現物取引ペアがサポートされていないこと', () => {
    // BTC/USDTはサポートされていないはず
    const spotPair = SUPPORTED_TRADING_PAIRS.find(pair => pair.symbol === 'BTC/USDT');
    expect(spotPair).toBeUndefined();
  });

  test('BTCUSD無期限先物がサポートされていないこと', () => {
    // BTCUSDはサポートされていないはず
    const usdPair = SUPPORTED_TRADING_PAIRS.find(pair => pair.symbol === 'BTCUSD');
    expect(usdPair).toBeUndefined();
  });

  test('ETH関連の取引ペアがサポートされていないこと', () => {
    // ETH関連のペアは全て削除されているはず
    const ethPairs = SUPPORTED_TRADING_PAIRS.filter(pair => 
      pair.symbol.includes('ETH') || pair.base === 'ETH'
    );
    expect(ethPairs).toHaveLength(0);
  });

  test('BTC/USDT:USDTペアの詳細情報が正しいこと', () => {
    const btcUsdtPerp = SUPPORTED_TRADING_PAIRS.find(pair => pair.symbol === 'BTC/USDT:USDT');
    
    expect(btcUsdtPerp).toBeDefined();
    expect(btcUsdtPerp?.base).toBe('BTC');
    expect(btcUsdtPerp?.quote).toBe('USDT');
    expect(btcUsdtPerp?.name).toContain('Bitcoin');
    expect(btcUsdtPerp?.name).toContain('Perpetual');
  });

  test('デフォルト時間足が1時間であること', () => {
    expect(DEFAULT_TIMEFRAME).toBe('1h');
  });
});

describe('時間足表示情報テスト', () => {
  
  test('各時間足に適切な日本語ラベルが設定されていること', () => {
    const expectedLabels = {
      '15m': '15分',
      '30m': '30分', 
      '1h': '1時間',
      '4h': '4時間',
      '1d': '1日'
    };

    for (const [value, expectedLabel] of Object.entries(expectedLabels)) {
      const timeframe = SUPPORTED_TIMEFRAMES.find(tf => tf.value === value);
      expect(timeframe?.label).toBe(expectedLabel);
    }
  });

  test('各時間足に説明が設定されていること', () => {
    for (const timeframe of SUPPORTED_TIMEFRAMES) {
      expect(timeframe.description).toBeDefined();
      expect(timeframe.description.length).toBeGreaterThan(0);
      expect(timeframe.description).toContain('足データ');
    }
  });
});

describe('型安全性テスト', () => {
  
  test('TimeFrame型がサポートされる時間足と一致すること', () => {
    // TypeScriptの型チェックでこれが通ることを確認
    const validTimeframes: Array<'15m' | '30m' | '1h' | '4h' | '1d'> = 
      SUPPORTED_TIMEFRAMES.map(tf => tf.value);
    
    expect(validTimeframes).toHaveLength(5);
  });

  test('TradingPair型の構造が正しいこと', () => {
    for (const pair of SUPPORTED_TRADING_PAIRS) {
      expect(typeof pair.symbol).toBe('string');
      expect(typeof pair.name).toBe('string');
      expect(typeof pair.base).toBe('string');
      expect(typeof pair.quote).toBe('string');
      
      expect(pair.symbol.length).toBeGreaterThan(0);
      expect(pair.name.length).toBeGreaterThan(0);
      expect(pair.base.length).toBeGreaterThan(0);
      expect(pair.quote.length).toBeGreaterThan(0);
    }
  });
});
