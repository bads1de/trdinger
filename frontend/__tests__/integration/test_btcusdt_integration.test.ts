/**
 * BTC/USDT:USDT限定機能の統合テスト
 * 
 * フロントエンド、API、バックエンドの統合動作をテスト
 */

import { 
  SUPPORTED_TRADING_PAIRS, 
  SUPPORTED_TIMEFRAMES, 
  DEFAULT_TRADING_PAIR,
  DEFAULT_TIMEFRAME 
} from '@/constants';
import { 
  validateSymbol, 
  validateTimeframe, 
  getSupportedSymbols, 
  getSupportedTimeframes 
} from '@/lib/validation';

describe('BTC/USDT:USDT限定機能統合テスト', () => {
  
  describe('設定の一貫性テスト', () => {
    
    test('フロントエンドとバリデーション関数の設定が一致すること', () => {
      const frontendSymbols = SUPPORTED_TRADING_PAIRS.map(pair => pair.symbol);
      const validationSymbols = getSupportedSymbols().map(pair => pair.symbol);
      
      expect(frontendSymbols).toEqual(validationSymbols);
      expect(frontendSymbols).toEqual(['BTC/USDT:USDT']);
    });

    test('フロントエンドとバリデーション関数の時間足が一致すること', () => {
      const frontendTimeframes = SUPPORTED_TIMEFRAMES.map(tf => tf.value);
      const validationTimeframes = getSupportedTimeframes().map(tf => tf.value);
      
      expect(frontendTimeframes).toEqual(validationTimeframes);
      expect(frontendTimeframes).toEqual(['15m', '30m', '1h', '4h', '1d']);
    });

    test('デフォルト値がサポートされる値に含まれること', () => {
      expect(validateSymbol(DEFAULT_TRADING_PAIR)).toBe(true);
      expect(validateTimeframe(DEFAULT_TIMEFRAME)).toBe(true);
    });
  });

  describe('バリデーション機能テスト', () => {
    
    test('BTC/USDT:USDTが有効なシンボルとして認識されること', () => {
      expect(validateSymbol('BTC/USDT:USDT')).toBe(true);
    });

    test('サポートされていないシンボルが無効として認識されること', () => {
      const unsupportedSymbols = [
        'BTC/USDT',      // 現物
        'BTCUSD',        // USD建て無期限先物
        'ETH/USDT:USDT', // イーサリアム無期限先物
        'ETH/USDT',      // イーサリアム現物
        'INVALID/SYMBOL' // 無効なシンボル
      ];

      for (const symbol of unsupportedSymbols) {
        expect(validateSymbol(symbol)).toBe(false);
      }
    });

    test('要求された時間足が有効として認識されること', () => {
      const requiredTimeframes = ['15m', '30m', '1h', '4h', '1d'];
      
      for (const timeframe of requiredTimeframes) {
        expect(validateTimeframe(timeframe)).toBe(true);
      }
    });

    test('サポートされていない時間足が無効として認識されること', () => {
      const unsupportedTimeframes = [
        '1m',   // 1分足（要求されていない）
        '5m',   // 5分足
        '12h',  // 12時間足
        '1w',   // 週足
        '1M'    // 月足
      ];

      for (const timeframe of unsupportedTimeframes) {
        expect(validateTimeframe(timeframe)).toBe(false);
      }
    });
  });

  describe('データ構造の整合性テスト', () => {
    
    test('TradingPairオブジェクトの構造が正しいこと', () => {
      const btcUsdtPerp = SUPPORTED_TRADING_PAIRS[0];
      
      expect(btcUsdtPerp).toHaveProperty('symbol', 'BTC/USDT:USDT');
      expect(btcUsdtPerp).toHaveProperty('name');
      expect(btcUsdtPerp).toHaveProperty('base', 'BTC');
      expect(btcUsdtPerp).toHaveProperty('quote', 'USDT');
      
      expect(btcUsdtPerp.name).toContain('Bitcoin');
      expect(btcUsdtPerp.name).toContain('Perpetual');
    });

    test('TimeFrameInfoオブジェクトの構造が正しいこと', () => {
      for (const timeframe of SUPPORTED_TIMEFRAMES) {
        expect(timeframe).toHaveProperty('value');
        expect(timeframe).toHaveProperty('label');
        expect(timeframe).toHaveProperty('description');
        
        expect(typeof timeframe.value).toBe('string');
        expect(typeof timeframe.label).toBe('string');
        expect(typeof timeframe.description).toBe('string');
        
        expect(timeframe.label.length).toBeGreaterThan(0);
        expect(timeframe.description.length).toBeGreaterThan(0);
      }
    });
  });

  describe('型安全性テスト', () => {
    
    test('TimeFrame型の値が実際のサポート時間足と一致すること', () => {
      // TypeScriptの型チェックでこれが通ることを確認
      const timeframeValues = SUPPORTED_TIMEFRAMES.map(tf => tf.value);
      const expectedValues: Array<'15m' | '30m' | '1h' | '4h' | '1d'> = 
        ['15m', '30m', '1h', '4h', '1d'];
      
      expect(timeframeValues).toEqual(expectedValues);
    });

    test('シンボル文字列の形式が正しいこと', () => {
      const symbol = SUPPORTED_TRADING_PAIRS[0].symbol;
      
      // BTC/USDT:USDT形式であることを確認
      expect(symbol).toMatch(/^[A-Z]+\/[A-Z]+:[A-Z]+$/);
      expect(symbol.split('/')[0]).toBe('BTC');
      expect(symbol.split('/')[1].split(':')[0]).toBe('USDT');
      expect(symbol.split(':')[1]).toBe('USDT');
    });
  });

  describe('エラーハンドリングテスト', () => {
    
    test('空文字列のシンボルが無効として認識されること', () => {
      expect(validateSymbol('')).toBe(false);
    });

    test('空文字列の時間足が無効として認識されること', () => {
      expect(validateTimeframe('')).toBe(false);
    });

    test('nullやundefinedが適切に処理されること', () => {
      expect(validateSymbol(null as any)).toBe(false);
      expect(validateSymbol(undefined as any)).toBe(false);
      expect(validateTimeframe(null as any)).toBe(false);
      expect(validateTimeframe(undefined as any)).toBe(false);
    });

    test('大文字小文字の違いが適切に処理されること', () => {
      // 現在の実装では大文字小文字を区別する
      expect(validateSymbol('btc/usdt:usdt')).toBe(false);
      expect(validateTimeframe('1H')).toBe(false);
      expect(validateTimeframe('1D')).toBe(false);
    });
  });

  describe('後方互換性テスト', () => {
    
    test('既存のコンポーネントが新しい設定で動作すること', () => {
      // SymbolSelectorコンポーネントで使用される形式
      const symbolOptions = SUPPORTED_TRADING_PAIRS.map(pair => ({
        value: pair.symbol,
        label: pair.symbol
      }));
      
      expect(symbolOptions).toHaveLength(1);
      expect(symbolOptions[0].value).toBe('BTC/USDT:USDT');
      expect(symbolOptions[0].label).toBe('BTC/USDT:USDT');
    });

    test('TimeFrameSelectorコンポーネントで使用される形式', () => {
      const timeframeOptions = SUPPORTED_TIMEFRAMES.map(tf => ({
        value: tf.value,
        label: tf.label
      }));
      
      expect(timeframeOptions).toHaveLength(5);
      
      const expectedOptions = [
        { value: '15m', label: '15分' },
        { value: '30m', label: '30分' },
        { value: '1h', label: '1時間' },
        { value: '4h', label: '4時間' },
        { value: '1d', label: '1日' }
      ];
      
      expect(timeframeOptions).toEqual(expectedOptions);
    });
  });

  describe('パフォーマンステスト', () => {
    
    test('バリデーション関数が高速に動作すること', () => {
      const startTime = performance.now();
      
      // 1000回のバリデーションを実行
      for (let i = 0; i < 1000; i++) {
        validateSymbol('BTC/USDT:USDT');
        validateTimeframe('1h');
      }
      
      const endTime = performance.now();
      const duration = endTime - startTime;
      
      // 1000回の実行が100ms以内に完了することを確認
      expect(duration).toBeLessThan(100);
    });

    test('設定データの取得が高速に動作すること', () => {
      const startTime = performance.now();
      
      // 1000回の設定データ取得を実行
      for (let i = 0; i < 1000; i++) {
        getSupportedSymbols();
        getSupportedTimeframes();
      }
      
      const endTime = performance.now();
      const duration = endTime - startTime;
      
      // 1000回の実行が50ms以内に完了することを確認
      expect(duration).toBeLessThan(50);
    });
  });
});
