import { SUPPORTED_TRADING_PAIRS } from '@/constants';

describe('SUPPORTED_TRADING_PAIRS', () => {
  it('BTC と ETH の無期限先物をサポートする', () => {
    const symbols = SUPPORTED_TRADING_PAIRS.map((p) => p.symbol);
    expect(symbols).toContain('BTC/USDT:USDT');
    expect(symbols).toContain('ETH/USDT:USDT');
  });

  it('すべてのペアが必須フィールドを持つ', () => {
    for (const pair of SUPPORTED_TRADING_PAIRS) {
      expect(pair.symbol).toBeTruthy();
      expect(pair.name).toContain('/ USDT Perpetual');
      expect(pair.quote).toBe('USDT');
      // Bybit USDT無期限契約の CCXT 表記（BASE/QUOTE:SETTLE）
      expect(pair.symbol).toBe(`${pair.base}/USDT:USDT`);
    }
  });
});
