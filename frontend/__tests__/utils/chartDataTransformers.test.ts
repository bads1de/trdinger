/**
 * チャートデータ変換ユーティリティのテスト
 */

import {
  calculateDrawdown,
  transformEquityCurve,
  transformTradeHistory,
  calculateMonthlyReturns,
  calculateReturnDistribution,
  sampleData,
  calculateBuyHoldReturn,
  calculateMaxDrawdown,
  filterByDateRange,
} from '@/utils/chartDataTransformers';

import {
  generateMockEquityCurve,
  generateMockTradeHistory,
} from '../utils/chartTestUtils';

describe('chartDataTransformers', () => {
  describe('calculateDrawdown', () => {
    it('空の配列に対して空の配列を返す', () => {
      const result = calculateDrawdown([]);
      expect(result).toEqual([]);
    });

    it('単一ポイントのドローダウンを正しく計算する', () => {
      const equityCurve = [
        { timestamp: '2024-01-01', equity: 100000 }
      ];
      
      const result = calculateDrawdown(equityCurve);
      
      expect(result).toHaveLength(1);
      expect(result[0].drawdown_pct).toBe(0);
    });

    it('上昇トレンドでドローダウンが0になる', () => {
      const equityCurve = [
        { timestamp: '2024-01-01', equity: 100000 },
        { timestamp: '2024-01-02', equity: 105000 },
        { timestamp: '2024-01-03', equity: 110000 },
      ];
      
      const result = calculateDrawdown(equityCurve);
      
      result.forEach(point => {
        expect(point.drawdown_pct).toBe(0);
      });
    });

    it('下降トレンドでドローダウンを正しく計算する', () => {
      const equityCurve = [
        { timestamp: '2024-01-01', equity: 100000 },
        { timestamp: '2024-01-02', equity: 95000 },
        { timestamp: '2024-01-03', equity: 90000 },
      ];
      
      const result = calculateDrawdown(equityCurve);
      
      expect(result[0].drawdown_pct).toBe(0);
      expect(result[1].drawdown_pct).toBe(0.05); // 5%のドローダウン
      expect(result[2].drawdown_pct).toBe(0.10); // 10%のドローダウン
    });

    it('ピークからの回復を正しく処理する', () => {
      const equityCurve = [
        { timestamp: '2024-01-01', equity: 100000 },
        { timestamp: '2024-01-02', equity: 90000 },
        { timestamp: '2024-01-03', equity: 105000 },
      ];
      
      const result = calculateDrawdown(equityCurve);
      
      expect(result[0].drawdown_pct).toBe(0);
      expect(result[1].drawdown_pct).toBe(0.10);
      expect(result[2].drawdown_pct).toBe(0); // 新しいピーク
    });
  });

  describe('transformEquityCurve', () => {
    it('空の配列に対して空の配列を返す', () => {
      const result = transformEquityCurve([]);
      expect(result).toEqual([]);
    });

    it('資産曲線データを正しくチャート形式に変換する', () => {
      const equityCurve = [
        { timestamp: '2024-01-01T00:00:00Z', equity: 100000 },
        { timestamp: '2024-01-02T00:00:00Z', equity: 105000 },
      ];
      
      const result = transformEquityCurve(equityCurve);
      
      expect(result).toHaveLength(2);
      expect(result[0]).toMatchObject({
        equity: 100000,
        drawdown: 0,
      });
      expect(result[0].date).toBeGreaterThan(0);
      expect(result[0].formattedDate).toContain('2024');
    });

    it('ドローダウンをパーセンテージに変換する', () => {
      const equityCurve = [
        { timestamp: '2024-01-01T00:00:00Z', equity: 100000 },
        { timestamp: '2024-01-02T00:00:00Z', equity: 90000 },
      ];
      
      const result = transformEquityCurve(equityCurve);
      
      expect(result[1].drawdown).toBe(10); // 10%
    });
  });

  describe('transformTradeHistory', () => {
    it('空の配列に対して空の配列を返す', () => {
      const result = transformTradeHistory([]);
      expect(result).toEqual([]);
    });

    it('取引履歴を正しくチャート形式に変換する', () => {
      const trades = [
        {
          size: 1,
          entry_price: 50000,
          exit_price: 52000,
          pnl: 2000,
          return_pct: 0.04,
          entry_time: '2024-01-01T00:00:00Z',
          exit_time: '2024-01-02T00:00:00Z',
        }
      ];
      
      const result = transformTradeHistory(trades);
      
      expect(result).toHaveLength(1);
      expect(result[0]).toMatchObject({
        pnl: 2000,
        returnPct: 4, // パーセンテージに変換
        size: 1,
        type: 'long',
        isWin: true,
      });
      expect(result[0].entryDate).toBeGreaterThan(0);
      expect(result[0].exitDate).toBeGreaterThan(0);
    });

    it('ショート取引を正しく識別する', () => {
      const trades = [
        {
          size: -1,
          entry_price: 50000,
          exit_price: 48000,
          pnl: 2000,
          return_pct: 0.04,
          entry_time: '2024-01-01T00:00:00Z',
          exit_time: '2024-01-02T00:00:00Z',
        }
      ];
      
      const result = transformTradeHistory(trades);
      
      expect(result[0].type).toBe('short');
      expect(result[0].size).toBe(1); // 絶対値
    });

    it('負け取引を正しく識別する', () => {
      const trades = [
        {
          size: 1,
          entry_price: 50000,
          exit_price: 48000,
          pnl: -2000,
          return_pct: -0.04,
          entry_time: '2024-01-01T00:00:00Z',
          exit_time: '2024-01-02T00:00:00Z',
        }
      ];
      
      const result = transformTradeHistory(trades);
      
      expect(result[0].isWin).toBe(false);
      expect(result[0].returnPct).toBe(-4);
    });
  });

  describe('calculateMonthlyReturns', () => {
    it('空の配列に対して空の配列を返す', () => {
      const result = calculateMonthlyReturns([]);
      expect(result).toEqual([]);
    });

    it('月次リターンを正しく計算する', () => {
      const equityCurve = [
        { timestamp: '2024-01-01T00:00:00Z', equity: 100000 },
        { timestamp: '2024-01-15T00:00:00Z', equity: 105000 },
        { timestamp: '2024-01-31T00:00:00Z', equity: 110000 },
        { timestamp: '2024-02-15T00:00:00Z', equity: 108000 },
        { timestamp: '2024-02-29T00:00:00Z', equity: 115000 },
      ];
      
      const result = calculateMonthlyReturns(equityCurve);
      
      expect(result).toHaveLength(2);
      expect(result[0]).toMatchObject({
        year: 2024,
        month: 1,
        monthName: 'Jan',
      });
      expect(result[0].return).toBe(10); // 10%のリターン
      expect(result[1].month).toBe(2);
    });

    it('結果を年月順にソートする', () => {
      const equityCurve = [
        { timestamp: '2024-02-01T00:00:00Z', equity: 100000 },
        { timestamp: '2024-02-29T00:00:00Z', equity: 105000 },
        { timestamp: '2024-01-01T00:00:00Z', equity: 95000 },
        { timestamp: '2024-01-31T00:00:00Z', equity: 100000 },
      ];
      
      const result = calculateMonthlyReturns(equityCurve);
      
      expect(result[0].month).toBe(1); // 1月が最初
      expect(result[1].month).toBe(2); // 2月が次
    });
  });

  describe('calculateReturnDistribution', () => {
    it('空の配列に対して空の配列を返す', () => {
      const result = calculateReturnDistribution([]);
      expect(result).toEqual([]);
    });

    it('リターン分布を正しく計算する', () => {
      const trades = generateMockTradeHistory(100, 0.6);
      
      const result = calculateReturnDistribution(trades, 10);
      
      expect(result).toHaveLength(10);
      result.forEach(bin => {
        expect(bin.count).toBeGreaterThanOrEqual(0);
        expect(bin.frequency).toBeGreaterThanOrEqual(0);
        expect(bin.frequency).toBeLessThanOrEqual(100);
        expect(bin.rangeEnd).toBeGreaterThan(bin.rangeStart);
      });
    });
  });

  describe('sampleData', () => {
    it('データが最大ポイント数以下の場合はそのまま返す', () => {
      const data = [1, 2, 3, 4, 5];
      const result = sampleData(data, 10);
      expect(result).toEqual(data);
    });

    it('データが最大ポイント数を超える場合はサンプリングする', () => {
      const data = Array.from({ length: 1000 }, (_, i) => i);
      const result = sampleData(data, 100);
      
      expect(result.length).toBeLessThanOrEqual(100);
      expect(result[0]).toBe(0); // 最初の要素は保持
    });

    it('空の配列に対して空の配列を返す', () => {
      const result = sampleData([]);
      expect(result).toEqual([]);
    });
  });

  describe('calculateBuyHoldReturn', () => {
    it('空の配列に対して0を返す', () => {
      const result = calculateBuyHoldReturn([]);
      expect(result).toBe(0);
    });

    it('単一ポイントに対して0を返す', () => {
      const equityCurve = [{ timestamp: '2024-01-01', equity: 100000 }];
      const result = calculateBuyHoldReturn(equityCurve);
      expect(result).toBe(0);
    });

    it('Buy & Hold リターンを正しく計算する', () => {
      const equityCurve = [
        { timestamp: '2024-01-01', equity: 100000 },
        { timestamp: '2024-12-31', equity: 120000 },
      ];
      
      const result = calculateBuyHoldReturn(equityCurve);
      expect(result).toBe(0.2); // 20%のリターン
    });
  });

  describe('calculateMaxDrawdown', () => {
    it('空の配列に対して0を返す', () => {
      const result = calculateMaxDrawdown([]);
      expect(result).toBe(0);
    });

    it('最大ドローダウンを正しく計算する', () => {
      const equityCurve = [
        { timestamp: '2024-01-01', equity: 100000 },
        { timestamp: '2024-01-02', equity: 90000 },
        { timestamp: '2024-01-03', equity: 85000 },
        { timestamp: '2024-01-04', equity: 95000 },
      ];
      
      const result = calculateMaxDrawdown(equityCurve);
      expect(result).toBe(0.15); // 15%の最大ドローダウン
    });
  });

  describe('filterByDateRange', () => {
    it('日付範囲でデータを正しくフィルタリングする', () => {
      const data = [
        { timestamp: '2024-01-01T00:00:00Z', value: 1 },
        { timestamp: '2024-01-15T00:00:00Z', value: 2 },
        { timestamp: '2024-02-01T00:00:00Z', value: 3 },
        { timestamp: '2024-02-15T00:00:00Z', value: 4 },
      ];
      
      const startDate = new Date('2024-01-10');
      const endDate = new Date('2024-02-05');
      
      const result = filterByDateRange(data, startDate, endDate, 'timestamp');
      
      expect(result).toHaveLength(2);
      expect(result[0].value).toBe(2);
      expect(result[1].value).toBe(3);
    });

    it('範囲外のデータを除外する', () => {
      const data = [
        { timestamp: '2024-01-01T00:00:00Z', value: 1 },
        { timestamp: '2024-03-01T00:00:00Z', value: 2 },
      ];
      
      const startDate = new Date('2024-01-15');
      const endDate = new Date('2024-02-15');
      
      const result = filterByDateRange(data, startDate, endDate, 'timestamp');
      
      expect(result).toHaveLength(0);
    });
  });
});
