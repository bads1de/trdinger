import React from 'react';
import { render, screen, fireEvent } from '@testing-library/react';
import '@testing-library/jest-dom';
import { BaseBacktestConfigForm } from '@/components/backtest/BaseBacktestConfigForm';
import { BacktestConfig } from '@/types/backtest';

const baseConfig: BacktestConfig = {
  strategy_name: 'GA_STRATEGY',
  symbol: 'BTC/USDT:USDT',
  timeframe: '4h',
  start_date: '2025-01-01',
  end_date: '2025-06-30',
  initial_capital: 100000,
  commission_rate: 0.0004,
  strategy_config: { strategy_type: '', parameters: {} },
};

describe('BaseBacktestConfigForm', () => {
  it('既定の取引ペアとして BTC 無期限先物が表示される', () => {
    render(<BaseBacktestConfigForm config={baseConfig} onConfigChange={() => {}} />);
    const combos = screen.getAllByRole('combobox');
    // 1つ目が取引ペア、2つ目が時間軸
    expect(combos[0]).toHaveTextContent('BTC/USDT:USDT');
  });

  it('取引ペアの選択肢に ETH 無期限先物が含まれる', () => {
    render(<BaseBacktestConfigForm config={baseConfig} onConfigChange={() => {}} />);
    const combos = screen.getAllByRole('combobox');
    // Radix Select はキーボード操作（ArrowDown）で開く
    fireEvent.keyDown(combos[0], { key: 'ArrowDown' });
    // Radix Select は開いた状態で選択肢をポータルへ描画する
    expect(screen.getByText('ETH/USDT:USDT')).toBeInTheDocument();
    // トリガー表示と選択肢の両方に存在するため複数ヒット
    expect(screen.getAllByText('BTC/USDT:USDT').length).toBeGreaterThanOrEqual(2);
  });
});
