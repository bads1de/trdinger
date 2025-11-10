import React from 'react';
import { render, screen, fireEvent } from '@testing-library/react';
import '@testing-library/jest-dom';
import SymbolSelector from '../components/common/SymbolSelector';
import { TradingPair } from '@/types/market-data';

const mockSymbols: TradingPair[] = [
  { symbol: 'BTC/USDT', base: 'BTC', quote: 'USDT', type: 'spot' },
  { symbol: 'ETH/USDT', base: 'ETH', quote: 'USDT', type: 'spot' },
  { symbol: 'XRP/USDT', base: 'XRP', quote: 'USDT', type: 'spot' },
];

describe('SymbolSelector', () => {
  it('コンポーネントが正常にレンダリングされること', () => {
    render(
      <SymbolSelector
        symbols={mockSymbols}
        selectedSymbol="BTC/USDT"
        onSymbolChange={() => {}}
      />
    );
    expect(screen.getByLabelText('通貨ペア')).toBeInTheDocument();
    expect(screen.getByDisplayValue('BTC/USDT')).toBeInTheDocument();
  });

  it('symbolsプロパティで渡された通貨ペアがドロップダウンに表示されること', () => {
    render(
      <SymbolSelector
        symbols={mockSymbols}
        selectedSymbol="BTC/USDT"
        onSymbolChange={() => {}}
      />
    );
    const options = screen.getAllByRole('option');
    expect(options).toHaveLength(mockSymbols.length);
    expect(options[0]).toHaveTextContent('BTC/USDT');
    expect(options[1]).toHaveTextContent('ETH/USDT');
    expect(options[2]).toHaveTextContent('XRP/USDT');
  });

  it('通貨ペアを選択するとonSymbolChangeコールバックが呼び出されること', () => {
    const handleSymbolChange = jest.fn();
    render(
      <SymbolSelector
        symbols={mockSymbols}
        selectedSymbol="BTC/USDT"
        onSymbolChange={handleSymbolChange}
      />
    );
    const select = screen.getByLabelText('通貨ペア');
    fireEvent.change(select, { target: { value: 'ETH/USDT' } });
    expect(handleSymbolChange).toHaveBeenCalledWith('ETH/USDT');
  });

  it('loadingプロパティがtrueの場合、「読み込み中...」と表示されること', () => {
    render(
      <SymbolSelector
        selectedSymbol=""
        onSymbolChange={() => {}}
        loading={true}
      />
    );
    expect(screen.getByText('読み込み中...')).toBeInTheDocument();
  });

  it('symbolsが空の場合、「利用可能な通貨ペアがありません」と表示されること', () => {
    render(
      <SymbolSelector
        symbols={[]}
        selectedSymbol=""
        onSymbolChange={() => {}}
      />
    );
    expect(screen.getByText('利用可能な通貨ペアがありません')).toBeInTheDocument();
  });

  it('disabledプロパティがtrueの場合、ドロップダウンが無効化されること', () => {
    render(
      <SymbolSelector
        symbols={mockSymbols}
        selectedSymbol="BTC/USDT"
        onSymbolChange={() => {}}
        disabled={true}
      />
    );
    const select = screen.getByLabelText('通貨ペア');
    expect(select).toBeDisabled();
  });
});
