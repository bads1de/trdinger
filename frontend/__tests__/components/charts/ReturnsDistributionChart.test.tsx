/**
 * リターン分布チャートコンポーネントのテスト
 */

import React from 'react';
import { render, screen } from '@testing-library/react';
import '@testing-library/jest-dom';
import ReturnsDistributionChart from '@/components/backtest/charts/ReturnsDistributionChart';
import { generateMockTradeHistory } from '../../utils/chartTestUtils';

// Recharts のモック
jest.mock('recharts', () => ({
  ResponsiveContainer: ({ children }: { children: React.ReactNode }) => (
    <div data-testid="responsive-container">{children}</div>
  ),
  BarChart: ({ children }: { children: React.ReactNode }) => (
    <div data-testid="bar-chart">{children}</div>
  ),
  Bar: ({ dataKey }: { dataKey: string }) => (
    <div data-testid={`bar-${dataKey}`} />
  ),
  XAxis: ({ dataKey }: { dataKey: string }) => (
    <div data-testid={`x-axis-${dataKey}`} />
  ),
  YAxis: () => <div data-testid="y-axis" />,
  CartesianGrid: () => <div data-testid="cartesian-grid" />,
  Tooltip: () => <div data-testid="tooltip" />,
  ReferenceLine: ({ x }: { x: number }) => (
    <div data-testid="reference-line" data-x={x} />
  ),
}));

describe('ReturnsDistributionChart', () => {
  const mockTrades = generateMockTradeHistory(100, 0.6);

  beforeEach(() => {
    jest.clearAllMocks();
  });

  it('正常なデータでチャートをレンダリングする', () => {
    render(
      <ReturnsDistributionChart
        data={mockTrades}
        title="リターン分布"
      />
    );

    // チャートコンテナが表示される
    expect(screen.getByTestId('responsive-container')).toBeInTheDocument();
    expect(screen.getByTestId('bar-chart')).toBeInTheDocument();
    
    // 頻度バーが表示される
    expect(screen.getByTestId('bar-frequency')).toBeInTheDocument();
    
    // 軸が表示される
    expect(screen.getByTestId('x-axis-rangeLabel')).toBeInTheDocument();
    expect(screen.getByTestId('y-axis')).toBeInTheDocument();
    
    // グリッドとツールチップが表示される
    expect(screen.getByTestId('cartesian-grid')).toBeInTheDocument();
    expect(screen.getByTestId('tooltip')).toBeInTheDocument();
  });

  it('ゼロライン（損益分岐点）を表示する', () => {
    render(
      <ReturnsDistributionChart
        data={mockTrades}
        showZeroLine={true}
        title="リターン分布"
      />
    );

    // ゼロラインの参照線が表示される
    const referenceLine = screen.getByTestId('reference-line');
    expect(referenceLine).toBeInTheDocument();
    expect(referenceLine).toHaveAttribute('data-x', '0');
  });

  it('カスタムビン数でヒストグラムを生成する', () => {
    render(
      <ReturnsDistributionChart
        data={mockTrades}
        bins={15}
        title="リターン分布"
      />
    );

    // チャートが正常にレンダリングされる
    expect(screen.getByTestId('bar-chart')).toBeInTheDocument();
  });

  it('空のデータでエラー状態を表示する', () => {
    render(
      <ReturnsDistributionChart
        data={[]}
        title="リターン分布"
      />
    );

    // エラー状態が表示される
    expect(screen.getByText('データがありません')).toBeInTheDocument();
  });

  it('ローディング状態を表示する', () => {
    render(
      <ReturnsDistributionChart
        data={mockTrades}
        loading={true}
        title="リターン分布"
      />
    );

    // ローディング状態が表示される
    expect(screen.getByText('チャートを読み込み中...')).toBeInTheDocument();
  });

  it('エラー状態を表示する', () => {
    const errorMessage = 'リターン分布の計算に失敗しました';
    
    render(
      <ReturnsDistributionChart
        data={mockTrades}
        error={errorMessage}
        title="リターン分布"
      />
    );

    // エラー状態が表示される
    expect(screen.getByText('エラーが発生しました')).toBeInTheDocument();
    expect(screen.getByText(errorMessage)).toBeInTheDocument();
  });

  it('カスタムの高さとクラス名を適用する', () => {
    const customHeight = 450;
    const customClassName = 'custom-distribution-chart';
    
    render(
      <ReturnsDistributionChart
        data={mockTrades}
        height={customHeight}
        className={customClassName}
        title="リターン分布"
      />
    );

    // カスタムクラスが適用される
    const container = screen.getByTestId('responsive-container').closest('.bg-gray-800\\/30');
    expect(container).toHaveClass(customClassName);
  });

  it('サブタイトルを表示する', () => {
    const subtitle = '取引リターンの統計分布';
    
    render(
      <ReturnsDistributionChart
        data={mockTrades}
        title="リターン分布"
        subtitle={subtitle}
      />
    );

    expect(screen.getByText(subtitle)).toBeInTheDocument();
  });

  it('アクションボタンを表示する', () => {
    const actions = (
      <button data-testid="stats-button">統計情報</button>
    );
    
    render(
      <ReturnsDistributionChart
        data={mockTrades}
        title="リターン分布"
        actions={actions}
      />
    );

    expect(screen.getByTestId('stats-button')).toBeInTheDocument();
  });

  it('勝ち取引のみのデータでも正しく表示する', () => {
    const winOnlyTrades = mockTrades.filter(trade => trade.pnl > 0);
    
    render(
      <ReturnsDistributionChart
        data={winOnlyTrades}
        title="リターン分布"
      />
    );

    // チャートが正常にレンダリングされる
    expect(screen.getByTestId('bar-chart')).toBeInTheDocument();
    expect(screen.getByTestId('bar-frequency')).toBeInTheDocument();
  });

  it('負け取引のみのデータでも正しく表示する', () => {
    const lossOnlyTrades = mockTrades.filter(trade => trade.pnl < 0);
    
    render(
      <ReturnsDistributionChart
        data={lossOnlyTrades}
        title="リターン分布"
      />
    );

    // チャートが正常にレンダリングされる
    expect(screen.getByTestId('bar-chart')).toBeInTheDocument();
    expect(screen.getByTestId('bar-frequency')).toBeInTheDocument();
  });

  it('単一の取引データでも正しく表示する', () => {
    const singleTrade = [mockTrades[0]];
    
    render(
      <ReturnsDistributionChart
        data={singleTrade}
        title="リターン分布"
      />
    );

    // チャートが正常にレンダリングされる
    expect(screen.getByTestId('bar-chart')).toBeInTheDocument();
  });
});
