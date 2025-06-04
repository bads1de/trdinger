/**
 * 資産曲線チャートコンポーネントのテスト
 */

import React from 'react';
import { render, screen } from '@testing-library/react';
import '@testing-library/jest-dom';
import EquityCurveChart from '@/components/backtest/charts/EquityCurveChart';
import { generateMockChartEquityData } from '../../utils/chartTestUtils';

// Recharts のモック
jest.mock('recharts', () => ({
  ResponsiveContainer: ({ children }: { children: React.ReactNode }) => (
    <div data-testid="responsive-container">{children}</div>
  ),
  LineChart: ({ children }: { children: React.ReactNode }) => (
    <div data-testid="line-chart">{children}</div>
  ),
  Line: ({ dataKey }: { dataKey: string }) => (
    <div data-testid={`line-${dataKey}`} />
  ),
  XAxis: ({ dataKey }: { dataKey: string }) => (
    <div data-testid={`x-axis-${dataKey}`} />
  ),
  YAxis: ({ domain }: { domain?: string[] }) => (
    <div data-testid="y-axis" data-domain={domain?.join(',')} />
  ),
  CartesianGrid: () => <div data-testid="cartesian-grid" />,
  Tooltip: () => <div data-testid="tooltip" />,
  Legend: () => <div data-testid="legend" />,
  ReferenceLine: ({ y }: { y: number }) => (
    <div data-testid="reference-line" data-y={y} />
  ),
}));

describe('EquityCurveChart', () => {
  const mockData = generateMockChartEquityData(50);

  beforeEach(() => {
    jest.clearAllMocks();
  });

  it('正常なデータでチャートをレンダリングする', () => {
    render(
      <EquityCurveChart
        data={mockData}
        initialCapital={100000}
        title="資産曲線"
      />
    );

    // チャートコンテナが表示される
    expect(screen.getByTestId('responsive-container')).toBeInTheDocument();
    expect(screen.getByTestId('line-chart')).toBeInTheDocument();
    
    // 資産曲線ラインが表示される
    expect(screen.getByTestId('line-equity')).toBeInTheDocument();
    
    // 軸が表示される
    expect(screen.getByTestId('x-axis-date')).toBeInTheDocument();
    expect(screen.getByTestId('y-axis')).toBeInTheDocument();
    
    // グリッドとツールチップが表示される
    expect(screen.getByTestId('cartesian-grid')).toBeInTheDocument();
    expect(screen.getByTestId('tooltip')).toBeInTheDocument();
  });

  it('Buy & Hold 比較線を表示する', () => {
    render(
      <EquityCurveChart
        data={mockData}
        initialCapital={100000}
        buyHoldReturn={0.2}
        showBuyHold={true}
        title="資産曲線"
      />
    );

    // Buy & Hold ラインが表示される
    expect(screen.getByTestId('line-buyHold')).toBeInTheDocument();
    
    // 凡例が表示される
    expect(screen.getByTestId('legend')).toBeInTheDocument();
  });

  it('初期資本の参照線を表示する', () => {
    render(
      <EquityCurveChart
        data={mockData}
        initialCapital={100000}
        showInitialCapital={true}
        title="資産曲線"
      />
    );

    // 初期資本の参照線が表示される
    const referenceLine = screen.getByTestId('reference-line');
    expect(referenceLine).toBeInTheDocument();
    expect(referenceLine).toHaveAttribute('data-y', '100000');
  });

  it('空のデータでエラー状態を表示する', () => {
    render(
      <EquityCurveChart
        data={[]}
        initialCapital={100000}
        title="資産曲線"
      />
    );

    // エラー状態が表示される
    expect(screen.getByText('データがありません')).toBeInTheDocument();
  });

  it('ローディング状態を表示する', () => {
    render(
      <EquityCurveChart
        data={mockData}
        initialCapital={100000}
        loading={true}
        title="資産曲線"
      />
    );

    // ローディング状態が表示される
    expect(screen.getByText('チャートを読み込み中...')).toBeInTheDocument();
  });

  it('エラー状態を表示する', () => {
    const errorMessage = 'データの読み込みに失敗しました';
    
    render(
      <EquityCurveChart
        data={mockData}
        initialCapital={100000}
        error={errorMessage}
        title="資産曲線"
      />
    );

    // エラー状態が表示される
    expect(screen.getByText('エラーが発生しました')).toBeInTheDocument();
    expect(screen.getByText(errorMessage)).toBeInTheDocument();
  });

  it('カスタムの高さとクラス名を適用する', () => {
    const customHeight = 600;
    const customClassName = 'custom-chart-class';
    
    render(
      <EquityCurveChart
        data={mockData}
        initialCapital={100000}
        height={customHeight}
        className={customClassName}
        title="資産曲線"
      />
    );

    // カスタムクラスが適用される
    const container = screen.getByTestId('responsive-container').closest('.bg-gray-800\\/30');
    expect(container).toHaveClass(customClassName);
  });

  it('サブタイトルを表示する', () => {
    const subtitle = 'BTC/USDT - 2024年1月〜12月';
    
    render(
      <EquityCurveChart
        data={mockData}
        initialCapital={100000}
        title="資産曲線"
        subtitle={subtitle}
      />
    );

    expect(screen.getByText(subtitle)).toBeInTheDocument();
  });

  it('アクションボタンを表示する', () => {
    const actions = (
      <button data-testid="export-button">エクスポート</button>
    );
    
    render(
      <EquityCurveChart
        data={mockData}
        initialCapital={100000}
        title="資産曲線"
        actions={actions}
      />
    );

    expect(screen.getByTestId('export-button')).toBeInTheDocument();
  });

  it('Y軸のドメインを自動調整する', () => {
    const dataWithVariation = [
      { date: Date.now(), equity: 95000, drawdown: 5, formattedDate: '2024-01-01' },
      { date: Date.now() + 86400000, equity: 110000, drawdown: 0, formattedDate: '2024-01-02' },
    ];
    
    render(
      <EquityCurveChart
        data={dataWithVariation}
        initialCapital={100000}
        title="資産曲線"
      />
    );

    // Y軸が表示される（ドメインは自動調整）
    expect(screen.getByTestId('y-axis')).toBeInTheDocument();
  });

  it('大量データをサンプリングして表示する', () => {
    const largeData = generateMockChartEquityData(2000); // 2000ポイント
    
    render(
      <EquityCurveChart
        data={largeData}
        initialCapital={100000}
        maxDataPoints={1000}
        title="資産曲線"
      />
    );

    // チャートが正常にレンダリングされる
    expect(screen.getByTestId('line-chart')).toBeInTheDocument();
  });
});
