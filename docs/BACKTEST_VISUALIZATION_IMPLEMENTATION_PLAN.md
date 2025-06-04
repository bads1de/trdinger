# バックテストシステム可視化機能実装計画

## 📋 プロジェクト概要

バックテストシステムの可視化機能を Next.js フロントエンドに統合し、資産曲線、ドローダウン、取引履歴等のグラフを表示する機能を実装します。

## 🔍 現状調査結果

### バックエンドの可視化機能

#### 利用可能なデータ構造

```typescript
interface BacktestResult {
  // 基本情報
  strategy_name: string;
  symbol: string;
  timeframe: string;
  start_date: string;
  end_date: string;
  initial_capital: number;
  commission_rate: number;

  // パフォーマンス指標
  performance_metrics: {
    total_return: number;
    sharpe_ratio: number;
    max_drawdown: number;
    win_rate: number;
    total_trades: number;
    equity_final: number;
    buy_hold_return: number;
    exposure_time: number;
    sortino_ratio: number;
    calmar_ratio: number;
  };

  // 資産曲線データ
  equity_curve: Array<{
    timestamp: string;
    equity: number;
    drawdown_pct: number;
  }>;

  // 取引履歴データ
  trade_history: Array<{
    size: number;
    entry_price: number;
    exit_price: number;
    pnl: number;
    return_pct: number;
    entry_time: string;
    exit_time: string;
  }>;
}
```

#### 既存の可視化機能

- **BacktestService**: JSON 形式で equity_curve、trade_history を出力
- **EnhancedBacktestService**: matplotlib 基盤のヒートマップ生成（plot_heatmaps）
  - HTML ファイルとして保存される最適化ヒートマップ
  - `save_heatmap`機能でファイル出力
  - 現在はバックエンドのみで生成、フロントエンド統合が必要
- **データ変換**: backtesting.py の結果を標準化された JSON 形式に変換

### フロントエンドの現状

- **技術スタック**: Next.js 15 + React 18 + TypeScript + TailwindCSS
- **現在の表示**: テーブル形式のみ（BacktestResultsTable、PerformanceMetrics）
- **チャートライブラリ**: 未導入
- **UI 設計**: ダークモード、エンタープライズデザイン
- **API 統合**: useApiCall フック、エラーハンドリング、ローディング状態管理
- **データフェッチ**: RESTful API、ページネーション対応、フィルタリング機能

## 🛠 技術仕様

### チャートライブラリ選択

**推奨: Recharts**

#### 選択理由

- ✅ React 専用で統合が簡単
- ✅ TypeScript 完全対応
- ✅ 宣言的で保守しやすい
- ✅ 金融チャートの基本機能を提供
- ✅ 既存の TailwindCSS + ダークモードテーマと親和性が高い

#### 代替案比較

| ライブラリ | 統合性     | パフォーマンス | カスタマイズ性 | 学習コスト | 推奨度        |
| ---------- | ---------- | -------------- | -------------- | ---------- | ------------- |
| Recharts   | ⭐⭐⭐⭐⭐ | ⭐⭐⭐         | ⭐⭐⭐         | ⭐⭐⭐⭐⭐ | **第 1 選択** |
| Chart.js   | ⭐⭐⭐     | ⭐⭐⭐⭐⭐     | ⭐⭐⭐⭐       | ⭐⭐⭐     | 第 2 選択     |
| D3.js      | ⭐⭐       | ⭐⭐⭐⭐⭐     | ⭐⭐⭐⭐⭐     | ⭐         | 将来検討      |

### データ転送方法

**JSON 形式での直接転送**

- 既存の BacktestService の出力をそのまま活用
- 軽量で高速
- リアルタイム更新対応
- キャッシュ効率が良い

### API エンドポイント設計

#### 既存エンドポイント活用

```
GET /api/backtest/results/{id}
```

- equity_curve と trade_history が既に含まれている
- 追加のエンドポイントは当面不要

#### 将来的な拡張（パフォーマンス最適化時）

```
GET /api/backtest/results/{id}/charts
GET /api/backtest/results/{id}/equity-curve
GET /api/backtest/results/{id}/drawdown
GET /api/backtest/results/{id}/returns-distribution
GET /api/backtest/results/{id}/heatmap        # 既存ヒートマップファイル取得
```

#### 既存ヒートマップ統合

- **課題**: EnhancedBacktestService で生成される HTML ヒートマップの統合
- **解決策**:
  1. HTML ファイルを iframe 埋め込み
  2. データ抽出して Recharts で再実装
  3. 静的ファイル配信エンドポイントの追加

## 🏗 実装計画

### Phase 1: 基盤整備（1-2 日）

#### 1.1 依存関係の追加

```bash
cd frontend
npm install recharts
npm install --save-dev @types/recharts
```

#### 1.2 チャート共通コンポーネント作成

```
frontend/components/backtest/charts/
├── ChartContainer.tsx        # 共通コンテナ
├── ChartTheme.ts            # テーマ設定
└── types.ts                 # 型定義
```

### Phase 2: 基本チャート実装（3-4 日）

#### 2.1 資産曲線チャート（最優先）

- **ファイル**: `EquityCurveChart.tsx`
- **機能**:
  - 時系列での資産推移表示
  - Buy & Hold との比較
  - ズーム・パン機能
  - ツールチップでの詳細表示

#### 2.2 ドローダウンチャート

- **ファイル**: `DrawdownChart.tsx`
- **機能**:
  - ドローダウン期間の可視化
  - 最大ドローダウンのハイライト
  - 回復期間の表示

#### 2.3 取引散布図

- **ファイル**: `TradeScatterChart.tsx`
- **機能**:
  - 利益/損失の分布
  - 取引サイズとの相関
  - 勝率の可視化

### Phase 3: 高度なチャート実装（2-3 日）

#### 3.1 リターン分布

- **ファイル**: `ReturnsDistribution.tsx`
- **機能**:
  - ヒストグラム表示
  - 正規分布との比較
  - 統計情報の表示

#### 3.2 月次リターンヒートマップ

- **ファイル**: `MonthlyReturns.tsx`
- **機能**:
  - 月次パフォーマンスの可視化
  - 季節性の分析
  - 年次比較

### Phase 4: UI 統合（2-3 日）

#### 4.1 PerformanceMetrics コンポーネント拡張

```typescript
// 新しいタブ構成
const tabs = [
  { id: "overview", label: "概要" },
  { id: "charts", label: "チャート" }, // 新規追加
  { id: "heatmap", label: "ヒートマップ" }, // 既存機能統合
  { id: "trades", label: "取引履歴" },
];
```

#### 4.2 レスポンシブ対応

- モバイル・タブレット対応
- チャートサイズの動的調整
- タッチ操作対応

#### 4.3 ダークモード対応

- 既存の Tailwind テーマとの統合
- チャートカラーパレットの最適化

#### 4.4 既存ヒートマップ統合

- HTML ヒートマップの iframe 表示
- ファイルアクセス権限の設定
- エラーハンドリングの実装

## 📊 実装するチャートの詳細

### 1. 資産曲線チャート

```typescript
interface EquityCurveProps {
  equityCurve: Array<{
    timestamp: string;
    equity: number;
    drawdown_pct: number;
  }>;
  initialCapital: number;
  buyHoldReturn?: number;
}
```

### 2. ドローダウンチャート

```typescript
interface DrawdownChartProps {
  equityCurve: Array<{
    timestamp: string;
    equity: number;
    drawdown_pct: number;
  }>;
  maxDrawdown: number;
}
```

### 3. 取引散布図

```typescript
interface TradeScatterProps {
  trades: Array<{
    entry_time: string;
    exit_time: string;
    pnl: number;
    return_pct: number;
    size: number;
  }>;
}
```

## 🎯 成功指標

### 機能要件

- ✅ 資産曲線の表示
- ✅ ドローダウンの可視化
- ✅ 取引履歴の散布図表示
- ✅ リターン分布の表示
- ✅ 月次リターンヒートマップ

### 非機能要件

- ✅ レスポンシブデザイン
- ✅ ダークモード対応
- ✅ 高速レンダリング（1000 点以上のデータ）
- ✅ アクセシビリティ対応

### ユーザビリティ

- ✅ 直感的な操作
- ✅ 詳細情報のツールチップ
- ✅ ズーム・パン機能
- ✅ データエクスポート機能

## 🚀 次のステップ

1. **Phase 1 の実装開始**: Recharts の導入と基盤整備
2. **プロトタイプ作成**: 資産曲線チャートの基本実装
3. **ユーザーフィードバック**: 初期版でのユーザビリティテスト
4. **段階的拡張**: 追加チャートの実装
5. **パフォーマンス最適化**: 大量データ対応

## 🔧 技術的実装詳細

### コンポーネント設計パターン

#### 1. チャート共通インターフェース

```typescript
interface BaseChartProps {
  data: any[];
  loading?: boolean;
  error?: string;
  height?: number;
  className?: string;
  theme?: "light" | "dark";
}

interface ChartContainerProps extends BaseChartProps {
  title: string;
  subtitle?: string;
  children: React.ReactNode;
  actions?: React.ReactNode;
}
```

#### 2. データ変換ユーティリティ

```typescript
// utils/chartDataTransformers.ts
export const transformEquityCurve = (equityCurve: EquityCurveData[]) => {
  return equityCurve.map((point) => ({
    date: new Date(point.timestamp).getTime(),
    equity: point.equity,
    drawdown: point.drawdown_pct * 100,
    formattedDate: format(new Date(point.timestamp), "yyyy-MM-dd"),
  }));
};

export const transformTradeHistory = (trades: TradeData[]) => {
  return trades.map((trade) => ({
    entryDate: new Date(trade.entry_time).getTime(),
    exitDate: new Date(trade.exit_time).getTime(),
    pnl: trade.pnl,
    returnPct: trade.return_pct * 100,
    size: Math.abs(trade.size),
    type: trade.size > 0 ? "long" : "short",
  }));
};
```

### パフォーマンス最適化戦略

#### 1. データサンプリング

```typescript
const sampleData = (data: any[], maxPoints: number = 1000) => {
  if (data.length <= maxPoints) return data;

  const step = Math.ceil(data.length / maxPoints);
  return data.filter((_, index) => index % step === 0);
};
```

#### 2. 仮想化とレイジーローディング

```typescript
const ChartLazyLoader = ({ children, threshold = 0.1 }) => {
  const [isVisible, setIsVisible] = useState(false);
  const ref = useRef();

  useEffect(() => {
    const observer = new IntersectionObserver(
      ([entry]) => setIsVisible(entry.isIntersecting),
      { threshold }
    );

    if (ref.current) observer.observe(ref.current);
    return () => observer.disconnect();
  }, [threshold]);

  return <div ref={ref}>{isVisible ? children : <ChartSkeleton />}</div>;
};
```

### テスト戦略

#### 1. 単体テスト

```typescript
// __tests__/components/charts/EquityCurveChart.test.tsx
describe("EquityCurveChart", () => {
  it("renders equity curve correctly", () => {
    const mockData = generateMockEquityCurve();
    render(<EquityCurveChart data={mockData} />);

    expect(screen.getByTestId("equity-curve-chart")).toBeInTheDocument();
  });

  it("handles empty data gracefully", () => {
    render(<EquityCurveChart data={[]} />);
    expect(screen.getByText("データがありません")).toBeInTheDocument();
  });
});
```

#### 2. 統合テスト

```typescript
// __tests__/integration/backtest-charts.test.tsx
describe("Backtest Charts Integration", () => {
  it("displays all charts when backtest result is loaded", async () => {
    const mockResult = generateMockBacktestResult();

    render(<PerformanceMetrics result={mockResult} />);

    // チャートタブをクリック
    fireEvent.click(screen.getByText("チャート"));

    // 各チャートが表示されることを確認
    await waitFor(() => {
      expect(screen.getByTestId("equity-curve-chart")).toBeInTheDocument();
      expect(screen.getByTestId("drawdown-chart")).toBeInTheDocument();
      expect(screen.getByTestId("trade-scatter-chart")).toBeInTheDocument();
    });
  });
});
```

## 🎨 デザインシステム統合

### カラーパレット（ダークモード対応）

```typescript
export const chartColors = {
  primary: "#3B82F6", // blue-500
  success: "#10B981", // emerald-500
  danger: "#EF4444", // red-500
  warning: "#F59E0B", // amber-500
  neutral: "#6B7280", // gray-500

  // グラデーション
  equityGradient: ["#3B82F6", "#1D4ED8"],
  drawdownGradient: ["#EF4444", "#DC2626"],

  // 背景
  chartBackground: "#111827", // gray-900
  gridColor: "#374151", // gray-700
  textColor: "#F9FAFB", // gray-50
};
```

### レスポンシブブレークポイント

```typescript
export const chartBreakpoints = {
  mobile: { width: "100%", height: 300 },
  tablet: { width: "100%", height: 400 },
  desktop: { width: "100%", height: 500 },
  large: { width: "100%", height: 600 },
};
```

## 📈 段階的リリース計画

### MVP (Minimum Viable Product) - Week 1

- ✅ Recharts 導入
- ✅ 資産曲線チャート
- ✅ 基本的なツールチップ
- ✅ ダークモード対応

### Enhanced Version - Week 2

- ✅ ドローダウンチャート
- ✅ 取引散布図
- ✅ ズーム・パン機能
- ✅ データエクスポート

### Advanced Version - Week 3

- ✅ リターン分布
- ✅ 月次リターンヒートマップ
- ✅ パフォーマンス最適化
- ✅ アニメーション効果

### Enterprise Version - Week 4

- ✅ カスタムインジケーター
- ✅ 比較機能（複数戦略）
- ✅ 高度なフィルタリング
- ✅ レポート生成機能

## 🔍 品質保証

### コードレビューチェックリスト

- [ ] TypeScript 型安全性
- [ ] パフォーマンス（大量データ対応）
- [ ] アクセシビリティ（WCAG 2.1 AA 準拠）
- [ ] レスポンシブデザイン
- [ ] エラーハンドリング
- [ ] テストカバレッジ（80%以上）

### ブラウザ対応

- ✅ Chrome 90+
- ✅ Firefox 88+
- ✅ Safari 14+
- ✅ Edge 90+

---

**実装開始準備完了** ✅

## ⚠️ 重要な追加考慮事項

### セキュリティ・パフォーマンス

- **CORS 設定**: バックエンドとの適切な通信設定
- **データ検証**: チャートデータの入力検証
- **XSS 対策**: HTML ヒートマップ表示時のセキュリティ
- **メモリ管理**: 大量データ処理時のメモリリーク防止
- **バンドルサイズ**: Recharts ライブラリの最適化

### アクセシビリティ・ユーザビリティ

- **WCAG 2.1 AA 準拠**: スクリーンリーダー対応
- **キーボードナビゲーション**: チャート操作のキーボード対応
- **カラーコントラスト**: 色覚異常対応
- **エラー境界**: React エラー境界の実装
- **ローディング状態**: スケルトン UI、プログレスバー

### データ管理・最適化

- **キャッシュ戦略**: チャートデータのブラウザキャッシュ
- **データ圧縮**: 大量データの効率的転送
- **リアルタイム更新**: WebSocket/SSE 検討
- **オフライン対応**: ネットワーク切断時の対応
- **ユーザー設定**: チャート設定の永続化

### 運用・監視

- **パフォーマンス監視**: チャートレンダリング性能
- **エラー監視**: チャート表示エラーの追跡
- **ユーザー分析**: チャート利用状況の分析
- **A/B テスト**: UI 改善のための実験機能

**次のアクション**: Phase 1 の実装開始 - Recharts の導入と基盤整備
