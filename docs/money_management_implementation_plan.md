# トレーディングシステム資金管理機能 実装計画書

## 1. 現在のコードベース分析結果

### 1.1 既存の資金管理関連機能

#### 現在実装されている機能

- **BaseStrategy.calculate_position_size()**: 基本的なリスクパーセンテージベースのポジションサイジング
- **BacktestForm**: フロントエンドでの初期資金（initial_capital）・手数料率（commission_rate）設定 UI
- **backtesting.py ライブラリ統合**: 基本的な資金管理（cash, commission 設定）

#### 現在の実装レベル

```python
# 既存のポジションサイズ計算（非常に基本的）
def calculate_position_size(self, price: float, risk_percent: float = 1.0) -> float:
    available_cash = self.equity
    risk_amount = available_cash * (risk_percent / 100)
    position_size = risk_amount / price
    return position_size
```

### 1.2 不足している機能

#### 高度なポジションサイジング手法

- Kelly Criterion（ケリー基準）
- Risk Parity（リスクパリティ）
- 動的ポジションサイジング
- 相関を考慮した資金配分

#### リスク管理機能

- 最大ドローダウン制限
- ポジション制限（単一ポジション・総ポジション）
- VaR（Value at Risk）計算
- ポートフォリオレベルでのリスク管理

#### 資金管理設定

- 複数戦略への資金配分
- 動的リバランシング
- リスク予算管理

## 2. 推奨する資金管理手法と実装優先度

### 2.1 Phase 1: 基本的なポジションサイジング手法

#### 1. 固定金額ポジションサイジング

- **説明**: 各取引で固定金額を投資
- **実装難易度**: 低
- **適用場面**: 初心者向け、リスク制御重視

#### 2. 固定比率ポジションサイジング

- **説明**: 総資産の固定比率を投資
- **実装難易度**: 低
- **適用場面**: 一般的な戦略

#### 3. Kelly Criterion（ケリー基準）

- **説明**: 数学的に最適なポジションサイズを計算
- **計算式**: f\* = (bp - q) / b
  - f\*: 最適投資比率
  - b: 勝率時の利益率
  - p: 勝率
  - q: 負率 (1-p)
- **実装難易度**: 中
- **適用場面**: 統計的優位性がある戦略
- **注意**: フルケリーは非常にアグレッシブ

#### 4. Half Kelly（ハーフケリー）

- **説明**: Kelly Criterion の 50%を使用する保守的なアプローチ
- **計算式**: f\* = 0.5 × (bp - q) / b
- **実装難易度**: 中
- **適用場面**: リスクを抑えつつ成長を狙う戦略
- **メリット**:
  - ドローダウンリスクの大幅削減
  - より安定した成長曲線
  - 実用的なリスクレベル

### 2.2 Phase 2: 高度なリスク管理

#### 1. Risk Parity（リスクパリティ）

- **説明**: リスク寄与度を均等にする資金配分
- **実装難易度**: 高
- **適用場面**: 複数戦略・複数資産のポートフォリオ

#### 2. 最大ドローダウン制限

- **説明**: ドローダウンが閾値を超えた場合の取引停止
- **実装難易度**: 中
- **適用場面**: 全戦略共通

## 3. 段階的実装計画

### Phase 1: 基盤整備（1-2 週間）

#### 1.1 バックエンド基盤

```
backend/app/core/services/money_management/
├── __init__.py
├── money_management_service.py      # メインサービス
├── position_sizing/
│   ├── __init__.py
│   ├── base_calculator.py          # 基底クラス
│   ├── fixed_amount.py             # 固定金額
│   ├── fixed_ratio.py              # 固定比率
│   ├── kelly_criterion.py          # ケリー基準（フル）
│   ├── half_kelly.py               # ハーフケリー
│   └── risk_parity.py              # リスクパリティ
├── risk_management/
│   ├── __init__.py
│   ├── drawdown_manager.py         # ドローダウン管理
│   ├── position_limits.py          # ポジション制限
│   └── var_calculator.py           # VaR計算
└── portfolio_manager.py            # ポートフォリオ管理
```

#### 1.2 データモデル拡張

```python
# 新規テーブル: money_management_settings
class MoneyManagementSettings(Base):
    __tablename__ = "money_management_settings"

    id = Column(Integer, primary_key=True)
    strategy_id = Column(String, nullable=False)
    position_sizing_method = Column(String, nullable=False)  # 'fixed_amount', 'fixed_ratio', 'kelly', 'half_kelly', 'risk_parity'
    position_sizing_params = Column(JSON, nullable=False)
    max_drawdown_limit = Column(Float, nullable=True)
    max_position_size = Column(Float, nullable=True)
    risk_budget = Column(Float, nullable=True)
    created_at = Column(DateTime, server_default=func.now())
    updated_at = Column(DateTime, server_default=func.now(), onupdate=func.now())
```

### Phase 2: コア機能実装（2-3 週間）

#### 2.1 ポジションサイジング計算器

```python
class PositionSizingCalculator:
    def __init__(self, method: str, params: Dict[str, Any]):
        self.method = method
        self.params = params
        self.calculator = self._create_calculator()

    def calculate(self,
                 current_equity: float,
                 price: float,
                 historical_returns: List[float] = None,
                 win_rate: float = None,
                 avg_win: float = None,
                 avg_loss: float = None) -> float:
        return self.calculator.calculate(
            current_equity, price, historical_returns, win_rate, avg_win, avg_loss
        )
```

#### 2.2 Kelly Criterion & Half Kelly 実装

```python
class KellyCriterionCalculator(BasePositionSizingCalculator):
    def __init__(self, kelly_multiplier: float = 1.0):
        """
        Args:
            kelly_multiplier: Kelly fractionの乗数
                            1.0 = Full Kelly
                            0.5 = Half Kelly
                            0.25 = Quarter Kelly
        """
        self.kelly_multiplier = kelly_multiplier

    def calculate(self, current_equity: float, price: float,
                 win_rate: float, avg_win: float, avg_loss: float) -> float:
        if win_rate is None or avg_win is None or avg_loss is None:
            raise ValueError("Kelly Criterion requires win_rate, avg_win, and avg_loss")

        # Kelly formula: f* = (bp - q) / b
        b = avg_win / abs(avg_loss)  # odds
        p = win_rate
        q = 1 - p

        kelly_fraction = (b * p - q) / b

        # Kelly fractionにmultiplierを適用
        adjusted_kelly = kelly_fraction * self.kelly_multiplier

        # 安全制限（最大25%）
        adjusted_kelly = max(0, min(adjusted_kelly, 0.25))

        position_value = current_equity * adjusted_kelly
        return position_value / price

class HalfKellyCalculator(KellyCriterionCalculator):
    """Half Kelly専用クラス（使いやすさのため）"""
    def __init__(self):
        super().__init__(kelly_multiplier=0.5)
```

### Phase 3: UI/UX 実装（1-2 週間）

#### 3.1 資金管理設定モーダル（ユーザー好みに基づく）

```typescript
interface MoneyManagementSettings {
  positionSizingMethod:
    | "fixed_amount"
    | "fixed_ratio"
    | "kelly"
    | "half_kelly"
    | "risk_parity";
  positionSizingParams: {
    fixedAmount?: number;
    fixedRatio?: number;
    kellyMultiplier?: number; // 1.0 = Full Kelly, 0.5 = Half Kelly, 0.25 = Quarter Kelly
    riskBudget?: number;
  };
  riskManagement: {
    maxDrawdownLimit?: number;
    maxPositionSize?: number;
    stopTradingOnDrawdown?: boolean;
  };
}
```

#### 3.2 バックテスト設定からの値継承

- 現在のバックテスト設定（initial_capital, commission_rate）を自動で引き継ぎ
- デフォルト値の自動設定機能

### Phase 4: 統合・テスト（1 週間）

#### 4.1 既存システムとの統合

- backtesting.py ライブラリとの統合
- BaseStrategy クラスの拡張
- バックテストサービスでの資金管理適用

#### 4.2 テスト戦略（TDD アプローチ）

```
backend/tests/unit/money_management/
├── test_position_sizing_calculators.py
├── test_kelly_criterion.py
├── test_risk_management.py
└── test_money_management_service.py

backend/tests/integration/
└── test_backtest_with_money_management.py
```

## 4. API 設計

### 4.1 新規エンドポイント

```
POST /api/money-management/settings          # 設定保存
GET  /api/money-management/settings/{id}     # 設定取得
PUT  /api/money-management/settings/{id}     # 設定更新
POST /api/money-management/calculate-position # ポジションサイズ計算
GET  /api/money-management/methods           # 利用可能な手法一覧
```

### 4.2 既存バックテスト API の拡張

```python
class BacktestRequest(BaseModel):
    # 既存フィールド
    strategy_name: str
    symbol: str
    timeframe: str
    start_date: str
    end_date: str
    initial_capital: float
    commission_rate: float
    strategy_config: StrategyConfig

    # 新規フィールド
    money_management: Optional[MoneyManagementSettings] = None
```

## 5. 実装における技術的考慮事項

### 5.1 backtesting.py ライブラリとの統合

- Strategy クラスの拡張でポジションサイジングロジックを組み込み
- 既存の self.buy()、self.sell()メソッドの拡張

### 5.2 パフォーマンス最適化

- ポジションサイズ計算のキャッシュ機能
- 大量データでの効率的な計算

### 5.3 エラーハンドリング

- 不正なパラメータに対する適切なエラーメッセージ
- 計算不可能な場合のフォールバック機能

## 6. 成功指標とテスト計画

### 6.1 機能テスト

- 各ポジションサイジング手法の正確性検証
- リスク管理機能の動作確認
- UI/UX の使いやすさテスト

### 6.2 パフォーマンステスト

- 大量データでの計算速度
- メモリ使用量の最適化

### 6.3 統合テスト

- 既存バックテストシステムとの互換性
- エンドツーエンドでの動作確認

## 7. 今後の拡張計画

### 7.1 高度な機能

- 機械学習ベースのポジションサイジング
- リアルタイムリスク監視
- 複数戦略の自動リバランシング

### 7.2 外部システム連携

- 実取引システムとの統合
- リスク管理システムとの連携

---

**実装開始予定**: 計画承認後即座
**完了予定**: 4-6 週間後
**責任者**: Trdinger Development Team
