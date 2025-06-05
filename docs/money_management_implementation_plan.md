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

- 動的ポジションサイジング
- 相関を考慮した資金配分

#### リスク管理機能

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

- **説明**: 各取引で固定金額を投資します。
- **実装難易度**: 低
- **適用場面**: 初心者向け、またはリスクを一定額に固定したい場合。

#### 2. 固定比率ポジションサイジング

- **説明**: 総資産の固定比率（デフォルト 100%）を投資します。100%の場合、利用可能な全資金を意味します。
- **実装難易度**: 低
- **適用場面**: 資産の成長/減少に合わせて投資額をスケールさせたい一般的な戦略。

#### 3. ボラティリティベースポジションサイジング

- **説明**: 市場のボラティリティに基づいてポジションサイズを調整します。ボラティリティが高い場合はポジションサイズを小さくし、ボラティリティが低い場合はポジションサイズを大きくします。
- **計算例**: `ポジションサイズ = (基準リスク額 / (ATR * 価格))`, `基準リスク額 = 資本 × リスク許容度`
- **実装難易度**: 中
- **適用場面**: リスクを市場環境に適応させたい戦略。ATR（Average True Range）などのボラティリティ指標を利用します。

### 2.2 Phase 2: 高度なリスク管理

(このセクションは現在、具体的な項目がありません。将来的に他の高度なリスク管理機能が追加される可能性があります。)

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
│   └── volatility_based.py         # ボラティリティベース
├── risk_management/
│   ├── __init__.py
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
    position_sizing_method = Column(String, nullable=False)  # 'fixed_amount', 'fixed_ratio', 'volatility_based'
    position_sizing_params = Column(JSON, nullable=False)
    max_position_size = Column(Float, nullable=True)
    risk_budget = Column(Float, nullable=True)
    created_at = Column(DateTime, server_default=func.now())
    updated_at = Column(DateTime, server_default=func.now(), onupdate=func.now())
```

### Phase 2: コア機能実装（2-3 週間）

#### 2.1 ポジションサイジング計算器

```python
# 各ポジションサイジング手法の基底クラス
class BasePositionSizingCalculator(ABC):
    @abstractmethod
    def calculate(self, current_equity: float, price: float, **kwargs) -> float:
        pass

class FixedAmountCalculator(BasePositionSizingCalculator):
    def calculate(self, current_equity: float, price: float, amount: float) -> float:
        # amount は固定投資額
        return amount / price

class FixedRatioCalculator(BasePositionSizingCalculator):
    def calculate(self, current_equity: float, price: float, ratio: float) -> float:
        # ratio は投資比率 (例: 0.01 = 1%)
        position_value = current_equity * ratio
        return position_value / price

class VolatilityBasedCalculator(BasePositionSizingCalculator):
    def calculate(self, current_equity: float, price: float, atr: float, risk_percentage: float) -> float:
        # atr: Average True Range
        # risk_percentage: 許容するリスク割合 (例: 0.02 = 資本の2%)
        risk_amount_per_share = atr
        value_at_risk = current_equity * risk_percentage
        if risk_amount_per_share == 0: # ATRが0の場合のフォールバック
            return (current_equity * 0.01) / price # 例として資本の1%
        position_size = value_at_risk / (risk_amount_per_share * price) # 修正: priceを乗算
        return position_size

class PositionSizingCalculator:
    def __init__(self, method: str, params: Dict[str, Any]):
        self.method = method
        self.params = params
        self.calculator = self._create_calculator()

    def _create_calculator(self) -> BasePositionSizingCalculator:
        if self.method == "fixed_amount":
            return FixedAmountCalculator()
        elif self.method == "fixed_ratio":
            return FixedRatioCalculator()
        elif self.method == "volatility_based":
            return VolatilityBasedCalculator()
        else:
            raise ValueError(f"Unknown position sizing method: {self.method}")

    def calculate(self, current_equity: float, price: float, **kwargs) -> float:
        """
        選択された手法に基づいてポジションサイズを計算します。
        kwargs には各手法が必要とする追加パラメータが含まれます。
        例:
        - fixed_amount: amount (固定額)
        - fixed_ratio: ratio (比率)
        - volatility_based: atr (ATR値), risk_percentage (リスク許容度)
        """
        # 必要なパラメータをkwargsから取得し、calculatorに渡す
        # 各calculatorのcalculateメソッドのシグネチャに合わせてparamsを渡す必要がある
        if self.method == "fixed_amount":
            return self.calculator.calculate(current_equity, price, amount=self.params.get("fixedAmount"))
        elif self.method == "fixed_ratio":
            # デフォルト100% (ratio=1.0)
            ratio = self.params.get("fixedRatio", 1.0)
            return self.calculator.calculate(current_equity, price, ratio=ratio)
        elif self.method == "volatility_based":
            # atrとrisk_percentageはAPI経由または内部で計算/設定される想定
            atr = kwargs.get("atr") # APIリクエストや内部計算から渡される
            risk_percentage = self.params.get("volatilityRiskPercentage")
            if atr is None or risk_percentage is None:
                raise ValueError("ATR and volatilityRiskPercentage are required for volatility_based sizing.")
            return self.calculator.calculate(current_equity, price, atr=atr, risk_percentage=risk_percentage)
        else:
            # _create_calculatorでエラーになるはずだが念のため
            raise ValueError(f"Unknown position sizing method: {self.method}")

```

### Phase 3: UI/UX 実装（1-2 週間）

#### 3.1 資金管理設定モーダル（ユーザー好みに基づく）

```typescript
interface MoneyManagementSettings {
  positionSizingMethod: "fixed_amount" | "fixed_ratio" | "volatility_based";
  positionSizingParams: {
    fixedAmount?: number; // 固定金額
    fixedRatio?: number; // 固定比率 (例: 0.01 = 1%, デフォルト 1.0 = 100%)
    volatilityRiskPercentage?: number; // ボラティリティベースのリスク許容度 (例: 0.02 = 資本の2%)
    atrPeriod?: number; // ATR計算期間 (ボラティリティベース用)
  };
  riskManagement: {
    maxPositionSize?: number;
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
├── test_position_sizing_calculators.py # 固定金額・固定比率のテストを含む
├── test_volatility_based_sizing.py
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
  # Request Body Example (volatility_based の場合):
  # {
  #   "method": "volatility_based",
  #   "params": { "volatilityRiskPercentage": 0.02, "atrPeriod": 14 },
  #   "current_equity": 100000,
  #   "price": 50000,
  #   "symbol": "BTC/USDT", // ATR計算に必要なら
  #   "timeframe": "1d"     // ATR計算に必要なら
  # }
  # Response Body Example:
  # { "position_size": 0.002 }

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

- Risk Parity（リスクパリティ）
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
