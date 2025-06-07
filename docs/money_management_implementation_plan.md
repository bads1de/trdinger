# トレーディングシステム資金管理機能 実装計画書

## 1. 設計方針

### 1.1 既存システム活用アプローチ

**基本方針：**

- 既存の`BacktestRequest.strategy_config.parameters`を活用
- 新しい DB テーブルや API エンドポイントは作成しない
- `BaseStrategy.calculate_position_size`を拡張してエントリーごとの動的計算を実現

### 1.2 現在の実装状況

```python
def calculate_position_size(self, price: float, risk_percent: float = 1.0) -> float:
    available_cash = self.equity
    risk_amount = available_cash * (risk_percent / 100)
    position_size = risk_amount / price
    return position_size
```

```python
class BacktestRequest(BaseModel):
    strategy_name: str
    symbol: str
    timeframe: str
    start_date: datetime
    end_date: datetime
    initial_capital: float
    commission_rate: float
    strategy_config: StrategyConfig
```

## 2. 実装する資金管理手法

### 2.1 基本的なポジションサイジング手法

#### 1. 固定金額ポジションサイジング

- **説明**: 各取引で固定金額を投資
- **計算**: `ポジションサイズ = 固定金額 / 価格`

#### 2. 固定比率ポジションサイジング

- **説明**: 総資産の固定比率を投資（デフォルト 100%）
- **計算**: `ポジションサイズ = (資産 × 比率) / 価格`

#### 3. ボラティリティベースポジションサイジング

- **説明**: ATR に基づいてリスクを調整
- **計算**: `ポジションサイズ = (資産 × リスク許容度) / (ATR × 価格)`

## 3. 実装計画

### 3.1 ファイル構成

```text
backend/app/core/strategies/position_sizing/
├── __init__.py
├── base_calculator.py          # 基底クラス
└── calculators.py              # 3つの計算器実装
```

### 3.2 パラメータ設定方法

既存の`BacktestRequest.strategy_config.parameters`に資金管理設定を追加：

```python
{
  "strategy_type": "some_strategy",
  "parameters": {
    "money_management": {
      "method": "fixed_ratio",  # "fixed_amount" | "fixed_ratio" | "volatility_based"
      "params": {
        "fixed_amount": 10000,           # 固定金額手法用
        "fixed_ratio": 1.0,              # 固定比率手法用（1.0 = 100%）
        "volatility_risk_percentage": 0.02,  # ボラティリティ手法用
        "atr_period": 14                 # ATR計算期間
      }
    }
    # その他の戦略パラメータ...
  }
}
```

### 3.3 ポジションサイジング計算器の実装

```python
# backend/app/core/strategies/position_sizing/base_calculator.py
from abc import ABC, abstractmethod

class BasePositionSizingCalculator(ABC):
    @abstractmethod
    def calculate(self, current_equity: float, price: float, **kwargs) -> float:
        pass

# backend/app/core/strategies/position_sizing/calculators.py
class FixedAmountCalculator(BasePositionSizingCalculator):
    def calculate(self, current_equity: float, price: float, amount: float) -> float:
        return amount / price

class FixedRatioCalculator(BasePositionSizingCalculator):
    def calculate(self, current_equity: float, price: float, ratio: float) -> float:
        position_value = current_equity * ratio
        return position_value / price

class VolatilityBasedCalculator(BasePositionSizingCalculator):
    def calculate(self, current_equity: float, price: float, atr: float, risk_percentage: float) -> float:
        value_at_risk = current_equity * risk_percentage
        if atr == 0:
            return (current_equity * 0.01) / price  # フォールバック
        return value_at_risk / (atr * price)

class PositionSizingCalculator:
    def __init__(self, method: str, params: dict):
        self.method = method
        self.params = params
        self.calculator = self._create_calculator()

    def _create_calculator(self) -> BasePositionSizingCalculator:
        calculators = {
            "fixed_amount": FixedAmountCalculator(),
            "fixed_ratio": FixedRatioCalculator(),
            "volatility_based": VolatilityBasedCalculator()
        }
        if self.method not in calculators:
            raise ValueError(f"Unknown position sizing method: {self.method}")
        return calculators[self.method]

    def calculate(self, current_equity: float, price: float, **kwargs) -> float:
        if self.method == "fixed_amount":
            return self.calculator.calculate(current_equity, price,
                                           amount=self.params.get("fixed_amount"))
        elif self.method == "fixed_ratio":
            return self.calculator.calculate(current_equity, price,
                                           ratio=self.params.get("fixed_ratio", 1.0))
        elif self.method == "volatility_based":
            atr = kwargs.get("atr")
            risk_percentage = self.params.get("volatility_risk_percentage")
            if atr is None or risk_percentage is None:
                raise ValueError("ATR and volatility_risk_percentage are required")
            return self.calculator.calculate(current_equity, price,
                                           atr=atr, risk_percentage=risk_percentage)
```

### 3.4 BaseStrategy の拡張

```python
# backend/app/core/strategies/base_strategy.py に追加
def calculate_position_size(self, price: float, **kwargs) -> float:
    # 資金管理設定を取得
    money_management = getattr(self, 'money_management_config', None)

    if money_management:
        from .position_sizing.calculators import PositionSizingCalculator

        calculator = PositionSizingCalculator(
            method=money_management['method'],
            params=money_management['params']
        )

        # ボラティリティベースの場合はATRを計算
        if money_management['method'] == 'volatility_based':
            atr = self.calculate_atr(money_management['params'].get('atr_period', 14))
            return calculator.calculate(self.equity, price, atr=atr)
        else:
            return calculator.calculate(self.equity, price)
    else:
        # 既存のロジック（後方互換性）
        risk_percent = kwargs.get('risk_percent', 1.0)
        available_cash = self.equity
        risk_amount = available_cash * (risk_percent / 100)
        return risk_amount / price

def calculate_atr(self, period: int = 14) -> float:
    # ATR計算の実装
    if len(self.data) < period:
        return 0.0

    high_low = self.data.High - self.data.Low
    high_close = abs(self.data.High - self.data.Close.shift(1))
    low_close = abs(self.data.Low - self.data.Close.shift(1))

    true_range = pd.concat([high_low, high_close, low_close], axis=1).max(axis=1)
    return true_range.rolling(window=period).mean().iloc[-1]
```

### 3.5 テスト計画

```text
backend/tests/unit/strategies/position_sizing/
├── test_calculators.py
└── test_base_strategy_integration.py

backend/tests/integration/
└── test_backtest_with_money_management.py
```

## 4. 実装手順

### 4.1 Step 1: ポジションサイジング計算器の作成

1. `backend/app/core/strategies/position_sizing/` ディレクトリ作成
2. 基底クラスと 3 つの計算器を実装
3. ユニットテストの作成

### 4.2 Step 2: BaseStrategy の拡張

1. `calculate_position_size` メソッドの拡張
2. `calculate_atr` メソッドの追加
3. 資金管理設定の注入機能

### 4.3 Step 3: BacktestService の修正

1. 戦略クラス生成時に `money_management_config` を設定
2. パラメータの検証とデフォルト値設定

### 4.4 Step 4: 統合テスト

1. 各手法でのバックテスト実行テスト
2. エラーハンドリングの確認

## 5. 技術的考慮事項

### 5.1 エラーハンドリング

- パラメータ不正時の適切なエラーメッセージ
- ATR 計算不可能時のフォールバック機能

### 5.2 後方互換性

- 既存の戦略コードは変更なしで動作
- 資金管理設定がない場合は従来通りの動作

### 5.3 パフォーマンス

- ATR 計算の最適化
- エントリーごとの計算効率化

---

**実装予定**: 1-2 週間
**テスト期間**: 1 週間
