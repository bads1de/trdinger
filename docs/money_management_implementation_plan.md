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
    # ATR計算の実装（backtesting.py用）
    if len(self.data) < period:
        return 0.0

    # backtesting.pyのデータ形式に対応
    from .indicators import ATR

    # ATRインジケーターを使用（既に計算済みの場合は再利用）
    if not hasattr(self, '_atr_cache'):
        self._atr_cache = {}

    if period not in self._atr_cache:
        # ATRを計算してキャッシュ
        atr_values = self.I(ATR, self.data.High, self.data.Low, self.data.Close, period)
        self._atr_cache[period] = atr_values

    # 最新のATR値を返す
    atr_values = self._atr_cache[period]
    return float(atr_values[-1]) if len(atr_values) > 0 else 0.0
```

### 3.5 backtesting.py での動的ポジションサイジング実装

```python
# 戦略クラスでの使用例（SMACrossStrategy）
def next(self):
    # 現在価格を取得
    current_price = self.data.Close[-1]

    # ゴールデンクロス: 短期SMAが長期SMAを上抜け → 買いシグナル
    if crossover(self.sma1, self.sma2):
        # 動的ポジションサイズを計算
        position_size = self.calculate_position_size(current_price)

        # サイズを指定して買い注文
        self.buy(size=position_size)

    # デッドクロス: 短期SMAが長期SMAを下抜け → 売りシグナル
    elif crossover(self.sma2, self.sma1):
        # 現在のポジションを決済
        if self.position:
            self.position.close()
```

### 3.6 BacktestService の修正

```python
# backend/app/core/services/backtest_service.py の _create_strategy_class メソッド修正
def _create_strategy_class(self, strategy_config: Dict[str, Any]) -> Type[Strategy]:
    """
    戦略設定から戦略クラスを取得し、パラメータを設定
    """
    strategy_type = strategy_config["strategy_type"]
    parameters = strategy_config.get("parameters", {})

    # 資金管理設定を抽出
    money_management = parameters.get("money_management")

    if strategy_type == "SMA_CROSS":
        # 戦略パラメータを設定
        if "n1" in parameters:
            SMACrossStrategy.n1 = parameters["n1"]
        if "n2" in parameters:
            SMACrossStrategy.n2 = parameters["n2"]

        # 資金管理設定を注入
        if money_management:
            SMACrossStrategy.money_management_config = money_management

        return SMACrossStrategy

    # 他の戦略も同様に処理...
```

### 3.7 SMACrossStrategy の BaseStrategy 継承への修正

```python
# backend/app/core/strategies/sma_cross_strategy.py の修正
from .base_strategy import BaseStrategy

class SMACrossStrategy(BaseStrategy):  # BaseStrategy を継承
    """
    SMAクロス戦略（資金管理対応版）
    """

    # デフォルトパラメータ
    n1 = 20
    n2 = 50

    def init(self):
        """指標の初期化"""
        close = self.data.Close
        self.sma1 = self.I(SMA, close, self.n1)
        self.sma2 = self.I(SMA, close, self.n2)

    def next(self):
        """売買ロジック（資金管理対応）"""
        current_price = self.data.Close[-1]

        # ゴールデンクロス
        if crossover(self.sma1, self.sma2):
            # 動的ポジションサイズを計算
            position_size = self.calculate_position_size(current_price)
            self.buy(size=position_size)

        # デッドクロス
        elif crossover(self.sma2, self.sma1):
            if self.position:
                self.position.close()
```

### 3.8 テスト計画

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

### 4.3 Step 3: SMACrossStrategy の修正

1. BaseStrategy を継承するよう変更
2. 動的ポジションサイジングの実装
3. 資金管理設定の対応

### 4.4 Step 4: BacktestService の修正

1. 戦略クラス生成時に `money_management_config` を設定
2. パラメータの検証とデフォルト値設定

### 4.5 Step 5: 統合テスト

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

## 6. SMA Cross 戦略での実装テスト例

### 6.1 固定金額ポジションサイジングのテスト

```python
# テスト用のリクエスト設定
backtest_request = {
    "strategy_name": "SMA_CROSS_FIXED_AMOUNT",
    "symbol": "BTCUSDT",
    "timeframe": "1h",
    "start_date": "2024-01-01",
    "end_date": "2024-03-01",
    "initial_capital": 100000,
    "commission_rate": 0.001,
    "strategy_config": {
        "strategy_type": "SMA_CROSS",
        "parameters": {
            "n1": 20,
            "n2": 50,
            "money_management": {
                "method": "fixed_amount",
                "params": {
                    "fixed_amount": 10000  # 毎回1万円分を投資
                }
            }
        }
    }
}
```

### 6.2 固定比率ポジションサイジングのテスト

```python
# 資産の50%を投資する設定
"money_management": {
    "method": "fixed_ratio",
    "params": {
        "fixed_ratio": 0.5  # 50%
    }
}
```

### 6.3 ボラティリティベースポジションサイジングのテスト

```python
# ATRベースのリスク管理
"money_management": {
    "method": "volatility_based",
    "params": {
        "volatility_risk_percentage": 0.02,  # 2%のリスク
        "atr_period": 14
    }
}
```

### 6.4 期待される結果

1. **固定金額**: 各取引で一定額を投資、資産増減に関係なく安定
2. **固定比率**: 資産増加に伴いポジションサイズも増加、複利効果
3. **ボラティリティベース**: 市場の変動に応じてリスクを調整、より安全

---

**実装予定**: 1-2 週間
**テスト期間**: 1 週間
