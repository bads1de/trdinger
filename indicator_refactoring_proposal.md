# テクニカル指標設定の管理方法の改善提案 (改訂版)

## 1. はじめに

現状のテクニカル指標に関する設定は、複数のファイルに分散しており、メンテナンス性の低下を招いています。この課題に対し、「指標の追加・修正は、**単一ファイルへの追記**で完結させる」ことを最終目標とした、より高度なリファクタリング計画を提案します。

この計画では、Pythonの**デコレータ**を活用し、指標自身がその定義（メタデータ）を持つ「自己登録」の仕組みを導入します。

## 2. 根本原因と解決方針

問題の根本原因は「関心事の分散」です。これを解決するため、指標の「計算ロジック」と「メタデータ」を物理的に同じ場所に記述し、それ以外の登録作業を完全に自動化します。

-   **方針:**
    1.  **デコレータの導入**: 指標の計算ロジック（関数）に、そのメタデータ（名前、カテゴリ、パラメータ等）を付与するデコレータを作成します。
    2.  **レジストリの自動構築**: アプリケーション起動時に、指定されたディレクトリ（`technical_indicators/`）をスキャンし、デコレータが付与された指標を自動的に発見・登録する仕組みを構築します。

## 3. 提案する具体的な設計

### 3.1. `indicator_registry.py` の役割変更

このファイルは、手動で指標を登録する場所ではなく、**指標を登録するためのツール（デコレータ）**と、**自動登録された指標を格納する器**を提供します。

**`indicator_registry.py` の実装イメージ:**

```python
from dataclasses import dataclass, field
from typing import List, Dict, Any, Callable, Optional
import pkgutil
import importlib
from . import technical_indicators # インポート対象のパッケージ

@dataclass
class IndicatorParameter:
    name: str
    type: str
    default: Any
    description: str

@dataclass
class IndicatorDefinition:
    name: str
    category: str
    description: str
    function: Callable
    params: List[IndicatorParameter] = field(default_factory=list)

# グローバルな指標レジストリ
INDICATOR_REGISTRY: Dict[str, IndicatorDefinition] = {}

def register_indicator(name: str, category: str, description: str = "", params: Optional[List[IndicatorParameter]] = None):
    """
    指標計算関数をレジストリに登録するためのデコレータ
    """
    def decorator(func: Callable):
        definition = IndicatorDefinition(
            name=name,
            category=category,
            description=description,
            function=func,
            params=params or []
        )
        if name in INDICATOR_REGISTRY:
            raise ValueError(f"Indicator '{name}' is already registered.")
        INDICATOR_REGISTRY[name] = definition
        return func
    return decorator

def autodiscover_indicators():
    """
    technical_indicatorsパッケージ内のモジュールを自動インポートし、
    デコレータによる指標の登録を実行する。
    """
    if INDICATOR_REGISTRY: # 既に登録済みの場合はスキップ
        return

    path = technical_indicators.__path__
    name = technical_indicators.__name__

    for _, module_name, _ in pkgutil.iter_modules(path, name + "."):
        importlib.import_module(module_name)

# アプリケーション初期化時に一度だけ呼び出す
autodiscover_indicators()
```

### 3.2. 計算ロジックファイル (`technical_indicators/*.py`)

各指標の計算関数に、新しく作成した `@register_indicator` デコレータを付けるだけです。

**`technical_indicators/momentum.py` の実装イメージ:**

```python
import pandas as pd
from ..indicator_registry import register_indicator, IndicatorParameter

@register_indicator(
    name="RSI",
    category="Momentum",
    description="Relative Strength Index. A momentum oscillator that measures the speed and change of price movements.",
    params=[
        IndicatorParameter(name="length", type="int", default=14, description="The look-back period.")
    ]
)
def calculate_rsi(data: pd.DataFrame, length: int = 14) -> pd.Series:
    """RSIの計算ロジック"""
    # ...（計算ロジックはここに記述）
    close = data['close']
    delta = close.diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=length).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=length).mean()
    rs = gain / loss
    rsi = 100 - (100 / (1 + rs))
    return rsi

# 他のMomentum系指標も同様にデコレータを付けて定義
# @register_indicator(...)
# def calculate_stoch(...):
```

### 3.3. アプリケーション初期化処理

FastAPIなどのアプリケーションのエントリーポイントで、一度だけ `autodiscover_indicators()` を呼び出す処理を追加します。これにより、サーバー起動時にすべての指標が自動的に登録されます。

## 4. 新しいワークフロー

**新しい指標「MACD」を追加する場合:**

1.  **`technical_indicators/trend.py`** を開きます。
2.  ファイルの末尾に、計算ロジックと `@register_indicator` デコレータを追記します。

    ```python
    # trend.py の末尾に追記

    @register_indicator(
        name="MACD",
        category="Trend",
        description="Moving Average Convergence Divergence.",
        params=[
            IndicatorParameter(name="fast_period", type="int", default=12, description="Fast EMA period."),
            IndicatorParameter(name="slow_period", type="int", default=26, description="Slow EMA period."),
            IndicatorParameter(name="signal_period", type="int", default=9, description="Signal line EMA period.")
        ]
    )
    def calculate_macd(data: pd.DataFrame, fast_period: int = 12, slow_period: int = 26, signal_period: int = 9) -> pd.DataFrame:
        """MACDの計算ロジック"""
        # ...（計算ロジックを記述）
        fast_ema = data['close'].ewm(span=fast_period, adjust=False).mean()
        slow_ema = data['close'].ewm(span=slow_period, adjust=False).mean()
        macd_line = fast_ema - slow_ema
        signal_line = macd_line.ewm(span=signal_period, adjust=False).mean()
        histogram = macd_line - signal_line
        return pd.DataFrame({'MACD': macd_line, 'Signal': signal_line, 'Histogram': histogram})
    ```

これだけで、新しい指標の追加は完了です。他のファイル（レジストリ等）を手動で編集する必要は一切ありません。

## 5. 期待されるメリット

-   **究極のメンテナンス性**: 指標の追加・修正は、ロジックが記述されたファイルへの追記・修正のみで完結します。
-   **高い可読性と凝集度**: ロジックとその定義が常に一体となっているため、コードが非常に読みやすくなります。
-   **プラグインのような拡張性**: 指標は完全に自己完結したコンポーネントとなり、追加や（将来的には）無効化が容易になります。
-   **堅牢性の向上**: 手動での登録作業がなくなるため、登録漏れやタイプミスといったヒューマンエラーを撲滅できます。

## 6. 次のステップ

本提案にご同意いただけるようでしたら、以下の手順でリファクタリング作業を進めます。

1.  `indicator_registry.py` を作成し、`@register_indicator` デコレータと自動検出の仕組み (`autodiscover_indicators`) を実装します。
2.  アプリケーションの初期化処理に `autodiscover_indicators()` の呼び出しを追加します。
3.  既存の指標計算関数にデコレータを付与し、新しい仕組みに移行します。（まずは1〜2個の指標でテストします）
4.  `indicator_orchestrator.py` 等の関連モジュールが、新しい `INDICATOR_REGISTRY` を参照するように修正します。
5.  すべての指標の移行完了後、古い設定ファイル (`indicator_definitions.py`, `indicator_config.py`) を削除します。

ご確認のほど、よろしくお願いいたします。