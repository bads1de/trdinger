# backtesting.py 実装チェックリスト

## 🎯 実装状況サマリー

### ✅ **適切に実装済み**
- [x] BacktestService (backtesting.py使用)
- [x] SMACrossStrategy (標準的な実装)
- [x] 基本的なAPI設計
- [x] データベース統合

### ❌ **修正が必要**
- [ ] 独自実装の削除 (StrategyExecutor)
- [ ] データ形式の統一
- [ ] 指標ライブラリの統一
- [ ] エラーハンドリングの改善

---

## 🚨 緊急対応項目（1-2週間）

### 1. **重複実装の削除**

#### 削除対象ファイル
```bash
# 以下のファイルを削除
rm backend/backtest/engine/strategy_executor.py
rm backend/backtest/engine/indicators.py
rm -rf backend/backtest/engine/
```

#### 修正対象ファイル
- `backend/backtest/runner.py`
- `backend/app/api/backtest.py` (必要に応じて)

### 2. **runner.py の修正**

```python
# 変更前
from backtest.engine.strategy_executor import StrategyExecutor

# 変更後
from app.core.services.backtest_service import BacktestService

def run_backtest(config: dict) -> dict:
    """バックテストを実行"""
    try:
        # BacktestServiceを使用
        backtest_service = BacktestService()
        result = backtest_service.run_backtest(config)
        return result
    except Exception as e:
        return {'error': str(e)}
```

### 3. **データ形式の統一**

```python
# 全OHLCVデータの列名を統一
STANDARD_COLUMNS = ['Open', 'High', 'Low', 'Close', 'Volume']

def standardize_ohlcv_data(df):
    """OHLCV列名を標準化"""
    column_mapping = {
        'open': 'Open', 'high': 'High', 'low': 'Low',
        'close': 'Close', 'volume': 'Volume'
    }
    return df.rename(columns=column_mapping)
```

---

## 🔧 機能強化項目（2-4週間）

### 4. **TA-Libの導入**

```bash
# requirements.txtに追加
TA-Lib==0.4.25
```

```python
# 戦略での使用例
import talib

class ImprovedSMACrossStrategy(Strategy):
    n1 = 20
    n2 = 50
    
    def init(self):
        # TA-Libを使用
        self.sma1 = self.I(talib.SMA, self.data.Close, self.n1)
        self.sma2 = self.I(talib.SMA, self.data.Close, self.n2)
    
    def next(self):
        if crossover(self.sma1, self.sma2):
            self.buy()
        elif crossover(self.sma2, self.sma1):
            self.sell()
```

### 5. **最適化機能の拡張**

```python
# SAMBO optimizerの活用
class BacktestService:
    def optimize_strategy_advanced(self, config, optimization_params):
        """高度な最適化機能"""
        bt = Backtest(data, strategy_class, **backtest_params)
        
        stats = bt.optimize(
            method='sambo',  # モデルベース最適化
            max_tries=200,
            maximize='Sharpe Ratio',
            return_heatmap=True,
            **optimization_params
        )
        
        return stats
```

### 6. **エラーハンドリングの強化**

```python
class RobustBacktestService:
    def run_backtest(self, config):
        """堅牢なバックテスト実行"""
        try:
            # 設定検証
            self._validate_config(config)
            
            # データ取得・検証
            data = self._get_validated_data(config)
            
            # 戦略クラス生成・検証
            strategy_class = self._create_validated_strategy(config)
            
            # バックテスト実行
            bt = Backtest(data, strategy_class, **self._get_backtest_params(config))
            stats = bt.run()
            
            return self._format_results(stats, config)
            
        except ValueError as e:
            raise ValueError(f"Configuration error: {e}")
        except Exception as e:
            raise RuntimeError(f"Backtest execution failed: {e}")
```

---

## 🚀 高度な機能（1-2ヶ月）

### 7. **マルチタイムフレーム対応**

```python
from backtesting.lib import resample_apply

class MultiTimeFrameStrategy(Strategy):
    def init(self):
        # 日足データから週足指標を計算
        self.weekly_sma = resample_apply(
            'W-FRI', SMA, self.data.Close, 50
        )
        
        # 日足指標
        self.daily_sma = self.I(SMA, self.data.Close, 20)
    
    def next(self):
        # 複数時間軸の条件
        if (self.data.Close[-1] > self.daily_sma[-1] and
            self.daily_sma[-1] > self.weekly_sma[-1]):
            self.buy()
```

### 8. **ヒートマップ可視化**

```python
from backtesting.lib import plot_heatmaps

class BacktestService:
    def generate_optimization_heatmap(self, config):
        """最適化ヒートマップ生成"""
        bt = Backtest(data, strategy_class)
        
        heatmap = bt.optimize(
            n1=range(5, 50, 5),
            n2=range(20, 100, 10),
            constraint=lambda p: p.n1 < p.n2,
            return_heatmap=True
        )
        
        # ヒートマップ可視化
        plot_heatmaps(heatmap, filename='optimization_heatmap.html')
        
        return heatmap
```

### 9. **複数戦略比較**

```python
from backtesting.lib import MultiBacktest

class StrategyComparisonService:
    def compare_strategies(self, data, strategies, params):
        """複数戦略の比較"""
        results = {}
        
        for strategy_name, strategy_class in strategies.items():
            bt = Backtest(data, strategy_class, **params)
            stats = bt.run()
            results[strategy_name] = stats
        
        return self._generate_comparison_report(results)
```

---

## 🧪 テスト強化

### 10. **包括的テストスイート**

```python
# tests/test_backtest_service.py
import pytest
from backtesting.test import GOOG
from app.core.services.backtest_service import BacktestService

class TestBacktestService:
    def test_basic_backtest(self):
        """基本バックテストのテスト"""
        config = {
            'strategy_name': 'SMA_CROSS',
            'symbol': 'BTC/USDT',
            'timeframe': '1d',
            'start_date': '2023-01-01',
            'end_date': '2023-12-31',
            'initial_capital': 100000,
            'commission_rate': 0.001,
            'strategy_config': {
                'strategy_type': 'SMA_CROSS',
                'parameters': {'n1': 20, 'n2': 50}
            }
        }
        
        service = BacktestService()
        result = service.run_backtest(config)
        
        assert 'performance_metrics' in result
        assert 'total_return' in result['performance_metrics']
    
    def test_optimization(self):
        """最適化機能のテスト"""
        # 最適化テストの実装
        pass
    
    def test_error_handling(self):
        """エラーハンドリングのテスト"""
        # エラーケースのテスト
        pass
```

---

## 📊 パフォーマンス最適化

### 11. **大量データ処理対応**

```python
class OptimizedBacktestService:
    def run_large_dataset_backtest(self, config):
        """大量データ対応バックテスト"""
        # データの分割処理
        data_chunks = self._split_data(config)
        
        results = []
        for chunk in data_chunks:
            bt = Backtest(chunk, strategy_class)
            stats = bt.run()
            results.append(stats)
        
        # 結果の統合
        return self._merge_results(results)
```

### 12. **キャッシュ機能**

```python
from functools import lru_cache

class CachedBacktestService:
    @lru_cache(maxsize=100)
    def get_cached_indicators(self, data_hash, indicator_config):
        """指標計算結果のキャッシュ"""
        return self._calculate_indicators(data_hash, indicator_config)
```

---

## 📋 実装チェックポイント

### Phase 1: 緊急対応 ✅
- [ ] StrategyExecutor削除
- [ ] runner.py修正
- [ ] データ形式統一
- [ ] 基本テスト実行

### Phase 2: 機能強化 🔄
- [ ] TA-Lib導入
- [ ] 最適化機能拡張
- [ ] エラーハンドリング強化
- [ ] テストカバレッジ向上

### Phase 3: 高度な機能 🚀
- [ ] マルチタイムフレーム
- [ ] ヒートマップ可視化
- [ ] 複数戦略比較
- [ ] パフォーマンス最適化

---

## 🎯 成功指標

### 技術指標
- [ ] テストカバレッジ > 80%
- [ ] バックテスト実行時間 < 5秒 (10,000バー)
- [ ] メモリ使用量 < 1GB (大量データ)
- [ ] エラー率 < 1%

### 品質指標
- [ ] コード重複率 < 5%
- [ ] 循環的複雑度 < 10
- [ ] ドキュメントカバレッジ > 90%
- [ ] 静的解析エラー = 0

---

**このチェックリストに従って段階的に実装を進めることで、backtesting.pyライブラリを最大限活用した堅牢なバックテストシステムを構築できます。**
