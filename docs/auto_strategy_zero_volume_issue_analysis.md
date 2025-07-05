# オートストラテジー取引量0問題の根本原因分析

## 問題の概要

オートストラテジー機能で生成されたストラテジーを実行すると、取引量が必ず0になってしまう問題が発生している。

## 調査結果

### 1. 問題の根本原因

#### 1.1 パラメータ受け渡しの不整合

**GAエンジンでの処理 (`ga_engine.py:_evaluate_individual`)**:
```python
# 戦略クラス生成
strategy_class = self.strategy_factory.create_strategy_class(gene)

# バックテスト実行
backtest_config = self._fixed_backtest_config.copy()
backtest_config["strategy_class"] = strategy_class  # 直接クラスを設定

result = self.backtest_service.run_backtest(backtest_config)
```

**BacktestServiceでの処理 (`backtest_service.py:run_backtest`)**:
```python
# パラメータを取得して渡す
stats = bt.run(**config.get("strategy_config", {}).get("parameters", {}))
```

**問題**: GAエンジンからは`strategy_config`が渡されていないため、空の辞書`{}`がパラメータとして渡される。

#### 1.2 戦略クラス初期化の問題

**GeneratedStrategyクラスの初期化 (`strategy_factory.py`)**:
```python
def __init__(self, broker=None, data=None, params=None):
    super().__init__(broker, data, params)
    
    # strategy_geneパラメータを期待
    current_gene = getattr(self, "strategy_gene", None)
    self.gene = current_gene if current_gene is not None else gene
```

**問題**: `strategy_gene`パラメータが渡されないため、フォールバック処理に依存している。

#### 1.3 実装の一貫性不足

**テスト実行時の正しい実装 (`auto_strategy_service.py:test_strategy_generation`)**:
```python
test_config["strategy_config"] = {
    "strategy_type": "GENERATED_TEST",
    "parameters": {"strategy_gene": gene.to_dict()},
}
```

**問題**: GAエンジンでは異なるアプローチを使用しており、一貫性がない。

### 2. 影響範囲

- GA実行時の個体評価で取引量が0になる
- 最適化プロセスが正しく機能しない
- 生成された戦略の品質評価が不正確

### 3. 修正方針

#### 3.1 GAエンジンでのパラメータ設定修正
- `_evaluate_individual`メソッドで`strategy_config`と`parameters`を正しく設定
- `strategy_gene`パラメータを含める

#### 3.2 BacktestServiceでのパラメータ処理改善
- `strategy_class`が直接渡される場合と`strategy_config`が渡される場合の両方に対応

#### 3.3 デバッグ機能の強化
- 取引量計算の詳細ログを追加
- 条件評価の結果をログに出力

#### 3.4 統合テストの実装
- 修正後の動作を検証するテストを作成

## 次のステップ

1. GAエンジンでのパラメータ設定修正
2. BacktestServiceでのパラメータ処理改善
3. StrategyFactoryでのデバッグ強化
4. 統合テストの実装と実行

## 関連ファイル

- `backend/app/core/services/auto_strategy/engines/ga_engine.py`
- `backend/app/core/services/backtest_service.py`
- `backend/app/core/services/auto_strategy/factories/strategy_factory.py`
- `backend/app/core/services/auto_strategy/services/auto_strategy_service.py`
