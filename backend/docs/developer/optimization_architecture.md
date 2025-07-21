# ハイパーパラメータ最適化アーキテクチャ

## 概要

このドキュメントでは、ハイパーパラメータ最適化システムのアーキテクチャ、設計思想、実装詳細について説明します。

## アーキテクチャ概要

```
┌─────────────────┐    ┌─────────────────┐    ┌─────────────────┐
│   Frontend      │    │   API Layer     │    │  Service Layer  │
│                 │    │                 │    │                 │
│ OptimizationUI  │───▶│ MLTrainingAPI   │───▶│ MLTrainingService│
│                 │    │                 │    │                 │
└─────────────────┘    └─────────────────┘    └─────────────────┘
                                                        │
                                                        ▼
                                               ┌─────────────────┐
                                               │ Optimization    │
                                               │ Layer           │
                                               │                 │
                                               │ ┌─────────────┐ │
                                               │ │BaseOptimizer│ │
                                               │ └─────────────┘ │
                                               │        │        │
                                               │ ┌──────┴──────┐ │
                                               │ │   Factory   │ │
                                               │ └─────────────┘ │
                                               │        │        │
                                               │ ┌──────┴──────┐ │
                                               │ │ Concrete    │ │
                                               │ │ Optimizers  │ │
                                               │ └─────────────┘ │
                                               └─────────────────┘
```

## 設計原則

### 1. 単一責任原則 (SRP)
- 各オプティマイザーは特定の最適化手法のみを担当
- ファクトリーパターンでオプティマイザー生成を分離
- サービス層で最適化ワークフローを統合

### 2. 開放閉鎖原則 (OCP)
- BaseOptimizerを継承して新しい最適化手法を追加可能
- 既存コードを変更せずに機能拡張

### 3. リスコフ置換原則 (LSP)
- 全てのオプティマイザーはBaseOptimizerと置換可能
- 統一されたインターフェース

### 4. インターフェース分離原則 (ISP)
- 最小限のインターフェースを定義
- 不要な依存関係を排除

### 5. 依存性逆転原則 (DIP)
- 抽象に依存し、具象に依存しない
- OptimizerFactoryによる依存性注入

## コンポーネント詳細

### BaseOptimizer (抽象クラス)

```python
class BaseOptimizer(ABC):
    @abstractmethod
    def optimize(
        self,
        objective_function: Callable[[Dict[str, Any]], float],
        parameter_space: Dict[str, ParameterSpace],
        n_calls: int = 50,
        **kwargs: Any
    ) -> OptimizationResult:
        pass
```

**責任:**
- 共通インターフェースの定義
- パラメータ検証
- ログ出力
- 結果オブジェクト生成

### BayesianOptimizer

```python
class BayesianOptimizer(BaseOptimizer):
    def optimize(self, ...):
        # scikit-optimizeを使用したベイジアン最適化
        return self._optimize_with_skopt(...)
```

**特徴:**
- ガウス過程による効率的な最適化
- 獲得関数による次の評価点選択
- 少ない試行回数で高精度

### GridSearchOptimizer

```python
class GridSearchOptimizer(BaseOptimizer):
    def optimize(self, ...):
        # パラメータグリッドを生成して網羅的探索
        param_grid = self._generate_parameter_grid(...)
        return self._execute_grid_search(...)
```

**特徴:**
- パラメータ空間の網羅的探索
- 確実な最適解発見
- 組み合わせ爆発の制御

### RandomSearchOptimizer

```python
class RandomSearchOptimizer(BaseOptimizer):
    def optimize(self, ...):
        # ランダムサンプリングによる探索
        return self._execute_random_search(...)
```

**特徴:**
- ランダムサンプリング
- 早期停止機能
- 高次元パラメータ空間に対応

### OptimizerFactory

```python
class OptimizerFactory:
    SUPPORTED_METHODS = {
        "bayesian": BayesianOptimizer,
        "grid": GridSearchOptimizer,
        "random": RandomSearchOptimizer,
    }
    
    @classmethod
    def create_optimizer(cls, method: str) -> BaseOptimizer:
        # 手法名に基づいてオプティマイザーを生成
```

**責任:**
- オプティマイザーの生成
- 手法名の正規化
- 設定の適用

## データフロー

### 1. リクエスト処理

```
Frontend Request
    ↓
API Validation
    ↓
OptimizationSettings Creation
    ↓
MLTrainingService.train_model()
```

### 2. 最適化実行

```
MLTrainingService
    ↓
OptimizerFactory.create_optimizer()
    ↓
optimizer.optimize()
    ↓
objective_function() (multiple calls)
    ↓
OptimizationResult
```

### 3. 目的関数実行

```
objective_function(params)
    ↓
BaseMLTrainer.train_model()
    ↓
Model Training & Evaluation
    ↓
F1 Score (return value)
```

## エラーハンドリング

### 1. バリデーションエラー
- パラメータ空間の妥当性検証
- 目的関数の呼び出し可能性確認
- 設定値の範囲チェック

### 2. 実行時エラー
- 目的関数実行エラーの捕捉
- ペナルティスコアの返却
- ログ出力とエラー追跡

### 3. リソースエラー
- メモリ不足の検出
- タイムアウト処理
- 適切なクリーンアップ

## パフォーマンス最適化

### 1. メモリ管理
- 最適化履歴の効率的な保存
- 不要なオブジェクトの適切な削除
- ガベージコレクションの考慮

### 2. 計算効率
- 目的関数の軽量化
- 並列処理の回避（リソース競合防止）
- 早期停止による無駄な計算の削減

### 3. スケーラビリティ
- パラメータ数に応じた試行回数調整
- 大規模データセットでの最適化制限
- 段階的な最適化アプローチ

## テスト戦略

### 1. ユニットテスト
- 各オプティマイザーの個別テスト
- パラメータ空間の検証テスト
- エラーハンドリングのテスト

### 2. 統合テスト
- サービス層の統合テスト
- API層との連携テスト
- エンドツーエンドテスト

### 3. パフォーマンステスト
- 実行時間の測定
- メモリ使用量の監視
- スケーラビリティテスト

## 拡張性

### 新しい最適化手法の追加

1. **BaseOptimizerを継承**
```python
class NewOptimizer(BaseOptimizer):
    def optimize(self, ...):
        # 新しい最適化ロジック
        pass
```

2. **FactoryにMethodを追加**
```python
SUPPORTED_METHODS = {
    # 既存の手法
    "new_method": NewOptimizer,
}
```

3. **テストの追加**
```python
def test_new_optimizer():
    # 新しいオプティマイザーのテスト
    pass
```

### 新しいパラメータ型の追加

1. **ParameterSpaceの拡張**
```python
@dataclass
class ParameterSpace:
    type: str  # "real", "integer", "categorical", "new_type"
    # 新しい型用のフィールド追加
```

2. **各オプティマイザーでの対応**
```python
def _sample_random_parameters(self, parameter_space):
    # 新しい型の処理を追加
    elif param_config.type == "new_type":
        # 新しい型のサンプリングロジック
```

## セキュリティ考慮事項

### 1. 入力検証
- パラメータ範囲の妥当性確認
- 悪意のある入力の検出
- SQLインジェクション対策

### 2. リソース制限
- 最大試行回数の制限
- 実行時間の制限
- メモリ使用量の制限

### 3. ログ管理
- 機密情報の除外
- 適切なログレベル設定
- 監査ログの記録

## 監視とメトリクス

### 1. パフォーマンスメトリクス
- 最適化実行時間
- メモリ使用量
- CPU使用率

### 2. 品質メトリクス
- 最適化成功率
- 収束率
- スコア改善率

### 3. 運用メトリクス
- API呼び出し回数
- エラー発生率
- ユーザー利用状況
