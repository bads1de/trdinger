# メモリーリーク分析レポート

## 概要
OptunaとAutoML機能使用時に発生するメモリーリークの原因を調査し、特定した問題点と解決策をまとめました。

## 🔍 特定されたメモリーリークの原因

### 1. グローバルインスタンスによるメモリ保持

#### 問題箇所
```python
# backend/app/services/ml/ml_training_service.py:685
ml_training_service = MLTrainingService(trainer_type="ensemble", automl_config=None)

# backend/app/services/ml/orchestration/background_task_manager.py:214
background_task_manager = BackgroundTaskManager()

# backend/app/services/ml/model_manager.py:494
model_manager = ModelManager()

# backend/app/services/ml/orchestration/ml_training_orchestration_service.py:28
training_status = {
    "is_training": False,
    "progress": 0,
    # ... その他の状態
}
```

#### 問題点
- グローバルインスタンスがアプリケーション終了まで保持される
- 内部で大量のモデルやデータを保持し続ける
- ガベージコレクションの対象にならない

### 2. AutoFeatCalculatorのメモリリーク

#### 問題箇所
```python
# backend/app/services/ml/feature_engineering/automl_features/autofeat_calculator.py:43
self.autofeat_model = None  # fit_transform後に大量メモリを保持
```

#### 問題点
- `fit_transform`実行後、`autofeat_model`が大量のメモリを保持
- `clear_model()`メソッドはあるが、適切なタイミングで呼び出されない
- AutoFeat内部の属性（`feateng_cols_`, `featsel_`, `model_`, `scaler_`）が残存

### 3. OptunaOptimizerのメモリリーク

#### 問題箇所
```python
# backend/app/services/optimization/optuna_optimizer.py:46
self.study: Optional[optuna.Study] = None
```

#### 問題点
- 最適化完了後も`study`オブジェクトを保持
- Optunaの内部データ（trials、sampler、pruner）が蓄積
- 明示的なクリーンアップメソッドが存在しない

### 4. EnsembleTrainerのメモリリーク

#### 問題箇所
```python
# backend/app/services/ml/ensemble/ensemble_trainer.py:46
self.ensemble_model = None  # 複数のモデルを保持
```

#### 問題点
- 複数のMLモデルを同時に保持
- アンサンブル学習で大量のモデルデータが蓄積
- `cleanup_resources`メソッドが不完全

### 5. TSFreshCalculatorのメモリリーク

#### 問題箇所
```python
# backend/app/services/ml/feature_engineering/automl_features/tsfresh_calculator.py:138
extracted_features = extract_features(...)  # 大量の特徴量データ
```

#### 問題点
- `extract_features`で大量のメモリを使用
- 特徴量データが適切に解放されない
- 並列処理（n_jobs）でメモリ使用量が増大

### 6. バックグラウンドタスクのリソース管理不備

#### 問題箇所
```python
# backend/app/services/ml/orchestration/background_task_manager.py:63-64
self._task_resources[task_id] = resources or []
self._cleanup_callbacks[task_id] = cleanup_callbacks or []
```

#### 問題点
- タスク終了後もリソースが残存する場合がある
- クリーンアップコールバックが完全に実行されない
- 例外発生時のリソース解放が不完全

### 7. DataPreprocessorのキャッシュ問題

#### 問題箇所
```python
# backend/app/utils/data_preprocessing.py:289
data_preprocessor = DataPreprocessor()
```

#### 問題点
- `DataPreprocessor`がグローバルインスタンスとして生成されている
- 内部で`imputer`や`scaler`のインスタンスをキャッシュとして保持（`self.imputers`, `self.scalers`）
- `clear_cache()`メソッドは存在するが、どこからも呼び出されておらず、キャッシュが解放されない

## 🚨 メモリリークの影響

### 症状
1. **メモリ使用量の継続的増加**
   - AutoML機能使用後にメモリが解放されない
   - 複数回の実行でメモリ使用量が累積

2. **パフォーマンス低下**
   - システム全体の動作が重くなる
   - 他のプロセスに影響を与える

3. **システム不安定化**
   - メモリ不足によるクラッシュの可能性
   - 長時間運用での問題発生

## 💡 解決策

### 1. グローバルインスタンスの改善

```python
# 解決策: ファクトリーパターンまたはコンテキストマネージャーの使用
class MLServiceFactory:
    @staticmethod
    def create_training_service(**kwargs):
        return MLTrainingService(**kwargs)
    
    @staticmethod
    def cleanup_service(service):
        if hasattr(service, 'cleanup_resources'):
            service.cleanup_resources()
```

### 2. AutoFeatCalculatorの改善

```python
# 解決策: 確実なリソース解放
def clear_model(self):
    if self.autofeat_model is not None:
        # 内部属性を個別にクリア
        for attr in ['feateng_cols_', 'featsel_', 'model_', 'scaler_']:
            if hasattr(self.autofeat_model, attr):
                setattr(self.autofeat_model, attr, None)
        
        # モデル自体をクリア
        self.autofeat_model = None
        
        # 強制ガベージコレクション
        import gc
        gc.collect()
```

### 3. OptunaOptimizerの改善

```python
# 解決策: 最適化完了後のクリーンアップ
def cleanup(self):
    if self.study is not None:
        # Studyの内部データをクリア
        self.study.trials.clear()
        self.study = None
        
        # ガベージコレクション
        import gc
        gc.collect()
```

### 4. コンテキストマネージャーの活用

```python
# 解決策: with文でのリソース管理
@contextmanager
def ml_training_context(**kwargs):
    service = MLTrainingService(**kwargs)
    try:
        yield service
    finally:
        service.cleanup_resources()

# 使用例
with ml_training_context() as service:
    result = service.train_model(data)
```

### 5. 定期的なメモリクリーンアップ

```python
# 解決策: 定期的なクリーンアップ処理
def periodic_memory_cleanup():
    # グローバルインスタンスのクリーンアップ
    if hasattr(ml_training_service, 'cleanup_resources'):
        ml_training_service.cleanup_resources()
    
    # バックグラウンドタスクのクリーンアップ
    background_task_manager.cleanup_all_tasks()
    
    # 強制ガベージコレクション
    import gc
    collected = gc.collect()
    logger.info(f"定期クリーンアップ完了: {collected}オブジェクト回収")
```

## 🔧 実装優先度

### 高優先度（即座に対応）
1. **OptunaOptimizerのクリーンアップメソッド追加**
2. **AutoFeatCalculatorの確実なリソース解放**
3. **グローバルインスタンスのクリーンアップ強化**

### 中優先度（次回リリース）
1. **コンテキストマネージャーの導入**
2. **ファクトリーパターンの実装**
3. **定期的なメモリクリーンアップ機能**

### 低優先度（長期的改善）
1. **アーキテクチャの根本的見直し**
2. **依存性注入パターンの導入**
3. **メモリ使用量監視システムの構築**

## 📊 検証方法

### メモリ使用量の監視
```python
import psutil
import gc

def monitor_memory_usage(operation_name):
    process = psutil.Process()
    before = process.memory_info().rss / 1024 / 1024
    
    # 操作実行
    yield
    
    gc.collect()
    after = process.memory_info().rss / 1024 / 1024
    diff = after - before
    
    print(f"{operation_name}: メモリ変化 {diff:+.2f}MB")
```

### テストケース
1. **AutoML機能の連続実行テスト**
2. **Optuna最適化の繰り返しテスト**
3. **長時間運用でのメモリ使用量監視**

## 📝 まとめ

メモリーリークの主要原因は以下の通りです：

1. **グローバルインスタンスによる永続的なメモリ保持**
2. **AutoFeat/Optuna/TSFreshのリソース管理不備**
3. **バックグラウンドタスクのクリーンアップ不完全**
4. **循環参照による解放阻害**

これらの問題を段階的に解決することで、メモリーリークを根本的に改善できます。
