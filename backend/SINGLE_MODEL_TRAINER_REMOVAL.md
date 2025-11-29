# SingleModelTrainer 削除に関する注意事項

## 概要

SingleModelTrainer は廃止され、全て EnsembleTrainer で統一されました。
単一モデルが必要な場合は、EnsembleTrainer に`models=["lightgbm"]`のように渡すことで実現できます。

## 移行完了したファイル

- ✅ `ml_training_service.py` - SingleModelTrainer への依存を削除
- ✅ `optimization_service.py` - SingleModelTrainer への依存を削除
- ✅ `ensemble_trainer.py` - 単一モデルモード対応を追加

## SingleModelTrainer 削除可能条件

以下の確認が完了したら削除可能：

- [x] 新しいテストが全てパス
- [x] 既存のテストが全てパス（SingleModelTrainer 関連を除く）
- [ ] test_trainer_refactoring.py から SingleModelTrainer インポートを削除
- [ ] single_model_trainer.py ファイルを削除
- [ ] **init**.py から SingleModelTrainer エクスポートを削除

## 後方互換性

以下のインターフェースは引き続き動作:

```python
# これまでの使い方（内部でEnsembleTrainerに変換される）
service = MLTrainingService(trainer_type="single", single_model_config={"model_type": "lightgbm"})

# 新しい推奨の使い方
service = MLTrainingService(
    trainer_type="ensemble",
    ensemble_config={"models": ["lightgbm"]}
)
```
