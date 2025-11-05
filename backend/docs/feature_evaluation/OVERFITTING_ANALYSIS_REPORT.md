# 過学習分析レポート

## 概要

108 特徴量を使用した 3 モデル（TabNet、XGBoost、LightGBM）の過学習を詳細に分析しました。

## 1. 過学習度の定量評価

過学習度 = (Train_Score - Test_Score) / Train_Score × 100

### 評価基準

- 0-5%: 問題なし ✅
- 5-10%: 軽度の過学習 ⚠️
- 10-20%: 中程度の過学習 ⚠️⚠️
- 20%+: 深刻な過学習 ❌

### 各モデルの過学習度

| モデル   | Train ROC-AUC | Test ROC-AUC | 過学習度 | 評価        |
| -------- | ------------- | ------------ | -------- | ----------- |
| XGBoost  | 1.0000        | 0.9959       | 0.41%    | ✅ 問題なし |
| LightGBM | 1.0000        | 0.9955       | 0.45%    | ✅ 問題なし |
| TabNet   | 0.9811        | 0.9622       | 1.92%    | ✅ 問題なし |

## 2. モデルごとの過学習診断

### XGBoost

**過学習リスクレベル**: ✅ 低

**スコア詳細**:

- Training ROC-AUC: 1.0000
- Validation ROC-AUC: 0.9926
- Test ROC-AUC: 0.9959
- Train-Test Gap: 0.0041

**主要な原因分析**:

- モデルは適切に汎化できています

**推奨される対策**:

- 現在のハイパーパラメータは適切です
- 更なる性能向上のため、特徴量エンジニアリングを検討

### LightGBM

**過学習リスクレベル**: ✅ 低

**スコア詳細**:

- Training ROC-AUC: 1.0000
- Validation ROC-AUC: 0.9917
- Test ROC-AUC: 0.9955
- Train-Test Gap: 0.0045

**主要な原因分析**:

- モデルは適切に汎化できています

**推奨される対策**:

- 現在のハイパーパラメータは適切です
- 更なる性能向上のため、特徴量エンジニアリングを検討

### TabNet

**過学習リスクレベル**: ✅ 低

**スコア詳細**:

- Training ROC-AUC: 0.9811
- Validation ROC-AUC: 0.9731
- Test ROC-AUC: 0.9622
- Train-Test Gap: 0.0188

**主要な原因分析**:

- モデルは適切に汎化できています

**推奨される対策**:

- 現在のハイパーパラメータは適切です
- 更なる性能向上のため、特徴量エンジニアリングを検討

## 3. クロスバリデーション結果

5-Fold Cross-Validation の結果:

| モデル   | 平均スコア | 標準偏差 | 安定性 |
| -------- | ---------- | -------- | ------ |
| XGBoost  | 0.9954     | 0.0017   | ✅ 高  |
| LightGBM | 0.9953     | 0.0018   | ✅ 高  |

## 4. 最適な特徴量数の推奨

### XGBoost

**推奨特徴量数**: 20

- Test ROC-AUC: 0.9977
- Train-Test Gap: 0.0023

**特徴量数別の性能**:

| 特徴量数 | Train ROC-AUC | Test ROC-AUC | Gap    |
| -------- | ------------- | ------------ | ------ |
| 20       | 1.0000        | 0.9977       | 0.0023 |
| 40       | 1.0000        | 0.9973       | 0.0027 |
| 60       | 1.0000        | 0.9968       | 0.0032 |
| 80       | 1.0000        | 0.9962       | 0.0038 |
| 108      | 1.0000        | 0.9968       | 0.0032 |

### LightGBM

**推奨特徴量数**: 20

- Test ROC-AUC: 0.9975
- Train-Test Gap: 0.0025

**特徴量数別の性能**:

| 特徴量数 | Train ROC-AUC | Test ROC-AUC | Gap    |
| -------- | ------------- | ------------ | ------ |
| 20       | 1.0000        | 0.9975       | 0.0025 |
| 40       | 1.0000        | 0.9969       | 0.0031 |
| 60       | 1.0000        | 0.9969       | 0.0031 |
| 80       | 1.0000        | 0.9969       | 0.0031 |
| 108      | 1.0000        | 0.9961       | 0.0039 |

## 5. 正則化の効果

### XGBoost

| reg_alpha | Train ROC-AUC | Test ROC-AUC | Gap    |
| --------- | ------------- | ------------ | ------ |
| 0         | 1.0000        | 0.9969       | 0.0031 |
| 0.01      | 1.0000        | 0.9965       | 0.0035 |
| 0.1       | 1.0000        | 0.9963       | 0.0037 |
| 1.0       | 1.0000        | 0.9962       | 0.0038 |

### LightGBM

| reg_alpha | Train ROC-AUC | Test ROC-AUC | Gap    |
| --------- | ------------- | ------------ | ------ |
| 0         | 1.0000        | 0.9965       | 0.0035 |
| 0.01      | 1.0000        | 0.9964       | 0.0036 |
| 0.1       | 1.0000        | 0.9970       | 0.0030 |
| 1.0       | 1.0000        | 0.9968       | 0.0032 |

## 6. 過学習リスクの総合評価

### ✅ 総合評価: 優良

全てのモデルが適切に汎化できており、過学習のリスクは低いです。

## 7. 推奨される対策（優先度順）

1. **正則化パラメータの調整**

   - XGBoost: reg_alpha=0.1, reg_lambda=0.1 を試す
   - LightGBM: reg_alpha=0.1, reg_lambda=0.1 を試す

2. **特徴量選択**

   - 重要度の低い特徴量を削除し、60-80 特徴量に削減
   - 相関の高い特徴量を統合

3. **モデルの複雑度調整**

   - XGBoost: max_depth を 4-5 に削減
   - LightGBM: num_leaves を 23-31 に調整

4. **データ拡張**

   - より多くのトレーニングデータを収集
   - データ拡張技術の適用を検討

5. **アンサンブル手法**
   - 複数モデルのアンサンブルで汎化性能を向上
   - スタッキングやブレンディングの活用

## 8. 生成されたファイル

### グラフ

1. `overfitting_analysis_plots/train_val_test_comparison.png` - トレーニング/検証/テストスコア比較
2. `overfitting_analysis_plots/learning_curves.png` - 学習曲線
3. `overfitting_analysis_plots/validation_curves_xgboost.png` - XGBoost バリデーションカーブ
4. `overfitting_analysis_plots/validation_curves_lightgbm.png` - LightGBM バリデーションカーブ
5. `overfitting_analysis_plots/cv_score_distribution.png` - CV スコア分布
6. `overfitting_analysis_plots/regularization_effect.png` - 正則化の効果
7. `overfitting_analysis_plots/feature_count_vs_overfitting.png` - 特徴量数と過学習の関係

### データ

- `overfitting_analysis_results.json` - 全分析結果（JSON 形式）
