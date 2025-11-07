# 特徴量評価スクリプト

このディレクトリには、機械学習モデルの特徴量を評価・分析するためのスクリプトが含まれています。

## 概要

特徴量エンジニアリングの品質を評価し、モデル性能への影響を分析するための包括的なツールセットを提供します。

## スクリプト一覧

### 1. analyze_feature_importance.py
- **目的**: 複数のML手法で特徴量重要度を統合分析
- **実行**: 
  ```bash
  python -m scripts.feature_evaluation.analyze_feature_importance
  ```
- **出力**:
  - `feature_importance_analysis.json` - 統合スコアと詳細分析結果

#### 詳細機能
- LightGBM、Permutation Importance、SHAP値による多角的評価
- 統合スコアによる総合評価
- 重要度ランキングと推奨削除候補の提示

---

### 2. benchmark_all_features.py
- **目的**: 全特徴量を用いたベンチマーク評価
- **実行**: 
  ```bash
  python -m scripts.feature_evaluation.benchmark_all_features
  ```
- **出力**:
  - `data/feature_evaluation/benchmark_results.json` - ベンチマーク結果

#### 詳細機能
- 全特徴量を使用した基準性能の計測
- TimeSeriesSplitによるクロスバリデーション
- RMSE、MAE、R²などの評価指標

---

### 3. evaluate_feature_performance.py
- **目的**: 全モデル（LightGBM、TabNet、XGBoost）での特徴量性能検証
- **実行**: 
  ```bash
  python -m scripts.feature_evaluation.evaluate_feature_performance
  python -m scripts.feature_evaluation.evaluate_feature_performance --models lightgbm
  python -m scripts.feature_evaluation.evaluate_feature_performance --models lightgbm xgboost
  python -m scripts.feature_evaluation.evaluate_feature_performance --models all
  ```
- **出力**:
  - `data/feature_evaluation/{model_name}_feature_performance_evaluation.json`
  - `data/feature_evaluation/{model_name}_performance_comparison.csv`
  - `data/feature_evaluation/all_models_feature_performance_evaluation.json`
  - `data/feature_evaluation/all_models_performance_comparison.csv`

#### 詳細機能
- DBから実データを取得して評価
- TimeSeriesSplitでクロスバリデーション
- 統合スコア下位N%削除の複数シナリオ評価
- モデル固有の特徴量重要度ベース削除
- 推奨事項の自動生成

#### コマンドライン引数
- `--models`: 評価するモデルを指定（lightgbm、xgboost、all）
- `--symbol`: 分析対象シンボル（デフォルト: BTC/USDT:USDT）
- `--limit`: データ取得件数（デフォルト: 2000）

---

### 4. evaluate_models.py
- **目的**: モデル性能の詳細評価と比較
- **実行**: 
  ```bash
  python -m scripts.feature_evaluation.evaluate_models
  ```
- **出力**:
  - `data/feature_evaluation/model_evaluation_results.json`

#### 詳細機能
- 複数モデルでの性能比較
- 特徴量セットごとの評価
- 詳細なメトリクス分析

---

### 5. integrate_feature_evaluation_results.py
- **目的**: 複数の評価結果を統合して分析レポート生成
- **実行**: 
  ```bash
  python -m scripts.feature_evaluation.integrate_feature_evaluation_results
  ```
- **出力**:
  - `data/feature_evaluation/integrated_evaluation_report.json`
  - 統合分析レポート

#### 詳細機能
- 複数評価の結果統合
- 総合推奨事項の生成
- 削減可能特徴量の特定

---

### 6. overfitting_analysis.py
- **目的**: 過学習の検出と分析
- **実行**: 
  ```bash
  python -m scripts.feature_evaluation.overfitting_analysis
  ```
- **出力**:
  - `data/feature_evaluation/overfitting_analysis_results.json`
  - 過学習診断レポート

#### 詳細機能
- 訓練/検証誤差の差分分析
- 学習曲線の生成
- 過学習リスクの定量評価
- 正則化パラメータの推奨

---

## 実行順序の推奨

特徴量評価を体系的に行う場合、以下の順序で実行することを推奨します：

1. **benchmark_all_features.py** - ベースライン性能の確立
2. **analyze_feature_importance.py** - 特徴量重要度の分析
3. **evaluate_feature_performance.py** - 全モデルでの性能検証
4. **overfitting_analysis.py** - 過学習リスクの評価
5. **integrate_feature_evaluation_results.py** - 総合レポート生成

## 出力ファイルの場所

全ての評価結果は以下のディレクトリに保存されます：
```
backend/data/feature_evaluation/
```

## 依存関係

以下のPythonパッケージが必要です：

- lightgbm
- xgboost

- scikit-learn
- pandas
- numpy
- shap

## 注意事項

1. **データベース接続**: 実データを使用するスクリプトはデータベースへの接続が必要です
2. **計算リソース**: 大規模な評価には相応の計算時間とメモリが必要です


## トラブルシューティング




```bash

```

### メモリ不足エラー

`--limit`パラメータでデータ取得件数を減らしてください：
```bash
python -m scripts.feature_evaluation.evaluate_feature_performance --limit 1000
```

## 関連ドキュメント

- [モデル評価レポート](../../docs/feature_evaluation/MODEL_EVALUATION_REPORT.md)
- [過学習分析レポート](../../docs/feature_evaluation/OVERFITTING_ANALYSIS_REPORT.md)

## 開発者向け情報

### コードスタイル

- Black（行長88文字）
- Isort（profile=black）
- MyPy（厳密な型チェック）
- Flake8

### テスト

```bash
pytest tests/feature/ -v