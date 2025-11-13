# Optuna最適化 - クイックスタートガイド

このガイドでは、Optunaを使用したハイパーパラメータ最適化の基本的な使い方を説明します。

---

## 目次

1. [基本的な使い方](#基本的な使い方)
2. [ユースケース別コマンド](#ユースケース別コマンド)
3. [パラメータ設定ガイド](#パラメータ設定ガイド)
4. [結果の確認方法](#結果の確認方法)
5. [トラブルシューティング](#トラブルシューティング)
6. [パフォーマンスチューニング](#パフォーマンスチューニング)

---

## 基本的な使い方

### 1. ベースライン測定

最初に固定パラメータで性能を測定します。これが比較基準となります。

```bash
cd backend

# LightGBMのベースライン
python -m scripts.feature_evaluation.evaluate_feature_performance \
    --models lightgbm
```

**実行時間**: 約5-10秒  
**出力**: `backend/scripts/results/feature_analysis/lightgbm_feature_performance_evaluation.json`

### 2. Optuna最適化実行

ベースラインを取得したら、Optuna最適化を実行します。

```bash
# 標準設定（50試行）
python -m scripts.feature_evaluation.evaluate_feature_performance \
    --models lightgbm \
    --enable-optuna \
    --n-trials 50
```

**実行時間**: 約30-60秒（試行回数に依存）  
**推奨**: 初回は30試行から開始

### 3. 結果の比較

最適化結果とベースラインを比較します。

```bash
# 結果ファイルの確認
cat backend/scripts/results/feature_analysis/lightgbm_performance_comparison.csv
```

結果には以下が含まれます：
- ベースライン性能
- 最適化後の性能
- 改善率
- ベストパラメータ

---

## ユースケース別コマンド

### クイック検証（5-10分）

開発中やデバッグ時に使用。

```bash
python -m scripts.feature_evaluation.evaluate_feature_performance \
    --models lightgbm \
    --enable-optuna \
    --n-trials 20 \
    --limit 1000 \
    --optuna-timeout 600
```

**設定**:
- 試行回数: 20回（少ない）
- データ件数: 1000件（少ない）
- タイムアウト: 10分

**用途**: 
- 実装の動作確認
- 設定のテスト
- 素早いフィードバック

### 標準評価（20-30分）

通常の評価に使用。バランスの取れた設定。

```bash
python -m scripts.feature_evaluation.evaluate_feature_performance \
    --models lightgbm \
    --enable-optuna \
    --n-trials 50 \
    --symbol BTC/USDT:USDT \
    --timeframe 1h \
    --limit 2000 \
    --optuna-timeout 1800
```

**設定**:
- 試行回数: 50回（標準）
- データ件数: 2000件（標準）
- タイムアウト: 30分

**用途**:
- 通常の性能評価
- 本番前の検証
- モデル比較

### 詳細評価（1-2時間）

最高品質の最適化が必要な場合に使用。

```bash
python -m scripts.feature_evaluation.evaluate_feature_performance \
    --models lightgbm xgboost \
    --enable-optuna \
    --n-trials 100 \
    --symbol BTC/USDT:USDT \
    --timeframe 1h \
    --limit 3000 \
    --optuna-timeout 7200
```

**設定**:
- 試行回数: 100回（多い）
- データ件数: 3000件（多い）
- タイムアウト: 2時間
- 複数モデル: LightGBM + XGBoost

**用途**:
- 本番環境用の最適化
- 最高精度の追求
- 複数モデルの徹底比較

### 複数シンボル評価

複数の暗号通貨ペアで評価する場合。

```bash
# BTC
python -m scripts.feature_evaluation.evaluate_feature_performance \
    --models lightgbm \
    --enable-optuna \
    --n-trials 50 \
    --symbol BTC/USDT:USDT

# ETH
python -m scripts.feature_evaluation.evaluate_feature_performance \
    --models lightgbm \
    --enable-optuna \
    --n-trials 50 \
    --symbol ETH/USDT:USDT

# SOL
python -m scripts.feature_evaluation.evaluate_feature_performance \
    --models lightgbm \
    --enable-optuna \
    --n-trials 50 \
    --symbol SOL/USDT:USDT
```

**用途**:
- シンボル別の最適パラメータ発見
- 汎化性能の確認
- マルチアセット戦略の構築

### 複数タイムフレーム評価

異なる時間軸で評価する場合。

```bash
# 1時間足
python -m scripts.feature_evaluation.evaluate_feature_performance \
    --models lightgbm \
    --enable-optuna \
    --timeframe 1h

# 4時間足
python -m scripts.feature_evaluation.evaluate_feature_performance \
    --models lightgbm \
    --enable-optuna \
    --timeframe 4h

# 1日足
python -m scripts.feature_evaluation.evaluate_feature_performance \
    --models lightgbm \
    --enable-optuna \
    --timeframe 1d
```

**用途**:
- タイムフレーム別の最適化
- スキャルピング vs スイングトレード
- 最適な取引期間の特定

---

## パラメータ設定ガイド

### コマンドラインオプション

| オプション | 説明 | デフォルト | 推奨値 |
|-----------|------|-----------|--------|
| `--models` | 使用モデル | - | `lightgbm` or `xgboost` |
| `--enable-optuna` | Optuna有効化 | False | True |
| `--n-trials` | 試行回数 | 50 | 30-100 |
| `--optuna-timeout` | タイムアウト（秒） | なし | 1800-7200 |
| `--symbol` | シンボル | BTC/USDT:USDT | 対象資産 |
| `--timeframe` | 時間足 | 1h | 1h, 4h, 1d |
| `--limit` | データ件数 | 2000 | 1000-3000 |

### 試行回数の選択基準

| 試行回数 | 実行時間 | 精度 | 推奨シーン |
|---------|---------|------|-----------|
| 10-20 | 5-15分 | ⭐⭐☆☆☆ | デバッグ・動作確認 |
| 30-50 | 15-30分 | ⭐⭐⭐⭐☆ | 通常の評価 |
| 50-100 | 30-60分 | ⭐⭐⭐⭐⭐ | 本番用最適化 |
| 100+ | 1時間以上 | ⭐⭐⭐⭐⭐ | 最高品質の最適化 |

**推奨**: 
- 開発中: 20-30試行
- 検証: 50試行
- 本番: 100試行

### タイムアウトの設定

```bash
# タイムアウトなし（推奨しない）
--enable-optuna --n-trials 100

# 1時間のタイムアウト（推奨）
--enable-optuna --n-trials 100 --optuna-timeout 3600

# 2時間のタイムアウト（詳細評価）
--enable-optuna --n-trials 200 --optuna-timeout 7200
```

**ベストプラクティス**:
- 常にタイムアウトを設定
- 試行回数 × 20秒 を目安に設定
- 例: 100試行 → 2000秒（33分）のタイムアウト

---

## 結果の確認方法

### 1. JSONファイルの確認

```bash
# 結果ファイルの場所
cd backend/scripts/results/feature_analysis/

# 最新の結果を表示
cat lightgbm_feature_performance_evaluation.json | python -m json.tool
```

### 2. 重要な指標の抽出

```python
import json

# JSONファイルを読み込み
with open('lightgbm_feature_performance_evaluation.json', 'r') as f:
    results = json.load(f)

# 重要な指標を表示
print(f"Optuna有効: {results['optuna_enabled']}")
print(f"試行回数: {results['n_trials']}")
print(f"RMSE: {results['cv_results']['rmse_mean']:.6f}")
print(f"MAE: {results['cv_results']['mae_mean']:.6f}")
print(f"R2: {results['cv_results']['r2_mean']:.6f}")

# ベストパラメータ
print("\nベストパラメータ:")
for key, value in results['best_params'].items():
    print(f"  {key}: {value}")
```

### 3. CSVファイルでの比較

```bash
# パフォーマンス比較CSVを確認
cat lightgbm_performance_comparison.csv
```

出力例:
```csv
Metric,Baseline,Optimized,Improvement
RMSE,0.007655,0.006957,-9.12%
MAE,0.005155,0.004228,-17.99%
R2,-0.222883,-0.004640,+97.92%
Training Time,0.13s,0.021s,-83.85%
```

### 4. 最適化履歴の可視化

```python
import json
import matplotlib.pyplot as plt

# 結果の読み込み
with open('lightgbm_feature_performance_evaluation.json', 'r') as f:
    results = json.load(f)

# 最適化履歴を取得
history = results['optimization_history']
trials = [h['trial'] for h in history]
values = [h['value'] for h in history]

# プロット
plt.figure(figsize=(10, 6))
plt.plot(trials, values, 'b-', marker='o')
plt.xlabel('Trial Number')
plt.ylabel('Objective Value (R2 Score)')
plt.title('Optuna Optimization History')
plt.grid(True)
plt.savefig('optuna_history.png')
```

---

## トラブルシューティング

### よくある問題と解決方法

#### 1. メモリ不足エラー

**症状**:
```
MemoryError: Unable to allocate array
```

**解決方法**:
```bash
# データ件数を減らす
python -m scripts.feature_evaluation.evaluate_feature_performance \
    --enable-optuna \
    --limit 1000  # 2000 → 1000に削減
```

#### 2. 実行時間が長すぎる

**症状**: 1時間以上経過しても完了しない

**解決方法**:
```bash
# タイムアウトを設定
python -m scripts.feature_evaluation.evaluate_feature_performance \
    --enable-optuna \
    --n-trials 50 \
    --optuna-timeout 1800  # 30分でタイムアウト
```

#### 3. データが取得できない

**症状**:
```
ValueError: No data available for symbol
```

**確認方法**:
```bash
# データベースの確認
python -c "
from database.connection import SessionLocal
from database.repositories.ohlcv_repository import OHLCVRepository

db = SessionLocal()
repo = OHLCVRepository(db)
df = repo.get_ohlcv_dataframe('BTC/USDT:USDT', '1h', limit=100)
print(f'データ件数: {len(df)}')
db.close()
"
```

**解決方法**:
```bash
# データ収集を実行
python -m scripts.collect_ohlcv_data \
    --symbol BTC/USDT:USDT \
    --timeframe 1h \
    --days 90
```

#### 4. Optunaがクラッシュする

**症状**:
```
optuna.exceptions.StorageInternalError
```

**解決方法**:
```bash
# 試行回数を減らす
python -m scripts.feature_evaluation.evaluate_feature_performance \
    --enable-optuna \
    --n-trials 20  # 50 → 20に削減

# または、タイムアウトを設定
--optuna-timeout 600  # 10分
```

#### 5. 結果が改善しない

**症状**: 最適化後もベースラインと同等かそれ以下

**原因と対策**:
1. **試行回数が不足**
   ```bash
   --n-trials 100  # 50 → 100に増加
   ```

2. **データが不足**
   ```bash
   --limit 3000  # 2000 → 3000に増加
   ```

3. **ローカル最適解に陥っている**
   - 別のシードで再実行
   - パラメータ範囲を調整

---

## パフォーマンスチューニング

### 実行速度の最適化

#### 1. データ件数の調整

```bash
# 高速（開発用）
--limit 1000

# 標準（評価用）
--limit 2000

# 高精度（本番用）
--limit 3000
```

**トレードオフ**:
- データ少 → 高速だが精度低
- データ多 → 遅いが精度高

#### 2. 試行回数の調整

```bash
# 段階的アプローチ（推奨）

# Step 1: クイック評価
python -m scripts.feature_evaluation.evaluate_feature_performance \
    --enable-optuna --n-trials 20

# Step 2: 有望であれば詳細評価
python -m scripts.feature_evaluation.evaluate_feature_performance \
    --enable-optuna --n-trials 100
```

#### 3. 並列実行

複数のシンボルやモデルを同時に評価する場合:

```bash
# 別々のターミナルで実行

# Terminal 1: BTC
python -m scripts.feature_evaluation.evaluate_feature_performance \
    --models lightgbm --enable-optuna --symbol BTC/USDT:USDT

# Terminal 2: ETH
python -m scripts.feature_evaluation.evaluate_feature_performance \
    --models lightgbm --enable-optuna --symbol ETH/USDT:USDT
```

### 精度の最適化

#### 1. TimeSeriesSplit分割数の調整

デフォルトでは5分割ですが、より厳密な評価には:

```python
# evaluate_feature_performance.pyで設定
tscv = TimeSeriesSplit(n_splits=10)  # 5 → 10に増加
```

**注意**: 分割数を増やすと実行時間が増加

#### 2. パラメータ範囲のカスタマイズ

より広範囲を探索する場合:

```python
# backend/app/services/optimization/ensemble_parameter_space.py

# LightGBM用パラメータ範囲を拡張
def suggest_lightgbm_params(self, trial):
    return {
        'num_leaves': trial.suggest_int('num_leaves', 10, 300),  # 20-200 → 10-300
        'learning_rate': trial.suggest_float('learning_rate', 0.0001, 0.5),  # 0.001-0.3 → 0.0001-0.5
        # ...
    }
```

---

## ベストプラクティス

### 1. 段階的アプローチ

```bash
# ステップ1: ベースライン測定（必須）
python -m scripts.feature_evaluation.evaluate_feature_performance \
    --models lightgbm

# ステップ2: クイック最適化（20試行）
python -m scripts.feature_evaluation.evaluate_feature_performance \
    --models lightgbm --enable-optuna --n-trials 20

# ステップ3: 標準最適化（50試行）
python -m scripts.feature_evaluation.evaluate_feature_performance \
    --models lightgbm --enable-optuna --n-trials 50

# ステップ4: 詳細最適化（100試行、必要な場合のみ）
python -m scripts.feature_evaluation.evaluate_feature_performance \
    --models lightgbm --enable-optuna --n-trials 100
```

### 2. 結果の記録

```bash
# ログファイルに出力を保存
python -m scripts.feature_evaluation.evaluate_feature_performance \
    --models lightgbm \
    --enable-optuna \
    --n-trials 50 \
    2>&1 | tee optimization_log_$(date +%Y%m%d_%H%M%S).txt
```

### 3. 定期的な再最適化

市場環境は変化するため、定期的に再最適化することを推奨:

- **頻度**: 月1回または四半期に1回
- **トリガー**: 性能劣化が観測された場合
- **方法**: 最新データで再度最適化を実行

---

## 次のステップ

### 学習リソース

1. **詳細ドキュメント**
   - [`OPTUNA_OPTIMIZATION_REPORT.md`](../../../OPTUNA_OPTIMIZATION_REPORT.md) - 詳細な最適化レポート
   - [`HYPERPARAMETER_OPTIMIZATION_SUMMARY.md`](../../../HYPERPARAMETER_OPTIMIZATION_SUMMARY.md) - プロジェクトサマリー
   - [`README.md`](README.md) - 機能評価スクリプトの全体ガイド

2. **関連実装**
   - [`backend/app/services/optimization/optuna_optimizer.py`](../../app/services/optimization/optuna_optimizer.py) - Optunaコア実装
   - [`backend/app/services/optimization/ensemble_parameter_space.py`](../../app/services/optimization/ensemble_parameter_space.py) - パラメータ空間定義

3. **外部リソース**
   - [Optuna公式ドキュメント](https://optuna.readthedocs.io/)
   - [LightGBMパラメータガイド](https://lightgbm.readthedocs.io/en/latest/Parameters.html)
   - [XGBoostパラメータガイド](https://xgboost.readthedocs.io/en/latest/parameter.html)

### 推奨される次のアクション

1. **XGBoostでの最適化**
   ```bash
   python -m scripts.feature_evaluation.evaluate_feature_performance \
       --models xgboost --enable-optuna --n-trials 50
   ```

2. **アンサンブルモデルの構築**
   - LightGBM + XGBoostの組み合わせ
   - スタッキングやブレンディング

3. **本番環境への適用**
   - 最適化されたパラメータをデフォルト設定に反映
   - 継続的なモニタリング体制の構築

---

## サポート

問題が発生した場合:

1. **ログの確認**: エラーメッセージを詳細に確認
2. **テストの実行**: `pytest tests/scripts/test_evaluate_with_optuna.py -v`
3. **イシュー報告**: GitHubでイシューを作成

---

**最終更新**: 2025年11月13日  
**バージョン**: 1.0.0  
**著者**: Trdinger開発チーム