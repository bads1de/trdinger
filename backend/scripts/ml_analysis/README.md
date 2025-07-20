# 特徴量ノイズ分析スクリプト

このスクリプトは機械学習モデルの特徴量を分析し、精度を落としているノイズ特徴量を特定・可視化します。

## 機能

### 1. 特徴量重要度分析
- LightGBMの特徴量重要度計算
- 重要度の可視化（バープロット）
- 低重要度特徴量の特定

### 2. Permutation Importance分析
- 特徴量をシャッフルした時の性能変化を測定
- 負の影響を与える特徴量の特定
- 標準偏差付きの可視化

### 3. 特徴量相関分析
- 特徴量間の相関係数計算
- 高相関特徴量ペアの特定
- 相関ヒートマップの生成

### 4. 低分散特徴量検出
- 各特徴量の分散計算
- 低分散（情報量の少ない）特徴量の特定

### 5. ノイズ特徴量の総合判定
以下の基準でノイズ特徴量を特定：
- 特徴量重要度下位20%
- 負のPermutation Importance
- 高相関特徴量（相関係数 > 0.95）
- 低分散特徴量（分散 < 0.01）

### 6. モデル性能比較
- ベースラインモデル（全特徴量使用）
- フィルタリングモデル（ノイズ特徴量除去後）
- 性能改善の定量評価

### 7. 結果の可視化とレポート
- 特徴量重要度プロット
- Permutation Importanceプロット
- 相関ヒートマップ
- モデル性能比較グラフ
- Markdownレポート自動生成

## 使用方法

### 1. バッチファイルで実行（推奨）
```bash
# Windowsの場合
backend\scripts\ml_analysis\run_feature_analysis.bat
```

### 2. Pythonスクリプト直接実行
```bash
# プロジェクトルートから実行
python backend\scripts\ml_analysis\feature_noise_analysis.py
```

### 3. カスタム設定で実行
```python
from backend.scripts.ml_analysis.feature_noise_analysis import FeatureNoiseAnalyzer

# 分析器を初期化（カスタム設定）
analyzer = FeatureNoiseAnalyzer(symbol="ETH/USDT", timeframe="4h")

# データ読み込み
df = analyzer.load_data(limit=10000)

# 以下、分析実行...
```

## 必要な依存関係

```bash
pip install pandas numpy matplotlib seaborn scikit-learn lightgbm
```

## 出力ファイル

分析結果は `backend/scripts/ml_analysis/results/` に保存されます：

### 可視化ファイル
- `lightgbm_feature_importance.png` - 特徴量重要度
- `permutation_importance_analysis.png` - Permutation Importance
- `feature_correlation_heatmap.png` - 相関ヒートマップ
- `model_performance_comparison.png` - モデル性能比較

### レポートファイル
- `feature_noise_analysis_report.md` - 分析結果レポート

## 設定可能なパラメータ

### FeatureNoiseAnalyzer
- `symbol`: 取引ペア（デフォルト: "BTC/USDT"）
- `timeframe`: 時間軸（デフォルト: "1h"）

### 分析パラメータ
- `limit`: 取得データ数（デフォルト: 5000）
- `correlation_threshold`: 高相関閾値（デフォルト: 0.95）
- `variance_threshold`: 低分散閾値（デフォルト: 0.01）
- `n_repeats`: Permutation Importance繰り返し回数（デフォルト: 5）

## 分析結果の解釈

### 特徴量重要度
- 高い値：モデルの予測に重要な特徴量
- 低い値：予測への寄与が小さい特徴量

### Permutation Importance
- 正の値：特徴量が予測性能に貢献
- 負の値：特徴量が予測性能を悪化させる（ノイズ）
- ゼロ付近：特徴量の影響が小さい

### 相関分析
- 高相関（>0.95）：冗長な特徴量ペア
- 一方を除去することで次元削減可能

### 性能改善
- 正の改善：ノイズ除去が効果的
- 負の改善：重要な特徴量も除去された可能性

## トラブルシューティング

### よくあるエラー

1. **データが見つからない**
   - データベースにOHLCVデータが存在するか確認
   - シンボルと時間軸の設定を確認

2. **メモリ不足**
   - `limit`パラメータを小さくする
   - 相関ヒートマップの特徴量数を制限

3. **ライブラリエラー**
   - 必要な依存関係をインストール
   - Pythonバージョンの確認（3.8以上推奨）

### パフォーマンス最適化

- データ量を調整（`limit`パラメータ）
- 特徴量数を事前に制限
- 並列処理の活用（将来の拡張）

## 注意事項

- 分析には時間がかかる場合があります（データ量に依存）
- 結果は使用するデータとモデルに依存します
- 特徴量除去は慎重に行ってください
- 定期的な再分析を推奨します

## 今後の拡張予定

- SHAP値による詳細な特徴量分析
- 複数モデルでの比較分析
- 時系列特有の特徴量選択手法
- 自動特徴量生成機能
- インタラクティブな可視化
