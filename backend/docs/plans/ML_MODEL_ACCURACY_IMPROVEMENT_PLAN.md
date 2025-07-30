# MLモデル精度改善計画

## 概要

本ドキュメントは、トレーニングで生成されたモデルの精度が50%以下という問題を解決するための多角的な分析と改善策をまとめたものです。コードベースやDBスキーマの詳細な分析に基づき、具体的な改善プランを提案します。

## 問題の現状

- **現在のモデル精度**: 50%以下
- **期待される精度**: 70%以上
- **影響範囲**: MLトレーニングシステム全体
- **緊急度**: 高（精度がランダム予測以下）

## 実施した分析

### 1. MLシステムの全体アーキテクチャ分析

#### 分析対象ファイル
- `backend/app/api/ml_training.py`
- `backend/app/services/ml/orchestration/ml_training_orchestration_service.py`
- `backend/app/services/ml/base_ml_trainer.py`
- `backend/app/services/ml/config/ml_config.py`
- `backend/app/services/ml/model_manager.py`

#### 主要な発見
- デフォルトでスタッキングアンサンブルが使用されている
- LightGBM、XGBoost、Gradient Boosting、Random Forestをベースモデルとして使用
- AutoML（TSFresh、AutoFeat）による特徴量エンジニアリングが実装済み
- モデルの永続化とバージョン管理の仕組みが整備されている

### 2. データベーススキーマとデータフローの確認

#### 分析対象ファイル
- `backend/database/models.py`
- `backend/app/services/backtest/data/data_integration_service.py`

#### 主要な発見
- TimescaleDBを使用した効率的な時系列データ保存
- OHLCV、ファンディングレート、建玉残高、Fear & Greed Indexの各データモデル
- 適切なインデックス設定によるクエリパフォーマンスの最適化
- データマージ機能による複数ソースの統合

### 3. 特徴量エンジニアリングの実装調査

#### 分析対象ファイル
- `backend/app/services/ml/feature_engineering/feature_engineering_service.py`
- `backend/app/services/ml/feature_engineering/enhanced_feature_engineering_service.py`
- `backend/app/services/ml/feature_engineering/automl_features/`
- `backend/app/services/ml/feature_engineering/optimized_crypto_features.py`
- `backend/app/services/ml/feature_engineering/enhanced_crypto_features.py`

#### 主要な発見
- 手動特徴量エンジニアリングとAutoMLの両方が実装済み
- **問題点**: 特徴量スケーリングが無効になっている
- 豊富な特徴量グループ（価格、出来高、テクニカル、時間関連など）
- TSFreshとAutoFeatによる自動特徴量生成機能

### 4. モデルトレーニングプロセスの分析

#### 分析対象ファイル
- `backend/app/services/ml/ml_training_service.py`
- `backend/app/services/ml/ensemble/ensemble_trainer.py`
- `backend/app/services/ml/ensemble/bagging.py`
- `backend/app/services/ml/ensemble/stacking.py`

#### 主要な発見
- アンサンブル学習（バギングとスタッキング）の完全な実装
- 時系列クロスバリデーションの試み
- モデルの最適化とハイパーパラメータチューニング機能
- **問題点**: 時系列データに適した検証が不十分

### 5. データ品質と前処理の詳細調査

#### 分析対象ファイル
- `backend/app/utils/data_preprocessing.py`
- `backend/app/utils/data_cleaning_utils.py`
- `backend/app/utils/data_validation.py`
- `backend/app/services/auto_strategy/utils/data_coverage_analyzer.py`

#### 主要な発見
- SimpleImputerを使用した高品質な欠損値補完
- **問題点**: Z-scoreベースの外れ値検出が金融データに不適切
- データクリーニングと検証の充実した機能
- 異常値検出と処理のメカニズム

### 6. ラベル生成メカニズムの分析

#### 分析対象ファイル
- `backend/app/utils/label_generation.py`

#### 主要な発見
- 複数の閾値計算方法（固定、分位数、標準偏差、適応的）
- **問題点**: デフォルト設定で固定閾値（±2%）が使用されている
- 動的閾値計算機能は実装済みだが使用されていない
- ラベル分布の検証機能

## 特定した問題の根本原因

### 1. ラベル生成の問題

#### 問題点
- 固定閾値（±2%）が市場のボラティリティ変化に対応できない
- 低ボラティリティ時：ほとんどのラベルが「レンジ」になり不均衡
- 高ボラティリティ時：「上昇」「下落」ラベルが多くなりすぎる

#### 影響
- 予測が困難なレンジ相場が増加
- クラス不均衡によるモデル性能の低下
- 市場状態を適切に反映できない

### 2. データ前処理の問題

#### 問題点
- Z-scoreベースの外れ値検出が金融データの急な変動を誤検出
- 重要な市場シグナルが外れ値として除去される
- マルチソースデータの時間軸不整合

#### 影響
- 予測に重要な情報の損失
- ノイズの増加
- データ品質の低下

### 3. 特徴量エンジニアリングの問題

#### 問題点
- 特徴量スケーリングが無効になっている
- 特徴量選択が不十分で次元の呪いが発生
- 時系列データの特性を考慮した特徴量が不足

#### 影響
- 特徴量のスケール不整合
- 重要な特徴量が埋もれる
- 時系列パターンの捉え方が不十分

### 4. モデル設定と評価の問題

#### 問題点
- 時系列クロスバリデーションが不適切
- 評価指標が精度（accuracy）のみに焦点
- データ量や特性に合わない複雑なモデル

#### 影響
- 過剰適合（オーバーフィッティング）のリスク
- 時系列データの未来情報leakage
- 実際の取引パフォーマンスとの乖離

## 改善策の策定

### 1. ラベル生成システムの再設計

#### 改善内容
1. **動的閾値の導入**
   - 固定閾値から市場ボラティリティに応じた動的閾値へ変更
   - ボラティリティレジームを考慮した閾値調整

2. **ラベル分布の最適化**
   - クラス不均衡を解消するための目標分布設定
   - 時系列特性を考慮したラベル生成

#### 実装コード例
```python
# backend/app/utils/label_generation.py に追加
def generate_labels_with_dynamic_threshold(
    self,
    price_data: pd.Series,
    volatility_window: int = 24,
    threshold_multiplier: float = 0.5,
    min_threshold: float = 0.005,
    max_threshold: float = 0.05,
) -> Tuple[pd.Series, Dict[str, Any]]:
    """
    ボラティリティに基づく動的閾値でラベルを生成
    """
    # ボラティリティの計算
    volatility = price_data.pct_change().rolling(volatility_window).std()
    
    # 動的閾値の計算
    dynamic_threshold = volatility * threshold_multiplier
    dynamic_threshold = dynamic_threshold.clip(min_threshold, max_threshold)
    
    # ラベル生成
    price_change = price_data.pct_change().shift(-1)
    labels = pd.Series(1, index=price_change.index, dtype=int)  # レンジ
    labels[price_change > dynamic_threshold] = 2  # 上昇
    labels[price_change < -dynamic_threshold] = 0  # 下落
    
    return labels.iloc[:-1], {
        "method": "dynamic_volatility",
        "volatility_window": volatility_window,
        "threshold_multiplier": threshold_multiplier,
        "avg_threshold": dynamic_threshold.mean()
    }
```

#### 設定ファイルの更新
```python
# backend/app/services/ml/config/ml_config.py の修正
DEFAULT_LABEL_CONFIG = {
    "method": "dynamic_volatility",  # 固定閾値から動的閾値へ変更
    "volatility_window": 24,
    "threshold_multiplier": 0.5,
    "min_threshold": 0.005,
    "max_threshold": 0.05,
    "target_distribution": {"up": 0.35, "down": 0.35, "range": 0.30}
}
```

#### 期待される効果
- 市場状態に応じた適切なラベル生成
- クラス分布の均衡化（目標: 上昇35%、下落35%、レンジ30%）
- 予測困難なレンジ相場の削減

### 2. データ前処理パイプラインの強化

#### 改善内容
1. **金融時系列データに適した前処理**
   - Z-scoreベースからIQR（四分位範囲）ベースの外れ値検出へ変更
   - 時系列データの特性を考慮した欠損値補完

2. **マルチソースデータの整合性確保**
   - 異なるデータソースの時間軸を整合させる
   - データ品質モニタリングの実装

#### 実装コード例
```python
# backend/app/utils/data_preprocessing.py に追加
class FinancialDataPreprocessor(DataPreprocessor):
    """
    金融時系列データに特化した前処理クラス
    """
    
    def remove_financial_outliers(
        self,
        df: pd.DataFrame,
        columns: List[str],
        method: str = "iqr",
        threshold: float = 3.0
    ) -> pd.DataFrame:
        """
        金融データに適した外れ値除去
        """
        result_df = df.copy()
        
        for col in columns:
            if method == "iqr":
                # IQR（四分位範囲）ベースの外れ値検出
                Q1 = result_df[col].quantile(0.25)
                Q3 = result_df[col].quantile(0.75)
                IQR = Q3 - Q1
                lower_bound = Q1 - threshold * IQR
                upper_bound = Q3 + threshold * IQR
                
                outliers = (result_df[col] < lower_bound) | (result_df[col] > upper_bound)
                result_df.loc[outliers, col] = np.nan
        
        return result_df
```

#### 期待される効果
- 重要な市場シグナルの保持
- データ品質の向上
- ノイズの削減

### 3. 特徴量エンジニアリングの最適化

#### 改善内容
1. **特徴量スケーリングの有効化**
   - `feature_engineering_service.py`でスケーリングを有効化
   - ロバストスケーリング（外れ値に強い）の導入

2. **特徴量選択の最適化**
   - 時系列データに適した特徴量選択手法の導入
   - 予測力の低い特徴量の自動削除

#### 実装コード例
```python
# backend/app/services/ml/feature_engineering/feature_engineering_service.py の修正
def preprocess_features(
    self,
    df: pd.DataFrame,
    scale_features: bool = True,  # Trueに変更
    remove_outliers: bool = True,
    outlier_threshold: float = 3.0,
    scaling_method: str = "robust"  # スケーリング方法を追加
) -> pd.DataFrame:
    """
    特徴量の前処理（改良版）
    """
    # ... 既存のコード ...
    
    # 特徴量スケーリング
    if scale_features:
        result_df = self._scale_features(
            result_df, 
            numeric_columns, 
            method=scaling_method
        )
    
    return result_df
```

#### 期待される効果
- 特徴量のスケール不整合解消
- 次元の呪い回避
- 重要な特徴量の抽出

### 4. モデルトレーニングと評価の改善

#### 改善内容
1. **モデルアーキテクチャの最適化**
   - データ量と特性に応じたモデル選択
   - 時系列クロスバリデーションの適切な実装

2. **評価指標の多様化**
   - 金融取引に適した評価指標の導入
   - 時系列特性を考慮したバックテスト

#### 実装コード例
```python
# backend/app/services/ml/time_series_validation.py の新規作成
class TimeSeriesValidator:
    """
    時系列データに適した検証を行うクラス
    """
    
    def time_series_split(
        self,
        data: pd.DataFrame,
        n_splits: int = 5,
        test_size: float = 0.2
    ) -> List[Tuple[pd.DataFrame, pd.DataFrame]]:
        """
        時系列データの分割
        """
        splits = []
        data_length = len(data)
        test_length = int(data_length * test_size)
        
        for i in range(n_splits):
            # 時系列順に分割
            train_end = int(data_length * (1 - test_size) * (i + 1) / n_splits)
            test_start = train_end
            test_end = min(train_end + test_length, data_length)
            
            train_data = data.iloc[:train_end]
            test_data = data.iloc[test_start:test_end]
            
            splits.append((train_data, test_data))
        
        return splits
```

#### 期待される効果
- 過剰適合の防止
- 時系列データの未来情報leakage防止
- 実際の取引パフォーマンスとの整合性向上

## 実装スケジュール

| フェーズ | タスク | 期間 | 優先度 |
|---------|------|------|-------|
| 1 | ラベル生成システムの再設計 | 1週間 | 高 |
| 2 | データ前処理パイプラインの強化 | 1週間 | 高 |
| 3 | 特徴量エンジニアリングの最適化 | 1週間 | 中 |
| 4 | モデルトレーニングと評価の改善 | 1週間 | 中 |
| 5 | 統合テストとパフォーマンス検証 | 1週間 | 高 |

## 期待される成果

- **精度向上**: 50%以下 → 70%以上
- **F1スコア向上**: 50%以下 → 70%以上
- **トレーニング時間**: 20%削減
- **モデルの安定性**: 標準偏差の減少
- **解釈性**: 特徴量重要度の明確化

## リスクと対策

| リスク | 対策 |
|-------|------|
| データ品質の問題 | データ品質モニタリングの実装 |
| 過剰適合 | 時系列クロスバリデーションの厳格な実施 |
| 計算リソース不足 | 特徴量選択による次元削減 |
| モデル性能の不安定性 | アンサンブル学習の適切な適用 |

## まとめ

本改善計画は、MLモデルの精度が50%以下という問題を解決するための包括的なアプローチを提供します。ラベル生成、データ前処理、特徴量エンジニアリング、モデル設定、評価方法の5つの主要領域に焦点を当て、具体的な実装コードと共に改善策を提案しました。

これらの改善策を段階的に実装することで、モデルの精度を70%以上に向上させることが期待できます。各改善策は独立しているため、優先度の高いものから順に実装可能です。

まずは「ラベル生成システムの再設計」から着手することをお勧めします。これは最も直接的な影響を与える改善策であり、比較的実装が容易です。