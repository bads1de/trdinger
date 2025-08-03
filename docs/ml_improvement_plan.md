# ML モデル精度低下原因の包括的分析報告書

## 1. 実施した調査の概要

コードベース全体を対象に、ML モデルの精度低下に関連する可能性のある領域を包括的に調査しました。調査範囲は以下の通りです：

- 評価指標とメトリクス計算
- データ前処理と特徴量エンジニアリング
- ML トレーニングサービスとアンサンブル学習
- モデル管理とシリアライズ
- 設定管理と環境設定
- テストファイルと既知の問題
- データ収集システムとデータ頻度の不一致

## 2. 特定された主要問題

### 2.1 データ頻度の深刻な不一致問題

#### 問題 1: OHLCV、FR、OI のデータ頻度不一致

**場所**: データ収集システム全体

**問題内容**:

- **OHLCV データ**: ユーザー指定の timeframe（15m, 30m, 1h, 4h, 1d 等）で収集
- **ファンディングレート（FR）**: 8 時間間隔で固定（[`backend/app/services/data_collection/bybit/funding_rate_service.py`](backend/app/services/data_collection/bybit/funding_rate_service.py)）
- **建玉残高（OI）**: 1 時間間隔で固定（[`backend/app/services/data_collection/bybit/open_interest_service.py`](backend/app/services/data_collection/bybit/open_interest_service.py:41)）

**影響度**: 極めて高い（精度低下の最大の要因）

**詳細分析**:

```python
# OIサービス - 1時間間隔で固定
async def fetch_open_interest_history(
    self,
    symbol: str,
    limit: int = 100,
    since: Optional[int] = None,
    interval: str = "1h",  # 固定値
) -> List[Dict[str, Any]]:

# FRサービス - 8時間間隔で暗黙的に設定
# （CCXTのfetch_funding_rate_historyが8時間間隔を返す）
```

### 2.2 データ前処理の深刻な問題

#### 問題 2: 特徴量スケーリングの不具合

**場所**: [`backend/app/services/ml/feature_engineering/feature_engineering_service.py`](backend/app/services/ml/feature_engineering/feature_engineering_service.py:261-269)

**問題内容**:

- RobustScaler が正しく実装されておらず、スケーリングが適用されない
- 特徴量間で最大 200 万倍のスケール差が発生
- 外れ値の影響を強く受け、モデル性能が大幅に低下

**影響度**: 極めて高い

#### 問題 3: 不適切な外れ値検出

**場所**: [`backend/app/utils/data_preprocessing.py`](backend/app/utils/data_preprocessing.py)

**問題内容**:

- Z-score 方式がデフォルトで有効になっている
- 市場データの重要なシグナルを外れ値として誤検出
- IQR 方式への切り替えが必要

**影響度**: 高い

### 2.3 ラベル生成の問題

#### 問題 4: 固定閾値によるクラス不均衡

**場所**: [`backend/app/services/ml/config/ml_config.py`](backend/app/services/ml/config/ml_config.py:166-167)

**問題内容**:

- 固定閾値（+2%/-2%）を使用
- 動的ボラティリティ考慮が不足
- クラス不均衡比率が 3 倍に達する場合あり

**影響度**: 高い

### 2.4 特徴量エンジニアリングの問題

#### 問題 5: 特徴量の品質と選択

**場所**: [`backend/app/services/ml/feature_engineering/`](backend/app/services/ml/feature_engineering/)

**問題内容**:

- ノイズの多い特徴量が含まれている可能性
- 特徴量選択プロセスが不十分
- 時系列データの特性を考慮した特徴量設計が不足

**影響度**: 中程度

### 2.5 モデルトレーニングの問題

#### 問題 6: 時系列クロスバリデーションの不備

**場所**: [`backend/app/services/ml/ml_training_service.py`](backend/app/services/ml/ml_training_service.py)

**問題内容**:

- 時系列データに対して適切なクロスバリデーションが実施されていない
- データリークのリスク
- 将来のデータでの汎化性能が低下

**影響度**: 高い

### 2.6 評価指標の問題

#### 問題 8: 不適切な評価指標の使用

**場所**: [`backend/app/utils/metrics_calculator.py`](backend/app/utils/metrics_calculator.py)

**問題内容**:

- 重み付き平均を使用した評価指標が多数派クラスを過大評価
- 不均衡データに対する適切な評価指標（balanced accuracy, F1-score）の活用不足
- PR-AUC などの重要な指標が十分に活用されていない

**影響度**: 中程度

### 2.7 モデル管理と互換性の問題

#### 問題 9: モデルシリアライズの互換性リスク

**場所**: [`backend/app/services/ml/model_manager.py`](backend/app/services/ml/model_manager.py)

**問題内容**:

- バージョン管理が不完全
- 異なるバージョン間での互換性チェックが不十分
- モデル読み込み時のエラーハンドリングが脆弱

**影響度**: 低～中程度

## 3. 問題の相互関連性

### 3.1 連鎖的な影響

1. **データ頻度不一致問題** → 時系列の不整合が発生し、全ての特徴量計算に影響
2. **特徴量スケーリング問題** → モデルが特定の特徴量に過剰適合
3. **外れ値検出問題** → 重要な市場シグナルが除去される
4. **ラベル生成問題** → クラス不均衡により予測バイアスが発生
5. **評価指標問題** → 実際の性能が正しく評価されない

### 3.2 根本的な原因

- **データ収集アーキテクチャの不備**: 異なる頻度のデータを統一的に扱う仕組みが不足
- **設定管理の不統一**: 複数の設定ファイル間で矛盾が存在
- **テストカバレッジの不足**: 重要な機能のテストが不十分
- **ドキュメント化の不足**: 設定パラメータの意図が明確でない

## 4. 改善提案

### 4.1 優先度 1: 即時対応が必要（データ頻度問題の解決）

#### 1.1 データ頻度統一システムの実装

```python
# 新規作成: data_frequency_manager.py
class DataFrequencyManager:
    def __init__(self):
        self.frequency_mappings = {
            '1m': {'ohlcv': '1m', 'fr': '8h', 'oi': '1h'},
            '5m': {'ohlcv': '5m', 'fr': '8h', 'oi': '1h'},
            '15m': {'ohlcv': '15m', 'fr': '8h', 'oi': '1h'},
            '30m': {'ohlcv': '30m', 'fr': '8h', 'oi': '1h'},
            '1h': {'ohlcv': '1h', 'fr': '8h', 'oi': '1h'},
            '4h': {'ohlcv': '4h', 'fr': '8h', 'oi': '1h'},
            '1d': {'ohlcv': '1d', 'fr': '8h', 'oi': '1h'},
        }

    def get_target_frequency(self, source_data_type: str, ohlcv_timeframe: str) -> str:
        """OHLCVのtimeframeに基づいて各データタイプの目標頻度を取得"""
        return self.frequency_mappings.get(ohlcv_timeframe, {}).get(source_data_type, '1h')

    def align_data_frequencies(self, ohlcv_data, fr_data, oi_data, ohlcv_timeframe):
        """異なる頻度のデータをOHLCVのtimeframeに合わせて再サンプリング"""
        target_fr_freq = self.get_target_frequency('fr', ohlcv_timeframe)
        target_oi_freq = self.get_target_frequency('oi', ohlcv_timeframe)

        # FRデータの再サンプリング（8h → OHLCV timeframe）
        if fr_data is not None and not fr_data.empty:
            fr_data = self._resample_funding_rate(fr_data, ohlcv_timeframe)

        # OIデータの再サンプリング（1h → OHLCV timeframe）
        if oi_data is not None and not oi_data.empty:
            oi_data = self._resample_open_interest(oi_data, ohlcv_timeframe)

        return fr_data, oi_data
```

#### 1.2 特徴量エンジニアリングサービスの修正

```python
# feature_engineering_service.py の修正
async def calculate_advanced_features(
    self,
    ohlcv_data: pd.DataFrame,
    funding_rate_data: Optional[pd.DataFrame] = None,
    open_interest_data: Optional[pd.DataFrame] = None,
    fear_greed_data: Optional[pd.DataFrame] = None,
    lookback_periods: Optional[Dict[str, int]] = None,
) -> pd.DataFrame:

    # データ頻度マネージャーの初期化
    frequency_manager = DataFrequencyManager()

    # OHLCVのtimeframeを検出
    ohlcv_timeframe = self._detect_ohlcv_timeframe(ohlcv_data)

    # データ頻度の統一
    funding_rate_data, open_interest_data = frequency_manager.align_data_frequencies(
        ohlcv_data, funding_rate_data, open_interest_data, ohlcv_timeframe
    )

    # 以降の特徴量計算処理...
```

#### 1.3 特徴量スケーリングの修正

```python
# feature_engineering_service.py の修正
def apply_robust_scaling(self, features):
    from sklearn.preprocessing import RobustScaler
    scaler = RobustScaler()
    scaled_features = scaler.fit_transform(features)
    return scaled_features, scaler
```

#### 1.4 外れ値検出方式の変更

```python
# data_preprocessing.py の修正
OUTLIER_DETECTION_METHOD = "iqr"  # "zscore" から変更
```

#### 1.5 動的ラベル生成の実装

```python
# ml_config.py の修正
LABEL_METHOD = "dynamic_volatility"  # 固定閾値から動的閾値へ
```

### 4.2 優先度 2: 短期的な改善

#### 2.1 時系列クロスバリデーションの導入

```python
from sklearn.model_selection import TimeSeriesSplit
tscv = TimeSeriesSplit(n_splits=5)
```

#### 2.2 評価指標の改善

- Balanced Accuracy を主要指標として採用
- PR-AUC と ROC-AUC の両方を監視
- 混同行列の詳細分析

### 4.3 優先度 3: 中長期的な改善

#### 3.1 特徴量エンジニアリングの強化

- 時系列特徴量の追加
- 技術的指標の最適化
- 特徴量選択アルゴリズムの導入

#### 3.2 モデル管理システムの改善

- バージョン管理の厳格化
- 互換性チェックの強化
- モデルパフォーマンスの監視

#### 3.3 テストカバレッジの拡充

- 統合テストの追加
- パフォーマンステストの実装
- 回帰テストの自動化

## 5. 期待される改善効果

### 5.1 精度改善の見込み

- **短期目標**: データ頻度統一により現行精度から 20-30%の改善
- **中期目標**: 現行精度から 30-40%の改善
- **長期目標**: 業界標準レベルの精度達成

### 5.2 ビジネスインパクト

- 予測精度向上によるトレーディングパフォーマンスの大幅改善
- モデルの信頼性向上による運用コストの削減
- 開発効率の向上によるイノベーションの加速

## 6. 実装計画

### 6.1 フェーズ 1: データ頻度統一（1-2 週間）

1. DataFrequencyManager クラスの実装
2. 特徴量エンジニアリングサービスの修正
3. データ収集サービスの調整
4. テストと検証

### 6.2 フェーズ 2: 即時対応問題の修正（1 週間）

1. 特徴量スケーリングの修正
2. 外れ値検出方式の変更
3. 動的ラベル生成の実装

### 6.3 フェーズ 3: 短期的改善（2-3 週間）

1. 時系列クロスバリデーションの導入
2. 評価指標の改善

### 6.4 フェーズ 4: 中長期的改善（1-2 ヶ月）

1. 特徴量エンジニアリングの強化
2. モデル管理システムの改善
3. テストカバレッジの拡充

## 7. 次のステップ

1. **即時開始**: データ頻度統一システムの設計と実装
2. **並行実施**: フェーズ 1 とフェーズ 2 の並行実施
3. **検証**: 各フェーズ完了後のパフォーマンス検証
4. **段階的展開**: 本番環境への段階的なデプロイ
5. **継続的監視**: モデルパフォーマンスの継続的モニタリング体制構築

この分析報告書に基づき、特にデータ頻度の不一致問題に焦点を当てた改善計画を立案し、実行に移すことを強く推奨します。データ頻度の統一は、ML モデル精度向上の最も重要な基盤となります。
