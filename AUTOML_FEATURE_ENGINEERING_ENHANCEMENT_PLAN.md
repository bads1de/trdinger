# 🤖 AutoML 特徴量エンジニアリング強化計画

## 📋 プロジェクト概要

**プロジェクト名**: AutoML 特徴量エンジニアリング強化  
**目的**: 現在の手動特徴量生成システムを AutoML ライブラリで補完し、より高度な特徴量の自動発見を実現  
**対象**: Trdinger トレーディングプラットフォーム  
**期間**: 4 週間（段階的実装）

---

## 🔍 現状分析

### ✅ **現在の特徴量エンジニアリングシステム**

既に非常に包括的な手動特徴量生成システムが実装済み：

#### **実装済み特徴量カテゴリ**

1. **価格特徴量** (`PriceFeatureCalculator`)

   - 移動平均比率、モメンタム、価格レンジ
   - ローソク足パターン（実体、ヒゲ）
   - 価格位置、ギャップ分析

2. **ボラティリティ特徴量**

   - 実現ボラティリティ、ATR、ボラティリティスパイク
   - ボラティリティレジーム、変化率

3. **出来高特徴量**

   - VWAP、出来高比率、価格・出来高トレンド
   - 出来高スパイク、出来高加重指標

4. **テクニカル特徴量** (`TechnicalFeatureCalculator`)

   - RSI、MACD、ストキャスティクス、CCI、ROC
   - 市場レジーム、パターン認識
   - ダイバージェンス、サポート・レジスタンス

5. **市場データ特徴量** (`MarketDataFeatureCalculator`)

   - ファンディングレート特徴量（24 種類）
   - 建玉残高特徴量（15 種類）
   - 複合特徴量（FR × OI）

6. **時間的特徴量** (`TemporalFeatureCalculator`)

   - 取引セッション、曜日効果
   - 周期的エンコーディング（sin/cos）
   - セッション重複時間

7. **相互作用特徴量** (`InteractionFeatureCalculator`)
   - ボラティリティ × モメンタム
   - 出来高 × トレンド
   - FR × RSI、OI × 価格変動

#### **現在の特徴量統計**

- **総特徴量数**: 約 100 個
- **カテゴリ数**: 7 カテゴリ
- **実装ファイル数**: 7 ファイル
- **コード行数**: 約 2,000 行

### 🎯 **強化の必要性**

現在のシステムは優秀だが、以下の限界がある：

1. **手動設計の限界**: 人間が考えつかない複雑な特徴量の組み合わせ
2. **統計的特徴量の不足**: 高次統計量、フラクタル次元、エントロピー系
3. **時系列専用特徴量の不足**: 周波数領域、スペクトル解析、自己相関
4. **特徴量選択の自動化**: 重要な特徴量の自動識別
5. **計算効率の最適化**: 冗長な特徴量の自動除去

---

## 🚀 AutoML ライブラリ選定

### **主要候補ライブラリ比較**

| ライブラリ       | 特徴                   | 時系列対応 | 金融データ適性 | 学習コスト | 性能     |
| ---------------- | ---------------------- | ---------- | -------------- | ---------- | -------- |
| **TSFresh**      | 時系列専用、100+特徴量 | ⭐⭐⭐⭐⭐ | ⭐⭐⭐⭐⭐     | ⭐⭐⭐     | ⭐⭐⭐⭐ |
| **Featuretools** | DFS、汎用性高          | ⭐⭐⭐⭐   | ⭐⭐⭐         | ⭐⭐       | ⭐⭐⭐⭐ |
| **AutoFeat**     | 遺伝的プログラミング   | ⭐⭐       | ⭐⭐⭐         | ⭐⭐⭐⭐   | ⭐⭐⭐   |

### **選定結果**

#### **第 1 優先: TSFresh**

- **理由**: 時系列・金融データに最適化
- **特徴量数**: 100 以上の統計的特徴量
- **自動選択**: 仮説検定による特徴量選択
- **実績**: 金融時系列での豊富な実績

#### **第 2 優先: Featuretools**

- **理由**: 既存特徴量との相互作用発見
- **DFS**: Deep Feature Synthesis
- **柔軟性**: カスタム変換関数対応

#### **第 3 優先: AutoFeat**

- **理由**: 軽量で高速な特徴量選択
- **最適化**: 遺伝的アルゴリズム
- **補完**: 最終的な特徴量選択に活用

---

## 📅 段階的実装計画

### **Phase 1: TSFresh 統合（2 週間）**

#### **Week 1: 基盤実装**

- [ ] TSFresh 依存関係追加
- [ ] TSFreshFeatureCalculator クラス実装
- [ ] 既存システムとの統合テスト
- [ ] 基本的な時系列特徴量生成

#### **Week 2: 高度な機能**

- [ ] カスタム特徴量設定
- [ ] 特徴量選択の自動化
- [ ] 性能最適化
- [ ] 金融データ専用設定

### **Phase 2: Featuretools 統合（1 週間）**

#### **Week 3: DFS 実装**

- [ ] Featuretools 依存関係追加
- [ ] Deep Feature Synthesis 設定
- [ ] 既存特徴量との相互作用発見
- [ ] カスタム変換関数実装

### **Phase 3: 最適化・統合（1 週間）**

#### **Week 4: 最終統合**

- [ ] AutoFeat 特徴量選択統合
- [ ] 性能ベンチマーク
- [ ] UI 更新（特徴量選択設定）
- [ ] ドキュメント作成

---

## 🔧 技術実装詳細

### **1. TSFresh 統合実装**

#### **新規ファイル構成**

```
backend/app/core/services/ml/feature_engineering/
├── automl_features/
│   ├── __init__.py
│   ├── tsfresh_calculator.py      # TSFresh特徴量計算
│   ├── featuretools_calculator.py # Featuretools DFS
│   ├── autofeat_selector.py       # AutoFeat選択
│   └── automl_config.py           # AutoML設定
└── enhanced_feature_service.py    # 統合サービス
```

#### **TSFreshFeatureCalculator 実装**

```python
"""
TSFresh特徴量計算クラス

時系列データから100以上の統計的特徴量を自動生成し、
仮説検定による特徴量選択を実行します。
"""

import logging
import pandas as pd
import numpy as np
from typing import Dict, List, Optional
from tsfresh import extract_features, select_features
from tsfresh.utilities.dataframe_functions import impute
from tsfresh.feature_extraction import ComprehensiveFCParameters

logger = logging.getLogger(__name__)


class TSFreshFeatureCalculator:
    """
    TSFresh特徴量計算クラス

    時系列データから統計的特徴量を自動生成します。
    """

    def __init__(self):
        """初期化"""
        self.feature_cache = {}
        self.selected_features = None

    def calculate_tsfresh_features(
        self,
        df: pd.DataFrame,
        target: Optional[pd.Series] = None,
        feature_selection: bool = True,
        custom_settings: Optional[Dict] = None
    ) -> pd.DataFrame:
        """
        TSFresh特徴量を計算

        Args:
            df: OHLCV価格データ
            target: ターゲット変数（特徴量選択用）
            feature_selection: 特徴量選択を実行するか
            custom_settings: カスタム特徴量設定

        Returns:
            TSFresh特徴量が追加されたDataFrame
        """
        try:
            # データを時系列形式に変換
            ts_data = self._prepare_timeseries_data(df)

            # 特徴量設定
            if custom_settings is None:
                # 金融データ用カスタム設定
                settings = self._get_financial_feature_settings()
            else:
                settings = custom_settings

            # 特徴量抽出
            logger.info("TSFresh特徴量抽出を開始...")
            extracted_features = extract_features(
                ts_data,
                column_id="id",
                column_sort="time",
                default_fc_parameters=settings,
                impute_function=impute,
                n_jobs=4  # 並列処理
            )

            # 特徴量選択（ターゲットがある場合）
            if feature_selection and target is not None:
                logger.info("TSFresh特徴量選択を開始...")
                selected_features = select_features(
                    extracted_features,
                    target,
                    fdr_level=0.05  # False Discovery Rate
                )
                self.selected_features = selected_features.columns.tolist()
                result_features = selected_features
            else:
                result_features = extracted_features

            # 元のDataFrameに結合
            result_df = df.copy()

            # インデックスを合わせて結合
            if len(result_features) == len(df):
                for col in result_features.columns:
                    result_df[f"TSF_{col}"] = result_features[col].values

            logger.info(f"TSFresh特徴量生成完了: {len(result_features.columns)}個")
            return result_df

        except Exception as e:
            logger.error(f"TSFresh特徴量計算エラー: {e}")
            return df

    def _prepare_timeseries_data(self, df: pd.DataFrame) -> pd.DataFrame:
        """時系列データをTSFresh形式に変換"""
        ts_data = []

        # 各価格系列を個別の時系列として扱う
        for col in ['Open', 'High', 'Low', 'Close', 'Volume']:
            if col in df.columns:
                for i, (timestamp, value) in enumerate(df[col].items()):
                    ts_data.append({
                        'id': col,
                        'time': i,
                        'value': value
                    })

        return pd.DataFrame(ts_data)

    def _get_financial_feature_settings(self) -> Dict:
        """金融データ用の特徴量設定"""
        # 金融時系列に適した特徴量のみを選択
        financial_settings = {
            # 統計的特徴量
            'mean': None,
            'median': None,
            'std': None,
            'var': None,
            'skewness': None,
            'kurtosis': None,

            # 分位点
            'quantile': [{'q': 0.1}, {'q': 0.25}, {'q': 0.75}, {'q': 0.9}],

            # 自己相関
            'autocorrelation': [{'lag': i} for i in [1, 2, 3, 5, 10, 20]],

            # 周波数領域
            'fft_coefficient': [{'coeff': i, 'attr': 'real'} for i in range(10)],
            'fft_coefficient': [{'coeff': i, 'attr': 'imag'} for i in range(10)],

            # エントロピー
            'sample_entropy': None,
            'approximate_entropy': [{'m': 2, 'r': 0.1}],

            # トレンド
            'linear_trend': [{'attr': 'slope'}, {'attr': 'intercept'}],
            'agg_linear_trend': [
                {'attr': 'slope', 'chunk_len': 10, 'f_agg': 'mean'},
                {'attr': 'slope', 'chunk_len': 10, 'f_agg': 'std'}
            ],

            # 極値
            'number_peaks': [{'n': 3}, {'n': 5}, {'n': 10}],
            'number_cwt_peaks': [{'n': 3}, {'n': 5}],

            # 変化点
            'change_quantiles': [
                {'ql': 0.0, 'qh': 0.2, 'isabs': False},
                {'ql': 0.8, 'qh': 1.0, 'isabs': False}
            ],

            # 複雑性
            'lempel_ziv_complexity': [{'bins': 10}],
            'fourier_entropy': [{'bins': 10}],
        }

        return financial_settings

    def get_feature_names(self) -> List[str]:
        """生成される特徴量名のリストを取得"""
        if self.selected_features:
            return [f"TSF_{name}" for name in self.selected_features]
        else:
            # デフォルトの特徴量名（推定）
            return [
                "TSF_mean", "TSF_std", "TSF_skewness", "TSF_kurtosis",
                "TSF_autocorr_1", "TSF_autocorr_5", "TSF_autocorr_10",
                "TSF_fft_coeff_0_real", "TSF_fft_coeff_1_real",
                "TSF_sample_entropy", "TSF_linear_trend_slope",
                "TSF_number_peaks_3", "TSF_change_quantiles_low",
                "TSF_lempel_ziv_complexity"
            ]
```

### **2. 統合サービス実装**

```python
"""
拡張特徴量エンジニアリングサービス

既存の手動特徴量生成システムにAutoML特徴量を統合します。
"""

import logging
from typing import Dict, Optional
import pandas as pd

from .feature_engineering_service import FeatureEngineeringService
from .automl_features.tsfresh_calculator import TSFreshFeatureCalculator
from .automl_features.featuretools_calculator import FeaturetoolsCalculator
from .automl_features.autofeat_selector import AutoFeatSelector

logger = logging.getLogger(__name__)


class EnhancedFeatureEngineeringService(FeatureEngineeringService):
    """
    拡張特徴量エンジニアリングサービス

    既存の手動特徴量にAutoML特徴量を追加します。
    """

    def __init__(self):
        """初期化"""
        super().__init__()

        # AutoML特徴量計算クラス
        self.tsfresh_calculator = TSFreshFeatureCalculator()
        self.featuretools_calculator = FeaturetoolsCalculator()
        self.autofeat_selector = AutoFeatSelector()

    def calculate_enhanced_features(
        self,
        ohlcv_data: pd.DataFrame,
        funding_rate_data: Optional[pd.DataFrame] = None,
        open_interest_data: Optional[pd.DataFrame] = None,
        fear_greed_data: Optional[pd.DataFrame] = None,
        lookback_periods: Optional[Dict[str, int]] = None,
        automl_config: Optional[Dict] = None,
        target: Optional[pd.Series] = None
    ) -> pd.DataFrame:
        """
        拡張特徴量を計算（手動 + AutoML）

        Args:
            ohlcv_data: OHLCV価格データ
            funding_rate_data: ファンディングレートデータ
            open_interest_data: 建玉残高データ
            fear_greed_data: Fear & Greed Index データ
            lookback_periods: 計算期間設定
            automl_config: AutoML設定
            target: ターゲット変数（特徴量選択用）

        Returns:
            拡張特徴量が追加されたDataFrame
        """
        try:
            # デフォルト設定
            if automl_config is None:
                automl_config = {
                    'tsfresh_enabled': True,
                    'featuretools_enabled': True,
                    'autofeat_enabled': True,
                    'feature_selection': True
                }

            # 1. 既存の手動特徴量を計算
            logger.info("手動特徴量を計算中...")
            result_df = self.calculate_advanced_features(
                ohlcv_data=ohlcv_data,
                funding_rate_data=funding_rate_data,
                open_interest_data=open_interest_data,
                fear_greed_data=fear_greed_data,
                lookback_periods=lookback_periods
            )

            manual_feature_count = len(result_df.columns)
            logger.info(f"手動特徴量生成完了: {manual_feature_count}個")

            # 2. TSFresh特徴量を追加
            if automl_config.get('tsfresh_enabled', True):
                logger.info("TSFresh特徴量を計算中...")
                result_df = self.tsfresh_calculator.calculate_tsfresh_features(
                    df=result_df,
                    target=target,
                    feature_selection=automl_config.get('feature_selection', True)
                )
                tsfresh_count = len(result_df.columns) - manual_feature_count
                logger.info(f"TSFresh特徴量追加完了: {tsfresh_count}個")

            # 3. Featuretools特徴量を追加
            if automl_config.get('featuretools_enabled', True):
                logger.info("Featuretools特徴量を計算中...")
                result_df = self.featuretools_calculator.calculate_dfs_features(
                    df=result_df,
                    target=target
                )
                featuretools_count = len(result_df.columns) - manual_feature_count - tsfresh_count
                logger.info(f"Featuretools特徴量追加完了: {featuretools_count}個")

            # 4. AutoFeat特徴量選択
            if automl_config.get('autofeat_enabled', True) and target is not None:
                logger.info("AutoFeat特徴量選択を実行中...")
                result_df = self.autofeat_selector.select_features(
                    df=result_df,
                    target=target
                )
                final_count = len(result_df.columns)
                logger.info(f"AutoFeat選択完了: {final_count}個の特徴量を選択")

            total_features = len(result_df.columns)
            logger.info(f"拡張特徴量生成完了: 総計{total_features}個の特徴量")

            return result_df

        except Exception as e:
            logger.error(f"拡張特徴量計算エラー: {e}")
            # エラー時は手動特徴量のみ返す
            return self.calculate_advanced_features(
                ohlcv_data=ohlcv_data,
                funding_rate_data=funding_rate_data,
                open_interest_data=open_interest_data,
                fear_greed_data=fear_greed_data,
                lookback_periods=lookback_periods
            )
```

### **3. 依存関係追加**

```python
# requirements.txt に追加
tsfresh>=0.21.0
featuretools>=1.31.0
autofeat>=2.1.0

# オプション（大規模データ用）
dask>=2023.1.0  # TSFresh並列処理用
```

---

## 🎨 フロントエンド UI 拡張

### **AutoML 特徴量設定コンポーネント**

```typescript
// frontend/components/ml/AutoMLFeatureSettings.tsx

interface AutoMLFeatureConfig {
  tsfresh_enabled: boolean;
  featuretools_enabled: boolean;
  autofeat_enabled: boolean;
  feature_selection: boolean;
  tsfresh_settings: {
    feature_count_limit: number;
    fdr_level: number;
    parallel_jobs: number;
  };
  featuretools_settings: {
    max_depth: number;
    max_features: number;
  };
}

export default function AutoMLFeatureSettings({
  settings,
  onChange,
}: {
  settings: AutoMLFeatureConfig;
  onChange: (settings: AutoMLFeatureConfig) => void;
}) {
  return (
    <Card>
      <CardHeader>
        <CardTitle className="flex items-center gap-2">
          <Bot className="h-5 w-5" />
          AutoML特徴量エンジニアリング設定
        </CardTitle>
        <p className="text-sm text-muted-foreground">
          自動特徴量生成・選択の設定
        </p>
      </CardHeader>
      <CardContent className="space-y-6">
        {/* TSFresh設定 */}
        <div className="space-y-4">
          <div className="flex items-center space-x-2">
            <Switch
              id="tsfresh-enabled"
              checked={settings.tsfresh_enabled}
              onCheckedChange={(enabled) =>
                onChange({ ...settings, tsfresh_enabled: enabled })
              }
            />
            <Label htmlFor="tsfresh-enabled">TSFresh時系列特徴量生成</Label>
          </div>

          {settings.tsfresh_enabled && (
            <div className="ml-6 space-y-3">
              <div>
                <Label>特徴量数制限</Label>
                <Slider
                  value={[settings.tsfresh_settings.feature_count_limit]}
                  onValueChange={([value]) =>
                    onChange({
                      ...settings,
                      tsfresh_settings: {
                        ...settings.tsfresh_settings,
                        feature_count_limit: value,
                      },
                    })
                  }
                  max={200}
                  min={20}
                  step={10}
                />
                <span className="text-sm text-muted-foreground">
                  {settings.tsfresh_settings.feature_count_limit}個
                </span>
              </div>
            </div>
          )}
        </div>

        {/* Featuretools設定 */}
        <div className="space-y-4">
          <div className="flex items-center space-x-2">
            <Switch
              id="featuretools-enabled"
              checked={settings.featuretools_enabled}
              onCheckedChange={(enabled) =>
                onChange({ ...settings, featuretools_enabled: enabled })
              }
            />
            <Label htmlFor="featuretools-enabled">
              Featuretools Deep Feature Synthesis
            </Label>
          </div>
        </div>

        {/* AutoFeat設定 */}
        <div className="space-y-4">
          <div className="flex items-center space-x-2">
            <Switch
              id="autofeat-enabled"
              checked={settings.autofeat_enabled}
              onCheckedChange={(enabled) =>
                onChange({ ...settings, autofeat_enabled: enabled })
              }
            />
            <Label htmlFor="autofeat-enabled">AutoFeat特徴量選択</Label>
          </div>
        </div>

        {/* 特徴量選択設定 */}
        <div className="space-y-4">
          <div className="flex items-center space-x-2">
            <Switch
              id="feature-selection"
              checked={settings.feature_selection}
              onCheckedChange={(enabled) =>
                onChange({ ...settings, feature_selection: enabled })
              }
            />
            <Label htmlFor="feature-selection">
              自動特徴量選択を有効にする
            </Label>
          </div>
        </div>

        {/* 情報表示 */}
        <div className="bg-blue-50 dark:bg-blue-900/20 p-4 rounded-lg">
          <div className="flex items-center gap-2 mb-2">
            <Info className="h-4 w-4 text-blue-500" />
            <span className="text-sm font-medium text-blue-700 dark:text-blue-300">
              AutoML特徴量について
            </span>
          </div>
          <ul className="text-xs text-blue-600 dark:text-blue-400 space-y-1">
            <li>• TSFresh: 100以上の統計的時系列特徴量</li>
            <li>• Featuretools: 既存特徴量の高次相互作用</li>
            <li>• AutoFeat: 遺伝的アルゴリズムによる選択</li>
            <li>• 予想処理時間: 5-15分（データサイズ依存）</li>
          </ul>
        </div>
      </CardContent>
    </Card>
  );
}
```

---

## 📊 期待効果

### **定量的効果**

| 指標               | 現在         | 強化後      | 改善率 |
| ------------------ | ------------ | ----------- | ------ |
| **特徴量数**       | ~100 個      | ~300 個     | +200%  |
| **特徴量カテゴリ** | 7 カテゴリ   | 10 カテゴリ | +43%   |
| **統計的特徴量**   | 基本統計のみ | 100+統計量  | +1000% |
| **自動選択**       | 手動         | 自動        | 100%   |
| **発見可能性**     | 限定的       | 高度        | +500%  |

### **定性的効果**

#### **✅ 機能強化**

1. **隠れたパターン発見**: 人間では発見困難な複雑な特徴量
2. **統計的妥当性**: 仮説検定による科学的特徴量選択
3. **時系列専用特徴量**: 周波数領域、エントロピー、フラクタル
4. **自動最適化**: 冗長特徴量の自動除去
5. **スケーラビリティ**: 大規模データへの対応

#### **✅ 開発効率向上**

1. **自動化**: 手動特徴量設計の負荷軽減
2. **実験速度**: 高速な特徴量探索
3. **再現性**: 標準ライブラリによる一貫性
4. **保守性**: 実績のあるライブラリ活用

---

## ⚠️ リスク管理

### **主要リスク**

| リスク               | 影響度 | 対策                             |
| -------------------- | ------ | -------------------------------- |
| **計算時間増加**     | 中     | 並列処理、キャッシュ、段階的実行 |
| **メモリ使用量増加** | 中     | データ型最適化、バッチ処理       |
| **特徴量爆発**       | 高     | 自動選択、閾値設定               |
| **過学習リスク**     | 高     | 交差検証、正則化強化             |
| **依存関係競合**     | 低     | 仮想環境、バージョン固定         |

### **対策詳細**

#### **1. 性能最適化**

```python
# 並列処理設定
TSFRESH_N_JOBS = 4
FEATURETOOLS_N_JOBS = 2

# メモリ制限
MAX_FEATURES_PER_BATCH = 50
MEMORY_LIMIT_GB = 8

# キャッシュ設定
FEATURE_CACHE_SIZE = 100
CACHE_TTL_HOURS = 24
```

#### **2. 特徴量選択戦略**

```python
# 段階的選択
FEATURE_SELECTION_STAGES = [
    {'method': 'variance_threshold', 'threshold': 0.01},
    {'method': 'correlation_threshold', 'threshold': 0.95},
    {'method': 'statistical_test', 'fdr_level': 0.05},
    {'method': 'model_based', 'max_features': 100}
]
```

#### **3. 品質保証**

```python
# 特徴量品質チェック
QUALITY_CHECKS = [
    'null_ratio_check',      # 欠損値比率
    'constant_check',        # 定数特徴量
    'duplicate_check',       # 重複特徴量
    'correlation_check',     # 高相関特徴量
    'distribution_check'     # 分布の健全性
]
```

---

## 🧪 テスト戦略

### **テスト階層**

#### **1. 単体テスト**

```python
# backend/tests/feature_engineering/test_automl_features.py

class TestTSFreshCalculator:
    def test_basic_feature_extraction(self):
        """基本的な特徴量抽出テスト"""
        calculator = TSFreshFeatureCalculator()
        test_data = create_test_ohlcv_data()

        result = calculator.calculate_tsfresh_features(test_data)

        assert len(result.columns) > len(test_data.columns)
        assert any('TSF_' in col for col in result.columns)

    def test_feature_selection(self):
        """特徴量選択テスト"""
        calculator = TSFreshFeatureCalculator()
        test_data = create_test_ohlcv_data()
        test_target = create_test_target()

        result = calculator.calculate_tsfresh_features(
            test_data,
            target=test_target,
            feature_selection=True
        )

        # 選択後は特徴量数が減ることを確認
        assert len(calculator.selected_features) > 0
```

#### **2. 統合テスト**

```python
class TestEnhancedFeatureService:
    def test_full_pipeline(self):
        """完全パイプラインテスト"""
        service = EnhancedFeatureEngineeringService()
        test_data = create_comprehensive_test_data()

        result = service.calculate_enhanced_features(
            ohlcv_data=test_data['ohlcv'],
            funding_rate_data=test_data['funding_rate'],
            open_interest_data=test_data['open_interest'],
            target=test_data['target']
        )

        # 手動 + AutoML特徴量が含まれることを確認
        manual_features = [col for col in result.columns if not col.startswith(('TSF_', 'FT_', 'AF_'))]
        automl_features = [col for col in result.columns if col.startswith(('TSF_', 'FT_', 'AF_'))]

        assert len(manual_features) >= 100  # 既存特徴量
        assert len(automl_features) >= 50   # AutoML特徴量
```

#### **3. 性能テスト**

```python
class TestAutoMLPerformance:
    def test_processing_time(self):
        """処理時間テスト"""
        service = EnhancedFeatureEngineeringService()
        large_data = create_large_test_data(rows=10000)

        start_time = time.time()
        result = service.calculate_enhanced_features(large_data)
        processing_time = time.time() - start_time

        # 15分以内で完了することを確認
        assert processing_time < 900

    def test_memory_usage(self):
        """メモリ使用量テスト"""
        import psutil

        process = psutil.Process()
        initial_memory = process.memory_info().rss / 1024 / 1024  # MB

        service = EnhancedFeatureEngineeringService()
        large_data = create_large_test_data(rows=10000)
        result = service.calculate_enhanced_features(large_data)

        final_memory = process.memory_info().rss / 1024 / 1024  # MB
        memory_increase = final_memory - initial_memory

        # メモリ増加が8GB以下であることを確認
        assert memory_increase < 8192
```

---

## 📈 成功指標

### **定量的指標**

#### **特徴量生成**

- [ ] 総特徴量数: 300 個以上
- [ ] TSFresh 特徴量: 50 個以上
- [ ] Featuretools 特徴量: 30 個以上
- [ ] 特徴量選択率: 70%以上

#### **性能指標**

- [ ] 処理時間: 15 分以内
- [ ] メモリ使用量: 8GB 以内
- [ ] 特徴量品質: 95%以上が有効

#### **予測性能**

- [ ] F1 スコア改善: +5%以上
- [ ] 精度改善: +3%以上
- [ ] 過学習抑制: 検証スコア維持

### **定性的指標**

#### **開発効率**

- [ ] 特徴量探索時間: 50%短縮
- [ ] 実験サイクル: 30%高速化
- [ ] 新規特徴量発見: 月 10 個以上

#### **システム品質**

- [ ] 自動化率: 80%以上
- [ ] エラー率: 1%以下
- [ ] 再現性: 100%

---

## 🚀 デプロイ計画

### **段階的デプロイ**

#### **Stage 1: 開発環境**

- [ ] AutoML ライブラリインストール
- [ ] 基本機能実装・テスト
- [ ] 性能ベンチマーク

#### **Stage 2: ステージング環境**

- [ ] 統合テスト実行
- [ ] 負荷テスト実行
- [ ] UI 動作確認

#### **Stage 3: 本番環境**

- [ ] 段階的ロールアウト
- [ ] 監視体制構築
- [ ] フィードバック収集

### **監視項目**

```python
# 監視設定
MONITORING_METRICS = {
    'feature_generation_time': {'threshold': 900, 'unit': 'seconds'},
    'memory_usage': {'threshold': 8192, 'unit': 'MB'},
    'feature_count': {'min': 200, 'max': 500},
    'error_rate': {'threshold': 0.01, 'unit': 'ratio'},
    'cache_hit_rate': {'threshold': 0.8, 'unit': 'ratio'}
}
```

---

## 📚 学習リソース

### **技術ドキュメント**

1. [TSFresh Documentation](https://tsfresh.readthedocs.io/)
2. [Featuretools Documentation](https://featuretools.alteryx.com/)
3. [AutoFeat GitHub](https://github.com/cod3licious/autofeat)

### **参考論文**

1. "TSFRESH: Time Series Feature extraction based on scalable hypothesis tests"
2. "Deep Feature Synthesis: Towards Automating Data Science Endeavors"
3. "Automated Feature Engineering for Financial Time Series"

### **実装例**

1. [TSFresh Financial Examples](https://github.com/blue-yonder/tsfresh/tree/main/notebooks)
2. [Featuretools Time Series Guide](https://featuretools.alteryx.com/en/stable/guides/time_series.html)

---

## 🎯 まとめ

### **プロジェクトの価値**

この AutoML 特徴量エンジニアリング強化により、以下を実現します：

#### **✅ 技術的価値**

1. **特徴量数 3 倍増**: 100 個 → 300 個
2. **自動化率 80%**: 手動作業の大幅削減
3. **発見能力向上**: 人間では困難な複雑パターン発見
4. **科学的妥当性**: 統計的仮説検定による選択

#### **✅ ビジネス価値**

1. **予測精度向上**: F1 スコア +5%、精度 +3%
2. **開発効率化**: 特徴量探索時間 50%短縮
3. **競争優位性**: 高度な特徴量による差別化
4. **スケーラビリティ**: 大規模データ対応

#### **✅ 戦略的価値**

1. **イノベーション**: 最新 AutoML 技術の活用
2. **持続可能性**: 標準ライブラリによる保守性
3. **拡張性**: 新しい特徴量手法の容易な追加
4. **知見蓄積**: 金融データ特有のパターン発見

### **実装優先度**

**Phase 1 (最優先)**: TSFresh 統合

- 時系列・金融データに最適
- 即座に 100+特徴量追加
- 統計的妥当性確保

**Phase 2 (高優先)**: Featuretools 統合

- 既存特徴量との相互作用発見
- Deep Feature Synthesis
- カスタマイズ性

**Phase 3 (中優先)**: AutoFeat 統合

- 最終的な特徴量選択
- 遺伝的アルゴリズム最適化
- 性能向上

---

**既存の優秀な手動特徴量システムに AutoML の力を加えることで、世界クラスの特徴量エンジニアリングシステムを構築できます。**

**実装開始日**: 2025 年 7 月 22 日  
**完了予定日**: 2025 年 8 月 19 日（4 週間）  
**期待 ROI**: 300%以上

🚀 **AutoML 特徴量エンジニアリング強化で、Trdinger を次のレベルへ！**
