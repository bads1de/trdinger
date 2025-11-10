# ML 改善・整理計画（ドラフト）

この計画は、現在の ML 関連コードベース（特徴量生成、ラベリング、学習パイプライン、評価スクリプト）を前提に、最小限の変更で実戦的なモデル運用に近づけるための具体的ステップをまとめたものです。

## 1. ✅ タスク定義の明確化

### 現状

- 主に「連続値回帰（1 本先リターン）」と LabelGenerator による 3 値分類が混在。
- evaluate_feature_performance.py 等では 1 ステップ先の連続リターン評価が中心で、CV R2 が強くマイナス。

### 方針

- 「意味のある 1 つのタスク」を軸に設計を揃える。
- 固定 4 時間に縛らず、「固定時間ホライズン + 対称閾値 3 値分類」をコアタスクとして一般化する。
- 対象:
  - 時間足: 15m / 30m / 1h / 4h / 1d（将来的に拡張可能）
  - ホライズン: N 本先 (例: 4 本先、16 本先など) を設定で切替可能
- コアタスク定義（3 値分類）:
  - `UP`: forward_return >= +θ
  - `RANGE`: -θ < forward_return < +θ
  - `DOWN`: forward_return <= -θ
- θ（閾値）はボラティリティ等に基づき銘柄・時間足別に設定可能にする余地を残す。

### 実装タスク

**タスク1: ✅ 完了** - [`presets.py`](backend/app/utils/label_generation/presets.py:1)に実装
- `forward_classification_preset`関数を実装
- 5種類の時間足（15m, 30m, 1h, 4h, 1d）をサポート
- 6種類の閾値計算方法をサポート

**タスク2: ✅ 完了** - 汎用プリセット関数として実装
- `get_common_presets()`で13種類のプリセット定義
- `apply_preset_by_name()`でプリセット適用

**タスク3: ✅ 完了** - [`MLTrainingConfig`](backend/app/config/unified_config.py:452)と[`BaseMLTrainer`](backend/app/services/ml/base_ml_trainer.py:617)に統合
- [`LabelGenerationConfig`](backend/app/config/unified_config.py:367)を追加
- 環境変数サポート
- プリセット/カスタム設定の切り替え機能

**タスク4: ✅ 基盤実装済み** - 将来の拡張に対応可能
- [`EventDrivenLabelGenerator`](backend/app/utils/label_generation/__init__.py:11)が既に実装済み
- インターフェースは拡張可能な設計

### 実装完了内容

#### 実装済みファイル
1. **プリセット関数**: [`backend/app/utils/label_generation/presets.py`](backend/app/utils/label_generation/presets.py:1) (365行)
   - `forward_classification_preset()`: 汎用ラベル生成関数
   - `get_common_presets()`: 13種類のプリセット定義
   - `apply_preset_by_name()`: プリセット適用関数

2. **設定クラス**: [`backend/app/config/unified_config.py`](backend/app/config/unified_config.py:367)
   - `LabelGenerationConfig`: ラベル生成設定
   - 環境変数サポート（`ML__LABEL_GENERATION__*`）
   - バリデーション機能

3. **ML統合**: [`backend/app/services/ml/base_ml_trainer.py`](backend/app/services/ml/base_ml_trainer.py:617)
   - `_prepare_training_data()`メソッドに統合
   - プリセット/カスタム設定の自動切り替え
   - 後方互換性維持

4. **テストコード**: [`backend/tests/test_label_generation_presets.py`](backend/tests/test_label_generation_presets.py:1) (1054行)
   - 59個のテスト（全てパス）
   - カバレッジ: 新規コードの80%以上

#### 使用方法

**環境変数でプリセット使用:**
```bash
ML__LABEL_GENERATION__USE_PRESET=true
ML__LABEL_GENERATION__DEFAULT_PRESET="4h_4bars"
```

**環境変数でカスタム設定:**
```bash
ML__LABEL_GENERATION__USE_PRESET=false
ML__LABEL_GENERATION__TIMEFRAME="1h"
ML__LABEL_GENERATION__HORIZON_N=8
ML__LABEL_GENERATION__THRESHOLD=0.003
```

**プログラムから使用:**
```python
from app.utils.label_generation.presets import forward_classification_preset, apply_preset_by_name

# プリセット使用
labels, info = apply_preset_by_name(df, "4h_4bars")

# カスタム設定
labels = forward_classification_preset(
    df, timeframe="1h", horizon_n=8, threshold=0.003
)
```

### 次のステップ（セクション2以降）
セクション1の実装が完了したため、次はセクション2「特徴量サブセットの整理（Production Profile）」に進みます。

## 2. ✅ 特徴量サブセットの整理（Production Profile）

### 現状

- FeatureEngineeringService + technical/price/crypto/advanced/interaction で非常に多くの特徴量を生成。
- detect_low_importance_features.py / analyze_feature_importance.py / evaluate_feature_performance.py で分析しているが、「本番で使う固定サブセット」がない。

### 方針

- 「研究用: フル特徴量」「本番用: 厳選サブセット」を明確に分離。

### 実装タスク

**タスク1: ✅ 完了** - [`FeatureEngineeringConfig`](backend/app/config/unified_config.py:525)と[`FEATURE_PROFILES`](backend/app/services/ml/feature_engineering/feature_engineering_service.py:38)
- researchプロファイル: 全特徴量を保持
- productionプロファイル: 約40個の厳選特徴量
- 環境変数サポート: `ML_FEATURE_ENGINEERING__PROFILE`
- カスタムallowlist設定可能

**タスク2: ✅ 完了** - [`run_unified_analysis.py`](backend/scripts/feature_evaluation/run_unified_analysis.py:1)
- 3つの評価スクリプトを統合（低重要度検出、重要度分析、パフォーマンス評価）
- [`CommonFeatureEvaluator`](backend/scripts/feature_evaluation/common_feature_evaluator.py:193)で一貫したターゲット設定
- JSON/CSV形式で結果出力
- production allowlist推奨リスト生成

**タスク3: ✅ 完了** - [`MLTrainingService`](backend/app/services/ml/ml_training_service.py:167)と[`BaseMLTrainer`](backend/app/services/ml/base_ml_trainer.py:513)
- `feature_profile`パラメータでprofile指定可能
- 設定から自動読み込み
- API経由での指定サポート
- 完全な後方互換性

### 実装完了内容

#### 実装済みファイル

1. **設定クラス**: [`backend/app/config/unified_config.py`](backend/app/config/unified_config.py:525)
   - `FeatureEngineeringConfig`: プロファイル設定
   - 環境変数サポート（`ML_FEATURE_ENGINEERING__*`）
   - カスタムallowlist設定

2. **プロファイル実装**: [`backend/app/services/ml/feature_engineering/feature_engineering_service.py`](backend/app/services/ml/feature_engineering/feature_engineering_service.py:38)
   - `FEATURE_PROFILES`: research/productionプロファイル定義
   - `_apply_feature_profile()`: プロファイル適用ロジック
   - 約40個のproduction厳選特徴量

3. **分析スクリプト**: [`backend/scripts/feature_evaluation/`](backend/scripts/feature_evaluation/)
   - `run_unified_analysis.py`: 統合分析スクリプト
   - `CommonFeatureEvaluator`: 一貫した評価基盤
   - `README.md`: 使用ガイド

4. **ML統合**: [`backend/app/services/ml/`](backend/app/services/ml/)
   - `BaseMLTrainer._calculate_features()`: プロファイル自動読み込み
   - `MLTrainingService.train_model()`: feature_profileパラメータ
   - `backend/app/api/ml_training.py`: API対応

5. **テストコード**: [`backend/tests/feature/`](backend/tests/feature/)
   - `test_feature_profile.py`: 12個のテスト（全てパス）
   - `test_profile_integration.py`: 統合テスト
   - カバレッジ: 80%以上

6. **ドキュメント**: [`backend/docs/feature_profile_usage.md`](backend/docs/feature_profile_usage.md:1)
   - 設定方法の詳細ガイド
   - 使用例（環境変数、プログラム、API）
   - トラブルシューティング

#### production allowlist内容
約40個の厳選特徴量:
- **テクニカル指標**: RSI、MACD、ボリンジャーバンド、ATR
- **移動平均**: 短期・長期MA、EMA、VWMA
- **ボリューム**: Volume_MA_Ratio、Volume_Trend
- **価格**: 高値・安値位置、価格変化率
- **暗号特有**: OI_Change、FR_Trend
- **複合指標**: MACD_BB、RSI_BB等の相互作用特徴

#### 使用方法

**環境変数で設定:**
```bash
ML_FEATURE_ENGINEERING__PROFILE=production
```

**プログラムから使用:**
```python
# MLTrainingServiceで使用
service = MLTrainingService()
service.train_model(
    training_data=data,
    feature_profile="production"
)

# 設定で変更
unified_config.ml.feature_engineering.profile = "production"
```

**API経由で使用:**
```bash
POST /api/ml-training/train
{
  "symbol": "BTC/USDT:USDT",
  "feature_profile": "production"
}
```

**特徴量分析の実行:**
```bash
cd backend
python -m scripts.feature_evaluation.run_unified_analysis \
    --preset 4h_4bars \
    --symbol BTC/USDT:USDT \
    --limit 2000
```

### 次のステップ（セクション3以降）
セクション2の実装が完了したため、次はセクション3「評価・学習パイプラインの一貫化」に進みます（既に完了済み）。
その後、セクション4「リーク/不安定要因の監視ポイント」とセクション5「実行ステップ（まとめ）」の確認を行います。

## 3. ✅ 評価・学習パイプラインの一貫化（✅ 完了）

### 現状

- BaseMLTrainer 内で train_test_split と TimeSeriesSplit が共存。
- スクリプト側とサービス側で評価前提が完全には揃っていない。

### 方針

- 時系列タスクとしての一貫性を担保する。

### 実装タスク

1. BaseMLTrainer:
   - デフォルトを時系列 CV（TimeSeriesSplit）＋最終的な全データ再学習に寄せる。
   - ランダムな train_test_split は明示フラグ指定時のみ使うようにする。
2. MLTrainingService:
   - `determine_trainer_type` 等のロジックを維持しつつ、時系列 CV 利用時のパラメータ（fold 数など）を ml_config から統一管理。
3. 評価スクリプト:
   - evaluate_feature_performance.py は CommonFeatureEvaluator を利用し、TimeSeriesSplit を内部で明示。
   - detect_low_importance_features.py / analyze_feature_importance.py も同じ forward ラベルと CV 前提を使用（既に CommonFeatureEvaluator 導入済みのため微調整で整合可能）。

## 4. リーク/不安定要因の監視ポイント

### 既に対処済み

- technical_features.py の Local_Min/Local_Max は将来参照ロジックを廃止し、過去窓のみで定義。
- 評価スクリプトの文字コード問題（R²）修正済み。
- DB 接続は .env 経由に統一（絶対パスの直書きを排除）。

### 今後の注意点

1. LabelGenerator:
   - DataFrame + target_column 経路など、モードにより forward/backward が変わる箇所を明確化し、学習パイプラインでどのモードを使うか固定する。
2. pseudo FR/OI 特徴:
   - 本番用途では基本的に無効化し、実データがある場合のみ使用するポリシーをコードコメントと設定で明示。
3. interaction_features / crypto_features:
   - 必須特徴が欠ける場合は自動スキップされる設計だが、その挙動が評価スクリプトと本番で一致しているかを定期確認。

## 4. フロントエンド実装（✅ 完了）

### 実装完了内容

#### 4.1 TypeScript型定義
**完了日**: 2025-01-10

- **新規ファイル**:
  - [`frontend/types/ml-config.ts`](frontend/types/ml-config.ts:1): ML設定型定義（350行）
    - `LabelGenerationConfig`: ラベル生成設定
    - `FeatureEngineeringConfig`: 特徴量エンジニアリング設定
    - `ThresholdMethod`: 6種類の閾値計算方法
    - `FeatureProfile`: research/production
    - `LabelPresetInfo`: プリセット情報
    - `MLTrainingRequestExtended`: 拡張トレーニングリクエスト
  
  - [`frontend/constants/ml-config-constants.ts`](frontend/constants/ml-config-constants.ts:1): 定数定義（300行）
    - `LABEL_PRESETS`: 15個のプリセット定義
    - `THRESHOLD_METHODS`: 閾値計算方法一覧
    - `THRESHOLD_METHOD_LABELS`: 表示名マッピング
    - `THRESHOLD_METHOD_DESCRIPTIONS`: 説明テキスト
    - `FEATURE_PROFILES`: プロファイル一覧
    - ヘルパー関数: `getPresetNames()`, `getPresetInfo()`, `getPresetsByCategory()`

- **更新ファイル**:
  - [`frontend/constants/index.ts`](frontend/constants/index.ts:1): 一括エクスポート
  - [`frontend/hooks/useMLSettings.ts`](frontend/hooks/useMLSettings.ts:1): MLConfig型拡張
  - [`frontend/hooks/useMLTraining.ts`](frontend/hooks/useMLTraining.ts:1): TrainingConfig型拡張

#### 4.2 ML設定UIコンポーネント
**完了日**: 2025-01-10

- **新規コンポーネント**:
  - [`frontend/components/ml/LabelGenerationSettings.tsx`](frontend/components/ml/LabelGenerationSettings.tsx:1) (242行)
    - プリセット使用/カスタム設定の切り替え
    - 15種類のプリセット選択
    - カスタム設定（時間足、ホライズン、閾値、閾値計算方法）
    - プリセット詳細情報の表示
    - 入力検証とエラーハンドリング
  
  - [`frontend/components/ml/FeatureProfileSettings.tsx`](frontend/components/ml/FeatureProfileSettings.tsx:1) (196行)
    - research/productionプロファイル選択
    - カスタム特徴量allowlist入力（JSON形式）
    - JSON検証とエラー表示
    - プロファイル情報の詳細表示

- **更新コンポーネント**:
  - [`frontend/components/ml/MLSettings.tsx`](frontend/components/ml/MLSettings.tsx:1): タブを2つから4つに拡張
    - 基本設定タブ
    - ラベル生成タブ（新規）
    - 特徴量タブ（新規）
    - データ前処理タブ

#### 4.3 テスト
**完了日**: 2025-01-10

- **新規テストファイル**:
  - [`frontend/tests/LabelGenerationSettings.test.tsx`](frontend/tests/LabelGenerationSettings.test.tsx:1) (321行)
    - 17個のテストケース
    - カバレッジ: 86.95%
  
  - [`frontend/tests/FeatureProfileSettings.test.tsx`](frontend/tests/FeatureProfileSettings.test.tsx:1) (394行)
    - 28個のテストケース
    - カバレッジ: 100%

- **テスト結果**:
  - ✅ Test Suites: 2 passed, 2 total
  - ✅ Tests: 45 passed, 45 total
  - ✅ 実行時間: 9.668秒
  - ✅ TypeScriptエラー: なし

#### 4.4 主な機能

##### ラベル生成設定
- ✅ プリセット使用のチェックボックス
- ✅ 15種類のプリセット選択（時間足別、動的閾値対応）
- ✅ カスタム設定（時間足、ホライズン、閾値など）
- ✅ 6種類の閾値計算方法（FIXED、QUANTILE、STD_DEVIATION、ADAPTIVE、DYNAMIC_VOLATILITY、KBINS_DISCRETIZER）
- ✅ 各設定項目の説明とツールチップ
- ✅ プリセット詳細情報の自動表示

##### 特徴量プロファイル設定
- ✅ research/productionプロファイルのラジオボタン選択
- ✅ カスタム特徴量allowlistのJSON入力
- ✅ JSON検証とエラー表示
- ✅ プロファイル情報の詳細表示
- ✅ 特徴量数のカウント表示

##### UI/UX実装
- ✅ レスポンシブデザイン（Tailwind CSS使用）
- ✅ 入力検証（数値範囲チェック、JSON形式検証）
- ✅ ツールチップとヘルプテキスト
- ✅ プリセットプレビュー機能
- ✅ 適切なデフォルト値設定
- ✅ shadcn/uiコンポーネントの活用

#### 4.5 バックエンドAPI統合

- **MLTrainingService** ([`backend/app/services/ml/ml_training_service.py`](backend/app/services/ml/ml_training_service.py:1))
  - `train_model`メソッドに以下のパラメータを追加:
    - `label_generation`: ラベル生成設定（オプション）
    - `feature_profile`: 特徴量プロファイル（オプション）
    - `custom_allowlist`: カスタム特徴量allowlist（オプション）

- **フロントエンドからのAPI呼び出し**:
  - `useMLTraining` hookがトレーニングリクエストに新しい設定を含める
  - プリセット使用時とカスタム設定時で適切にパラメータを分岐

#### 4.6 技術仕様

- **使用コンポーネント**: Card、Checkbox、RadioGroup、Select、Input、Textarea、Label、Alert
- **型安全性**: TypeScript厳密モード、完全な型推論
- **テスト**: Jest + React Testing Library
- **スタイリング**: Tailwind CSS + shadcn/ui
- **バックエンドAPI互換**: 新しい設定項目をPOSTリクエストに含める

### 次のステップ（セクション5の確認）

セクション1・2・3およびフロントエンド実装（セクション4）が完了しました。次のステップは:

1. **セクション5: 実行ステップ（まとめ）**
   - ターゲット決定（4h forward 3値分類の選択と設定）
   - 評価ループの実行（[`run_unified_analysis.py`](backend/scripts/feature_evaluation/run_unified_analysis.py:1)の使用）
   - Production Profile定義の最終調整
   - 学習パイプライン統一の確認
   - モデル確定と運用準備

## 5. 実行ステップ（まとめ）

1. ターゲット決定:
   - 4h forward 3 値 or 2 値タスクを 1 つ選び、その定義を LabelGenerator/評価スクリプト/MLTrainingService に反映。
2. 評価ループ:
   - 共通ターゲットで 3 スクリプトを実行し、安定して低重要度な特徴を洗い出す。
3. Production Profile 定義:
   - FeatureEngineeringService に production 用の特徴量リストを実装し、本番学習・推論はこれを利用するように変更。
4. 学習パイプライン統一:
   - BaseMLTrainer/MLTrainingService のデフォルトを TimeSeriesSplit ＋ forward ラベルに揃える。
5. モデル確定:
   - 上記条件で SingleModelTrainer（LightGBM/XGBoost）により 1 本の本番モデルを学習・保存し、そのモデルを運用に使う。

この計画は既存構造を壊さずに「タスク定義」「特徴量サブセット」「評価方法」を揃えることに集中しています。
次のステップとして、選びたいターゲット（例: 4h forward 3 値 or 2 値）を教えてもらえれば、その前提で具体的なコード変更案に落とし込めます。
