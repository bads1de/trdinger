# ML モデル精度改善計画書

## 1. はじめに

### 1.1. 目的

本ドキュメントは、現在トレーニングされている ML モデルの精度が低い問題に対し、コードベース全体を多角的に分析し、その根本原因を特定するとともに、具体的な改善策を提案することを目的とします。当初の分析に加え、追加調査で判明した問題点も統合し、包括的な改善計画を策定します。

### 1.2. 問題の現状

- **現象**: 生成されるモデルの予測精度が 50%前後、またはそれ以下に留まっており、ランダム予測と大差ないレベル。
- **影響**: モデルの予測に基づいた意思決定の信頼性が極めて低く、実用的な戦略開発の妨げとなっている。

---

## 2. 根本原因の分析（初期調査）［現状反映版］

初期調査で列挙した 4 点のうち、現行コードでは 3 点がすでに改善済みであることを確認しました。残る論点は文末の「注意点」を参照してください。

### 2.1. ラベル生成の現状更新

- **現状**: `app/utils/label_generation.py` には FIXED/QUANTILE/STD_DEVIATION/ADAPTIVE/DYNAMIC_VOLATILITY が実装され、学習時は動的法が既定です（例: [`BaseMLTrainer._generate_dynamic_labels()`](backend/app/services/ml/base_ml_trainer.py:545-669)）。
- **修正**: 旧記述「±2%固定に依存」は削除。現状は動的・適応的な閾値生成を使用し、固定はフォールバック用途。

### 2.2. データ前処理の現状更新

- **現状**: 外れ値処理は既定で IQR 法、Z-score はオプション（例: [`DataPreprocessor._remove_outliers(method='iqr')`](backend/app/utils/data_preprocessing.py:214-283)）。前処理呼び出しでは IQR 指定が明示されています（例: [`FeatureEngineeringService.calculate_advanced_features()`](backend/app/services/ml/feature_engineering/feature_engineering_service.py:259-269)）。
- **修正**: 旧記述「Z-score 既定で重要情報が失われる」は削除。現状は IQR 既定で堅牢化済み。

### 2.3. 特徴量スケーリングの現状更新

- **現状**: 特徴量前処理で `scale_features=True` かつ `scaling_method='robust'` を使用（例: [`feature_engineering_service`からの前処理呼び出し](backend/app/services/ml/feature_engineering/feature_engineering_service.py:259-269)、[`BaseMLTrainer._prepare_training_data`](backend/app/services/ml/base_ml_trainer.py:682-692)）。
- **修正**: 旧記述「スケーリング無効（scale_features=False）」は削除。現状はロバストスケーリングを採用済み。

### 2.4. 検証プロセスの現状更新

- **現状**: デフォルトで時系列分割を使用。ウォークフォワード用の `TimeSeriesSplit` 実装もあり（例: [`_split_data`](backend/app/services/ml/base_ml_trainer.py:731-803), [`_time_series_cross_validate`](backend/app/services/ml/base_ml_trainer.py:805-893)）。
- **修正**: 旧記述「ランダム分割でデータリーク」は削除。現状は時系列順の分割に改善済み。

【注意点】
以降の「追加調査」の 3.3（マージ tolerance）と 3.4（ML 指標の FR/OI 未連携）は現時点でも有効な改善対象です。

---

## 3. 追加で特定した問題の根本原因（追加調査）

前回の分析に加え、`app/services/`ディレクトリを中心にコードベースをさらに詳細に調査した結果、モデルの低精度に寄与している可能性のある、これまで指摘されていなかった 4 つの重要な問題点を新たに特定しました。

### 3.1. GA の多様性の欠如：似た戦略ばかりが生き残る「共食い」問題（表現緩和）

- **現状**: 類似度は指標タイプの Jaccard に加え、条件構造、リスク管理、TP/SL、ポジションサイジングの数値近接を重み付きで統合済み（例: [`FitnessSharing._calculate_similarity`](backend/app/services/auto_strategy/engines/fitness_sharing.py:101-159)）。
- **改善余地**: AST ベースの構造比較や各パラメータ距離の正規化、重みのデータ駆動最適化を導入し、局所最適への収束をさらに抑止。
- **表記修正**: 「表面的な類似性に強依存」の断定を緩和し、「多面的ではあるがさらなる拡張余地あり」に更新。

### 3.2. ポジションサイジングの現状更新（過去懸念・現状解消）

- **現状**: `abs(size) < 1` は割合としてそのまま使用し、`abs(size) ≥ 1` のみ整数化（例: [`_adjust_position_size_for_backtesting`](backend/app/services/auto_strategy/factories/strategy_factory.py:226-260)）。
- **修正**: 本節の「割合が 0/1 に丸められる懸念」は現状と不一致のため削除（または「過去の懸念。現状は改善済み」に変更）。

### 3.3. データマージの不整合：時間軸のズレによる「未来予知」

- **問題点**: `app/services/data_mergers/`内の各マージャークラス（`fr_merger.py`, `oi_merger.py`）で、`pd.merge_asof`を使用する際の`tolerance`（許容時間差）が、FR（ファンディングレート）で 8 時間、OI（建玉残高）で 1 日と、比較的長く設定されています。
- **なぜ問題か**: 例えば、1 時間足の OHLCV データに対し、最大 8 時間前の FR データが「現在のデータ」としてマージされる可能性があります。これは、本来その時点では利用できないはずの情報を利用していることになり、一種の**データリーク**を引き起こします。モデルは、この時間的なズレに含まれる「未来の情報」を学習してしまい、バックテスト上でのみ高い精度を達成する可能性があります。
- **リスク**: バックテスト結果が非現実的なほど良好になり、実際の市場では機能しないモデルが「優秀」だと誤って判断されます。

### 3.4. ML 指標計算時のデータ不足：重要な特徴量の欠落

- **問題点**: `app/services/auto_strategy/calculators/indicator_calculator.py`の`calculate_indicator`メソッドが、ML 指標（`ML_UP_PROB`など）を計算する際、`ml_orchestrator`を呼び出していますが、その際に`funding_rate_data`と`open_interest_data`を渡していません（`None`が渡されている）。
- **なぜ問題か**: `MLOrchestrator`は、これらの補助データが存在する場合、それらを用いてより高度な特徴量を生成するように設計されています。しかし、現状の実装ではこのデータ連携が機能しておらず、ML モデルは OHLCV データのみからなる不完全な特徴量セットで予測を行っています。
- **リスク**: モデルが市場のセンチメントや資金フローといった重要な情報を学習できず、予測精度が大幅に低下します。

---

## 4. 改善計画（初期調査分）

本章のうち、4.1〜4.4の一部は現行コードで既に実施済みであるため、完了/維持とする。残存の課題は5章に集約。

### 4.1. ラベル生成の動的化（現状）

- 状態: 実装・導入済み。固定閾値はフォールバック用途で残す。
- 維持/検証: レジーム別の分布監視、動的法のハイパラ最適化（窓、倍率）。

### 4.2. データ前処理の堅牢化：重要なシグナルを守る

- **改善内容**: 外れ値の検出方法を、Z-score からより外れ値に強い**IQR（四分位範囲）**ベースの手法に変更します。
- **具体的な実装**: `app/utils/data_preprocessing.py`の`_remove_outliers`メソッドを修正し、`Q1 - 1.5*IQR`から`Q3 + 1.5*IQR`の範囲外の値を外れ値として扱うロジックを追加します。
- **期待される効果**: 市場の急騰・急落といった本質的な変動を保持しつつ、真に異常なノイズのみを除去することで、データ品質を向上させます。

### 4.3. 特徴量スケーリングの有効化と最適化

- **改善内容**: 特徴量スケーリングを有効化し、外れ値の影響を受けにくい**ロバストスケーリング（RobustScaler）**をデフォルトのスケーリング手法として採用します。
- **具体的な実装**: `app/services/ml/feature_engineering/feature_engineering_service.py`の`preprocess_features`メソッドのデフォルト引数を`scale_features=True`に変更し、`_scale_features`メソッド内で`RobustScaler`を使用するロジックを追加します。
- **期待される効果**: 全ての特徴量が同等のスケールでモデルに貢献できるようになり、学習の安定性と精度が向上します。

### 4.4. 検証プロセスの厳格化：データリークの撲滅

- **改善内容**: データの分割方法を、時系列を正しく扱う**時系列スプリット**（`TimeSeriesSplit`）に変更します。
- **具体的な実装**: `app/services/ml/base_ml_trainer.py`の`_split_data`メソッドを修正し、`sklearn.model_selection.TimeSeriesSplit`を使用して、常に過去のデータで学習し、未来のデータでテストするように分割ロジックを修正します。
- **期待される効果**: データリークを防ぎ、モデルの真の汎化性能（未知のデータに対する予測能力）を正しく評価できるようになります。

---

## 5. 追加改善計画（追加調査分＋新規追記）

### 5.1. GA の多様性促進（拡張案）

- **改善内容**: 既存の多面的類似度に加え、AST ベースの構造差分、指標パラメータ距離の正規化、重みのデータ駆動最適化（ベイズ最適化等）を導入。
- **具体実装**: `_calculate_similarity` を拡張し、条件式の構文木比較と距離学習の導入。

### 5.2. ポジションサイジング（現状維持）

- **現状**: 割合サイズの丸め問題は解消済み。特に追加改修なし。

### 5.3. データマージの厳格化（必須）

- **改善内容**: `pd.merge_asof` の `tolerance` を足幅未満に動的設定（例: 1h 足 →`59min`）。
- **具体実装**: マージ呼出し側で OHLCV の分解能を検出し、[`FRMerger.merge_fr_data`](backend/app/services/data_mergers/fr_merger.py:58-66)/[`OIMerger.merge_oi_data`](backend/app/services/data_mergers/oi_merger.py:58-66) へ厳格値を伝播。

### 5.4. ML 指標計算のデータ連携修正（必須）

- **改善内容**: `GeneratedStrategy` 初期化時に `self.data.df` の `funding_rate`/`open_interest` を抽出して `IndicatorCalculator.calculate_indicator` の ML 分岐へ受け渡し。
- **具体実装**: `strategy_factory.py` の `GeneratedStrategy.init()` で Series を抽出し、ML 指標の計算呼出しに渡す。

### 5.5. 予測確率のキャリブレーションと閾値最適化（新規）

- **背景**: 現状メトリクス（例: AUC-ROC 58.6%、PR-AUC 0%）より確率校正と閾値の不整合が示唆。
- **改善内容**: Platt scaling / Isotonic Regression による確率キャリブレーションを導入し、評価基準（F1/Youden/利益最大化）に基づく閾値最適化を検証パイプラインに組込み。
- **期待効果**: 陽性の適切な確率分布化と PR-AUC/F1 の改善。

### 5.6. クラス不均衡・冗長特徴の対策（新規）

- **改善内容**: クラス重み/サンプルウェイト、Focal 系損失設定、相関・VIF に基づく冗長特徴抑制（ATR/TR 偏重の緩和）。
- **期待効果**: バランス精度・MCC の改善、過度なボラ特徴依存の緩和。

---

## 6. 結論・まとめ（現状反映版）

- 初期調査の 4 点のうち、ラベル生成・外れ値処理・スケーリング・分割は現状で改善済み。記述を更新しました。
- 追加調査では、以下を最優先で改修すべきです。
  - データマージ tolerance の厳格化（リークリスク低減）
  - ML 指標への FR/OI 連携（重要特徴の欠落を解消）
- さらに本版で新規に追記した対策（確率キャリブレーション＋閾値最適化、不均衡・冗長特徴対策）を導入し、PR-AUC 低迷や ATR/TR 偏重を是正します。

これらの対策を段階的に実施することで、汎化性能と安定性の向上が期待できます。
