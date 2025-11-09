# ML改善・整理計画（ドラフト）

この計画は、現在のML関連コードベース（特徴量生成、ラベリング、学習パイプライン、評価スクリプト）を前提に、最小限の変更で実戦的なモデル運用に近づけるための具体的ステップをまとめたものです。

## 1. タスク定義の明確化

### 現状
- 主に「連続値回帰（1本先リターン）」とLabelGeneratorによる3値分類が混在。
- evaluate_feature_performance.py 等では1ステップ先の連続リターン評価が中心で、CV R2が強くマイナス。

### 方針
- 「意味のある1つのタスク」を軸に設計を揃える。
- 第一候補タスク：
  - 4時間先リターンの3値分類（up / range / down）または2値分類（一定閾値以上の上昇か否か）。

### 実装タスク
1. app/utils/label_generation 以下を用いて、以下のような標準ラベルを定義:
   - `UP`: forward 4h リターン >= 上昇閾値（例: +0.2%）
   - `DOWN`: forward 4h リターン <= 下落閾値（例: -0.2%）
   - `RANGE`: その間
2. LabelGenerator / utils に「4h forward用プリセット」を追加（既存API互換を維持）。
3. MLパイプライン（BaseMLTrainer/MLTrainingService）で、デフォルトターゲットとしてこのラベルを選択できる設定キーを追加。

## 2. 特徴量サブセットの整理（Production Profile）

### 現状
- FeatureEngineeringService + technical/price/crypto/advanced/interaction で非常に多くの特徴量を生成。
- detect_low_importance_features.py / analyze_feature_importance.py / evaluate_feature_performance.py で分析しているが、「本番で使う固定サブセット」がない。

### 方針
- 「研究用: フル特徴量」「本番用: 厳選サブセット」を明確に分離。

### 実装タスク
1. FeatureEngineeringService に production 用の allowlist を導入:
   - 例: `FEATURE_PROFILE = {"research": None, "production": ["RSI", "MACD", "MA_Long", "BB_Position", ...]}`
   - production指定時はallowlistにない列を最終出力からドロップ。
2. detect_low_importance_features.py / analyze_feature_importance.py / evaluate_feature_performance.py を共通のターゲット設定＋CommonFeatureEvaluator経由で実行し、
   - 複数モデル・複数シナリオで一貫して低重要度な特徴を抽出。
   - その結果を元に production allowlist を更新（手動レビュー込み）。
3. MLTrainingService/BaseMLTrainer の本番利用時は production プロファイルを使うよう設定可能にする（例: `feature_profile="production"`）。

## 3. 評価・学習パイプラインの一貫化

### 現状
- BaseMLTrainer 内で train_test_split と TimeSeriesSplit が共存。
- スクリプト側とサービス側で評価前提が完全には揃っていない。

### 方針
- 時系列タスクとしての一貫性を担保する。

### 実装タスク
1. BaseMLTrainer:
   - デフォルトを時系列CV（TimeSeriesSplit）＋最終的な全データ再学習に寄せる。
   - ランダムな train_test_split は明示フラグ指定時のみ使うようにする。
2. MLTrainingService:
   - `determine_trainer_type` 等のロジックを維持しつつ、時系列CV利用時のパラメータ（fold数など）を ml_config から統一管理。
3. 評価スクリプト:
   - evaluate_feature_performance.py は CommonFeatureEvaluator を利用し、TimeSeriesSplitを内部で明示。
   - detect_low_importance_features.py / analyze_feature_importance.py も同じ forward ラベルとCV前提を使用（既にCommonFeatureEvaluator導入済みのため微調整で整合可能）。

## 4. リーク/不安定要因の監視ポイント

### 既に対処済み
- technical_features.py の Local_Min/Local_Max は将来参照ロジックを廃止し、過去窓のみで定義。
- 評価スクリプトの文字コード問題（R²）修正済み。
- DB接続は .env 経由に統一（絶対パスの直書きを排除）。

### 今後の注意点
1. LabelGenerator:
   - DataFrame + target_column 経路など、モードによりforward/backwardが変わる箇所を明確化し、学習パイプラインでどのモードを使うか固定する。
2. pseudo FR/OI 特徴:
   - 本番用途では基本的に無効化し、実データがある場合のみ使用するポリシーをコードコメントと設定で明示。
3. interaction_features / crypto_features:
   - 必須特徴が欠ける場合は自動スキップされる設計だが、その挙動が評価スクリプトと本番で一致しているかを定期確認。

## 5. 実行ステップ（まとめ）

1. ターゲット決定:
   - 4h forward 3値 or 2値タスクを1つ選び、その定義をLabelGenerator/評価スクリプト/MLTrainingServiceに反映。
2. 評価ループ:
   - 共通ターゲットで3スクリプトを実行し、安定して低重要度な特徴を洗い出す。
3. Production Profile定義:
   - FeatureEngineeringServiceにproduction用の特徴量リストを実装し、本番学習・推論はこれを利用するように変更。
4. 学習パイプライン統一:
   - BaseMLTrainer/MLTrainingServiceのデフォルトをTimeSeriesSplit＋forwardラベルに揃える。
5. モデル確定:
   - 上記条件でSingleModelTrainer（LightGBM/XGBoost）により1本の本番モデルを学習・保存し、そのモデルを運用に使う。

この計画は既存構造を壊さずに「タスク定義」「特徴量サブセット」「評価方法」を揃えることに集中しています。
次のステップとして、選びたいターゲット（例: 4h forward 3値 or 2値）を教えてもらえれば、その前提で具体的なコード変更案に落とし込めます。
