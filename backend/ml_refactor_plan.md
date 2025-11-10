# ML 改善・整理計画（ドラフト）

この計画は、現在の ML 関連コードベース（特徴量生成、ラベリング、学習パイプライン、評価スクリプト）を前提に、最小限の変更で実戦的なモデル運用に近づけるための具体的ステップをまとめたものです。

## 1. タスク定義の明確化

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

1. app/utils/label_generation 以下を用いて、以下のような汎用 forward ラベルプリセットを定義:
   - 引数: `timeframe` (例: "15m", "30m", "1h", "4h", "1d"), `horizon_n` (N 本先), `threshold` (対称閾値, 例: 0.2%), `price_column`。
   - 出力: `UP` / `RANGE` / `DOWN` の 3 値ラベル列。
2. LabelGenerator / utils に「forward_classification_preset」を追加:
   - 既存 API 互換を維持しつつ、上記引数で任意時間足・任意 N 本先を指定可能。
   - 将来のトリプルバリア法や 2 値分類タスク追加に対応できる拡張性あるインターフェースにする。
3. ML パイプライン（BaseMLTrainer/MLTrainingService）で、
   - デフォルトターゲットとしてこのプリセットラベルを選択できる設定キーを追加
   - 少なくとも「15m, 30m, 1h, 4h, 1d × 代表的 N 本先」の組み合わせを設定で切替可能にする。
4. 将来拡張としてのトリプルバリア法対応を明示:
   - 現フェーズでは採用しないが、LabelGenerator のプリセット設計を拡張可能にしておき、
     triple_barrier_preset のような手法を後から追加できるインターフェースにする。
   - トリプルバリア法は、TP/SL/時間バリアに基づく実運用寄りのラベリングとして有用だが、
     実装・検証コストが高いため、本計画のスコープ外とし、将来の高度化タスクとして位置付ける。

## 2. 特徴量サブセットの整理（Production Profile）

### 現状

- FeatureEngineeringService + technical/price/crypto/advanced/interaction で非常に多くの特徴量を生成。
- detect_low_importance_features.py / analyze_feature_importance.py / evaluate_feature_performance.py で分析しているが、「本番で使う固定サブセット」がない。

### 方針

- 「研究用: フル特徴量」「本番用: 厳選サブセット」を明確に分離。

### 実装タスク

1. FeatureEngineeringService に production 用の allowlist を導入:
   - 例: `FEATURE_PROFILE = {"research": None, "production": ["RSI", "MACD", "MA_Long", "BB_Position", ...]}`
   - production 指定時は allowlist にない列を最終出力からドロップ。
2. detect_low_importance_features.py / analyze_feature_importance.py / evaluate_feature_performance.py を共通のターゲット設定＋ CommonFeatureEvaluator 経由で実行し、
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
