# ML サービス リファクタリング提案（再構成版）

対象: [`backend/app/services/ml/`](backend/app/services/ml/)

目的:

- 責務分離の徹底、重複排除、設定/ログ/インターフェースの標準化により、保守性・拡張性・可観測性・性能を総合的に向上させる。

構成:

1. 全体課題と基本方針
2. 個別リファクタリング提案（抜粋）
3. utils 配下の具体的改善提案
4. 実装優先度と進め方
5. 期待効果（定量指標）
6. 付録（参照・実装例）

---

## 1. 全体課題と基本方針

課題（要約）

- 責務の混在: `BaseMLTrainer` やオーケストレーション層に複数責務が集中。
- コード重複: モデルラッパー群や評価ロジックの重複。
- 設定の煩雑さ: Pydantic モデルと辞書の手動相互変換。
- デッドコード: 参照されないファイルの残存（例: ensemble 周辺の旧実装）。

基本方針

- 単一責任の原則で層分解（config/feature/model/evaluation/orchestrator）。
- 共通処理のユーティリティ化と抽象化（Strategy/Registry/Factory）。
- 設定・ログ・エラーハンドリングの標準化。
- 互換 API は段階的に非推奨化し、移行ガイドを明示。

---

## 2. 個別リファクタリング提案（抜粋）

[ ] 2.4 設定管理の改善

- 課題:
  - `config/ml_config_manager.py` の `get_config_dict` が手動マッピングで脆弱。
  - `base_ml_trainer.py` の `_create_automl_config_from_dict` も手動変換。
- 提案:
  - Pydantic モデル（`MLConfig`, `AutoMLConfig`）にシリアライズ API を実装。
  - `to_dict()/from_dict()` は Pydantic v2 の `.model_dump()` を内部使用。
- 実装例（抜粋）:
  - [`ml_config.MLConfig.to_dict()`](backend/app/services/ml/config/ml_config.py:1)
  - [`ml_config.MLConfig.from_dict()`](backend/app/services/ml/config/ml_config.py:1)
  - Python.example():
    ```python
    class MLConfig(BaseModel):
        # fields...
        def to_dict(self) -> dict:
            return self.model_dump()
        @classmethod
        def from_dict(cls, data: dict) -> "MLConfig":
            return cls(**data)
    ```

[ ] 2.15 インターフェースの未活用

- 課題:
  - `MLPredictionInterface`, `MLTrainingInterface`, `MLServiceInterface` は定義済だが実装で未活用。
- 提案:
  - 既存サービス（例: MLTrainingService）に実装適用。
  - 依存性注入（DI）を導入し、モック容易性とテスト性を向上。
  - 仕様を使用実態に合わせ再定義（返却型と例外契約を固定）。

[ ] 2.20 ログメッセージの重複パターン

- 課題:
  - 「ML トレーニング…エラー」「AutoML…」などの定型が乱立。絵文字使用に一貫性なし。
- 提案:
  - `MLLogMessages`（定数/関数）でメッセージ規約を一元管理。
  - レベルごとの運用基準と構造化ログ（extra）方針を定義。
- 期待効果:
  - 運用時のノイズ低減、検索性・相関分析性の向上。

[ ] 2.22 ML 設定管理の分散

- 課題:
  - `MLConfig`, `MLConfigManager`, `AutoMLConfig` に加えハードコード設定が混在。
- 提案:
  - `BaseMLConfig` を基底に階層化、継承で共通と専用の境界を明確化。
  - 検証/デフォルト/環境変数オーバーライドを統一処理。

[ ] 2.23 インターフェース実装の不完全性

- 課題:
  - `MLOrchestrator` が `MLPredictionInterface` を実装するが未完部分あり。
- 提案:
  - 抽象基底クラス(ABC)で未実装検出を強制。契約（inputs/outputs/exceptions）を明文化。

---

## 3. utils 配下の具体的改善提案

[ ] 3.1 preprocessing と validation の境界再定義

- 課題:
  - 欠損/外れ値処理が [`data_preprocessing.preprocess_features()`](backend/app/utils/data_preprocessing.py:157) と [`data_validation.clean_dataframe()`](backend/app/utils/data_validation.py:349) に重複・分散。
- 提案:
  - validation は「検出/レポート（最小限の安全置換）」に限定。
  - preprocessing は「補完/外れ値除去/スケーリング」を一元化。
  - パイプライン: 検出 → 補完 → 標準化（順序を固定化）。
- 効果:
  - 二重処理防止、順序依存の不具合低減、テスト容易性向上。

[ ] 3.3 型・スキーマ整合性の標準化

- 課題:
  - [`standardize_ohlcv_columns`](backend/app/utils/data_standardization.py:10) が列名変換中心で型や Index 保証が弱い。
- 提案:
  - オプション追加: ensure_datetime_index, sort_index, coerce_dtypes。
  - DataCleaner 側は「標準化後の監査」に縮小。
- 効果:
  - 受け渡し時点での一回の標準化で後工程が安定。

[ ] 3.4 data_converter と data_utils の役割明確化

- 課題:
  - dtype ポリシーの不一致、Series/Array 変換の散在。
- 提案:
  - 変換ポリシー（例: 学習=float32、評価/保存=float64）を設定化。
  - シリーズ/配列の入口は data_utils に集約、converter は入出力表現変換に限定。

[ ] 3.5 Safe\* 系戻り型と NaN/Inf 方針の統一

- 提案:
  - すべて `pd.Series` を返す規約に統一（Index は基準 Series を継承）。
  - NaN/Inf 置換ポリシーを `DataValidator` 定数へ集約・上書き可能化。

[ ] 3.6 metrics_calculator の移行

- 提案:
  - `calculate_detailed_metrics` に PendingDeprecationWarning を付与し、利用側を `EnhancedMetricsCalculator` へ置換。
  - `get_default_metrics` は初期化用途に限定。

[ ] 3.7 unified_error_handler の契約分離

- 提案:
  - API 向け: 例外送出（HTTP 例外化）/ ML 向け: 値返却 の 2 系統を明示公開。
  - `create_error_response` のスキーマ固定（JSON-Schema 相当）。

[ ] 3.8 database_utils の方言戦略

- 提案:
  - DialectStrategy 導入（PostgreSQL/SQLite 実装）。例外粒度で分岐、バッチ設定は注入。
  - ユニットテストで重複/部分成功/コミット失敗/再試行/ログ検証を網羅。

[ ] 3.9 label_generation のスケール最適化

- 提案:
  - 統計量のキャッシュ（std/quantile 等）、近似（t-digest）やサンプリングで加速。
  - 戻りメタは TypedDict/Dataclass へ移行。

[ ] 3.11 API ユーティリティの時刻統一

- 提案:
  - `DateTimeHelper.now_utc_iso()` を追加して UTC+Z ISO に統一。
  - `APIResponseHelper`/`error_response`/`api_response` は DateTimeHelper に委譲。

[ ] 3.12 循環依存の回避

- 提案:
  - `core_utils/transform` と `domain/cleaners` に層分割。ローカル import で循環回避。

[ ] 3.13 ログ文言/レベル標準化

- 提案:
  - info=進捗/件数、warning=回復可能、error=失敗時。重要パラメータは構造化ログ(extra)。

[ ] 3.14 テスト整備（単体/プロパティ/回帰）

- 提案:
  - hypothesis で safe\_\* 数値安定性、時系列一貫性の順序不変性、DB 挿入の各種シナリオを網羅。

---

## 4. 実装優先度と進め方

高優先度（即着手）

- 2.4 設定管理の改善（Pydantic の正規化 API）
- 3.1 前処理/検証の境界再定義とパイプライン一本化
- 3.6 メトリクス計算の移行ガイド整備（非推奨警告付与）

中優先度（次スプリント）

- 2.15/2.23 インターフェース適用と ABC 化、DI 導入
- 3.3 型・スキーマ標準化（OHLCV 強化）
- 3.11 時刻取り扱いの統一

低優先度（継続的改善）

- 2.20/3.13 ログ標準化・テンプレート化
- 3.8 DB 方言戦略 + テスト強化
- 3.9 分布最適化・ストリーミング化
- 3.12 層分割と循環依存対策

進め方（推奨プロセス）

1. 設計合意（契約/型/ログ/設定の規約を合意）
2. フラグ併用による段階移行（旧 API は PendingDeprecation）
3. 移行時のカナリア/回帰テストを優先導入
4. 文書化（How-to/ガイド/プレイブック）

---

## 5. 期待効果（定量指標）

- コード重複削減: 40–50%
- 変更影響範囲の縮小: 修正箇所 70%削減
- テスト容易性: インターフェース/契約標準化によりカバレッジ向上
- パフォーマンス: メモリ/CPU の安定化（無駄なコピー/二重処理の排除）
- 開発効率: 新規モデル/特徴量追加の所要時間 60%短縮

---

## 6. 付録（参照・実装例）

A) 参照リンク（抜粋）

- 前処理と検証の重複箇所
  - [`data_preprocessing.preprocess_features`](backend/app/utils/data_preprocessing.py:157)
  - [`data_validation.clean_dataframe`](backend/app/utils/data_validation.py:349)
- 型/スキーマ整合性
  - [`data_standardization.standardize_ohlcv_columns`](backend/app/utils/data_standardization.py:10)
  - [`DataCleaner.validate_ohlcv_data`](backend/app/utils/data_cleaning_utils.py:160)

B) 実装スニペット（ガイド用）

- 設定モデル（Pydantic v2）.python():

  ```python
  class AutoMLConfig(BaseModel):
      enable: bool = True
      max_trials: int = 50
      random_state: int = 42

      def to_dict(self) -> dict:
          return self.model_dump()

      @classmethod
      def from_dict(cls, data: dict) -> "AutoMLConfig":
          return cls(**data)
  ```

- 統一ログテンプレート.python():
  ```python
  class MLLogMessages:
      TRAIN_START = "ML training started"
      TRAIN_FAIL = "ML training failed"
      PREDICT_START = "Prediction started"
      PREDICT_FAIL = "Prediction failed"
      # builderやformat関数でextraへ構造化情報を付加
  ```

C) 非推奨化の流れ

- ステップ 1: 警告付与（PendingDeprecationWarning）
- ステップ 2: 参照置換の PR 群
- ステップ 3: 次期での削除予告
- ステップ 4: 削除・移行完了

---

総括

- 本提案は、設定・前処理・評価・インターフェース・ログ・DB 方言・テストの 7 領域を横断して、保守性と運用性のボトルネックを体系的に削減する。段階移行とテスト先行で、安全に品質向上と開発速度を両立できる。
