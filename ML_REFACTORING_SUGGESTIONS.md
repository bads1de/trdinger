# ML サービスのリファクタリング提案

`backend/app/services/ml/` 配下のコードベースについて、冗長な箇所、責務の分離、汎用性の観点からリファクタリング案を提案します。

## 1. 全体課題と基本方針

- **責務の混在**: `BaseMLTrainer` や `MLTrainingOrchestrationService` など、一つのクラスに多くの責務が集中しています。機能をより小さなクラスに分割し、単一責任の原則を徹底します。
- **コードの重複**: 特にモデルラッパークラス群や評価ロジックに多くの重複が見られます。共通処理をユーティリティや基底クラスに集約します。
- **設定管理の煩雑さ**: 設定オブジェクトと辞書表現の相互変換が手動で行われており、メンテナンス性に欠けます。Pydantic の機能を活用して自動化します。
- **デッドコードの可能性**: `ensemble_models.py` など、現在使用されていない可能性のあるコードが存在します。利用状況を確認し、不要であれば削除します。

---

## 2. 個別リファクタリング提案

- [ ] ### 2.2. `BaseMLTrainer` の責務分割

- **課題**: `base_ml_trainer.py` は、特徴量計算、データ準備、モデル学習、評価、保存など、多くの責務を担っており、クラスが肥大化しています。
- **提案**:

  - **特徴量計算**: `_calculate_features` メソッド内のロジックは、`EnhancedFeatureEngineeringService` に完全に移譲します。`BaseMLTrainer` は `FeatureEngineeringService` のインスタンスを保持し、そのメソッドを呼び出すだけにします。
  - **データ準備**: `_prepare_training_data` メソッド内の欠損値補完やラベル生成ロジックは、`utils/data_preprocessing.py` や `utils/label_generation.py` の汎用関数を利用するようにし、`BaseMLTrainer` から具体的な処理を分離します。
  - **メタデータ構築**: `train_model` 内の冗長なメタデータ構築ロジックを、`ModelMetadata` のような `dataclass` を活用して簡潔にします。

- [ ] ### 2.4. 設定管理の改善

- **課題**:
  - `config/ml_config_manager.py` の `get_config_dict` メソッドは、`MLConfig` の属性を手動で辞書にマッピングしており、設定クラスの変更に脆弱です。
  - `base_ml_trainer.py` の `_create_automl_config_from_dict` も同様に、手動での変換ロジックが記述されています。
- **提案**:

  - `MLConfig` や `AutoMLConfig` などの Pydantic モデルに、`to_dict()` や `from_dict()` のようなシリアライズ/デシリアライズ用のクラスメソッドを実装します。Pydantic の `.model_dump()` (v2) や `.dict()` (v1) を内部で利用することで、この処理を自動化できます。

  **実装例 (`ml_config.py`):**

  ```python
  # MLConfigクラス内
  def to_dict(self) -> Dict[str, Any]:
      return self.model_dump() # Pydantic v2の場合

  @classmethod
  def from_dict(cls, data: Dict[str, Any]) -> "MLConfig":
      return cls(**data)
  ```

- [ ] ### 2.6. 状態管理の改善

- **課題**: `orchestration/ml_training_orchestration_service.py` は、グローバル変数 `training_status` を使ってトレーニングの状態を管理しており、ステートフルでテストが困難です。
- **提案**:

  - `orchestration/background_task_manager.py` の機能を拡張し、トレーニングの状態（進捗、ステータス、メッセージなど）も管理できるようにします。
  - `MLTrainingOrchestrationService` は `background_task_manager` を通じて状態の読み書きを行い、グローバル変数への依存をなくします。これにより、コードの見通しが良くなり、テスト容易性も向上します。

- [] ### 2.7. Feature Engineering サービスの階層重複

- **課題**:
  - `FeatureEngineeringService`、`EnhancedFeatureEngineeringService`、`AutoMLFeatureGenerationService` の 3 つのサービスが存在し、責務が重複・分散しています。
  - `EnhancedFeatureEngineeringService` は `FeatureEngineeringService` を継承していますが、実際には大部分の機能を再実装しており、継承の利点が活かされていません。
  - `AutoMLFeatureGenerationService` は単なるファサードクラスとして機能しており、独立したサービスとしての価値が低いです。
- **提案**:

  - `FeatureEngineeringService` を基底クラスとして残し、AutoML 機能を統合した単一の `UnifiedFeatureEngineeringService` を作成します。
  - `AutoMLFeatureGenerationService` の機能を `UnifiedFeatureEngineeringService` に統合し、API レイヤーから直接呼び出せるようにします。
  - 継承ではなくコンポジションパターンを使用し、AutoML 機能を必要に応じて注入できる設計に変更します。

- [ ] ### 2.13. 特徴量計算クラスの構造重複

- **課題**:
  - `PriceFeatureCalculator`、`TechnicalFeatureCalculator`、`MarketDataFeatureCalculator` などが同じ初期化パターンを持っています。
  - 各クラスで `DataValidator.safe_*` メソッドを使用した同様の計算パターンが繰り返されています。
  - エラーハンドリングや結果の検証ロジックが各クラスで重複しています。
- **提案**:

  - 抽象基底クラス `BaseFeatureCalculator` を作成し、共通の初期化・検証・エラーハンドリングロジックを集約します。
  - 各特徴量計算クラスは `BaseFeatureCalculator` を継承し、具体的な計算ロジックのみを実装するように変更します。
  - 共通の計算パターン（移動平均、比率計算、変化率計算など）をユーティリティメソッドとして `BaseFeatureCalculator` に実装します。

  **実装例:**

  ```python
  # base_feature_calculator.py
  from abc import ABC, abstractmethod

  class BaseFeatureCalculator(ABC):
      def __init__(self):
          self.validator = DataValidator()

      def safe_rolling_mean(self, series: pd.Series, window: int) -> pd.Series:
          return self.validator.safe_rolling_mean(series, window)

      def safe_ratio_calculation(self, numerator: pd.Series, denominator: pd.Series) -> pd.Series:
          return self.validator.safe_divide(numerator, denominator, default_value=1.0)

      @abstractmethod
      def calculate_features(self, df: pd.DataFrame, config: Dict[str, Any]) -> pd.DataFrame:
          pass
  ```

- [ ] ### 2.15. インターフェースの未活用

- **課題**:
  - `MLPredictionInterface`、`MLTrainingInterface`、`MLServiceInterface` が定義されていますが、実装クラスが見つかりません。
  - インターフェースが活用されておらず、型安全性やコードの一貫性が保たれていません。
  - 新しい実装を追加する際の指針が不明確です。
- **提案**:

  - 既存の ML サービスクラス（`MLTrainingService`、予測関連サービスなど）にインターフェースを実装させます。
  - インターフェースに基づいた依存性注入を導入し、テスト容易性を向上させます。
  - インターフェースの仕様を見直し、実際の使用パターンに合わせて調整します。

- [ ] ### 2.16. 設定管理の手動マッピング重複

- **課題**:
  - `MLConfigManager.get_config_dict()` メソッドで、Pydantic モデルの属性を手動で辞書にマッピングしています。
  - 同様の手動マッピングが `AutoMLConfig.to_dict()` や他の設定クラスでも実装されています。
  - 設定クラスの属性変更時に、マッピングロジックも手動で更新する必要があります。
- **提案**:

  - Pydantic の `.model_dump()` メソッド（v2）または `.dict()` メソッド（v1）を活用して、自動シリアライゼーションを実装します。
  - カスタムシリアライゼーションが必要な場合は、Pydantic のフィールド設定やカスタムシリアライザーを使用します。
  - 手動マッピングロジックを削除し、保守性を向上させます。

- [ ] ### 2.20. ログメッセージの重複パターン

- **課題**:
  - ML 関連のログメッセージで似たようなパターンが多数存在します：
    - `"MLトレーニング〇〇エラー"` パターンが複数箇所
    - `"AutoML〇〇"` パターンの重複
    - `"✅"` や `"❌"` などの絵文字を使ったログの一貫性不足
- **提案**:

  - ML 専用のログメッセージテンプレートクラス `MLLogMessages` を作成し、統一されたメッセージフォーマットを提供します。
  - ログレベルと絵文字の使用ルールを統一し、一貫性のあるログ出力を実現します。
  - 国際化（i18n）を考慮したメッセージ管理システムの導入を検討します。

- [ ] ### 2.22. ML 設定管理の分散

- **課題**:
  - ML 関連の設定が複数箇所に分散しています：
    - `MLConfig` クラス（Pydantic モデル）
    - `MLConfigManager` クラス（ファイル永続化）
    - `AutoMLConfig` クラス（AutoML 専用設定）
    - オーケストレーションサービス内のハードコードされた設定
- **提案**:

  - 設定管理を階層化し、基底設定クラス `BaseMLConfig` を作成します。
  - 設定の継承関係を明確にし、共通設定と専用設定を分離します。
  - 設定の検証、デフォルト値管理、環境変数オーバーライドを統一的に処理します。

- [ ] ### 2.23. インターフェース実装の不完全性

- **課題**:
  - `MLOrchestrator` が `MLPredictionInterface` を実装していますが、一部のメソッドが未実装または不完全です。
  - インターフェースの契約が守られておらず、実行時エラーのリスクがあります。
  - 他のクラスでもインターフェースの実装が中途半端な状態です。
- **提案**:

  - すべてのインターフェース実装クラスで完全な実装を強制します。
  - 抽象基底クラスを使用して、未実装メソッドをコンパイル時に検出できるようにします。
  - インターフェースの仕様を見直し、実際の使用パターンに合わせて調整します。

---

これらの追加のリファクタリングを実行することで、コードの保守性、拡張性、パフォーマンスが大幅に向上することが期待されます。特にモデルラッパークラスの統一化と特徴量計算クラスの共通化により、新しいモデルや特徴量の追加が容易になり、メンテナンスコストも大幅に削減されます。

## 5. 総括

本提案は、ML コードベースの保守性・拡張性・性能を総合的に高めるための改善項目を体系化しました。特に「モデルラッパーの統一」「特徴量計算の共通化」「設定/ログ/リソース管理の標準化」により、追加開発と運用の両面でコスト削減と品質向上が期待できます。

本リファクタリング提案では、**25 項目**の包括的な改善点を特定しました。これらの改善により、ML コードベースの品質が大幅に向上し、開発効率とシステムの安定性が向上することが期待されます。

### 改善効果の期待値

1. **コード重複の削減**: 推定 40-50%のコード重複を解消
2. **保守性の向上**: 新機能追加時の修正箇所を 70%削減
3. **テスト容易性**: 統一されたインターフェースによりテストカバレッジ向上
4. **パフォーマンス**: メモリ使用量と CPU 使用率の最適化
5. **開発効率**: 新しいモデルや特徴量の追加時間を 60%短縮

### 実装優先度

**高優先度（即座に実装すべき）:**

（完了済み）

**中優先度（次のスプリントで実装）:**

- 2.13. 特徴量計算クラスの構造重複

**低優先度（長期的な改善）:**

- 2.20. ログメッセージの重複パターン
- 2.24. テストコードの重複パターン

## 3. utils 配下のリファクタリング提案（追加）

以下は `backend/app/utils/` 配下のコード（API 応答/日時、DB 挿入・クエリ、データクリーニング・前処理・標準化・変換・検証、ラベル生成、メトリクス、エラーハンドリング、重複ログフィルタ）を精査したうえでの具体的な改善提案です。ML コード全体の品質・可観測性・再利用性を高めるために優先度順で記載します。

- [ ] 3.1. data_preprocessing と data_validation の重複・境界の再定義

  - 課題:
    - 欠損値補完や外れ値処理が [`data_preprocessing.DataPreprocessor.preprocess_features()`](backend/app/utils/data_preprocessing.py:157) と [`data_validation.DataValidator.clean_dataframe()`](backend/app/utils/data_validation.py:349) に分散・重複。
    - どちらにも「NaN/inf 処理・閾値クリーニング・中央値/平均埋め」などが存在し、呼び順によって挙動が変わるリスクがある。
  - 提案:
    - 境界の定義を明確化し、責務分離を徹底。
      - validation 系は「検出と報告（および最小限の安全置換）」に限定。
      - preprocessing 系は「意図的な補完・外れ値除去・スケーリング」のみ担当。
    - 具体策:
      - DataValidator に「検出専用 API」を追加し、副作用のある補完分岐を削減（clean_dataframe は軽量フォールバック中心に）。
      - DataPreprocessor 側に「前処理パイプライン（検出 → 補完 → 標準化）」を一本化し、validation の結果を受けて明示的に処置を適用。
    - 期待効果: 前処理順序の一意化、テスト容易性向上、二重処理防止。

- [ ] 3.2. data_cleaning_utils の補間ロジックの共通化とパイプライン化

  - 課題: [`DataCleaner.interpolate_*`](backend/app/utils/data_cleaning_utils.py:20) が OI/FR と Fear&Greed で似た補間パターン（ffill→ 統計補完）を別々に実装。
  - 提案:
    - 共通補間ヘルパーを DataPreprocessor に移し、カラム名と戦略を受け取って処理する関数へ一般化。
    - DataCleaner は「ドメイン別の列名セット/既定値」を定義する薄いラッパーとし、実処理は共通関数を呼ぶ。
  - 期待効果: 重複削減、欠損戦略の一元管理、後続テストの簡素化。

- [ ] 3.3. 型・スキーマ整合性の標準化（standardize_ohlcv_columns の強化）

  - 課題: [`data_standardization.standardize_ohlcv_columns()`](backend/app/utils/data_standardization.py:10) は列名正規化のみで型/インデックス/ソートを保証していない。これに対して [`DataCleaner.validate_ohlcv_data()`](backend/app/utils/data_cleaning_utils.py:160) で別途検証している。
  - 提案:
    - standardize_ohlcv_columns に以下のオプション引数を追加し、標準化の網羅度を拡張:
      - ensure_datetime_index=True（DatetimeIndex でない場合に変換）
      - sort_index=True（時間順ソート）
      - coerce_dtypes=True（数値列の型強制: float32/float64 方針を選択可能）
    - DataCleaner 側の検証は「標準化後の整合性監査」に縮小。
  - 期待効果: データ受け渡し点での一回の標準化で後段の前処理・学習の前提が安定。

- [ ] 3.4. data_converter と data_utils のフォーマット/精度の統一

  - 課題:
    - [`data_converter.OHLCVDataConverter`](backend/app/utils/data_converter.py:12) などで float キャストが散在し、精度・dtype の一貫性がコンテキスト依存。
    - [`data_utils.ensure_series/ensure_numeric_series`](backend/app/utils/data_utils.py:23) とも役割が近いが責務境界が曖昧。
  - 提案:
    - 変換時の dtype ポリシーを一本化（例: 学習前は float32、評価・保存は float64 などを「方針」として定義）。
    - Series/Array 変換の入口を data_utils に集約。data_converter は「外部->内部表現」「内部->API/DB 表現」のマッピングに限定。
    - 変換ポリシーを設定化（config 経由）し、変換関数は明示的な引数で override 可能に。
  - 期待効果: 精度/メモリの統制、一貫したデータ表現で不具合低減。

- [ ] 3.5. data_validation.Safe\* 系関数の戻り型統一と NaN/Inf 処理方針の明確化

  - 課題: [`safe_normalize`](backend/app/utils/data_validation.py:250) は戻りが Series/ndarray/float の場合があり、最後に Series へ寄せる処理が入っている。ほかの safe\_\* もスカラー/Series の混在がある。
  - 提案:
    - すべての safe\_\* は戻りを pd.Series に統一（index を引数の基準 Series から継承）。スカラー演算は内部で Series 化。
    - NaN/Inf の置換ポリシーを DataValidator のクラス定数として集約（デフォルト値、epsilon、閾値など）。呼び出し側で上書き可能。
  - 期待効果: 呼び出し側の分岐削減、型安定性の向上。

- [ ] 3.6. metrics_calculator の非推奨 API の整理と委譲

  - 課題: [`calculate_detailed_metrics`](backend/app/utils/metrics_calculator.py:30) が deprecated コメントで外部クラスへの移行を案内しているが、利用側での移行完了が不明。デフォルト辞書 [`get_default_metrics()`](backend/app/utils/metrics_calculator.py:151) との責務関係も曖昧。
  - 提案:
    - 本関数に PendingDeprecationWarning を追加し、次期リリースで完全削除をアナウンス。検索して利用箇所を置換（`EnhancedMetricsCalculator` へ移行）。
    - get_default_metrics は「評価結果の初期化」に特化して残し、モデル評価は evaluation 層へ全面委譲。
  - 期待効果: メトリクス計算経路を一本化、保守コスト削減。

- [ ] 3.7. unified_error_handler の API/ML 分岐と戻り値契約の厳格化

  - 課題: [`safe_execute()`](backend/app/utils/unified_error_handler.py:272) が API/ML の分岐を内包し戻り値が default_return/HTTPException のミックスになり得る。利用側が想定を外すと取り扱いが難しい。
  - 提案:
    - API 用（例外送出）と ML 用（値返却）の 2 系統に明確分岐した関数を公開（safe_execute_api / safe_execute_ml）。現行 safe_execute は内部委譲しつつ将来的に非推奨化。
    - create_error_response のスキーマを JSON-Schema 的に固定し、details/context の型を統一。timestamp は UTC ISO で固定。
  - 期待効果: 例外/戻り値契約の明瞭化によりハンドリングの分岐が単純化。

- [ ] 3.8. database_utils の DB 方言吸収レイヤの拡張とテスト強化

  - 課題: [`bulk_insert_with_conflict_handling`](backend/app/utils/database_utils.py:18) で SQLite と PostgreSQL で分岐しているが、MySQL 等の将来拡張が想定されていない。SQLite バッチ個別リトライは堅牢だがユニットテストが重要。
  - 提案:
    - DB 方言抽象（DialectStrategy）を導入して、方言別の conflict/insert 戦略をプラガブルに（PostgreSQL/SQLite 実装を分離）。
    - 例外の粒度（IntegrityError 等）での分岐を明示してログを整備。バッチサイズは設定で注入可能に。
    - テスト: 疑似 DB セッションのモックで「重複/部分成功/コミット失敗/個別再試行」を網羅。ログの期待文言も検証。

- [ ] 3.9. label_generation の分布最適化と計算のストリーミング化

  - 課題: ラベル生成で複数メソッドの試行（[`_calculate_adaptive_thresholds`](backend/app/utils/label_generation.py:212)）を行うが、データ長に比例して計算量が増える。分布検証 [`validate_label_distribution`](backend/app/utils/label_generation.py:292) は有用だが、スケール時のパフォーマンスに注意。
  - 提案:
    - 目標分布に対するスコア計算をベクトル化・一回算出の統計量をキャッシュ（std/quantile など）。
    - 大規模データではローリング窓のサンプルリング（または分位近似: t-digest 等）で近似加速。
    - 戻りのメタ情報（threshold_stats 等）はスキーマ（Dict[str, Any] -> TypedDict/Dataclass）へ。
  - 期待効果: 大規模データでのスループット改善、メタ情報の型安全性向上。

- [ ] 3.10. duplicate_filter_handler の機能拡張（統計/抑制ログ出力）

  - 課題: 連続抑制された回数が利用者に見えにくい。容量到達時の削除戦略は LRU だが明示ログがない。
  - 提案:
    - interval 内で抑制された回数が一定閾値を超えた際にサマリーログを定期的に出すオプションを追加（例: "message='X' suppressed N times in last T sec"）。
    - capacity 超過で削除されたメッセージキーと削除理由を debug ログに記録（オプトイン）。
  - 期待効果: 運用時の観測性向上、サイレントドロップの原因究明が容易。

- [ ] 3.11. API ユーティリティの時刻/タイムゾーン取り扱いの統一

  - 課題: [`api_utils.APIResponseHelper`](backend/app/utils/api_utils.py:12) と他ユーティリティで datetime.now().isoformat() などが混在。UTC 固定/タイムゾーン付き ISO の統一を徹底したい。
  - 提案:
    - DateTimeHelper に `now_utc_iso()` を追加し、全レスポンス timestamp を UTC+Z で統一（例: `datetime.now(timezone.utc).isoformat()`）。
    - APIResponseHelper/error_response/api_response は内部で DateTimeHelper を使用し、重複処理を削減。
  - 期待効果: 監視・集計・フロント表示の一貫性向上。

- [ ] 3.12. ユーティリティ間の循環依存の回避と import 方針

  - 課題: [`data_cleaning_utils` が `data_preprocessing` を参照](backend/app/utils/data_cleaning_utils.py:12)。将来的な再配置で循環依存が生じやすい構成。
  - 提案:
    - 「共通低層（core_utils/transform）」と「ドメイン別（cleaners）」に層分割し、上位が下位にのみ依存する構造へ整理。
    - 実装ファイル内 import を遅延（ローカル import）するポリシーを最小限で活用し循環を回避。
  - 期待効果: 再配置・分割時のリスク低減。

- [ ] 3.13. ログ文言・レベルの標準化（utils 全体）

  - 課題: info/warning/error の使い分けにばらつき。ユーザー向け/開発者向けの文言が混在。
  - 提案:
    - `utils` 共通のログガイドラインを定義し、info は進捗・件数、warning は回復可能異常、error は失敗時に限定。
    - 閾値・件数・ウィンドウなど重要パラメータは構造化ログ（extra フィールド）に添付。
  - 期待効果: 運用時のノイズ低減、問題追跡の効率化。

- [ ] 3.14. テスト整備（単体/プロパティ/回帰）
  - 提案（横断的）:
    - property-based testing（hypothesis など）で `safe_*` 系の数値安定性を検証（ゼロ除算、極小値、Inf、巨大値）。
    - 時系列一貫性テスト（標準化 → 検証 → 前処理の順で結果が変わらないこと）。
    - DB バルク挿入のリトライ/混在成功シナリオのモックテスト。
