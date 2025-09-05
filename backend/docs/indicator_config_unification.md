# テクニカル指標設定の一元化計画（設計提案）

作成日: 2025-09-05
対象ディレクトリ: `backend/`

## 要約

- 現状、テクニカル指標の設定・登録・戦略向け特性が、複数のファイルとYAMLに分散しています。
- 単一の指標を追加・修正するたびに、最低でも 2〜4 箇所へ変更が波及しやすく、重複と不整合の温床になっています。
- 目標は「Single Source of Truth」を確立し、追加・修正を1ファイル（1エントリ）に集約することです。
- 具体的には、指標のメタ情報・pandas-ta連携・パラメータ仕様・カテゴリ・出力定義を単一スキーマに統合し、そこから
  - レジストリ初期化
  - 自動ストラテジー用の一覧・カテゴリ別抽出
  - pandas-ta呼び出し
  をすべて生成/参照します。
- YAML は「データソース依存の閾値・条件（戦略ロジック向けの可変設定）」に限定し、技術的メタ（型・必須データ・出力構造等）は単一スキーマに集約します。

---

## 現状調査の要点

- 指標詳細・登録
  - `backend/app/services/indicators/config/indicator_definitions.py`
    - 多数の `IndicatorConfig` をハードコードで登録。
    - pandas-ta 連携定義 `PANDAS_TA_CONFIG` と `POSITIONAL_DATA_FUNCTIONS` も併記（約L1634〜）。
    - `initialize_all_indicators()` がモジュール末尾で呼ばれ、インポート時副作用でレジストリ登録が走る。
- 設定クラス
  - `backend/app/services/indicators/config/indicator_config.py`
    - `IndicatorConfig`, `ParameterConfig`, `IndicatorConfigRegistry` を定義。
    - `IndicatorConfig.__post_init__()` が `PANDAS_TA_CONFIG` を参照して `parameters` を構築（隠れ依存）。
    - `normalize_params()` で `length/period` 変換や特殊マッピングを実装。
- パラメータ管理
  - `backend/app/services/indicators/parameter_manager.py`
    - `PARAMETER_MAPPINGS` に別系統のパラメータマッピング辞書を保持。
    - `map_parameters()` と `validate/generate` 系ロジックが存在。
- 戦略系の定数・一覧
  - `backend/app/services/auto_strategy/config/constants.py`
    - カテゴリ別の大量の指標名リストやキュレート集合、IDマッピング生成などを保持。
    - YAML ベースの特性とのマージも行う（`YamlIndicatorUtils` を通じて）。
- YAML（戦略向け可変設定）
  - `backend/app/services/auto_strategy/config/technical_indicators_config.yaml`
    - 指標ごとの `type`（カテゴリ）や `scale_type` を含む場合が多数。
    - 閾値・条件テンプレートなど「戦略ロジック依存の値」が主だが、技術的メタと重複している箇所がある。

---

## 問題点の整理

- 重複・分散
  - カテゴリ・スケール・出力定義などが `indicator_definitions.py` と YAML に重複。
  - パラメータのエイリアス/マッピングが `IndicatorConfig.normalize_params()` と `parameter_manager.py` の `PARAMETER_MAPPINGS` の両方に存在。
  - `PANDAS_TA_CONFIG` と各 `IndicatorConfig.param_map` で似通った変換が二重定義。
- 隠れ依存と副作用
  - `IndicatorConfig.__post_init__()` がグローバル `PANDAS_TA_CONFIG` に依存し、初期化順序に敏感。
  - `config/__init__.py` で `initialize_all_indicators()` を実行しており、インポートで副作用が発生。
- メンテナンス性低下
  - 新規指標の追加に、複数ファイルの同期更新が必要。
  - 戦略側（YAML）が技術的メタ（カテゴリ・スケール）を重複保持し、改修時不整合が発生しやすい。

---

## 達成目標（品質基準）

- 追加・修正点を原則「1箇所」に集約（Single Source of Truth）。
- パラメータ名は内部的に標準化（例: `length` を正規形）、外部API（pandas-ta/アダプタ）向けに自動マッピング。
- 戦略YAMLから技術メタを排除し、閾値・条件など運用チューニング項目のみに限定。
- インポート時副作用を撤廃し、明示的な初期化シーケンスへ。
- 既存呼び出しとの互換層を提供（段階的移行）。

---

## 解決方針（アーキテクチャ）

1. 単一スキーマの導入（指標仕様の宣言的定義）
   - 格納場所候補: `backend/app/services/indicators/config/indicators.schema.yaml` または `indicators.schema.json`。
   - 含める情報:
     - 識別子（`name`, `aliases`）
     - カテゴリ（`category`）
     - 結果タイプ（`result_type`）
     - スケール（`scale_type`）
     - 必須データ（`required_data`）
     - 出力名（`output_names` と `default_output`、複数結果用）
     - パラメータ仕様（標準名・型・既定値・範囲・説明）
     - pandas-ta 連携（`function`, `param_aliases`, `data_column(s)`/`multi_column`）
     - アダプタ関数名（存在する場合のみ）
   - これを唯一の「真実のソース」にする。

2. 自動レジストリ生成
   - 起動時にスキーマを読み込み、`IndicatorConfigRegistry` を構築。
   - `IndicatorConfig` はスキーマの1エントリからインスタンス化し、副作用なし。
   - `PANDAS_TA_CONFIG` と `param_map`、`PARAMETER_MAPPINGS` はスキーマの `param_aliases` に統合して廃止（互換層で段階的に移行）。

3. パラメータ正規化の一本化
   - 正規形（例: `length`）をスキーマで定義。
   - `normalize_params()` はスキーマ由来の `param_aliases` を参照する汎用実装のみに簡素化。

4. 戦略YAMLの役割を限定
   - YAML（例: `technical_indicators_config.yaml`）から `type`/`scale_type` のような技術メタを段階的に削除。
   - 閾値・条件テンプレートなど、戦略チューニングに関わる部分のみ維持。
   - もし読みやすさのために `type` 等を残すなら、レジストリと突合・不整合検知のCIチェックを導入。

5. 明示的初期化（副作用排除）
   - `config/__init__.py` の即時初期化を廃止し、`indicator_orchestrator.py` 等のエントリで初期化を明示呼び出し。
   - 循環依存や順序依存のバグを回避。

6. 互換レイヤ
   - 既存の `IndicatorConfig.__post_init__()` は無効化/空実装化し、スキーマ経由のパラメータ生成に置換。
   - 旧API（`normalize_params` の特殊ケース、`PARAMETER_MAPPINGS` など）を内部的に新スキーマへ委譲。
   - ログで非推奨警告を出し、段階的に削除予定を明記。

---

## 設計詳細

- 標準パラメータ名
  - 内部の正規形は `length` を主とし、`period` 等は `param_aliases` で吸収。
  - 一貫性のない指標（例: `fast`, `slow`, `signal` 等）は、そのまま標準名として採用し、エイリアスでTA-Lib/pandas-ta差を吸収。

- データカラム定義
  - `single`/`multi` の別と、必要なカラム名をスキーマで明記（例: `data_column: "Close"` or `data_columns: ["High", "Low", ...]`）。
  - 大文字/小文字解決は既存の `TechnicalIndicatorService._resolve_column_name()` を流用しつつ、スキーマの宣言から生成。

- 出力構造
  - `result_type` が `single` の場合は単一配列/Series を返却、`complex`/`multiple` では `output_names` 長と一致する形を保証。

- カテゴリとキュレート集合
  - カテゴリはスキーマで一元管理し、`auto_strategy/config/constants.py` 側のリストはレジストリから生成。
  - 「キュレート集合」はスキーマの `tags` などでフラグ付けし、`CURATED_TECHNICAL_INDICATORS` をビルド時に自動組立（手動維持も可）。

- pandas-ta 呼び出し
  - 現行の `PANDAS_TA_CONFIG` をスキーマの `integration: pandas_ta` セクションへ吸収。
  - `TechnicalIndicatorService._calculate_with_pandas_ta()` はスキーマ由来の情報のみを使って解決。

---

## データフロー（新）

1. 起動時にスキーマを読み込み、`IndicatorConfigRegistry` を構築（明示呼び出し）。
2. 戦略層はレジストリからサポート指標一覧・カテゴリ別一覧を取得。
3. 指標計算時は、レジストリ→スキーマ情報を元に pandas-ta or アダプタで計算。
4. YAML は「閾値・条件テンプレート」だけを参照。技術メタはレジストリから取得/検証。

---

## マイグレーション手順（段階的移行）

1. スキーマ定義の土台を作成
   - 既存の `PANDAS_TA_CONFIG` と `IndicatorConfig` 群をサンプルとして、10〜20指標分をスキーマ化して PoC。
   - `IndicatorConfig` 生成ローダを実装（副作用なし）。

2. レジストリ初期化の切替
   - `config/__init__.py` の自動初期化を削除し、`indicator_orchestrator.py` 側で明示初期化。
   - 既存参照箇所で初期化が一度だけ起動するよう整理。

3. パラメータ正規化/マッピングの一本化
   - `IndicatorConfig.normalize_params()` をスキーマ起点の汎用ロジックに差替。
   - `parameter_manager.py` の `PARAMETER_MAPPINGS` をスキーマへ移管し、互換レイヤ経由で呼び出し。

4. YAML のスリム化
   - `technical_indicators_config.yaml` から `type`/`scale_type` を徐々に削減（または検証専用フィールド化）。
   - CI でレジストリ定義との不整合を検出し、差分アラートを出す。

5. 全指標のスキーマ移行
   - 重要/頻出指標から順次移行、段階ごとにレグレッションテストを実行。
   - 完了後、`PANDAS_TA_CONFIG`/`PARAMETER_MAPPINGS`/ハードコード登録を撤去。

6. ドキュメント整備
   - 「新規指標を追加する手順」を1ページに集約（追加はスキーマ1行と任意のアダプタ関数のみ）。

---

## 影響範囲

- 生成系/呼び出し系
  - `TechnicalIndicatorService`（`indicator_orchestrator.py`）のpandas-ta連携部は、参照元を `PANDAS_TA_CONFIG` からスキーマへ変更。
- 参照系
  - `auto_strategy/config/constants.py` の一覧/カテゴリ派生ロジックは、レジストリからの生成に切替（静的リストは減少）。
- テスト
  - 主要インジケータで入出力のスナップショット/境界値テストを更新。

---

## リスクと対策

- スキーマと実装の乖離
  - 対策: スキーマ→レジストリ生成の検証テストと、pandas-ta 呼び出しのスモークテストをCIに追加。
- パフォーマンス（起動時ロード）
  - 対策: スキーマは1回読み込み・キャッシュ。必要なら生成済みPythonモジュールをCIで自動生成。
- 互換性
  - 対策: 旧APIは互換レイヤで維持し、非推奨ログを出す。移行完了後にクリーニング。
- 循環依存
  - 対策: 初期化タイミングを統一し、副作用実行を禁止。エントリポイントから初期化を明示呼び出し。

---

## 実装計画（作業WBS）

- フェーズ1（設計/PoC）
  - 最小セットのスキーマ定義（10〜20指標）。
  - ローダ/バリデータ/レジストリ生成のPoC。
  - `TechnicalIndicatorService` の読み替えPoC。
- フェーズ2（切替基盤）
  - 明示初期化化＆副作用除去。
  - パラメータ正規化の一本化。
  - CIルール（不整合検出）。
- フェーズ3（全面移行）
  - 全指標のスキーマ移行。
  - 重複定義の削除（`PANDAS_TA_CONFIG`、`PARAMETER_MAPPINGS`、静的リスト）。
  - 戦略YAMLのスリム化。
- フェーズ4（仕上げ）
  - ドキュメント/チーム教育。
  - デグレ検知追加、保守性向上タスク。

---

## 期待される効果

- 新規指標の追加/修正が、原則「スキーマ1エントリ」＋「（必要時）アダプタ関数1箇所」で完結。
- 重複定義が排除され、不整合・回帰のリスクが低下。
- YAML は運用チューニング専用となり、戦略作成の自由度を維持しながら技術メタの整合性を確保。

---

## 付記（過去リファクタ方針との整合）

- 既存メモリの方針（`pd.Series` のインライン化、統一エラーハンドリング等）と同様に、
  「複雑化・二重化した層を排除し、宣言的な単一ソースから動的生成する」方向性で統一します。
- 本計画は、無駄なラッパ/変換の重複を削ぎ、可観測性（ログ/CI検証）を強化するものです。

## 影響を受けるファイル（詳細）

- `backend/app/services/indicators/config/indicator_definitions.py`
  - `PANDAS_TA_CONFIG`（L1634〜）、`POSITIONAL_DATA_FUNCTIONS`（L1831〜）、`initialize_all_indicators()` 実行（L1888）
  - 多数の `indicator_registry.register(...)` によるハードコード登録

- `backend/app/services/indicators/config/__init__.py`
  - `initialize_all_indicators` の自動実行によるインポート時副作用（L14–L17）

- `backend/app/services/indicators/indicator_orchestrator.py`
  - `PANDAS_TA_CONFIG` 参照（L79, L136, L316 付近）
  - `POSITIONAL_DATA_FUNCTIONS` 参照（L510 付近）

- `backend/app/services/indicators/data_validation.py`
  - `PANDAS_TA_CONFIG` 参照（L14, L82, L194, L258 付近）

- `backend/app/services/indicators/config/indicator_config.py`
  - `__post_init__()` 内で `PANDAS_TA_CONFIG` を参照（L104–L110）
  - `get_parameter_ranges()` / `normalize_params()` で `PANDAS_TA_CONFIG` に依存

- `backend/app/services/indicators/parameter_manager.py`
  - `PARAMETER_MAPPINGS`（L112〜）と `IndicatorParameterManager.map_parameters()` による別系統マッピング

- 戦略側（Auto Strategy）
  - `backend/app/services/auto_strategy/config/constants.py`
    - 静的リスト群（`VOLUME_INDICATORS`/`MOMENTUM_INDICATORS`/`TREND_INDICATORS`/`VOLATILITY_INDICATORS` 等）
    - `VALID_INDICATOR_TYPES` の結合ロジック、`CURATED_TECHNICAL_INDICATORS`、`INDICATOR_CHARACTERISTICS` の YAML マージ
  - `backend/app/services/auto_strategy/utils/common_utils.py`
    - `YamlIndicatorUtils` による `technical_indicators_config.yaml` のロード/適用
  - `backend/app/services/auto_strategy/generators/condition_generator.py`
    - `YamlIndicatorUtils` ベースのしきい値/条件生成
  - `backend/app/services/auto_strategy/generators/random_gene_generator.py`
    - `VALID_INDICATOR_TYPES` / `CURATED_TECHNICAL_INDICATORS` 参照
  - `backend/app/services/auto_strategy/models/strategy_models.py`
    - `VALID_INDICATOR_TYPES` 参照と検証ロジック
  - `backend/app/services/auto_strategy/config/auto_strategy_config.py`
    - `VALID_INDICATOR_TYPES` をデフォルトに使用

- スクリプト
  - `backend/scripts/comprehensive_test.py`
    - `indicator_registry` に依存（移行後も動作確認要）


## 削除候補と削除タイミング（段階的）

- フェーズ2以降で削除/無効化できるもの（準備が整い次第）
  - `backend/app/services/indicators/config/__init__.py` の自動初期化行
    - 削除条件: 明示初期化へ切替後（アプリのエントリポイントでレジストリを初期化）
  - `IndicatorConfig.__post_init__()` の `PANDAS_TA_CONFIG` 依存処理
    - 削除条件: スキーマ由来の `parameters` 構築へ完全移行後

- フェーズ3で削除するもの（スキーマ移行完了後）
  - `PANDAS_TA_CONFIG` / `POSITIONAL_DATA_FUNCTIONS`（`indicator_definitions.py`）
    - 削除条件: `indicator_orchestrator.py` / `data_validation.py` がスキーマ参照に切替完了
  - `PARAMETER_MAPPINGS` と `IndicatorParameterManager.map_parameters()` のマッピング系
    - 削除条件: パラメータ正規化/マッピングがスキーマに一元化し、呼び出し側を置換完了
  - `auto_strategy/config/constants.py` の巨大な静的リスト群と `VALID_INDICATOR_TYPES`
    - 削除/縮退条件: レジストリからカテゴリ別・全体の一覧を取得する実装へ切替完了

- 削除はしないがスリム化するもの
  - `technical_indicators_config.yaml`
    - 技術メタ（`type`/`scale_type`）は段階的に削除/検証専用に縮退。閾値・条件テンプレート中心に維持
  - `CURATED_TECHNICAL_INDICATORS`
    - 残す場合はスキーマの `tags` 等で自動抽出に置換可能（完全自動化も選択肢）


## フェーズ別チェックリスト（実行管理）

### フェーズ1（PoC）

- [ ] 代表 10〜20 指標をスキーマ化（`name/category/result_type/scale/required_data/params/aliases` ほか）
- [ ] スキーマ ローダ/バリデータ/`IndicatorConfigRegistry` 生成 PoC
- [ ] `TechnicalIndicatorService` の pandas-ta 呼び出しをスキーマ参照に切替（PoC 範囲）
- [ ] スモークテスト（移行済み指標の入出力確認）

### フェーズ2（切替基盤）

- [ ] 明示初期化へ移行（`config/__init__.py` の自動初期化を停止）
- [ ] `normalize_params()` をスキーマ起点の汎用ロジックへ置換（旧特殊分岐は互換レイヤ経由）
- [ ] `data_validation.py` をスキーマ参照で必要データ長/NaN生成に対応
- [ ] CI で YAML とレジストリ（スキーマ）定義の不整合検出を追加

### フェーズ3（全面移行）

- [ ] 全指標のスキーマ移行完了
- [ ] `indicator_orchestrator.py` / `data_validation.py` の `PANDAS_TA_CONFIG` 依存を撤廃
- [ ] `indicator_definitions.py` から `PANDAS_TA_CONFIG` / `POSITIONAL_DATA_FUNCTIONS` を削除
- [ ] `IndicatorConfig.__post_init__()` の `PANDAS_TA_CONFIG` 依存ロジックを削除
- [ ] `parameter_manager.py` の `PARAMETER_MAPPINGS` と `map_parameters()` を削除（または薄い互換ラッパ化）
- [ ] 戦略側：`VALID_INDICATOR_TYPES` / 静的リスト依存をレジストリ参照に切替（`random_gene_generator.py`/`strategy_models.py`/`auto_strategy_config.py` 等）
- [ ] YAML をスリム化（`type`/`scale_type` の撤去または検証専用化）と `YamlIndicatorUtils` の更新

### フェーズ4（仕上げ/クリーンアップ）

- [ ] `auto_strategy/config/constants.py` の不要な静的リストを削除または動的生成へ置換
- [ ] `IndicatorConfig.param_map` など重複マッピングの最終削除（スキーマ一元化確認後）
- [ ] `backend/scripts/comprehensive_test.py` をスキーマ経路での検証へ更新
- [ ] ドキュメント最終更新＆非推奨 API の削除告知
