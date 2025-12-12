# Auto Strategy: 完了済みの改修内容 (Completed Improvements)

このドキュメントは、`AUTO_STRATEGY_ANALYSIS.md` で特定され、既に実装が完了した課題を移動・記録したものです。

## 1. 構造的な改善 (Structural Improvements)

### 1.1 単一タイムフレームの強制 (Single Timeframe Constraint)

> **実装完了日: 2024-12-09**
>
> - `IndicatorGene` に `timeframe: Optional[str]` フィールドを追加
> - `GAConfig` に MTF 設定（`enable_multi_timeframe`, `available_timeframes`, `mtf_indicator_probability`）を追加
> - `IndicatorGenerator` で MTF 指標生成対応
> - `MultiTimeframeDataProvider` クラスを新設（リサンプリング・キャッシュ機能）
> - `IndicatorCalculator` / `UniversalStrategy` で MTF データ使用対応
> - `GeneValidator` でタイムフレームバリデーション追加
> - `DictConverter` でシリアライズ/デシリアライズ対応

**解決された問題:**
従来のシステムは単一タイムフレームに依存していました。改修により、各指標が異なるタイムフレームを持つことが可能になり、MTF 戦略の探索が可能になりました。

### 1.2 ステートレス・ロジック (Stateless Logic)

> **実装完了日: 2025-12-09**
>
> - `StatefulCondition` モデルを新設
> - `StateTracker` クラスを新設（イベント記録・判定機能）
> - `ConditionEvaluator` に `evaluate_stateful_condition` / `check_and_record_trigger` メソッドを追加
> - `StrategyGene` に `stateful_conditions` フィールドを追加
> - `UniversalStrategy` に `StateTracker` を統合

**解決された問題:**
過去の状態やイベント順序を考慮できない問題を解決しました。「条件 A 発生後 N 本以内に条件 B」といったシーケンス制御が可能になりました。

### 1.3 階層構造の欠如 (Flat Structure)

> **実装完了日: 2025-12-09**
>
> - `ConditionGroup` に `operator` (AND/OR) を追加し、再帰構造をサポート
> - `ConditionEvaluator` で再帰的な評価ロジックを実装
> - `ComplexConditionsStrategy` で確率的に階層構造を生成するロジックを追加
> - `DictConverter` / `GeneValidator` / `GeneticOperators` を階層構造に対応修正

**解決された問題:**
フラットな OR 条件しか生成できなかった問題を解決し、任意の深さの論理ツリー（(A AND B) OR C など）を扱えるようになりました。

### 1.4 ListEncoder のアップグレード (ListEncoder Upgrade)

> **実装完了日: 2025-12-10**
>
> - `ListEncoder` をリファクタリングし、`timeframe` エンコードに対応
> - `ConditionGroup` の再帰的エンコード（Flattening）を追加
> - `StatefulCondition` のエンコード（Lookback/Cooldown/Direction 含む）に対応
> - 効果的なロング/ショート条件の分離エンコードへの考慮（`get_effective_long_conditions`利用）
> - 指標パラメータのエンコードロジックを改善

**解決された問題:**
ML モデルがタイムフレームの違い、複雑な条件構造、およびステートフルな条件ロジックを認識できなかった問題を解決しました。これにより、Hybrid GA がより高度な戦略のポテンシャルを正しく評価できるようになります。

## 2. 評価・最適化プロセスの改善 (Optimization Improvements)

### 2.1 フィットネス評価の統一不全

> **解決日: 2025-12-09**
>
> - `EvolutionRunner.run_evolution` メソッドにロジックを統合（単一・多目的共通化）
> - `ga_engine.py` の条件分岐を削除
> - HallOfFame / ParetoFront のポリモーフィックな扱いを実装

**解決された問題:**
単一目的と多目的最適化のコード重複を解消し、統一的に扱えるようになりました。

### 2.2 ヒューリスティックな定数（Magic Numbers）の使用

> **解決日: 2025-12-09**
>
> - `GAConfig` に `zero_trades_penalty` と `constraint_violation_penalty` パラメータを追加
> - `individual_evaluator.py` でこれらの設定値を参照するように変更

**解決された問題:**
固定値のペナルティによる進化の停滞リスクを排除し、設定可能にしました。

### 2.3 TPSL データスライスの動的化 (Dynamic TPSL Data Slicing)

> **解決日: 2025-12-10**
>
> - `UniversalStrategy.next()` 内の固定値「30」を `atr_period + 1` に基づく動的なスライスサイズに変更
> - True Range 計算に必要なバッファ (+1) を自動的に確保
> - テストケースを追加 (`test_universal_strategy_data_slicing.py`)

**解決された問題:**
GA が大きな `atr_period`（例: 50）を選択した場合に、ATR 計算に必要なデータが不足し、不正確な TPSL 値（NaN や 0）が算出される問題を修正しました。これにより、「長期間のボラティリティを参照する安定した戦略」の生成が可能になりました。

### 2.4 GAConfig のフィールド欠損 (Implicit Configuration)

> **解決日: 2025-12-11**
>
> - `GAConfig` クラス（`ga_runtime.py`）に `zero_trades_penalty` と `constraint_violation_penalty` フィールドを追加
> - `IndividualEvaluator` での暗黙的参照（`getattr`）を廃止し、明示的なプロパティ参照に変更
> - 設定の継承とデフォルト値の扱いを統一

**解決された問題:**
コード内でハードコードされたデフォルト値が優先され、`GASettings` や設定ファイルからのペナルティ値変更が反映されない問題を修正しました。これにより、意図した通りの進化圧力（ペナルティ）を個体に適用できるようになりました。

## 3. 設定と拡張性の改善 (Scalability & Configuration Improvements)

### 3.1 パラメータ範囲プリセット (Parameter Range Presets)

> **解決日: 2025-12-10**
>
> - `ParameterConfig` に `presets` フィールドと `get_range_for_preset()` メソッドを追加
> - `IndicatorParameterManager.generate_parameters()` に `preset` 引数を追加
> - `GAConfig` に `parameter_range_preset` フィールドを追加
> - テストケースを追加 (`test_parameter_range_presets.py`)

**解決された問題:**
パラメータの探索範囲がグローバルに固定されており、指標用途（短期/中期/長期）に応じた探索範囲の切り替えができなかった問題を解決しました。これにより、「短期トレンド用の RSI（期間 5-15）」と「長期トレンド用の RSI（期間 50-100）」を明確に区別して探索できるようになりました。

### 3.2 ハードコードされた制限

> **解決日: 2025-12-09**
>
> - `StrategyGene.MAX_INDICATORS` を `GAConfig.max_indicators` に移行
> - `ConditionGenerator` の条件数制限を `GAConfig` から取得するように変更

**解決された問題:**
探索空間を人為的に狭めていたハードコード制限を撤廃しました。

### 3.3 パラメータ依存関係検証 (Parameter Dependency Validation)

> **解決日: 2025-12-10**
>
> - `IndicatorConfig` に `parameter_constraints` フィールドと `validate_constraints()` メソッドを追加
> - `GeneValidator.validate_indicator_gene()` でパラメータ依存関係制約を検証するよう拡張
> - 以下の指標に `fast < slow` 制約を追加:
>   - **momentum.py**: MACD, PPO, APO, UO (fast < medium < slow), MASSI, AO, STC
>   - **volume.py**: ADOSC, KVO, PVO
>   - **trend.py**: AMAT
> - テストケースを追加 (`test_parameter_constraints.py`)

**解決された問題:**
パラメータ間の論理的整合性（例: MACD の `fast < slow`）をチェックするロジックが欠如していた問題を解決しました。これにより、無効なパラメータ設定（例: fast=50, slow=10）を持つ個体が生成される前に検出・排除され、GA の探索効率が向上しました。

### 3.4 プライスアクション制限の緩和 (Price Action Validation Relaxation)

> **解決日: 2025-12-10**
>
> - `GeneValidator._is_trivial_condition` の判定ロジックを緩和
> - 異なる価格データ同士の比較（`close > open`, `close > high` など）をプライスアクションとして許容
> - 同一価格データの比較（`close > close`）と無意味な閾値比較のみをトリビアルとして排除
> - テストケースを更新

**解決された問題:**
`Close > Open`（陽線判定）や `Close > High[1]`（前日高値ブレイク）などの単純なローソク足比較が一律「トリビアル」として無効化されていた問題を解決しました。これにより、高頻度取引やスキャルピング的なプライスアクションロジック、および他の条件との組み合わせで強力なシグナルとなるパターンの生成が可能になりました。

### 3.5 TPSL（利確・損切）の柔軟性不足 (TPSL Flexibility)

> **解決日: 2025-12-11**
>
> - `StrategyGene` に `long_tpsl_gene` / `short_tpsl_gene` フィールドを追加し、Long/Short 個別設定を可能化
> - `UniversalStrategy` で売買方向に応じた TPSL 遺伝子の動的選択ロジックを実装
> - `TPSLService` を Calculator パターンでリファクタリングし、拡張性を向上
> - `RandomGeneGenerator` および遺伝的操作（交叉・突然変異）を Long/Short 分離に対応

**解決された問題:**
TPSL 設定が戦略全体で単一のオブジェクトに制限されていた問題を解決しました。ロングとショートで異なるリスク設定を持つことや、将来的に市場環境に応じてロジックを切り替える基盤が整いました。

### 3.6 注文執行と TPSL の強化 (Order Execution & TPSL Enhancements)

> **解決日: 2025-12-11**
>
> - **悲観的約定モデル (Pessimistic Execution)**: 同一足内で SL/TP 両方に到達した場合、常に SL を優先するロジックを実装し、「幻の利益」問題を解決。
> - **トレーリングストップ (Trailing Stop)**: 価格推移に合わせて SL を有利な方向に更新するロジックを実装（`TPSLGene`に追加）。
> - **トレーリングテイクプロフィット (Trailing TP)**: TP 到達後も即時決済せず、利益確保ラインを追従させて利益を伸ばす機能を実装。
> - **内部状態管理**: `UniversalStrategy`内で SL/TP 価格を自前管理し、`backtesting.py`のブラックボックス依存を排除。

**解決された問題:**
ティックデータがない環境（1 分足のみ）でのバックテスト信頼性を向上させました。また、従来の固定的な SL/TP では実現できなかった「利益を伸ばして損失を限定する」動的なポジション管理が可能になりました。

### 3.7 エントリータイプの多様化 (Entry Type Diversification)

> **解決日: 2025-12-11**
>
> - `EntryType` Enum を追加（MARKET/LIMIT/STOP/STOP_LIMIT の 4 種類）
> - `EntryGene` データクラスを新設（注文タイプ、オフセット比率、有効期限を定義）
> - `StrategyGene` に `entry_gene` / `long_entry_gene` / `short_entry_gene` フィールドを追加
> - `EntryExecutor` サービスを新設（backtesting.py への limit/stop パラメータ計算）
> - `UniversalStrategy.next()` に `EntryExecutor` を統合
> - `RandomGeneGenerator` / `EntryGenerator` で自動生成対応
> - `DictConverter` でシリアライズ/デシリアライズ対応

**解決された問題:**
成行注文のみだった戦略実行に指値・逆指値注文を追加しました。GA が自動的に「有利な価格での約定を狙う指値戦略」や「ブレイクアウトを狙う逆指値戦略」を探索できるようになり、エントリー手法の多様性が向上しました。

## 4. システムアーキテクチャ・運用上の改善

### 4.1 エラーハンドリングとデバッグ情報の損失

> **解決日: 2025-12-09**
>
> - `ParallelEvaluator` にエラー種別詳細分類機能を追加
> - 最近のエラー履歴（最大 20 件）を保持し、デバッグ情報として取得可能

**解決された問題:**
評価エラーの詳細が不明でデバッグが困難だった問題を解決しました。

### 4.2 フィットネス共有の計算量 ($O(N^2)$)

> **解決日: 2025-12-09**
>
> - KD-Tree (`scipy.spatial.cKDTree`) を使用した空間分割による近傍探索で $O(N \log N)$ に改善
> - 大規模集団向けのサンプリングベース近似を実装

**解決された問題:**
大規模集団での計算ボトルネックを解消しました。

### 4.3 設定の整合性チェック不足

> **解決日: 2025-12-09**
>
> - `GASettings` にパラメータ間の整合性チェックを追加

**解決された問題:**
相互依存するパラメータの不正設定を防止しました。

## 5. 詳細コンポーネントの改善 (Component Improvements)

### 5.2 ポジションサイジングのバックテスト統合不全

> **解決日: 2025-12-09**
>
> - `PositionSizingService.calculate_position_size_fast()` メソッドを実装
> - `UniversalStrategy` に高速版メソッドを統合

**解決された問題:**
重い計算によるバックテスト速度の低下を解消しました。

### 5.3 Hybrid GA（ML 統合）のボトルネックとデータ汚染リスク

> **解決日: 2025-12-09**
>
> - `HybridIndividualEvaluator` で LRU キャッシュを活用
> - データ取得失敗時のハンドリングを強化

**解決された問題:**
DB アクセス過多によるボトルネックと、不完全データによる学習汚染を防止しました。

### 5.4 Hybrid GA のキャッシュバイパス修正 (Cache Bypass in Hybrid Mode)

> **解決日: 2025-12-10**
>
> - `HybridIndividualEvaluator._perform_single_evaluation` で `_get_cached_data` を呼び出し、取得したデータを `run_backtest` に `preloaded_data` として渡すように修正
> - `_fetch_ohlcv_data` のキャッシュキーにプレフィックス `"ohlcv"` を追加し、基底クラスの `backtest_data` とのキー衝突を回避
> - テストケースを追加（`test_preloaded_data_passed_to_backtest`, `test_ohlcv_cache_key_has_prefix`）

**解決された問題:**
Hybrid モードでのバックテスト実行時にデータキャッシュが効かず、毎回データベースアクセスが発生していたパフォーマンス問題を解決しました。また、OHLCV データと基底クラスのバックテストデータが同じキャッシュキーを使用することによるデータ衝突リスクを排除しました。

### 5.5 UniversalStrategy の責務過多と実装乖離

> **解決日: 2025-12-09**
>
> - `UniversalStrategy.next()` で `TPSLService` を正しく呼び出すように修正

**解決された問題:**
遺伝子で指定された高度なリスク計算が実際には無視されていた重大なバグを修正しました。

### 5.6 キャッシュ管理の不備 (Memory Leak Risk)

> **解決日: 2025-12-09**
>
> - `IndividualEvaluator` の `_data_cache` を `cachetools.LRUCache` に置換

**解決された問題:**
無制限なキャッシュ肥大化によるメモリ枯渇リスクを解消しました。

### 5.7 並列処理のゾンビ化リスク

> **解決日: 2025-12-09**
>
> - `ParallelEvaluator` に `use_process_pool` パラメータを追加
> - タイムアウト時のプロセス強制終了メカニズムを実装

**解決された問題:**
スレッドベースの並列処理によるタイムアウト処理の不備とリソース枯渇リスクを解消しました。

### 5.8 類似度計算の精度不足

> **解決日: 2025-12-09**
>
> - 特徴ベクトル化に指標タイプと条件演算子の情報を追加

**解決された問題:**
戦略の多様性を正しく評価できていなかった問題を改善しました。

### 5.9 並列評価のプロセス安全性 (Parallel Evaluation Process Safety)

> **解決日: 2025-12-10**
>
> - `ParallelEvaluator` に `worker_initializer` と `worker_initargs` を追加
> - Pickle できないオブジェクト（DB セッション等）のメインプロセスからの転送を回避
> - ワーカープロセス内でのサービスの安全な再構築をサポート

**解決された問題:**
ProcessPoolExecutor 使用時に DB セッションなどが原因で Pickle エラーが発生し、真の並列処理（マルチコア活用）が妨げられていた問題を解決しました。

## 6. 実装上の重大な欠陥修正 (Critical Bug Fixes)

### 6.1 Stateful Condition のロジック不備

> **解決日: 2025-12-09**
>
> - `StatefulCondition` に `direction` フィールドを追加
> - `UniversalStrategy` でステートフル条件成立時の売買方向判定を実装

**解決された問題:**
ステートフル条件が成立しても取引が実行されないバグを修正しました。

### 6.2 Position Sizing の静的キャッシュ問題

> **解決日: 2025-12-09**
>
> - `UniversalStrategy._calculate_position_size` からキャッシュを削除
> - エントリー毎に `equity` を参照して再計算

**解決された問題:**
複利効果が無効化されていた問題を修正しました。

## 7. 検証方法論の改善

### 7.1 堅牢性検証 (Walk-Forward Analysis) の欠如

> **解決日: 2025-12-09**
>
> - `IndividualEvaluator` に WFA（Walk-Forward Analysis）機能を統合

**解決された問題:**
過学習のリスクが高い単純な検証手法から、より堅牢な評価手法へ移行しました。
