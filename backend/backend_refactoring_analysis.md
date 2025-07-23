# バックエンドコードベースの分析とリファクタリング提案

## 1. 概要

このドキュメントは、`backend`ディレクトリ全体のコードベースを分析し、品質、保守性、拡張性を向上させるためのリファクタリング案をまとめたものです。コードは全体的に機能分割が意識されていますが、いくつかの領域でさらなる改善が可能です。

## 2. 全体的な評価

### 良い点

- **FastAPI の適切な活用**: API ルーター、Pydantic モデル、依存性注入などが効果的に使用されています。
- **ディレクトリ構造**: `api`, `core`, `database` といった基本的な責務分離ができています。
- **統一エラーハンドリング**: `UnifiedErrorHandler` の導入により、エラー処理の共通化が図られています。
- **設定管理**: `unified_config.py` により、設定が一元管理されています。

### 主要な課題

- **責務の不完全な分離**: 一部の API ルーターにビジネスロジックが残存しており、サービス層への委譲が不完全です。
- **機能の重複と分散**: 特に機械学習関連の機能が複数のファイルに分散し、見通しが悪くなっています。
- **一貫性の欠如**: データベースセッションの管理や API レスポンスの形式に若干の揺らぎが見られます。
- **過剰な分割**: `auto_strategy` パッケージ内のサブディレクトリが細かすぎ、全体の把握を困難にしている可能性があります。

---

## 3. 主要な課題と改善提案

#### 課題: `MLOrchestrator` と `MLTrainingService` の役割重複

- **現状**: `MLOrchestrator` と `MLTrainingService` の両方がモデルの学習や予測に関与しており、役割分担が不明確です。
- **提案**:
  - `MLTrainingService` を学習のメインサービスとし、`MLOrchestrator` は予測信号の生成や、より高レベルのワークフロー管理に特化させる、あるいは機能を `MLTrainingService` に統合して `MLOrchestrator` を廃止することを検討します。

#### 課題: `auto_strategy` パッケージの過剰な分割

- **現状**: `auto_strategy` パッケージ内が `calculators`, `engines`, `evaluators`, `factories`, `generators`, `managers`, `models`, `operators`, `services`, `utils` と非常に細かく分割されています。
- **問題点**:
  - クラス間の関係性が複雑になり、コードの追跡が困難になります。
  - 小さな責務のためにファイルが乱立し、全体像の把握を妨げます。
- **提案**:
  - **関連機能のマージ**: 例えば、`calculators` と `evaluators` は密接に関連しているため、`evaluation` や `logic` のような単一のパッケージに統合することを検討します。
  - **`engines` と `managers` の役割明確化**: `GeneticAlgorithmEngine` と `ExperimentManager` の役割を再評価し、責務が重複している場合は統合を検討します。現状では `ExperimentManager` が `GAEngine` をラップする形になっていますが、よりシンプルな構成にできる可能性があります。

#### 課題: モデルクラス内のシリアライゼーションロジック

- **現状**: `gene_strategy.py` の `StrategyGene` クラス内に `to_dict`, `from_dict` などのシリアライゼーションメソッドが存在します。
- **問題点**: モデルクラスがシリアライゼーションという異なる責務を持ってしまっています。
- **提案**:
  - シリアライゼーションロジックは `gene_serialization.py` の `GeneSerializer` に完全に移譲し、`StrategyGene` クラスからはこれらのメソッドを削除します。

### 3.3. コードの複雑性と一貫性

#### ✅ 完了: 依存性注入とセッション管理の不統一

- **修正内容**:
  - `app/core/dependencies.py`の`get_backtest_service()`を依存性注入パターンに変更
  - `app/api/ml_training.py`、`app/api/ml_management.py`の`get_data_service()`を依存性注入パターンに変更
  - `app/core/services/data_collection/bybit/bybit_service.py`で`SessionLocal()`の直接使用を`get_db()`に変更
  - `app/core/services/backtest_service.py`で`SessionLocal()`の直接使用を`get_db()`に変更
  - `app/core/services/auto_strategy/services/ml_orchestrator.py`のセッション管理を改善

#### ✅ 完了: API レスポンス形式の不統一

- **修正内容**:
  - `app/api/auto_strategy.py`で GAGenerationResponse、MultiObjectiveResultResponse、GAResultResponse を`APIResponseHelper`に変更
  - `app/api/ml_training.py`で MLTrainingResponse、手動レスポンス辞書を`APIResponseHelper`に変更
  - `app/api/ml_management.py`で手動レスポンス辞書を`APIResponseHelper`に変更

## 4. まとめ

提案されたリファクタリングを実施することで、以下の効果が期待できます。

- **保守性の向上**: 機能の凝集度が高まり、責務が明確になることで、コードの理解と修正が容易になります。
- **テスト容易性の向上**: 各コンポーネントが疎結合になることで、単体テストやモックを使用したテストが書きやすくなります。
- **バグの削減**: 状態管理がシンプルになり、一貫性が保たれることで、潜在的なバグを未然に防ぎます。
- **拡張性の向上**: コードベースが整理されることで、将来的な機能追加が容易になります。

まずは、影響範囲が限定的で効果の高い「API ルーター内のビジネスロジックのサービス層への移動」と「ML トレーニング関連機能の統合」から着手することを推奨します。
