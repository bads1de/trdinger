# バックエンドコードベース リファクタリング計画書

## 1. はじめに

### 1.1. 目的
本ドキュメントは、`backend`コードベースの現状を分析し、保守性、可読性、拡張性、およびパフォーマンスを向上させるための具体的なリファクタリング計画を提案するものです。

### 1.2. 分析概要
コードベース全体をレビューした結果、多くの機能が堅牢に実装されている一方で、いくつかの領域で改善の余地があることが確認されました。主な課題は以下の通りです。

- **依存性注入の不徹底**: 一部のAPIルーターでサービスが直接インスタンス化されており、パフォーマンスとテスト容易性に影響を与えています。
- **責務の重複**: 複数のサービスクラスやオーケストレーションサービス間で、責務が重複・分散している箇所が見られます。
- **コードの重複**: APIルーター内に、明らかなコードの重複（バグ）が存在します。
- **設定管理の複雑性**: 統一設定システムは強力ですが、一部で`dataclasses`と`pydantic-settings`が混在しており、簡素化の余地があります。
- **サービス層の肥大化**: 特に機械学習と自動戦略生成に関連するサービスは、多くの責務を抱え複雑化しています。

---

## 2. 主要なリファクタリング項目

### 2.1. 依存性注入（DI）の徹底と統一

- **問題点**:
  - `app/api/backtest.py`や`app/api/data_collection.py`などのAPIルーター内で、オーケストレーションサービスがリクエストごとに直接インスタンス化されています。(`orchestration_service = BacktestOrchestrationService()`)
  - これにより、リクエストごとに不要なオブジェクトが生成され、パフォーマンスが低下する可能性があります。また、単体テスト時のモック化が困難になります。

- **改善案**:
  - FastAPIの依存性注入システム（`Depends`）を全面的に採用します。
  - `app/api/dependencies.py`に、全てのオーケストレーションサービスの依存関係解決関数を定義し、各APIエンドポイントは`Depends`を通じてサービスを受け取るように統一します。

- **実装例 (`app/api/backtest.py`)**:
  ```python
  # 変更前
  @router.get("/results", response_model=BacktestResultsResponse)
  async def get_backtest_results(...):
      async def _get_results():
          orchestration_service = BacktestOrchestrationService()
          return await orchestration_service.get_backtest_results(...)
      return await UnifiedErrorHandler.safe_execute_async(_get_results)

  # 変更後
  from app.api.dependencies import get_backtest_orchestration_service

  @router.get("/results", response_model=BacktestResultsResponse)
  async def get_backtest_results(
      ...,
      orchestration_service: BacktestOrchestrationService = Depends(get_backtest_orchestration_service)
  ):
      async def _get_results():
          return await orchestration_service.get_backtest_results(...)
      return await UnifiedErrorHandler.safe_execute_async(_get_results)
  ```

- **期待される効果**:
  - パフォーマンスの向上（不要なインスタンス生成の削減）。
  - テスト容易性の向上（依存関係のモック化が容易になる）。
  - コードの可読性と一貫性の向上。

### 2.2. コードの重複排除（バグ修正）

- **問題点**:
  - `app/api/funding_rates.py`に`get_funding_rates`関数が2つ重複して定義されています。
  - `app/api/ml_management.py`に`get_models`関数が2つ重複して定義されています。
  - これらは明らかなバグであり、意図しない挙動を引き起こす原因となります。

- **改善案**:
  - 各ファイルから重複している関数定義を削除し、正しいエンドポイントのみを残します。

- **期待される効果**:
  - バグの修正とコードの信頼性向上。

### 2.3. サービス層の責務の明確化

- **問題点**:
  - `MLOrchestrator`が特徴量計算、モデル予測、データ取得など、多くの責務を担っており、肥大化しています。
  - `AutoStrategyService`と`ExperimentManager`の間の責務分担が複雑で、見通しが悪くなっています。
  - データ収集ロジックが`historical_data_service.py`と各データタイプ（`funding_rate_service.py`など）のサービスに分散しています。

- **改善案**:
  - **MLサービス**:
    - `MLOrchestrator`は、**予測の実行**と、そのために必要な**特徴量計算の呼び出し**に責務を限定します。
    - `MLTrainingOrchestrationService`が、データ取得からモデル学習、保存までの一連の**学習パイプライン**全体の管理責任を負います。
  - **AutoStrategyサービス**:
    - `AutoStrategyService`をAPI層とコアロジックをつなぐFacade（窓口）として位置づけ、実験の開始・停止・結果取得などのエントリーポイントとします。
    - `ExperimentManager`は、単一のGA実験のライフサイクル（実行、進捗管理、永続化）に特化させます。
  - **データ収集サービス**:
    - `DataCollectionOrchestrationService`に、OHLCV、FR、OI、Fear & Greedなど、全てのデータ収集タスクの開始と管理ロジックを集約します。
    - 各データタイプ別のサービスクラス（例: `BybitFundingRateService`）は、特定の取引所APIとの通信とデータ変換のみに責務を限定します。

- **期待される効果**:
  - 各クラスの責務が明確になり、コードの理解と修正が容易になる（単一責任原則）。
  - サービス間の結合度が下がり、コンポーネントの再利用性が向上する。

### 2.4. 設定管理の簡素化

- **問題点**:
  - `app/config/unified_config.py`内の`MLConfig`クラスが`dataclasses.dataclass`で定義されており、他の`pydantic-settings`ベースの設定クラスと形式が異なっています。

- **改善案**:
  - `MLConfig`およびその内部クラス（`MLDataProcessingConfig`など）も`pydantic-settings`の`BaseSettings`を継承するように変更し、全ての設定クラスの形式を統一します。
  - これにより、環境変数からの設定読み込みや型検証の仕組みが統一され、管理が簡素化されます。

- **期待される効果**:
  - 設定管理の一貫性と可読性の向上。
  - 環境変数による設定オーバーライドの容易化。

---

## 3. 推奨されるリファクタリング手順

以下の順序でリファクタリングを進めることを推奨します。

1.  **【優先度: 高】バグ修正**: `funding_rates.py`と`ml_management.py`の重複した関数定義を即座に修正します。
2.  **【優先度: 高】依存性注入の統一**: 全てのAPIルーターで`Depends`を使用するように修正します。これは影響範囲が広く、多くの改善の基盤となります。
3.  **【優先度: 中】サービス層の責務分離**: まずは責務が明確なデータ収集サービスから着手し、次にMLサービス、最後に複雑なAutoStrategyサービスの順でリファクタリングを進めます。
4.  **【優先度: 低】設定管理の簡素化**: 全ての機能が安定して動作することを確認した後、設定管理クラスの統一を行います。

この計画に従ってリファクタリングを進めることで、コードベース全体の品質が向上し、将来の機能追加やメンテナンスがより効率的に行えるようになります。
