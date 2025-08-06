# 設計ドキュメント

## 概要

ML 関連とオートストラテジー全般の包括的なテストスイートは、Trdinger プラットフォームの計算精度、パフォーマンス、統合性を検証するための統合テストフレームワークです。このシステムは、財務計算の精度問題を洗い出し、ML 機能とオートストラテジー機能の正確性を保証することを目的としています。

### 設計目標

- **計算精度の保証**: すべての財務計算で Decimal 型を使用し、8 桁精度を維持
- **パフォーマンス検証**: 市場データ処理 < 100ms、戦略シグナル生成 < 500ms の要件を満たす
- **統合テスト**: ML 機能とオートストラテジー機能の連携を検証
- **自動化**: 継続的な品質保証のための自動テスト実行
- **包括性**: エッジケース、エラーハンドリング、セキュリティを含む全面的なテスト

## アーキテクチャ

### 全体アーキテクチャ

```
┌─────────────────────────────────────────────────────────────┐
│                    テストオーケストレーター                      │
├─────────────────────────────────────────────────────────────┤
│  ML テスト     │  オートストラテジー  │  統合テスト  │  パフォーマンス │
│  フレームワーク  │  テストフレームワーク │  フレームワーク │  テストスイート │
├─────────────────────────────────────────────────────────────┤
│                    共通テストインフラ                          │
│  • モックシステム  • テストデータ管理  • レポート生成           │
├─────────────────────────────────────────────────────────────┤
│                 Trdinger コアシステム                         │
│  FastAPI + SQLAlchemy + CCXT + ML Pipeline                 │
└─────────────────────────────────────────────────────────────┘
```

### レイヤー構造

**テストオーケストレーション層**

- テスト実行の調整と管理
- テスト結果の集約とレポート生成
- 並行テスト実行の制御

**テストフレームワーク層**

- 各ドメイン固有のテストロジック
- テストケースの定義と実行
- アサーション機能

**共通インフラ層**

- モックシステム（CCXT API、データベース）
- テストデータ管理とフィクスチャ
- ユーティリティ関数

**システム層**

- 既存の Trdinger プラットフォーム
- テスト対象のコンポーネント

### 設計決定の根拠

**1. レイヤー分離アプローチ**

- 各テストドメインを独立したフレームワークとして設計
- 保守性と拡張性を向上
- テストの並行実行を可能にする

**2. 共通インフラの抽出**

- モックシステムとテストデータ管理を共通化
- コードの重複を削減
- 一貫したテスト環境を提供

## コンポーネントとインターフェース

### 1. ML テストフレームワーク

```python
class MLTestFramework:
    """ML 機能の包括的テストを実行するフレームワーク"""

    async def test_feature_calculation_accuracy(self) -> TestResult:
        """特徴量計算の精度テスト"""

    async def test_model_prediction_accuracy(self) -> TestResult:
        """ML 予測の精度テスト"""

    async def test_ensemble_model_weighting(self) -> TestResult:
        """アンサンブルモデルの重み付けテスト"""

    async def test_feature_engineering_pipeline(self) -> TestResult:
        """特徴量エンジニアリングパイプラインテスト"""
```

**主要機能:**

- Decimal 型を使用した財務計算の精度検証
- ML モデルの予測精度と信頼度スコア検証
- 特徴量エンジニアリングの正確性検証
- アンサンブルモデルの重み付け計算検証

### 2. オートストラテジーテストフレームワーク

```python
class AutoStrategyTestFramework:
    """オートストラテジー機能の包括的テストを実行するフレームワーク"""

    async def test_genetic_algorithm_fitness(self) -> TestResult:
        """遺伝的アルゴリズムの適応度関数テスト"""

    async def test_strategy_parameter_generation(self) -> TestResult:
        """戦略パラメータ生成テスト"""

    async def test_position_sizing_calculation(self) -> TestResult:
        """ポジションサイジング計算テスト"""

    async def test_tp_sl_calculation(self) -> TestResult:
        """TP/SL 計算テスト"""

    async def test_strategy_backtest_metrics(self) -> TestResult:
        """戦略バックテスト指標テスト"""
```

**主要機能:**

- 遺伝的アルゴリズムの適応度関数検証
- 戦略パラメータの有効性検証
- リスク管理計算の精度検証
- バックテスト指標（シャープレシオ、最大ドローダウン）の正確性検証

### 3. 統合テストフレームワーク

```python
class IntegrationTestFramework:
    """ML とオートストラテジーの統合テストを実行するフレームワーク"""

    async def test_ml_strategy_signal_integration(self) -> TestResult:
        """ML モデルと戦略シグナルの統合テスト"""

    async def test_market_data_update_latency(self) -> TestResult:
        """市場データ更新レイテンシテスト"""

    async def test_signal_generation_performance(self) -> TestResult:
        """シグナル生成パフォーマンステスト"""

    async def test_portfolio_update_performance(self) -> TestResult:
        """ポートフォリオ更新パフォーマンステスト"""

    async def test_concurrent_processing(self) -> TestResult:
        """並行処理と競合状態テスト"""
```

**主要機能:**

- ML モデルとオートストラテジーの連携検証
- パフォーマンス要件の検証（100ms、500ms、1 秒）
- 並行処理の安全性検証
- システム統合の正確性検証

### 4. テクニカルインジケータテストスイート

```python
class TechnicalIndicatorTestSuite:
    """テクニカルインジケータの計算精度テスト"""

    async def test_moving_averages(self) -> TestResult:
        """移動平均（SMA/EMA）の計算精度テスト"""

    async def test_rsi_calculation(self) -> TestResult:
        """RSI 計算精度テスト"""

    async def test_macd_calculation(self) -> TestResult:
        """MACD 計算精度テスト"""

    async def test_bollinger_bands(self) -> TestResult:
        """ボリンジャーバンド計算精度テスト"""

    async def test_stochastic_oscillator(self) -> TestResult:
        """ストキャスティクス計算精度テスト"""

    async def test_atr_calculation(self) -> TestResult:
        """ATR 計算精度テスト"""

    async def test_custom_indicators(self) -> TestResult:
        """カスタムインジケータ計算精度テスト"""
```

### 5. セキュリティテストフレームワーク

```python
class SecurityTestFramework:
    """データ保護と入力検証のテスト"""

    async def test_financial_data_encryption(self) -> TestResult:
        """財務データ暗号化テスト"""

    async def test_audit_logging(self) -> TestResult:
        """監査ログ記録テスト"""

    async def test_input_validation(self) -> TestResult:
        """入力データ検証テスト"""

    async def test_environment_variable_security(self) -> TestResult:
        """環境変数セキュリティテスト"""

    async def test_sensitive_data_masking(self) -> TestResult:
        """機密データマスキングテスト"""
```

### 6. パフォーマンステストスイート

```python
class PerformanceTestSuite:
    """システムパフォーマンスの検証"""

    async def test_market_data_processing_latency(self) -> TestResult:
        """市場データ処理レイテンシテスト（< 100ms）"""

    async def test_strategy_execution_performance(self) -> TestResult:
        """戦略実行パフォーマンステスト（< 500ms）"""

    async def test_portfolio_update_latency(self) -> TestResult:
        """ポートフォリオ更新レイテンシテスト（< 1秒）"""

    async def test_resource_management(self) -> TestResult:
        """リソース管理テスト"""

    async def test_memory_usage_monitoring(self) -> TestResult:
        """メモリ使用量監視テスト"""

    async def test_database_connection_pooling(self) -> TestResult:
        """データベース接続プールテスト"""
```

## データモデル

### テスト結果データモデル

```python
from decimal import Decimal
from datetime import datetime
from enum import Enum
from typing import Optional, Dict, Any
from pydantic import BaseModel

class TestStatus(Enum):
    PASSED = "passed"
    FAILED = "failed"
    SKIPPED = "skipped"
    ERROR = "error"

class TestResult(BaseModel):
    """個別テスト結果"""
    test_name: str
    status: TestStatus
    execution_time_ms: Decimal
    expected_value: Optional[Any] = None
    actual_value: Optional[Any] = None
    error_message: Optional[str] = None
    timestamp: datetime
    correlation_id: str

class TestSuiteResult(BaseModel):
    """テストスイート結果"""
    suite_name: str
    total_tests: int
    passed_tests: int
    failed_tests: int
    skipped_tests: int
    error_tests: int
    total_execution_time_ms: Decimal
    coverage_percentage: Decimal
    test_results: List[TestResult]
    timestamp: datetime

class PerformanceBenchmark(BaseModel):
    """パフォーマンスベンチマーク"""
    operation_name: str
    target_latency_ms: Decimal
    actual_latency_ms: Decimal
    throughput_ops_per_second: Decimal
    memory_usage_mb: Decimal
    cpu_usage_percentage: Decimal
    timestamp: datetime
```

### テストデータフィクスチャ

```python
class TestDataFixtures:
    """テスト用データフィクスチャ"""

    @staticmethod
    def get_sample_market_data() -> Dict[str, Any]:
        """サンプル市場データ"""
        return {
            "symbol": "BTC/USDT:USDT",
            "price": Decimal("50000.12345678"),
            "volume": Decimal("1000.00000000"),
            "timestamp": datetime.now(),
            "ohlcv": [
                [1640995200000, Decimal("49000.00000000"), Decimal("51000.00000000"),
                 Decimal("48500.00000000"), Decimal("50000.12345678"), Decimal("1000.00000000")]
            ]
        }

    @staticmethod
    def get_sample_strategy_parameters() -> Dict[str, Any]:
        """サンプル戦略パラメータ"""
        return {
            "rsi_period": 14,
            "rsi_overbought": Decimal("70.0"),
            "rsi_oversold": Decimal("30.0"),
            "stop_loss_percentage": Decimal("0.02"),
            "take_profit_percentage": Decimal("0.04"),
            "position_size_percentage": Decimal("0.1")
        }

    @staticmethod
    def get_expected_calculation_results() -> Dict[str, Decimal]:
        """期待される計算結果"""
        return {
            "sma_20": Decimal("49500.12345678"),
            "ema_20": Decimal("49750.12345678"),
            "rsi_14": Decimal("65.5"),
            "macd_signal": Decimal("150.25"),
            "bollinger_upper": Decimal("51000.12345678"),
            "bollinger_lower": Decimal("48000.12345678")
        }
```

## エラーハンドリング

### エラー分類と対応戦略

**1. 計算精度エラー**

```python
class CalculationAccuracyError(Exception):
    """計算精度に関するエラー"""
    def __init__(self, expected: Decimal, actual: Decimal, tolerance: Decimal):
        self.expected = expected
        self.actual = actual
        self.tolerance = tolerance
        super().__init__(f"計算精度エラー: 期待値 {expected}, 実際値 {actual}, 許容誤差 {tolerance}")
```

**2. パフォーマンス要件違反**

```python
class PerformanceViolationError(Exception):
    """パフォーマンス要件違反エラー"""
    def __init__(self, operation: str, target_ms: Decimal, actual_ms: Decimal):
        self.operation = operation
        self.target_ms = target_ms
        self.actual_ms = actual_ms
        super().__init__(f"パフォーマンス要件違反: {operation} - 目標 {target_ms}ms, 実際 {actual_ms}ms")
```

**3. 外部 API 障害対応**

```python
class ExternalAPITestError(Exception):
    """外部API テストエラー"""
    pass

async def handle_ccxt_api_failure(test_function):
    """CCXT API 障害時のフォールバック処理"""
    try:
        return await test_function()
    except ccxt.NetworkError:
        # モックデータを使用してテスト継続
        return await test_function(use_mock_data=True)
    except ccxt.ExchangeError as e:
        # テストスキップまたはエラー記録
        raise ExternalAPITestError(f"取引所API エラー: {str(e)}")
```

### エラーログとトレーシング

```python
import structlog
from uuid import uuid4

logger = structlog.get_logger()

class TestExecutionContext:
    """テスト実行コンテキスト"""
    def __init__(self):
        self.correlation_id = str(uuid4())
        self.start_time = datetime.now()

    async def log_test_start(self, test_name: str):
        """テスト開始ログ"""
        logger.info(
            "テスト開始",
            correlation_id=self.correlation_id,
            test_name=test_name,
            timestamp=self.start_time
        )

    async def log_test_error(self, test_name: str, error: Exception):
        """テストエラーログ"""
        logger.error(
            "テストエラー",
            correlation_id=self.correlation_id,
            test_name=test_name,
            error_type=type(error).__name__,
            error_message=str(error),
            timestamp=datetime.now()
        )
```

## テスト戦略

### 1. 単体テスト戦略

**財務計算テスト**

- 既知の期待値を使用した精度検証
- Decimal 型の使用確認
- ROUND_HALF_UP の適用確認
- エッジケース（ゼロ、負の値、極端な値）のテスト

**ML モデルテスト**

- 決定論的シードを使用した再現可能テスト
- モデル予測の一貫性検証
- 信頼度スコアの範囲検証（0-1）
- 特徴量エンジニアリングの正確性検証

### 2. 統合テスト戦略

**システム間連携テスト**

- ML モデルとオートストラテジーの連携
- 市場データフローの検証
- リアルタイム更新の正確性
- 並行処理の安全性

**パフォーマンステスト**

- レイテンシ要件の検証
- スループット測定
- リソース使用量監視
- スケーラビリティテスト

### 3. エンドツーエンドテスト戦略

**完全なトレーディングフロー**

- 市場データ取得から戦略実行まで
- ポートフォリオ更新の正確性
- エラー状況での動作確認
- セキュリティ要件の検証

### 4. 継続的テスト戦略

**自動化パイプライン**

- コミット時の自動テスト実行
- 定期的な回帰テスト
- パフォーマンス監視
- テストカバレッジ追跡

**テストレポート**

- 詳細なテスト結果レポート
- パフォーマンスベンチマーク
- カバレッジ分析
- 傾向分析とアラート

### 設計の利点

**1. 包括性**

- ML、オートストラテジー、統合、パフォーマンス、セキュリティを網羅
- エッジケースとエラー状況を含む全面的なテスト

**2. 精度保証**

- Decimal 型による財務計算の精度保証
- 既知の期待値による検証
- 厳密な許容誤差設定

**3. パフォーマンス監視**

- リアルタイムパフォーマンス測定
- 要件違反の即座な検出
- リソース使用量の継続監視

**4. 保守性**

- モジュラー設計による拡張性
- 共通インフラによるコード再利用
- 構造化ログによるトレーサビリティ

**5. 自動化**

- 継続的な品質保証
- 人的エラーの削減
- 迅速なフィードバックループ
