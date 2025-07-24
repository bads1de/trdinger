# テストガイド

このディレクトリには、Trdinger プラットフォームのバックエンドテストが含まれています。

## ディレクトリ構造

```
tests/
├── unit/                    # 単体テスト
├── integration/             # 統合テスト
│   ├── test_comprehensive.py      # 包括的統合テスト
│   └── test_market_validation.py  # 市場検証テスト
├── e2e/                     # エンドツーエンドテスト
│   └── test_complete_workflow.py  # 完全ワークフローテスト
├── utils/                   # テストユーティリティ
│   ├── helpers.py              # ヘルパー関数
│   ├── fixtures.py             # 共通フィクスチャ
│   └── data_generators.py      # テストデータ生成
├── conftest.py              # pytest設定とフィクスチャ
├── run_tests.py             # テスト実行スクリプト
└── README.md                # このファイル
```

## テストカテゴリ

### 🔧 単体テスト (unit)

- 個別コンポーネントの動作確認
- 高速実行、外部依存なし
- マーカー: `@pytest.mark.unit`

### 🔗 統合テスト (integration)

- コンポーネント間の連携確認
- データベース、API 統合テスト
- マーカー: `@pytest.mark.integration`

### 🚀 エンドツーエンドテスト (e2e)

- 完全なワークフロー確認
- 実際のユーザーシナリオ
- マーカー: `@pytest.mark.e2e`

### 📊 市場検証テスト (market_validation)

- 実際の市場データでの検証
- パフォーマンス・精度確認
- マーカー: `@pytest.mark.market_validation`

## テスト実行方法

### 基本実行

```bash
# 全テスト実行
python tests/run_tests.py --full

# 高速テスト（unit + integration）
python tests/run_tests.py --quick

# カテゴリ別実行
python tests/run_tests.py -c unit
python tests/run_tests.py -c integration
python tests/run_tests.py -c e2e
```

### 詳細オプション

```bash
# カバレッジレポート付き
python tests/run_tests.py -c unit --coverage

# 並列実行
python tests/run_tests.py -c integration --parallel

# 特定のテストファイル
python tests/run_tests.py -t tests/integration/test_comprehensive.py

# 特定のテストメソッド
python tests/run_tests.py -t tests/integration/test_comprehensive.py::TestTPSLPositionSizingIntegration::test_tpsl_position_sizing_interaction
```

### 直接 pytest 実行

```bash
# マーカー指定
pytest -m unit
pytest -m "integration and not slow"
pytest -m "e2e or market_validation"

# 詳細出力
pytest -v tests/integration/

# カバレッジ
pytest --cov=app --cov-report=html tests/
```

## テストマーカー

| マーカー            | 説明                   | 実行時間 |
| ------------------- | ---------------------- | -------- |
| `unit`              | 単体テスト             | 高速     |
| `integration`       | 統合テスト             | 中程度   |
| `e2e`               | エンドツーエンドテスト | 低速     |
| `slow`              | 時間のかかるテスト     | 低速     |
| `market_validation` | 市場検証テスト         | 中程度   |
| `performance`       | パフォーマンステスト   | 中程度   |
| `security`          | セキュリティテスト     | 中程度   |

## テスト作成ガイドライン

### 1. テストファイル命名規則

```
test_<機能名>.py          # 単体テスト
test_<機能名>_integration.py  # 統合テスト
test_<機能名>_e2e.py      # E2Eテスト
```

### 2. テストクラス構造

```python
import pytest
from tests.utils.helpers import performance_monitor, assert_financial_precision
from tests.utils.data_generators import TestDataGenerator

@pytest.mark.integration
class TestFeatureName:
    """機能名のテスト"""

    def setup_method(self):
        """テストセットアップ"""
        self.data_generator = TestDataGenerator()

    def test_specific_functionality(self):
        """特定機能のテスト"""
        with performance_monitor("テスト名"):
            # テストロジック
            pass

    def teardown_method(self):
        """テスト後処理"""
        pass
```

### 3. 財務計算テスト

```python
from decimal import Decimal
from tests.utils.helpers import assert_financial_precision

def test_financial_calculation():
    """財務計算テスト例"""
    # Decimalを使用
    price = Decimal("50000.12345678")

    # 計算実行
    result = calculate_something(price)

    # 精度検証
    assert_financial_precision(float(result), expected_value, tolerance=1e-8)
```

### 4. パフォーマンステスト

```python
from tests.utils.helpers import performance_monitor
from tests.utils.data_generators import PerformanceTestHelper

@pytest.mark.performance
def test_performance():
    """パフォーマンステスト例"""
    helper = PerformanceTestHelper()

    def operation():
        # テスト対象の処理
        return process_data()

    result, execution_time = helper.measure_execution_time(operation)
    result, memory_used = helper.measure_memory_usage(operation)

    # パフォーマンス要件確認
    assert execution_time < 1.0  # 1秒以内
    assert memory_used < 100     # 100MB以内
```

## CI/CD 統合

### GitHub Actions 例

```yaml
- name: Run Tests
  run: |
    # 高速テスト（PR時）
    python tests/run_tests.py --quick --coverage

    # 全テスト（main branch）
    python tests/run_tests.py --full --coverage --parallel
```

### テスト結果レポート

- カバレッジレポート: `htmlcov/index.html`
- JUnit XML: `--junit-xml=test-results.xml`
- パフォーマンスレポート: テスト出力に含まれる

## トラブルシューティング

### よくある問題

1. **データベース接続エラー**

   ```bash
   # テスト用データベースの確認
   pytest tests/conftest.py::test_database_connection -v
   ```

2. **メモリ不足**

   ```bash
   # 並列実行数を調整
   pytest -n 2  # 2プロセスで実行
   ```

3. **テストデータ不足**
   ```bash
   # データ生成の確認
   python -c "from tests.utils.data_generators import TestDataGenerator; print(TestDataGenerator.generate_ohlcv_data(100))"
   ```

### デバッグ

```bash
# 詳細ログ出力
pytest -v -s tests/integration/test_comprehensive.py

# 特定のテストのみ実行
pytest tests/integration/test_comprehensive.py::TestTPSLPositionSizingIntegration::test_tpsl_position_sizing_interaction -v -s

# pdbデバッガー使用
pytest --pdb tests/integration/test_comprehensive.py
```

## 貢献ガイドライン

1. **新しいテストを追加する場合**:

   - 適切なカテゴリ（unit/integration/e2e）に配置
   - 適切なマーカーを付与
   - `tests/utils/` のヘルパー関数を活用

2. **テストデータが必要な場合**:

   - `TestDataGenerator` を使用
   - 再現可能なシードを設定
   - 極端なケースも含める

3. **パフォーマンステストの場合**:

   - `@pytest.mark.slow` または `@pytest.mark.performance` を付与
   - 適切な閾値を設定
   - メモリリークチェックを含める

4. **財務計算テストの場合**:
   - `Decimal` 型を使用
   - `assert_financial_precision` で精度確認
   - エッジケース（ゼロ、負の値等）を含める
