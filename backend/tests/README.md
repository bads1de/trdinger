# Backend Tests - 統合テストスイート

## 📋 概要

バックエンドテストを機能ベースで整理・統合したテストスイートです。
TDD原則に基づき、各機能を包括的にテストし、保守性と拡張性を向上させています。

## 🏗️ 構造

```
backend/tests/
├── __init__.py                 # 共通フィクスチャとユーティリティ
├── unit/                       # 単体テスト
│   ├── test_data_processing.py # データ処理統合テスト
│   ├── test_indicators.py      # 指標統合テスト
│   ├── test_backtest.py        # バックテスト統合テスト
│   └── __init__.py
├── integration/                # 統合テスト
│   ├── test_data_flow.py       # データフロー統合テスト
│   ├── test_strategy_execution.py # 戦略実行統合テスト
│   ├── test_ml_pipeline.py     # MLパイプライン統合テスト
│   └── __init__.py
├── utils/                      # ユーティリティテスト
│   ├── test_warnings_and_deprecations.py # 警告・非推奨機能テスト
│   └── __init__.py
├── performance/                # パフォーマンステスト（既存）
├── system/                     # システムテスト（既存）
├── auto_strategy/              # オートストラテジーテスト（既存）
└── README.md                   # このファイル
```

## 🎯 統合の成果

### ✅ 改善点

1. **ファイル数の削減**: 20+ファイル → 7ファイル（メイン統合ファイル）
2. **命名規則の統一**: 機能ベースの一貫した命名
3. **責任の明確化**: 各テストファイルの役割が明確
4. **重複除去**: 類似テストの統合による保守性向上
5. **統合テストの充実**: エンドツーエンドの包括的テスト

### 📊 統合内容

#### unit/test_data_processing.py
- データ変換（OHLCV, FundingRate, OpenInterest）
- データ処理（クリーニング、バリデーション、補間）
- データ型最適化
- エラー処理

#### unit/test_indicators.py
- トレンド指標（SAR, SMA, EMA, WMA）
- モメンタム指標（RSI, MACD, STOCH）
- ボラティリティ指標（ATR, BBANDS）
- 出来高指標（MFI, OBV, AD）
- MAVP, SQUEEZE, MFIなどの特殊指標

#### unit/test_backtest.py
- BacktestExecutorの実行ワークフロー
- StrategyFactoryの戦略生成
- FractionalBacktest統合
- 戦略実行とポジションサイジング

#### integration/test_data_flow.py
- CCXT ↔ DB ↔ API形式変換
- データ処理パイプライン
- 指標計算ワークフロー
- パフォーマンス・スケーラビリティ

#### integration/test_strategy_execution.py
- 戦略生成から実行までの完全フロー
- バックテスト実行ワークフロー
- 戦略パラメータ最適化
- エラーハンドリング

#### integration/test_ml_pipeline.py
- 特徴量エンジニアリング
- MLデータ準備
- モデル学習シミュレーション
- 交差検証と評価

#### utils/test_warnings_and_deprecations.py
- pandas非推奨機能（fillna methodパラメータ）
- 指標計算時の警告
- ログ削除検証
- 互換性テスト

## 🚀 実行方法

### 全テスト実行
```bash
cd backend
pytest tests/
```

### 特定カテゴリの実行
```bash
# 単体テストのみ
pytest tests/unit/

# 統合テストのみ
pytest tests/integration/

# 特定のテストファイル
pytest tests/unit/test_data_processing.py

# 特定のテストクラス
pytest tests/unit/test_data_processing.py::TestDataConversionIntegrated
```

### パフォーマンステスト
```bash
# パフォーマンステストを含む
pytest tests/ --durations=10

# 遅いテストを特定
pytest tests/ --durations=0
```

## 📈 カバレッジレポート

```bash
pytest tests/ --cov=backend --cov-report=html
```

## 🔧 テスト作成ガイドライン

### 1. TDD原則の遵守
- **Red**: 失敗するテストを書く
- **Green**: テストを通す最小限のコードを書く
- **Refactor**: コードを改善する

### 2. テスト構造
```python
import pytest
from unittest.mock import Mock

class TestFeature:
    """機能別のテストクラス"""

    @pytest.fixture
    def setup_data(self):
        """テストデータ準備"""
        return {"key": "value"}

    def test_success_case(self, setup_data):
        """成功ケースのテスト"""
        assert True

    def test_error_case(self, setup_data):
        """エラーケースのテスト"""
        with pytest.raises(ValueError):
            raise ValueError("test error")
```

### 3. 命名規則
- テストクラス: `Test[Feature]`
- テストメソッド: `test_[description]`
- フィクスチャ: `[data]_fixture`

### 4. 統合テストのベストプラクティス
- **独立性**: 各テストが他のテストに依存しない
- **完全性**: エンドツーエンドのワークフローをカバー
- **保守性**: 変更に対する影響を最小限に
- **速さ**: 実行時間が短い

## 📋 移行履歴

### 統合前の構造（問題点）
```
/tests/
├── test_backtest_executor.py     # 単独ファイル
├── test_data_conversion.py       # 分散
├── test_data_processor.py        # 分散
├── test_data_validation.py       # 分散
├── test_indicator_*.py          # 複数ファイル
└── ... (20+ files)
```

### 統合後の構造（改善点）
```
/tests/
├── unit/test_data_processing.py    # 統合済み
├── unit/test_indicators.py         # 統合済み
├── unit/test_backtest.py           # 統合済み
├── integration/test_data_flow.py   # 新規統合
└── ... (7 files)
```

## 🎯 品質基準

- **カバレッジ**: 目標80%以上
- **実行時間**: 全テスト10秒以内
- **保守性**: 重複コードなし、明確な責任分担
- **拡張性**: 新機能追加時の影響最小化

## 🔍 トラブルシューティング

### よくある問題

1. **ImportError**
   ```bash
   # PYTHONPATHの確認
   export PYTHONPATH=$PYTHONPATH:/path/to/backend
   ```

2. **Fixtureエラー**
   ```bash
   # conftest.pyの確認
   pytest --fixtures
   ```

3. **遅いテスト**
   ```bash
   # 実行時間分析
   pytest --durations=10
   ```

## 📞 サポート

テストに関する質問や問題は、開発チームまでお問い合わせください。