# TA-Lib移行テストスイート

このディレクトリには、TA-Lib移行に関連するテストファイルが含まれています。

## 📁 テストファイル構成

### 基本テスト
- **`test_talib_basic.py`** - TALibAdapterの基本機能テスト
- **`test_quick.py`** - クイックテスト（開発時の動作確認用）
- **`test_final.py`** - 最終統合テスト

### 包括的テスト
- **`test_extended_suite.py`** - 拡張テストスイート（包括的な機能テスト）
- **`test_updated_indicators.py`** - 更新された指標クラスのテスト
- **`test_stress.py`** - ストレステスト（大規模データ、並行処理等）

### 互換性テスト
- **`../compatibility_test.py`** - 既存システムとの互換性テスト

## 🚀 テスト実行方法

### 個別テスト実行

```bash
# 基本テスト
cd backend/tests/talib_migration
python test_talib_basic.py

# クイックテスト
python test_quick.py

# 最終テスト
python test_final.py

# 拡張テストスイート
python test_extended_suite.py

# ストレステスト
python test_stress.py

# 更新された指標テスト
python test_updated_indicators.py
```

### 互換性テスト実行

```bash
cd backend/tests
python compatibility_test.py
```

### pytest実行

```bash
cd backend
pytest tests/talib_migration/ -v
```

## 📊 テスト内容

### 1. 基本機能テスト (`test_talib_basic.py`)
- TALibAdapterの基本機能
- SMA, EMA, RSI, MACD計算
- エラーハンドリング
- パフォーマンス比較

### 2. 拡張テストスイート (`test_extended_suite.py`)
- 複数データサイズでの包括的テスト
- エッジケーステスト
- 指標クラステスト
- backtesting関数テスト
- データ一貫性テスト

### 3. ストレステスト (`test_stress.py`)
- 大規模データ処理テスト
- 並行処理テスト
- メモリリークテスト
- 高速連続計算テスト
- 極端な値でのテスト

### 4. 互換性テスト (`compatibility_test.py`)
- 既存API互換性
- 計算精度テスト
- フォールバック機能テスト

## ✅ 期待される結果

### 成功時の出力例
```
🔬 TA-lib移行 拡張テストスイート
======================================================================
📊 成功: 71/71 (100.0%)

🎉 全てのテストが成功しました！
✅ TA-lib移行は完全に成功しています
🚀 パフォーマンス、精度、一貫性すべてが確認されました
```

### パフォーマンス指標
- **計算速度**: 10-1000倍の高速化
- **精度**: 差分 < 1e-10 (完全一致)
- **メモリ**: リーク無し
- **並行処理**: 10スレッド同時実行成功

## 🔧 トラブルシューティング

### よくある問題

1. **TA-Libインポートエラー**
   ```
   ImportError: No module named 'talib'
   ```
   **解決方法**: TA-Libをインストール
   ```bash
   pip install TA-Lib
   # または
   conda install -c conda-forge ta-lib
   ```

2. **パス関連エラー**
   ```
   ModuleNotFoundError: No module named 'app'
   ```
   **解決方法**: 正しいディレクトリから実行
   ```bash
   cd backend/tests/talib_migration
   python test_*.py
   ```

3. **データ型エラー**
   ```
   TA-Lib計算エラー: input array type is not double
   ```
   **解決方法**: これは予想される動作（フォールバック機能が作動）

### デバッグ方法

1. **詳細ログ有効化**
   ```python
   import logging
   logging.basicConfig(level=logging.DEBUG)
   ```

2. **個別関数テスト**
   ```python
   from app.core.services.indicators.talib_adapter import TALibAdapter
   import pandas as pd
   
   data = pd.Series([1, 2, 3, 4, 5] * 20)
   result = TALibAdapter.sma(data, 10)
   print(result)
   ```

## 📋 テスト結果の解釈

### 成功指標
- ✅ 全テスト成功率 100%
- ✅ 計算精度 < 1e-10
- ✅ API互換性 100%
- ✅ メモリリーク無し

### 警告レベル
- ⚠️ 一部テスト失敗 (95-99%)
- ⚠️ 計算精度 1e-6 - 1e-10
- ⚠️ 軽微なメモリ増加 (<50MB)

### エラーレベル
- ❌ 多数テスト失敗 (<95%)
- ❌ 計算精度 > 1e-6
- ❌ API互換性問題
- ❌ 大幅なメモリリーク (>100MB)

## 🔄 継続的テスト

### 開発時
1. コード変更後は `test_quick.py` で動作確認
2. 新機能追加時は対応するテストを追加
3. リリース前は全テストスイート実行

### CI/CD統合
```yaml
# GitHub Actions例
- name: Run TA-Lib Migration Tests
  run: |
    cd backend
    python -m pytest tests/talib_migration/ -v
    python tests/compatibility_test.py
```

## 📚 関連ドキュメント

- [TA_LIB_MIGRATION_REPORT.md](../../../TA_LIB_MIGRATION_REPORT.md) - 移行完了報告書
- [FINAL_TEST_REPORT.md](../../../FINAL_TEST_REPORT.md) - 最終テスト報告書
- [backend/tests/unit/test_talib_adapter.py](../unit/test_talib_adapter.py) - ユニットテスト
