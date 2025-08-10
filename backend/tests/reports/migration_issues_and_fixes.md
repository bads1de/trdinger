# talib → pandas-ta 移行: 発見された問題と修正

## 修正済みの問題

### 1. handle_talib_errors の残存
**ファイル**: `backend/app/services/indicators/technical_indicators/price_transform.py`
**問題**: `handle_talib_errors` のインポートが残存
**修正**: `handle_pandas_ta_errors` に変更

```python
# 修正前
from ..utils import (
    handle_talib_errors,  # ← 古いインポート
    ...
)

# 修正後
from ..utils import (
    handle_pandas_ta_errors,  # ← 新しいインポート
    ...
)
```

### 2. PandasTAError インポートの不整合
**ファイル**: 
- `backend/app/services/indicators/technical_indicators/cycle.py`
- `backend/app/services/indicators/technical_indicators/volatility.py`

**問題**: PandasTAError がインポートされていない
**修正**: 両ファイルに PandasTAError のインポートを追加

```python
# 修正前
from ..utils import (
    ensure_numpy_array,
    format_indicator_result,
    handle_pandas_ta_errors,
    validate_input,
)

# 修正後
from ..utils import (
    PandasTAError,  # ← 追加
    ensure_numpy_array,
    format_indicator_result,
    handle_pandas_ta_errors,
    validate_input,
)
```

## 残存する問題と修正提案

### 1. TechnicalIndicatorService の互換性問題 (高優先度)

**問題**: `_map_data_key_to_param` メソッドが存在しない

**修正提案**:
```python
# backend/app/services/indicators/indicator_orchestrator.py に追加

def _map_data_key_to_param(self, data_key: str) -> str:
    """データキーを関数パラメータ名にマッピング"""
    mapping = {
        'close': 'close',
        'open': 'open', 
        'high': 'high',
        'low': 'low',
        'volume': 'volume'
    }
    return mapping.get(data_key.lower(), data_key)
```

### 2. エラーハンドリングの統一 (中優先度)

**問題**: ValueError と PandasTAError の使い分けが不統一

**修正提案**:
```python
# indicator_orchestrator.py の修正
try:
    # 既存のロジック
    if actual_column is None:
        raise PandasTAError(  # ValueError → PandasTAError
            f"必要なカラム '{data_key}' がDataFrameにありません。"
            f"利用可能なカラム: {list(df.columns)}"
        )
except Exception as e:
    if isinstance(e, PandasTAError):
        raise
    else:
        raise PandasTAError(f"指標計算エラー: {str(e)}")
```

### 3. パラメータバリデーションの強化 (中優先度)

**問題**: 無効なパラメータ（負の値、ゼロ）の検証が不十分

**修正提案**:
```python
# pandas_ta_utils.py の _validate_data 関数を拡張

def _validate_parameters(length: int, min_length: int = 1) -> None:
    """パラメータの検証"""
    if not isinstance(length, int):
        raise PandasTAError(f"期間は整数である必要があります: {type(length)}")
    
    if length <= 0:
        raise PandasTAError(f"期間は正の値である必要があります: {length}")
    
    if length < min_length:
        raise PandasTAError(f"期間が小さすぎます: 最小{min_length}, 実際{length}")

# 各指標関数で使用
@_handle_errors
def sma(data: Union[np.ndarray, pd.Series], length: int) -> np.ndarray:
    """単純移動平均"""
    _validate_parameters(length)  # ← 追加
    series = _to_series(data)
    _validate_data(series, length)
    result = ta.sma(series, length=length)
    return result.values
```

## 推奨される実装順序

### Phase 1: 緊急修正 (即座に実施)
1. TechnicalIndicatorService の `_map_data_key_to_param` メソッド実装
2. エラーハンドリングの PandasTAError への統一

### Phase 2: 品質向上 (1-2日以内)
1. パラメータバリデーションの強化
2. 全指標での動作確認テスト
3. エッジケースのテスト追加

### Phase 3: 最適化 (1週間以内)
1. パフォーマンス最適化
2. メモリ使用量の改善
3. ドキュメントの更新

## テスト実行コマンド

修正後は以下のテストを実行して確認してください：

```bash
# 基本機能テスト
cd backend && python -m pytest tests/test_pandas_ta_basic.py -v

# 移行漏れ検出テスト
cd backend && python -m pytest tests/test_talib_migration_comprehensive.py::TestTalibMigrationComprehensive::test_no_talib_error_handling_remaining -v

# エッジケーステスト
cd backend && python -m pytest tests/test_migration_edge_cases.py::TestMigrationEdgeCases::test_empty_dataframe -v
```

## 移行完了の確認項目

- [ ] TechnicalIndicatorService の修正完了
- [ ] 全テストの PASSED 確認
- [ ] 実データでの動作確認
- [ ] パフォーマンス劣化なし
- [ ] メモリリークなし

これらの修正により、talib から pandas-ta への移行が完全に完了します。
