# エラーハンドリング推奨事項

## 🎯 結論: `@handle_pandas_ta_errors`は必要

### **❌ 完全削除は危険**

テスト結果から、デコレーターを完全に削除することは**危険**であることが判明しました。

### **📊 問題の実例**

#### pandas-ta 直接使用の問題

```python
# 空データ → None (エラーなし)
ta.atr(empty_series, empty_series, empty_series, length=14)  # None

# 短いデータ → None (エラーなし)
ta.atr(short_series, short_series, short_series, length=14)  # None

# NaNデータ → 全NaN (エラーなし)
ta.atr(nan_series, nan_series, nan_series, length=3)  # [nan, nan, nan, ...]
```

#### デコレーター付きの利点

```python
# 適切なエラーメッセージ
PandasTAError: atr: 計算結果がNoneです
PandasTAError: atr: データ長(2)が最小長(14)未満です
PandasTAError: atr: 計算結果が全てNaNです
```

## ✅ 推奨アプローチ: 軽量版デコレーター

### **1. 簡素化されたデコレーター**

**変更前**: 複雑な検証（約 50 行）

```python
def handle_pandas_ta_errors(func):
    # 複雑なエラー分類
    # 詳細なログ出力
    # 多重検証...
```

**変更後**: 軽量版（約 20 行）

```python
def handle_pandas_ta_errors(func):
    @wraps(func)
    def wrapper(*args, **kwargs):
        result = func(*args, **kwargs)

        # 重要な異常ケースのみチェック
        if result is None:
            raise PandasTAError(f"{func.__name__}: 計算結果がNoneです")
        if isinstance(result, np.ndarray) and np.all(np.isnan(result)):
            raise PandasTAError(f"{func.__name__}: 計算結果が全てNaNです")

        return result
```

### **2. 段階的適用戦略**

#### Phase 1: 重要な指標にのみ適用

- **volatility.py**: ATR, Bollinger Bands 等（価格に直結）
- **momentum.py**: RSI, MACD 等（シグナル生成に重要）
- **pattern_recognition.py**: パターン検出（誤検出防止）

#### Phase 2: 軽量エラーハンドリング

- **trend.py**: 基本的な入力検証のみ
- **volume.py**: 出来高関連（比較的安全）

#### Phase 3: 統計・数学系

- **statistics.py**: 統計関数（pandas 標準で十分）
- **math_transform.py**: 数学変換（エラー少ない）

## 🚀 実装ガイドライン

### **1. 重要度による分類**

#### 🔴 高リスク（デコレーター必須）

```python
@handle_pandas_ta_errors
def rsi(data, length=14):
    # RSIは取引シグナルに直結するため重要
```

#### 🟡 中リスク（軽量検証）

```python
def sma(data, length=20):
    if len(data) == 0:
        raise ValueError("データが空です")
    result = ta.sma(data, length=length)
    if result.isna().all():
        raise ValueError("計算結果が全てNaNです")
    return result.values
```

#### 🟢 低リスク（最小限）

```python
def simple_math_transform(data):
    # 数学変換は比較的安全
    return np.log(data)
```

### **2. パフォーマンス重視の実装**

```python
# ❌ 避けるべき（重い）
if np.all(np.isnan(result)) and len(result) > 1000:  # 大きな配列で重い

# ✅ 推奨（軽い）
if len(result) > 0 and np.all(np.isnan(result[:min(10, len(result))])):  # サンプルチェック
```

## 📋 移行計画

### **Step 1: 軽量デコレーターの導入** ✅

- `handle_pandas_ta_errors`を簡素化
- 重要な異常ケースのみチェック

### **Step 2: 段階的適用**

1. **volatility.py**: デコレーター維持
2. **momentum.py**: デコレーター維持
3. **trend.py**: 軽量検証追加
4. **その他**: 必要に応じて

### **Step 3: テストカバレッジ**

- 異常データでのテスト追加
- エラーハンドリングの検証
- パフォーマンステスト

## 🎯 期待される効果

### **品質向上**

- 異常データの早期検出
- 適切なエラーメッセージ
- デバッグ効率向上

### **パフォーマンス**

- 軽量化により処理速度向上
- 必要最小限の検証
- メモリ使用量削減

### **保守性**

- 一貫したエラーハンドリング
- 予測可能な動作
- テストの簡素化

## 💡 ベストプラクティス

1. **重要度に応じた適用**: 全てに適用せず、重要な指標のみ
2. **軽量化**: 重い検証は避け、重要な異常ケースのみ
3. **一貫性**: 同じパターンのエラーハンドリング
4. **テスト**: 異常データでの動作確認
5. **ドキュメント**: エラーハンドリングの方針を明記
