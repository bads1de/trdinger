# インジケーターJSON形式ガイド

## 概要

このドキュメントでは、新しいJSON形式のインジケーター設定について説明します。従来のパラメータ埋め込み文字列（例：`f"APO_{fast_period}_{slow_period}"`）から、より構造化されたJSON形式への移行を行いました。

## JSON形式の利点

1. **可読性の向上**: パラメータが明確に構造化される
2. **拡張性**: 新しいパラメータの追加が容易
3. **型安全性**: パラメータの型と範囲の検証が可能
4. **国際化対応**: パラメータ名の多言語対応が可能

## 基本構造

### JSON形式
```json
{
  "indicator": "RSI",
  "parameters": {
    "period": 14
  }
}
```

### 複数パラメータの例
```json
{
  "indicator": "APO",
  "parameters": {
    "fast_period": 12,
    "slow_period": 26,
    "matype": 0
  }
}
```

## 後方互換性

既存のレガシー形式（文字列）も引き続きサポートされます：

- `"RSI_14"` → `{"indicator": "RSI", "parameters": {"period": 14}}`
- `"APO_12_26"` → `{"indicator": "APO", "parameters": {"fast_period": 12, "slow_period": 26}}`

## 使用方法

### 設定の取得
```python
from app.core.services.indicators.config import indicator_registry

# インジケーター設定の取得
config = indicator_registry.get("RSI")
if config:
    print(config.parameters)
```

### JSON形式の名前生成
```python
from app.core.services.indicators.config import indicator_registry

# JSON形式での名前生成
json_name = indicator_registry.generate_json_name("RSI", {"period": 14})
print(json_name)  # {"indicator": "RSI", "parameters": {"period": 14}}
```

### レガシー形式の名前生成
```python
from app.core.services.indicators.config import indicator_registry

# レガシー形式での名前生成
legacy_name = indicator_registry.generate_legacy_name("RSI", {"period": 14})
print(legacy_name)  # "RSI_14"
```

### 互換性管理
```python
from app.core.services.indicators.config import compatibility_manager

# 互換性モードの有効化
compatibility_manager.enable_compatibility_mode()

# レガシー形式の解決
resolved = compatibility_manager.resolve_indicator_name("RSI_14")
print(resolved)  # {"indicator": "RSI", "parameters": {"period": 14}}
```

## サポートされているインジケーター

### モメンタム系
- **RSI**: `{"indicator": "RSI", "parameters": {"period": 14}}`
- **APO**: `{"indicator": "APO", "parameters": {"fast_period": 12, "slow_period": 26, "matype": 0}}`
- **PPO**: `{"indicator": "PPO", "parameters": {"fast_period": 12, "slow_period": 26, "matype": 0}}`
- **MACD**: `{"indicator": "MACD", "parameters": {"fast_period": 12, "slow_period": 26, "signal_period": 9}}`

### トレンド系
- **SMA**: `{"indicator": "SMA", "parameters": {"period": 14}}`
- **EMA**: `{"indicator": "EMA", "parameters": {"period": 14}}`

### ボラティリティ系
- **ATR**: `{"indicator": "ATR", "parameters": {"period": 14}}`
- **BB**: `{"indicator": "BB", "parameters": {"period": 20, "std_dev": 2.0}}`

### 出来高系
- **OBV**: `{"indicator": "OBV", "parameters": {}}`
- **ADOSC**: `{"indicator": "ADOSC", "parameters": {"fast_period": 3, "slow_period": 10}}`

## アダプターでの使用

### 新しい形式での計算
```python
from app.core.services.indicators.adapters.momentum_adapter import MomentumAdapter

# RSI計算（自動的にJSON形式対応の名前が生成される）
rsi_result = MomentumAdapter.rsi(data, period=14)
print(rsi_result.name)  # "RSI_14" (互換性モード有効時)

# APO計算
apo_result = MomentumAdapter.apo(data, fast_period=12, slow_period=26, matype=0)
print(apo_result.name)  # "APO_12_26" (互換性モード有効時)
```

## 設定の拡張

新しいインジケーターを追加する場合：

```python
from app.core.services.indicators.config import (
    IndicatorConfig, 
    ParameterConfig, 
    IndicatorResultType,
    indicator_registry
)

# 新しいインジケーター設定
new_config = IndicatorConfig(
    indicator_name="CUSTOM",
    required_data=["close"],
    result_type=IndicatorResultType.SINGLE,
    legacy_name_format="{indicator}_{period}"
)

# パラメータの追加
new_config.add_parameter(ParameterConfig(
    name="period",
    default_value=20,
    min_value=2,
    max_value=100,
    description="計算期間"
))

# レジストリに登録
indicator_registry.register(new_config)
```

## 移行ガイド

### 段階的移行
1. **フェーズ1**: 互換性モードを有効にして既存コードを維持
2. **フェーズ2**: 新しいコードでJSON形式を使用
3. **フェーズ3**: 既存コードを段階的にJSON形式に移行
4. **フェーズ4**: レガシー形式のサポートを段階的に削除

### 移行チェックリスト
- [ ] 互換性モードが有効になっていることを確認
- [ ] 新しいインジケーター設定が正しく登録されていることを確認
- [ ] 既存のテストが正常に動作することを確認
- [ ] 新しいJSON形式でのテストを追加
- [ ] ドキュメントの更新

## トラブルシューティング

### よくある問題

1. **設定が見つからない**
   ```python
   config = indicator_registry.get("UNKNOWN")
   if config is None:
       print("インジケーター設定が見つかりません")
   ```

2. **レガシー形式の解析失敗**
   ```python
   from app.core.services.indicators.config import migrator
   
   parsed = migrator.parse_legacy_name("INVALID_FORMAT")
   if parsed is None:
       print("レガシー形式の解析に失敗しました")
   ```

3. **互換性モード無効時のエラー**
   ```python
   from app.core.services.indicators.config import compatibility_manager
   
   try:
       result = compatibility_manager.resolve_indicator_name("RSI_14")
   except ValueError as e:
       print(f"互換性モードが無効です: {e}")
   ```

## 今後の予定

- [ ] 全インジケーターのJSON形式対応完了
- [ ] フロントエンドでのJSON形式サポート
- [ ] パフォーマンス最適化
- [ ] 追加のバリデーション機能
