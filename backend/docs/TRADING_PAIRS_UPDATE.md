# 取引ペア設定更新ドキュメント（実際に確認済み）

## 概要

このドキュメントは、Trdinger システムの取引ペア設定を BTC、ETH、XRP、BNB、SOL の先物とスポットペアに更新した内容を説明します。

**重要**: この設定は実際にBybitで利用可能なペアのみを含んでいます。

## 更新内容

### 対象通貨
- **BTC** (Bitcoin)
- **ETH** (Ethereum) 
- **XRP** (Ripple)
- **BNB** (Binance Coin)
- **SOL** (Solana)

### 対象市場
- **スポット市場**: 現物取引
- **先物市場**: 永続契約（Perpetual Futures）

## サポートされている取引ペア（実際に確認済み）

### Bitcoin (BTC)
| 市場 | シンボル | 説明 |
|------|----------|------|
| スポット | `BTC/USDT` | Bitcoin/USDT スポット |
| 先物 | `BTC/USDT:USDT` | Bitcoin USDT永続契約 |
| 先物 | `BTCUSD` | Bitcoin USD永続契約 |

### Ethereum (ETH)
| 市場 | シンボル | 説明 |
|------|----------|------|
| スポット | `ETH/USDT` | Ethereum/USDT スポット |
| スポット | `ETH/BTC` | Ethereum/Bitcoin ペア |
| 先物 | `ETH/USDT:USDT` | Ethereum USDT永続契約 |
| 先物 | `ETHUSD` | Ethereum USD永続契約 |

### XRP (Ripple)
| 市場 | シンボル | 説明 |
|------|----------|------|
| スポット | `XRP/USDT` | XRP/USDT スポット |
| 先物 | `XRP/USDT:USDT` | XRP USDT永続契約 |

### BNB (Binance Coin)
| 市場 | シンボル | 説明 |
|------|----------|------|
| スポット | `BNB/USDT` | BNB/USDT スポット |
| 先物 | `BNB/USDT:USDT` | BNB USDT永続契約 |

### SOL (Solana)
| 市場 | シンボル | 説明 |
|------|----------|------|
| スポット | `SOL/USDT` | SOL/USDT スポット |
| 先物 | `SOL/USDT:USDT` | SOL USDT永続契約 |

## シンボル正規化

システムは様々な表記形式を自動的に正規化します：

### スポット正規化例
```
BTC-USDT → BTC/USDT
ETH-USDT → ETH/USDT
ETHBTC → ETH/BTC
ETH-BTC → ETH/BTC
```

### 先物正規化例（USDT永続契約）
```
BTCUSDT → BTC/USDT:USDT
ETHUSDT → ETH/USDT:USDT
XRPUSDT → XRP/USDT:USDT
BNBUSDT → BNB/USDT:USDT
SOLUSDT → SOL/USDT:USDT
```

### レガシーUSD永続契約
```
BTCUSD → BTCUSD (そのまま)
ETHUSD → ETHUSD (そのまま)
```

## 設定ファイル

### 更新されたファイル
- `backend/app/config/market_config.py`

### 主要な変更点
1. `SUPPORTED_SYMBOLS` リストを実際に確認済みのペアに更新
2. `SYMBOL_MAPPING` 辞書を正しい形式に修正
3. 確認できないペアを削除

## 使用方法

### 基本的な使用例

```python
from app.config.market_config import MarketDataConfig

# シンボルの検証
is_valid = MarketDataConfig.validate_symbol("BTC/USDT")
print(is_valid)  # True

# シンボルの正規化
normalized = MarketDataConfig.normalize_symbol("BTCUSDT")
print(normalized)  # "BTC/USDT:USDT"

# サポートされているシンボル一覧
symbols = MarketDataConfig.SUPPORTED_SYMBOLS
print(len(symbols))  # 13
```

### データ収集での使用例

```python
from data_collector.collector import DataCollector

collector = DataCollector()

# Bitcoin スポットデータの収集
await collector.collect_historical_data("BTC/USDT", "1d", 30)

# Bitcoin USDT永続契約データの収集
await collector.collect_historical_data("BTC/USDT:USDT", "1h", 100)

# Ethereum スポットデータの収集
await collector.collect_historical_data("ETH/USDT", "4h", 50)

# Solana USDT永続契約データの収集
await collector.collect_historical_data("SOL/USDT:USDT", "1h", 100)
```

## テスト

### テストファイル
- `backend/tests/unit/test_updated_market_config.py`
- `backend/scripts/test_updated_symbols.py`

### テスト実行
```bash
# ユニットテスト
cd backend && python -m pytest tests/unit/test_updated_market_config.py -v

# 統合テスト
cd backend && python scripts/test_updated_symbols.py
```

## 統計情報

- **総シンボル数**: 13 (実際に確認済み)
- **対象通貨数**: 5 (BTC, ETH, XRP, BNB, SOL)
- **スポットペア数**: 6
- **先物ペア数**: 7 (USDT永続契約 5 + USD永続契約 2)
- **サポートされている取引所**: Bybit

## 注意事項

1. **確認済みペア**: この設定は実際にBybitで利用可能なペアのみを含んでいます
2. **API制限**: Bybit API には地域制限がある場合があります
3. **レート制限**: API 呼び出し時はレート制限に注意してください
4. **市場時間**: 先物市場は24時間、スポット市場も基本的に24時間利用可能です
5. **シンボル形式**: CCXTライブラリでの正しいシンボル形式を使用しています

## 確認方法

この設定の各ペアは以下の方法で確認されました：
- TradingViewでのBybitペア確認
- Bybit公式サイトでの取引ペア情報
- 他の取引所での一般的なペア存在確認

## 今後の拡張

必要に応じて以下の通貨ペアの追加を検討できます（事前確認必須）：
- ADA (Cardano)
- DOT (Polkadot)
- AVAX (Avalanche)
- LTC (Litecoin)
- UNI (Uniswap)

## 更新履歴

| 日付 | 変更内容 | 担当者 |
|------|----------|--------|
| 2024-XX-XX | BTC, ETH, XRP, BNB, SOL の先物・スポットペア追加（実際に確認済み） | Trdinger Team |

---

このドキュメントは取引ペア設定の更新内容を説明しています。
質問や問題がある場合は、開発チームまでお問い合わせください。
