# テクニカル指標機能追加プラン

## 概要

既存のfunding rate、open interestと同じ実装パターンに従って、テクニカル指標の計算・保存・表示機能を追加します。

## 目標

- OHLCVデータを基にテクニカル指標を自動計算
- 計算結果をデータベースに保存
- フロントエンドでの表示・操作機能
- 既存機能との一貫したUI/UX

## 実装範囲

### 対象テクニカル指標

**Phase 1（基本指標）**:
- SMA（Simple Moving Average）- 単純移動平均
- EMA（Exponential Moving Average）- 指数移動平均
- RSI（Relative Strength Index）- 相対力指数

**Phase 2（応用指標）**:
- MACD（Moving Average Convergence Divergence）
- ボリンジャーバンド
- ストキャスティクス

### 機能要件

1. **データ計算・保存**
   - OHLCVデータから各指標を計算
   - 計算結果をTimescaleDBに保存
   - 増分計算対応（新しいデータのみ計算）

2. **API エンドポイント**
   - GET `/technical-indicators` - 指標データ取得
   - POST `/technical-indicators/calculate` - 指標計算・保存
   - GET `/technical-indicators/current` - 現在値取得
   - POST `/technical-indicators/bulk-calculate` - 一括計算

3. **フロントエンド機能**
   - 指標データテーブル表示
   - 指標計算ボタン
   - 指標タイプ・パラメータ選択
   - 既存UIとの統合

## 技術仕様

### 依存関係

```python
# 新規追加
pandas-ta>=0.3.14b  # テクニカル指標計算ライブラリ
numpy>=1.21.0       # 数値計算（既存の可能性あり）
```

### データベース設計

**SQLite対応のテーブル設計**:

```python
class TechnicalIndicatorData(Base):
    __tablename__ = "technical_indicator_data"

    id = Column(Integer, primary_key=True, autoincrement=True)
    symbol = Column(String(50), nullable=False, index=True)
    timeframe = Column(String(10), nullable=False, index=True)
    indicator_type = Column(String(20), nullable=False)  # 'SMA', 'EMA', 'RSI', 'MACD'
    period = Column(Integer, nullable=False)             # 期間（20, 14等）
    value = Column(Float, nullable=False)                # メイン値
    signal_value = Column(Float, nullable=True)          # シグナル線（MACD等）
    histogram_value = Column(Float, nullable=True)       # ヒストグラム（MACD等）
    timestamp = Column(DateTime(timezone=True), nullable=False, index=True)
    created_at = Column(DateTime(timezone=True), server_default=func.now())
    updated_at = Column(DateTime(timezone=True), server_default=func.now(), onupdate=func.now())

    # SQLite対応インデックス定義
    __table_args__ = (
        Index("idx_ti_symbol_type_timestamp", "symbol", "indicator_type", "timestamp"),
        Index("idx_ti_timestamp_symbol", "timestamp", "symbol"),
        Index("uq_symbol_type_period_timestamp", "symbol", "indicator_type", "period", "timestamp", unique=True),
    )
```

## 実装計画

### Phase 1: バックエンド基盤（3-4日）

#### 1. 依存関係追加
- [ ] `backend/requirements.txt` に pandas-ta 追加（既存のpandas, numpyを活用）
- [ ] パッケージインストール・動作確認
- [ ] SQLite環境での動作テスト

#### 2. データベースモデル
- [ ] `backend/database/models.py` に `TechnicalIndicatorData` クラス追加
- [ ] マイグレーション実行

#### 3. リポジトリクラス
- [ ] `backend/database/repositories/technical_indicator_repository.py` 作成
  - データ取得・保存メソッド
  - 期間指定クエリ
  - 重複チェック機能

#### 4. サービスクラス
- [ ] `backend/app/core/services/technical_indicator_service.py` 作成
  - SMA, EMA, RSI 計算ロジック
  - OHLCVデータ取得・変換
  - バッチ計算機能

### Phase 2: API エンドポイント（2-3日）

#### 5. APIエンドポイント
- [ ] `backend/app/api/technical_indicators.py` 作成
  - GET `/technical-indicators` - データ取得
  - POST `/technical-indicators/calculate` - 計算・保存
  - GET `/technical-indicators/current` - 現在値
  - POST `/technical-indicators/bulk-calculate` - 一括計算

#### 6. ルーター統合
- [ ] `backend/app/main.py` にルーター追加
- [ ] API動作テスト

### Phase 3: フロントエンド（3-4日）

#### 7. データテーブルコンポーネント
- [ ] `frontend/components/TechnicalIndicatorDataTable.tsx` 作成
  - 指標データ表示
  - ソート・フィルタ機能
  - 既存テーブルと同じデザイン

#### 8. 操作コンポーネント
- [ ] `frontend/components/common/TechnicalIndicatorCalculationButton.tsx` 作成
- [ ] `frontend/components/common/IndicatorTypeSelector.tsx` 作成
- [ ] `frontend/components/common/IndicatorPeriodSelector.tsx` 作成

#### 9. メインページ統合
- [ ] `/data` ページにテクニカル指標タブ追加
- [ ] 既存のOHLCV、FR、OIタブと同じデザインパターン
- [ ] レスポンシブデザイン対応

### Phase 4: テスト・最適化（2日）

#### 10. テスト
- [ ] バックエンドAPIテスト
- [ ] 計算精度テスト
- [ ] フロントエンド表示テスト
- [ ] パフォーマンステスト

#### 11. ドキュメント
- [ ] API仕様書更新
- [ ] README更新

## ファイル構成

### 新規作成ファイル

```
backend/
├── database/repositories/
│   └── technical_indicator_repository.py
├── app/
│   ├── api/
│   │   └── technical_indicators.py
│   └── core/services/
│       └── technical_indicator_service.py
frontend/
├── components/
│   ├── TechnicalIndicatorDataTable.tsx
│   └── common/
│       ├── TechnicalIndicatorCalculationButton.tsx
│       ├── IndicatorTypeSelector.tsx
│       └── IndicatorPeriodSelector.tsx
```

### 既存修正ファイル

```
backend/
├── requirements.txt          # pandas-ta 追加
├── database/models.py        # TechnicalIndicatorData 追加
└── app/main.py              # ルーター追加
frontend/
└── app/data/page.tsx        # テクニカル指標タブ追加
```

## 技術的考慮事項

### パフォーマンス
- 大量データの計算は非同期処理
- バッチサイズの最適化
- SQLite用インデックス設計の最適化
- メモリ効率的なデータ処理

### エラーハンドリング
- 計算エラーの適切な処理
- データ不足時の対応
- API レート制限対応

### 拡張性
- 新しい指標の追加容易性
- パラメータのカスタマイズ
- 複合指標への対応

## 期待される成果

1. **自動化**: OHLCVデータから自動的にテクニカル指標を計算
2. **履歴管理**: 指標の時系列データを保存・表示
3. **リアルタイム**: 最新データでの指標計算
4. **一貫性**: 既存機能と統一されたUI/UX
5. **拡張性**: 将来的な指標追加への対応

## リスク・課題

1. **計算精度**: ライブラリの計算結果の検証が必要
2. **パフォーマンス**: 大量データ処理時の最適化
3. **データ整合性**: OHLCVデータとの同期
4. **UI複雑性**: 多数の指標・パラメータの管理

## 次のステップ

1. 依存関係の追加とテスト環境での動作確認
2. データベースモデルの実装と動作テスト
3. 基本的なSMA計算機能の実装
4. 段階的な機能追加とテスト

---

**総見積もり時間**: 10-13日
**優先度**: 中（既存機能が安定してから実装）
**担当者**: バックエンド・フロントエンド開発者
