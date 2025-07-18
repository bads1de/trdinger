# リファクタリング候補一覧

## 概要
コードベース全体を調査した結果、以下のリファクタリング候補を特定しました。これらは責務の分離、コードの重複削除、構造化の改善を目的としています。

## 🔴 高優先度（重要度：高）

### 1. APIルーター内のビジネスロジック残存問題

#### 1.1 external_market.py
**問題**: データ収集ロジックが直接記述されている
```python
# 問題のあるコード（L214-230）
async with ExternalMarketDataCollector() as collector:
    result = await collector.collect_incremental_external_market_data(
        symbols=symbols, db_session=db
    )
```
**解決策**: ExternalMarketOrchestrationServiceの作成

#### 1.2 funding_rates.py & open_interest.py
**問題**: 収集ロジックとエラーハンドリングが混在
```python
# 問題のあるコード（L147-175）
service = BybitFundingRateService()
repository = FundingRateRepository(db)
result = await service.fetch_and_save_funding_rate_data(...)
```
**解決策**: FundingRateOrchestrationService、OpenInterestOrchestrationServiceの作成

#### 1.3 data_reset.py
**問題**: データ削除ロジックが直接記述されている
**解決策**: DataManagementOrchestrationServiceの作成

### 2. エラーハンドリングの不統一

#### 2.1 try-catch vs UnifiedErrorHandler
**問題**: 一部のAPIで古いtry-catch方式が残存
```python
# 問題のあるコード（external_market.py L214）
try:
    async with ExternalMarketDataCollector() as collector:
        # ...
except Exception as e:
    raise HTTPException(...)
```
**解決策**: 全APIで`UnifiedErrorHandler.safe_execute_async`に統一

#### 2.2 open_interest.py の混在パターン
**問題**: 同一ファイル内でエラーハンドリング方式が混在
- L134-162: try-catch方式
- L163: UnifiedErrorHandler方式

### 3. scriptsディレクトリの重複コード

#### 3.1 データベース接続・初期化パターンの重複
**重複箇所**:
- `direct_db_check.py` L19-20
- `verify_persistence.py` L130-131
- `check_database_status.py` L42-47
- `check_latest_results.py` 類似パターン

**解決策**: `scripts/utils/db_utils.py`の作成

#### 3.2 エラーハンドリングパターンの重複
**重複箇所**:
- `direct_db_check.py` L142-146
- `verify_persistence.py` L171-174
- `check_database_status.py` 類似パターン

**解決策**: 共通エラーハンドリング関数の作成

#### 3.3 ログ出力パターンの重複
**重複箇所**:
- 統計情報出力パターン
- 進捗表示パターン
- 結果表示パターン

## 🟡 中優先度（重要度：中）

### 4. ディレクトリ構造の改善

#### 4.1 database/repository.py の役割不明確
**問題**: レガシー互換性のための再エクスポートファイル
**解決策**: 段階的な廃止とインポートパスの統一

#### 4.2 scriptsディレクトリの構造化不足
**問題**: 
- ユーティリティ関数が各ファイルに散在
- 共通処理の重複
- 命名規則の不統一

**解決策**: 
```
scripts/
├── utils/
│   ├── db_utils.py          # データベース関連ユーティリティ
│   ├── logging_utils.py     # ログ関連ユーティリティ
│   └── analysis_utils.py    # 分析関連ユーティリティ
├── database/
│   ├── init_database.py
│   ├── check_database_status.py
│   └── verify_persistence.py
└── analysis/
    ├── direct_db_check.py
    ├── check_latest_results.py
    └── debug_*.py
```

### 5. 命名規則の不統一

#### 5.1 APIルーターのprefix不統一
**問題**: 
- 一部は`prefix="/api"`
- 一部はprefixなし
- 一部は独自prefix

**解決策**: 統一的なprefix規則の確立

#### 5.2 サービスクラス命名の不統一
**問題**:
- `*Service` vs `*OrchestrationService`
- 責務の境界が不明確

## 🟢 低優先度（重要度：低）

### 6. コードの最適化

#### 6.1 インポートの最適化
**問題**: 未使用インポートや重複インポートが散在

#### 6.2 型ヒントの不足
**問題**: 一部のファイルで型ヒントが不完全

#### 6.3 docstringの不統一
**問題**: docstringの形式が統一されていない

## 📋 実装計画

### Phase 1: 高優先度の解決（推定工数: 2-3日）
1. ExternalMarketOrchestrationService作成
2. FundingRateOrchestrationService作成  
3. OpenInterestOrchestrationService作成
4. DataManagementOrchestrationService作成
5. エラーハンドリングの統一

### Phase 2: 中優先度の解決（推定工数: 1-2日）
1. scriptsディレクトリの構造化
2. 共通ユーティリティの作成
3. 命名規則の統一

### Phase 3: 低優先度の解決（推定工数: 1日）
1. コードの最適化
2. 型ヒント・docstringの改善

## 🎯 期待される効果

### 短期的効果
- APIルーターの責務明確化
- エラーハンドリングの統一
- コードの重複削除

### 長期的効果
- 保守性の向上
- テスタビリティの向上
- 新機能追加の効率化
- チーム開発の効率化

## 📝 注意事項

1. **段階的実装**: 既存機能への影響を最小化するため段階的に実装
2. **テスト重視**: 各段階でテストを実施し、機能の正常性を確認
3. **後方互換性**: 可能な限り既存のインターフェースを維持
4. **ドキュメント更新**: リファクタリング後はドキュメントも更新
