# 指標設定分散問題移行計画書

## 文書情報
- **作成日**: 2025年3月
- **作成者**: 指標設定移行プロジェクトチーム
- **バージョン**: 1.0
- **ステータス**: 設計フェーズ

## 目次
1. [現在の問題点の詳細分析](#問題分析)
2. [YAML駆動型アーキテクチャの設計](#アーキテクチャ設計)
3. [段階的移行計画](#移行計画)
4. [依存関係マップ](#依存関係マップ)
5. [リスクと解決策](#リスク解決策)
6. [テスト戦略](#テスト戦略)
7. [影響を受けるコンポーネント分析](#影響を受けるコンポーネント分析)

---

<a name="問題分析"></a>
## 1. 現在の問題点の詳細分析

### 分散構造の現状

オートストラテジーシステムでは指標設定が以下の5つのファイルに分散されており、保守性を大きく低下させています：

#### 1. `constants.py` (823行)
- **役割**: カテゴリ別指標リストの管理
```python
TREND_INDICATORS = [
    "SMA", "EMA", "WMA", ..., "CWMA"
]
MOMENTUM_INDICATORS = [
    "RSI", "MACD", ..., "CTI"
]
VOLATILITY_INDICATORS = [...],
VOLUME_INDICATORS = [...]
```

#### 2. `indicator_definitions.py` (2,503行)
- **役割**: 各指標の詳細設定（登録処理）
```python
# CWMA配置例
cwma_config = IndicatorConfig(
    indicator_name="CWMA",
    adapter_function=TrendIndicators.cwma,
    required_data=["close"],
    result_type=IndicatorResultType.SINGLE,
    scale_type=IndicatorScaleType.PRICE_RATIO,
    category="trend",
)
```

#### 3. `indicator_config.py` (クラス定義点)
- **役割**: IndicatorConfigクラス構造定義

#### 4. `unified_config.py` (設定統合点)
- **役割**: システム設定の統合ハブ

#### 5. YAML設定ファイル群 (存在点散逸)
- **役割**: 部分的設定活用（データソース依存）
- **問題**: 場所の一貫性なく、再利用性低下

### 具体的な保守性問題

1. **CWMA指標追加事例 (問題顕在化)**
   - 前提: CWMA (Central Weighted Moving Average) の新規追加
   - 必要修正5箇所:
     - `constants.py`: TREND_INDICATORS リスト更新
     - `indicator_definitions.py`: Adapter関数バインド・設定
     - `indicator_config.py`: 構造クラス最定義時補助
     - `unified_config.py`: 統合参照追加
     - YAMLファイル: 適切属性選択
   - **所要時間**: 各箇所での류戸解答時間発生 → 全体作業コスト増大

2. **構造的欠陥 (長期影響)**
   - **情報散逸**: 単一指標設定が多階層分散
   - **依存複雑化**: ファイル間連携高度要請
   - **エラー繁発性**: 一箇所修正波及全域検証義務
   - **汎用性不足**: 類似ロジック新規策定時コピー多重発

3. **同期問題 (constants.pyとindicator_definitions.pyの複製指標リスト問題)**
   - **問題**: カテゴリ別指標リストがconstants.pyとindicator_definitions.pyで重複定義されているため、一方の変更が他方に反映されない
   - **影響**: 指標追加時における見逃しや不整合発生、システム全体での信頼性低下
   - **具体例**: 新規指標"CWMA"の追加時に、constants.pyのTREND_INDICATORSにのみ追加され、indicator_definitions.pyの登録処理で漏れが発生
   - **長期影響**: メンテナンスコスト増大とバグの潜在化

---

<a name="アーキテクチャ設計"></a>
## 2. YAML駆動型アーキテクチャの設計

### 目標アーキテクチャ概要

集中管理配置図:
```
YAMLセントラル設定 → 動的ローダー → 統合レジストリ
                           ↓
                      システムアブリケーション
```

主要利点:
- 設定情報集中化: 全指標設定1YAMLファイル
- 動的柔軟対応: 実行時設定更新・新規指標追加
- 保守簡便化: 設定変更最小限度ファイル編集

### YAML設定ファイル構造設計

#### 1. 主設定ファイル: `backend/app/config/indicators.yaml`

```yaml
# indicators.yaml
metadata:
  version: "1.0"
  last_updated: "2025-12-03"
  description: "Unified Technical Indicators Configuration"

# カテゴリ別指標グループ
categories:
  trend:
    description: "トレンド指標群"
    indicators: ["SMA", "EMA", "CWMA", ...]

  momentum:
    description: "モメンタム指標群"
    indicators: ["RSI", "MACD", "CTI", ...]

  volatility:
    description: "ボラティリティ指標群"
    indicators: ["ATR", "BBANDS", ...]

  volume:
    description: "出来高指標群"
    indicators: ["AD", "OBV", ...]

# 個別指標設定
indicators:
  SMA:
    name: "SMA"
    category: "trend"
    technical_function: "TrendIndicators.sma"
    data_requirements: ["close"]
    result_type: "single"
    scale_type: "price_ratio"
    parameters:
      period:
        type: "integer"
        default: 20
        min: 2
        max: 200
        description: "SMA期間"

  CWMA:  # 新規指標例
    name: "CWMA"
    category: "trend"
    technical_function: "TrendIndicators.cwma"
    data_requirements: ["close"]
    result_type: "single"
    scale_type: "price_ratio"
    parameters:
      length:
        type: "integer"
        default: 10
        min: 2
        max: 200
        description: "CWMA計算期間"
```

#### 2. ローダー構造設計

##### YAML Indicator Loaderクラス:

```python
class YamlIndicatorLoader:
    def __init__(self, config_path: str):
        self.config_path = config_path
        self._config_cache = None

    def load_indicators(self) -> Dict[str, IndicatorConfig]:
        """YAMLから指標設定を読み込み"""
        pass

    def validate_config(self, config: Dict) -> bool:
        """設定妥当性検証"""
        pass

    def update_registry(self) -> None:
        """indicator_registry へ設定反映"""
        pass
```

#### 3. Registry更新アーキテクチャ

##### 更新後IndicatorRegistry構造:

```python
class ConfigDrivenIndicatorRegistry(IndicatorConfigRegistry):
    """YAML設定駆動型レジストリ"""

    def load_from_yaml(self, config_path: str) -> None:
        """YAMLからの動的設定読み込み"""
        loader = YamlIndicatorLoader(config_path)
        configs = loader.load_indicators()

        for indicator_name, config in configs.items():
            self.register(config)

    def reload_configurations(self) -> None:
        """実行時設定再読込み(開発・保守用途)"""
        pass
```

---

<a name="移行計画"></a>
## 3. 段階的移行計画

### 移行フェーズ全体計画

```
Phase 1 (設計・プロトタイプ作成) → Phase 2 (設定移行) → Phase 3 (テスト)
```

### Phase 1: YAMLプロトタイプ開発 (1-2週間)
**目的**: 新アーキテクチャの検証・基礎実装

#### 作業項目:
1. **YAML設定ファイル作成**
    ```
    touch backend/app/config/indicators.yaml
    ```

2. **YamlIndicatorLoaderクラスの実装**
   ```
   cd backend/app/services/indicators/config
   touch yaml_indicator_loader.py
   ```

3. **ConfigDrivenIndicatorRegistry実装**
   ```
   # indicator_config.py 拡張
   vim indicator_config.py
   ```

#### 検証方法:
- YAMLパーサー機能確認
- 基本指標5種設定読込み成功検証
- Registry更新正確性確認

### Phase 2: 設定移行実装 (3-4週間)
**目的**: 全指標設定のYAML移行・既存コード適応

#### 作業項目:
1. **全指標YAML移行**
   ```
   python scripts/migrate_indicators_to_yaml.py
   ```

2. **indicator_definitions.py簡素化**
   ```
   # 手動設定をYAML読込みに移行
   vim indicator_definitions.py
   ```

3. **constants.py の動的生成処理実装**
   ```
   vim constants.py  # YAMLからカテゴリリスト生成
   ```

4. **不要ファイル削除**
   **削除対象ファイル:**
   - `constants.py`: TREND_INDICATORSリストなどが重複しているため、YAML駆動型に移行後は削除 (823行)
   - `indicator_definitions.py`: 手動設定がYAML読込みに移行されるため、全ファイル削除可能 (2,503行)
   - `indicator_config.py`: 構造クラスが新規アーキテクチャで不要になるため削除

   **削除理由と注意点:**
   - `constants.py`: 重複定義のリストがYAML集中管理なしになるため、削除対象。注意: YAMLからの動的生成処理が安定稼働を確認後の削除のみ
   - `indicator_definitions.py`: 全設定手動記述がYAML駆動に移行完了後に削除可能。注意: 削除前にバックアップ確実に作成
   - `indicator_config.py`: 新アーキテクチャで構造クラスが冗長となるため削除。注意: 依存コンポーネントへの移行影響を確認後

   **削除タイミング:**
   - 全指標設定のYAML移行完了後、各ファイルの移行確認テスト通過後
   - 順序: `constants.py` → `indicator_config.py` → `indicator_definitions.py`

   **バックアップ戦略:**
   ```
   # バックアップ作成
   cd backend/app/services/indicators
   cp constants.py constants.py.backup.$(date +%Y%m%d)
   cp indicator_definitions.py indicator_definitions.py.backup.$(date +%Y%m%d)
   cp indicator_config.py indicator_config.py.backup.$(date +%Y%m%d)

   # 削除実行
   rm constants.py indicator_definitions.py indicator_config.py

   # 復元（緊急時）
   cp constants.py.backup.$(date +%Y%m%d) constants.py
   cp indicator_definitions.py.backup.$(date +%Y%m%d) indicator_definitions.py
   cp indicator_config.py.backup.$(date +%Y%m%d) indicator_config.py
   ```

   **削除検証:**
   - 削除前: 全テスト実行確認
   - 削除後: システム起動テスト・主要機能検証
   - 復元テスト: バックアップからのリストア実行確認

#### 検証方法:
- 全指標設定読込み確認
- 実行時Registry更新検証
- 既存戦略互換性テスト

### Phase 3: テスト・最適化 (1-2週間)
**目的**: 移行品質保証・性能最適化

#### 作業項目:
1. **回帰テスト実行**
2. **負荷テスト実施**
3. **メモリ使用量最適化**

<a name="依存関係マップ"></a>
## 4. 依存関係マップ

### 起動時依存関係

```
メインアプリケーション (main.py)
     ↓
UnifiedConfig (unified_config.py)
     ↓
YamlIndicatorLoader (yaml_indicator_loader.py)
     ↓
indicator_definitions.py (簡素化版)
     ↓
indicator_registry (更新後実装)
```

### ランタイム依存関係

```
YAML設定ファイル
     ↓
動的設定読込み
     ↓
IndicatorConfigRegistry
     ↓
個別指標生成・条件生成・戦略実行
```

### 相互依存関係マトリックス

| コンポーネント | constants.py | indicator_definitions.py |indicator_config.py | unified_config.py |
|---------------|-------------|-------------------------|-------------------|------------------|
| YamlIndicatorLoader | カテゴリ参照 | 簡素化・活用 | 設定更新 | 統合連携 |
| 条件生成器 | カテゴリ活用 | 設定読取 | 構造利用 | - |
| 戦略実行器 | - | 指標生成 | - | 設定活用 |
| parameter_manager.py | PARAMETER_MAPPINGS重複統合 | - | YAML移行連携 | - |
| data_validation.py | - | 指標最小長統合 | - | YAML統合連携 |
| indicator_orchestrator.py | - | - | IndicatorConfig削除対応 | YAML駆動型移行連携 |

### 外部API依存

```
PANDAS_TAライブラリ
     ↓
TrendIndicators, MomentumIndicators, ...
     ↓
indicator_definitions.py
     ↓
動的関数バインド
```

---

<a name="リスク解決策"></a>
## 5. リスクと解決策

### High Risk項目 (発生確率: 高 / 影響: 大)

#### 1. 設定検証エラー (実行時クラッシュ発生要因)
**影響**: システム起動失敗・運用停止
**解決策**:
- YAMLスキーマ検証実装
- 起動時完全性チェック
- フォールバック設定保持

#### 2. 指標計算不整合 (予測値乖離)
**影響**: 戦略性能低下・誤信号生成
**解決策**:
- 二重計算検証システム
- 比較テスト完全実施
- 段階的デプロイ方針

### Medium Risk項目 (発生確率: 中 / 影響: 中)

#### 1. パフォーマンス低下 (メモリ・CPU負荷増大)
**影響**: 応答遅延・処理能力低下
**解決策**:
- キャッシュ機構導入
- 遅延読み込み実装
- 性能監視強化

#### 2. 互換性破壊 (既存戦略無効化)
**影響**: 一部機能使用不可
**解決策**:
- ライトスルーモード実装
- 互換性維持フラグ利用
- 段階的移行方式

#### 3. 追加ファイル変更リスク
**影響**: 移行作業中に影響を受けるコンポーネントの変更により生じる予期せぬ動作不良や統合時の問題発生
**解決策**:
- 影響分析の事前実施と変更影響範囲の明確化
- 単体テストの強化と統合テストの多重実施
- ロールバック計画の準備とバックアップ体制の確保

### Low Risk項目 (発生確率: 低 / 影響: 小)

#### 1. 人作業ミス (YAML記法誤り)
**影響**: 設定読込み失敗
**解決策**:
- YAML文法検証ツール導入
- 自動フォーマッター適用
- テンプレート設定提供

### リスク緩和総合戦略

#### 1. テスト重視アプローチ
```
リスク評価 → 多層テスト実装 → 継続的インテグレーション
                  ↓
           自動検証強化 → 品質保証体制確立
```

#### 2. 段階的移行パターン
```
検証ビルド → カナリアデプロイ → 全体移行 → 監視強化
       ↓               ↓              ↓          ↓
   単体テスト     トラヒック分割  機能テスト  アラート監視
```

---

<a name="テスト戦略"></a>
## 6. テスト戦略

### 多層テスト構造

#### 1. Unit Tests (単体テスト - Phase毎実施)
```python
# test_yaml_indicator_loader.py
def test_yaml_parsing():
    """YAML設定読込みテスト"""
    loader = YamlIndicatorLoader("backend/app/config/indicators.yaml")
    configs = loader.load_indicators()
    assert "SMA" in configs
    assert configs["SMA"].category == "trend"

def test_indicator_registration():
    """指標登録テスト"""
    registry = ConfigDrivenIndicatorRegistry()
    registry.load_from_yaml("backend/app/config/indicators.yaml")
    assert registry.list_indicators() is not None
```

#### 2. Integration Tests (統合テスト - Phase 3重点)
```python
def test_end_to_end_indicator_generation():
    """指標生成統合テスト"""
    # YAMLから設定読込み
    # 指標インスタンス生成
    # 計算実行・検証
    pass

def test_strategy_compatibility():
    """戦略互換性テスト"""
    # 既存戦略読み込み
    # 新旧両システムでの実行
    # 結果比較・乖離検知
    pass
```

#### 3. Performance Tests (性能テスト)
```python
def test_indicators_loading_performance():
    """指標読込み性能テスト"""
    # 測定: YAML⇒Registry設定時間
    # 閾値: 既存比±10%以内
    pass

def test_runtime_indicator_calculation():
    """実行時指標計算テスト"""
    # 測定: 単一指標・複合指標計算時間
    pass

def test_load_test_indicators():
    """ロードテスト: 指標生成機能の応答負荷テスト"""
    # 測定: 高負荷時（大量のリクエスト同時処理）での指標生成時間と失敗率
    # シナリオ: 最大同時接続数での指標計算実行
    # 閾値: 平均応答時間5秒以内、失敗率1%以下
    pass

def test_memory_usage_evaluation():
    """メモリ使用量評価テスト"""
    # 測定: 全指標設定読込み時のメモリ消費量とリーク検知
    # 評価点: メモリピーク使用量 baseline 比 ±20%以内
    # 継続監視: 長時間運用時のメモリリーク有無
    pass
```

### テスト実行シナリオ

#### Phase 1検証:
```bash
python -m pytest tests/unit/test_yaml_loader.py -v
python -m pytest tests/integration/test_basic_indicators.py
```

#### Phase 2検証:
```bash
python scripts/test_migration_compatibility.py
python -m pytest tests/system/test_full_integration.py
```

#### Phase 3品質保証:
```bash
python scripts/performance_baseline.py --baseline
python scripts/load_test_indicators.py --duration=300
```

### 継続的テスト体制

#### 1. CI/CD Pipeline統合
```yaml
# .github/workflows/test.yaml
- name: Run Indicator Tests
  run: |
    python -m pytest tests/indicators/ -v --cov=backend.app.services.indicators
    python scripts/test_migration_coverage.py
```

#### 2. カバレッジ目標
- **目標カバレッジ**: branches=85%, lines=90%
- **重要コンポーネント**: 設定読込み・指標登録・計算処理
- **監視対象**: エッジケース・エラー処理

### テストデータ戦略

#### 1. Test Data Generator
```python
# テスト用指標設定自動生成
def generate_test_indicator_configs(count=50):
    """テスト用指標設定生成"""
    templates = ["trend", "momentum", "volatility"]
    # YAML形式テストデータ生成
    pass
```

#### 2. 現実性検証データ
- **価格データ**: 過去1年分のBTC/USDT時系列
- **指標数**: 主要10指標 + 新規5指標
- **シナリオ数**: 50以上の取引条件パターン

---

## 実行コマンド一覧

### 設定準備
```bash
mkdir -p backend/docs
```

### 開発環境構築
```bash
cd backend/app/services/indicators/config
touch yaml_indicator_loader.py
cd ../../../config
touch indicators.yaml
```

### テスト実行
```bash
python -m pytest tests/indicators/ -v
python scripts/test_migration_compatibility.py
```

---

## 結論

この移行計画により、オートストラテジーシステムの指標設定分散問題を解決し、以下を実現します：

1. **保守性向上**: 単一YAMLでの集中管理
2. **拡張性確保**: 新指標追加の簡易化
3. **リスク低減**: 段階的移行による安全保障
4. **運用効率**: 動的設定更新による柔軟性
5. **追加推奨事項**: 開発者トレーニングの実施と運用監視体制の強化

想定される効果:
- 新指標追加コスト: 5箇所→1箇所 (80%削減)
- 修正時間: 30分→5分 (83%改善)
- 全体保守性: 大幅向上

### 追加推奨事項

1. **開発者トレーニング**: YAML設定管理と動的ローダー利用に関するトレーニングを実施し、全開発メンバーのスキルレベル統一
2. **運用監視体制**: 移行後のパフォーマンスと安定性を継続監視するための監視ダッシュボード構築とアラート設定

**承認状況**: ☐ 要承認 ☐ 承認済
**担当者**: [プロジェクトチーム]
**次回レビュ**: [日付]

---

<a name="影響を受けるコンポーネント分析"></a>
## 7. 影響を受けるコンポーネント分析

### 移行対象ファイルの詳細分析

#### 1. parameter_manager.py
**影響度**: 高
**変更内容**:

PARAMETER_MAPPINGS重複統合 - YAML移行
- 既存のPARAMETER_MAPPINGS辞書の重複定義をYAML中心化へ移行
- YAML設定からの動的パラメータ読込み処理の実装

**対応計画**:
- YAML設定からPARAMETER_MAPPINGSを動的生成する処理の追加
- 既存パラメータの保存性を確保した対応実装
- 移行テスト実施による影響確認

#### 2. data_validation.py
**影響度**: 中
**変更内容**:

指標最小長定義 - YAML統合
- 指標計算時の最小データ長定義をYAML中心化
- 複数の characteristically validationルールの統合

**対応計画**:
- YAML設定からの最小長定義読込み処理の実装
- 既存validationルールの移行・統合
- 互換性維持のためのテスト強化

#### 3. indicator_orchestrator.py
**影響度**: 高
**変更内容**:

IndicatorConfig削除 - YAML駆動型移行
- IndicatorConfigクラスの使用をYAML駆動型設定へ置き換え
- 動的設定読み込みに基づく新たな処理フローの実装

**対応計画**:
- YAML設定からの動的設定読み込み処理の実装
- IndicatorConfig依存コードのYAML駆動型置き換え
- 段階的テストによる移行確認

### 総合的対応計画
- 影響度別優先順位付け: indicator_orchestrator.py > parameter_manager.py > data_validation.py
- 各ファイルの変更影響をテストで事前検証
- 移行段階でのバックアップ・ロールバック体制確保

このように、3つのファイルの具体的な影響度と対応計画を明確に記載した。移行計画書の拡張を実現した。