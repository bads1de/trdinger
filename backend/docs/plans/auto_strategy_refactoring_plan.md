## 方針
- ご指摘の通り、auto_strategy 周りは機能追加と検証強化で複雑性が増しています
- まず全体構造と依存関係を把握し、責務分離と凝集度の観点でボトルネックを特定しました
- その上で、安全に段階的に進められるリファクタ提案とテスト戦略を提示します

## 現状アーキテクチャの概要（要点）
- generators
  - RandomGeneGenerator: 戦略遺伝子全体の生成。TP/SL遺伝子やロング・ショート条件、ポジションサイジング遺伝子の組立も担当
  - SmartConditionGenerator: インジケータ特性に応じた条件（含ORグループ）を生成。price vs trend 補完、オシレータ閾値など複雑
- evaluators
  - ConditionEvaluator: 条件単体/ORグループを評価。右オペランドが文字列（SMA/EMA/MACDなど）の場合の解決ロジックが肥大
- calculators
  - IndicatorCalculator: TechnicalIndicatorService 経由で計算し、Strategy インスタンス上に属性登録（SMA, EMA, MACD_0/1 など）
  - TPSLCalculator, PositionSizingHelper: TP/SL とサイズ計算
- factories
  - StrategyFactory: StrategyGene から backtesting.py 互換の Strategy クラスを動的生成。initで指標を登録、nextで条件を評価しサイジング/TP/SLを適用
- models/utils
  - Condition, ConditionGroup, StrategyGene 各種。ORグループ対応、シリアライズ/バリデーション、品質評価の補助ユーティリティなど

## 問題点（構造的課題）
1) 責務の集約過多と横断的関与
- RandomGeneGenerator が「TP/SL有効化方針」「OR グループ正規化」「デフォルトフォールバック」など戦術レベルの知見を内包し過ぎ
- SmartConditionGenerator に「価格 vs トレンド補完」「レンジ/トレンドのゲーティング」「インジ特性→しきい値化」「OR 候補合成」まで詰め込まれており高凝集だが肥大

2) 評価器のオペランド解決が散逸
- ConditionEvaluator が右オペランドの素名解決、マルチアウトプット（MACD_0/1, BB_0/1/2）など多岐にわたり、IndicatorCalculator 側の命名規則/登録と強い暗黙依存

3) StrategyFactory の next() が多機能化
- 条件評価、TP/SL 計算、サイジング、backtesting の制約吸収などの分岐が集中
- デバッグログや最小エントリ保証などのポリシーも混在

4) 汎用ルールの分散
- 「price vs open フォールバック」「technical_only でのTP/SLバイアス」等が generator 側に散在し、テストや他モード拡張時の影響範囲が読みにくい

5) 命名と仕様の暗黙共有
- 指標名の命名規則（MACD_0/1, BB_0/1/2, KELTNER_1 など）が evaluator と calculator の間で暗黙リンク

## 推奨リファクタリング（段階的・安全策）

### 段階1: 参照テーブル化と責務の明確化（低リスク）
- IndicatorNameResolver（新規小コンポーネント）の導入
  - 役割: 条件（left/right operand）の文字列→Strategy属性名の解決を一元化
  - ConditionEvaluator はこの Resolver に依存して値取得のみ担う
  - 命名規則（MACD/MACDEXTの0/1、BBのUpper/Middle/Lower、KELTNERのMiddleなど）をテーブル化
- ThresholdPolicy（新規）
  - 役割: RSI/CCI/ADXなど指標種別→プロファイル別（aggressive/normal/conservative）閾値計算を一元化
  - SmartConditionGenerator はポリシーを呼び出して閾値を得るだけに簡素化
- PriceVsTrendPolicy（新規）
  - 役割: price vs trend 補完で使用する「右オペランド候補を gene に含まれるトレンド系から安全選定」ロジックを共通化

### 段階2: ジェネレータ層の分割（中リスク）
- ConditionAssembly（新規）
  - 役割: 生成した素条件を OR/AND 構造に束ねる純粋ユーティリティ
  - RandomGeneGenerator の OR 正規化や fallback の注入はここに移管
- EntryPolicy（新規）
  - 役割: long/short の最低保証（price vs open など）を定義。モード別（technical_only, ml_only, mixed）での基本ルールを切替

### 段階3: StrategyFactory のスリム化（中〜高リスクだが効果大）
- OrderExecutionPolicy（新規）
  - 役割: TP/SL の優先順位、サイズ調整（backtesting制約対応）、買付可能性チェック等の執行ポリシーを分離
- StrategyFactory の next() は
  - 条件の真偽→ポジション方向決定→OrderExecutionPolicy に委譲するだけに

### 段階4: 命名規則とテストの強化（低リスク）
- IndicatorCalculator が設定する属性名の仕様書（ドキュメント）化とテスト追加
- Resolver のユニットテスト（例: 'SMA'→存在時は SMA、'MACD'→MACD_0、'BB_Middle_20'→BB_1 など）
- ConditionEvaluator は数値比較のみをテスト（名前解決は別モジュールの責務）

## 提案するディレクトリ/クラス追加（例）

- backend/app/services/auto_strategy/core/
  - indicator_name_resolver.py
  - threshold_policy.py
  - price_trend_policy.py
  - condition_assembly.py
  - entry_policy.py
  - order_execution_policy.py

## 段階実行プラン

- フェーズ1（小変更・高効果）
  1) IndicatorNameResolver を導入して ConditionEvaluator の文字列解決ロジックを移管
  2) ThresholdPolicy を導入して SmartConditionGenerator の閾値決定を集約
  3) 既存テストスイートを全実行（安全確認）

- フェーズ2（中規模）
  1) ConditionAssembly/EntryPolicy を導入し、RandomGeneGenerator の OR 正規化/フォールバック注入を移管
  2) SmartConditionGenerator の price vs trend 補完を PriceVsTrendPolicy に移管
  3) テスト更新（OR構造/price vs trend/thresholdの固定化に追随）

- フェーズ3（中〜大）
  1) OrderExecutionPolicy を導入し、StrategyFactory.next の執行部分を分離
  2) 回帰テスト（既存のBacktest統合・成功率・品質選別系）

各フェーズで変更範囲は1〜3ファイル程度に抑え、テストで段階確認します。

## 期待効果
- 可読性と保守性の改善：名前解決・閾値決定・構造化・執行が明確なモジュールで再利用可能
- 変更影響の局所化：新規指標や特性追加時に Resolver/Policy 側の追加のみで済み、ジェネレータや評価器を触らない
- テスト容易性：責務ごとにユニットテストを用意でき、回帰バグを早期検知

## リスクと抑制策
- リファクタ時の見落としによる挙動差異
  - 抑制: 既存の tests/auto_strategy/ 下のスイートを都度回し、差分ログでウォッチ
- 命名規則の過不足
  - 抑制: IndicatorCalculator 側の属性命名仕様を小ドキュメント化し、Resolver ユニットテストで網羅

## 次のステップ（提案）
- フェーズ1のスコープで着手（安全・短期）
  - core/indicator_name_resolver.py 追加
  - ConditionEvaluator から解決ロジックを移管（挙動不変）
  - 最小ユニットテスト追加（condition operand resolution）
- 実施後にテストを全実行し、次フェーズへ拡張

このプランで進めて良いですか？それとも対象範囲を更に絞る/広げる等のご希望がありますか。