# Auto-Strategy 技術オンリー拡張 レポート (第一弾)

日付: 2025-08-11

## 変更ハイライト
- OR 条件グループ対応: A AND (B OR C) 形式の戦略生成・評価・シリアライズ
- 指標初期化フォールバック: 失敗時に SMA(period) → RSI(14)
- 品質メトリクス/選別: スコアリングと最低品質判定、成立率>=60%の二次選別パイプライン

## 追加/更新ファイル
- app/services/auto_strategy/models/condition_group.py
- evaluators/condition_evaluator.py (ORグループ評価対応)
- generators/smart_condition_generator.py (ORグループ生成)
- models/gene_validation.py (ORグループ検証)
- models/gene_serialization.py (ORグループの辞書/JSON対応)
- factories/strategy_factory.py (指標フォールバック)
- utils/metrics.py (スコア/閾値)
- utils/selector.py (二次選別とランキング)

## 追加テスト
- test_or_group_entry.py: OR導入による成立性の低下がないことのスモーク
- test_edge_market_conditions.py: 急騰/急落/レンジで動作できること
- test_quality_selection.py: スコア関数・閾値のモック検証
- test_quality_selector_pipeline.py: 成立率→品質→スコアの一連の選別検証
- test_quality_metrics_brushup.py: スコア単調性/閾値の整合性

## テスト結果
- 実行コマンド: `pytest -q`
- 結果: 380 passed, 3 skipped（多数の既知のWarningあり）
- 回帰: なし（既存テストを含めてグリーン）

## 新たに可能な戦略例
- トレンド追随 + オシレーター確認: Close > SMA ∧ (RSI > 55 ∨ MACD > 0)
- ボリンジャーブレイクアウト: Close > BB_Upper ∧ (MACD > 0 ∨ SMA上向き)
- 逆張り + モメンタム抑制: Close < BB_Lower ∧ (RSI < 30 ∨ Close < Open)
- プルバック押し目: Close > EMA ∧ (Close < BB_Middle ∨ RSIが50上抜け)
- ショート対称系: Close < SMA ∧ (RSI < 45 ∨ MACD < 0)

## 既知事項/今後
- スコア重み・品質閾値は経験的値。将来的にプロファイル/TF別に調整可能に
- 多数戦略生成の性能テスト（時間/メモリプロファイル）を強化予定
- フロント→バック→DB保存のE2Eテスト（品質指標付）を追加予定

以上。

