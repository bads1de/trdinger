# ポジションサイジングとリスク管理の動的調整 実装計画

## 目的

ボラティリティ（例: ATR や標準偏差）とオープンインタレスト（OI）に基づく動的ポジションサイジングを実装し、Kelly Criterion の変種を活用してリスクを最適化する。また、VaR（Value at Risk）と Expected Shortfall（CVaR）を統合し、ドローダウンを抑制して Sharpe Ratio を最大化する。これにより、既存の`position_sizing_gene`を拡張し、仮想通貨取引の変動性に対応したロバストな戦略を実現する。

## 利点

- **ドローダウン抑制**: 市場のボラティリティが高い時期にポジションサイズを自動調整し、損失を最小限に抑える。
- **Sharpe Ratio 最大化**: Kelly Criterion により期待リターンを最大化しつつ、リスクをコントロール。
- **動的適応**: OI を考慮した流動性調整で、スリッページを防ぎ、実行可能性を向上。

## 既存コードベースの確認

- **PositionSizingGene**: `backend/app/services/auto_strategy/models/position_sizing_gene.py`で定義。現在の方法は`PositionSizingMethod`（FIXED_QUANTITY, FIXED_RATIO など）の静的パラメータ中心。
- **関連ファイル**:
  - `genetic_operators.py`: クロスオーバーとミューテーションで`position_sizing_gene`を扱う。
  - `random_gene_generator.py`: ランダム生成時にデフォルトの固定サイズを使用。
  - `strategy_factory.py`: バックテスト時に`position_sizing_service.calculate_position_size`で使用。
  - `fitness_sharing.py`: 類似度計算で`position_sizing_gene`を考慮。
  - シリアライザー（`dict_converter.py`, `list_encoder.py`など）: シリアライズ/デシリアライズ対応。
- **制限**: 現在の実装は静的。ボラティリティ/OI/VaR を動的に取り込む仕組みなし。バックテスト（`backtesting.py`）と ML（LightGBM/Optuna）との統合が必要。

## 必要な拡張点

- **Optimal F (改良型 Kelly Criterion)**: backtesting.py の optimalF を活用し、Kelly Criterion の改良版として使用。勝率とリスク・リワード比から最適 fraction を計算。
  - 調整: ボラティリティ（ATR）でスケーリング（例: optimal_f \* (1 / ATR)）、OI で流動性調整（例: OI が高いほどサイズ拡大）。
  - 保守的に half-optimal_f を適用し、過度なレバレッジを避ける。Kelly を別途実装する必要なし。
- **VaR/Expected Shortfall 統合**:
  - VaR: 歴史的シミュレーション or パラメトリック（正規分布）で 95%信頼区間の損失推定。
  - Expected Shortfall: VaR を超える損失の平均（テールリスク）。
  - ポジションサイズ = min(optimal_f サイズ, VaR 上限 / 期待損失）。
- **データソース**: OHLCV データ（`collect_ohlcv_data.py`）から ATR/OI を計算。`open_interest_data`フックでリアルタイム取得。
- **GA 統合**: `position_sizing_gene`に optimal_f/VaR パラメータ（例: ボラ調整係数, VaR 信頼水準）を遺伝子として追加。フィットネス関数に Sharpe Ratio とドローダウンを追加。

## 実装ステップ (TDD アプローチ)

1. **[ ] 依存ライブラリ追加**: `requirements.txt`に`scipy`（VaR 計算用）、既存の`numpy/pandas`活用。テストでモック。
2. **[ ] PositionSizingMethod 拡張**: 新しい enum 追加（`OPTIMAL_F_VOLATILITY`, `VAR_ADJUSTED`）。`position_sizing_gene.py`に optimal_f/VaR パラメータ（`volatility_factor`, `var_confidence`, `oi_threshold`）を追加。
3. **[ ] Optimal F 計算関数作成**: `utils/position_sizing_utils.py`に`calculate_optimal_f(win_rate, reward_risk_ratio, volatility, oi)`。backtesting.py の optimalF をラップし、ボラ/OI 調整を追加。
   - テスト: ユニットテストで固定入力に対する出力検証（例: 高ボラでサイズ縮小）。
4. **[ ] VaR/ES 計算関数**: 同ファイルに`calculate_var_es(returns, confidence=0.95)`。歴史的リターンから計算。
   - テスト: モックデータで VaR=5%損失、ES=7%損失を確認。
5. **[ ] calculate_position_size 拡張**: `strategy_factory.py`の`position_sizing_service`に動的ロジック追加。入力に市場データ（ATR, OI）渡す。
   - バックテスト統合: `backtest`サービスで各トレード時にリアルタイム計算。
6. **[ ] GA オペレーター更新**: `genetic_operators.py`で新パラメータのクロスオーバー/ミューテーション対応。`random_gene_generator.py`でランダム生成追加。
7. **[ ] フィットネス関数更新**: `fitness_sharing.py`に Sharpe Ratio（リターン/ボラ）と Max Drawdown を追加。VaR をペナルティとして。
8. **[ ] シリアライザー対応**: `dict_converter.py`などで新フィールドのシリアライズ追加。
9. **[ ] フロントエンド統合**: `frontend/hooks/useAutoStrategy.ts`で新パラメータ表示/編集。バックエンド API 拡張（`/auto-strategy/generate`）。
10. **[ ] 統合テスト**: `comprehensive_test.py`でエンドツーエンドバックテスト。Sharpe > 1.5, Drawdown < 20%のシナリオ検証。
11. **[ ] ドキュメント更新**: `auto_strategy_enhancement_plan.md`にセクション追加。

## 潜在的課題と解決

- **計算負荷**: VaR は歴史データ依存。解決: ローリングウィンドウ（100-200 バー）で効率化。
- **データ可用性**: OI は Bybit API から。解決: `useOpenInterestData`フック活用、フォールバックで固定値。
- **過剰適合**: GA で Kelly パラメータ最適化。解決: アウトオブサンプルテスト。
- **セキュリティ**: 動的サイズでレバレッジ上限（例: 5x）。`validators.py`でバリデーション。

## 評価指標

- **バックテスト**: Sharpe Ratio 向上（目標: +0.5）、Max Drawdown 低減（目標: -10%）。
- **GA 性能**: 多様性維持（Fitness Sharing）、収束速度。
- **リアルタイム**: 実行時間 < 1ms/トレード。

## ポジションサイジング調整ロジックのフローチャート

```mermaid
flowchart TD
    A[トレードシグナル発生] --> B[市場データ取得: ATR, OI, 過去リターン]
    B --> C[勝率 p とリスク・リワード比 b を計算<br/>(バックテスト or ML 予測から)]
    C --> D[Optimal F 計算: f = optimal_f(p, b)]
    D --> E[ボラティリティ調整: f_adj = f * (1 / ATR_normalized)]
    E --> F[OI 流動性調整: f_final = f_adj * min(1, OI / threshold)]
    F --> G[VaR/ES 計算: VaR = historical_var(returns, 0.95)]
    G --> H[サイズ上限: size = min(f_final * capital, VaR_limit / expected_loss)]
    H --> I[ポジション実行: buy/sell with size]
    I --> J[バックテスト/GA でフィットネス評価]
    style A fill:#f9f,stroke:#333
    style I fill:#bbf,stroke:#333
```

この計画は既存アーキテクチャを尊重し、モジュール化。実装後、Code モードで TDD 実施。
