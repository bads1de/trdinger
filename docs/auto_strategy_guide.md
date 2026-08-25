# オートストラティジーパイプライン解説（平語ガイド）

> 目的: 「個体（戦略候補）が生成されてから採用・棄却されるまで、何がどう判定されるか」を
> コードを読まなくても追跡できるようにする。全ての恣意的な数値には出典（ファイル名）を付ける。

---

## 1. 全体フロー

```
[1] データ収集 (scripts/data_ingestion/*)
        OHLCV / FundingRate / OpenInterest / LongShortRatio をPostgreSQLへ
        対応シンボル: BTC/USDT:USDT, ETH/USDT:USDT（app/config/constants.py SUPPORTED_MARKET_SYMBOLS）
   ↓
[2] GA進化 (scripts/run_auto_strategy → GeneticAlgorithmEngine)
        ランダム遺伝子の集団をバックテスト成績で進化させる
        RNGシード: GAConfig.random_state（既定42=決定論。--random-state で変更可）
   ↓
[3] 自動検証パイプライン (StrategyValidationService)
        進化結果を複数の「過学習チェック」に通し、合格/不合格と理由を出す
   ↓
[4] 検証ハーネス (scripts/validate_pipeline.py)
        複数期間×複数シードで[2][3]を実行し、合格者だけを
        「知らない未来データ（OOS窓）」で再評価。結果を履歴JSONLに蓄積
   ↓
[5] メタ分析トラッカー
        履歴全体で「IS側の指標がOOS成績を予測するか」を追跡
```

---

## 2. 個体はどこで死ぬか — 検証ゲート一覧

進化で生き残った個体でも、以下を1つでも落とせば不採用。**不合格理由は結果JSONの `reasons` に出る。**

| # | ゲート | 試験内容 | 合格基準 | 既定値の出典 |
|---|---|---|---|---|
| 1 | 最少取引数 | 全期間のトレード数 | ≥ min_trades（=10） | ga_config.py `validation_config.min_trades` |
| 2 | WFA頑健性 | 学習期間内を3分割し、見ていない時間帯で再評価 | fold合格率 ≥ 0.33（3fold中1回以上） | ga_config.py `wfa_n_folds` / `min_pass_rate` |
| 3 | PBO（過学習確率） | WFA fold間で成績上位入替の頻度を測定 | PBO ≤ 0.50 | `pbo_threshold` |
| 4 | 全期間市場比 | 戦略リターン − B&Hリターン | ≥ 0（市場に勝つ） | strategy_validation_service.py `_judge` |
| 5 | レジーム一貫性 | 学習期間を2等分し各スライスでも検査 | 各スライスで市場比≥0 かつ trades≥⌈10/2⌉ | `regime_splits=2`, fail-closed |
| 6 | パラメータ摂動耐性 | 主要パラメータを微小に動かして再評価 | 変異体生存率 ≥ 0.50 | `param_min_survival_rate` |
| 7 | 行動クローン排除 | 取引履歴のフィンガープリント（8キーのハッシュ）が完全一致する兄弟を統合 | グループ内最良の1体のみ通す | individual_evaluator.py `compute_trade_fingerprint` |

設計思想は一貫している: **「たまたま良かった」を弾くこと**。
（WFA=時間的偶然、摂動=パラメータの偶然、レジームゲート=特定相場依存、クローン=同一挙動の重複计数）

### 不合格理由の読み方（実例）

```
最少取引数 5.0 が閾値 10 未満          ← ゲート1: 取引が少なすぎて統計的に無意味
通期市場比リターン -0.34 が負値です     ← ゲート3: B&H保有に負けている
レジーム一致性なし slice2(2024-07...)   ← ゲート4: 前半期だけ機能する片寄り戦略
行動的同一クローン個体                  ← ゲート6: 別IDだが取引内容が同一な兄弟
```

---

## 3. 検証ハーネス（validate_pipeline.py）の仕組み

- **期間レジストリ**: 半年窓 2022H1〜2026H1（スクリプト先頭 `PERIOD_REGISTRY`）
- **GA実行**: 期間ごとにサブプロセスで run_auto_strategy。`--seeds N` で seed 42,43,… のN回実行
- **遺伝子重複排除**: 遺伝子を正規化→SHA256の先頭12桁（`gene_hash`）。同一遺伝子の再記録はスキップ
- **OOS評価**: 学習期間と同長の「直後の半年」を未来データとして初見評価。
  - 同一シンボル（通常検証）と別シンボル（`--eval-symbol`、cross_symbolフラグ付き）の両対応
- **評価シンボルは必ず行に明記**される（`eval_symbol`）。旧形式の行は読み込み時に自動補完
- **履歴**: `results/validation_history.jsonl` に行追加（追記専用）

### レジーム分類（⚠️恣意的な閾値）

- OOS窓のB&Hリターンで `bull > +20pp / sideways ±20pp / bear < -20pp`
- IS窓も同様に分類し、`is_regime / oos_regime / regime_match` 列を記録
- **この±20ppは研究用の便宜的な値。運用判断の根拠にする前に感度確認が必要**

---

## 4. メタトラッカーの現状（2026-08時点の知見）

履歴 n=78（同シンボル56 + クロス17 + その他）からの結論:

1. **IS側指標（fitness / マージン / is_excess）はOOS成績を予測しない**
   （Pearson/Spearmanすべて非有意または不安定）
2. **強気OOS窓では誰も市場に勝てない**: 0/21
3. **唯一の優位セル**: 「横ばいで進化した戦略を横ばい窓で稼働」12/24勝ち・中央値+1.4pp
   - 注意: サンプル24で勝率50%。幸運の範囲内の可能性はまだある
4. **市場間転移はほぼ無い**: BTC→ETH 0/9、ETH→BTC 5/8（ただし勝ち5件は暴落→暴落の1クラスタに偏る）

**実務的含意**: 現状システムの価値は「戦略発見」ではなく「ダメな戦略の高精度排除」。

### ウォークフォワード検証による訂正（2026-08）

上記4の「横ばい×横ばいのみ優位」は、**月次×先読みなし検出（n=1435）では再現しなかった**。
`scripts/verify_regime_rule.py` による検証の結果:

| 検出レジーム | 月次中央値 | 勝率 |
|---|---|---|
| bear | +2.04pp | 55% |
| bull | −1.07pp | 43% |
| **sideways** | **−9.53pp** | **11%** |

「sideways進化戦略をsideways検出月のみ稼働」ルール（R1）は中央値−10.13pp・勝率20%で棄却。
半年窓でのセル分析は細粒度では崩れる。観測詳細は `results/regime_rule_records.jsonl`。
※「bear検出月が最良」という逆パターンは新仮説であり、別データでの事前登録・再検証が必要。

---

## 5. よくある疑問

- **Q. 結果ファイルが大量にあるのは?**
  A. シード倍増で同期間の実行が複数保存されるため。ハーネスは gene_hash で統合するので二重計上されない。
- **Q. smokeモードで結果が保存されないのは?**
  A. 仕様（詳細保存無効）。動作確認専用。
- **Q. 1mのLong/Short Ratio警告は?**
  A. 1mのLSRデータは収集しておらず既定値1.0を使用。BTCも同じ挙動なので影響は対称。
- **Q. 判定に人的判断が入る箇所は?**
  A. 上記レジーム閾値(§3)と、メタ分析の「いつ判定を確定させるか」（n≥10で表示開始だが確定は未定）。

---

## 6. 主要ファイル地図

| ファイル | 役割 |
|---|---|
| `backend/scripts/run_auto_strategy.py` | GA実行CLI（--symbol/--random-state/--smoke等） |
| `backend/app/cli/config_builder.py` | CLI引数→GAConfig変換（random_stateセンチネル含む） |
| `backend/app/services/auto_strategy/services/strategy_validation_service.py` | 検証ゲート本体（§2の表の実装） |
| `backend/app/services/auto_strategy/core/evaluation/individual_evaluator.py` | 取引フィンガープリント |
| `backend/scripts/validate_pipeline.py` | 検証ハーネス＋メタトラッカー |
| `backend/results/validation_history.jsonl` | 観測履歴（追記専用JSONL） |
| `frontend/components/backtest/BaseBacktestConfigForm.tsx` | バックテスト設定UI（BTC/ETH選択可） |
