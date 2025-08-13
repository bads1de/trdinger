# Deprecated / 非推奨シンボル一覧

このレポートはプロジェクト内で検出された「非推奨（deprecated）」を示す箇所のリストです。
各行は「ファイルパス : 行番号範囲 : 抽出テキスト」の形式で示しています。
優先度は手動で「高/中/低」を付与していませんが、頻出・利用箇所が広いものほど注意度が高くなります。

---


- `backend/app/utils/index_alignment.py` : 229-250  
  後方互換性のためのレガシークラス（非推奨） / `MLWorkflowIndexManager` が非推奨コメントと警告を出しています。

- `backend/app/utils/data_validation.py` : 196-216  
  `safe_divide()` — 「非推奨」として warnings.warn(DeprecationWarning) を発している。

- `backend/app/utils/data_validation.py` : 242-264  
  `safe_correlation()` — 非推奨。

- `backend/app/utils/data_validation.py` : 274-297  
  `safe_multiply()` — 非推奨。

- `backend/app/utils/data_validation.py` : 312-334  
  `safe_pct_change()` — 非推奨。

- `backend/app/utils/data_validation.py` : 347-366  
  `safe_rolling_mean()` — 非推奨。

- `backend/app/utils/data_validation.py` : 408-426  
  `safe_normalize()` — 非推奨。

- `backend/app/utils/data_validation.py` : 449-465  
  `validate_dataframe()` — 非推奨（`validate_dataframe_with_schema()` を推奨）。

- `backend/app/utils/data_validation.py` : 525-541  
  `clean_dataframe()` — 非推奨（`clean_dataframe_with_schema()` を推奨）。

- `backend/app/utils/data_processing.py` : 546-556  
  IQR法による外れ値除去（非推奨。OutlierRemovalTransformer を推奨）。

- `backend/app/utils/data_processing.py` : 583-593  
  Z-score法による外れ値除去（非推奨）。

- `backend/app/utils/data_processing.py` : 620-628  
  安全なカテゴリカル変数エンコーディング（非推奨。CategoricalEncoderTransformer を推奨）。

- `backend/app/utils/data_processing.py` : 1136-1150  
  カテゴリカル変数を数値にエンコーディングする旧関数（非推奨）。

- `backend/app/services/ml/base_ml_trainer.py` : 926-929  
  従来のランダム分割（非推奨）警告メッセージ。

- `backend/app/services/auto_strategy/services/ml_orchestrator.py` : 645-653  
  `_get_default_indicators` が非推奨（警告ログあり）。

---

備考（実行時のヒット総数）:
- 検出件数: 27 ヒット（warnings.warn による DeprecationWarning / コメントによる「非推奨」表記等を含む）

推奨ワークフロー（安全に削除するため）:
1. 全ての非推奨シンボルの呼び出し箇所を検索して影響範囲を把握する（例: `git grep -n "safe_divide("` 等）。
2. 低リスクから順に「内部で新APIへ委譲するラッパ」に置き換える（挙動互換を保つ）。
3. CI（ユニットテスト／統合テスト）を実行して回帰がないことを確認する。
4. テスト通過後、deprecated マーカー（warnings.warn と docstring）を削除して本体を削除する。
5. 最後に不要 import を除去し lint を適用する。

次の自動支援（選択してください）:
- deprecated シンボルごとに「呼び出し箇所」を全自動で検索して一覧化したレポートを作る（推奨）  
- まずは `safe_*` 系（data_validation.py）をラッパ化して deprecated を内部委譲に変えるパッチを作る（低リスク）  
- 今はここで止める（手動で確認する）
