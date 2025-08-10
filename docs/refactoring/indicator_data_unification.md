## インジケータ計算のデータ型統一・変換最小化リファクタリング案

### 背景と目的
- 現状、オートストラテジーで利用するテクニカル指標は、pandas.DataFrame/Series と numpy.ndarray の相互変換を複数箇所で行っている。
- これにより、計算オーバーヘッド・型不整合の潜在バグ・メンテナンスコストが増加。
- 目的は「一気通貫のデータ型方針」と「変換回数の最小化」による性能・保守性の改善。

### 現状のデータフロー（要点）
- 入力: backtesting.py 由来の DataFrame（OHLCV）。
- orchestrator: backend/app/services/indicators/indicator_orchestrator.py
  - DataFrame からカラムを取り出し、to_numpy() → ensure_numpy_array() で numpy.float64 に統一。
  - adapter_function（各インジケータクラスのメソッド）を呼び出し。
- pandas-ta ラッパ: backend/app/services/indicators/pandas_ta_utils.py
  - 受け取った引数（np.ndarray or pd.Series）を _to_series() で Series に変換。
  - pandas_ta 関数を呼び、結果 Series/DF から .values で np.ndarray を返す。
- 出力: numpy.ndarray（もしくはタプル）を上位層へ返す。

結果として「DataFrame → numpy → Series → numpy」という往復が多くの指標で生じている。

参考（該当箇所の代表例）
- indicator_orchestrator.calculate_indicator: df[col].to_numpy() → ensure_numpy_array()
- pandas_ta_utils._to_series: np.ndarray を pd.Series へ

### 問題点
- 不必要な型変換（特に numpy → Series → numpy）によるオーバーヘッド
- 各レイヤーが型責務を重複して担っており、境界が不明瞭
- テスト観点での型揺らぎ（int 入力 → float64 出力などは妥当だが、どの層で強制されるかが分散）

### 方針の選択肢
1) pandas ベースで一気通貫
   - 長所: pandas-ta が Series を前提としており、最小変換で済みやすい。
   - 短所: 既存の外部 API が numpy を返す前提（多数のテスト・呼び出し側）を変更する影響が大。

2) numpy ベースで一気通貫
   - 長所: 計算実装が numpy 志向なら高速にしやすい。
   - 短所: pandas-ta は Series 入力を前提。内部で必ず Series 変換が必要なため往復が避けられない。

3) 境界設計で「変換最小化」（推奨）
   - 外部 API 契約（呼び出し側・テスト）を維持: 出力は numpy.ndarray。
   - 指標レイヤ内部では可能な限り pandas.Series を維持し、pandas-ta 呼び出し直前の変換を省略。
   - 具体的には、orchestrator から adapter_function へは Series を渡し、pandas_ta_utils 側で Series をそのまま扱う。戻りは .values で numpy に一度だけ変換。

### 推奨アーキテクチャ（詳細）
- 外部 I/F（サービス境界）: 入力 DataFrame、出力 numpy.ndarray（現状維持）。
- orchestrator 層: DataFrame からカラム取得時に to_numpy() せず「pd.Series のまま」渡す。
  - 例: array = df[col]（現状: df[col].to_numpy()）
  - ensure_numpy_array の適用箇所を削減（必要なら出力側で dtype を統一）。
- adapter 層（technical_indicators/*.py の各メソッド）
  - 引数型を Union[np.ndarray, pd.Series] に緩和（実装は入力が Series でも動作）。
  - 可能な限り pandas_ta_utils の関数へそのまま委譲。
  - 最終出力は np.ndarray（float64）で統一。
- pandas_ta_utils 層
  - 既に _to_series で Series 化しており、Series 入力なら変換をスキップ可能。
  - 戻り値は .values により numpy へ一度だけ変換（現状通り）。

この構成により、典型的なパスは「DataFrame → Series（維持） → pandas-ta → numpy（最終返却）」となり、往復変換が 1 回に減る。

### 影響範囲（主に修正/確認が必要な箇所）
- backend/app/services/indicators/indicator_orchestrator.py
  - required_data 構築で df[col] のまま渡す（to_numpy や ensure_numpy_array を外す/条件付きに）。
  - normalize_data_for_trig が必要な一部（ASIN/ACOS前処理など）は、Series 対応（内部で values を取る or 関数を Series も受けられる形に拡張）。
- backend/app/services/indicators/technical_indicators/*.py
  - メソッド引数型ヒントの緩和（Union[np.ndarray, pd.Series]）。
  - 入力側 ensure_numpy_array の乱用を削減（必要な検証は validate_input/validate_multi_input を見直し）。
- backend/app/services/indicators/utils.py
  - ensure_numpy_array は後方互換で残すが、呼び出しは最小化。
  - normalize_data_for_trig を Series 受け取りに対応（Union 受け取り、最後に np.asarray へ）。
- backend/app/services/indicators/pandas_ta_utils.py
  - 現状で Series 入力を優先的に扱える（追加修正は最小）。

### 具体的変更案（抜粋サンプル）
- orchestrator の required_data 作成（疑似差分）
  - Before: array = df[col].to_numpy(); required_data[param] = ensure_numpy_array(array)
  - After:  series = df[col]; required_data[param] = series
- adapter の型ヒント
  - def rsi(data: np.ndarray, period: int = 14) → def rsi(data: Union[np.ndarray, pd.Series], period: int = 14)
- normalize_data_for_trig
  - 引数を Union[np.ndarray, pd.Series] とし、最初に np.asarray(data, dtype=np.float64) で 1 回だけ配列化。

注: 本ドキュメントは方針書であり、コード変更は別 PR として段階的に適用。

### 性能影響の見積りとベンチ指標
- 想定改善点: 指標 1 回の計算につき、不必要な numpy→Series→numpy 変換 1 回分を削減。
- ベンチ方法（ローカル計測の例）
  - N=100k のダミー価格系列で、代表指標（RSI/MACD/ATR/BB）を 100 回反復。
  - 現行 vs 変更後で wall time を比較、少なくとも 5〜15% 程度の短縮を期待。
  - プロファイルツール: time.perf_counter、または pytest-benchmark。

### リスクと対策
- リスク: adapter 層の入力型想定ズレ（np.ndarray 前提の実装に Series が渡る）
  - 対策: すべての adapter を Union 受け取りに変更し、必要最小限で np.asarray を適用。
- リスク: 型変換削減に伴う dtype 逸脱（int → float64 への強制）
  - 対策: 戻り値を常に np.float64 に統一するユーティリティを導入（format_indicator_result で最終保証）。
- リスク: 一部の placeholder 実装（例: pandas_ta_aroon = stoch などの仮実装）の存在
  - 対策: 別タスクで正しいラッパへ差し替える。今回の主旨（型統一）と分離。

### 段階的マイグレーション手順
- Phase 0（準備）
  - tests の型期待を再確認（現状: 出力は numpy を前提）。
  - format_indicator_result に dtype=float64 の最終保証を追加（オプション）。
- Phase 1（最小変更での効果獲得）
  - orchestrator: df[col]（Series）を adapter に渡す変更。
  - normalize_data_for_trig の Series 対応（Union 受け取り）。
- Phase 2（adapter の整理）
  - technical_indicators/*.py のメソッド引数型ヒントを Union に変更。
  - 不要な ensure_numpy_array 呼び出しを削減。検証は validate_* を活用。
- Phase 3（ユーティリティの明確化）
  - utils: ensure_numpy_array の利用箇所を棚卸しし、境界（外部 I/F 直前/直後）に限定。
  - pandas_ta_utils: _to_series は維持。Series 入力優先の前提をコメントで明文化。
- Phase 4（ベンチ/テスト）
  - 代表ケースのベンチ計測を実施し、改善率を記録。
  - 既存 tests に加え、型/精度/エッジケース（NaN/inf/短系列）を網羅する回帰テストを追加。

### テスト計画
- 既存: backend/tests/indicators/*（全通過を確認）
- 追加:
  - 型安定性: int 入力でも出力 float64、Series 入力でも出力 numpy を保証。
  - データ境界: 最小長、NaN混在、極端値（非常に大/小）での健全性。
  - パフォーマンス: pytest-benchmark による主要指標の比較。

### 今後の拡張（範囲外だが言及）
- ML 指標: auto_strategy/services/ml_orchestrator.py の I/O も同方針で整備（DataFrame 入力→最終のみ numpy）。
- OI/FR の統合: 将来的な特徴量増に対しても Series 優先での前処理を標準化。

### 参考資料（出典）
- pandas-ta: https://github.com/twopirllc/pandas-ta
- pandas Series.to_numpy: https://pandas.pydata.org/pandas-docs/stable/reference/api/pandas.Series.to_numpy.html
- NumPy astype: https://numpy.org/doc/stable/reference/generated/numpy.ndarray.astype.html
- backtesting.py Data: https://kernc.github.io/backtesting.py/doc/backtesting/#data

