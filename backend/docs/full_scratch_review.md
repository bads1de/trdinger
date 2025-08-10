# `backend`ディレクトリのフルスクラッチ実装レビュー（実施方針に更新）

## 目的と方針
- 目的: コード削減と潜在的バグの回避を最優先に、低リスク・高効果の改善のみ段階導入
- 前提: Windows 実行環境を考慮（Unix 専用APIの無計画な適用は避ける）、ログは最小化
- 範囲: 本ドキュメントでは「やるべき」内容のみを記載し、当面やらない項目は削除済み

## 実施する改善（優先順）

### 1) FastAPI の response_model で API レスポンス標準化（小規模・効果大）
- 対象: `app/` の API ルート全般（`api_utils.py` のボイラープレート削減）
- 目的: レスポンス構造を宣言的に定義し、型/バリデーションを自動化
- 実施:
  - 各エンドポイントに Pydantic の ResponseModel を付与
  - エラーレスポンスも共通 ErrorSchema を定義し統一
- 効果: コード削減（中〜大）、整合性向上、フロント連携の安定
- 参考: FastAPI Response Model https://fastapi.tiangolo.com/tutorial/response-model/

### 2) tenacity によるリトライ/タイムアウトの標準化（局所導入）
- 対象: 外部API/ネットワーク/DB I/O を中心に（例: `services/data_collection/...`）
- 目的: 自作デコレータ/try-except 乱立の削減と安定化
- 実施:
  - `@retry`, `stop`, `wait`, `retry=retry_if_exception_type` などの組合せで失敗時の再試行を共通化
  - hard timeout は既存の OS 別タイムアウト（UnifiedTimeout）を維持
- 効果: コード削減（中）、エッジケース耐性の向上
- 参考: tenacity https://tenacity.readthedocs.io/en/latest/

### 3) 設定を pydantic-settings へ段階移行（低〜中リスク）
- 対象: `services/ml/config/ml_config_manager.py` など設定管理周辺
- 目的: 型安全・バリデーションの自動化で自作設定管理を縮小
- 実施:
  - 環境変数/ファイル設定を pydantic-settings に寄せる
  - フィールドバリデーションを Pydantic に集約
- 効果: コード削減（中）、設定不備の早期検知
- 参考: pydantic-settings https://docs.pydantic.dev/latest/integrations/pydantic_settings/

### 4) DataConverter の統一（pandas 中間表現 + Pydantic 検証）
- 対象: `data_conversion.py`（`OHLCVDataConverter` 等）
- 目的: 形式別の冗長ロジック削減と検証の一元化
- 実施:
  - 変換前は pandas DataFrame へ統一
  - 変換後は Pydantic モデルで最終スキーマ検証
  - `pd.to_numeric`, `np.asarray` 等の標準関数活用
- 効果: コード削減（中〜大）、NaN/Inf 混入などの不正検知
- 参考: pandas https://pandas.pydata.org/docs/

### 5) エラーハンドリングの契約統一（戻り値と例外の方針明確化）
- 対象: `app/utils/unified_error_handler.py` と呼び出し側
- 目的: 呼び出し側の if/else 減少、型の揺れによるバグ回避
- 実施:
  - ML系ハンドラは「常に default_return を返す」か「常に例外送出」かを関数単位で仕様明記
  - `safe_execute` の `default_value` と `default_return` の二重化は将来的に片方へ整理
- 効果: バグ回避（中）、呼び出し側の簡素化
- 参考: Python logging https://docs.python.org/3/library/logging.html

### 6) カスタム例外の圧縮（標準例外の活用）
- 対象: プロジェクト全体のカスタム例外
- 目的: 例外乱立の抑制と握り漏れ防止
- 実施:
  - `ValueError`/`TypeError`/`RuntimeError` で代替可能な箇所の移行
  - API 近傍のみ薄いドメイン例外を残す
- 効果: コード削減（中）、例外処理の一貫性向上
- 参考: Python Exceptions https://docs.python.org/3/library/exceptions.html

### 7) API グローバル例外ハンドラの最小導入
- 対象: アプリ初期化部（FastAPI `add_exception_handler`）
- 目的: エンドポイントごとの try/except を整理し、レスポンス形式を統一
- 実施:
  - `HTTPException`/`ValidationError`/必要なドメイン例外のみを対象に JSON 変換
  - 既存の OS 別タイムアウトや ML ユーティリティは維持
- 効果: コード削減（中）、詳細情報の欠落防止
- 参考: FastAPI Error Handling https://fastapi.tiangolo.com/tutorial/handling-errors/

### 8) Lint/Type Check（ruff/mypy）を最小設定で導入
- 対象: 重要モジュールから順次
- 目的: 未使用・未定義・型不整合などの潜在バグ検出
- 実施:
  - mypy は緩め設定から開始し、段階的に厳格化
  - 出力は最小化し、CI の早期失敗で品質確保
- 効果: バグ回避（中）
- 参考:
  - mypy https://mypy.readthedocs.io/en/stable/
  - ruff https://docs.astral.sh/ruff/

## 備考（環境依存の注意）
- Windows 環境では `signal.alarm` は利用不可。既存の OS 分岐（`UnifiedErrorHandler.handle_timeout`）を維持
- `concurrent.futures` によるタイムアウトは継続利用
- 参考:
  - concurrent.futures https://docs.python.org/3/library/concurrent.futures.html#concurrent.futures.Future.result
  - signal.alarm https://docs.python.org/3/library/signal.html#signal.alarm

## 今後の進め方（フェーズ分割）
- フェーズ1（低リスク・即効性）
  - response_model 導入（代表エンドポイント）
  - tenacity 導入（外部 I/O のみ）
  - エラーハンドリング契約の明記
  - テスト追加（API スキーマ、リトライ挙動、タイムアウト単体）
- フェーズ2（中規模）
  - pydantic-settings へ主要設定を移行
  - DataConverter の pandas+Pydantic 化（対象を限定し順次）
  - カスタム例外の圧縮
- フェーズ3（必要に応じて）
  - GA/Auto-Strategy 周りの表現最適化（小さな重複削減から）
