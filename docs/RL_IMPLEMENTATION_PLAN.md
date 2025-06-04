# 強化学習システム実装計画

## 1. 現在のコードベース分析

### 1.1 既存アーキテクチャ概要

**バックエンド構成：**
- **フレームワーク**: FastAPI + SQLAlchemy + TimescaleDB (PostgreSQL)
- **データ取得**: CCXT (Bybit API)
- **対象市場**: BTC専用（BTC/USDT、BTC/USDT:USDT、BTCUSD）
- **言語**: Python 3.x

**フロントエンド構成：**
- **フレームワーク**: Next.js 15 + React 18 + TypeScript
- **スタイリング**: TailwindCSS
- **アーキテクチャ**: コンポーネントベース設計

### 1.2 利用可能なデータ詳細

**既存データテーブル構造：**

1. **OHLCVData（価格データ）**
   - **フィールド**: symbol, timeframe, timestamp, open, high, low, close, volume
   - **インデックス**: 複合インデックス（symbol + timeframe + timestamp）
   - **制約**: ユニーク制約で重複防止
   - **最適化**: TimescaleDBハイパーテーブル対応

2. **FundingRateData（ファンディングレート）**
   - **フィールド**: symbol, funding_rate, funding_timestamp, next_funding_timestamp, mark_price, index_price
   - **特徴**: 8時間ごとの更新、市場センチメント指標として活用可能
   - **データ品質**: 高精度（小数点6桁）

3. **OpenInterestData（オープンインタレスト）**
   - **フィールド**: symbol, open_interest_value, data_timestamp
   - **特徴**: USD建て建玉残高、市場参加者の動向分析に活用
   - **更新頻度**: リアルタイム～1時間ごと

4. **TechnicalIndicatorData（テクニカル指標）**
   - **フィールド**: symbol, timeframe, indicator_type, period, value, signal_value, histogram_value, upper_band, lower_band
   - **対応指標**: SMA, EMA, RSI, MACD, ボリンジャーバンド, ATR, ストキャスティクス
   - **計算済み**: 既存の計算ロジックで事前計算済み

5. **DataCollectionLog（収集ログ）**
   - **用途**: データ品質管理、収集状況監視
   - **フィールド**: symbol, timeframe, start_time, end_time, records_collected, status, error_message

**データ収集・品質管理システム：**
- **収集方式**: Bybit API（CCXT経由）
- **バッチ処理**: 1000件ずつのバッチ処理で大量データ対応
- **重複防止**: データベースレベルでの重複チェック
- **エラーハンドリング**: 包括的なエラーログとリトライ機能
- **データ検証**: 価格関係の妥当性チェック（high >= max(open, close)等）

**データ量・パフォーマンス特性：**
- **時間軸**: 1m, 5m, 15m, 30m, 1h, 4h, 1d（7種類）
- **履歴期間**: 最大365日分の履歴データ収集可能
- **データベース**: SQLite（開発）/ PostgreSQL + TimescaleDB（本番）
- **クエリ最適化**: 複合インデックスによる高速時系列クエリ
- **メモリ効率**: pandas DataFrame統合でメモリ効率的な処理

### 1.3 既存API構造詳細

**主要エンドポイント：**
- `/api/market-data/ohlcv`: OHLCV データ取得（時系列クエリ対応）
- `/api/market-data/symbols`: サポート対象シンボル一覧
- `/api/market-data/timeframes`: サポート対象時間軸一覧
- `/api/funding-rates/`: ファンディングレート取得・収集
- `/api/open-interest/`: オープンインタレスト取得・収集
- `/api/technical-indicators/`: テクニカル指標取得・計算
- `/api/data-collection/`: 一括データ収集

**設計パターン詳細：**
- **Repository パターン**: `BaseRepository`を継承した型安全なデータアクセス
- **Service 層**: `BybitMarketDataService`等のビジネスロジック分離
- **FastAPI 依存性注入**: `get_db()`によるセッション管理
- **Pydantic データ検証**: 型安全性とバリデーション
- **統一エラーハンドリング**: `APIErrorHandler`による一貫したエラー処理
- **レスポンス標準化**: `APIResponseHelper`による統一フォーマット

**データ変換・検証システム：**
- **OHLCVDataConverter**: CCXT ↔ DB ↔ API形式の相互変換
- **DataValidator**: 価格データの妥当性検証・サニタイズ
- **DateTimeHelper**: ISO形式日時の統一処理
- **重複処理**: `bulk_insert_with_conflict_handling`による効率的な重複回避

**パフォーマンス最適化：**
- **バッチ処理**: 大量データの効率的な処理
- **インデックス活用**: 時系列クエリの高速化
- **接続プール**: SQLAlchemy QueuePoolによる接続管理
- **メモリ効率**: pandas統合による大量データ処理

## 2. 強化学習システム設計

### 2.1 環境（Environment）定義

**状態空間（State Space）設計：**

```python
# 基本設定
LOOKBACK_WINDOW = 20  # 過去20期間の価格データ
STATE_DIMENSION = 87  # 総状態次元数

状態ベクトル = [
    # 1. 正規化された価格データ（過去20期間）
    normalized_prices: [
        (close_t-19 - close_t-20) / close_t-20,  # リターン
        (high_t-19 - low_t-19) / close_t-19,     # 日中変動率
        log(volume_t-19 / volume_t-20),          # 出来高変化
        ...  # 20期間分
    ],  # 3 * 20 = 60次元

    # 2. テクニカル指標（既存TechnicalIndicatorDataから取得）
    technical_indicators: [
        rsi_14,           # RSI(14)
        macd_line,        # MACD線
        macd_signal,      # MACDシグナル
        macd_histogram,   # MACDヒストグラム
        sma_20,           # SMA(20)
        ema_12,           # EMA(12)
        bb_upper,         # ボリンジャーバンド上限
        bb_lower,         # ボリンジャーバンド下限
        atr_14,           # ATR(14)
        stoch_k,          # ストキャスティクス%K
        stoch_d,          # ストキャスティクス%D
    ],  # 11次元

    # 3. 市場構造データ（既存データベースから取得）
    market_structure: [
        funding_rate,              # 現在のファンディングレート
        funding_rate_change,       # ファンディングレート変化
        open_interest_normalized,  # 正規化されたオープンインタレスト
        oi_change_rate,           # オープンインタレスト変化率
    ],  # 4次元

    # 4. ポジション・ポートフォリオ情報
    portfolio_state: [
        position_size,        # ポジションサイズ（-1～1）
        unrealized_pnl,       # 未実現損益（正規化）
        holding_time,         # 保有時間（正規化）
        drawdown_from_peak,   # ピークからのドローダウン
    ],  # 4次元

    # 5. 時間的特徴量
    temporal_features: [
        sin(2π * hour / 24),      # 時刻の周期性
        cos(2π * hour / 24),
        sin(2π * day_of_week / 7), # 曜日の周期性
        cos(2π * day_of_week / 7),
        market_session_asia,       # アジア市場時間（0/1）
        market_session_europe,     # 欧州市場時間（0/1）
        market_session_us,         # 米国市場時間（0/1）
        volatility_regime,         # ボラティリティ体制（低/中/高）
    ],  # 8次元
]
```

**状態空間の特徴：**
- **総次元数**: 87次元（60 + 11 + 4 + 4 + 8）
- **正規化**: 全特徴量を[-1, 1]または[0, 1]に正規化
- **時系列性**: LSTMネットワーク対応のシーケンシャルデータ
- **既存データ活用**: データベースの全テーブルを効率的に活用

### 2.2 行動空間（Action Space）

**離散行動空間（推奨）：**
```python
# 基本行動セット（4行動）
actions = {
    0: "HOLD",     # ポジション維持・何もしない
    1: "BUY",      # ロングポジション開始（既存ポジションがない場合）
    2: "SELL",     # ショートポジション開始（既存ポジションがない場合）
    3: "CLOSE"     # 現在のポジションをクローズ
}

# 行動制約（Action Masking）
def get_valid_actions(current_position):
    if current_position == 0:  # ポジションなし
        return [0, 1, 2]  # HOLD, BUY, SELL
    else:  # ポジションあり
        return [0, 3]     # HOLD, CLOSE
```

**行動空間の特徴：**
- **シンプル性**: 4つの基本行動で理解しやすい
- **制約対応**: ポジション状態に応じた行動マスキング
- **既存システム統合**: `StrategyExecutor`の取引ロジックを活用
- **リスク管理**: 同時複数ポジション防止

### 2.3 報酬関数（Reward Function）

**多目的報酬設計：**
```python
reward = (
    α * return_reward +           # リターン報酬
    β * risk_adjusted_reward +    # リスク調整報酬
    γ * transaction_cost_penalty + # 取引コストペナルティ
    δ * drawdown_penalty          # ドローダウンペナルティ
)

where:
    return_reward = (current_portfolio_value - previous_portfolio_value) / previous_portfolio_value
    risk_adjusted_reward = return_reward / volatility
    transaction_cost_penalty = -0.001 * |action_change|  # 0.1% 取引コスト
    drawdown_penalty = -max(0, (peak_value - current_value) / peak_value - 0.05)  # 5%超のドローダウンにペナルティ
```

### 2.4 学習アルゴリズム選択

**推奨アルゴリズム：PPO（Proximal Policy Optimization）**
- **理由**: 安定性と性能のバランスが良い
- **特徴**: 連続・離散両方の行動空間に対応
- **実装**: Stable-Baselines3

## 3. 技術的実装詳細

### 3.1 新しい依存関係

**Python パッケージ（backend/requirements.txt に追加）：**
```
torch>=2.0.0
stable-baselines3>=2.0.0
gymnasium>=0.29.0
tensorboard>=2.14.0
optuna>=3.4.0
scikit-learn>=1.3.0
matplotlib>=3.7.0
seaborn>=0.12.0
```

### 3.2 ディレクトリ構造

**新規ディレクトリ構成：**
```
backend/
├── rl/                              # 強化学習モジュール
│   ├── __init__.py
│   ├── environments/                # 取引環境
│   │   ├── __init__.py
│   │   ├── trading_env.py          # メイン環境クラス（Gymnasium準拠）
│   │   ├── data_provider.py        # データ提供クラス（既存Repositoryを活用）
│   │   ├── reward_functions.py     # 報酬関数（複数戦略対応）
│   │   ├── state_processor.py      # 状態処理（特徴量エンジニアリング）
│   │   └── portfolio_manager.py    # ポートフォリオ管理（既存Tradeクラス拡張）
│   ├── agents/                      # RLエージェント
│   │   ├── __init__.py
│   │   ├── base_agent.py           # ベースエージェント（共通インターフェース）
│   │   ├── ppo_agent.py            # PPOエージェント（主力）
│   │   ├── dqn_agent.py            # DQNエージェント（比較用）
│   │   └── agent_factory.py        # エージェント生成（設定ベース）
│   ├── training/                    # 学習関連
│   │   ├── __init__.py
│   │   ├── trainer.py              # 学習実行（SB3統合）
│   │   ├── callbacks.py            # 学習コールバック（TensorBoard連携）
│   │   ├── hyperparams.py          # ハイパーパラメータ（Optuna対応）
│   │   └── experiment_manager.py   # 実験管理（結果保存・比較）
│   ├── evaluation/                  # 評価・バックテスト
│   │   ├── __init__.py
│   │   ├── backtester.py           # バックテスト実行（既存エンジン拡張）
│   │   ├── metrics.py              # 評価指標（既存指標 + RL特有指標）
│   │   └── visualizer.py           # 結果可視化（matplotlib/seaborn）
│   ├── models/                      # 学習済みモデル保存
│   │   ├── .gitkeep
│   │   ├── checkpoints/            # 学習チェックポイント
│   │   └── best_models/            # 最良モデル
│   └── utils/                       # ユーティリティ
│       ├── __init__.py
│       ├── feature_engineering.py  # 特徴量エンジニアリング（TI活用）
│       ├── data_preprocessing.py   # データ前処理（正規化・スケーリング）
│       ├── config.py               # RL設定（MarketDataConfig拡張）
│       └── model_utils.py          # モデル保存・読み込み
├── app/
│   ├── api/
│   │   └── rl_training.py          # RL学習API（既存APIパターン準拠）
│   └── core/
│       └── models/
│           └── rl_models.py        # RLデータモデル（Pydantic）
```



## 4. 開発フェーズ計画

### Phase 1: 基盤構築（1-2週間）

**目標**: 強化学習の基本インフラ構築

**詳細タスク:**
1. **依存関係管理**
   - `requirements.txt`への新規パッケージ追加
   - 仮想環境での依存関係テスト
   - GPU環境の設定確認

2. **ディレクトリ構造作成**
   - `backend/rl/`以下の全ディレクトリ作成
   - `__init__.py`ファイルの配置
   - `.gitkeep`ファイルの配置

3. **基本設定実装**
   - `rl/utils/config.py`: RL固有設定クラス
   - `MarketDataConfig`の拡張
   - 環境変数設定の追加

4. **データプロバイダー実装**
   - `RLDataProvider`クラス: 既存Repositoryの統合
   - データ取得インターフェースの統一
   - キャッシュ機能の実装

**成果物:**
- 完全なディレクトリ構造
- 基本設定システム
- データアクセス基盤

**検証方法:**
- 既存データベースからの全データ型取得テスト
- 設定ファイルの読み込みテスト
- データプロバイダーの基本動作確認

### Phase 2: 環境開発（2-3週間）

**目標**: 取引環境の完全実装

**詳細タスク:**
1. **状態空間設計・実装**
   - `StateProcessor`クラス: 特徴量エンジニアリング
   - 価格データの正規化・スケーリング
   - テクニカル指標の統合（既存`TechnicalIndicators`活用）
   - 市場構造データ（FR、OI）の組み込み
   - 時間的特徴量の生成

2. **行動空間実装**
   - 離散行動空間の定義（Hold/Buy/Sell/Close）
   - 行動制約の実装（ポジション状態による制限）
   - 行動マスキング機能

3. **報酬関数設計**
   - `RewardFunction`基底クラス
   - 複数報酬戦略の実装（リターン、リスク調整、取引コスト）
   - 報酬正規化・スケーリング機能

4. **ポートフォリオ管理**
   - `PortfolioManager`クラス（既存`Position`、`Trade`拡張）
   - 取引実行ロジック（既存`StrategyExecutor`パターン活用）
   - リスク管理機能

5. **環境クラス実装**
   - `TradingEnvironment`（Gymnasium準拠）
   - エピソード管理
   - 状態遷移ロジック

**成果物:**
- Gymnasium準拠の取引環境
- 柔軟な状態・報酬設計システム
- 既存システム統合済みポートフォリオ管理

**検証方法:**
- 環境の基本動作テスト（reset/step/render）
- ランダムエージェントでの長期実行テスト
- 報酬関数の妥当性・安定性検証
- 既存バックテストエンジンとの結果比較

### Phase 3: エージェント実装（2-3週間）

**目標**: 強化学習エージェントの実装と基本学習

**詳細タスク:**
1. **エージェント基盤実装**
   - `BaseAgent`抽象クラス: 共通インターフェース定義
   - `AgentFactory`クラス: 設定ベースのエージェント生成
   - エージェント設定管理システム

2. **PPOエージェント実装**
   - Stable-Baselines3のPPO統合
   - ネットワークアーキテクチャ設計（MLP/LSTM対応）
   - ハイパーパラメータ設定システム

3. **学習システム構築**
   - `Trainer`クラス: 学習実行管理
   - 学習ループの実装
   - チェックポイント機能
   - 早期停止機能

4. **学習監視システム**
   - TensorBoard統合
   - カスタムコールバック実装
   - 学習進捗の可視化
   - パフォーマンス指標の追跡

5. **実験管理システム**
   - `ExperimentManager`クラス
   - 実験設定の保存・管理
   - 結果比較機能

**成果物:**
- 完全なPPOエージェント実装
- 学習実行・監視システム
- 実験管理基盤

**検証方法:**
- 短期間学習での収束確認
- 学習曲線の妥当性検証
- TensorBoardでの監視機能確認
- 複数実験の並行実行テスト

### Phase 4-6: 評価・UI統合・最適化（4-6週間）

**Phase 4: 評価システム（1-2週間）**
- バックテスト機能と評価指標の実装
- 結果可視化システムの構築

**Phase 5: UI統合（1週間）**
- RL学習APIとフロントエンドコンポーネントの実装
- 学習進捗・結果表示機能

**Phase 6: 最適化・本番化（2-3週間）**
- ハイパーパラメータ最適化とパフォーマンス改善
- 本番環境対応とドキュメント整備

## 5. リスク管理と考慮事項

### 5.1 技術的リスク

**リスク1: 過学習**
- **対策**: 
  - 交差検証の実装
  - 早期停止機能
  - 正則化手法の適用

**リスク2: 計算リソース不足**
- **対策**:
  - GPU利用の最適化
  - 分散学習の検討
  - クラウドリソースの活用

**リスク3: データ品質問題**
- **対策**:
  - データ検証機能の強化
  - 異常値検出システム
  - データクリーニングパイプライン

### 5.2 パフォーマンス要件

**学習時間**: 初期学習 24-48時間以内
**推論時間**: リアルタイム推論 < 100ms
**メモリ使用量**: < 8GB RAM
**ストレージ**: モデル保存 < 1GB

### 5.3 既存システムへの影響

**最小限の影響**:
- 既存APIの変更なし
- データベーススキーマの変更なし
- フロントエンドの既存機能に影響なし

**新規追加のみ**:
- 新しいAPIエンドポイント
- 新しいフロントエンドコンポーネント
- 新しいバックエンドモジュール

## 6. 成功指標

### 6.1 技術的指標

- **学習安定性**: 学習曲線の収束
- **推論速度**: < 100ms/予測
- **システム可用性**: > 99%

### 6.2 性能指標

- **シャープレシオ**: > 1.0
- **最大ドローダウン**: < 20%
- **勝率**: > 55%
- **年間リターン**: ベンチマーク（Buy&Hold）を上回る

---

**注意**: この計画は既存のコードベースとアーキテクチャを最大限活用し、BTCのみに焦点を当てた実装となっています。段階的な実装により、リスクを最小化しながら確実に強化学習システムを構築できます。
