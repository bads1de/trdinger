# Trdinger - トレーディング戦略バックテストサービス

仮想通貨のトレーディング戦略を定義し、過去データを用いてその有効性を検証（バックテスト）できるWebサービスです。

## 🚀 機能

- **戦略定義**: テクニカル指標と売買ルールを組み合わせた戦略を定義
- **バックテスト実行**: 過去データを使用して戦略の有効性を検証
- **結果分析**: シャープレシオ、最大ドローダウン、勝率などの詳細な分析
- **視覚的表示**: 損益曲線や取引履歴をわかりやすく表示

## 🛠 技術スタック

### フロントエンド
- **Next.js 15** - React フレームワーク
- **TypeScript** - 型安全性
- **Tailwind CSS** - スタイリング
- **Recharts** - チャート表示

### バックエンド
- **Python** - バックテストエンジン
- **Pandas & NumPy** - データ処理
- **自作テクニカル指標ライブラリ** - SMA, EMA, RSI, MACD, ボリンジャーバンド等

### API
- **Next.js API Routes** - RESTful API
- **JSON** - データ交換形式

## 📦 インストール

### 前提条件
- Node.js 18+
- Python 3.10+
- npm または yarn

### セットアップ

1. **リポジトリのクローン**
```bash
git clone https://github.com/bads1de/trdinger.git
cd trdinger
```

2. **フロントエンドの依存関係をインストール**
```bash
npm install
```

3. **バックエンドの依存関係をインストール**
```bash
cd backend
pip install -r requirements.txt
cd ..
```

## 🚀 使用方法

### 開発サーバーの起動
```bash
npm run dev
```

ブラウザで `http://localhost:3000` にアクセスしてください。

### テストの実行

**フロントエンドのテスト**
```bash
npm test
```

**バックエンドのテスト**
```bash
cd backend
python -m pytest tests/ -v
```

## 📊 使用例

### 1. 戦略定義
1. `/strategy` ページで新しい戦略を作成
2. テクニカル指標を追加（例：SMA 20期間、RSI 14期間）
3. エントリールールを設定（例：`close > 100`）
4. エグジットルールを設定（例：`close < 95`）
5. 戦略を保存

### 2. バックテスト実行
1. `/backtest` ページで保存した戦略を選択
2. バックテスト期間と初期資金を設定
3. バックテストを実行
4. 結果を確認

## 🧪 テスト

### テクニカル指標のテスト
```bash
cd backend
python -m pytest tests/test_indicators.py -v
```

### 戦略実行エンジンのテスト
```bash
cd backend
python -m pytest tests/test_strategy_executor.py -v
```

## 📁 プロジェクト構造

```
trdinger/
├── src/                    # Next.js フロントエンド
│   ├── app/               # App Router
│   │   ├── api/          # API Routes
│   │   ├── strategy/     # 戦略定義ページ
│   │   └── backtest/     # バックテストページ
│   ├── components/        # React コンポーネント
│   ├── types/            # TypeScript 型定義
│   └── utils/            # ユーティリティ関数
├── backend/               # Python バックエンド
│   ├── src/
│   │   ├── backtest_engine/  # バックテストエンジン
│   │   │   ├── indicators.py      # テクニカル指標
│   │   │   └── strategy_executor.py  # 戦略実行
│   │   └── backtest_runner.py     # バックテスト実行スクリプト
│   └── tests/            # Python テスト
└── docker/               # Docker 設定（将来実装）
```

## 🔧 対応テクニカル指標

- **SMA** (Simple Moving Average) - 単純移動平均
- **EMA** (Exponential Moving Average) - 指数移動平均
- **RSI** (Relative Strength Index) - 相対力指数
- **MACD** (Moving Average Convergence Divergence) - MACD
- **ボリンジャーバンド** - Bollinger Bands
- **ストキャスティクス** - Stochastic Oscillator
- **ATR** (Average True Range) - 平均真の値幅

## 📈 パフォーマンス指標

- **総リターン** - 投資期間全体の収益率
- **シャープレシオ** - リスク調整後リターン
- **最大ドローダウン** - 最大下落率
- **勝率** - 勝ち取引の割合
- **プロフィットファクター** - 総利益/総損失
- **平均利益/損失** - 取引あたりの平均損益

## 🚧 今後の実装予定

- [ ] データベース統合（TimescaleDB）
- [ ] 外部データソース連携（取引所API）
- [ ] より多くのテクニカル指標
- [ ] 複雑な戦略ロジック
- [ ] ポートフォリオバックテスト
- [ ] リアルタイムデータ対応
- [ ] Docker コンテナ化
- [ ] LLM統合（戦略提案）

## 📄 ライセンス

MIT License

## 🤝 コントリビューション

プルリクエストやイシューの報告を歓迎します！

## 📞 サポート

質問や問題がある場合は、GitHubのIssuesページでお知らせください。
