# 🤝 コントリビューションガイド

Trdingerプロジェクトへのコントリビューションを歓迎します！このガイドでは、プロジェクトに貢献する方法について説明します。

## 📋 目次

- [開発環境のセットアップ](#開発環境のセットアップ)
- [コーディング規約](#コーディング規約)
- [テスト駆動開発](#テスト駆動開発)
- [プルリクエストの流れ](#プルリクエストの流れ)
- [イシューの報告](#イシューの報告)
- [コミットメッセージ規約](#コミットメッセージ規約)

## 🛠 開発環境のセットアップ

### 前提条件

- Node.js 18.0.0以上
- Python 3.10以上
- Git

### セットアップ手順

1. **リポジトリのフォーク**
   ```bash
   # GitHubでリポジトリをフォーク後
   git clone https://github.com/YOUR_USERNAME/trdinger.git
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

4. **開発サーバーの起動**
   ```bash
   npm run dev
   ```

## 📝 コーディング規約

### TypeScript/JavaScript

- **ESLint**と**Prettier**の設定に従う
- **関数とクラスにはJSDocコメントを必須**で記述
- **型安全性**を重視し、`any`型の使用は避ける
- **コンポーネント名**はPascalCase、**ファイル名**はkebab-case

```typescript
/**
 * ユーザー情報を取得する関数
 * 
 * @param userId - ユーザーID
 * @returns ユーザー情報のPromise
 * @throws {Error} ユーザーが見つからない場合
 */
async function fetchUser(userId: string): Promise<User> {
  // 実装
}
```

### Python

- **PEP 8**スタイルガイドに従う
- **docstring**は必須（Google形式推奨）
- **型ヒント**を積極的に使用
- **関数名**はsnake_case

```python
def calculate_sma(data: pd.Series, period: int) -> pd.Series:
    """単純移動平均線を計算します。
    
    Args:
        data: 価格データのSeries
        period: 移動平均の期間
        
    Returns:
        計算された移動平均のSeries
        
    Raises:
        ValueError: periodが0以下の場合
    """
    if period <= 0:
        raise ValueError("期間は1以上である必要があります")
    return data.rolling(window=period).mean()
```

## 🧪 テスト駆動開発

### TDDサイクル

1. **Red**: 失敗するテストを書く
2. **Green**: テストが通る最小限のコードを書く
3. **Refactor**: コードを改善する

### テストの実行

**フロントエンド**
```bash
npm test
npm run test:watch  # ウォッチモード
npm run test:coverage  # カバレッジ付き
```

**バックエンド**
```bash
cd backend
python -m pytest tests/ -v
python -m pytest tests/ --cov=src  # カバレッジ付き
```

### テストの書き方

**React コンポーネント**
```typescript
import { render, screen } from '@testing-library/react'
import { Home } from './page'

describe('Home', () => {
  it('メインタイトルが表示される', () => {
    render(<Home />)
    expect(screen.getByText('仮想通貨トレーディング戦略')).toBeInTheDocument()
  })
})
```

**Python 関数**
```python
import pytest
from backtest_engine.indicators import TechnicalIndicators

class TestTechnicalIndicators:
    def test_sma_calculation(self):
        """SMA計算のテスト"""
        data = pd.Series([1, 2, 3, 4, 5])
        result = TechnicalIndicators.sma(data, 3)
        expected = pd.Series([NaN, NaN, 2.0, 3.0, 4.0])
        pd.testing.assert_series_equal(result, expected)
```

## 🔄 プルリクエストの流れ

1. **イシューの確認**
   - 既存のイシューを確認
   - 新機能の場合は事前にイシューを作成

2. **ブランチの作成**
   ```bash
   git checkout -b feature/new-indicator
   git checkout -b fix/calculation-bug
   git checkout -b docs/update-readme
   ```

3. **開発とテスト**
   - TDDサイクルに従って開発
   - 全てのテストが通ることを確認

4. **コミット**
   ```bash
   git add .
   git commit -m "feat: RSI指標の計算機能を追加"
   ```

5. **プッシュとPR作成**
   ```bash
   git push origin feature/new-indicator
   ```

### PRのチェックリスト

- [ ] 全てのテストが通る
- [ ] コードカバレッジが80%以上
- [ ] ESLint/Flake8エラーがない
- [ ] 適切なドキュメントが追加されている
- [ ] 破壊的変更がある場合は明記されている

## 🐛 イシューの報告

### バグレポート

以下の情報を含めてください：

- **環境情報** (OS, ブラウザ, Node.js/Pythonバージョン)
- **再現手順**
- **期待される動作**
- **実際の動作**
- **スクリーンショット** (該当する場合)

### 機能リクエスト

- **機能の概要**
- **使用ケース**
- **期待される動作**
- **代替案** (あれば)

## 📝 コミットメッセージ規約

[Conventional Commits](https://www.conventionalcommits.org/)に従います：

```
<type>[optional scope]: <description>

[optional body]

[optional footer(s)]
```

### タイプ

- `feat`: 新機能
- `fix`: バグ修正
- `docs`: ドキュメントのみの変更
- `style`: コードの意味に影響しない変更（空白、フォーマット等）
- `refactor`: バグ修正や機能追加ではないコード変更
- `test`: テストの追加や修正
- `chore`: ビルドプロセスやツールの変更

### 例

```bash
feat(indicators): RSI計算機能を追加
fix(backtest): 手数料計算のバグを修正
docs(readme): インストール手順を更新
test(indicators): SMA計算のテストを追加
```

## 🎯 開発のベストプラクティス

### コードレビュー

- **建設的なフィードバック**を心がける
- **コードの意図**を理解してからコメント
- **代替案**があれば提案する
- **学習の機会**として捉える

### パフォーマンス

- **不要な再レンダリング**を避ける
- **大きなデータセット**の処理は最適化する
- **メモリリーク**に注意する

### セキュリティ

- **入力値の検証**を必ず行う
- **機密情報**をコードに含めない
- **依存関係**の脆弱性を定期的にチェック

## 📞 サポート

質問や相談がある場合：

- **GitHub Issues**: バグ報告や機能リクエスト
- **GitHub Discussions**: 一般的な質問や議論

## 📄 ライセンス

このプロジェクトに貢献することで、あなたのコントリビューションがMITライセンスの下でライセンスされることに同意したものとみなされます。

---

**ありがとうございます！** 🙏

あなたのコントリビューションがTrdingerをより良いプロジェクトにします。
