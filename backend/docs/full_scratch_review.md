# フルスクラッチ実装の調査レポート

## 1. はじめに

本レポートは、`backend`ディレクトリ内に存在する、標準ライブラリやサードパーティライブラリで代替可能なフルスクラッチ実装を特定し、改善案を提案するものです。コードの可読性、保守性、パフォーマンスの向上を目的とします。

## 2. 調査結果

### 2.1. データ処理・検証 (`app/utils/data_validation.py`)

- **対象箇所**: `DataValidator`クラス内の各種`safe_*`メソッド
- **問題点**: `numpy`や`pandas`の機能をラップしただけの関数が多く、コードが冗長になっています。特に、エラーハンドリングが各関数で個別に行われており、共通化の余地があります。
- **提案**:
    - `safe_divide`: `np.divide`や単純な除算と`replace`の組み合わせで実現可能です。
    - `safe_rolling_mean`: `series.rolling(window).mean().fillna(fill_value)` を直接利用することで、より簡潔に記述できます。
    - `safe_correlation`: `series1.rolling(window).corr(series2)` を利用できます。
- **メリット**: コードの簡潔化、`numpy`や`pandas`の最適化された実装によるパフォーマンス向上が期待できます。

### 2.2. 特徴量計算 (`app/services/ml/feature_engineering/`)

- **対象箇所**: `price_features.py`, `market_data_features.py` などの特徴量計算クラス
- **問題点**: 移動平均や変化率などの基本的な計算が、`DataValidator`のセーフラッパー経由で実装されており、冗長になっています。
- **提案**: `pandas`のメソッド（`.rolling()`, `.mean()`, `.std()`, `.pct_change()`など）を直接利用し、エラーハンドリングはデコレータなどで共通化します。
- **メリット**: コード量の削減、責務の明確化、`pandas`のパフォーマンス最適化の活用。

### 2.3. 評価指標計算 (`app/services/ml/models/*.py`)

- **対象箇所**: 各モデルラッパー（`lightgbm_wrapper.py`など）の`_train_model_impl`メソッド内の評価指標計算部分。
- **問題点**: `accuracy_score`, `f1_score`などを個別に呼び出しており、コードが重複しています。
- **提案**: `app.services.ml.evaluation.enhanced_metrics.EnhancedMetricsCalculator`に実装されている`calculate_comprehensive_metrics`メソッドを利用して、評価指標を一括で計算します。
- **メリット**: コードの重複排除、評価指標の追加・変更が容易になり、保守性が向上します。

### 2.4. データベース操作 (`app/utils/database_utils.py`)

- **対象箇所**: `_sqlite_insert_with_ignore`メソッド
- **問題点**: SQLiteでの重複無視挿入を、Pythonのループ処理で1件ずつ実装しています。
- **提案**: SQLAlchemyの`Insert`オブジェクトと`on_conflict_do_nothing`を組み合わせることで、より効率的な一括挿入が可能です。これはPostgreSQLの実装 (`_postgresql_insert_with_conflict`) と同様のアプローチです。
- **メリット**: データベースへのラウンドトリップが減少し、大幅なパフォーマンス向上が見込めます。

### 2.5. ポジションサイジング計算 (`app/services/auto_strategy/calculators/position_sizing_helper.py`)

- **対象箇所**: `_calculate_atr_from_data`メソッド
- **問題点**: ATR（Average True Range）の計算を独自に実装しています。
- **提案**: `talib.ATR`や、他のテクニカル分析ライブラリに存在する関数を利用します。`app.services.indicators.technical_indicators.volatility.VolatilityIndicators.atr` に既にTa-libを利用した実装が存在するため、これを再利用するのが望ましいです。
- **メリット**: 実績のあるライブラリを利用することによる信頼性の向上と、コードの簡素化。

## 3. まとめ

`backend`ディレクトリ全体を通して、`numpy`、`pandas`、`scikit-learn`、`talib`といったライブラリの機能をより直接的に活用することで、多くのカスタム実装を置き換えることが可能です。

特に、データ処理や検証、特徴量計算のユーティリティ関数は、ライブラリの標準的な使い方に統一することで、コード量を削減し、可読性とパフォーマンスを向上させることができます。また、評価指標の計算やデータベース操作に関しても、共通のサービスやORMの機能を活用することで、より堅牢で保守性の高いコードを実現できます。
