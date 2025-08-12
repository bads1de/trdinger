# `auto_strategy` パッケージのコード重複と統合に関する分析

`backend/app/services/auto_strategy/` ディレクトリ以下のコードベースを分析し、コードの重複、責務の集約、および統合の可能性について以下にまとめる。

## 1. TP/SL計算ロジックの重複と統合

**現状**:
- `calculators/tpsl_calculator.py`: 基本的なTP/SL価格計算ロジック。
- `services/tpsl_calculator_service.py`: 複数の計算方式（固定、RR比、ボラティリティ、統計）を統合する高レベルなサービス。
- `generators/statistical_tpsl_generator.py`, `generators/volatility_based_generator.py`: 特定の方式に基づいたTP/SL値を生成する。
- `models/gene_tpsl.py`: 遺伝子情報からTP/SL値を計算するメソッドを持つ。

**問題点**:
- `tpsl_calculator.py` と `tpsl_calculator_service.py` の役割が重複しており、どちらを主として使うべきか不明確。
- `StrategyFactory` が低レベルな `tpsl_calculator` を直接利用しており、高レベルなサービス (`tpsl_calculator_service`) をバイパスしている。

**提案**:
- **責務の集約**: `tpsl_calculator_service.py` を `TPSLService` のような名前に変更し、TP/SL計算に関する唯一のエントリーポイントとする。
- **ロジックの移動**: `tpsl_calculator.py` のロジックは、`TPSLService` のプライベートメソッドとして統合するか、より低レベルな計算ヘルパーとして明確に位置付ける。
- **依存関係の統一**: `StrategyFactory` を含め、すべてのTP/SL計算は新しい `TPSLService` を経由するように依存関係を修正する。これにより、計算ロジックが一元管理され、将来の拡張性が向上する。

## 2. ポジションサイジング計算ロジックの簡素化

**現状**:
- `calculators/position_sizing_calculator.py`: `PositionSizingCalculatorService` が遺伝子に基づいた詳細な計算とフォールバックロジックを持つ。
- `services/position_sizing_service.py`: `PositionSizingService` が `PositionSizingCalculatorService` をラップし、遺伝子がない場合のデフォルト処理を追加している。

**問題点**:
- `service` が `calculator` をラップする構造は冗長である可能性がある。2つのファイルにロジックが分散しており、見通しが悪い。

**提案**:
- **クラスの統合**: `PositionSizingService` のロジックを `PositionSizingCalculatorService` に統合し、`services/position_sizing_service.py` を削除する。
- **命名の変更**: 統合後のクラス名を `PositionSizingService` に変更し、責務が「ポジションサイジングに関するサービス全般」であることを明確にする。

## 3. 遺伝子関連処理の集約

**現状**:
- `models/gene_encoder.py`: `StrategyGene` を `list[float]` にエンコードする。
- `models/gene_decoder.py`: `list[float]` を `StrategyGene` にデコードする。
- `models/gene_serialization.py`: `StrategyGene` を `dict` や `json` にシリアライズ/デシリアライズする。

**問題点**:
- 戦略遺伝子の「表現形式の変換」という責務が3つのファイルに分散している。

**提案**:
- **責務の集約**: `gene_encoder.py` と `gene_decoder.py` の機能を `gene_serialization.py` の `GeneSerializer` クラスに統合する。
- **メソッドの追加**: `GeneSerializer` に `to_list()` と `from_list()` のようなメソッドを追加し、エンコード/デコード処理を担わせる。これにより、遺伝子の変換ロジックが一箇所にまとまり、管理が容易になる。

## 4. 条件生成ロジックの責務分離

**現状**:
- `generators/smart_condition_generator.py`: 指標の特性やスケールに基づいた複雑な条件生成ロジックが集中している。

**問題点**:
- 単一のクラスに多くのロジックが詰め込まれており、可読性とメンテナンス性が低い。
- 指標の閾値やトレンド判定などの汎用的なロジックが、このジェネレーター内に閉じてしまっている。

**提案**:
- **ポリシーの抽出**: `smart_condition_generator.py` 内の汎用的なロジックを、`core/` ディレクトリ内の新しいポリシーファイルとして抽出する。
    - 例: `core/momentum_policy.py`, `core/volatility_policy.py` など。
    - 既存の `core/threshold_policy.py` を拡張し、より多くの指標の閾値決定ロジックを担わせる。
- **ジェネレーターの責務削減**: `smart_condition_generator.py` は、抽出された各ポリシーを組み合わせて条件を組み立てる「オーケストレーター」としての役割に専念させる。

## 5. サービスレイヤーの再検討

**現状**:
- `orchestration/auto_strategy_orchestration_service.py` (API層に近い) -> `services/auto_strategy_service.py` (薄いファサード) -> `managers/experiment_manager.py` (実質的な処理) という呼び出し階層になっている。

**問題点**:
- `auto_strategy_service.py` が非常に薄いラッパーとして機能しており、抽象化レイヤーとして冗長に感じられる可能性がある。

**提案**:
- **階層の簡素化**: `AutoStrategyOrchestrationService` が直接 `ExperimentManager` を利用する構成を検討する。
- **責務の明確化**: もし `AutoStrategyService` を残すのであれば、単なるファサード以上の明確なビジネスロジック（複数のマネージャーを協調させるなど）を持たせるように責務を再定義する。現状の機能範囲であれば、階層を一つ減らすことでコードの追跡が容易になる。
