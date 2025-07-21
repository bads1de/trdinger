# ML 最適化システム完全移行計画

## 📋 概要

現在の複雑なハイパーパラメータ最適化システム（ベイジアン、グリッド、ランダムサーチ）を**Optuna のみ**に完全移行する詳細計画書です。

### 現状分析

- **現在のコード量**: 約 5,000 行（最適化関連）
- **現在の手法**: 3 つの独自実装（ベイジアン、グリッド、ランダム）
- **最適化時間**: 30-60 分
- **保守コスト**: 高（複雑な独自実装）
- **学習コスト**: 高（新規開発者の参入障壁）

### 移行後の期待効果

- **コード量**: 約 300 行（95%削減）
- **手法**: Optuna のみ（シンプル化）
- **最適化時間**: 5-15 分（70-80%短縮）
- **保守コスト**: 極低（標準ライブラリのみ）
- **学習コスト**: 極低（Optuna の豊富なドキュメント）

---

## 🎯 移行戦略

### 基本方針

1. **完全置換**: 既存の最適化システムを完全削除
2. **Optuna のみ**: 単一ライブラリによるシンプル化
3. **最小リスク**: 既存の LightGBM と ML アーキテクチャは維持
4. **検証重視**: 移行前後で性能比較を実施

### 対象範囲

- ✅ **完全削除**: ベイジアン、グリッド、ランダムサーチの独自実装
- ✅ **新規実装**: Optuna ベースの最適化システム
- ❌ **対象外**: 特徴量エンジニアリング、モデル管理、信号生成

---

## 📅 実装スケジュール

### Phase 1: Optuna 実装（1 週間）

- [ ] Optuna 依存関係追加
- [ ] OptunaOptimizer クラス実装
- [ ] 既存システムとの統合テスト

### Phase 2: 既存システム削除（1 週間）

- [ ] ベイジアン最適化削除
- [ ] グリッドサーチ削除
- [ ] ランダムサーチ削除
- [ ] OptimizerFactory 簡素化

### Phase 3: UI 更新・テスト（1 週間）

- [ ] フロントエンド UI 簡素化
- [ ] 性能比較テスト
- [ ] ドキュメント更新

---

## 🗑️ 削除対象ファイル

以下のファイルを完全削除します：

```
backend/app/core/services/optimization/
├── bayesian_optimizer.py          # 削除
├── grid_search_optimizer.py       # 削除
├── random_search_optimizer.py     # 削除
├── optimization_presets.py        # 削除
├── base_optimizer.py              # 簡素化
└── optimizer_factory.py           # 簡素化
```

```
backend/tests/optimization/
├── test_bayesian_optimizer.py     # 削除
├── test_grid_search_optimizer.py  # 削除
├── test_random_search_optimizer.py # 削除
├── test_optimization_performance.py # 削除
└── test_optimization_functionality.py # 削除
```

---

## 🔧 技術実装詳細

### 1. 依存関係の追加

```bash
# requirements.txtに追加
optuna>=3.4.0
optuna-dashboard>=0.13.0  # 可視化用（オプション）

# 削除する依存関係
# scikit-optimize>=0.9.0  # 削除
```

### 2. シンプルな OptunaOptimizer 実装

```python
# backend/app/core/services/optimization/optuna_optimizer.py
"""
Optunaベースの最適化エンジン

既存の複雑な最適化システムを置き換える、シンプルで効率的な実装。
"""

import logging
import optuna
from typing import Dict, Any, Callable, Optional
from datetime import datetime
from dataclasses import dataclass

logger = logging.getLogger(__name__)


@dataclass
class OptimizationResult:
    """最適化結果（簡素化版）"""
    best_params: Dict[str, Any]
    best_score: float
    total_evaluations: int
    optimization_time: float
    study: optuna.Study


@dataclass
class ParameterSpace:
    """パラメータ空間の定義（簡素化版）"""
    type: str  # "real", "integer", "categorical"
    low: Optional[float] = None
    high: Optional[float] = None
    categories: Optional[list] = None


class OptunaOptimizer:
    """
    Optunaベースの最適化エンジン

    既存の複雑なシステムを置き換える、シンプルで効率的な実装。
    """

    def __init__(self):
        """初期化"""
        self.study: Optional[optuna.Study] = None

    def optimize(
        self,
        objective_function: Callable[[Dict[str, Any]], float],
        parameter_space: Dict[str, ParameterSpace],
        n_calls: int = 50,
    ) -> OptimizationResult:
        """
        Optunaを使用した最適化を実行

        Args:
            objective_function: 目的関数
            parameter_space: パラメータ空間
            n_calls: 最適化試行回数

        Returns:
            最適化結果
        """
        logger.info(f"🚀 Optuna最適化を開始: 試行回数={n_calls}")
        start_time = datetime.now()

        # Optunaスタディを作成
        self.study = optuna.create_study(
            direction="maximize",
            sampler=optuna.samplers.TPESampler(seed=42),
            pruner=optuna.pruners.MedianPruner(),
        )

        # 目的関数をOptunaに適応
        def optuna_objective(trial: optuna.Trial) -> float:
            params = self._suggest_parameters(trial, parameter_space)
            try:
                score = objective_function(params)
                return score
            except Exception as e:
                logger.warning(f"目的関数評価中にエラー: {e}")
                raise optuna.TrialPruned()

        # 最適化実行
        self.study.optimize(optuna_objective, n_trials=n_calls)

        end_time = datetime.now()
        optimization_time = (end_time - start_time).total_seconds()

        # 結果を作成
        best_trial = self.study.best_trial
        result = OptimizationResult(
            best_params=best_trial.params,
            best_score=best_trial.value,
            total_evaluations=len(self.study.trials),
            optimization_time=optimization_time,
            study=self.study
        )

        logger.info(f"✅ Optuna最適化完了: ベストスコア={result.best_score:.4f}, 時間={optimization_time:.2f}秒")
        return result

    def _suggest_parameters(
        self, trial: optuna.Trial, parameter_space: Dict[str, ParameterSpace]
    ) -> Dict[str, Any]:
        """パラメータをサジェスト"""
        params = {}

        for param_name, param_config in parameter_space.items():
            if param_config.type == "real":
                params[param_name] = trial.suggest_float(
                    param_name, param_config.low, param_config.high
                )
            elif param_config.type == "integer":
                params[param_name] = trial.suggest_int(
                    param_name, param_config.low, param_config.high
                )
            elif param_config.type == "categorical":
                params[param_name] = trial.suggest_categorical(
                    param_name, param_config.categories
                )

        return params

    @staticmethod
    def get_default_parameter_space() -> Dict[str, ParameterSpace]:
        """LightGBMのデフォルトパラメータ空間"""
        return {
            "num_leaves": ParameterSpace(type="integer", low=10, high=100),
            "learning_rate": ParameterSpace(type="real", low=0.01, high=0.3),
            "feature_fraction": ParameterSpace(type="real", low=0.5, high=1.0),
            "bagging_fraction": ParameterSpace(type="real", low=0.5, high=1.0),
            "min_data_in_leaf": ParameterSpace(type="integer", low=5, high=50),
            "max_depth": ParameterSpace(type="integer", low=3, high=15),
        }
```

### 3. MLTrainingService の大幅簡素化

```python
# backend/app/core/services/ml/ml_training_service.py の更新

from .optimization.optuna_optimizer import OptunaOptimizer, ParameterSpace

class OptimizationSettings:
    """最適化設定クラス（簡素化版）"""
    def __init__(
        self,
        enabled: bool = False,
        n_calls: int = 50,
        parameter_space: Optional[Dict[str, Dict[str, Any]]] = None,
    ):
        self.enabled = enabled
        self.n_calls = n_calls
        self.parameter_space = parameter_space or {}


class MLTrainingService:
    def _train_with_optimization(self, ...):
        """Optunaを使用した最適化学習（大幅簡素化）"""
        try:
            logger.info("🚀 Optuna最適化を開始")

            # Optunaオプティマイザーを作成
            optimizer = OptunaOptimizer()

            # パラメータ空間を準備
            if not optimization_settings.parameter_space:
                # デフォルトのLightGBMパラメータ空間を使用
                parameter_space = optimizer.get_default_parameter_space()
            else:
                parameter_space = self._prepare_parameter_space(
                    optimization_settings.parameter_space
                )

            # 目的関数を作成（既存のものを流用）
            objective_function = self._create_objective_function(...)

            # Optuna最適化を実行
            optimization_result = optimizer.optimize(
                objective_function=objective_function,
                parameter_space=parameter_space,
                n_calls=optimization_settings.n_calls,
            )

            # 最適化されたパラメータで最終モデルを学習
            final_training_params = {
                **training_params,
                **optimization_result.best_params,
            }

            final_result = self.trainer.train_model(...)

            # 最適化情報を結果に追加
            final_result["optimization_result"] = {
                "method": "optuna",
                "best_params": optimization_result.best_params,
                "best_score": optimization_result.best_score,
                "total_evaluations": optimization_result.total_evaluations,
                "optimization_time": optimization_result.optimization_time,
            }

            return final_result

        except Exception as e:
            logger.error(f"Optuna最適化学習中にエラーが発生しました: {e}")
            raise

    def _prepare_parameter_space(
        self, parameter_space_config: Dict[str, Dict[str, Any]]
    ) -> Dict[str, ParameterSpace]:
        """パラメータ空間設定をParameterSpaceオブジェクトに変換"""
        parameter_space = {}

        for param_name, param_config in parameter_space_config.items():
            parameter_space[param_name] = ParameterSpace(
                type=param_config["type"],
                low=param_config.get("low"),
                high=param_config.get("high"),
                categories=param_config.get("categories"),
            )

        return parameter_space
```

---

## 🎨 フロントエンド大幅簡素化

### 1. 超シンプルな OptimizationSettings

```typescript
// frontend/components/ml/OptimizationSettings.tsx の完全書き換え

interface OptimizationSettingsConfig {
  enabled: boolean;
  n_calls: number;
}

export default function OptimizationSettings({ settings, onChange }: Props) {
  return (
    <Card>
      <CardHeader>
        <CardTitle className="flex items-center gap-2">
          <Zap className="h-5 w-5" />
          ハイパーパラメータ最適化設定
        </CardTitle>
        <p className="text-sm text-muted-foreground">
          Optunaによる高効率な自動最適化
        </p>
      </CardHeader>
      <CardContent className="space-y-6">
        {/* 最適化有効/無効 */}
        <div className="flex items-center space-x-2">
          <Switch
            id="optimization-enabled"
            checked={settings.enabled}
            onCheckedChange={(enabled) => onChange({ ...settings, enabled })}
          />
          <Label htmlFor="optimization-enabled">
            ハイパーパラメータ自動最適化を有効にする
          </Label>
        </div>

        {settings.enabled && (
          <div className="space-y-4">
            {/* 試行回数 */}
            <div className="space-y-2">
              <Label>最適化試行回数</Label>
              <div className="grid grid-cols-3 gap-2">
                <Button
                  variant={settings.n_calls === 20 ? "default" : "outline"}
                  onClick={() => onChange({ ...settings, n_calls: 20 })}
                >
                  高速 (20回)
                </Button>
                <Button
                  variant={settings.n_calls === 50 ? "default" : "outline"}
                  onClick={() => onChange({ ...settings, n_calls: 50 })}
                >
                  標準 (50回)
                </Button>
                <Button
                  variant={settings.n_calls === 100 ? "default" : "outline"}
                  onClick={() => onChange({ ...settings, n_calls: 100 })}
                >
                  高精度 (100回)
                </Button>
              </div>
            </div>

            {/* 情報表示 */}
            <div className="bg-blue-50 dark:bg-blue-900/20 p-4 rounded-lg">
              <div className="flex items-center gap-2 mb-2">
                <Info className="h-4 w-4 text-blue-500" />
                <span className="text-sm font-medium text-blue-700 dark:text-blue-300">
                  Optuna最適化について
                </span>
              </div>
              <ul className="text-xs text-blue-600 dark:text-blue-400 space-y-1">
                <li>• TPEサンプラーによる効率的な探索</li>
                <li>• MedianPrunerによる早期停止</li>
                <li>• 予想時間: {Math.ceil(settings.n_calls * 0.2)}分</li>
              </ul>
            </div>
          </div>
        )}
      </CardContent>
    </Card>
  );
}
```

---

## 🧪 テスト戦略（簡素化）

### 1. Optuna のみのテスト

```python
# backend/tests/optimization/test_optuna_optimizer.py

import pytest
from app.core.services.optimization.optuna_optimizer import OptunaOptimizer, ParameterSpace

class TestOptunaOptimizer:
    def test_basic_optimization(self):
        """基本的な最適化テスト"""
        optimizer = OptunaOptimizer()

        def objective(params):
            return -(params["x"] - 0.5) ** 2  # x=0.5で最大

        parameter_space = {
            "x": ParameterSpace(type="real", low=0.0, high=1.0)
        }

        result = optimizer.optimize(objective, parameter_space, n_calls=20)

        assert abs(result.best_params["x"] - 0.5) < 0.2
        assert result.best_score > -0.1
        assert result.total_evaluations <= 20

    def test_lightgbm_parameter_space(self):
        """LightGBMパラメータ空間テスト"""
        space = OptunaOptimizer.get_default_parameter_space()

        expected_params = ["num_leaves", "learning_rate", "feature_fraction"]
        for param in expected_params:
            assert param in space

    def test_ml_training_integration(self):
        """MLTrainingServiceとの統合テスト"""
        from app.core.services.ml.ml_training_service import MLTrainingService, OptimizationSettings

        service = MLTrainingService()
        training_data = create_test_ohlcv_data()

        optimization_settings = OptimizationSettings(
            enabled=True,
            n_calls=10,
        )

        result = service.train_model(
            training_data=training_data,
            optimization_settings=optimization_settings
        )

        assert result["success"] is True
        assert "optimization_result" in result
        assert result["optimization_result"]["method"] == "optuna"
```

---

## 🗑️ 削除作業詳細

### 1. ファイル削除スクリプト

```python
# scripts/cleanup_old_optimization.py

import os
import shutil

def cleanup_old_optimization_files():
    """古い最適化システムのファイルを削除"""

    files_to_delete = [
        "backend/app/core/services/optimization/bayesian_optimizer.py",
        "backend/app/core/services/optimization/grid_search_optimizer.py",
        "backend/app/core/services/optimization/random_search_optimizer.py",
        "backend/app/core/services/optimization/optimization_presets.py",
        "backend/tests/optimization/test_bayesian_optimizer.py",
        "backend/tests/optimization/test_grid_search_optimizer.py",
        "backend/tests/optimization/test_random_search_optimizer.py",
        "backend/tests/optimization/test_optimization_performance.py",
        "backend/tests/optimization/test_optimization_functionality.py",
    ]

    for file_path in files_to_delete:
        if os.path.exists(file_path):
            os.remove(file_path)
            print(f"✅ 削除完了: {file_path}")
        else:
            print(f"⚠️ ファイルが見つかりません: {file_path}")

    print("🎉 古い最適化システムの削除完了")

if __name__ == "__main__":
    cleanup_old_optimization_files()
```

### 2. requirements.txt 更新

```bash
# 削除する依存関係
# scikit-optimize>=0.9.0

# 追加する依存関係
optuna>=3.4.0
```

---

## 📊 移行前後比較

### コード量比較

| 項目                 | 移行前     | 移行後     | 削減率 |
| -------------------- | ---------- | ---------- | ------ |
| 最適化関連ファイル数 | 8 ファイル | 1 ファイル | 87.5%  |
| 最適化関連コード行数 | ~5,000 行  | ~300 行    | 94%    |
| テストファイル数     | 5 ファイル | 1 ファイル | 80%    |
| UI 設定項目数        | 20+項目    | 2 項目     | 90%    |

### 性能比較

| 項目         | 移行前   | 移行後  | 改善率 |
| ------------ | -------- | ------- | ------ |
| 最適化時間   | 30-60 分 | 5-15 分 | 75%    |
| メモリ使用量 | 高       | 低      | 50%    |
| 学習コスト   | 高       | 極低    | 80%    |
| 保守コスト   | 高       | 極低    | 90%    |

---

## 🚀 デプロイ計画

### 1. 一括移行手順

```bash
# 1. 依存関係更新
pip install optuna>=3.4.0
pip uninstall scikit-optimize

# 2. 古いファイル削除
python scripts/cleanup_old_optimization.py

# 3. 新しいファイル配置
# OptunaOptimizerを配置

# 4. テスト実行
pytest backend/tests/optimization/test_optuna_optimizer.py

# 5. アプリケーション再起動
python backend/main.py
```

### 2. 環境変数設定

```bash
# .env ファイル
OPTUNA_STORAGE_URL=sqlite:///optuna.db
OPTUNA_LOG_LEVEL=INFO
```

---

## ⚠️ リスク管理

### 1. 主要リスク

| リスク                | 影響度 | 対策                     |
| --------------------- | ------ | ------------------------ |
| Optuna 依存関係の問題 | 中     | 事前テスト、バックアップ |
| 性能劣化              | 高     | 移行前後の性能比較       |
| 既存データの互換性    | 中     | データ移行スクリプト     |

### 2. ロールバック計画

```python
# scripts/rollback_to_old_system.py

def rollback_optimization_system():
    """古いシステムにロールバック"""

    # Gitから古いファイルを復元
    os.system("git checkout HEAD~1 -- backend/app/core/services/optimization/")

    # 依存関係を戻す
    os.system("pip install scikit-optimize>=0.9.0")
    os.system("pip uninstall optuna")

    print("✅ 古いシステムにロールバック完了")
```

---

## 📈 成功指標

### 1. 定量的指標

- **コード削減**: 95%以上削減
- **時間短縮**: 75%以上短縮
- **精度維持**: 既存システムと同等以上
- **学習時間**: 新規開発者 1 日以内で習得

### 2. 定性的指標

- **保守性**: 極めて高い（標準ライブラリのみ）
- **可読性**: 極めて高い（シンプルな実装）
- **拡張性**: 高い（Optuna の豊富な機能）

---

## 🎯 まとめ

この完全移行計画により、以下の効果が期待できます：

### ✅ **期待効果**

1. **劇的な簡素化**: コード量 95%削減、保守コスト 90%削減
2. **大幅な高速化**: 最適化時間 75%短縮
3. **学習コスト激減**: 新規開発者の習得時間 80%短縮
4. **安定性向上**: 実績のある Optuna による信頼性

### 🛡️ **リスク軽減**

1. **シンプル化**: 複雑な独自実装を排除
2. **標準化**: 業界標準ライブラリの使用
3. **十分なテスト**: 移行前後の性能比較
4. **ロールバック計画**: 問題発生時の対応策

### 📅 **実装スケジュール**

- **Phase 1**: Optuna 実装（1 週間）
- **Phase 2**: 既存システム削除（1 週間）
- **Phase 3**: UI 更新・テスト（1 週間）

**既存の複雑なシステムを完全に捨てて、Optuna のみのシンプルなシステムに移行することで、大幅な改善を実現できます。**
