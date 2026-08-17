"""保存済み戦略をWFAで直接評価し、フォールドごとの取引数・リターンを確認するデバッグスクリプト。

使い方:
    uv run python -m scripts.debug_fold_trades results/auto_strategy/noseed_noval_2026-08-17.json
"""

import json
import sys
from pathlib import Path

from app.services.auto_strategy.config.ga_config import GAConfig
from app.services.auto_strategy.core.evaluation.evaluation_strategies import (
    EvaluationStrategy,
)
from app.services.auto_strategy.core.evaluation.individual_evaluator import (
    IndividualEvaluator,
)
from app.services.auto_strategy.serializers.serialization import GeneSerializer
from app.services.backtest.services.backtest_service import BacktestService


def main() -> None:
    if len(sys.argv) < 2:
        print("使用法: uv run python -m scripts.debug_fold_trades <result.json>")
        sys.exit(1)

    result_path = Path(sys.argv[1])
    with open(result_path, encoding="utf-8") as fh:
        data = json.load(fh)

    best = data.get("best_strategy")
    raw_gene = (best or {}).get("raw_gene")
    if not raw_gene:
        print("best_strategy.raw_gene がありません")
        sys.exit(1)

    serializer = GeneSerializer()
    gene = serializer.dict_to_strategy_gene(raw_gene)

    ga_config = GAConfig.from_dict(data["ga_config"])
    # 検証時と同じ WFA 設定に揃える
    vc = ga_config.validation_config
    ec = ga_config.evaluation_config
    ec.enable_walk_forward = True
    ec.wfa_n_folds = vc.wfa_n_folds
    ec.wfa_train_ratio = vc.wfa_train_ratio
    ec.wfa_anchored = vc.wfa_anchored
    ec.evaluation_mode = "walk_forward"
    ec.oos_split_ratio = 0.0

    backtest_config = data["backtest_config"]

    service = BacktestService()
    try:
        evaluator = IndividualEvaluator(service)
        strategy = EvaluationStrategy(evaluator, max_workers=2)
        report = strategy.execute_report(gene, backtest_config, ga_config)
        print(f"mode: {report.mode}, agg: {report.primary_aggregated_fitness}")
        for scenario in report.scenarios:
            pm = scenario.performance_metrics or {}
            meta = scenario.metadata or {}
            print(
                f"  {scenario.name}: fitness={list(scenario.fitness)} "
                f"passed={scenario.passed} "
                f"trades={pm.get('total_trades')} "
                f"return={pm.get('total_return')} "
                f"sharpe={pm.get('sharpe_ratio')} "
                f"max_dd={pm.get('max_drawdown')} "
                f"window={meta.get('start_date')}~{meta.get('end_date')}"
            )
            if pm.get("early_terminated"):
                print(f"    !! early_terminated: {meta.get('termination_reason')}")
    finally:
        service.cleanup()


if __name__ == "__main__":
    main()
