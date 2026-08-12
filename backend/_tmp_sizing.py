import glob
import json
import logging

logging.basicConfig(level=logging.WARNING)

from typing import Any


def main() -> None:
    files = sorted(glob.glob("results/auto_strategy/*.json"))
    d = json.load(open(files[-1], encoding="utf-8"))
    raw = d["best_strategy"]["raw_gene"]

    from app.services.auto_strategy.config.ga import GAConfig
    from app.services.auto_strategy.genes.position_sizing import (
        PositionSizingGene,
    )
    from app.services.auto_strategy.positions.position_sizing_service import (
        PositionSizingService,
    )

    gene = PositionSizingGene.from_dict(psg := raw["position_sizing_gene"])
    service = PositionSizingService()

    current_price = 43740.573619999996

    print("=== calculate_position_size_fast (trade_history=None) ===")
    size = service.calculate_position_size_fast(
        gene=gene,
        account_balance=100000.0,
        current_price=current_price,
        market_data=None,
    )
    print("result size:", size)

    print()
    print("=== _calculate_with_calculator 詳細 ===")
    from app.services.auto_strategy.utils.normalization import normalize_enum_name

    method_val = normalize_enum_name(gene.method)
    calc = service._calculator_factory.create_calculator(method_val)
    result = calc.calculate(
        gene,
        100000.0,
        current_price,
        market_data={},
        trade_history=None,
    )
    print("position_size:", result["position_size"])
    print("details:", json.dumps(result.get("details", {}), indent=1, ensure_ascii=False))
    print("warnings:", result.get("warnings"))

    print()
    print("=== entry_decision_engine の変換ロジックを再現 ===")
    position_size = float(result["position_size"])
    final_units = max(0.028387014244613075, min(31.23013040630173, position_size))
    print("final_units:", final_units)
    fraction = (final_units * current_price) / 100000.0
    print("fraction:", fraction)
    if fraction < 1.0:
        print("buy(size=fraction):", fraction)
    else:
        print("buy(size=floor(final_units)):", float(int(final_units)))


if __name__ == "__main__":
    main()