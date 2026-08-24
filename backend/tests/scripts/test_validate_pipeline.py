"""validate_pipeline ヘルパーのテスト."""

import math

from scripts.validate_pipeline import (
    build_oos_config,
    extract_candidate_records,
    norm_gene,
    pearson,
    rank,
    regime_min_margin_pp,
    spearman,
)


class TestCorrelations:
    def test_pearson_perfect_positive(self):
        assert pearson([1.0, 2.0, 3.0], [2.0, 4.0, 6.0]) > 0.999

    def test_pearson_insufficient_n(self):
        assert math.isnan(pearson([1.0, 2.0], [1.0, 2.0]))
        assert math.isnan(pearson([], []))

    def test_pearson_zero_variance(self):
        assert math.isnan(pearson([1.0, 1.0, 1.0], [1.0, 2.0, 3.0]))

    def test_spearmon_rank_ties(self):
        xs = [1.0, 1.0, 2.0]
        rs = rank(xs)
        assert rs[0] == rs[1] == 1.5
        assert rs[2] == 3.0
        # 単調増加なら +1
        assert spearman([1.0, 2.0, 3.0, 4.0], [10.0, 20.0, 30.0, 40.0]) > 0.999


class TestNormGene:
    def test_ignores_ids_and_timestamps(self):
        a = {"id": "aaaa1111", "generated_at": "2026-01-01", "x": 1}
        b = {"id": "ffff2222", "generated_at": "2027-01-01", "x": 1}
        assert norm_gene(a) == norm_gene(b)

    def test_differs_on_real_change(self):
        a = {"parameters": {"length": 20}}
        b = {"parameters": {"length": 22}}
        assert norm_gene(a) != norm_gene(b)


class TestRegimeMinMargin:
    def test_returns_min_times_hundred(self):
        v = {
            "regime_consistency": {"slices": [{"excess": 0.13}, {"excess": -0.02}, {}]}
        }
        assert regime_min_margin_pp(v) == -2.0

    def test_missing_returns_none(self):
        assert regime_min_margin_pp({}) is None


def _result_fixture() -> dict:
    strategies = {
        "gid-aaaa": {"indicators": [], "id": "gid-aaaa"},
        "gid-bbbb": {"indicators": [{"type": "RSI"}], "id": "gid-bbbb"},
    }
    validations = {
        "gid-aaaa": {
            "passed": True,
            "primary_fitness": 1.5,
            "reasons": [],
            "report_summary": {
                "scenarios": [
                    {
                        "performance_metrics": {
                            "total_return": 0.10,
                            "buy_hold_return": 0.05,
                        }
                    }
                ]
            },
            "param_perturbation": {"survival_rate": 1.0},
            "regime_consistency": {"slices": [{"excess": 0.08}]},
        },
        "gid-bbbb": {
            "passed": False,
            "primary_fitness": None,
            "reasons": ["WFA合格率 0.00 が閾値未満"],
            "report_summary": {},
        },
    }
    return {
        "validated_strategies": strategies,
        "validation_results": validations,
    }


class TestExtractCandidateRecords:
    def test_extracts_metrics_and_dedupes(self):
        result = _result_fixture()
        records = extract_candidate_records(result)
        assert len(records) == 2
        passed = next(r for r in records if r["passed"])
        assert passed["fitness"] == 1.5
        assert passed["is_excess_pp"] == (0.10 - 0.05) * 100
        assert passed["pert_survival"] == 1.0
        assert passed["regime_min_margin_pp"] == 8.0
        failed = next(r for r in records if not r["passed"])
        assert failed["fitness"] is None
        assert failed["reasons"]

    def test_duplicate_genes_collapse(self):
        result = _result_fixture()
        result["validated_strategies"]["gid-cccc"] = {"indicators": [], "id": "zzzz"}
        result["validation_results"]["gid-cccc"] = dict(
            result["validation_results"]["gid-aaaa"]
        )
        records = extract_candidate_records(result)
        assert len(records) == 2


class TestBuildOOSConfig:
    def test_shifts_by_period_length(self):
        cfg = build_oos_config(
            {
                **{
                    "symbol": "BTC/USDT:USDT",
                    "timeframe": "4h",
                    "initial_capital": 100000.0,
                    "commission_rate": 0.0004,
                    "slippage": 0.0001,
                },
                "start_date": "2022-01-01",
                "end_date": "2022-06-30",
            }
        )
        assert cfg is not None
        assert cfg["start_date"] == "2022-07-01"
        # 同長窓（180日）なので H2 の末日ではなく 12/28
        assert cfg["end_date"] == "2022-12-28"

    def test_invalid_dates_return_none(self):
        assert build_oos_config({"start_date": "", "end_date": ""}) is None


class TestGeneHashAndHistoryDedup:
    """遺伝子ハッシュと履歴重複排除のテスト."""

    def test_gene_hash_stable_across_ids(self):
        from scripts.validate_pipeline import gene_hash

        a = {"id": "aaaa1111-1111-1111-1111-111111111111", "x": 1}
        b = {"id": "ffff2222-2222-2222-2222-222222222222", "x": 1}
        assert gene_hash(a) == gene_hash(b)
        assert len(gene_hash(a)) == 12

    def test_extract_includes_gene_hash(self):
        records = extract_candidate_records(_result_fixture())
        assert all(len(r["gene_hash"]) == 12 for r in records)

    def test_select_unique_rows_by_hash(self):
        from scripts.validate_pipeline import select_unique_rows

        rows = [
            {"gene_hash": "AAA", "oos_excess_pp": 1.0},
            {"gene_hash": "AAA", "oos_excess_pp": 9.9},  # 重複
            {"gene_hash": "BBB", "oos_excess_pp": -1.0},
        ]
        out = select_unique_rows(rows)
        assert len(out) == 2

    def test_legacy_rows_deduped_by_metric_tuple(self):
        from scripts.validate_pipeline import select_unique_rows

        legacy = {
            "period": "2022H1",
            "fitness": 0.5,
            "regime_min_margin_pp": 10.0,
            "pert_survival": 1.0,
            "is_excess_pp": -20.0,
            "oos_excess_pp": 25.0,
        }
        rows = [dict(legacy), dict(legacy), {**legacy, "fitness": 0.6}]
        out = select_unique_rows(rows)
        assert len(out) == 2

    def test_append_history_skips_duplicates(self, tmp_path):
        from scripts.validate_pipeline import append_history

        target = tmp_path / "history.jsonl"
        first = [{"gene_hash": "AAA", "period": "P", "passed": True}]
        second = [
            {"gene_hash": "AAA", "period": "P", "passed": True},
            {"gene_hash": "BBB", "period": "P", "passed": True},
        ]
        assert append_history(target, first) == 1
        assert append_history(target, second) == 1
        lines = target.read_text(encoding="utf-8").splitlines()
        assert len(lines) == 2


class TestCrossSymbol:
    """クロスシンボルOOS評価サポートのテスト."""

    def test_build_oos_config_symbol_override(self):
        cfg = build_oos_config(
            {
                "start_date": "2022-01-01",
                "end_date": "2022-06-30",
                **{
                    "symbol": "BTC/USDT:USDT",
                    "timeframe": "4h",
                },
            },
            symbol="ETH/USDT:USDT",
        )
        assert cfg is not None
        assert cfg["symbol"] == "ETH/USDT:USDT"

    def test_record_key_distinguishes_eval_symbol(self):
        from scripts.validate_pipeline import record_key

        base = {"gene_hash": "AAA"}
        btc = record_key(base)
        eth = record_key({**base, "eval_symbol": "ETH/USDT:USDT"})
        assert btc != eth

    def test_select_unique_keeps_same_gene_across_symbols(self):
        from scripts.validate_pipeline import select_unique_rows

        rows = [
            {"gene_hash": "AAA"},
            {"gene_hash": "AAA", "eval_symbol": "ETH/USDT:USDT"},
            {"gene_hash": "AAA", "eval_symbol": "ETH/USDT:USDT"},  # 完全重複
        ]
        assert len(select_unique_rows(rows)) == 2
