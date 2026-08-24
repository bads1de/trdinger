"""validate_pipeline ヘルパーのテスト."""

import json
import math

from scripts.validate_pipeline import (
    append_history,
    build_oos_config,
    build_oos_row,
    extract_candidate_records,
    load_history,
    norm_gene,
    normalize_legacy_row,
    pearson,
    rank,
    record_key,
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


class TestGASymbolSupport:
    """GAフェーズのシンボル対応テスト."""

    def test_build_oos_config_defaults_to_result_symbol(self):
        from scripts.validate_pipeline import build_oos_config

        cfg = build_oos_config(
            {
                "symbol": "ETH/USDT:USDT",
                "start_date": "2025-01-01",
                "end_date": "2025-06-30",
            }
        )
        # 結果ファイルのシンボルを継承する（BTC定数に落ちない）
        assert cfg is not None
        assert cfg["symbol"] == "ETH/USDT:USDT"

    def test_find_latest_result_filters_by_symbol(self, tmp_path, monkeypatch):
        import scripts.validate_pipeline as vp

        results_dir = tmp_path / "results" / "auto_strategy"
        results_dir.mkdir(parents=True)

        payload_btc = {
            "backtest_config": {
                "symbol": "BTC/USDT:USDT",
                "start_date": "2025-01-01",
                "end_date": "2025-06-30",
            }
        }
        payload_eth = {
            "backtest_config": {
                "symbol": "ETH/USDT:USDT",
                "start_date": "2025-01-01",
                "end_date": "2025-06-30",
            }
        }
        (results_dir / "strategy_a.json").write_text(
            json.dumps(payload_btc), encoding="utf-8"
        )
        (results_dir / "strategy_b.json").write_text(
            json.dumps(payload_eth), encoding="utf-8"
        )

        monkeypatch.setattr(vp, "PROJECT_ROOT", tmp_path)

        hit = vp.find_latest_result_for_period(
            "2025-01-01", "2025-06-30", symbol="ETH/USDT:USDT"
        )
        assert hit is not None and hit.name == "strategy_b.json"

        miss = vp.find_latest_result_for_period(
            "2025-01-01", "2025-06-30", symbol="SOL/USDT:USDT"
        )
        assert miss is None


class TestBuildOosRowSymbolSemantics:
    """build_oos_row / normalize_legacy_row のシンボル意味論テスト。

    回帰: native実行(ETH)行がモジュール定数BTCへフォールバックし、
    クロス行(eval=BTC)と誤重複していた問題。
    """

    CANDIDATE = {"gid": "abc", "gene_hash": "h1", "passed": True}
    OUTCOME = {"oos_excess_pp": 1.5}

    def test_native_row_eval_symbol_equals_ga_symbol(self):
        row = build_oos_row(
            "2022H1", "ETH/USDT:USDT", dict(self.CANDIDATE), dict(self.OUTCOME), None
        )
        assert row["eval_symbol"] == "ETH/USDT:USDT"
        assert row["cross_symbol"] is False

    def test_cross_row_flags_when_eval_differs_from_ga(self):
        row = build_oos_row(
            "2022H1",
            "ETH/USDT:USDT",
            dict(self.CANDIDATE),
            dict(self.OUTCOME),
            "BTC/USDT:USDT",
        )
        assert row["eval_symbol"] == "BTC/USDT:USDT"
        assert row["cross_symbol"] is True

    def test_cross_row_false_when_eval_equals_ga_explicitly(self):
        row = build_oos_row(
            "2022H1",
            "BTC/USDT:USDT",
            dict(self.CANDIDATE),
            dict(self.OUTCOME),
            "BTC/USDT:USDT",
        )
        assert row["cross_symbol"] is False

    def test_normalize_legacy_row_prefers_ga_symbol_over_constant(self):
        legacy = {"ga_symbol": "ETH/USDT:USDT", "oos_excess_pp": -14.0}
        fixed = normalize_legacy_row(legacy)
        assert fixed["eval_symbol"] == "ETH/USDT:USDT"
        assert fixed["cross_symbol"] is False

    def test_normalize_legacy_row_oldest_defaults_to_btc_constant(self):
        fixed = normalize_legacy_row({"oos_excess_pp": 20.0})
        assert fixed["eval_symbol"] == "BTC/USDT:USDT"
        assert fixed["cross_symbol"] is False

    def test_normalize_is_idempotent_and_preserves_existing_flags(self):
        row = {
            "ga_symbol": "ETH/USDT:USDT",
            "eval_symbol": "BTC/USDT:USDT",
            "cross_symbol": True,
        }
        once = normalize_legacy_row(dict(row))
        assert normalize_legacy_row(dict(once)) == once

    def test_append_history_distinguishes_same_gene_across_eval_symbols(self, tmp_path):

        hist = tmp_path / "history.jsonl"
        native = build_oos_row(
            "2022H1", "ETH/USDT:USDT", dict(self.CANDIDATE), dict(self.OUTCOME), None
        )
        cross = build_oos_row(
            "2022H1",
            "ETH/USDT:USDT",
            dict(self.CANDIDATE),
            dict(self.OUTCOME),
            "BTC/USDT:USDT",
        )
        # 遺伝子同一でも評価シンボルが違えば別観測として両方記録される
        assert append_history(hist, [native]) == 1
        assert append_history(hist, [cross]) == 1
        assert len(list(hist.read_text(encoding="utf-8").splitlines())) == 2
        # 同一キーの再追加はスキップ
        assert append_history(hist, [dict(cross)]) == 0
        keys = {record_key(r) for r in load_history(hist)}
        assert len(keys) == 2


class TestRegimeTagging:
    """classify_regime / レジーム列付与 / シード統合のテスト"""

    def test_classify_regime_boundaries(self):
        from scripts.validate_pipeline import classify_regime

        assert classify_regime(20.1) == "bull"
        assert classify_regime(-20.1) == "bear"
        assert classify_regime(20.0) == "sideways"
        assert classify_regime(-20.0) == "sideways"
        assert classify_regime(0.0) == "sideways"

    def test_build_oos_row_adds_regime_columns(self):
        cand = {"gid": "g1", "gene_hash": "h", "passed": True, "is_bh_pp": -56.0}
        outcome = {"oos_excess_pp": 5.0, "oos_return_pp": 2.0, "oos_bh_pp": -64.0}
        row = build_oos_row("2022H1", "BTC/USDT:USDT", cand, outcome, None)
        assert row["is_regime"] == "bear"
        assert row["oos_regime"] == "bear"
        assert row["regime_match"] is True

    def test_build_oos_row_regime_mismatch(self):
        cand = {"gid": "g1", "gene_hash": "h", "is_bh_pp": 70.0}
        outcome = {"oos_excess_pp": -3.0, "oos_bh_pp": -10.0}
        row = build_oos_row("2023H1", "BTC/USDT:USDT", cand, outcome, None)
        assert row["is_regime"] == "bull"
        assert row["oos_regime"] == "sideways"
        assert row["regime_match"] is False

    def test_build_oos_row_records_seed_values(self):
        row = build_oos_row(
            "2022H1",
            "BTC/USDT:USDT",
            {"gid": "x", "gene_hash": "h"},
            {"oos_excess_pp": 1.0},
            None,
            seed_values=[42, 43],
        )
        assert row["seed_values"] == [42, 43]

    def test_extract_candidate_records_includes_is_bh(self, tmp_path):

        bh = 0.1234
        result = {
            "validated_strategies": {"a1": {"entry_rules": [], "exit_rules": []}},
            "validation_results": {
                "a1": {
                    "passed": True,
                    "primary_fitness": 0.5,
                    "report_summary": {
                        "scenarios": [
                            {
                                "performance_metrics": {
                                    "total_return": 0.2,
                                    "buy_hold_return": bh,
                                }
                            }
                        ]
                    },
                }
            },
        }
        recs = extract_candidate_records(result)
        assert len(recs) == 1
        assert math.isclose(recs[0]["is_bh_pp"], bh * 100.0)

    def test_merge_result_files_dedupes_and_namespaces_gids(self, tmp_path):
        import json as _json

        from scripts.validate_pipeline import merge_result_files

        gene_a = {"entry_rules": [{"type": "sma_crossover"}], "exit_rules": []}
        gene_b = {"entry_rules": [{"type": "rsi_extremes"}], "exit_rules": []}
        f1 = tmp_path / "r1.json"
        f2 = tmp_path / "r2.json"
        f1.write_text(
            _json.dumps(
                {
                    "backtest_config": {"symbol": "X"},
                    "validated_strategies": {"aaa": gene_a},
                    "validation_results": {"aaa": {"passed": True}},
                }
            ),
            encoding="utf-8",
        )
        # f2: 同じ遺伝子(gene_a)と別遺伝子(gene_b)を含む
        f2.write_text(
            _json.dumps(
                {
                    "backtest_config": {"symbol": "X"},
                    "validated_strategies": {"bbb": gene_a, "ccc": gene_b},
                    "validation_results": {
                        "bbb": {"passed": True},
                        "ccc": {"passed": True},
                    },
                }
            ),
            encoding="utf-8",
        )
        merged = merge_result_files([f1, f2])
        assert merged is not None
        # 重複遺伝子は排除され、ユニーク2件のみ
        assert len(merged["validation_results"]) == 2
        gids = set(merged["validation_results"].keys())
        # gidは接頭辞付きで一意化され、生遺伝子との対応が保たれる
        for gid in gids:
            assert gid in merged["validated_strategies"]

    def test_find_all_results_for_period_returns_all_seeds(self, tmp_path, monkeypatch):
        import scripts.validate_pipeline as vp

        base = tmp_path / "results" / "auto_strategy"
        base.mkdir(parents=True)
        cfg = {"start_date": "2022-01-01", "end_date": "2022-06-30", "symbol": "ETH"}
        for name in ("strategy_a.json", "strategy_b.json"):
            (base / name).write_text(
                json.dumps({"backtest_config": dict(cfg)}), encoding="utf-8"
            )
        (base / "other.json").write_text(
            json.dumps({"backtest_config": {**cfg, "symbol": "BTC"}}),
            encoding="utf-8",
        )
        monkeypatch.setattr(vp, "PROJECT_ROOT", tmp_path)
        paths = vp.find_all_results_for_period("2022-01-01", "2022-06-30", "ETH")
        assert len(paths) == 2
