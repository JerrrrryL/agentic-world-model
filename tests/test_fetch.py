"""Tests for upstream fetching — the selection logic, which decides what we pay to download."""

from __future__ import annotations

import json

import pytest

from hv.traj import fetch


@pytest.fixture
def listing() -> list[tuple[str, int]]:
    """A miniature of the real dataset tree, including the shapes that must be rejected."""
    return [
        (".gitattributes", 36113),
        ("README.md", 2668),
        ("viewer_data/claude__gsm8k.json", 5_000_000),
        # wanted: a core benchmark under a selected config
        ("claude_non_api_max_claude-opus-4-8_10h_run1/gsm8k_Qwen_Qwen3-1.7B-Base_17315721/solve_out.txt", 900_000),
        ("claude_non_api_max_claude-opus-4-8_10h_run1/gsm8k_Qwen_Qwen3-1.7B-Base_17315721/metrics.json", 80),
        # rejected: file we do not want (workspace snapshot, huge log)
        ("claude_non_api_max_claude-opus-4-8_10h_run1/gsm8k_Qwen_Qwen3-1.7B-Base_17315721/error.log", 162_000_000),
        ("claude_non_api_max_claude-opus-4-8_10h_run1/gsm8k_Qwen_Qwen3-1.7B-Base_17315721/task/train.py", 4000),
        # rejected: observation-group benchmark
        ("claude_non_api_max_claude-opus-4-8_10h_run1/healthbench_Qwen_Qwen3-4B-Base_1/solve_out.txt", 500),
        # rejected: config not in the batch
        ("opencode_zai_glm-5_10h_run2/gsm8k_Qwen_Qwen3-4B-Base_2/solve_out.txt", 500),
        # a second selected config
        ("codex_non_api_high_gpt-5.4_10h_run1/bfcl_google_gemma-3-4b-pt_16934887/solve_out.txt", 700_000),
    ]


class TestSelect:
    def test_picks_only_core_benchmarks_of_selected_configs(self, listing):
        got = {p for p, _ in fetch.ptb_select(listing)}
        assert got == {
            "claude_non_api_max_claude-opus-4-8_10h_run1/gsm8k_Qwen_Qwen3-1.7B-Base_17315721/solve_out.txt",
            "claude_non_api_max_claude-opus-4-8_10h_run1/gsm8k_Qwen_Qwen3-1.7B-Base_17315721/metrics.json",
            "codex_non_api_high_gpt-5.4_10h_run1/bfcl_google_gemma-3-4b-pt_16934887/solve_out.txt",
        }

    def test_excludes_the_files_that_make_the_dataset_29gb(self, listing):
        got = {p for p, _ in fetch.ptb_select(listing)}
        assert not any("error.log" in p or "/task/" in p or p.startswith("viewer_data") for p in got)

    def test_repo_root_files_are_not_three_deep_and_so_never_match(self, listing):
        assert not any(p in ("README.md", ".gitattributes") for p, _ in fetch.ptb_select(listing))

    def test_observation_group_is_available_on_request(self, listing):
        got = {p for p, _ in fetch.ptb_select(listing, benchmarks=fetch.PTB_OBSERVE_BENCHMARKS)}
        assert got == {
            "claude_non_api_max_claude-opus-4-8_10h_run1/healthbench_Qwen_Qwen3-4B-Base_1/solve_out.txt"
        }

    def test_a_benchmark_name_must_match_the_whole_prefix_segment(self, listing):
        # "aime2025" must not be selected by a request for "aime", nor vice versa.
        rows = [("cfg/aime2026_Qwen_Qwen3-4B-Base_9/metrics.json", 10)]
        assert fetch.ptb_select(rows, configs=("cfg",), benchmarks=("aime2025",)) == []
        assert fetch.ptb_select(rows, configs=("cfg",), benchmarks=("aime2026",)) == rows

    def test_empty_configs_means_every_configuration(self, listing):
        # ALL_CONFIGS is the empty tuple: the config filter is the one that widens
        # to the whole release, because the file filter alone already keeps the
        # download to traces.
        got = {p for p, _ in fetch.ptb_select(listing, configs=fetch.ALL_CONFIGS)}
        assert "opencode_zai_glm-5_10h_run2/gsm8k_Qwen_Qwen3-4B-Base_2/solve_out.txt" in got
        assert not any("error.log" in p or "/task/" in p for p in got)

    def test_viewer_data_is_never_selected_even_with_every_config(self, listing):
        # It sits at the same depth as a run directory and would otherwise match.
        got = {p for p, _ in fetch.ptb_select(listing, configs=fetch.ALL_CONFIGS)}
        assert not any(p.startswith("viewer_data") for p in got)

    def test_an_empty_file_filter_selects_nothing(self, listing):
        # Widening `files` is what would pull the whole 28.9 GB release, so an
        # empty filter must mean nothing, never everything.
        assert fetch.ptb_select(listing, files=()) == []


class TestListingCache:
    def test_uses_the_cache_without_touching_the_network(self, tmp_path, monkeypatch):
        cache = tmp_path / ".file_list.json"
        cache.write_text(json.dumps([["a/b/c.txt", 12]]))

        def explode(*a, **kw):  # any HTTP call here is a bug
            raise AssertionError("hit the network despite a warm cache")

        monkeypatch.setattr("requests.get", explode)
        assert fetch.ptb_list_files(cache) == [("a/b/c.txt", 12)]


class TestDefaults:
    def test_core_and_observe_benchmarks_are_disjoint(self):
        assert not set(fetch.PTB_CORE_BENCHMARKS) & set(fetch.PTB_OBSERVE_BENCHMARKS)

    def test_default_batch_covers_both_cli_families(self):
        fams = {c.split("_")[0] for c in fetch.PTB_DEFAULT_CONFIGS}
        assert fams == {"claude", "codex"}
