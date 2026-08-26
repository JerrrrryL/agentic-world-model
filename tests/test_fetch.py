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
        ("viewer_data/index.json", 1_088_938),
        ("viewer_data/claude__gsm8k.json", 5_000_000),
        # Upstream keeps viewer_data flat, so the two rows above are rejected by
        # the depth check alone. This one is the shape the named guard exists
        # for: if upstream ever nests it per config, nothing else stops 5.3 GB.
        ("viewer_data/gsm8k_Qwen_Qwen3-4B-Base_3/solve_out.txt", 5_000_000),
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
        # An empty `configs` switches the config filter off, and viewer_data
        # occupies the slot a config name would. The named guard is what stops
        # it; deleting it must fail this test, hence the nested row in the
        # fixture — the flat ones alone are caught by the depth check.
        got = {p for p, _ in fetch.ptb_select(listing, configs=fetch.ALL_CONFIGS)}
        assert not any(p.startswith("viewer_data") for p in got)

    def test_the_catalog_is_not_selected_either(self, listing):
        # fetch_posttrainbench adds PTB_CATALOG on its own. ptb_select stays the
        # thing that keeps viewer_data out, whatever it is handed.
        for cfg in (fetch.PTB_DEFAULT_CONFIGS, fetch.ALL_CONFIGS, ("viewer_data",)):
            got = {p for p, _ in fetch.ptb_select(listing, configs=cfg, files=("index.json",))}
            assert fetch.PTB_CATALOG not in got

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


@pytest.fixture
def fake_hub(tmp_path, listing, monkeypatch) -> list[str]:
    """Record what fetch_posttrainbench asks the hub for, and write stubs for it."""
    import huggingface_hub

    asked: list[str] = []

    def download(repo_id, repo_type, filename, local_dir):
        asked.append(filename)
        out = tmp_path / filename
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text("{}")
        return str(out)

    monkeypatch.setattr(fetch, "ptb_list_files", lambda *a, **kw: listing)
    monkeypatch.setattr(huggingface_hub, "hf_hub_download", download)
    return asked


class TestCatalog:
    def test_every_batch_takes_the_catalog(self, tmp_path, fake_hub):
        fetch.fetch_posttrainbench(dest=tmp_path, workers=2)
        assert fetch.PTB_CATALOG in fake_hub
        # and nothing else out of viewer_data's 5.3 GB came with it
        assert [p for p in fake_hub if p.startswith("viewer_data")] == [fetch.PTB_CATALOG]

    def test_it_can_be_turned_off_for_a_traces_only_mirror(self, tmp_path, fake_hub):
        fetch.fetch_posttrainbench(dest=tmp_path, workers=2, catalog=False)
        assert not any(p.startswith("viewer_data") for p in fake_hub)

    def test_a_second_run_does_not_re_download_it(self, tmp_path, fake_hub):
        fetch.fetch_posttrainbench(dest=tmp_path, workers=2)
        fake_hub.clear()
        fetch.fetch_posttrainbench(dest=tmp_path, workers=2)
        assert fake_hub == []

    def test_reading_it_never_triggers_a_download(self, tmp_path):
        with pytest.raises(FileNotFoundError):
            fetch.ptb_catalog(tmp_path)

    def test_round_trips(self, tmp_path):
        path = tmp_path / fetch.PTB_CATALOG
        path.parent.mkdir(parents=True)
        path.write_text(json.dumps({"runs": [{"run_id": "x", "accuracy": 0.5}]}))
        assert fetch.ptb_catalog(tmp_path)["runs"][0]["run_id"] == "x"


class TestDefaults:
    def test_core_and_observe_benchmarks_are_disjoint(self):
        assert not set(fetch.PTB_CORE_BENCHMARKS) & set(fetch.PTB_OBSERVE_BENCHMARKS)

    def test_default_batch_covers_both_cli_families(self):
        fams = {c.split("_")[0] for c in fetch.PTB_DEFAULT_CONFIGS}
        assert fams == {"claude", "codex"}
