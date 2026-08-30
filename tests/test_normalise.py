"""The normalisation keys are what the recipe file gets counted by, so a wrong
key is a wrong statistic that nothing downstream can detect: a mislabelled step
just moves one run from one column to another and every total still adds up.
"""

from __future__ import annotations

import pytest

from awm.analysis import normalise as nz


class TestAlgoFamily:
    def test_a_step_gets_one_family_and_the_objective_wins(self) -> None:
        """"GRPO on top of the SFT checkpoint" is a GRPO step, not an SFT one."""
        assert nz.algo_family("GRPO round 2 from the SFT v2 checkpoint") == "grpo"

    def test_rejection_sampling_outranks_the_sft_that_consumes_it(self) -> None:
        assert nz.algo_family("rejection-sampling SFT (RFT)", "trl.SFTTrainer") == "rft"

    @pytest.mark.parametrize(
        "name,expected",
        [
            ("SFT", "sft"),
            ("full-parameter fine-tune of Qwen3-4B-Base", "sft"),
            ("Stage 2 — low-LR continuation", "sft"),
            ("LoRA merge into base weights", "merge"),
            ("Promote runs/full_run1 to final_model", "package"),
            ("on-policy DPO", "dpo"),
            ("EOS repair of the kept checkpoint", "decode-config"),
            ("none — no post-training survived", "none"),
        ],
    )
    def test_the_vocabulary_covers_what_the_corpus_says(self, name: str, expected: str) -> None:
        assert nz.algo_family(name) == expected

    def test_an_unmatchable_step_is_other_rather_than_the_nearest_guess(self) -> None:
        """A silent fallback to ``sft`` would make the residual invisible."""
        assert nz.algo_family("Pipeline as summarised by the agent at completion") == "other"
        assert nz.algo_family("") == "other"
        assert nz.algo_family(None) == "other"

    def test_a_trainer_class_in_the_framework_field_names_the_objective(self) -> None:
        """Several steps are named "stage 3" and say what they are only in
        ``framework``. ``\\bgrpo\\b`` does not match ``GRPOTrainer``."""
        assert nz.algo_family("stage 3", framework="trl.GRPOTrainer") == "grpo"
        assert nz.algo_family("stage 3", framework="trl.SFTTrainer") == "sft"
        assert nz.algo_family("stage 3", framework="trl.DPOTrainer") == "dpo"

    def test_lora_alone_does_not_name_an_objective(self) -> None:
        """``peft`` says how the weights were parameterised, not what was optimised.
        A LoRA step with no stated objective is ``other``, not a guessed ``sft``."""
        assert nz.algo_family("stage 3", peft="lora r=64 alpha=128") == "other"


class TestDatasetId:
    @pytest.mark.parametrize(
        "name",
        [
            "openai/gsm8k",
            "openai/gsm8k (main, train)",
            "GSM8K train split",
            "openai/gsm8k (few-shot system-prompt pool for SFT)",
        ],
    )
    def test_four_spellings_of_gsm8k_land_on_one_id(self, name: str) -> None:
        assert nz.dataset_id(name) == "openai/gsm8k"

    def test_metamath_folds_across_its_subset_notations(self) -> None:
        for name in ("meta-math/MetaMathQA", "meta-math/MetaMathQA (GSM_* subset)",
                     "MetaMathQA GSM_Rephrased"):
            assert nz.dataset_id(name) == "meta-math/metamathqa"

    def test_a_file_the_agent_wrote_is_not_a_hub_dataset(self) -> None:
        """``data/rft.jsonl`` matches ``org/name``; counting it as public would put
        self-generated data in the public column."""
        assert nz.dataset_id("data/rft.jsonl") == "local:rft.jsonl"
        assert nz.dataset_id("work/rft_all.jsonl") == "local:rft_all.jsonl"

    def test_prose_with_a_slash_in_it_is_not_an_id(self) -> None:
        got = nz.dataset_id("synthetic problems, GSM-style percentages/ratios/averages (8k)",
                            kind="handwritten")
        assert got == "handwritten"

    def test_a_script_path_is_not_an_id_either(self) -> None:
        got = nz.dataset_id("GSM8K train split, rendered in the exact Inspect/evaluate.py format")
        assert got == "openai/gsm8k"

    def test_self_generated_data_is_not_the_dataset_it_was_sampled_from(self) -> None:
        """These strings all name GSM8K; none of them is a pull of GSM8K."""
        for name in ("self-sampled correct GSM8K solutions from the SFT-v2 checkpoint",
                     "rejection-sampled CoT on GSM8K train questions",
                     "STaR self-generated verified solutions from the round-1 model"):
            assert nz.dataset_id(name) == "synthetic:self"

    def test_a_stronger_model_is_a_different_namespace_from_the_model_itself(self) -> None:
        assert nz.dataset_id("teacher explanations from Qwen2.5-Math-7B") == "synthetic:teacher"
        assert nz.dataset_id("some rows", kind="synthetic-other-model") == "synthetic:teacher"

    def test_kind_only_breaks_ties_the_string_leaves_open(self) -> None:
        """The string is the evidence; ``kind`` is the extractor's judgement about it."""
        assert nz.dataset_id("openai/gsm8k", kind="public") == "openai/gsm8k"
        assert nz.dataset_id("rejection-sampled from gsm8k", kind="public") == "synthetic:self"

    def test_an_undescribable_name_is_unknown_not_a_bucket(self) -> None:
        assert nz.dataset_id("replay slice of the stage-1 SFT mixture") == "unknown"
        assert nz.dataset_id("") == "unknown"
        assert nz.dataset_id(None) == "unknown"


class TestAnnotate:
    def _recipe(self) -> dict:
        return {
            "run": "e/r",
            "algorithms": [
                {"order": 1, "name": "SFT", "framework": "trl.SFTTrainer"},
                {"order": 2, "name": "GRPO", "framework": "trl.GRPOTrainer"},
                {"order": 3, "name": "LoRA merge into base weights"},
                {"order": 4, "name": "copied to final_model"},
            ],
            "datasets": [{"name": "openai/gsm8k", "kind": "public"}],
        }

    def test_the_pipeline_is_the_training_steps_only(self) -> None:
        """Merge and packaging are weight surgery, not stages; counting them
        would make every pipeline one or two steps longer than it was."""
        assert nz.annotate(self._recipe())["pipeline"] == ["sft", "grpo"]

    def test_the_verbatim_strings_survive_beside_the_keys(self) -> None:
        out = nz.annotate(self._recipe())
        assert out["algorithms"][1]["name"] == "GRPO"
        assert out["algorithms"][1]["family"] == "grpo"
        assert out["datasets"][0]["dataset_id"] == "openai/gsm8k"

    def test_it_does_not_mutate_its_input(self) -> None:
        original = self._recipe()
        nz.annotate(original)
        assert "family" not in original["algorithms"][0]

    def test_a_recipe_with_no_algorithms_annotates_to_an_empty_pipeline(self) -> None:
        out = nz.annotate({"run": "e/r"})
        assert out["pipeline"] == [] and out["algorithms"] == [] and out["datasets"] == []
