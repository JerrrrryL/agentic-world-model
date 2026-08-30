"""The evidence checker is the only unfaked number in the recipe file's audit:
every other quality figure comes from a model saying it looked. So the way this
module fails is by getting lenient — a paraphrase that passes turns "99.7% of
anchors verify" from a measurement into a restatement of the extractor's own
confidence.
"""

from __future__ import annotations

from awm.analysis import evidence

DIGEST = """# run: e/r
# recipe-bearing events: 3

--- [23] turn=9 say ---
I'll upsample MetaMathQA to 70% of the mixture and keep GSM8K train at 30%.
--- [31] turn=12 shell ---
command: python3 train.py --lr 2e-5 --num_train_epochs 3
description: Launch the SFT run
--- [88] turn=40 say ---
RFT (self-sample K=4×2 temps, keep correct, dedup, cap 3/q): added ~21K rows.
I keep the ≤4 shortest solutions per problem.
--- [90] turn=41 write ---
content: trainer = SFTTrainer(
    model=base,
    args=cfg,
)…[truncated]
"""

BLOCKS = evidence.parse_digest(DIGEST)


class TestParseDigest:
    def test_it_finds_every_block_and_keys_them_by_event_index(self) -> None:
        assert sorted(BLOCKS) == [23, 31, 88, 90]

    def test_a_block_body_stops_at_the_next_header(self) -> None:
        assert "MetaMathQA" in BLOCKS[23] and "train.py" not in BLOCKS[23]

    def test_the_last_block_runs_to_the_end_of_the_file(self) -> None:
        assert "SFTTrainer(" in BLOCKS[90]

    def test_the_preamble_before_the_first_header_is_not_a_block(self) -> None:
        """``# run:`` lines are metadata the extractor was handed, not evidence."""
        assert all("# run:" not in body for body in BLOCKS.values())


class TestCheck:
    def _check(self, quote, i):
        return evidence.check(quote, i, BLOCKS, DIGEST)

    def test_a_verbatim_quote_in_the_right_block_is_ok(self) -> None:
        assert self._check("upsample MetaMathQA to 70% of the mixture", 23) == "ok"

    def test_a_quote_the_digest_wrapped_differently_still_matches(self) -> None:
        """The digest re-emits commands with its own indentation; comparing bytes
        would report a difference that is not one."""
        assert self._check("trainer = SFTTrainer(\n  model=base,\n  args=cfg,\n)", 90) == "ok"

    def test_the_truncation_marker_the_digest_added_is_not_the_extractor_s_fault(self) -> None:
        assert self._check("trainer = SFTTrainer(…[truncated]", 90) == "ok"

    def test_a_real_quote_filed_under_the_wrong_event_is_not_absent(self) -> None:
        """A broken anchor and an invented quote are different defects: the first
        still has evidence behind it, it just cannot be re-checked cheaply."""
        assert self._check("upsample MetaMathQA to 70% of the mixture", 31) == "wrong-block"

    def test_text_that_is_nowhere_in_the_digest_is_absent(self) -> None:
        assert self._check("we then ran DPO on the preference pairs", 23) == "absent"

    def test_a_paraphrase_of_the_right_block_is_still_absent(self) -> None:
        """This is the case the whole module exists for. The block says the words
        in a different order; a checker loose enough to pass this measures nothing."""
        assert self._check("kept GSM8K train at 30% and upsampled MetaMathQA to 70%", 23) == "absent"

    def test_a_unicode_character_retyped_as_ascii_is_not_an_invention(self) -> None:
        """Five of the corpus's 735 anchors missed only on ``×`` written ``x`` or
        ``≤`` written ``<=``."""
        assert self._check("RFT (self-sample K=4x2 temps, keep correct, dedup, cap 3/q)", 88) == "ok"
        assert self._check("keep the <=4 shortest solutions per problem", 88) == "ok"

    def test_the_fold_cannot_bridge_two_different_words(self) -> None:
        """It is character-level and applied to both sides; nothing about it
        makes ``dedup`` match ``deduplicate``."""
        assert self._check("RFT (self-sample K=4x2 temps, keep correct, deduplicate", 88) == "absent"

    def test_an_elision_joining_two_real_spans_of_the_block_is_its_own_verdict(self) -> None:
        assert self._check("I'll upsample MetaMathQA ... GSM8K train at 30%", 23) == "elided"

    def test_an_elision_whose_fragments_are_out_of_order_does_not_pass(self) -> None:
        """"A ... B" claims B follows A. Accepting it reversed would let a quote
        assert an ordering the digest contradicts."""
        assert self._check("GSM8K train at 30% ... I'll upsample MetaMathQA", 23) == "absent"

    def test_an_elision_across_two_blocks_is_a_broken_anchor_not_a_clean_one(self) -> None:
        got = self._check("upsample MetaMathQA to 70% ... python3 train.py --lr 2e-5", 23)
        assert got == "wrong-block"

    def test_a_quote_too_short_to_anchor_anything_is_flagged_rather_than_passed(self) -> None:
        """``SFT`` is in every run in the corpus; matching it proves nothing."""
        assert self._check("SFT", 90) == "too-short"

    def test_a_missing_quote_or_index_is_reported_not_skipped(self) -> None:
        assert self._check(None, 23) == "no-anchor"
        assert self._check("upsample MetaMathQA to 70% of the mixture", None) == "no-anchor"

    def test_an_index_the_digest_does_not_contain_falls_through_to_the_whole_text(self) -> None:
        """The extractor can cite an event the digest filter dropped. That is a
        broken anchor if the text is elsewhere, absent if it is nowhere."""
        assert self._check("upsample MetaMathQA to 70% of the mixture", 9999) == "wrong-block"
        assert self._check("we then ran DPO on the pairs", 9999) == "absent"

    def test_without_the_whole_digest_a_miss_cannot_be_called_wrong_block(self) -> None:
        assert evidence.check("python3 train.py --lr 2e-5", 23, BLOCKS) == "absent"


class TestAudit:
    def test_it_tallies_both_fields_and_names_each_verdict(self) -> None:
        recipe = {
            "algorithms": [
                {"name": "SFT", "evidence_i": 31, "evidence_quote": "python3 train.py --lr 2e-5"},
                {"name": "RFT", "evidence_i": 23, "evidence_quote": "invented out of thin air"},
            ],
            "datasets": [
                {"name": "metamath", "evidence_i": 23, "evidence_quote": "upsample MetaMathQA to 70%"},
            ],
        }
        assert evidence.audit(recipe, DIGEST) == {"ok": 2, "absent": 1}

    def test_a_recipe_with_no_anchors_tallies_to_nothing_rather_than_raising(self) -> None:
        assert evidence.audit({"run": "e/r"}, DIGEST) == {}
