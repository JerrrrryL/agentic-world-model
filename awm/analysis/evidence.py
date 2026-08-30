"""Check an extracted recipe's evidence anchors against the digest they cite.

Every ``algorithms[]`` and ``datasets[]`` entry in the recipe file carries an
``evidence_i`` (an event index) and an ``evidence_quote`` (text that is supposed
to appear at that event). The reviewer agents were told to ``grep`` the digest
before accepting a quote — but an agent *saying* it grepped is not a grep, and
this is the one part of the audit that needs no model at all: the string is
either in that block or it is not.

Four verdicts, because they are four different defects:

``ok``
    verbatim inside the cited block, up to the normalisation below.
``elided``
    the quote joins two real spans of the cited block with ``...``, in order.
    The claim is evidenced; the quote just is not a single span. Cosmetic.
``wrong-block``
    the text is in the digest, but not where the record says. The claim still
    has evidence behind it; the anchor is broken, so nobody can re-check it
    cheaply.
``absent``
    nowhere in the digest. This is the extractor writing prose and calling it a
    quote — the failure the whole audit exists to catch.

Both sides are normalised before comparing, in exactly two ways, and each was
added because a strict comparison reported a difference that was not one:

*   **Whitespace is collapsed.** The digest re-wraps long commands, so a quote
    of one will not match byte-for-byte even when it is honest. Also strips the
    ``…[truncated]`` marker the digest itself appends to a capped block.
*   **A short closed set of characters is folded to ASCII** (:data:`FOLD`).
    Five of the 735 anchors in the gsm8k corpus missed only because the
    extractor retyped ``×`` as ``x`` or ``≤`` as ``<=``. That is a transcription
    slip, not an invention, and reporting it beside a genuinely fabricated quote
    would make the count useless.

What this module deliberately does *not* do is fuzzy-match. There is no edit
distance, no token overlap, no "close enough" prefix — the two normalisations
are character-level, applied to both sides, and cannot turn a paraphrase into a
match. That strictness is the whole value: if a future extraction reports zero
``absent``, that should mean the extractor quoted honestly, not that the checker
got accommodating. :data:`MIN_ANCHOR` and :data:`FOLD` are the parts that could
rot; grow ``FOLD`` only for characters that render near-identically, never for
words.
"""

from __future__ import annotations

import re
from collections import Counter

#: A quote shorter than this anchors nothing — ``"SFT"`` appears in every run.
MIN_ANCHOR = 8

#: Every verdict :func:`check` can return, worst-is-last within severity. Exists
#: so a caller can zero-fill :func:`audit`'s tally: a ``Counter`` reports an
#: absent key as ``0``, which is the right reading here and the opposite of the
#: convention everywhere else in the recipe file, where an absent key is NA. A
#: row with no anchors at all then has ``{}`` where every other row has counts,
#: and a columnar reader loads the difference as null rather than zero.
VERDICTS = ("ok", "elided", "wrong-block", "absent", "too-short", "no-anchor")

#: Characters an extractor predictably retypes. Applied to both sides, so it can
#: only merge spellings that look the same, never bridge different words.
FOLD = {
    "×": "x",      # × multiplication sign
    "≤": "<=",     # ≤
    "≥": ">=",     # ≥
    "→": "->",     # →
    "‘": "'", "’": "'",          # curly single quotes
    "“": '"', "”": '"',          # curly double quotes
    "–": "-", "—": "-",          # en dash, em dash
    " ": " ",      # non-breaking space
    "−": "-",      # − minus sign
}
_FOLD_RE = re.compile("|".join(re.escape(k) for k in FOLD))

_HEADER = re.compile(r"^--- \[(\d+)\] turn=\S+ (\S+) ---$", re.M)
_TRUNCATION = ("…[truncated]", "...[truncated]")
_ELISION = re.compile(r"\s*\.\.\.\s*|\s*…\s*")


def parse_digest(text: str) -> dict[int, str]:
    """Split a rendered digest into ``{event index: block body}``."""
    hits = list(_HEADER.finditer(text))
    out: dict[int, str] = {}
    for n, match in enumerate(hits):
        end = hits[n + 1].start() if n + 1 < len(hits) else len(text)
        out[int(match.group(1))] = text[match.end() : end]
    return out


def normalise(text: str) -> str:
    """Collapse whitespace and fold the characters in :data:`FOLD`."""
    return _FOLD_RE.sub(lambda m: FOLD[m.group()], re.sub(r"\s+", " ", text)).strip()


def _strip_marker(quote: str) -> str:
    for marker in _TRUNCATION:
        if quote.endswith(marker):
            return quote[: -len(marker)].strip()
    return quote


def _in_order(parts: list[str], haystack: str) -> bool:
    """Every fragment present, and in the order the quote claims."""
    at = 0
    for part in parts:
        at = haystack.find(part, at)
        if at < 0:
            return False
        at += len(part)
    return True


def check(quote: str | None, evidence_i: int | None, blocks: dict[int, str],
          digest: str | None = None) -> str:
    """Classify one anchor.

    ``digest`` is the whole digest text, needed to tell ``wrong-block`` from
    ``absent``; omit it and a miss outside the cited block reads as ``absent``.
    """
    if not quote or evidence_i is None:
        return "no-anchor"

    want = normalise(_strip_marker(normalise(quote)))
    if len(want) < MIN_ANCHOR:
        return "too-short"

    here = normalise(blocks.get(int(evidence_i), ""))
    whole = normalise(digest) if digest is not None else here

    if want in here:
        return "ok"

    parts = [p for p in _ELISION.split(want) if len(p) >= MIN_ANCHOR]
    if len(parts) > 1:
        if _in_order(parts, here):
            return "elided"
        if _in_order(parts, whole):
            return "wrong-block"

    return "wrong-block" if want in whole else "absent"


def audit(recipe: dict, digest: str) -> Counter:
    """Tally the verdicts over one recipe's algorithm and dataset anchors."""
    blocks = parse_digest(digest)
    tally: Counter[str] = Counter()
    for field in ("algorithms", "datasets"):
        for item in recipe.get(field) or []:
            tally[check(item.get("evidence_quote"), item.get("evidence_i"), blocks, digest)] += 1
    return tally
