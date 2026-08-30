"""Give the free-text fields of an extracted recipe a groupable key.

The extractor is asked for a *faithful* description of what an agent did, so it
writes what the trajectory says: 344 algorithm steps across the 143 gsm8k train
runs carry 326 distinct ``name`` strings, and 391 dataset entries carry 263.
Every one of those strings is right. None of them can be counted.

So each free-text field gets a companion key, and the original is kept beside
it. ``algo_family`` maps a step onto one of a closed vocabulary; ``dataset_id``
maps a dataset onto its Hugging Face id where there is one. Both are pure
functions of the string — no lookup table of run ids, nothing that has to be
re-derived when the extraction is re-run.

Two decisions worth knowing about:

*   **A step gets one family, by priority, not a set of tags.** An "SFT on the
    LoRA-merged checkpoint, then copied to final_model" matches ``sft``,
    ``merge`` and ``package``; calling it all three makes the family counts sum
    to more than the steps. The priority order below is "most specific claim
    about what the step *did*" — a preference/RL objective beats rejection
    sampling beats plain SFT beats the weight surgery afterwards.
*   **A path is not a dataset id.** ``data/rft.jsonl`` and ``meta-math/MetaMathQA``
    both match ``org/name``. The first is a file the agent wrote itself and
    belongs under ``local:``, and treating it as a public dataset would put
    self-generated data in the public column — which is the one number this
    corpus is most likely to be asked for.

``PATTERNS`` and ``ALIASES`` are the parts that rot. When the extraction is
re-run, check the residual: ``algo_family`` should leave almost nothing in
``other``, and a growing ``other`` means the vocabulary stopped describing the
corpus rather than that the corpus stopped having algorithms.
"""

from __future__ import annotations

import re

#: Ordered: the first pattern that matches wins. See the module docstring on why
#: a step gets one family rather than every tag it matches.
PATTERNS: list[tuple[str, str]] = [
    # No trailing \b: the only signal for some steps is a trainer class name in
    # ``framework``, and ``\bgrpo\b`` does not match ``trl.GRPOTrainer``.
    ("dpo", r"\bdpo"),
    ("orpo", r"\borpo"),
    ("kto", r"\bkto"),
    ("simpo", r"\bsimpo"),
    ("ppo", r"\bppo"),
    ("grpo", r"\bgrpo"),
    ("rloo", r"\brloo"),
    ("reinforce", r"\breinforce"),
    (
        "rft",
        r"rejection[ _-]?sampl|\brft\b|\bstar\b|self[- ]?train|expert[- ]iter"
        r"|answer-verification|hard-prompt min",
    ),
    ("distill", r"distill"),
    (
        "sft",
        r"\bsft|fine[- ]?tun|finetun|supervised|continuation|refinement|re-anchor"
        r"|continued|calibration pass|second epoch|epoch over|training launch|train(_v\d+)?\.py",
    ),
    ("merge", r"merge|soup|averag|interpolat"),
    ("quantise", r"quantiz|quantis|awq|gptq|bnb|4-?bit|8-?bit"),
    ("decode-config", r"\beos\b|generation_config|stop-vs-continue|stop-calibration|greedy generation"),
    (
        "package",
        r"packag|export|promote|copy|copied|finali[sz]|submission director|final_model"
        r"|checkpoint select|select.*checkpoint|pick.*checkpoint|ship",
    ),
    ("none", r"^none\b|no post-training"),
]

#: Families that change weights. The rest — merge, quantise, decode-config,
#: package — are what the agent did to the weights afterwards, and counting them
#: as training stages inflates every pipeline by one or two steps.
TRAINING_FAMILIES = frozenset(
    {"dpo", "orpo", "kto", "simpo", "ppo", "grpo", "rloo", "reinforce", "rft", "distill", "sft"}
)

#: Spellings of the same dataset. The key is what falls out of :func:`_raw_id`
#: or a bare-name match; the value is the id to count under.
ALIASES: dict[str, str] = {
    "gsm8k": "openai/gsm8k",
    "openai/gsm8k": "openai/gsm8k",
    "metamathqa": "meta-math/metamathqa",
    "metamath": "meta-math/metamathqa",
    "meta-math/metamathqa": "meta-math/metamathqa",
    "openmathinstruct-2": "nvidia/openmathinstruct-2",
    "openmathinstruct2": "nvidia/openmathinstruct-2",
    "omi2": "nvidia/openmathinstruct-2",
    "nvidia/openmathinstruct-2": "nvidia/openmathinstruct-2",
    "orca-math-word-problems-200k": "microsoft/orca-math-word-problems-200k",
    "microsoft/orca-math-word-problems-200k": "microsoft/orca-math-word-problems-200k",
    "numinamath-cot": "ai-mo/numinamath-cot",
    "ai-mo/numinamath-cot": "ai-mo/numinamath-cot",
}

#: First path segments that mean "a file this agent wrote", not an org on the Hub.
_LOCAL_ROOTS = frozenset(
    {"data", "data_clean", "runs", "run", "work", "out", "output", "outputs", "sft_data",
     "checkpoints", "ckpt", "logs", "tmp", "rft", "epochs", "scripts", "artifacts"}
)
_FILE_SUFFIX = re.compile(
    r"\.(jsonl|json|parquet|csv|txt|arrow|pt|bin|safetensors|py|sh|yaml|yml|md|log)\b", re.I
)
_ORG_NAME = re.compile(r"\b([A-Za-z0-9][\w.-]*/[\w.-]+)")
#: A Hub id is not two English words with a slash between them. Every real id in
#: this corpus carries a digit, hyphen, underscore or dot somewhere; prose like
#: "percentages/ratios/averages" carries none, and without this it counts as a
#: dataset that was never pulled.
_ID_SHAPED = re.compile(r"[-_0-9.]")

#: Names that describe data the agent produced rather than downloaded. Checked
#: before the bare-name aliases so "self-generated GSM8K solutions" does not
#: count as a pull of ``openai/gsm8k``.
_SELF = re.compile(
    r"(?i)self[- ]?(sampled|generated|produced)|rejection[- ]?sampled|\bstar\b|on-policy"
    r"|model'?s own|sampled from the|generated by the (trained|current|sft|base)"
)
_TEACHER = re.compile(r"(?i)teacher|distilled from|generated by (gpt|claude|qwen2\.5|deepseek|a stronger)")


def algo_family(name: str | None, framework: str | None = None, peft: str | None = None) -> str:
    """Map one pipeline step onto a closed vocabulary. ``other`` when nothing fits."""
    blob = " | ".join(x for x in (name, framework, peft) if x).lower()
    if not blob:
        return "other"
    for family, pattern in PATTERNS:
        if re.search(pattern, blob):
            return family
    return "other"


def _raw_id(name: str) -> str | None:
    """The ``org/name`` in a dataset string, or ``None`` if it is a local path."""
    match = _ORG_NAME.search(name)
    if not match:
        return None
    candidate = match.group(1)
    if _FILE_SUFFIX.search(candidate) or candidate.count("/") > 1:
        return None
    if not _ID_SHAPED.search(candidate):
        return None
    if candidate.split("/", 1)[0].lower() in _LOCAL_ROOTS:
        return None
    return candidate.lower()


def dataset_id(name: str | None, kind: str | None = None) -> str:
    """Canonicalise a dataset name.

    Returns a Hugging Face id, or one of the ``synthetic:``/``local:``/``unknown``
    namespaces. ``kind`` only breaks ties the string itself leaves open — it is
    the extractor's judgement, and the string is the evidence.
    """
    text = (name or "").strip()
    if not text:
        return "unknown"

    if _SELF.search(text) or kind == "synthetic-self":
        return "synthetic:self"
    if _TEACHER.search(text) or kind == "synthetic-other-model":
        return "synthetic:teacher"

    raw = _raw_id(text)
    if raw:
        return ALIASES.get(raw, raw)

    bare = re.sub(r"[^a-z0-9/_.-]+", " ", text.lower())
    for token in bare.split():
        if token in ALIASES:
            return ALIASES[token]
    for alias, canonical in ALIASES.items():
        if "/" not in alias and re.search(rf"\b{re.escape(alias)}\b", bare):
            return canonical

    if kind == "handwritten":
        return "handwritten"
    path = _ORG_NAME.search(text)
    if path:
        return "local:" + path.group(1).rsplit("/", 1)[-1].lower()
    return "unknown"


def annotate(recipe: dict) -> dict:
    """Add ``family`` to every algorithm and ``dataset_id`` to every dataset.

    Returns a new dict; the verbatim strings the extractor wrote are untouched.
    """
    out = dict(recipe)
    out["algorithms"] = [
        {**a, "family": algo_family(a.get("name"), a.get("framework"), a.get("peft"))}
        for a in recipe.get("algorithms") or []
    ]
    out["datasets"] = [
        {**d, "dataset_id": dataset_id(d.get("name"), d.get("kind"))}
        for d in recipe.get("datasets") or []
    ]
    out["pipeline"] = [
        a["family"] for a in out["algorithms"] if a["family"] in TRAINING_FAMILIES
    ]
    return out
