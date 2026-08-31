"""Score a jsonl of ``{id, question, gold}`` items with a checkpoint (custom evaluators).

Generation uses vLLM when importable, else transformers. Scoring is GSM8K-style:
the last number in the completion is compared numerically with the gold
(``#### 42`` or a bare number). Writes ``items.jsonl`` (per item: ``correct``,
``pred``, ``gold``, ``output``) and ``metrics.json`` (``value``, ``n``, ``stderr``).

    python -m awm.wm.score_items --model CKPT --items items.jsonl --out DIR [--limit N]
"""

from __future__ import annotations

import argparse
import json
import math
import re
from pathlib import Path

NUM_RE = re.compile(r"-?\d[\d,]*(?:\.\d+)?")
PROMPT = "Question: {q}\nAnswer: Let's think step by step."


def last_number(text: str) -> str | None:
    if "####" in text:
        text = text.split("####")[-1]
    nums = NUM_RE.findall(text)
    return nums[-1].replace(",", "") if nums else None


def same_number(a: str | None, b: str | None) -> bool:
    if a is None or b is None:
        return False
    try:
        return math.isclose(float(a), float(b), rel_tol=0, abs_tol=1e-6)
    except ValueError:
        return a.strip() == b.strip()


def generate(model_path: str, prompts: list[str], max_new_tokens: int, batch_size: int) -> list[str]:
    try:
        from vllm import LLM, SamplingParams  # type: ignore

        llm = LLM(model=model_path, gpu_memory_utilization=0.85, max_model_len=4096)
        outs = llm.generate(prompts, SamplingParams(temperature=0.0, max_tokens=max_new_tokens))
        return [o.outputs[0].text for o in outs]
    except ImportError:
        pass
    import torch  # type: ignore
    from transformers import AutoModelForCausalLM, AutoTokenizer  # type: ignore

    tok = AutoTokenizer.from_pretrained(model_path)
    tok.padding_side = "left"
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    model = AutoModelForCausalLM.from_pretrained(model_path, torch_dtype=torch.bfloat16,
                                                 device_map="auto")
    model.eval()
    texts: list[str] = []
    for i in range(0, len(prompts), batch_size):
        batch = prompts[i:i + batch_size]
        enc = tok(batch, return_tensors="pt", padding=True).to(model.device)
        with torch.no_grad():
            gen = model.generate(**enc, max_new_tokens=max_new_tokens, do_sample=False,
                                 pad_token_id=tok.pad_token_id)
        for j in range(len(batch)):
            texts.append(tok.decode(gen[j, enc["input_ids"].shape[1]:], skip_special_tokens=True))
    return texts


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--model", required=True)
    ap.add_argument("--items", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--max-new-tokens", type=int, default=512)
    ap.add_argument("--batch-size", type=int, default=8)
    a = ap.parse_args()

    rows = [json.loads(line) for line in Path(a.items).read_text().splitlines() if line.strip()]
    if a.limit:
        rows = rows[: a.limit]
    prompts = [PROMPT.format(q=r["question"]) for r in rows]
    outputs = generate(a.model, prompts, a.max_new_tokens, a.batch_size)

    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    correct = 0
    with (out / "items.jsonl").open("w") as fh:
        for r, text in zip(rows, outputs):
            pred = last_number(text)
            gold = last_number(str(r["gold"]))
            ok = same_number(pred, gold)
            correct += ok
            fh.write(json.dumps({"id": r["id"], "correct": ok, "pred": pred, "gold": gold,
                                 "output": text[-600:]}) + "\n")
    n = len(rows)
    value = correct / n if n else 0.0
    (out / "metrics.json").write_text(json.dumps({
        "value": value, "n": n, "stderr": math.sqrt(value * (1 - value) / n) if n else None,
        "metric": "accuracy",
    }, indent=2))
    print(f"{correct}/{n} = {value:.4f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
