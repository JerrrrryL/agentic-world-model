"""Stage the Prime Intellect speedrun task directory from the pinned submodule.

There is one task, so unlike the AIRS adapter this generates nothing — it only
materialises the files that belong to upstream. Everything else in the task
directory is ours and is committed: the Dockerfile, the compose file, task.toml,
solution/solve.sh, and tests/ (our record checker, which upstream has no
equivalent of).

Why these files are staged rather than committed: the release carries no
LICENSE, so copyright is reserved by default and nothing grants us the right to
redistribute it in another repository. GitHub's terms do grant the right to fork
it on GitHub, which is what third_party/frontier-automated-speedrun is. So the
bytes live there, pinned to a commit, and land in the task directory only when
somebody runs this.

``run.sh`` and ``verify.py`` are a separate case and ARE committed: they were
never published in the release. Agents read them during their runs and the tool
results in the published traces contain both files verbatim, so ours are
recovered from that record, not copied from a repository.
"""

from __future__ import annotations

import argparse
import shutil
import subprocess
from pathlib import Path

from awm.paths import REPO_ROOT

UPSTREAM_ROOT = REPO_ROOT / "third_party" / "frontier-automated-speedrun"
UPSTREAM_URL = "https://github.com/PrimeIntellect-ai/frontier-automated-speedrun"
TASK_DIR = REPO_ROOT / "tasks" / "speedrun_pi" / "track3_optimizer"

#: ``upstream path -> path inside the task directory``. Both are verbatim.
STAGED = {
    "program.md": "instruction.md",
    "train_gpt_simple.py": "environment/train_gpt_simple.py",
}

HEADER = (
    "<!-- Verbatim copy of {src} from {url} @ {sha} .\n"
    "     Staged by `python -m awm.adapters.speedrun_pi stage`; not committed, because the\n"
    "     upstream release carries no LICENSE. Our deviations from the official protocol\n"
    "     are recorded in task.toml, never here. -->\n\n"
)


def upstream_commit() -> str:
    return subprocess.run(
        ["git", "-C", str(UPSTREAM_ROOT), "rev-parse", "HEAD"],
        capture_output=True,
        text=True,
        check=True,
    ).stdout.strip()


def stage(task_dir: Path = TASK_DIR) -> list[Path]:
    if not (UPSTREAM_ROOT / "program.md").exists():
        raise FileNotFoundError(
            f"upstream not checked out at {UPSTREAM_ROOT}; run "
            "`git submodule update --init third_party/frontier-automated-speedrun`"
        )
    sha = upstream_commit()
    written = []
    for src_name, dest_rel in STAGED.items():
        src = UPSTREAM_ROOT / src_name
        dest = task_dir / dest_rel
        dest.parent.mkdir(parents=True, exist_ok=True)
        if dest_rel.endswith(".md"):
            body = src.read_text(encoding="utf-8")
            dest.write_text(
                HEADER.format(src=src_name, url=UPSTREAM_URL, sha=sha) + body, encoding="utf-8"
            )
        else:
            shutil.copy2(src, dest)
        written.append(dest)
    return written


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("command", choices=["stage"])
    ap.add_argument("--task-dir", type=Path, default=TASK_DIR)
    args = ap.parse_args(argv)
    for p in stage(args.task_dir):
        print(f"staged {p.relative_to(REPO_ROOT)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
