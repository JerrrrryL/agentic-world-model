"""Command line entry point: ``awm``.

Thin wiring over the library. Everything it prints is meant to be readable in a
terminal and greppable in a log; anything a program should consume comes out of
the library or the parquet index instead.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from awm import paths


def _fetch(args: argparse.Namespace) -> int:
    from awm.traj import fetch

    sources = list(fetch.FETCHERS) if args.source == "all" else [args.source]
    for source in sources:
        print(f"fetching {source} ...")
        if source == "posttrainbench":
            benchmarks = fetch.PTB_CORE_BENCHMARKS
            if args.observe or args.all:
                benchmarks = benchmarks + fetch.PTB_OBSERVE_BENCHMARKS
            result = fetch.fetch_posttrainbench(
                configs=fetch.ALL_CONFIGS if args.all else fetch.PTB_DEFAULT_CONFIGS,
                benchmarks=benchmarks,
            )
        else:
            result = fetch.FETCHERS[source]()
        print("  " + str(result))
    return 0


def _convert_pi(limit: int | None) -> int:
    from awm.traj import convert_pi

    raw = paths.raw_dir("pi_speedrun")
    if not (raw / "traces").is_dir():
        print(f"pi_speedrun not fetched at {raw} — run `awm traj fetch pi_speedrun`", file=sys.stderr)
        return 1
    metas = convert_pi.convert_all(raw, paths.events_dir("pi_speedrun"), limit=limit)
    print(f"pi_speedrun: {len(metas)} runs -> {paths.events_dir('pi_speedrun')}")
    return 0


def _convert_ptb(limit: int | None) -> int:
    from awm.traj import posttrainbench as ptb

    raw = paths.raw_dir("posttrainbench")
    if not raw.is_dir():
        print(f"posttrainbench not fetched at {raw} — run `awm traj fetch posttrainbench`",
              file=sys.stderr)
        return 1
    out = paths.events_dir("posttrainbench")
    runs = list(ptb.iter_run_dirs(raw))
    if limit:
        runs = runs[:limit]
    failed = []
    skipped = []
    for run in runs:
        try:
            ptb.convert_run_dir(run, out)
        except ptb.NoAgentOutput:
            # The CLI died before emitting anything — `opencode: command not
            # found`, an unknown flag, no CUDA. 41 runs, none with a
            # metrics.json. Nothing was published to convert, so this is not a
            # conversion failure and must not red the exit code.
            skipped.append(run.run_id)
        except Exception as exc:  # one malformed run must not abandon the batch
            # run_id, not the directory name: two agent configurations hold the
            # same 28 run names, so the bare name identifies neither.
            failed.append((run.run_id, exc))
    ok = len(runs) - len(failed) - len(skipped)
    print(f"posttrainbench: {ok}/{len(runs)} runs -> {out}")
    if skipped:
        print(f"  skipped {len(skipped)} run(s) with no agent output")
    for name in skipped:
        print(f"  SKIPPED {name}: no agent output", file=sys.stderr)
    for name, exc in failed:
        print(f"  FAILED {name}: {type(exc).__name__}: {exc}", file=sys.stderr)
    return 1 if failed else 0


def _convert(args: argparse.Namespace) -> int:
    rc = 0
    if args.source in ("all", "pi_speedrun"):
        rc |= _convert_pi(args.limit)
    if args.source in ("all", "posttrainbench"):
        rc |= _convert_ptb(args.limit)
    return rc


def _run(args: argparse.Namespace) -> int:
    """Run a task directory under Harbor with the host paths it needs.

    The generated compose files require absolute host paths (compose resolves a
    relative volume source against its own directory), and those paths differ per
    machine because ``data/`` is a symlink. Rather than committing one layout,
    the task files demand the variables and this fills them in.
    """
    import os
    import subprocess

    task_dir = Path(args.task).resolve()
    if not (task_dir / "task.toml").exists():
        print(f"no task.toml under {task_dir}", file=sys.stderr)
        return 1

    env = dict(os.environ)
    env.setdefault("AWM_AIRS_PREPARED", str(paths.data_root().resolve() / "assets/airs/prepared"))
    env.setdefault("AWM_FINEWEB_DIR", str(paths.data_root().resolve() / "assets/fineweb10B"))

    cmd = [args.harbor, "run", "-p", str(task_dir), "-a", args.agent, "-o", str(args.jobs_dir)]
    if args.model:
        cmd += ["-m", args.model]
    cmd += args.harbor_arg
    print(" ".join(cmd))
    return subprocess.run(cmd, env=env).returncode


def _index(args: argparse.Namespace) -> int:
    from awm.traj import index

    df = index.build()
    path = index.save(df)
    print(f"{len(df)} runs -> {path}")
    if len(df):
        by_source = df.groupby("source", dropna=False).size()
        for source, n in by_source.items():
            print(f"  {source}: {n}")
    return 0


def _scope_list(args: argparse.Namespace) -> int:
    from awm import scope

    entries = scope.load(args.bench)
    if args.self_run:
        entries = [e for e in entries if e.self_run]
    if args.ids_only:
        print("\n".join(e.id for e in entries))
        return 0
    width = max((len(e.id) for e in entries), default=0)
    for e in entries:
        runs = f"x{e.n_runs}" if e.n_runs > 1 else "  "
        note = "" if e.self_run else "  (analysis only)"
        print(f"{e.id:<{width}} {runs}  {e.metric.get('name', '')}{note}")
    total = sum(e.n_runs for e in entries)
    print(f"\n{len(entries)} tasks, {total} runs")
    return 0


def _scope_check(args: argparse.Namespace) -> int:
    from awm import scope

    issues = scope.check()
    for issue in issues:
        print(f"  - {issue}")
    print(f"{len(issues)} issue(s)")
    return 1 if issues else 0


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(prog="awm", description=__doc__)
    p.add_argument("--data-root", type=Path, help="override AWM_DATA_ROOT for this call")
    sub = p.add_subparsers(dest="group", required=True)

    traj = sub.add_parser("traj", help="fetch and convert trajectories").add_subparsers(
        dest="cmd", required=True
    )

    f = traj.add_parser("fetch", help="download an upstream release into raw/")
    f.add_argument("source", choices=["all", "pi_speedrun", "posttrainbench"])
    f.add_argument(
        "--all",
        action="store_true",
        help="posttrainbench: every agent configuration and benchmark (7.3 GB, 1842 runs) "
        "instead of the default four configurations on the five core benchmarks (0.6 GB)",
    )
    f.add_argument(
        "--observe",
        action="store_true",
        help="posttrainbench: also take the two LLM-judged benchmarks "
        "(arenahardwriting, healthbench)",
    )
    f.set_defaults(func=_fetch)

    c = traj.add_parser("convert", help="convert raw/ into the unified event schema")
    c.add_argument("--source", default="all", choices=["all", "pi_speedrun", "posttrainbench"])
    c.add_argument("--limit", type=int, help="convert at most N runs per source")
    c.set_defaults(func=_convert)

    i = traj.add_parser("index", help="rebuild the run index")
    i.set_defaults(func=_index)

    r = sub.add_parser(
        "run",
        help="run a task directory under Harbor, supplying this machine's host paths",
    )
    r.add_argument("task", help="path to a task directory (tasks/airs/<name>, ...)")
    r.add_argument("-a", "--agent", default="oracle")
    r.add_argument("-m", "--model")
    r.add_argument("-o", "--jobs-dir", type=Path, default=Path("data/smoke/jobs"))
    r.add_argument("--harbor", default="harbor", help="path to the harbor executable")
    r.add_argument(
        "--harbor-arg",
        action="append",
        default=[],
        metavar="ARG",
        help="pass one argument through to harbor; repeat for more",
    )
    r.set_defaults(func=_run)

    sc = sub.add_parser("scope", help="query the task registry").add_subparsers(
        dest="cmd", required=True
    )
    sl = sc.add_parser("list", help="list tasks in scope")
    sl.add_argument("--bench", choices=["posttrainbench", "airs", "speedrun_pi"])
    sl.add_argument(
        "--self-run", action="store_true", help="only tasks we run ourselves, not analysis-only"
    )
    sl.add_argument("--ids-only", action="store_true")
    sl.set_defaults(func=_scope_list)

    sk = sc.add_parser("check", help="reconcile the registry against upstream and the docs")
    sk.set_defaults(func=_scope_check)
    return p


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    if args.data_root:
        import os

        os.environ["AWM_DATA_ROOT"] = str(args.data_root)
    return args.func(args)


if __name__ == "__main__":
    raise SystemExit(main())
