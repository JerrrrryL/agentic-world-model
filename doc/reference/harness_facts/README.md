# Benchmark harness 事实清单(2026-08-24 调研)

六份调研原文由调研 agent 实际 clone 仓库、读取 HF 数据集、下载样本轨迹后产出,英文,每条事实标 VERIFIED / UNVERIFIED。本页是中文索引,接入时以原文为准。设计结论见 `doc/spec/2026-08-24-repo-architecture.md`;**PoC 10 题(Speedrun 1 + AIRS 8 + PostTrainBench 1)的逐题 GPU 需求见 [gpu_requirements.md](gpu_requirements.md)**(2026-08-25)。

| 基准 | 原文 | 任务定义格式 | 官方 harness | 自有 agent 插入点 | 公开轨迹 | 仓库状态 / License |
|---|---|---|---|---|---|---|
| PostTrainBench | [posttrainbench.md](posttrainbench.md) | 单一 prompt 模板 × (4 模型 × 7 基准);每基准目录 = evaluate.py + info.json + test_data.json | bash `run_task.sh` + Apptainer + HTCondor(完整运行器) | `agents/<name>/solve.sh` | HF 1,842 run / 28.9 GB / Apache-2.0 / 不 gated;Claude Code、Codex 原生 JSONL 逐行加时间戳 + system_monitor + judge | 活跃(3 个月 105 commit),MIT |
| AIRS-Bench | [airs_bench.md](airs_bench.md) | project_description.md + prepare.py + evaluate_prepare.py + evaluate.py + metadata.yaml(与 harness 无关) | aira-dojo(公开版**跑不了 AIRS**)/ MLGym | 无;需自建运行器 | 无 | 停更,CC BY-NC |
| Speedrun(PI) | [pi_speedrun.md](pi_speedrun.md) | program.md + train_gpt_simple.py;run.sh / verify.py 可从轨迹恢复 | 自研 launcher(bwrap + netns + 代理),未开源 | 任意 headless CLI agent + `/goal` | 41 run / 50 MB,已统一为四类事件 schema + subagents + scratchpad + manifest | 8 月发布后静默,无 LICENSE |
| RE-Bench + MALT | [rebench_malt.md](rebench_malt.md) | METR Task Standard `TaskFamily`;METR 已迁到 Inspect 版(Docker Hub 有镜像) | Vivaria(旧)/ Inspect(现) | 替换 Inspect solver | MALT 点击式 gated,default 1.66 GB,DAG 节点;人类转录未发布 | 原仓停更,MIT |
| AutoLab | [autolab.md](autolab.md) | Harbor task 目录(task.toml + instruction.md + environment/ + tests/test.sh) | Harbor(锁 0.3.0) | Harbor BaseAgent | live-lab 161 run 打包在网站 JS chunk 里;仓库内 4 条 ATIF | 6 月后停更,无 LICENSE |
| LLM-Speedrunner | [llm_speedrunner.md](llm_speedrunner.md) | 记录链复现(已被 PI 设定取代) | Slurm + Aider,Meta 内网路径硬编码 | — | 无 | 停更,CC BY-NC |

首批只接前三家;AutoLab 一节同时是 Harbor 框架本身(task 格式、agent 接口、ATIF、超时)的事实来源。
