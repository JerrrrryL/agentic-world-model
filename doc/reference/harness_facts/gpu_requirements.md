# Harness 事实清单:PoC 10 题逐题 GPU 需求

**用途**:PoC 自跑范围(3 个 benchmark、10 道题)的硬件事实——GPU 型号/数量、单次训练时长、墙钟预算、显存、存储、harness 与已知的坑。估计值标 ≈(按 1×H100/H200 折算);官方协议为准绳。范围决策见《benchmark 案例与数据清单》二 / 3.2 / 3.4 / 3.5,机器可读清单在 `splits/`(2026-08-27 起取代 `scope/`,见 `splits/airs/gpu-heavy-8-v1.yaml`);核定日期 2026-08-25。

## 0. 汇总

| Benchmark | 题数 | 每题硬件 | 墙钟/题 | GPU·h / seed | harness |
|---|---|---|---|---|---|
| NanoGPT Speedrun(PI 设定) | 1 | 8×H200(H100 可替) | 24h(PoC 切片;官方 ≥3 seed、续跑 ≤8 天) | 192 | program.md 协议,自建 run.sh/verify.py |
| AIRS-Bench(GPU-heavy 8 题) | 8 | 1×H100/H200 | 24h(官方协议) | 192 | Harbor + `awm/adapters/airs.py`(公开版 aira-dojo 跑不了 AIRS) |
| PostTrainBench | 1 配置 | 1×H100 | 10h(官方协议) | 10 | 官方 harness |
| **合计** | **10** | — | — | **≈394 / seed** | — |

3 seeds ≈ 1,180 GPU·h。除 Speedrun 独占整节点外全部单卡;没有任务需要 >80GB 显存或多机。

## 1. NanoGPT Speedrun(Prime Intellect 设定)—— 1 题

- **硬件**:整节点 8×H200;指标是 `train_steps`(与硬件无关),8×H100 可替换,只降实验吞吐 ~30–50%,不影响分数。
- **实测节奏**(取自 Fable 5 公开轨迹):1 trial ≈ 8.5–9 min;8-trial 验证 ≈ 60–70 min;`RUN_TIMEOUT` 2h;24h ≈ 90–150 次 1-trial 实验。
- **显存/存储**:124M GPT,显存远低于单卡上限;FineWeb 分片只读挂载,磁盘 ~20–30 GB。
- **协议**:官方 ≥3 seed 并行 × ~24h 初筛 → 只续跑最优,单 run ≤8 天(≈770–1,900 GPU·h/配置);PoC = 1 seed × 24h = 192 GPU·h。
- **坑**:`run.sh` / `verify.py` 未随仓库公开,需按 program.md 自建(8 seed `0xC0FFEE+0..7`、门槛 3.27859、一步=一次 fwd-bwd);沙箱 = bwrap + 网络命名空间 + 只放行模型 API 的 logging proxy。
- **轨迹**:41 run 公开(events / subagents / scratchpad / manifest),Tier A 可直接分析。

## 2. AIRS-Bench GPU-heavy 8 题

官方协议:每 run 24h × 1×H200,≥10 seeds;允许联网 + 193 个 ≤2021 缓存预训练模型(最新为 deberta-v3-large);agent 从零写码、训模、产 `submission.csv`。
harness:论文用 aira-dojo Greedy(搜索树 journal)与 MLGym ReAct(.traj),但**公开版 aira-dojo 跑不了 AIRS**(任务注册表只有 MLE-bench);我们经 Harbor + `awm/adapters/airs.py` 从 `metadata.yaml` 生成 task 目录自跑,只评最终 `submission.csv`(MLGym 口径),搜索结构来自 agent 遥测。官方轨迹未发布(S2=0),须自产。

| # | 任务 | 训练/测试集(大小) | SOTA(方法 / 分数) | 逼近路径 | 单次训练 ≈ | 显存 ≈ | 24h 完整实验数 |
|---|---|---|---|---|---|---|---|
| 1 | CvMolecularPropertyPredictionQm9MeanAbsoluteError | 110,831 / 10,000(49MB) | EquiformerV2 / 0.021 | 从零训等变 GNN(SchNet→PaiNN 级;SOTA 级需 1 天+) | 3–12h | <20GB | 2–6 |
| 2 | GMolecularPropertyPredictionQm9MeanAbsoluteError | 同上 | EquiformerV2 / 7.53 | 同上 | 3–12h | <20GB | 2–6 |
| 3 | R2AbsMolecularPropertyPredictionQm9MeanAbsoluteError | 同上 | EquiformerV2 / 0.033 | 同上 | 3–12h | <20GB | 2–6 |
| 4 | U0MolecularPropertyPredictionQm9MeanAbsoluteError | 同上 | EquiformerV2 / 5.83 | 同上 | 3–12h | <20GB | 2–6 |
| 5 | GraphRegressionZincMae | 220,011 / 5,000(21MB) | ESA(图注意力)/ 0.017 | 从零训 GNN / 图 Transformer | 1–4h | <20GB | 5–15 |
| 6 | SentimentAnalysisYelpReviewFullAccuracy | 650,000 / 50,000(323MB) | SplitEE(BERT 系)/ 0.778 | 微调 roberta/deberta(base→large) | base 1–2h;large 3–5h | 20–40GB | 4–10 |
| 7 | QuestionAnsweringDuoRCAccuracy(ParaphraseRC) | 69,524 / 15,857(38MB) | ALBERT / 0.4648 | 微调 ALBERT / Longformer(长上下文) | 2–6h | 20–40GB | 3–8 |
| 8 | ReadingComprehensionSquadExactMatch | 87,599 / 10,570(16MB) | SplaXBERT(BERT 系)/ 0.858 | 微调 roberta/deberta-large | 1–2h | 20–40GB | 8–15 |

已知的坑:
- **SQuAD 缓存泄漏**:模型缓存含 `deepset/roberta-large-squad2`、`distilbert-*-distilled-squad` 等现成 checkpoint,零训练即近 SOTA;自跑须从缓存剔除 `*squad*` 系列,否则该题退化为轻任务。
- **QM9 四题同质**:同数据、同方法族,搜索树高度相似——用于压资源调度信号,不用于假设多样性分析(多样性用备选 5 题补)。
- CPU 路线(TF-IDF / GBDT / 手工描述符)在这 8 题全部远离 SOTA——GPU 是硬需求,这正是选它们的原因。

落选备选(轻 GPU / CPU,保留在《清单》3.4):WinoGrande(~1h)、SICK×2(分钟级)、TS×3(CPU 可跑)。

## 3. PostTrainBench —— 1 配置

- **官方**:1×H100 × 10h / 配置;给 base LLM + 目标基准,agent 自选任意后训练策略(SFT/LoRA/RL/数据合成)。
- **PoC 配置(暂定,可换)**:**Qwen3-4B × GSM8K**——客观判分、数据小、10h 内多种后训练路线可行;更快冒烟可用 Qwen3-1.7B × GSM8K。
- **显存/存储**:4B 模型 LoRA/SFT 单卡 80GB 充裕(全参 bf16+AdamW 偏紧,LoRA 更稳);base 权重 ~8GB。
- **坑**:已知 hack 形态(测试集训练、下载现成 instruct 权重、盗 API key 生成数据)需反作弊巡检,官方有裁判日志格式可对照。
- **轨迹**:官方 20 核心配置全轨迹已公开(HF `aisa-group/PostTrainBench-Trajectories`,含 system monitor 时间轴);自产这 1 配置主要用于对齐官方轨迹格式、校准解析器。
