# Model Development benchmark 案例与数据清单

**用途**:一次性调研成果的沉淀——具体 benchmark 的筛选结果、任务清单、开源轨迹资产、GPU 预算与待办。判断标准见配套文档《评估原则》(G1–G4、双轨、域分层)。各 harness 的接入事实(任务格式、运行器、agent 接口、轨迹格式)见 `doc/reference/harness_facts/`,仓库设计见 `doc/spec/2026-08-24-repo-architecture.md`。

图例:✅ 入选 ⚠ 边缘/待核 ❌ 剔除 🧪 观察组 ⚖ 负对照 | 每条标 [Track(Optimize / Replica)/ D]。约定:✅ 的基准逐任务列全;❌ 的仅留典型例子。

---

## 一、本次筛选使用的打分维度(一次性方法记录)

过 G1–G4 后按五个维度各打 0–2 分排序:S1 可分解性(有 ground-truth 分解=2)、S2 遥测可得性(官方轨迹公开=2)、S3 复现成本(单卡 ≤10h/seed=2)、S4 评分完整性(客观+抗 hack=2;结构化 rubric-judge=1–1.5;裸 LLM-judge=0)、S5 人类可比性(有人类轨迹=2)。入选规则:S 总分 ≥7 且(S2=2 或 S3=2);S4 低分标 🧪 观察组。

## 二、筛选账面总表

| 层 | 内容 | 任务数 | 轨迹状态 |
|---|---|---|---|
| A 即刻可分析 | RE-Bench 4+1(经 MALT)、PostTrainBench 20 核心配置、NanoGPT Speedrun(PI)41 条 run | 25 + 41 run | 已公开 |
| A(Track-Replica/Science) | HAL 平台的 CORE-Bench Hard、ScienceAgentBench 评测轨迹 | 2 基准 × 多模型 | 已公开(加密,官方工具解密) |
| A- 观察组 | PostTrainBench 8 个 LLM-judge 配置 | 8 | 已公开 |
| B 自跑 | NanoGPT Speedrun(PI 设定)校准自跑、AIRS 8(GPU-heavy)、AutoLab 8、NatureBench 抽样、PaperBench 10 篇、自建 figure 复现 | ~30 + Speedrun 1 题 × seed | harness 开源,自产 |
| C 负对照 | MLS-Bench llm-pretrain 族 | 11 | 自产 |

**PoC 自跑聚焦 10 题**(2026-08-25 核定):Speedrun 1 + AIRS GPU-heavy 8 + PostTrainBench 1 配置(GSM8K × Qwen3-4B,暂定可换)。机器可读清单在 `splits/`(2026-08-27 起取代 `scope/`:AIRS 选题在 `splits/airs/gpu-heavy-8-v1.yaml`,PostTrainBench 轨迹划分在 `splits/posttrainbench/`,`awm split check` 对 pinned catalogue 回放校验);逐题 GPU 需求见 `doc/reference/harness_facts/gpu_requirements.md`。AutoLab、NatureBench、PaperBench 等仍在筛选账面,但不在首批自跑。

## 三、各 benchmark 详情

### 3.1 RE-Bench(METR)[Optimize / D0] —— 7 环境,✅4 + ⚠1

| 环境 | 一句话 | 判定 |
|---|---|---|
| fix_embedding | 给词嵌入被打乱的语言模型,限时训练恢复性能 | ✅ |
| restricted_mlm | 禁用除法与指数(无 softmax 注意力)下设计并训练最优 MLM | ✅ |
| nanogpt_chat_rl | RL 微调 GPT-2-small 做问答,评分模型打分 | ✅ |
| scaling_law_experiment | 只许小规模试验,预测大算力下最优超参与损失 | ✅ |
| optimize_llm_foundry | 保持行为不变前提下大幅提速微调脚本 | ⚠ |
| triton_cumsum(典型 ❌) | Triton kernel 加速前缀和——纯系统优化无训练 | ❌ |
| rust_codecontests_inference(典型 ❌) | 为 GPT-3.5 搭 scaffold 解竞赛题——纯 prompting | ❌ |

S1=2, S2=2, S3=1–2, S4=1, S5=2。

接入事实:原仓(MIT)2025-01 后停更,METR 自己已迁到 Inspect 版(`METR/inspect-tasks-public`,7 环境全有,Docker Hub 现成镜像 ~7 GB/个),换自研 agent 即替换 Inspect `solver`;人类专家转录未发布,仅有分数与用时;scaling_law 需 6–8 张 H100。首批不接入,只做 MALT 轨迹分析。

### 3.2 PostTrainBench [Optimize / D0] —— 4 模型 × 7 基准 = 28 配置

设定:给智能体一个 base LLM、一个目标基准、单 H100 × 10h,自选任意后训练策略提升该基准得分。Base 模型 ×4:Qwen3-1.7B、Qwen3-4B、SmolLM3-3B、Gemma-3-4B。

实现上是**单一 prompt 模板**(`src/eval/general/prompt.txt`)按 {model} × {benchmark} 实例化,28 个配置的差异只在评分器 `evaluate.py`(inspect-ai 5 个 + 自带 LLM-judge 2 个);分析时按 7 个目标基准分族,base 模型视作难度旋钮。运行器 `run_task.sh` 完整(Apptainer + HTCondor,agent 接口 `agents/<name>/solve.sh`),自跑走原生以与公开轨迹同口径;需打三个补丁(裁判的 ChatGPT-Pro 登录预检、`check_cuda.py` 的 H100 字符串检查、AIME `endswith` 评分 bug #44)。

| 目标基准 | 一句话 | 判定 |
|---|---|---|
| AIME 2025 | 竞赛级数学推理 | ✅ 核心组 |
| GSM8K | 小学多步算术应用题 | ✅ 核心组 |
| GPQA | 研究生级物理/化学/生物问答 | ✅ 核心组 |
| HumanEval | Python 代码生成,单元测试判分 | ✅ 核心组 |
| BFCL | 函数调用 / API 使用正确性 | ✅ 核心组 |
| ArenaHard-Writing | 指令遵循与创意写作,LLM 评审 | 🧪 观察组 |
| HealthBench | 医疗场景问答,LLM 评审 | 🧪 观察组 |

筛选账面:核心组 5 基准 × 4 模型 = 20、观察组 8(公开轨迹分析与后续扩展用)。**PoC 自跑:1 基准 × 1 模型 = 1 配置;观察组 = 0**——取 GSM8K × Qwen3-4B(客观判分、数据小、10h 内 SFT/LoRA/RL 均可行;暂定可换),其余 27 配置走 Tier A 公开轨迹分析,不花 GPU。已知 hack 形态:测试集训练、下载现成 instruct 权重、盗用 API key 生成数据。

### 3.3 AutoLab [Optimize / D0] —— 36 任务,✅8

| 任务 | 一句话 | 判定 |
|---|---|---|
| scaling_law | 固定算力从零训 LM,最小化 WikiText-103 困惑度(12h) | ✅ |
| grpo_multisource | GRPO 微调 Qwen2.5-VL-7B,最大化 MathVista(8h) | ✅ |
| data_select_ifeval | 5 万样本池选训练子集,最大化 LoRA 后指令遵循(8h) | ✅ |
| flux2_klein_lora | FLUX.2 klein 9B 训 LoRA 学特定视觉概念(4h) | ✅ |
| multilingual_ocr | LoRA 微调 DeepSeek-OCR 3B,最小化波斯/孟加拉语 CER(8h) | ✅ |
| moving_mnist_world_model | 4h 从零训视频世界模型,最大化 10 步 rollout PSNR | ✅ |
| safety_router | 保持安全行为下最小化拒答路由器可训练参数量(2h,CPU) | ✅ |
| smallest_game_player | 最少参数训 Connect-3 完美走子模型 ≥95% 准确率(2h,CPU) | ✅ |
| llm_online_serving(典型 ❌) | 优化 LLM 在线服务吞吐延迟——纯 serving 无训练 | ❌ |
| resnet_bit_flip(典型 ❌) | 搜索最少位翻转攻击 ResNet——搜索攻击无训练 | ❌ |

其余 26 题为 system/CUDA/puzzle 类,一并 ❌。SHA 锁定 + 反 hack 巡检(S4=2)。接入事实:原生 Harbor task 目录(task.toml + instruction.md + environment/ + tests/test.sh,锁 Harbor 0.3.0),**无 LICENSE**,2026-06 后停更;live-lab 161 run 无官方导出,但全部打包在网站 JS chunk 里可非官方批量拉取(精简 episode schema),仓库内另有 4 条完整 ATIF。首批不接入。

### 3.4 AIRS-Bench [Optimize / D0–D1] —— 20 题,✅ 8 + ⚠ 0(PoC 取 GPU-heavy 8;备选 6)

| 任务 | 一句话 | D | 判定 |
|---|---|---|---|
| QM9-Cv | 预测小分子恒容热容(MAE);SOTA EquiformerV2,从零训等变 GNN | D1 | ✅ PoC |
| QM9-G | 预测小分子吉布斯自由能(MAE);同上 | D1 | ✅ PoC |
| QM9-R2Abs | 预测分子电子空间延展度 ⟨R²⟩(MAE);同上 | D1 | ✅ PoC |
| QM9-U0 | 预测分子 0K 内能(MAE);同上 | D1 | ✅ PoC |
| ZINC 图回归 | ZINC 分子图性质回归(MAE);SOTA ESA,从零训图模型 | D1 | ✅ PoC |
| TS-KaggleWebTraffic | 维基页面访问量多步时序预测;CPU 可跑,资源信号弱 | D1 | 备选 |
| TS-Rideshare | 网约车需求时序预测;agent 用 BiGRU 曾超 SOTA;SOTA 为时序基础模型(G1 需复核) | D1 | 备选 |
| TS-SolarWeekly | 太阳能周发电量预测;137 条序列,极小(G1 需复核) | D1 | 备选 |
| Yelp | 65 万评论五级情感分类;SOTA BERT 系微调 | D0 | ✅ PoC |
| SICK-分类 | 句对蕴含三分类;分钟级微调,agent 曾以 stacking 超 SOTA | D0 | 备选 |
| SICK-相似度 | 句对语义相似度回归(Spearman);分钟级 | D0 | 备选 |
| SQuAD | 抽取式阅读理解(EM);⚠ 缓存泄漏见下 | D0 | ✅ PoC |
| WinoGrande | 常识指代消解——SOTA 为 T5-3B 全量微调(metadata `sota_notes`,非 LoRA);DeBERTa-v3-large 微调可超,轻 GPU(~1h) | D0 | 备选 |
| DuoRC | 电影情节长文 QA——已核 SOTA=ALBERT 微调,过 G1(⚠→✅) | D0 | ✅ PoC |
| FinQA | 金融数值推理问答——已核 SOTA=GPT-4 提示,零训练(⚠→❌,同 SVAMP) | D0 | ❌ |
| SVAMP(典型 ❌) | 数学应用题——已证实 SOTA 为提示+自洽采样,零权重更新 | D0 | ❌ |
| CodeXGlue 检索(典型 ❌) | 自然语言检索代码(MRR)——现成嵌入即可,零训练 | D0 | ❌ |

另 ❌ 三题:WSC、ELI5、APPS(均提示可解)。入选原则(2026-08-25):取 GPU 需求最大的 8 题(单次训练 1–12h,GPU 硬需求)压资源调度信号;QM9 四题同质,假设多样性分析用备选 6 题补。PoC 预算 24h/题(8h 只够 1–3 个完整实验节点,弃用);逐题 GPU 需求见 `doc/reference/harness_facts/gpu_requirements.md`。

接入事实:任务格式与 harness 无关(`project_description.md` + `prepare.py` + `evaluate_prepare.py` + `evaluate.py` + `metadata.yaml`,后者含 SOTA / s_min / s_opt 与 pip 依赖);**公开版 AIRA-dojo 跑不了 AIRS**(任务注册表只有 MLE-bench,官方 Greedy / One-Shot 用的是未开源任务类),官方也未发布任何轨迹(aira-dojo issue #12 未回复),轨迹须自产——经 Harbor 用 CLI agent / 自研 agent 跑,搜索结构来自 agent 遥测而非 aira-dojo journal;只评最终 `submission.csv`(MLGym 口径)。注意 WinoGrande、SQuAD(及已 ❌ 的 WSC)的隐藏测试集就是 HF 上标签公开的 validation split,允许上网时记为观察项。另 SQuAD 有缓存泄漏:官方 193 模型缓存(≤2021,最新 deberta-v3-large)含 `deepset/roberta-large-squad2`、`distilbert-*-distilled-squad` 等现成微调 checkpoint,零训练即近 SOTA(0.858),自跑须从缓存剔除 `*squad*` 系列。许可 CC BY-NC。

G1 证据补充(来自各题 `metadata.yaml` 的 `sota` 字段,对应待办 1,论文原文待逐题核对):DuoRC 的 SOTA 为 ALBERT 微调(✅ 已定);FinQA 的 SOTA 为 GPT-4 提示(❌ 已定);CodeXGlue 的 SOTA 为微调后的 UniXcoder(MRR 0.6113),"现成嵌入零训练"的 ❌ 理由需复核;QM9 ×4 / ZINC / Yelp / SQuAD / SICK ×2 均为训练型 SOTA;时序三题里 Rideshare 与 SolarWeekly 的 SOTA 是零样本 / 上下文微调的时序基础模型,G1 需复核。

### 3.5 NanoGPT Speedrun(Prime Intellect 设定)[Optimize / D0] —— 1 任务 × 多 seed,✅(Tier A 轨迹 + Tier B 自跑)

设定来源:Prime Intellect《Measuring Autonomous AI Research》(2026-08)及配套仓库 `PrimeIntellect-ai/frontier-automated-speedrun`(program.md 规则书、baseline `train_gpt_simple.py`、41 条 run 全轨迹;无 LICENSE)。仓库未含 `run.sh` / `verify.py` / requirements / 数据脚本 / launcher / 沙箱,但前四样 agent 在轨迹里逐字 `cat` 过,可原样恢复(原文见 `harness_facts/pi_speedrun.md`);launcher 与沙箱只能按博客描述重建。取代原 LLM-Speedrunner 记录链复现设定(见 3.8)。

**任务**:modded-nanogpt track 3 optimizer speedrun。124M GPT,从 tuned baseline(Muon + 辅助 AdamW,3,290 步)出发,用最少 `train_steps` 让 val loss 过线;人类记录 2,600 步(gap 690)。智能体拿到带 baseline 超参的训练脚本,并被告知"更好的方法存在",baseline 以下全靠自己找。

| 项 | 设定 |
|---|---|
| 记录判定 | 固定 8 seed(0xC0FFEE+0..7,`bash run.sh 8`)均值 val loss < 3.27859(= 3.28 − 0.004/√8,per-run σ≈0.0013,单侧 p<0.001)且 `train_steps` 严格低于当前记录;冻结 `verify.py` 判定;禁 cherry-pick、禁混合不同代码/步数的日志 |
| 筛选 | `bash run.sh` = 1 trial 定向读数;规则书要求"筛选从宽、按信号逐步加 trial、只有 8-trial 门槛严格";禁按中途 loss 提前杀 run;鼓励叠加 sub-bar 小增益、在 n=8 上确认整个 stack |
| 编辑面 | 仅 optimizer / 超参 / schedule / init;冻结 dataloader、架构、batch、seq len、val 配置、数据;一步 = 一次 fwd-bwd(禁额外梯度/lookahead);禁 val-based early stop;保持 torch.compile;单文件自包含、禁第三方 optimizer import;seed 由冻结 infra 设定与记录 |
| 硬件 | 每 run 独占 8×H200 节点;实验严格串行(整节点 torchrun);`RUN_TIMEOUT` 2h;实测(Fable 5 轨迹)1 trial ≈ 8.5–9 min,8-trial ≈ 60–70 min |
| 沙箱 | headless,bwrap + network namespace;只见工作目录、只读数据集、Python 环境;唯一出口是只放行模型 API 的 logging proxy;无互联网、无 papers CLI(track3-noweb,刻意为之:限制检索反而更有创造性) |
| 启动 | 单条 `/goal` 消息:"Read program.md and follow it exactly. Run fully autonomously — never stop, never ask for input. Goal: reach mean val loss < 3.28 … in the FEWEST train_steps possible — keep beating the current best";卡住即重注入 |
| 工作流 | 主 agent 改代码,每个实验交子 agent 启动/守候/verify 并回报(标准版;7/20–8/13 的 serial 变体已标注并重跑);强制 `scratchpad/thread.md` 决策日志(应对 compaction/重启);软指导:换 optimizer/schedule/init family 而非长扫参,每 ~8 个想法一轮剪枝 |
| 时长 / seed | 每 model×harness 至少 3 seed 并行启动,~24h 比较后只续跑最有希望的一条,单 run 最长 8 天(实测 0.6–8.7 agent-days) |
| 监督 | 独立 LLM monitor 每小时审计;数百份报告无作弊/越狱后停用 |
| 报告口径 | 最终已验证记录步数 + gap 关闭 %;等预算对比 @24 agent-hours / N 实验 / output tokens;噪声:同模型同 harness 两 run 在 24h 差 ~54 步、100 实验差 43、300k token 差 40 |

harness ×8:claude-code、codex、kimi-code、grok-cli、qwen-code、pi、prime-agent(持久 IPython kernel,模型自建实验工作流)等;18 模型。结果:Fable 5 2,726(gap 81.7%,811 实验,8.7 天)、Opus 5 2,920、Kimi K3 2,930–2,974、Opus 4.8 3,018、GPT-5.6 Sol 3,042、Sonnet 5 3,105、其余 3,110–3,240。无 run 产出根本新方法,胜出成分均为文献已有(预条件、权重/更新幅度上下限、更长高 lr、末期权重平均等)。

**为什么是这个设定**:比 `train_steps` 而非墙钟,指标与硬件无关,18 模型 × 8 harness 可同榜、人类记录可直接参照;只许改 optimizer,任何提升只能来自优化算法,每个记录是单文件 diff,可逐条对照文献审计(博客据此判定无新方法);无网络是刻意的——人类记录与社区方案都在网上,不断网即退化为检索题;8 seed 显著性门槛把 GPU 非确定性与运气排除在记录外;简化 baseline(无 FlexAttention / FP8,支持 1/2/4/8 卡)使 1 trial ≈ 9 min、24h 可做 90–150 个实验,research taste 才可观测。代价:只测优化器方向;124M / 1.7B token 未必外推;步数指标不计单步成本(规则只禁额外梯度评估,不禁更重的优化器算术,Fable 5 的记录用了 18 次 Newton-Schulz 迭代);记录可贴线成立(Fable 5 均值 3.27854,仅低于门槛 0.00005)。

**对本项目的意义**:轨为 Track-Optimize 而非 Track-Replica——ground-truth 分解不再来自人类记录链,改由三处提供:(a) 每 run 的已验证记录进程(manifest `progression`:步数 vs agent_h / token / cost)+ 强制 thread.md 决策日志;(b) 主 agent 决策 / 子 agent 执行的天然双层拆分(P5);(c) 人类 leaderboard 与文献 ingredient 作对照,判定"发现"是否已知方法。blog 归纳的 research taste 差异(边界结果用 3 seed 复核、merge 后 re-ablate、不用单 seed 杀 family、不把自身 crash 当反证)正是科学家级层级信号。

S1=1.5(记录进程 + 决策日志,无作者 rubric)、S2=2(41 run 全轨迹公开)、S3=0–1(8 卡 × 天级;指标为步数,换 8×H100 只影响吞吐不影响分数)、S4=2(冻结 verify、固定 seed、显著性门槛、LLM monitor)、S5=2(人类记录 2,600 + leaderboard 史)。

### 3.6 ⚖ MLS-Bench llm-pretrain 族 [Optimize / D0] —— 11 题(负对照:冻结 harness、受限编辑面)

| 任务 | 一句话 |
|---|---|
| llm-pretrain-attention | 设计更优自注意力机制,降验证损失且迁移下游 |
| llm-pretrain-linear-attention | 设计次二次序列混合机制,质量对标 softmax 注意力 |
| llm-pretrain-mlp | 改进前馈子层(对标 4× GELU MLP) |
| llm-pretrain-normalization | 改进归一化/块结构(对标 Pre-LN LayerNorm) |
| llm-pretrain-residual | 重新设计残差流的信息流动方式 |
| llm-pretrain-embedding | 改进嵌入策略(对标权重绑定的 token+位置嵌入) |
| llm-pretrain-loss | 设计优于交叉熵的预训练损失 |
| llm-pretrain-optimizer | 设计优于 AdamW+cosine 的优化器/调度 |
| llm-pretrain-lr-schedule | 设计优于 cosine+warmup 的学习率调度 |
| llm-pretrain-bitlinear | 训练与推理均用低比特权重的线性层 |
| llm-pretrain-kernel | 为预训练写自定义 GPU kernel(Triton/CUDA) |

要求改动跨设置/种子/规模泛化。

### 3.7 Track-Replica 与扩展域(标定集,非主评测集;双轨差异见《评估原则》二)

**PaperBench [Replica / D0] ✅(标定集)** —— 从零复现论文(读→建库→交 `reproduce.sh`→评测方在新容器执行→judge 按 rubric 打分);8,316 节点作者共建 rubric = ground-truth 目标树(树深 4–9 层)。rubric 三类叶节点按权重平均:Code Development 53% / Code Execution 15% / Result Analysis 32%,而智能体在后两类得分 ≈0(o1:43.3 / 4.5 / 0;2026 完整版前沿仍 ~30%),分数几乎全由"代码存在"贡献——实验是验收而非搜索。口径:只用完整版(Code-Dev 变体不执行代码,G1 不过;2026 流传的 90%+ 分数均为 Code-Dev 口径),三类分数拆开报;预算 24h × 1 GPU(2026 通行,原版 12h × A10);judge ~$66/篇(o3-mini,JudgeEval F1 0.83)。S1=2、S2=0(官方未发布轨迹,HAL 亦无,须自产)、S3=1、S4=1、S5=1.5(8 位 PhD、3–4 篇、41.4%@48h)——总分不过入选线,凭 S1 保留为层级恢复的标定集。公开发布集 23 篇(主集 20 篇 ICML 2024 Spotlight/Oral;semantic-self-consistency 与 stay-on-topic 为 NeurIPS 2024 workshop 开发集);实跑取每篇外部数据 <10GB 且 Result Analysis 权重高的交集子集(构成见待办 8):

| 论文(短名) | 一句话 |
|---|---|
| adaptive-pruning | APT:自适应剪枝+微调,提升预训练 LM 训练与推理效率 |
| all-in-one | 一体化 simulation-based inference 框架 |
| bam | Batch-and-match:基于得分散度的黑盒变分推断 |
| bbox | BBox-Adapter:黑盒 LLM 的轻量适配 |
| bridging-data-gaps | 对抗噪声迁移学习弥合扩散模型数据缺口 |
| fre | 功能奖励编码的无监督零样本强化学习 |
| ftrl | 微调 RL 模型本质是遗忘缓解问题 |
| lbcs | 性能约束下最小核心集的精化 coreset 选择 |
| lca-on-the-line | 用类别分类树基准化 OOD 泛化 |
| mechanistic-understanding | DPO 与毒性:对齐算法的机制性理解 |
| pinn | PINN 训练难点的损失景观视角 |
| rice | 用解释机制打破强化学习训练瓶颈 |
| robust-clip | 无监督对抗微调获得鲁棒 CLIP 视觉嵌入 |
| sample-specific-masks | 视觉重编程提示的逐样本掩码 |
| sapg | SAPG:拆分-聚合策略梯度(大规模并行 RL) |
| self-composing-policies | 可自组合策略的可扩展持续强化学习 |
| self-expansion | 适配器混合的预训练模型自扩展持续学习 |
| semantic-self-consistency(dev) | 语义加权的自洽推理增强 |
| sequential-neural-score-estimation | 条件得分扩散模型的似然自由推断 |
| stay-on-topic(dev) | Classifier-Free Guidance 用于语言模型主题保持 |
| stochastic-interpolant | 数据相关耦合的随机插值生成模型 |
| test-time-model-adaptation | 仅前向传播的测试时模型自适应 |
| what-will-my-model-forget | 预测语言模型精调中将被遗忘的样本 |

**NatureBench [Optimize / D2] ✅(预算与轨迹待核)** —— 90 题蒸馏自 Nature 系论文,对标各论文已发表 SOTA;NatureGym 逐任务容器化;最强模型仅 17.8% 任务超 SOTA。任务以论文 ID 标识,逐题一句话说明存于独立 task package,待抽样时随包拉取补录(待办 7)。领域构成(90 题):

| 领域 | 题数 |
|---|---|
| Cellular Omics(细胞组学) | 31 |
| Protein Biology(蛋白质生物学) | 16 |
| Biomedical Modeling(生物医学建模) | 14 |
| Physical Modeling(物理建模) | 13 |
| Molecular Design(分子设计) | 11 |
| Relational Reasoning(关系推理) | 5 |

PoC 每领域抽 1–2 题共 ~8。

**Replica(Faraday 论文)[Replica / D0–D2] ❌ 已核实未开源(2026-08-24)→ 自建模板**:310 个 figure 级复现任务/100 篇论文——从 PDF 删掉一张结果图,智能体凭正文+图注、限定时间与算力、看不到原图复现它。核实:论文页无代码链接、官宣与官网未提发布、GitHub/HF 无踪迹;按闭源规划。替代:配方公开,在已选论文上自建,选材与 AIRS/NatureBench 论文源对齐,形成"同一论文的优化 vs 复现"配对。

**CORE-Bench Hard [Replica / D1–D2] ⚠**:270 任务/90 篇(CS、社科、医学),装依赖、跑通给定代码、答结果问题;层级来自复现步骤而非模型设计;作零成本 Track-Replica 轨迹来源,结论单列。

**ScienceAgentBench [Optimize / D2] ⚠**:102 任务/44 篇数据驱动发现;G2 弱、训练占比待核;仅作扩展域轨迹素材。

### 3.8 剔除与降级(典型例子)

| 对象 | 处理 | 原因 |
|---|---|---|
| LLM-Speedrunner(记录链复现) | 替换 | 改用 Prime Intellect 开放式 speedrun 设定(3.5):Track-Replica → Track-Optimize;原 R1–R20 多为架构/精度/系统改动,在 track 3 只许改 optimizer 的编辑面下大多越界;人类记录链仅保留作 ingredient 对照 |
| MLE-bench | 降级 | tabular 题 G1 边缘(典型:梯度提升一把梭的表格赛)、方案污染、成本高;AutoMind 公开的 Lite 15 题实验包可部分升 Tier A |
| ResearchCodeBench | 剔除 | component 补全无智能体闭环(典型:在给定仓库内补一段核心实现,跑通即得分) |
| ResearchClawBench | 剔除→待复核 | 原按域排除;域扩展后待调研 G1–G4 |

## 四、开源轨迹资产清单

| 资产 | 内容 | 入口与状态 |
|---|---|---|
| MALT(METR) | 公开 split 7,179 条 run / 169 任务 / 18 模型,4,426 条含推理轨迹;主要来自 HCAST 与 RE-Bench;带 reward-hacking / sandbagging 标注;轨迹为带 parent 指针的 DAG 节点 | HF: metr-evals/malt-transcripts-public,点击式 gated(在 HF 页面同意条款即可),default 配置 1.66 GB |
| RE-Bench 原始发布 | 71 次人类专家 attempt 全转录 + Claude 3.5 Sonnet / o1-preview 运行转录 | METR 官方发布 |
| PostTrainBench-Trajectories | 1,842 run / 62 个 agent 配置 / 约 33 种 agent+模型;完整 agent trace(各 CLI 原生 JSONL 逐行加时间戳)、评估结果、污染判定、反作弊裁判日志、系统监控日志(CPU/GPU/磁盘)、耗时、文本版工作区快照;实测 28.9 GB / 218,361 文件 | HF: aisa-group/PostTrainBench-Trajectories,Apache-2.0,不 gated |
| HAL(Princeton) | 26,597 条 rollouts / 9 基准,全部评测日志公开(加密防污染),逐 token 成本追踪;CORE-Bench Hard 与 ScienceAgentBench 官方榜 | hal.cs.princeton.edu + hal-decrypt |
| AutoMind 实验包 | MLE-Bench Lite 15 任务 × 3 种子完整日志、解法与中间结果(~33G) | Google Drive 公开链接(见其仓库) |
| PI frontier-automated-speedrun | 41 run / 18 模型 / 8 harness 完整轨迹(events:文本、thinking、tool call/result;subagents;scratchpad 含 thread.md 决策日志)+ manifest(逐 run 记录进程、agent_h、token、cost、工具计数、validity 标注)+ program.md + baseline 脚本;~50 MB;18 个记录 PR(baseline→记录 diff + 8-seed 验证);run.sh / verify.py 未随仓库公开但可从轨迹逐字恢复 | GitHub 公开,无需登记;无 LICENSE |
| PI experiments-autonomous-speedrunning | 早期波次(v1/novelty/v2/v3,Opus 4.7 vs GPT-5.5):plan.md、THREAD.md、runs.jsonl 台账、~1 万条 run log 与全部 variants | GitHub 公开 |
| AutoLab live-lab | 161 run / 23 题 / 7 模型,精简 episode schema | 无官方导出;打包在网站 JS chunk 内,可非官方批量拉取 |
| AIRS-Bench | — | 官方未发布任何轨迹(aira-dojo issue #12 未回复),须自产 |
| ML-Agent 专家轨迹 | 1 万条(MLAgentBench 4 题 + MLE-bench 5 题,每条 ≤15 步/30 分钟) | 发布状态待核实 |
| Replica | — | 已核实未开源(2026-08-24) |

## 五、GPU 预算估算

定义:一次迭代 = 选定子集 × 1 agent 配置 × 1 seed。AutoLab 取 task.toml 官方 timeout;AIRS 取官方协议;Speedrun 取 PI 协议(≥3 seed × ~24h 初筛,续跑最优至 ≤8 天;24h 墙钟 ≈ 90–150 次 1-trial 实验);≈ 推算,⚠ 粗估。

### 5.1 各组件单价

| 组件 | 硬件 | 单任务预算 | 任务数 | 一次迭代 GPU·h |
|---|---|---|---|---|
| Tier A 轨迹分析 | 无 GPU | 下载解析(~10–50 GB;PI Speedrun 50 MB) | 33+HAL+41 run | **0** |
| Speedrun PoC(24h 等预算切片) | 8×H100/H200 | 1 seed × 24h 墙钟(≈90–150 实验) | 1 | ≈192 |
| Speedrun 官方协议 | 8×H200 | 3 seed × 24h 初筛(≈576)+ 最优续跑 1–7 天(≈190–1,350) | 1 | ≈770–1,900 / 配置 |
| AIRS PoC 版(GPU-heavy 8 题) | 1×H200 | 24h(官方协议;单次训练 1–12h) | 8 | ≈192 |
| AutoLab 8 题 | 1×H100/L40S | solve 12/8/8/8/4/4h + 2 题 CPU;评测 ~12.5h | 8 | ≈56 |
| PostTrainBench 自跑(PoC:GSM8K × Qwen3-4B) | 1×H100 | 官方 10h/配置 | 1 | 10 |
| ⚠ PaperBench 10 篇 | 1 GPU + 评判 API | rollout 视论文;评分 ~$66/篇 | 10 | 待核 |
| ⚠ NatureBench 抽样 | 待核(容器化) | 时限待核 | 8 | 待核 |
| ⚠ 自建 figure 复现 | 1 GPU/任务 | figure 级,预期 < PaperBench 整篇 | 5–10 | 待核 |
| ⚠ MLS-Bench llm-pretrain | 2×H100 DDP | ~16–24 GPU·h/任务 | 11 | ≈180–260 |
| Backbone 推理(自托管) | 1×H100/H200 常驻 vLLM | 与墙钟同长;或 API 计费(PI 实测 Speedrun:claude-code 系 ≈ $250–350/24h,Fable 5 全程 8.7 天 $3,200) | — | 单列 |

### 5.2 预算方案

| 方案 | 内容 | GPU·h | 墙钟 |
|---|---|---|---|
| S0 纯分析 | Tier A 轨迹(含 HAL) | **0** | 数天(CPU) |
| S1 PoC 聚焦 10 题 | Speedrun 1 seed@24h + AIRS 8 题@24h + PostTrainBench 1 配置@10h | **≈394** | 2–3 天 |
| S2 标准一轮 | S1 加 seed:Speedrun 3 seed@24h + AIRS 8 题 × 3 seed + PostTrainBench 1 配置 × 3 次 | ≈1,180 | 3–5 天(3 节点并行;单节点则 Speedrun 串行 +2 天) |
| S2+ 扩展轮 | S2 + AutoLab 8 + NatureBench 8 + PaperBench 10 + 自建复现抽样 | ≈1,180 + 56 + ⚠ | 4–6 天 |
| S3 论文级 | 官方协议 × 3–10 seeds 全量(Speedrun 每 model×harness 配置 ≈770–1,900) | ≈5,000–6,500+ | 1–2 周 |

锚点:RPM 论文一次端到端评估(20 题 × 10 seeds × 24h × H200)= 4,800 GPU·h;S1 约为其 8%,S2 约为其 25%。AutoLab 两个 CPU 任务可作零 GPU 冒烟测试;Speedrun 先跑 1 seed × 2–4h(≈16–32 GPU·h)验证 run.sh / verify.py / 沙箱。

## 六、方法论参照案例:Faraday/Replica(arXiv 2608.13331)

1. 评审设计:逐任务自动生成 rubric(生成与评审分属不同模型)、每 rollout 判 3 次取平均、五维评分含"科学完整性"(显式抓 hard-coding 与伪造数据);验证为低噪声且与人类一致。
2. 双层架构:27B "scientist" 模型持有实验计划、选假设、读中间结果,指挥 5T 级 coding agent 执行(CAT 范式);turn-level "洞见时刻"加权提供步级信用分配。
3. Post-train 证据:提示工程无法弥合差距;GRPO 后训练的 27B 在留出复现任务上超过 Claude Opus 4.8 与 GPT-5.5(同分布 ML 73% 任务占优、留出科学域 60%)——支持"提示不行则后训练"路线,配方:GRPO + rubric 奖励 + 步级加权。

## 七、待办清单

1. ~~逐题核对 AIRS-Bench,落实 ⚠/暂定~~ 已核(2026-08-25):FinQA ❌(SOTA=GPT-4 提示)、DuoRC ✅(ALBERT 微调);PoC 收窄为 GPU-heavy 8 题,备选 6 题的 G1 复核(时序两题)仅在启用备选时需要。
2. AutoLab live-lab 轨迹:无官方导出,需要时从网站 JS chunk 批量拉取(首批不接入,暂缓)。
3. HF 完成 MALT 登记,拉 RE-Bench task family 转录,验证拆解结构可恢复。
4. 下载 PostTrainBench 1–2 条 run,写解析脚本(trace 与 system_monitor 时间轴对齐)。
5. 定义"层级恢复"第一版操作化指标(子目标 DAG vs ground-truth 图编辑距离)。
6. HAL 轨迹试点:下载 CORE-Bench Hard rollouts,跑通 hal-decrypt,评估格式兼容度。
7. NatureBench 核查:时限/硬件(G2)、许可证、公开运行数据;抽样 8 题并随 task package 补录逐题一句话说明。
8. PaperBench 子集核定:<10GB 与 Result Analysis 权重高(stay-on-topic 0.64、ftrl 0.53、sapg 0.48、rice / fre 0.47、pinn 0.44、bridging-data-gaps 0.43、lbcs 0.40)取交集;估 rollout GPU 与评判成本;轨迹须自产。
9. ResearchClawBench 域扩展后复核 G1–G4。
10. ML-Agent 1 万条专家轨迹发布状态核实。
11. 自建 figure 级复现任务:按 Replica 配方评估成本,选材与 AIRS/NatureBench 论文源对齐,先建 5–10 个。
12. 轨迹解析器增加双层拆分(科学家级决策 vs 编码级执行),在 MALT/PostTrainBench 现有 trace 上验证可分离性;PI Speedrun 轨迹的主 agent / 子 agent 拆分是天然双层,可作对照。
13. 拉取 frontier-automated-speedrun 41 run 轨迹:解析 manifest `progression`(记录进程 vs agent_h/token/cost)、events 主/子 agent 拆分、scratchpad/thread.md 决策日志;作为层级恢复(待办 5)与双层解析(待办 12)的首个试点。
14. 自建 Speedrun 复现环境:run.sh / verify.py 从轨迹恢复;verifier 须自建——PI 的 verify.py 只看单文件末行 val loss,不核种子与脚本一致性,改为按日志头部的脚本源码哈希分组、找齐 8 个 seed(0xC0FFEE+0..7、门槛 3.27859)判定;网络隔离(bwrap + 网络命名空间 + 只放行模型 API 的代理)在 Harbor 下如何做待验证;8×H100 节点先跑 1 trial baseline 复现 3290 步 ≈ 3.277,再上 24h 切片。
15. 仓库接入(见 `doc/spec/2026-08-24-repo-architecture.md`):PostTrainBench 原生 + 补丁分支,AIRS 经 Harbor adapter,PI 手写 Harbor task 目录;Phase 0 先在本机拉 PI 41 run 与 PostTrainBench 首批子集并统一成事件流。
16. AIRS 自跑镜像/缓存构建时剔除 `*squad*` 微调 checkpoint(SQuAD 缓存泄漏,见 3.4);Speedrun 换 8×H100 时先重跑 1 次 8-trial baseline 校准 σ。
