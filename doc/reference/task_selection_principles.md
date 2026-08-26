# Model Development 研究任务:评估原则

**用途**:长期沉淀的判断原则,用于判断一个任务是否属于 Model Development 研究任务、其轨迹是否值得分析、实验设计要守住哪些底线。具体的 benchmark 筛选结果、轨迹资产与预算数据见配套文档《Model Development benchmark 案例与数据清单》。

---

## 一、核心定义

Model Development 研究任务 = 在显式资源约束下,围绕"训练一个模型"展开的端到端发现过程:智能体自主拆解任务、构造子目标、分配实验预算,并对"子目标 ↔ 最终目标"的层次关联负责。三要素缺一不可:代码实现、训练、资源调度。

## 二、目标形态:双轨(Track-Optimize / Track-Replica)

**Track-Optimize 开放优化**:目标是把一个连续指标推得尽可能高(对标 SOTA 或无上界),没有标准答案。例:AIRS-Bench、AutoLab、PostTrainBench、RE-Bench、NanoGPT Speedrun(Prime Intellect 设定:从 baseline 起压 train_steps,人类记录只作 headroom 参照)。

**Track-Replica 复现**:目标是匹配一个已发表的结果。复现同样需要假设驱动的探索(论文总是欠规范、细节缺失),且天然自带 ground-truth 目标分解——这是层级分析最稀缺的标注。例:PaperBench(作者共建 rubric 树)、自建 figure 级复现(原图即 ground truth)。

**实质差异:实验在两轨里扮演的角色不同。** Track-Optimize 的实验是**搜索**——每次实验检验一个假设,结果决定下一步往哪走;Track-Replica 的实验是**验收**——论文已经规定了要实现什么、跑什么、得到什么数字,实验只回答"对不对",不对就回去 debug。两轨都真的跑实验(G1 不分轨),但只有前者的实验序列本身携带"研究该往哪走"的决策;后者的决策主要是工程排期。

| | Track-Optimize(开放优化) | Track-Replica(复现) |
|---|---|---|
| 目标 | 把指标推高,没有标准答案 | 匹配已知的数字 / 图表 |
| "该做什么"由谁定 | 智能体自己:提假设 → 跑 → 看分 → 定下一个方向 | 论文已规定:实现 X、跑 Y、得到表 Z |
| 一次实验的意义 | 一个假设的检验,结果改变搜索方向 | 一次复现,结果只是对 / 不对,不对则 debug |
| 分数从哪来 | 100% 来自跑出来的指标,不跑就是 0 | 部分来自"代码存在"(PaperBench 约 53% 权重),其余来自跑通与对数;实测智能体后两类得分 ≈0 |
| 层级的性质 | **假设层级**:先试哪个方向、哪条线继续深挖、预算怎么分 | **工程分解层级**:论文由哪些部分组成、先搭哪块、实现到哪了 |
| ground truth | 没有"正确分解",只有分数与人类 / leaderboard 参照 | 作者写好的 rubric 树 / 原图 = 正确分解 |
| 资源调度的含义 | 真决策:预算花在哪个假设上 | 主要是工程排期:先实现哪块、reproduce 能否在时限内跑完 |
| 对层级分析的价值 | **主信号源**(科学家级决策,P5) | **标定集**:校准"层级恢复"指标本身(智能体自建分解 vs 作者 rubric) |
| 典型例子 | AIRS-Bench、Speedrun(PI)、PostTrainBench、RE-Bench、AutoLab | PaperBench、自建 figure 级复现 |

**当前阶段先聚焦 Track-Optimize。** 理由:(1) 我们要观测的"子目标 ↔ 最终目标的层次关联 + 资源分配"只在假设层级里完整出现;(2) 分数 100% 由实验产生,G1 无歧义;(3) 公开轨迹充足(Speedrun 41 run、MALT、PostTrainBench),可零成本起步。Track-Replica 保留为标定集:PaperBench 是唯一带人类标注分解树的候选,用来检验"智能体自建分解 vs 作者 rubric"的恢复指标,不作主评测集。

## 三、四条门槛(全过才算 Model Development 研究任务)

**G1 训练/实验在环(统一,不分轨)**
一句话:不真正跑训练/实验就拿不到分。
定义:达成得分(或复现)的主要路径必须实际执行训练或实验;纯 prompting、纯推理管线搭建、纯 kernel/系统优化、只写代码不执行的均不算。
判据:核对该任务 reference solution 或已发表 SOTA 路径中是否存在真实训练执行(optimizer.step() 级操作)。必须逐任务核对,不能按 benchmark 整体判断。
例:同在 AIRS-Bench,SVAMP 的 SOTA 是少样本提示+自洽采样(不过关),WinoGrande 的 SOTA 是 T5-3B 全量微调(过关,依据其 `metadata.yaml` 的 `sota_notes`);PaperBench 的 Code-Dev 变体只写代码不跑实验(不过关),完整版须跑通复现(过关)。

**G2 显式资源预算**
一句话:时间与算力稀缺,花在哪是真决策。
定义:任务规范写明墙钟或 GPU 配额,超时影响得分。没有预算约束,"资源调度"这一层就观测不到。
例:PostTrainBench = 单 H100 × 10 小时;AutoLab 每题在 task.toml 中写死 timeout 与 GPU 数;Speedrun(PI)= 8×H200 节点,报告口径同时给 agent-hours / 实验数 / output tokens 三种预算。

**G3 目标结构(分轨;这是两轨的真正分野)**
Track-Optimize:连续可优化指标 + 明确 headroom——分数是刻度尺,不是及格线。例:AIRS 的归一化分数、AutoLab 的困惑度/PSNR、Speedrun 的 train_steps(baseline 3,290 → 人类 2,600)。
Track-Replica:连续的复现完成度 + 附带 ground-truth 目标结构(论文、图表、人类记录史)。例:PaperBench 的 rubric 满足率(须按 Code Development / Execution / Result Analysis 三类拆开报,只用完整版)。
共同排除:pass/fail 型任务——看不到增量式子目标优化的过程。

**G4 端到端自主权**
一句话:智能体当项目负责人,不是填空。
定义:自行设计验证信号(如交叉验证)、决定实验顺序、选择提交物;受限编辑面的 component 补全不满足。
例:MLS-Bench llm-pretrain 族冻结 harness、限制编辑面,不过关——但正因它只违反这一条,被保留为负对照(见 P2)。

## 四、域分层原则(AI for Science 的处理方式)

域是范围决策(随项目阶段变),分层是方法论(不变)。任务域可以从纯 ML 扩到 AI for Science,但领域知识是混淆变量:智能体在基因组学任务上失败,分不清是层级规划能力不行,还是缺领域知识。

规则:每个任务标注领域知识依赖度——D0 纯 ML 方法域 / D1 常识级领域背景 / D2 专业科学知识;跨域结论必须按 D 分层报告;扩展域任务仍须通过同一套 G1–G4。

例:AutoLab 的 scaling_law = D0;AIRS 的 QM9 分子性质预测 = D1;NatureBench = D2。

## 五、分析与实验设计原则

**P1 轨迹优先**:方法论验证(层级恢复、telemetry 指标)先在零成本的公开轨迹上完成,GPU 只花在必须自产搜索结构的验证上。例:MALT、PostTrainBench-Trajectories、HAL 轨迹均可直接下载分析。

**P2 负对照常设**:保留"只违反 G4"的受限编辑面任务作为负对照,检验层级评估指标在自由度被人为剥夺时是否如预期退化。例:MLS-Bench llm-pretrain 族。

**P3 hack 样本不丢弃**:评分易被钻空子的任务不剔除,单独标为观察组——hack 行为本身是 human-preference misalignment 分析的一手素材。例:PostTrainBench 观察到的在测试集上训练、直接下载现成 instruct 权重。

**P4 预算口径入结论**:裁剪预算会改变智能体的探索行为;任何对比结论必须注明预算口径,并保留少量任务以官方全预算运行做校准。例:Speedrun 同时报告 @24h 等预算切片与最终记录两个口径。

**P5 双层解析**:轨迹解析须区分"科学家级决策"(实验计划、假设选择、中间结果解读)与"编码级执行";层级信号主要存在于前者。参照:Faraday 的 CAT 范式(scientist 模型指挥 coding agent)与其 turn-level 洞见加权。

**P6 评审的次优替代**:无客观指标时,结构化 rubric-judge——逐任务自动生成 rubric、每次评估多采样取平均、显式包含"科学完整性"维度(抓 hard-coding 结果、伪造数据)——可作为次优替代,已被验证可做到低噪声且与人类评估一致。
