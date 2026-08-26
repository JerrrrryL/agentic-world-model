## 调研现有的 benchmark:

- Focus 在 Model Development, 体现出的问题就是说要在一个有限的资源下. 一个明确具体的任务上, 执行端到端发现. 要足够复杂涉及 代码实现 / 训练 / 资源调度. 
- 有非常多不同种的Research Benchmark, 覆盖面非常广, 扩展到基础科学. 例如说 Nature Bench 和 Research Claw Bench. 这些会偏离我们Model Development的初衷. Focus on ML tasks, instead of AI4Science.

## 核心卖点

- 我们不考虑 from scratch 建立一个benchmark, 而是建立一套评估方法论, 为benchmark补充评估维度. 衡量模型在Model Devlopment 中如何拆解任务 (breakdown task and planning), 构造子目标进行优化, 如何建模子目标与最终目标之间的层次结构关联.
- 要想查到有意义的提升, 就需要去限定条件, 筛选task样本, 观察多样的指标
- 可以用分析的方式, 先去大规模跑benchmark, 然后分析模型究竟是在依靠什么地方来去优化. 然后分析什么时候work, 什么时候不work, 找到哪些和human preference 不align的地方
- telemetry 是核心重点, 我们观测到Agent system的成功, 以及 agent 观察到 模型开发的成功 是等价的
- 研究的层级是 Scientific Judgement above coding agents
- 如果单纯使用 prompt 无法让 模型遵循 hierarchy goal, 就需要Post-Train

## Benchmark 选择:

1. MLS Bench (component-level) ✅
2. [OpenAI] PaperBench ✅
3. [OpenAI] MLE-Bench (kaggle, 偏ML应用)
4. PostTrainBench (recipe/data) ✅
5. [Meta] AIRS-Bench (no-baseline) ✅
6. LLM-Speedrunner (model training & impl) ✅
7. AutoLab (includes Model Dev category) ✅
8. ResearchCodeBench 
9. ResearchClawBench

## Proof of Concept

- 定义清楚 Model Development Research Task, 从当前开源benchmark中筛选一个子集.
- 尝试确定清楚一个干净的执行setup, 可以自己复现, 也可以去获取对应的trajectory, 分析模型Auto Research的中间状态
- 还是先从systemetic evaluation入手, 想一些比较novel又实际的评估方式 (构成一篇benchmark), 然后可以考虑变成一些harness设计, 或者模型训练recipe

