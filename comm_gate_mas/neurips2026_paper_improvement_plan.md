# NeurIPS 2026 论文五天提升方案：Agent Auto Research 场景下的 Message Pruning 强化计划

## 0. 总目标

当前论文的核心发现是：在 two-agent LLM solver-checker 系统中，acknowledgment 消息虽然在单条删除时最 disruptive，但在整体剪枝时可以安全删除，并能减少大量 LLM inference calls。

五天内的目标不是重写一篇新论文，而是用 2–4 张 RTX 5090，在最短时间内补上最能提升 NeurIPS 评分的证据，把论文从：

> acknowledgment pruning 在 GSM8K / StrategyQA two-agent 系统中有效

提升为：

> LLM agent communication 中存在一种通用的分析误区：局部扰动敏感性会高估某些结构性冗余消息的全局必要性。本文提出 functional message analysis protocol，并在 controlled reasoning system 与真实 Auto Research agent workflow 中验证该现象、机制和实际成本收益。

预期提升方向：

- **Quality**：通过因果干预和 matched control 补强机制证据。
- **Significance**：通过 Auto Research agent workflow 证明不是 toy benchmark 现象。
- **Practicality**：通过 wall-clock latency / GPU cost profiling 证明不是只省 call count。
- **Robustness**：通过 learned detector 的跨域 online validation 减少 oracle / heuristic 依赖。
- **Clarity**：通过重构叙事，让论文从 “ack pruning trick” 变成 “agent communication analysis principle”。

---

## 1. 当前论文的主要短板

### 1.1 任务场景偏 toy

当前主结果主要来自 GSM8K、StrategyQA、ARC 等标准 reasoning benchmarks。虽然实验量不少，但 reviewer 很可能认为它仍然是一个 two-agent solver-checker toy phenomenon，而不是对真实 agent 系统有普遍意义的发现。

需要补：

- 多 agent workflow；
- 更真实的任务；
- research / retrieval / synthesis / critique 类型任务；
- 与当前 LLM agent / AI scientist / Auto Research 方向相关的场景。

### 1.2 核心机制仍偏相关性

论文提出 cascade dissolution：acknowledgment 会形成自强化 cascade，整体剪枝会让这些 cascade 消失而非累积错误。

但目前证据主要是：

- P(ack | previous ack) 较高；
- position-controlled logistic regression 显示 previous ack 提高后续 ack 概率；
- dose-response curve 较平；
- ack 信息量低。

这些是相关性证据，不是严格因果证明。

需要补：

- ack injection；
- ack chain breaking；
- position-matched non-ack pruning；
- length-matched non-ack pruning。

### 1.3 实际效率收益不够清楚

当前论文强调减少的是 inference calls，而不是 token-level compression。但 reviewer 会问：

- 真实 wall-clock latency 降了多少？
- GPU seconds 降了多少？
- batching 下是否仍然省？
- acknowledgment 很短，token savings 是否很小？
- 每条 message 是否真的对应一次完整 LLM invocation？

需要补：

- end-to-end latency；
- request-level latency；
- total input/output tokens；
- GPU time proxy；
- throughput under batching；
- call count 与 token count 分开报告。

### 1.4 5-class taxonomy 的完整贡献不够稳

当前最强证据其实是 ack vs non-ack。Fine-grained 5-class taxonomy 中部分标签一致性较弱，尤其是 information_provision 这类边界模糊的标签。

建议写作上收敛：

- 主结论聚焦 ack / low-content flow-control；
- 5-class taxonomy 作为 analysis protocol，而不是硬说每一类都可靠；
- 对 fine-grained labels 的结论降级为 exploratory。

### 1.5 Detector 的 deployment claim 偏强

当前 oracle 结果较强，heuristic 有一些结果，但 heuristic non-transferability 明显；DistilBERT online validation 只在单个 configuration 上做了。

需要补：

- TF-IDF + Logistic Regression online validation；
- DistilBERT / TF-IDF 在 Auto Research、StrategyQA、Llama 上的小规模验证；
- leave-one-domain / leave-one-backbone generalization。

---

## 2. 改进后的论文主叙事

### 2.1 原始叙事

> In two-agent solver-checker systems, acknowledgment messages are individually disruptive but aggregately prunable.

这个叙事有意思，但容易被认为只是一个小现象。

### 2.2 推荐叙事

> We identify a general failure mode in LLM agent communication analysis: local perturbation sensitivity can overestimate the global necessity of structurally redundant messages. Acknowledgment messages are the canonical case. We show that they are locally disruptive but globally prunable in controlled solver-checker systems and in a realistic Auto Research agent workflow. We further provide causal interventions, matched controls, measured efficiency gains, and practical learned detectors.

### 2.3 新 contribution 写法

建议改成四个贡献：

1. **Perturbation-pruning dissociation**  
   发现局部扰动敏感性和整体剪枝安全性之间的 dissociation。

2. **Functional message analysis protocol**  
   提出 taxonomy、annotation、single-message perturbation、aggregate pruning、non-inferiority / equivalence testing 的统一 protocol。

3. **Causal and controlled evidence**  
   通过 ack injection、chain breaking、position/length-matched non-ack controls 验证 cascade dissolution 机制，并排除 position / length / random shortening 等 confound。

4. **Practical validation in Auto Research agents**  
   在 retrieval-synthesis-critique 类型的 Auto Research workflow 中证明 ack / low-content flow-control pruning 能减少 LLM calls 和 wall-clock latency，同时不降低 judge-rated quality 和 citation grounding。

---

## 3. 五天内最重要的新增实验

优先级从高到低：

1. **Auto Research 3-agent workflow**
2. **真实 latency / GPU cost profiling**
3. **Ack injection causal intervention**
4. **Position / length-matched non-ack control**
5. **Learned detector 跨域 online validation**
6. **小规模 4-agent topology probe**

---

# 4. 实验一：Auto Research Agent Workflow

## 4.1 目的

证明该现象不是 GSM8K / StrategyQA 这种标准 QA benchmark 中的 toy phenomenon，而是出现在更真实的 LLM agent workflow 中。

核心 reviewer concern：

> This is interesting, but does it matter for real agent systems?

新增实验回答：

> Yes. In a three-agent Auto Research workflow involving retrieval, synthesis, and critique, acknowledgment / low-content flow-control pruning reduces LLM invocations and wall-clock latency without degrading answer quality or citation grounding.

---

## 4.2 Benchmark 名称

可以命名为：

- **AutoResearch-CommPrune**
- **ResearchAgent-CommPrune**
- **AutoResearch Message Pruning Benchmark**

建议主文使用：

> AutoResearch-CommPrune

---

## 4.3 任务形式

给定一个 research question，让 agent 从本地论文库中检索、分析、综合，并输出一个 grounded research answer。

每个任务包括：

- research question；
- local document corpus；
- retrieved evidence；
- multi-agent discussion trace；
- final answer；
- automatic / LLM-judge evaluation。

---

## 4.4 任务类型

建议构造 100 个任务，覆盖以下类型：

### A. Literature Comparison

示例：

> Compare message-level pruning and token-level compression methods for multi-agent LLM systems. Identify their pruning granularity, evaluation settings, and limitations.

### B. Evidence Extraction

示例：

> Find which papers report token-level compression rather than message-level pruning, and summarize their main evaluation metrics.

### C. Paper Review Critique

示例：

> Given the retrieved papers, write a NeurIPS-style critique of a claim that single-message perturbation sensitivity predicts message importance.

### D. Method Design

示例：

> Design an ablation plan to distinguish whether acknowledgment pruning works because of cascade dissolution or because of random conversation shortening.

### E. Citation-Grounded QA

示例：

> Answer the question using only evidence from the provided papers and cite supporting documents.

---

## 4.5 文档库构造

五天内不要做复杂 web browsing。建议用固定本地 corpus：

- 30–80 篇 LLM agents / multi-agent communication / token compression / agent pruning / AI scientist 相关论文；
- 每篇切成 chunks；
- 建立 BM25 或 embedding retrieval；
- 每个 task top-k 检索 evidence。

推荐语料：

- AutoGen；
- CAMEL；
- MetaGPT；
- AgentVerse；
- AgentDropout；
- multi-agent debate；
- token compression；
- context compression；
- message pruning；
- AI Scientist / auto research；
- tool-use agent benchmark；
- communication protocol papers。

---

## 4.6 Agent Workflow 设计

主实验用 3-agent workflow，简单、稳定、可控。

### 3-agent workflow

1. **Retriever Agent**
   - 输入 research question；
   - 检索 top-k evidence chunks；
   - 返回 evidence summary。

2. **Synthesizer Agent**
   - 阅读 evidence；
   - 生成初版 answer；
   - 标注引用来源。

3. **Critic Agent**
   - 检查 factuality、coverage、citation grounding、logical gaps；
   - 要求修改或确认可接受。

然后 Synthesizer 产出 final answer。

### 推荐交互流程

```text
User question
→ Retriever: search plan + retrieved evidence
→ Synthesizer: draft answer
→ Critic: critique / verification
→ Synthesizer: revised answer
→ Critic: final verification or remaining concerns
→ Synthesizer: final answer
```

这样自然会产生 acknowledgment 和 low-content flow-control messages。

---

## 4.7 Pruning Targets

### Target 1: ACK-only

严格 acknowledgment：

- “I agree.”
- “Correct.”
- “This looks good.”
- “The answer is correct.”
- “No issue found.”

这是和原论文主线一致的 target。

### Target 2: ACK+FLOW

更宽的 low-content flow-control messages：

- “I will now proceed.”
- “Moving to the next step.”
- “Let’s continue.”
- “The retrieved documents seem relevant.”
- “I will summarize now.”

注意：ACK+FLOW 应该标成 exploratory，不要让主结论依赖它。

推荐写法：

> For Auto Research workflows, we evaluate strict acknowledgment pruning as the primary intervention and a broader low-content flow-control pruning policy as an exploratory extension.

---

## 4.8 Conditions

主实验条件：

1. Baseline
2. Oracle ACK pruning
3. Learned ACK pruning
4. ACK+FLOW pruning, exploratory

如果时间紧，至少完成：

1. Baseline
2. Oracle ACK pruning
3. Heuristic or TF-IDF ACK pruning

---

## 4.9 评价指标

### 4.9.1 Quality Score

用 LLM judge 打分，维度：

- factual correctness；
- coverage；
- citation grounding；
- reasoning coherence；
- usefulness。

每项 1–5 分。

### 4.9.2 Pairwise Preference

让 judge 比较 baseline vs pruned：

- A wins；
- B wins；
- tie。

必须随机交换 A/B 顺序，防止 position bias。

### 4.9.3 Citation Grounding

自动或 judge 检查：

- 引用是否来自 retrieved evidence；
- claim 是否有 evidence support；
- hallucinated citation rate；
- unsupported claim rate。

### 4.9.4 Cost Metrics

必须记录：

- number of LLM calls；
- input tokens；
- output tokens；
- wall-clock latency；
- request-level latency；
- GPU time proxy；
- throughput under batching, optional。

---

## 4.10 推荐结果表

```markdown
| Workflow | Model | Pruning | N | Quality Δ | Pairwise Win/Tie/Loss | Citation Error Δ | Calls ↓ | Output Tokens ↓ | Wall-clock ↓ |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|
| AutoResearch-3Agent | Qwen2.5-7B | ACK oracle | 100 | +0.02 | 18/67/15 | +0.1% | 35% | 12% | 24% |
| AutoResearch-3Agent | Qwen2.5-7B | ACK learned | 100 | -0.03 | 16/69/15 | +0.2% | 31% | 10% | 21% |
| AutoResearch-3Agent | Qwen2.5-7B | ACK+FLOW | 100 | -0.08 | 15/62/23 | +0.5% | 48% | 19% | 33% |
| AutoResearch-3Agent | Llama-3.1-8B | ACK oracle | 50 | +0.04 | 11/31/8 | -0.1% | 28% | 15% | 19% |
```

表中数值是期望格式，不是实际结果。

---

## 4.11 可接受结果标准

理想结果：

- Calls reduction ≥ 25%；
- Wall-clock latency reduction ≥ 15%；
- Pairwise judge 中 tie + pruned win ≥ 70%；
- Quality Δ 在 ±0.1 / 5 内；
- Citation hallucination 不显著增加。

如果 ACK+FLOW 有轻微质量下降，也可以作为 boundary condition。

主结论只要 ACK-only 安全即可。

---

# 5. 实验二：真实 Cost / Latency Profiling

## 5.1 目的

回应 reviewer 对 “36–56% inference-call reduction 是否等价于真实成本下降” 的质疑。

需要把论文从：

> We reduce inference calls.

升级为：

> We reduce LLM invocations, output tokens, and measured wall-clock latency in a real agent workflow, while noting that token savings are smaller because acknowledgments are short.

---

## 5.2 Logging 字段

每个 task / condition 记录：

```json
{
  "task_id": "...",
  "condition": "baseline",
  "model": "Qwen2.5-7B-Instruct",
  "num_calls": 32,
  "num_suppressed_messages": 11,
  "input_tokens": 18423,
  "output_tokens": 3210,
  "wall_clock_sec": 74.2,
  "request_latency_sec_mean": 2.3,
  "request_latency_sec_p50": 1.8,
  "request_latency_sec_p95": 4.9,
  "gpu_time_proxy_sec": 51.8,
  "final_quality_score": 4.1
}
```

如果拿不到真实 GPU kernel time，用以下 proxy：

- sum of request latencies；
- decode tokens / tokens per second；
- vLLM reported latency；
- wall-clock under fixed concurrency。

---

## 5.3 实验设置

同一组 AutoResearch 100 tasks：

1. Baseline
2. ACK oracle pruning
3. ACK learned pruning
4. ACK+FLOW pruning, optional

固定：

- same tasks；
- same retrieval results；
- same model；
- same max turns；
- same decoding setting；
- same hardware；
- same batching / concurrency。

---

## 5.4 主文写法

不要写：

> pruning reduces cost by 56%.

建议写：

> Since acknowledgments are short, token-level savings are smaller than call-level savings. However, in orchestration regimes where each message triggers a separate LLM invocation, acknowledgment pruning reduces measured wall-clock latency by X% and LLM invocations by Y%. Token-level compression methods are complementary.

这样更诚实，也更不容易被 reviewer 攻击。

---

# 6. 实验三：Ack Injection 因果干预

## 6.1 目的

当前 cascade dissolution 主要是相关性证据。Ack injection 是五天内最强、最简单的因果实验。

核心问题：

> Does inserting an acknowledgment causally increase the probability of future acknowledgments?

---

## 6.2 实验设计

从已有 baseline traces 中采样 non-ack 位置，在这些位置插入不同类型消息，然后从插入点之后 regenerate。

条件：

1. Original trace
2. ACK injection
3. Neutral injection
4. Flow-control injection
5. Content-bearing short control, optional

### ACK injection examples

```text
Correct.
I agree.
That looks right.
Yes, proceed.
The previous step is valid.
```

### Neutral injection examples

```text
[Step recorded.]
Continue.
Message received.
```

### Flow-control examples

```text
Proceed to the next step.
Continue with the calculation.
Move on to verification.
```

---

## 6.3 指标

核心指标：

- future ack rate in next k messages；
- average ack chain length；
- probability of next message being ack；
- total number of generated messages；
- final accuracy / final judge quality；
- trajectory similarity。

---

## 6.4 推荐结果表

```markdown
| Intervention | N positions | P(next ack) | Ack rate next-3 | Chain length | Accuracy Δ |
|---|---:|---:|---:|---:|---:|
| Original | 500 | 0.31 | 0.28 | 1.4 | 0.0 |
| Neutral injection | 500 | 0.34 | 0.30 | 1.5 | -0.1 |
| ACK injection | 500 | 0.58 | 0.51 | 2.3 | +0.0 |
| Flow-control injection | 500 | 0.40 | 0.36 | 1.7 | -0.2 |
```

表中数值是期望格式，不是实际结果。

---

## 6.5 主文写法

如果结果显著，可以写：

> Controlled insertion of acknowledgment messages causally increased the probability of subsequent acknowledgments, supporting the cascade component of our explanation. This intervention rules out a purely correlational interpretation based only on message position or dialogue phase.

如果结果不显著，也可以写成 limitation：

> Ack injection did not significantly increase future ack probability outside naturally occurring ack-rich phases, suggesting that cascade dynamics depend on both message function and dialogue state.

但理想情况下应该先在 Qwen/GSM8K 上做，因为最可能成功。

---

# 7. 实验四：Position / Length-Matched Non-Ack Control

## 7.1 目的

排除以下 confounds：

- ack pruning 安全只是因为 ack 出现在不重要的位置；
- ack pruning 安全只是因为 ack 很短；
- ack pruning 安全只是因为删少量消息或缩短对话；
- ack pruning 和 random pruning 没区别。

---

## 7.2 实验条件

比较四种 pruning：

1. ACK pruning
2. Random same-count pruning
3. Position-matched non-ack pruning
4. Length-matched non-ack pruning
5. Speaker/round-matched non-ack pruning, optional

---

## 7.3 Matching 方法

对每条 ack message，找一个 non-ack control：

- same speaker；
- same round or closest round；
- similar character length or token length；
- not labeled as acknowledgment；
- if possible, same topology / same task type。

如果找不到 perfect match，就用 nearest neighbor matching。

---

## 7.4 指标

- accuracy delta；
- final answer flip rate；
- call reduction；
- future ack rate；
- trajectory similarity；
- judge quality for AutoResearch, optional。

---

## 7.5 推荐结果表

```markdown
| Pruning Target | N tasks | Suppression Rate | Accuracy Δ | Correct→Wrong Flip | Wrong→Correct Flip |
|---|---:|---:|---:|---:|---:|
| ACK | 500 | 56% | +0.4 pp | 1.0% | 1.4% |
| Random same-count | 500 | 56% | -4.8 pp | 6.1% | 1.3% |
| Position-matched non-ack | 500 | 56% | -3.6 pp | 4.9% | 1.3% |
| Length-matched non-ack | 500 | 56% | -2.8 pp | 4.0% | 1.2% |
```

表中数值是期望格式，不是实际结果。

---

## 7.6 主文写法

理想 claim：

> Ack pruning remains safe while position- and length-matched non-ack pruning degrades accuracy, indicating that pruning safety is not explained by message length, position, or conversation shortening alone.

这是非常能提高 reviewer 信心的实验。

---

# 8. 实验五：Learned Detector 跨域 Online Validation

## 8.1 目的

减少 reviewer 对 oracle / heuristic 的质疑。

当前问题：

- oracle detector 依赖 GPT-4o；
- heuristic detector 不迁移；
- DistilBERT online validation 只在单个 configuration。

目标：

> Show that lightweight learned detectors can replace hand-crafted heuristics in multiple settings.

---

## 8.2 Detector 类型

建议训练两个：

### A. TF-IDF + Logistic Regression

优点：

- 训练快；
- CPU 推理极快；
- 可解释；
- deployment 更 practical；
- 五天内很稳。

### B. DistilBERT

优点：

- 看起来更 neural；
- 和当前论文已有结果衔接；
- F1 可能略高。

如果只能选一个，优先 TF-IDF + Logistic Regression，因为稳定且快。

---

## 8.3 数据

训练集包括：

- 原 GSM8K traces；
- StrategyQA traces；
- ARC traces；
- AutoResearch traces。

标签：

- oracle ack/non-ack；
- 或 human-validated subset。

Split 必须按 task_id split，避免同一问题的消息泄漏。

---

## 8.4 Generalization 设置

推荐至少做两个：

### Setting 1: Cross-domain

Train: GSM8K + StrategyQA  
Test: AutoResearch

### Setting 2: Cross-backbone

Train: Qwen  
Test: Llama

### Setting 3: Mixed train, held-out task

Train: 70% tasks  
Validation: 20% tasks  
Test: 10% tasks

---

## 8.5 Online Validation

不要只报 classifier F1。必须 online pruning。

推荐表：

```markdown
| Model | Domain | Detector | N | F1 | Suppression | Accuracy / Quality Δ | Calls ↓ |
|---|---|---|---:|---:|---:|---:|---:|
| Qwen2.5-7B | GSM8K | TF-IDF | 300 | 0.86 | 43% | +0.3 pp | 41% |
| Qwen2.5-7B | StrategyQA | TF-IDF | 200 | 0.78 | 39% | -0.2 pp | 37% |
| Llama-3.1-8B | GSM8K | TF-IDF | 200 | 0.80 | 35% | +0.8 pp | 33% |
| Qwen2.5-7B | AutoResearch | TF-IDF | 100 | 0.82 | 31% | -0.03 / 5 | 29% |
```

表中数值是期望格式，不是实际结果。

---

## 8.6 主文写法

建议写：

> Learned detectors reduce dependence on brittle hand-crafted heuristics. While universal deployment still requires empirical screening, a lightweight TF-IDF classifier achieves online pruning behavior close to oracle labels across reasoning and Auto Research settings with negligible CPU overhead.

不要写：

> solved deployment.

要写：

> bridges part of the oracle-to-deployment gap.

---

# 9. 实验六：小规模 4-Agent Topology Probe

## 9.1 目的

当前论文主要是 two-agent topology。AutoResearch 3-agent 已经补了一部分，但如果资源允许，可以再补一个 4-agent 小规模 probe。

目的不是证明 universal，而是说明：

> pruning opportunity depends on topology-induced acknowledgment prevalence.

---

## 9.2 4-Agent Workflow

1. Planner
2. Retriever
3. Writer / Synthesizer
4. Critic / Verifier

流程：

```text
Planner: decomposes research question
Retriever: collects evidence
Writer: drafts answer
Critic: critiques answer
Writer: revises answer
Critic: verifies final answer
```

---

## 9.3 实验规模

- N = 30 tasks；
- Qwen2.5-7B；
- baseline vs ack pruning；
- report quality + calls + latency。

---

## 9.4 推荐表

```markdown
| Topology | Task | N | Ack Rate | Quality / Accuracy Δ | Calls ↓ | Wall-clock ↓ |
|---|---|---:|---:|---:|---:|---:|
| 2-agent solver-checker | GSM8K | 500 | high | +0.4 pp | 56% | — |
| 3-agent research | AutoResearch | 100 | medium | -0.02 / 5 | 35% | 24% |
| 4-agent research | AutoResearch | 30 | medium-high | -0.05 / 5 | 41% | 27% |
| debate-like | QA | 100 | low | N/A | low | low |
```

---

## 9.5 主文写法

> Ack pruning is not universally useful; it is useful in acknowledgment-rich feedback topologies. Topologies with near-zero acknowledgment prevalence naturally provide little pruning opportunity.

这会把 limitation 变成适用条件。

---

# 10. 五天执行计划

## Day 0：冻结实验矩阵与代码接口

目标：防止发散。

当天必须确定：

- AutoResearch task set；
- document corpus；
- agent workflow；
- pruning targets；
- logging schema；
- judge prompt；
- experiment matrix。

### 当天冻结的最小实验矩阵

| Experiment | N | Model | Conditions |
|---|---:|---|---|
| AutoResearch 3-agent | 100 | Qwen2.5-7B | baseline / ACK / ACK+FLOW |
| AutoResearch 3-agent | 50 | Llama-3.1-8B | baseline / ACK |
| Cost profiling | 100 | Qwen2.5-7B | baseline / ACK / ACK+FLOW |
| Ack injection | 500 positions | Qwen2.5-7B | original / ACK inject / neutral inject |
| Position-matched pruning | 300–500 tasks | Qwen2.5-7B | ACK / random / matched non-ack |
| Learned detector online | 100–300 tasks | Qwen + Llama | TF-IDF / DistilBERT |

### 当天产出

- `tasks.jsonl`
- `corpus/`
- `agent_workflow.py`
- `pruning_policy.py`
- `logging_schema.json`
- `judge_prompt.txt`
- `run_matrix.yaml`

---

## Day 1：搭建 AutoResearch Benchmark 和 Logging

### 上午

构造 100 个 research questions。

方法：

1. 先用 LLM 生成 150 个；
2. 人工筛到 100 个；
3. 保证每个问题能从 local corpus 中找到 evidence；
4. 每个问题分配 task type。

### 下午

实现 3-agent workflow。

必须确保：

- 每条 message 都记录 speaker、round、content、tokens、latency；
- 每次 LLM call 都记录开始/结束时间；
- pruning policy 可以在线拦截 message；
- final answer 单独保存。

### 晚上

跑 10-task smoke test。

检查：

- 是否有足够 ack / flow messages；
- 是否出现无限循环；
- final answer 是否格式稳定；
- judge 是否能解析；
- latency logging 是否正确。

---

## Day 2：跑 AutoResearch 主实验 + Judge

### 白天

运行：

- Qwen2.5-7B，100 tasks；
- baseline / ACK pruning / ACK+FLOW pruning；
- 如果资源允许，Llama-3.1-8B，50 tasks。

### 晚上

运行 judge：

- scalar scores；
- pairwise preference；
- citation grounding；
- unsupported claim check。

### 当天产出

- AutoResearch 主结果表；
- cost profiling 表；
- judge win/tie/loss 表；
- quality distribution plot；
- latency distribution plot。

---

## Day 3：因果实验 + Matched Controls

### 上午：Ack Injection

运行：

- Qwen/GSM8K 或 Qwen/AutoResearch traces；
- 500 sampled positions；
- original / ACK injection / neutral injection。

输出：

- P(next ack)；
- next-3 ack rate；
- chain length；
- accuracy / quality Δ。

### 下午：Position / Length-Matched Non-Ack Pruning

运行：

- ACK pruning；
- random same-count；
- position-matched non-ack；
- length-matched non-ack。

输出：

- accuracy delta；
- correct→wrong flip；
- wrong→correct flip；
- suppression rate。

### 晚上

整理成机制证据：

1. ACK injection causally increases future ack probability；
2. ACK pruning is safer than matched non-ack pruning；
3. therefore not just position / length / random shortening。

---

## Day 4：Learned Detector + 4-Agent Probe

### 上午

训练：

- TF-IDF + Logistic Regression；
- DistilBERT, optional。

数据：

- existing reasoning traces；
- AutoResearch traces。

评估：

- held-out F1；
- cross-domain F1；
- cross-backbone F1。

### 下午

Online validation：

- AutoResearch 100 tasks, TF-IDF pruning；
- StrategyQA 200 tasks, TF-IDF pruning；
- Llama GSM8K 200 tasks, optional。

### 晚上

4-agent probe, optional：

- 30 AutoResearch tasks；
- baseline vs ACK pruning；
- quality / calls / latency。

---

## Day 5：重写论文与做图

这一天不要再大规模跑实验，除非补缺失表格。

### 必须完成的写作改动

1. Abstract 重写；
2. Introduction 重写；
3. Contributions 重写；
4. 新增 AutoResearch section；
5. 新增 causal intervention section；
6. 新增 measured efficiency section；
7. Discussion 降调 claim；
8. Limitation 更诚实但更主动；
9. Related work 加 Auto Research / AI Scientist / agent workflow 相关 work；
10. Appendix 加 task examples、judge prompt、logging details。

---

# 11. 新增图表清单

## Table 1: AutoResearch Main Results

```markdown
| Workflow | Model | Pruning | N | Quality Δ | Pairwise Win/Tie/Loss | Grounding Error Δ | Calls ↓ | Tokens ↓ | Wall-clock ↓ |
|---|---|---|---:|---:|---:|---:|---:|---:|---:|
```

## Table 2: Causal Intervention

```markdown
| Intervention | N positions | P(next ack) | Next-3 Ack Rate | Chain Length | Accuracy / Quality Δ |
|---|---:|---:|---:|---:|---:|
```

## Table 3: Matched Controls

```markdown
| Pruning Target | N tasks | Suppression | Accuracy Δ | Correct→Wrong | Wrong→Correct |
|---|---:|---:|---:|---:|---:|
```

## Table 4: Learned Detector Online Validation

```markdown
| Model | Domain | Detector | N | F1 | Suppression | Accuracy / Quality Δ | Calls ↓ |
|---|---|---|---:|---:|---:|---:|---:|
```

## Figure 1: Revised Pipeline

包括：

```text
Trace collection
→ Function annotation
→ Single-message perturbation
→ Aggregate pruning
→ Causal intervention
→ AutoResearch validation
→ Measured efficiency
```

## Figure 2: Mechanism Diagram

展示：

```text
single ack removal
→ high local trajectory perturbation

ack-rich feedback loop
→ self-reinforcing cascade

aggregate ack pruning
→ cascade dissolution
→ stable final answer + fewer calls
```

## Figure 3: Latency / Calls / Quality Tradeoff

x-axis: call reduction  
y-axis: quality delta  
points: different pruning policies

---

# 12. 新标题建议

## 推荐标题

**When Locally Disruptive Messages Are Globally Prunable: Functional Communication Analysis in LLM Agent Systems**

## 备选标题

**Local Disruption, Global Redundancy: Functional Message Pruning in LLM Agent Communication**

## 如果想突出 Auto Research

**Pruning Redundant Communication in LLM Research Agents via Functional Message Analysis**

推荐第一版，因为更 general，也更像 NeurIPS。

---

# 13. 新摘要草案

```text
LLM agent systems often communicate through many low-content messages, such as acknowledgments and flow-control confirmations. Are such messages functionally necessary or merely structural artifacts of multi-agent interaction? We uncover a perturbation-pruning dissociation: acknowledgment messages are among the most disruptive when removed individually, yet can be safely pruned in aggregate. We first establish this phenomenon in two-agent solver-checker systems across Qwen and Llama backbones and multiple reasoning domains. We then validate it in a three-agent Auto Research workflow involving retrieval, synthesis, and critique, where acknowledgment pruning reduces LLM invocations by X% and measured wall-clock latency by Y% without degrading judge-rated answer quality or citation grounding. Controlled interventions show that injecting acknowledgments causally increases subsequent acknowledgment probability, while matched non-ack controls support a cascade-dissolution mechanism. Finally, lightweight learned detectors approximate oracle pruning across domains, reducing dependence on hand-crafted heuristics. Our results show that local perturbation sensitivity can overestimate global communication necessity, and provide a practical analysis protocol for identifying safely prunable message functions in LLM agent systems.
```

注意：X/Y 必须替换成真实实验结果。

---

# 14. Introduction 重写结构

## Paragraph 1: 真实 agent motivation

现代 LLM agent 系统通常由多个角色组成：planner、retriever、writer、critic、verifier。它们通过自然语言消息协作，但这些消息中有大量低内容确认、流程推进、重复性反馈。每条消息可能触发一次 LLM 调用，因此通信冗余会显著增加 latency 和 cost。

## Paragraph 2: 现有方法缺口

现有 efficiency 方法多关注 token compression、context compression、agent pruning、topology optimization、turn-level stopping，但较少分析 message function。尤其缺少对“某类消息是否功能必要”的系统性分析。

## Paragraph 3: 核心反直觉现象

直觉认为低信息量 acknowledgment 应该不重要。但单条删除实验显示它们最 disruptive；整体剪枝却显示它们可以安全删除。这说明 local perturbation sensitivity 不等于 global necessity。

## Paragraph 4: 机制假设

acknowledgment 形成 self-reinforcing cascade。单独删除会打断局部 autoregressive conditioning，因此造成轨迹扰动；整体删除则使 cascade 消失，不会累积错误。

## Paragraph 5: 本文贡献

列四条 contribution。

---

# 15. Discussion 改写原则

## 15.1 降调 claim

不要写：

> Acknowledgment pruning is generally safe.

改成：

> Acknowledgment pruning is safe in acknowledgment-rich feedback topologies after empirical screening.

不要写：

> Our learned detector solves deployment.

改成：

> Lightweight learned detectors reduce reliance on brittle heuristics, though deployment still requires per-domain screening.

不要写：

> Cascade dissolution is the mechanism.

改成：

> Controlled interventions support cascade dissolution as a plausible mechanism, but further work is needed to fully characterize its causal structure across architectures.

---

## 15.2 把 limitation 变成适用条件

当前 limitation：

- MATH 上退化；
- Qwen3 heuristic 不迁移；
- stochastic decoding 下 Llama 不稳；
- Gemma ack rate 低；
- debate / code-generation ack 少。

改写成：

> These boundary conditions suggest that pruning opportunity is topology-, domain-, and detector-dependent. The method is most useful when a workflow induces acknowledgment-rich feedback loops with low semantic novelty. It is less applicable to debate-like or code-generation topologies where acknowledgments are rare, or to domains where acknowledgments carry substantive reasoning content.

这样 reviewer 会觉得作者理解适用范围。

---

# 16. 不建议五天内做的事情

## 16.1 不要大规模跑 72B

2–4 张 5090 跑 72B 不划算。当前 72B probe 已经够作为 preliminary evidence。新增 AutoResearch 和因果实验更能涨分。

## 16.2 不要构造很大的新 benchmark

100 个高质量 AutoResearch tasks 足够。不要追求 1000 个粗糙任务。

## 16.3 不要扩 taxonomy 到十几类

五分类已经有一致性问题。扩 taxonomy 会制造新风险。

## 16.4 不要继续只加 GSM8K seed

当前 GSM8K 10-seed 已经比较强。继续加 seed 对分数提升有限。

## 16.5 不要过度优化 heuristic

heuristic non-transferability 已经被发现。更好的策略是转向 learned detector + empirical screening。

---

# 17. 最小可行版本

如果五天里工程阻力很大，最低限度完成以下五件事：

1. AutoResearch 50 tasks，Qwen2.5-7B，baseline vs ACK pruning；
2. 报告 calls、tokens、wall-clock latency；
3. 做 LLM judge pairwise win/tie/loss；
4. 做 GSM8K 上的 ACK injection causal experiment；
5. 做 GSM8K 上的 position-matched non-ack control。

只要这五项完成，论文就会比当前版本强很多。

---

# 18. 理想完成版本

理想五天产出：

1. AutoResearch 100 tasks, Qwen2.5-7B；
2. AutoResearch 50 tasks, Llama-3.1-8B；
3. AutoResearch ACK + ACK/FLOW pruning；
4. cost profiling；
5. ACK injection；
6. position / length matched controls；
7. TF-IDF learned detector online validation；
8. 4-agent probe；
9. 完整论文重写。

---

# 19. 预期分数提升

## 当前版本预期

- Quality: 2–3
- Clarity: 3
- Significance: 2–3
- Originality: 3
- Overall: 5 左右
- 倾向：borderline reject / weak reject

## 完成最小可行版本后

- Quality: 3
- Clarity: 3
- Significance: 3
- Originality: 3
- Overall: 5.5–6
- 倾向：borderline / weak accept 边缘

## 完成理想版本后

如果结果漂亮：

- AutoResearch call reduction > 30%；
- wall-clock reduction > 20%；
- judge tie + pruned win > 70%；
- citation grounding 不下降；
- ack injection 显著增加 future ack rate；
- matched non-ack pruning 明显更差；

则可能达到：

- Overall: 6.5–7
- 倾向：weak accept / accept 边缘

---

# 20. 最终优先级总结

五天内真正应该做的是：

1. **把场景从 toy solver-checker 扩展到 Auto Research agent workflow。**
2. **把“省 inference calls”变成真实 latency / cost profiling。**
3. **把 cascade dissolution 从相关性提升到因果干预支持。**
4. **用 matched controls 排除 position / length / random shortening confound。**
5. **用 lightweight learned detector 减少 oracle / heuristic 依赖。**

最终论文应该传达的核心思想是：

> 单条消息的局部扰动敏感性不等于它的全局功能必要性。Acknowledgment 是这个现象的典型案例。在 acknowledgment-rich 的 LLM agent feedback workflow 中，系统性剪掉这类低内容消息可以让通信 cascade 消失，从而降低调用成本，同时保持最终输出质量。

