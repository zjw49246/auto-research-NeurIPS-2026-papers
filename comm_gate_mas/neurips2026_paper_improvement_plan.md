# NeurIPS 2026 论文五天提升方案：Agent Auto Research 场景下的 Message Pruning 强化计划

> **版本说明**：本方案基于 comm_gate_mas_NeurIPS2026_v4_20260430.pdf 制定。latest_draft.pdf 经验证与 v4 文本内容完全一致（仅编译时间不同：v4 为 4/30 20:26，latest 为 5/1 01:19），因此本方案同时适用于两个版本。

## 0. 总目标

当前论文的核心发现是：在 two-agent LLM solver-checker 系统中，acknowledgment 消息虽然在单条删除时最 disruptive，但在整体剪枝时可以安全删除，并能减少大量 LLM inference calls。

五天内的目标不是重写一篇新论文，而是用 2–4 张 RTX 5090，在最短时间内补上最能提升 NeurIPS 评分的证据，把论文从：

> acknowledgment pruning 在 GSM8K / StrategyQA two-agent 系统中有效

提升为：

> LLM agent communication 中存在一种通用的分析误区：局部扰动敏感性会高估某些结构性冗余消息的全局必要性。本文提出 functional message analysis protocol，并在 controlled reasoning system 与真实 Auto Research agent workflow 中验证该现象、机制和实际成本收益。

### 执行前提：Agent Auto Research 高速执行

本方案假定使用 Agent Auto Research 执行，coding 和写作速度远超人工。因此：

- **真正的瓶颈是 GPU 推理时间**，不是人力编码或写作；
- 五天内完成理想版本（全部实验 + 论文重写）完全可行；
- Day 0-1 的 benchmark 搭建、代码实现、论文重写等工作可以在数小时内完成；
- 并行 GPU 实验是关键优化点：多个实验可以在不同时段排队执行。

### 时间瓶颈分析

| 任务 | Agent 耗时 | GPU 耗时 (4×RTX 5090) |
|---|---|---|
| 构造 100 个 research questions | 数小时 | 无 |
| 搭建 3-agent workflow 代码 | 数小时 | 无 |
| 100 tasks × 3 conditions (Qwen 7B) | 无 | ~8–12h |
| Ack injection 500 positions × 3 conditions | 无 | ~6–10h |
| Matched controls 500 tasks × 4 conditions | 无 | ~8–12h |
| Learned detector 训练 + online validation | 30min | ~4–6h |
| Judge 评价 (API) | 无 | ~2–4h |
| 论文重写 (abstract, intro, 新 sections) | 数小时 | 无 |

预期提升方向：

- **Quality**：通过因果干预和 matched control 补强机制证据。
- **Significance**：通过 Auto Research agent workflow 证明不是 toy benchmark 现象。
- **Practicality**：通过 wall-clock latency / GPU cost profiling 证明不是只省 call count。
- **Robustness**：通过 learned detector 的跨域 online validation 减少 oracle / heuristic 依赖。
- **Clarity**：通过重构叙事，让论文从 "ack pruning trick" 变成 "agent communication analysis principle"。

---

## 1. 当前论文的主要短板

### 1.1 任务场景偏 toy

当前主结果主要来自 GSM8K、StrategyQA、ARC 等标准 reasoning benchmarks。虽然实验量不少（15 个 configurations、3 backbones、3 topologies、7B/14B/72B），但 reviewer 很可能认为它仍然是一个 two-agent solver-checker toy phenomenon，而不是对真实 agent 系统有普遍意义的发现。

> **注意**：论文的实验广度（backbone × domain × topology × scale）已经较强，reviewer 的核心担忧不是实验量不够，而是**任务类型单一**——全部是 QA reasoning，缺少更真实的 agent workflow。

需要补：

- 多 agent workflow；
- 更真实的任务；
- research / retrieval / synthesis / critique 类型任务；
- 与当前 LLM agent / AI scientist / Auto Research 方向相关的场景。

### 1.2 核心机制仍偏相关性

论文提出 cascade dissolution：acknowledgment 会形成自强化 cascade，整体剪枝会让这些 cascade 消失而非累积错误。论文 Discussion 第 6 节自己承认 "cascade dissolution is a descriptive framework for the observed correlational patterns, not a confirmed causal mechanism"，并明确写道 "causal verification through controlled intervention remains needed"。

目前证据主要是：

- P(ack | previous ack) 较高（Qwen 90.7%, Llama 61.9%，均 p < 10⁻¹³）；
- position-controlled logistic regression 显示 previous ack 提高后续 ack 概率（Qwen OR=6.43, Llama OR=2.02）；
- dose-response curve 较平；
- ack 信息量低（Qwen bigram novelty 5.7%）。

这些是相关性证据，不是严格因果证明。

需要补：

- **ack injection（正向因果）**：在 non-ack 位置插入 ack，观察是否引发后续 ack cascade；
- **ack chain breaking（反向因果）**：在自然 ack 位置替换为 non-ack content，观察是否打断 cascade；
- **position-matched non-ack pruning**：排除位置 confound；
- **length-matched non-ack pruning**：排除长度 confound。

### 1.3 实际效率收益不够清楚

当前论文强调减少的是 inference calls，而不是 token-level compression。论文自己承认 "message-level suppression (36–56%) translates to lower token-level savings (~10% Qwen, ~28% Llama) because acknowledgments are shorter than substantive messages"。但 reviewer 会问：

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

当前最强证据其实是 ack vs non-ack（Cohen's κ=0.721，binary ack/non-ack κ=0.81）。Fine-grained 5-class taxonomy 中部分标签一致性较弱，尤其是 information_provision（κ=0.237）。

建议写作上收敛：

- 主结论聚焦 ack / low-content flow-control；
- 5-class taxonomy 作为 analysis protocol，而不是硬说每一类都可靠；
- 对 fine-grained labels 的结论降级为 exploratory。

### 1.5 Detector 的 deployment claim 偏强

当前 oracle 结果较强，heuristic 有一些结果，但 heuristic non-transferability 明显（F1: 0.845 → 0.213 跨域）；DistilBERT online validation 只在单个 configuration 上做了（Qwen/GSM8K, N=300, seed 42, Δ=+0.7pp）。论文自己承认 "generalization across datasets, backbones, and random seeds remains unverified"。

需要补：

- TF-IDF + Logistic Regression online validation；
- DistilBERT / TF-IDF 在 Auto Research、StrategyQA、Llama 上的小规模验证；
- leave-one-domain / leave-one-backbone generalization。

### 1.6 Llama heuristic over-suppression 的未解释发现

论文 Appendix S.12 揭示一个重要问题：Llama online heuristic 的 suppression rate 是 74.8%，但只有 32.9% 是真正的 acknowledgment，54% 包含数学内容。然而 over-suppression 没有造成准确率下降（甚至 +8pp）。

这说明 Llama 的很多非 ack 消息也是 redundant。这个发现目前论文只做了简单解释（"removal of redundant later-round messages"），但可以：

- 作为 secondary finding 深入分析；
- 为 ACK+FLOW 扩展提供 Llama 侧的佐证；
- 如果在 Auto Research 实验中也观察到类似现象，可以统一叙述。

---

## 2. 改进后的论文主叙事

### 2.1 原始叙事

> In two-agent solver-checker systems, acknowledgment messages are individually disruptive but aggregately prunable.

这个叙事有意思，但容易被认为只是一个小现象。

### 2.2 推荐叙事

> We identify a general failure mode in LLM agent communication analysis: local perturbation sensitivity can overestimate the global necessity of structurally redundant messages. Acknowledgment messages are the canonical case. We show that they are locally disruptive but globally prunable in controlled solver-checker systems and in a realistic Auto Research agent workflow. We further provide causal interventions, matched controls, measured efficiency gains, and practical learned detectors.

> **注意**：叙事升级的幅度必须和新增证据量匹配。如果因果实验结果弱或 Auto Research 实验中 ack rate 过低，则叙事应相应降级，避免 overclaim。

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

### 优先级排序（已根据 ROI 和对 reviewer 说服力重新排序）

| 优先级 | 实验 | 核心价值 |
|---|---|---|
| **P0** | Ack injection 因果干预 + chain breaking | 直接回应论文最大弱点——机制证据仅为相关性 |
| **P0** | Auto Research 3-agent workflow（含 cost profiling） | 扩展应用场景，同时解决 toy benchmark 和效率收益两个短板 |
| **P1** | Position / length-matched non-ack control | 排除 confound，提升 quality 评分 |
| **P1** | 论文重写（abstract, intro, contributions, 新 sections） | 叙事升级 |
| **P2** | Learned detector 跨域 online validation | 补充 deployment story |
| ~~P3~~ | ~~4-agent topology probe~~ | **不建议做**：30 tasks 统计检验力太弱，3-agent 已足够 |

> **变更说明**：原方案将 Auto Research 排在第一、Ack injection 排在第三。重新评估后，**Ack injection 因果干预提升到 P0**，因为它直接回应论文自己承认的最大弱点（"causal verification remains needed"），且 ROI 最高——可以在已有 GSM8K traces 上做，不需要新 benchmark。4-agent probe 从原方案中移除，因为 30 tasks 的规模下统计检验力不足以支撑任何有意义的结论。

---

# 4. 实验一：Ack Injection 因果干预 + Chain Breaking

## 4.1 目的

当前 cascade dissolution 主要是相关性证据。Ack injection + chain breaking 是五天内最强、最简单的因果实验，直接回应论文 Discussion 第 6 节自己提出的 "causal verification through controlled intervention remains needed"。

核心问题：

> Does inserting an acknowledgment causally increase the probability of future acknowledgments? Does replacing a natural acknowledgment with content causally break the cascade?

正向 + 反向双向验证远比单向实验更有说服力。

---

## 4.2 实验设计

### Part A: Ack Injection（正向因果）

从已有 baseline traces 中采样 non-ack 位置，在这些位置插入不同类型消息，然后从插入点之后 regenerate。

条件：

1. Original trace（不插入任何消息）
2. ACK injection
3. Neutral injection（控制组）
4. Flow-control injection
5. Content-bearing short control（可选）

#### ACK injection examples

```text
Correct.
I agree.
That looks right.
Yes, proceed.
The previous step is valid.
```

#### Neutral injection examples

```text
[Step recorded.]
Continue.
Message received.
```

#### Flow-control examples

```text
Proceed to the next step.
Continue with the calculation.
Move on to verification.
```

### Part B: Chain Breaking（反向因果）

从已有 baseline traces 中找到**自然 ack chain 起始位置**（即 ack 后面紧跟 ack 的位置），将该 ack 替换为 content-bearing non-ack 消息，然后从替换点之后 regenerate。

条件：

1. Original trace（保留自然 ack）
2. Replace ack with content message（例如替换成一句 reasoning guidance 或 verification）
3. Replace ack with neutral placeholder（控制组）
4. Simply remove ack（对照 remove ablation）

#### Content replacement examples

```text
Let me re-examine step 3 more carefully.
I noticed the calculation in the previous step uses the wrong formula.
Could you verify whether the final answer matches the constraints?
```

#### 设计原则

- 替换消息的**长度应与被替换 ack 匹配**（避免引入长度 confound）；
- 替换位置必须是**confirmed ack chain 的第一个 ack**（即 P(next msg is ack | current msg) > 0.5）；
- 每条替换消息必须是**语义合理的**（不能是随机文本），否则下游 LLM 可能因为不理解而产生异常行为。

---

## 4.3 指标

核心指标：

- future ack rate in next k messages（k=1, 3, 5）；
- average ack chain length after intervention point；
- probability of next message being ack；
- total number of generated messages；
- final accuracy / final judge quality；
- trajectory similarity（cosine similarity of final answers）。

---

## 4.4 推荐结果表

### Part A: Ack Injection

```markdown
| Intervention | N positions | P(next ack) | Ack rate next-3 | Chain length | Accuracy Δ |
|---|---:|---:|---:|---:|---:|
| Original | 500 | 0.31 | 0.28 | 1.4 | 0.0 |
| Neutral injection | 500 | 0.34 | 0.30 | 1.5 | -0.1 |
| ACK injection | 500 | 0.58 | 0.51 | 2.3 | +0.0 |
| Flow-control injection | 500 | 0.40 | 0.36 | 1.7 | -0.2 |
```

### Part B: Chain Breaking

```markdown
| Intervention | N positions | P(next ack) | Ack rate next-3 | Chain length | Accuracy Δ |
|---|---:|---:|---:|---:|---:|
| Original (natural ack) | 300 | 0.91 | 0.78 | 3.1 | 0.0 |
| Replace with content | 300 | 0.35 | 0.29 | 1.3 | +0.2 |
| Replace with neutral | 300 | 0.42 | 0.35 | 1.6 | -0.1 |
| Simply remove | 300 | 0.38 | 0.31 | 1.4 | +0.1 |
```

表中数值是期望格式，不是实际结果。

---

## 4.5 主文写法

如果正向 + 反向都显著：

> Controlled insertion of acknowledgment messages causally increased the probability of subsequent acknowledgments (P(next ack): 0.31 → 0.58 for ack injection vs. 0.34 for neutral control), while replacing natural acknowledgments with content messages broke the cascade (P(next ack): 0.91 → 0.35). This bidirectional evidence supports cascade dissolution as a causal mechanism, ruling out a purely correlational interpretation based on message position or dialogue phase.

如果只有正向显著：

> Ack injection causally increased future ack probability, supporting the cascade component. Chain breaking yielded mixed results, suggesting that cascade dynamics depend on both message function and dialogue state.

如果都不显著，写成 limitation：

> Ack injection did not significantly increase future ack probability outside naturally occurring ack-rich phases, suggesting that the sequential dependence observed in correlational analysis may reflect dialogue-phase effects rather than per-message causal chains.

建议**优先在 Qwen/GSM8K 上做**，因为 Qwen 的 cascade rate 最高（90.7%），最可能看到因果效应。

---

## 4.6 计算量估算

- Part A: 500 positions × 4 conditions = 2000 次 regeneration（每次从插入点到对话结束，平均 ~3-5 个 LLM calls）
- Part B: 300 positions × 4 conditions = 1200 次 regeneration
- 总计 ~3200 次 partial regeneration，Qwen 7B 在 4×5090 上估计 **6-10 小时**
- Agent 代码实现 ~2 小时

---

# 5. 实验二：Auto Research Agent Workflow + Cost Profiling

> **注意**：Cost profiling 不作为独立实验，而是作为 Auto Research 实验的内嵌组件。在 Auto Research 的每次 LLM call 中记录完整 timing 和 token 数据即可，零额外计算成本。

## 5.1 目的

证明该现象不是 GSM8K / StrategyQA 这种标准 QA benchmark 中的 toy phenomenon，而是出现在更真实的 LLM agent workflow 中。同时回应 reviewer 对 "36–56% inference-call reduction 是否等价于真实成本下降" 的质疑。

核心 reviewer concerns：

> This is interesting, but does it matter for real agent systems?

> You reduce inference calls, but what about actual wall-clock time and cost?

新增实验回答：

> Yes. In a three-agent Auto Research workflow involving retrieval, synthesis, and critique, acknowledgment pruning reduces LLM invocations by X% and measured wall-clock latency by Y% without degrading answer quality or citation grounding. Token-level savings are smaller because acknowledgments are short, but call-level savings translate to real latency reduction in sequential orchestration.

---

## 5.2 Ack Prevalence 验证（Day 0 关键门控）

**这是整个 Auto Research 实验的最大风险点。** 论文已经证明 ack 是特定 topology 的 emergent property：

- solver-checker: 36–56% ack suppression
- debate: 0.2% acks
- code-generation (HumanEval): 4% acks

3-agent research workflow（Retriever → Synthesizer → Critic）的 Critic → Synthesizer 反馈循环接近 solver-checker 结构，但 ack rate 可能只有 15–30%。

### Day 0 必须做的 pilot

在冻结实验矩阵之前，先跑 **10-task smoke test**，检查：

1. ack rate 是否 ≥ 15%；
2. 如果 < 15%，启用 fallback plan；
3. 如果 < 5%，放弃 ACK-only target，仅做 ACK+FLOW。

### Fallback plan（如果 ack rate < 15%）

调整 workflow 设计，增加 Critic → Synthesizer 的多轮验证循环：

```text
User question
→ Retriever: search plan + retrieved evidence
→ Synthesizer: draft answer
→ Critic: critique round 1
→ Synthesizer: revision 1
→ Critic: critique round 2 (verify fixes)
→ Synthesizer: revision 2
→ Critic: final verification or remaining concerns
→ Synthesizer: final answer
```

强制至少 3 轮 revision-verification 循环会模仿 solver-checker 的反复确认模式，增加 ack 产生的机会。

如果 ack rate 仍然低，将**主结论定位为 ACK+FLOW**（扩展到 low-content flow-control 消息），并在论文中明确写：

> In Auto Research workflows, strict acknowledgment prevalence is lower than in solver-checker systems, but broader low-content flow-control messages (acknowledgments + procedural confirmations) exhibit analogous pruning safety.

---

## 5.3 Benchmark 名称

建议主文使用：

> AutoResearch-CommPrune

---

## 5.4 任务形式

给定一个 research question，让 agent 从本地论文库中检索、分析、综合，并输出一个 grounded research answer。

每个任务包括：

- research question；
- local document corpus；
- retrieved evidence；
- multi-agent discussion trace；
- final answer；
- automatic / LLM-judge evaluation。

---

## 5.5 任务类型

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

## 5.6 文档库构造

用固定本地 corpus。Agent 可以在数小时内完成 corpus 构建。

- 30–80 篇相关论文；
- 每篇切成 chunks；
- 建立 BM25 或 embedding retrieval；
- 每个 task top-k 检索 evidence。

### 防止 circularity 的领域覆盖要求

为避免 reviewer 质疑 "用 multi-agent communication 领域的论文测试 multi-agent communication pruning 是 circular evaluation"，**corpus 必须包含多个领域的论文**：

- **主领域**（~40%）：LLM agents, multi-agent communication, agent pruning, token compression
- **相关领域**（~30%）：AI Scientist, auto research, tool-use agents
- **泛 AI/ML 领域**（~30%）：NLP (summarization, QA), CV (image captioning), RL (RLHF), 通用 ML (architecture search, hyperparameter tuning)

research questions 也应覆盖这三个层次，确保不是只问 multi-agent communication 的问题。

---

## 5.7 Agent Workflow 设计

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

## 5.8 Pruning Targets

### Target 1: ACK-only

严格 acknowledgment：

- "I agree."
- "Correct."
- "This looks good."
- "The answer is correct."
- "No issue found."

这是和原论文主线一致的 target。

### Target 2: ACK+FLOW

更宽的 low-content flow-control messages：

- "I will now proceed."
- "Moving to the next step."
- "Let's continue."
- "The retrieved documents seem relevant."
- "I will summarize now."

注意：ACK+FLOW 应该标成 exploratory，不要让主结论依赖它。**除非 pilot 发现 ack rate < 15%，此时 ACK+FLOW 成为主 target。**

推荐写法：

> For Auto Research workflows, we evaluate strict acknowledgment pruning as the primary intervention and a broader low-content flow-control pruning policy as an exploratory extension.

---

## 5.9 Conditions

主实验条件：

1. Baseline
2. Oracle ACK pruning
3. Learned ACK pruning（TF-IDF）
4. ACK+FLOW pruning（exploratory）

---

## 5.10 评价指标

### 5.10.1 Quality Score

用 LLM judge 打分，维度：

- factual correctness；
- coverage；
- citation grounding；
- reasoning coherence；
- usefulness。

每项 1–5 分。

#### Judge 一致性检验

必须做 judge calibration：

- 随机抽取 20 个 task 的 baseline answer，让 judge 打分两次（不同 prompt order），计算 intra-rater agreement；
- 如果 agreement < 0.7，调整 judge prompt 或换 judge model；
- 报告 judge 的 inter-item consistency（Cronbach's α 或 ICC）。

### 5.10.2 Pairwise Preference

让 judge 比较 baseline vs pruned：

- A wins；
- B wins；
- tie。

必须随机交换 A/B 顺序，防止 position bias。

### 5.10.3 Citation Grounding

自动或 judge 检查：

- 引用是否来自 retrieved evidence；
- claim 是否有 evidence support；
- hallucinated citation rate；
- unsupported claim rate。

### 5.10.4 Cost Metrics（内嵌 Cost Profiling）

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

> **实现注意**：这些 logging 只需在 agent workflow 代码中每次 LLM call 前后加 timestamp + token count 记录，Agent 实现 ~30 分钟。

---

## 5.11 推荐结果表

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

## 5.12 可接受结果标准

理想结果：

- Calls reduction ≥ 25%；
- Wall-clock latency reduction ≥ 15%；
- Pairwise judge 中 tie + pruned win ≥ 70%；
- Quality Δ 在 ±0.1 / 5 内；
- Citation hallucination 不显著增加。

如果 ACK+FLOW 有轻微质量下降，也可以作为 boundary condition。

主结论只要 ACK-only 安全即可。

### 如果 ack rate 低导致 call reduction < 15%

这仍然是有价值的结果，可以写成：

> In Auto Research workflows with lower acknowledgment prevalence (~15% vs. ~50% in solver-checker), pruning opportunity is correspondingly reduced but remains non-inferior in quality. This confirms that pruning safety generalizes while pruning yield is topology-dependent.

---

## 5.13 Cost Profiling 主文写法

不要写：

> pruning reduces cost by 56%.

建议写：

> Since acknowledgments are short, token-level savings are smaller than call-level savings. However, in orchestration regimes where each message triggers a separate LLM invocation, acknowledgment pruning reduces measured wall-clock latency by X% and LLM invocations by Y%. Token-level compression methods are complementary.

这样更诚实，也更不容易被 reviewer 攻击。

---

# 6. 实验三：Position / Length-Matched Non-Ack Control

## 6.1 目的

排除以下 confounds：

- ack pruning 安全只是因为 ack 出现在不重要的位置；
- ack pruning 安全只是因为 ack 很短；
- ack pruning 安全只是因为删少量消息或缩短对话；
- ack pruning 和 random pruning 没区别。

论文目前有的 confound control 比较弱：
- Token-matched random：只匹配了总删除量，没匹配位置（Table 2, Qwen 9.9% suppression vs ack 56.3%，差距巨大）；
- Length-only baseline：Qwen char<30 效果类似 heuristic，但 Llama 几乎无效（0.8% suppression）；
- 缺少 position-matched 和 speaker-matched 的 non-ack pruning。

---

## 6.2 实验条件

比较四种 pruning：

1. ACK pruning
2. Random same-count pruning
3. Position-matched non-ack pruning
4. Length-matched non-ack pruning
5. Speaker/round-matched non-ack pruning（可选）

---

## 6.3 Matching 方法

对每条 ack message，找一个 non-ack control：

- same speaker；
- same round or closest round；
- similar character length or token length；
- not labeled as acknowledgment；
- if possible, same topology / same task type。

如果找不到 perfect match，就用 nearest neighbor matching。

---

## 6.4 指标

- accuracy delta；
- final answer flip rate；
- call reduction；
- future ack rate；
- trajectory similarity；
- judge quality for AutoResearch（可选）。

---

## 6.5 推荐结果表

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

## 6.6 主文写法

理想 claim：

> Ack pruning remains safe while position- and length-matched non-ack pruning degrades accuracy, indicating that pruning safety is not explained by message length, position, or conversation shortening alone.

这是非常能提高 reviewer 信心的实验。

---

# 7. 实验四：Learned Detector 跨域 Online Validation

## 7.1 目的

减少 reviewer 对 oracle / heuristic 的质疑。

当前问题：

- oracle detector 依赖 GPT-4o；
- heuristic detector 不迁移（F1: 0.845 → 0.213）；
- DistilBERT online validation 只在单个 configuration（Qwen/GSM8K, N=300, seed 42）。

目标：

> Show that lightweight learned detectors can replace hand-crafted heuristics in multiple settings.

---

## 7.2 Detector 类型

建议训练两个：

### A. TF-IDF + Logistic Regression

优点：

- 训练快（秒级）；
- CPU 推理极快；
- 可解释；
- deployment 更 practical；
- 论文已有 offline F1=0.859 的结果，只需补 online validation。

### B. DistilBERT

优点：

- 论文已有 offline F1=0.867 和单个 online validation 的结果；
- 和当前论文已有结果衔接。

如果只能选一个，优先 TF-IDF + Logistic Regression，因为稳定且快。

---

## 7.3 数据

训练集包括：

- 原 GSM8K traces；
- StrategyQA traces；
- ARC traces；
- AutoResearch traces（新增）。

标签：

- oracle ack/non-ack；
- 或 human-validated subset。

Split 必须按 task_id split，避免同一问题的消息泄漏。

---

## 7.4 Generalization 设置

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

## 7.5 Online Validation

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

## 7.6 主文写法

建议写：

> Learned detectors reduce dependence on brittle hand-crafted heuristics. While universal deployment still requires empirical screening, a lightweight TF-IDF classifier achieves online pruning behavior close to oracle labels across reasoning and Auto Research settings with negligible CPU overhead.

不要写：

> solved deployment.

要写：

> bridges part of the oracle-to-deployment gap.

---

# 8. 五天执行计划

> **核心原则**：Agent 做 coding/writing，GPU 做 inference。并行化 GPU 实验是关键——不同实验可以在 GPU 空闲时段依次执行。

## Day 0：冻结实验矩阵 + Ack Prevalence Pilot + 代码骨架

目标：防止发散，**并验证 Auto Research workflow 的 ack rate**。

### 关键门控：10-task Pilot

Agent 在数小时内搭建 3-agent workflow 基础代码，然后立即跑 10-task pilot：

- 如果 ack rate ≥ 15%：继续原计划
- 如果 ack rate 10–15%：增加 Critic-Synthesizer 循环轮次（见 §5.2 fallback plan）
- 如果 ack rate < 10%：切换到 ACK+FLOW 作为主 target
- 如果 ack rate < 5%：放弃 Auto Research ACK-only，聚焦因果实验 + matched controls

### 当天冻结的最小实验矩阵

| Experiment | N | Model | Conditions |
|---|---:|---|---|
| Ack injection + chain breaking | 500+300 positions | Qwen2.5-7B | original / ACK inject / neutral inject / chain break |
| AutoResearch 3-agent | 100 | Qwen2.5-7B | baseline / ACK / ACK+FLOW |
| AutoResearch 3-agent | 50 | Llama-3.1-8B | baseline / ACK |
| Cost profiling（内嵌） | 100 | Qwen2.5-7B | baseline / ACK / ACK+FLOW |
| Position-matched pruning | 300–500 tasks | Qwen2.5-7B | ACK / random / matched non-ack |
| Learned detector online | 100–300 tasks | Qwen + Llama | TF-IDF / DistilBERT |

### 当天 Agent 产出

- `tasks.jsonl`（100 个 research questions）
- `corpus/`（多领域论文 corpus）
- `agent_workflow.py`（3-agent workflow + logging）
- `pruning_policy.py`（ACK / ACK+FLOW / oracle / learned）
- `logging_schema.json`
- `judge_prompt.txt`（含 calibration 检验设计）
- `run_matrix.yaml`
- `ack_injection.py`（因果实验代码）
- `chain_breaking.py`（反向因果实验代码）
- `matched_control.py`（matched pruning 代码）
- **10-task pilot 结果** + ack rate 确认

---

## Day 1：因果实验（GPU 密集）+ AutoResearch Benchmark 完善

### GPU 任务（排队执行）

1. **Ack injection**（Qwen/GSM8K, 500 positions × 4 conditions）
2. **Chain breaking**（Qwen/GSM8K, 300 positions × 4 conditions）

### Agent 并行任务

- 完善 100 个 research questions（LLM 生成 150 → 筛选到 100）；
- 完善 corpus（确保多领域覆盖）；
- 完善 3-agent workflow 代码（基于 pilot 调整）；
- 实现 judge 评价代码 + calibration 检验。

### 晚上

- 分析因果实验结果；
- 如果因果实验显著，确认 cascade dissolution 的因果证据；
- 如果不显著，调整 claim 并开始准备 limitation 写法。

---

## Day 2：AutoResearch 主实验 + Matched Controls

### GPU 任务（排队执行）

1. **AutoResearch baseline**（Qwen, 100 tasks）
2. **AutoResearch ACK pruning**（Qwen, 100 tasks）
3. **AutoResearch ACK+FLOW**（Qwen, 100 tasks）
4. **Position/length-matched controls**（Qwen/GSM8K, 500 tasks × 4 conditions）

### Agent 并行任务

- 整理 Day 1 因果实验结果为论文 section 草稿；
- 准备 matched controls 的 matching 代码。

### 晚上

- AutoResearch Llama 50 tasks（如果 GPU 空闲）；
- 运行 judge 评价：scalar scores、pairwise preference、citation grounding。

### 当天产出

- 因果实验主结果表；
- AutoResearch 主结果表（含 cost profiling）；
- Matched controls 结果表。

---

## Day 3：Learned Detector + Judge 评价 + 结果整理

### 上午

训练 learned detectors：

- TF-IDF + Logistic Regression（秒级训练）；
- DistilBERT（如已有代码，直接复用论文已有的训练 pipeline）。

数据：

- existing reasoning traces + AutoResearch traces。

评估：

- held-out F1；
- cross-domain F1（train: GSM8K+StratQA, test: AutoResearch）；
- cross-backbone F1（train: Qwen, test: Llama）。

### 下午

Online validation（GPU 任务）：

- AutoResearch 100 tasks, TF-IDF pruning；
- StrategyQA 200 tasks, TF-IDF pruning；
- Llama GSM8K 200 tasks（可选）。

### 晚上

- 整理所有结果表；
- 确认所有数字一致性；
- 绘制关键图表。

---

## Day 4–5：论文重写 + 图表 + 最终检查

Agent 可以在数小时内完成全部写作。不需要再大规模跑实验，除非补缺失数据。

### 必须完成的写作改动

1. Abstract 重写（含真实实验数字）；
2. Introduction 重写（新叙事结构）；
3. Contributions 重写（四条新 contributions）；
4. 新增 Auto Research section（主文，~1.5 页）；
5. 新增 causal intervention section（主文，~1 页）；
6. 新增 measured efficiency section（可融入 Auto Research section）；
7. 新增 matched controls section（主文，~0.5 页）；
8. Discussion 降调 claim；
9. Limitation 更诚实但更主动；
10. Related work 加 Auto Research / AI Scientist / agent workflow 相关 work；
11. Appendix 加 task examples、judge prompt、logging details、chain breaking details。

### NeurIPS 页数适配

当前主文约 14 页（到 references 之前），NeurIPS 2026 通常限制 9 页主文（+ references + appendix 不限）。新增 ~3 页内容意味着需要大幅压缩。

建议：

- **移入 appendix**：当前 §5.3 Backbone Communication Typology 的详细分析、§5.5 Boundary Conditions 的大部分细节、length-only baseline 讨论；
- **保留在主文**：Auto Research 实验（~1.5 页）、causal intervention（~1 页）、matched controls（~0.5 页）；
- **压缩**：当前 §4 Experimental Setup 可以精简（部分实现细节移到 appendix）；
- **合并**：Cost profiling 融入 Auto Research section，不单独占 section。

### Day 5 检查清单

- [ ] 所有数字与实验结果一致
- [ ] 所有表格引用正确
- [ ] Abstract 中的 X% / Y% 替换为真实数字
- [ ] NeurIPS 格式检查（页数、字体、margin）
- [ ] Appendix 中补充完整的 task examples、judge prompt、logging schema
- [ ] 确认 contribution 写法与实际结果匹配（不 overclaim）

---

# 9. 新增图表清单

## Table 1: Causal Intervention (Ack Injection + Chain Breaking)

```markdown
| Intervention | Direction | N positions | P(next ack) | Next-3 Ack Rate | Chain Length | Accuracy / Quality Δ |
|---|---|---:|---:|---:|---:|---:|
```

## Table 2: AutoResearch Main Results (含 Cost Profiling)

```markdown
| Workflow | Model | Pruning | N | Quality Δ | Pairwise Win/Tie/Loss | Grounding Error Δ | Calls ↓ | Tokens ↓ | Wall-clock ↓ |
|---|---|---|---:|---:|---:|---:|---:|---:|---:|
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
→ Causal intervention (ack injection + chain breaking)
→ Matched controls
→ AutoResearch validation
→ Measured efficiency
```

## Figure 2: Mechanism Diagram

展示：

```text
single ack removal
→ high local trajectory perturbation

ack injection at non-ack position
→ causally increases future ack rate (cascade induction)

ack replacement with content at ack position
→ causally breaks cascade (chain breaking)

aggregate ack pruning
→ cascade dissolution
→ stable final answer + fewer calls
```

## Figure 3: Latency / Calls / Quality Tradeoff

x-axis: call reduction  
y-axis: quality delta  
points: different pruning policies (ACK, ACK+FLOW, random, position-matched, length-matched)  
annotations: Auto Research vs. GSM8K vs. StrategyQA

---

# 10. 新标题建议

## 推荐标题

**When Locally Disruptive Messages Are Globally Prunable: Functional Communication Analysis in LLM Agent Systems**

## 备选标题

**Local Disruption, Global Redundancy: Functional Message Pruning in LLM Agent Communication**

## 如果想突出 Auto Research

**Pruning Redundant Communication in LLM Research Agents via Functional Message Analysis**

推荐第一版，因为更 general，也更像 NeurIPS。从 "Two-Agent LLM Systems" 到 "LLM Agent Systems" 的升级前提是确实加了 3-agent Auto Research 实验。

---

# 11. 新摘要草案

```text
LLM agent systems often communicate through many low-content messages, such as acknowledgments and flow-control confirmations. Are such messages functionally necessary or merely structural artifacts of multi-agent interaction? We uncover a perturbation-pruning dissociation: acknowledgment messages are among the most disruptive when removed individually, yet can be safely pruned in aggregate. We first establish this phenomenon in two-agent solver-checker systems across Qwen and Llama backbones and multiple reasoning domains. We then validate it in a three-agent Auto Research workflow involving retrieval, synthesis, and critique, where acknowledgment pruning reduces LLM invocations by X% and measured wall-clock latency by Y% without degrading judge-rated answer quality or citation grounding. Controlled causal interventions—acknowledgment injection and chain breaking—show that acknowledgments causally induce self-reinforcing cascades, and matched non-ack controls confirm that pruning safety is function-specific rather than an artifact of message position or length. Lightweight learned detectors approximate oracle pruning across domains, reducing dependence on hand-crafted heuristics. Our results show that local perturbation sensitivity can overestimate global communication necessity, and provide a practical analysis protocol for identifying safely prunable message functions in LLM agent systems.
```

注意：X/Y 必须替换成真实实验结果。

---

# 12. Introduction 重写结构

## Paragraph 1: 真实 agent motivation

现代 LLM agent 系统通常由多个角色组成：planner、retriever、writer、critic、verifier。它们通过自然语言消息协作，但这些消息中有大量低内容确认、流程推进、重复性反馈。每条消息可能触发一次 LLM 调用，因此通信冗余会显著增加 latency 和 cost。

## Paragraph 2: 现有方法缺口

现有 efficiency 方法多关注 token compression、context compression、agent pruning、topology optimization、turn-level stopping，但较少分析 message function。尤其缺少对"某类消息是否功能必要"的系统性分析。

## Paragraph 3: 核心反直觉现象

直觉认为低信息量 acknowledgment 应该不重要。但单条删除实验显示它们最 disruptive；整体剪枝却显示它们可以安全删除。这说明 local perturbation sensitivity 不等于 global necessity。

## Paragraph 4: 机制假设

acknowledgment 形成 self-reinforcing cascade。单独删除会打断局部 autoregressive conditioning，因此造成轨迹扰动；整体删除则使 cascade 消失，不会累积错误。**因果实验（ack injection + chain breaking）为这一机制提供了干预性证据。**

## Paragraph 5: 本文贡献

列四条 contribution。

---

# 13. Discussion 改写原则

## 13.1 降调 claim

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

> Controlled interventions—ack injection and chain breaking—support cascade dissolution as the primary mechanism, though further work is needed to fully characterize its causal structure across architectures.

> **注意**：如果因果实验结果弱，则进一步降调为 "provide partial support for" 而非 "support"。

---

## 13.2 把 limitation 变成适用条件

当前 limitation：

- MATH 上退化；
- Qwen3 heuristic 不迁移；
- stochastic decoding 下 Llama 不稳；
- Gemma ack rate 低；
- debate / code-generation ack 少。

改写成：

> These boundary conditions suggest that pruning opportunity is topology-, domain-, and detector-dependent. The method is most useful when a workflow induces acknowledgment-rich feedback loops with low semantic novelty. It is less applicable to debate-like or code-generation topologies where acknowledgments are rare, or to domains where acknowledgments carry substantive reasoning content.

这样 reviewer 会觉得作者理解适用范围。

### 新增适用条件（来自 Auto Research 实验）

如果 Auto Research 的 ack rate 比 solver-checker 低：

> Auto Research workflows exhibit lower acknowledgment prevalence (~15-25%) than solver-checker systems (~50%), resulting in proportionally smaller but still meaningful call reductions. This confirms that pruning yield scales with acknowledgment prevalence, providing practitioners a simple diagnostic (measure ack rate) to predict deployment value.

---

# 14. 不建议五天内做的事情

## 14.1 不要做 4-agent topology probe

原方案建议 30 tasks 的 4-agent probe。但 N=30 的统计检验力太弱，无法支撑任何有意义的结论。3-agent Auto Research 已经足够证明 "beyond two-agent"。如果想展示 topology 的影响，用论文已有的 debate (0.2% ack) 和 code-generation (4% ack) 作为 negative examples 即可。

## 14.2 不要大规模跑 72B

2–4 张 5090 跑 72B 不划算。当前 72B probe 已经够作为 preliminary evidence（Δ=+0.0pp, N=265, p=1.000）。新增 AutoResearch 和因果实验更能涨分。

## 14.3 不要构造很大的新 benchmark

100 个高质量 AutoResearch tasks 足够。不要追求 1000 个粗糙任务。

## 14.4 不要扩 taxonomy 到十几类

五分类已经有一致性问题（information_provision κ=0.237）。扩 taxonomy 会制造新风险。

## 14.5 不要继续只加 GSM8K seed

当前 GSM8K 10-seed 已经比较强。继续加 seed 对分数提升有限。

## 14.6 不要过度优化 heuristic

heuristic non-transferability 已经被发现。更好的策略是转向 learned detector + empirical screening。

---

# 15. 最小可行版本

即使在 Agent Auto Research 高速执行下，仍需设定最低标准以防 GPU 故障或实验失败：

1. GSM8K 上的 ACK injection causal experiment（500 positions, Qwen）；
2. GSM8K 上的 chain breaking experiment（300 positions, Qwen）；
3. GSM8K 上的 position-matched non-ack control（500 tasks）；
4. AutoResearch 50 tasks，Qwen2.5-7B，baseline vs ACK pruning；
5. 报告 calls、tokens、wall-clock latency；
6. 做 LLM judge pairwise win/tie/loss。

只要这六项完成，论文就会比当前版本显著增强——因果机制和 confound control 是当前版本最大的 gap。

---

# 16. 理想完成版本

Agent Auto Research 高速执行下的理想五天产出：

1. ACK injection 因果实验（Qwen/GSM8K, 500 positions）；
2. Chain breaking 反向因果实验（Qwen/GSM8K, 300 positions）；
3. Position / length matched controls（Qwen/GSM8K, 500 tasks）；
4. AutoResearch 100 tasks, Qwen2.5-7B（baseline / ACK / ACK+FLOW）；
5. AutoResearch 50 tasks, Llama-3.1-8B（baseline / ACK）；
6. 内嵌 cost profiling（全部 Auto Research 实验自动记录）；
7. TF-IDF learned detector 跨域 online validation；
8. 完整论文重写（abstract, intro, contributions, 新 sections, discussion）；
9. NeurIPS 格式适配（9 页主文 + appendix）。

---

# 17. 预期分数提升

## 当前版本预期

- Quality: 2–3
- Clarity: 3
- Significance: 2–3
- Originality: 3
- Overall: 5 左右
- 倾向：borderline reject / weak reject

## 完成最小可行版本后

- Quality: 3–4（因果实验 + matched controls 显著提升）
- Clarity: 3
- Significance: 3
- Originality: 3
- Overall: 6
- 倾向：borderline accept

## 完成理想版本后

如果结果漂亮：

- ACK injection 显著增加 future ack rate（P(next ack) 提升 > 0.15）；
- Chain breaking 显著降低 future ack rate；
- Matched non-ack pruning 明显更差（accuracy drop > 2pp vs ack pruning）；
- AutoResearch call reduction > 20%；
- wall-clock reduction > 15%；
- judge tie + pruned win > 70%；
- citation grounding 不下降；

则可能达到：

- Quality: 4
- Significance: 3–4
- Overall: 6.5–7
- 倾向：weak accept / accept 边缘

---

# 18. 最终优先级总结

五天内真正应该做的是：

1. **用 ack injection + chain breaking 把 cascade dissolution 从相关性提升到因果干预支持。**（P0，直接回应论文最大弱点）
2. **把场景从 toy solver-checker 扩展到 Auto Research agent workflow，同时内嵌真实 latency / cost profiling。**（P0，同时解决 toy benchmark 和效率收益两个短板）
3. **用 matched controls 排除 position / length / random shortening confound。**（P1，排除 confound）
4. **用 lightweight learned detector 减少 oracle / heuristic 依赖。**（P2，补充 deployment story）
5. **重写论文叙事，升级 contributions，适配 NeurIPS 格式。**（P1，和新增证据量匹配的叙事升级）

**不做**：4-agent probe（统计检验力不足）、大规模 72B 实验、taxonomy 扩展。

最终论文应该传达的核心思想是：

> 单条消息的局部扰动敏感性不等于它的全局功能必要性。Acknowledgment 是这个现象的典型案例。因果实验表明 acknowledgment 会引发自强化 cascade，而整体剪枝使 cascade 消失。在 acknowledgment-rich 的 LLM agent feedback workflow（从 two-agent solver-checker 到 three-agent Auto Research）中，系统性剪掉这类低内容消息可以降低调用成本和实际延迟，同时保持最终输出质量。
