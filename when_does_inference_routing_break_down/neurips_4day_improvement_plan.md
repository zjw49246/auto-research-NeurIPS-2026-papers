# NeurIPS 2026 论文 4 天冲刺提升方案

> 目标：在 4 天内，利用 agent auto research 能力与 2–4 张 RTX 5090，把当前论文从“一个有趣但偏诊断性的单模型观察”提升为“有温度控制、跨模型验证、运行时路由启发和机制解释的方法学论文”，从而显著提高 NeurIPS 中稿概率。

---

## 0. 当前论文的核心判断

当前论文的核心发现是：在 depth scaling（单条长 CoT）与 width scaling（SC@4）比较中，标准协议通常使用不同 temperature：CoT 使用较低温度，SC 使用较高温度。这会让策略比较混入 decoding temperature 的影响。论文发现，在 Qwen3-32B 上，standard protocol 下若干 domain 似乎出现了从 CoT 到 SC 的 high-budget strategy reversal；但 matched temperature 后，大多数 reversal 消失，只有 MMLU-Pro 仍然保留 SC 优势。

这个 idea 是有价值的，但当前版本仍有几个明显风险：

1. **单模型风险**：主要证据来自 Qwen3-32B，外推性不足。
2. **单 matched temperature 风险**：只做 τ=0.8 matched control，容易被质疑 cherry-pick。
3. **routing framing 不够实**：论文说 routing breakdown，但主要实验证明的是 strategy comparison confound，而不是真正 router 的 failure。
4. **deduplication confound**：16k CoT 的 dedup 对 competition math 影响明显，可能被认为引入新 artifact。
5. **机制解释不够直接**：MMLU-Pro 为什么保留 SC 优势，reasoning-heavy tasks 为什么没有保留，需要更直接的 vote/error-correlation evidence。
6. **multi-hop QA near-floor**：multi-hop QA 准确率接近 floor，不能承载强主张。

因此，4 天内的提升目标不是做“大而全”的实验，而是补最能打掉审稿人主要质疑的证据。

---

## 1. 新论文主线

建议把论文主线从：

> Inference routing breaks down at high budgets, and temperature matching reveals that most reversals are artifacts.

升级为：

> Static strategy-only inference routing is under-specified because strategy choice is entangled with decoding temperature and runtime generation state. A strategy × temperature analysis shows that many apparent depth-vs-width reversals are artifacts of temperature asymmetry, while the remaining width advantage is better explained by answer format, vote efficiency, and error correlation. Runtime signals such as repetition onset provide a lightweight path toward more reliable fallback routing.

这个新主线比原文更强，因为它不仅指出问题，还给出可行方向：

- router 不应该只在 `{CoT, SC}` 中选择；
- router 应该考虑 `{strategy, temperature, k, runtime state}`；
- static surface-feature router 可能不足；
- runtime repetition signal 可能是更有效的 routing/fallback cue。

---

## 2. 最终优先级

| 优先级 | 模块 | 目的 | 对中稿概率提升 |
|---|---|---|---|
| P0 | Strategy × temperature sweep | 打掉 “τ=0.8 cherry-pick” 质疑 | 极高 |
| P0 | Runtime repetition-based routing proof-of-concept | 从纯诊断升级为诊断 + 可行方向 | 极高 |
| P0 | Raw vs dedup + threshold sensitivity | 封堵 dedup artifact 质疑 | 高 |
| P0 | Cross-model mini replication | 缓解单模型质疑 | 高 |
| P1 | Vote efficiency / error correlation | 强化 domain-dependent mechanism | 高 |
| P1 | Routing transfer experiment | 让 routing breakdown framing 更成立 | 中高 |
| P1 | Multi-hop QA 重新定位或轻量聚合改进 | 避免 near-floor domain 拖累主线 | 中 |
| P2 | 写作重构与图表优化 | 降低过强 claim，提升可读性 | 高 |
| P3 | 2k/8k budget、更多 benchmark、更多模型 | 锦上添花 | 中 |

---

# 3. P0 实验设计

---

## 3.1 P0.1 Strategy × Temperature Sweep

### 目的

当前 matched-temperature control 只使用 τ=0.8。审稿人很可能质疑：

> 你们是不是选择了一个对 CoT 有利的 temperature？如果换成 τ=0.6 或 τ=1.0，结论还成立吗？

因此需要做完整的 strategy × temperature sweep，把核心 claim 从“单点 control”升级为“interaction analysis”。

### 实验设置

```yaml
model: Qwen3-32B
budget: 16k
strategies:
  - CoT
  - SC@4
temperatures:
  - 0.2
  - 0.4
  - 0.6
  - 0.8
  - 1.0
domains:
  primary:
    - Competition Math
    - GPQA Diamond
    - MMLU-Pro
  optional:
    - Multi-hop QA
seeds:
  full_sweep: 1
  key_points:
    temperatures: [0.6, 0.8]
    seeds: [42, 123, 456]
```

### 必须记录的指标

每个 run 需要记录：

1. accuracy；
2. generated token count；
3. effective reasoning length；
4. repetition onset position；
5. repetition rate；
6. answer extraction success rate；
7. SC vote entropy；
8. SC oracle@4；
9. SC vote@4。

### 主要表格

**Table: Strategy × Temperature Accuracy at 16k**

| Domain | Strategy | τ=0.2 | τ=0.4 | τ=0.6 | τ=0.8 | τ=1.0 | Best τ |
|---|---|---:|---:|---:|---:|---:|---:|
| Math | CoT | | | | | | |
| Math | SC@4 | | | | | | |
| GPQA | CoT | | | | | | |
| GPQA | SC@4 | | | | | | |
| MMLU-Pro | CoT | | | | | | |
| MMLU-Pro | SC@4 | | | | | | |

### 主要图

**Figure: Accuracy vs Temperature**

每个 domain 一个子图：

- x-axis: temperature；
- y-axis: accuracy；
- CoT 一条线；
- SC@4 一条线。

**Figure: Repetition Rate vs Temperature**

用于证明 CoT 的温度敏感性与 long-trace degeneration 有关。

### 预期结论

最理想的结果：

1. Math / GPQA 上 CoT 对 temperature 很敏感，τ=0.8 明显优于 τ=0.6；
2. SC@4 对 temperature 相对不敏感，或者提升幅度小于 CoT；
3. standard protocol 下的 SC advantage 在 sweep 中不稳定；
4. MMLU-Pro 上 SC@4 在多数 temperature 下仍优于 CoT，说明它是真实 width advantage。

### 论文中的推荐写法

> The standard protocol does not merely compare depth and width; it compares two different strategy-temperature recipes. On reasoning-heavy domains, the strategy × temperature interaction explains most of the apparent width advantage. On MMLU-Pro, however, SC remains favorable across temperature controls, suggesting a genuine answer-format and vote-denoising effect.

---

## 3.2 P0.2 Runtime Repetition-Based Routing Proof-of-Concept

### 目的

当前论文容易被认为是纯诊断性工作：指出 routing evaluation 有问题，但没有给出解决方向。为了提高 NeurIPS 中稿概率，建议加入一个轻量级 runtime routing proof-of-concept。

它不需要重新大量跑推理，可以主要使用已有 16k CoT 输出。

### 核心思想

静态 surface features 很难判断一个问题应该走 CoT 还是 SC，但生成时状态可能包含更强信号。例如，当长 CoT 出现 repetition loop 时，继续 depth scaling 已经浪费 token，应该 early stop 并切换到 width sampling。

### Runtime features

在已有 CoT 输出上计算：

```text
1. sliding-window n-gram repetition rate
2. unique token ratio
3. repeated sentence ratio
4. repeated paragraph ratio
5. effective reasoning length before repetition onset
6. repetition onset token index
7. whether final answer appears before repetition onset
8. local entropy proxy if logprobs are available
9. output length saturation ratio
```

### 任务 A：预测 CoT failure

Label：

```text
1 if CoT answer is wrong
0 if CoT answer is correct
```

模型：

- logistic regression；
- random forest / XGBoost 可选；
- threshold-only repetition rule。

报告：

| Feature Set | AUC | PR-AUC | Accuracy | Notes |
|---|---:|---:|---:|---|
| Surface features | | | | static baseline |
| Repetition features | | | | runtime signal |
| Surface + repetition | | | | combined |

### 任务 B：预测 discordant-pair strategy preference

只在 CoT 和 SC@4 discordant pairs 上评估。

Label：

```text
1 if SC correct and CoT wrong
0 if CoT correct and SC wrong
```

报告：

| Feature Set | AUC on Discordant Pairs | Accuracy | Regret Reduction |
|---|---:|---:|---:|
| Surface LR router | | | |
| Repetition router | | | |
| Combined router | | | |

### 任务 C：Early fallback 离线模拟

策略：

```text
Start with CoT.
If repetition score crosses threshold before budget is exhausted:
    stop CoT
    fallback to SC@4 answer from existing outputs
else:
    use CoT answer
```

这是离线 proxy simulation，不要包装成完整在线系统。

报告：

| Policy | Accuracy | Cost Proxy | Regret vs Oracle | Notes |
|---|---:|---:|---:|---|
| Always CoT | | | | |
| Always SC@4 | | | | |
| Static LR router | | | | |
| Repetition-trigger fallback | | | | proof-of-concept |
| Oracle instance router | | | | upper bound |

### 论文中的推荐写法

> This proof-of-concept is not intended as a complete router. Instead, it shows that runtime generation signals contain routing information unavailable to static surface-feature routers, suggesting that future inference routers should monitor generation state rather than predict strategy solely from the input.

### 风险控制

如果 repetition router 效果一般，也可以写：

> Runtime repetition features improve failure detection but do not fully close the oracle gap, indicating that runtime-aware routing is promising but requires richer state signals than n-gram repetition alone.

这样仍然是正面结果。

---

## 3.3 P0.3 Raw vs Dedup + Threshold Sensitivity

### 目的

当前论文的 deduplication 是高风险点。尤其在 competition math 上，dedup 可能导致明显 correct→wrong flip。如果不透明处理，审稿人可能认为 standard-protocol reversal 受到 dedup artifact 影响。

### 必做分析

1. raw CoT accuracy；
2. dedup CoT accuracy；
3. correct→wrong flips；
4. wrong→correct flips；
5. repetition threshold sensitivity；
6. flip case audit。

### Raw vs dedup 表格

| Domain | Raw CoT | Dedup CoT | Δ | correct→wrong | wrong→correct | Main conclusion depends on dedup? |
|---|---:|---:|---:|---:|---:|---|
| Competition Math | | | | | | |
| GPQA Diamond | | | | | | |
| MMLU-Pro | | | | | | |
| Multi-hop QA | | | | | | |

### Threshold sensitivity

使用三个阈值：

```text
20-gram overlap
30-gram overlap
50-gram overlap
```

或者当前检测器有多个参数，则选择三个最关键参数做 sensitivity。

| Threshold | Repetition Rate | Median Effective Tokens | CoT Accuracy | Regime Changed? |
|---|---:|---:|---:|---|
| 20-gram | | | | |
| 30-gram | | | | |
| 50-gram | | | | |

### Flip case audit

随机抽 50 个受 dedup 影响的样本，分类：

| Category | Meaning | Count |
|---|---|---:|
| true repetition before final answer | dedup likely valid | |
| repetition after final answer | answer should be preserved | |
| false positive repetition | detector error | |
| answer lost by truncation | extraction artifact | |
| incoherent loop | valid truncation | |

### 论文中的推荐写法

> Because repetition detection can itself affect answer extraction, we treat deduplication as a diagnostic rather than as the sole basis for any main conclusion. Our matched-temperature and temperature-sweep results do not rely on dedup-induced reversals.

---

## 3.4 P0.4 Cross-Model Mini Replication

### 目的

缓解单模型质疑。不需要完整复现所有实验，只需要证明 temperature confounding 不是单个 Qwen3-32B run 的偶然现象。

### 推荐模型

优先选择容易在 2–4 张 5090 上跑的模型：

```text
1. Qwen3-32B: main model
2. Qwen3-8B / Qwen2.5-14B: same-family smaller model
3. DeepSeek-R1-Distill-Qwen-14B or 32B: reasoning/distilled model
4. Optional: Llama-3.1-8B or quantized 70B
```

### 实验范围

为了控制时间，只跑两个关键 domain：

```text
Competition Math
MMLU-Pro
```

设置：

```yaml
budget: 16k
protocols:
  standard:
    CoT_temperature: 0.6
    SC_temperature: 0.8
  matched:
    CoT_temperature: 0.8
    SC_temperature: 0.8
samples_per_domain: 100-200
seeds: 1
```

### 输出表格

| Model | Domain | Standard Δ(SC-CoT) | Matched τ=0.8 Δ | Gap Reduction | Verdict |
|---|---|---:|---:|---:|---|
| Qwen3-32B | Math | | | | |
| Qwen-small | Math | | | | |
| R1-Distill | Math | | | | |
| Qwen3-32B | MMLU-Pro | | | | |
| Qwen-small | MMLU-Pro | | | | |
| R1-Distill | MMLU-Pro | | | | |

### 结果解释策略

#### 如果趋势一致

> The temperature confound is not an isolated artifact of Qwen3-32B; we observe the same qualitative gap reduction on additional models.

#### 如果趋势不一致

> The magnitude of temperature confounding is model-dependent. Reasoning-tuned models appear less prone to long-trace degeneration, which clarifies the boundary condition of our claim rather than invalidating it.

---

# 4. P1 实验设计

---

## 4.1 P1.1 Vote Efficiency / Error Correlation Analysis

### 目的

当前论文认为 MMLU-Pro 保留 genuine SC advantage，而 Math / GPQA 不保留。这需要更直接机制证据。

### 核心假设

- MMLU-Pro：structured multiple-choice answer，sample errors 更独立，majority vote 更有效。
- Math / GPQA：reasoning-heavy，错误更相关，多个 samples 往往错在类似方向，SC@4 只能追平 CoT。
- Multi-hop QA：free-text answer fragmentation 导致 oracle signal 无法被 vote 聚合。

### 计算指标

对 SC@4 的四个 samples 计算：

```text
SC@1
SC@2
SC@3
SC@4 vote accuracy
Oracle@4 = any sample correct
Oracle gap = Oracle@4 - Vote@4
Vote entropy
Pairwise answer agreement
Correct plurality rate
Wrong-answer clustering
```

### 主表

| Domain | SC@1 | Vote@4 | Oracle@4 | Oracle Gap | Vote Entropy | Pairwise Agreement | Correct Plurality Rate | Mechanism |
|---|---:|---:|---:|---:|---:|---:|---:|---|
| Math | | | | | | | | correlated reasoning errors |
| GPQA | | | | | | | | correlated / mixed errors |
| MMLU-Pro | | | | | | | | vote denoising works |
| Multi-hop QA | | | | | | | | free-text fragmentation |

### 推荐图

**Figure: Vote Efficiency vs SC Gain**

- x-axis: vote efficiency or oracle gap；
- y-axis: SC@4 gain over CoT；
- each point: domain。

### 论文中的推荐写法

> The residual SC advantage is better predicted by voting efficiency than by token budget alone. MMLU-Pro converts oracle coverage into voted accuracy, whereas reasoning-heavy domains exhibit correlated failures and free-text QA loses oracle signal during aggregation.

---

## 4.2 P1.2 Routing Transfer Experiment

### 目的

让 “routing instability / routing breakdown” framing 更成立。当前论文主要比较 strategies，还需要展示 router label 在 budget / temperature shift 下是否真的 mislead。

### 设置

| Setting | Train Label From | Test Reward From | Purpose |
|---|---|---|---|
| A | standard protocol 1k | standard protocol 16k | budget transfer |
| B | standard protocol 16k | matched τ=0.8 16k | temperature-confound misrouting |
| C | matched τ=0.8 16k | matched τ=0.8 16k | residual routing signal |
| D | domain-only | matched τ=0.8 16k | domain-level signal baseline |

### Routers

```text
1. Always CoT
2. Always SC@4
3. Static surface-feature LR router
4. Domain router
5. Runtime repetition router
6. Oracle instance router
```

### 输出表

| Router | Standard 16k Acc | Matched 16k Acc | Regret vs Oracle | Misroute Rate |
|---|---:|---:|---:|---:|
| Always CoT | | | | |
| Always SC@4 | | | | |
| Static LR router | | | | |
| Domain router | | | | |
| Runtime repetition router | | | | |
| Oracle router | | | | |

### 推荐结论

> Static strategy labels learned under one budget or temperature protocol do not transfer reliably. Most residual signal after temperature control is captured by domain/answer-format and runtime generation state, not by shallow surface features alone.

---

## 4.3 P1.3 Multi-hop QA 重新定位或轻量改进

### 问题

Multi-hop QA 当前准确率 near-floor，因此不适合和 Math / GPQA / MMLU-Pro 并列支撑强结论。

### 推荐做法 A：重新定位，低成本

在 Method 中明确：

> We treat multi-hop QA as an exploratory stress test for free-form answer aggregation rather than as a primary domain for strategy-superiority claims.

在 Results 中只用它说明：

1. free-text voting difficulty；
2. oracle-vote gap；
3. answer-space fragmentation。

### 推荐做法 B：轻量 semantic aggregation

如果有时间，补一个 answer clustering baseline：

```text
1. exact string voting
2. normalized string voting
3. embedding similarity clustering
4. LLM equivalence clustering
```

表格：

| Aggregation | Vote Accuracy | Oracle@4 | Oracle Gap |
|---|---:|---:|---:|
| Exact voting | | | |
| Normalized voting | | | |
| Embedding clustering | | | |
| LLM equivalence clustering | | | |

### 推荐写法

> Because the task operates near floor, we avoid using multi-hop QA as evidence for strategy superiority. Instead, it serves as a diagnostic case showing that free-form answer spaces can destroy much of the oracle signal produced by width scaling.

---

# 5. 写作重构方案

---

## 5.1 标题调整

当前标题 “When Does Inference Routing Break Down?” 容易被认为过大。建议改为更精确的版本。

### 推荐标题 1

```text
When Does Depth-vs-Width Routing Fail?
Temperature Confounding and Runtime Signals in Inference-Time Scaling
```

### 推荐标题 2

```text
Disentangling Strategy, Temperature, and Runtime Degeneration in Inference Routing
```

### 推荐标题 3

```text
Inference Routing Is Under-Specified:
Temperature Confounding in Depth-vs-Width Test-Time Scaling
```

---

## 5.2 Abstract 重写方向

Abstract 不要堆太多 p-value 和 domain 数字。建议结构：

1. 背景：inference routers assume stable strategy preference；
2. 问题：standard protocols entangle strategy and temperature；
3. 方法：matched controls + temperature sweep + cross-model mini replication；
4. 发现：most reasoning-domain reversals disappear or shrink；MMLU-Pro remains genuine；
5. 机制：repetition ceiling + vote efficiency + error correlation；
6. 启示：route over strategy-temperature recipes and runtime signals。

### 推荐 abstract 草稿

```text
Inference routers that choose between depth scaling and width scaling typically treat the optimal strategy as a stable property of each input. We show that this assumption is under-specified: standard evaluations conflate strategy choice with decoding temperature, assigning lower temperature to long chain-of-thought traces and higher temperature to self-consistency samples. Through matched-temperature controls and a strategy × temperature sweep on Qwen3-32B, we find that apparent high-budget reversals from CoT to SC largely disappear on reasoning-heavy domains, while MMLU-Pro retains a genuine width advantage. Mini cross-model replication shows that the magnitude of this confound is model-dependent but not isolated to a single run. Mechanistic analyses reveal that residual width gains are predicted by vote efficiency and answer format, whereas long CoT failures are associated with repetition-induced degeneration. Finally, a lightweight runtime-routing proof-of-concept shows that repetition signals contain routing information unavailable to static surface-feature routers. These results suggest that inference routers should select over strategy-temperature recipes and runtime states, not strategies alone.
```

---

## 5.3 Contributions 重写

建议从五条压缩为四条，每条更有分量。

### 新贡献列表

1. **Temperature confounding in depth-vs-width routing.**  
   We identify a systematic confound in standard inference-routing evaluations: strategy choice is entangled with decoding temperature.

2. **Factorial controls and cross-model evidence.**  
   We use matched-temperature controls, strategy × temperature sweeps, and mini cross-model replication to separate artifact-driven reversals from genuine width advantages.

3. **Mechanism: repetition ceiling and vote efficiency.**  
   We show that residual domain differences are explained by repetition-induced depth ceilings, answer format, error correlation, and majority-vote efficiency.

4. **Runtime-aware routing implication.**  
   We provide a lightweight proof-of-concept showing that runtime repetition signals can improve fallback routing beyond static surface-feature routers.

---

## 5.4 Discussion 重写重点

Discussion 应该避免过强 claim。

### 不建议写

```text
Inference routing breaks down at high budgets.
```

### 建议写

```text
Static strategy-only routing labels become unstable under budget and temperature shifts. This does not imply that routing is impossible; rather, it shows that routing over strategies alone is under-specified. A more faithful action space should include decoding temperature, sample count, answer format, and runtime generation state.
```

### 新 Discussion 小节建议

1. **Temperature as part of the routing action space**
2. **Static input features vs runtime generation signals**
3. **When width helps: vote efficiency and answer structure**
4. **When depth fails: repetition-induced saturation**
5. **Limitations of free-text majority voting**

---

## 5.5 Limitations 重写

主动承认限制，降低审稿攻击力度。

建议包括：

1. Main experiments still focus on Qwen3-32B；cross-model replication is mini-scale。
2. Temperature sweep covers selected domains and 16k budget only。
3. Runtime fallback is an offline proof-of-concept, not a full online router。
4. Multi-hop QA is exploratory due to near-floor performance。
5. Deduplication can affect answer extraction, so main conclusions are reported with raw/dedup sensitivity。
6. SC aggregation methods are simple; learned verifiers or semantic clustering may alter free-text results。

---

# 6. 图表重构清单

## 主文 Figure 1：Standard vs Matched Summary

展示 standard protocol 下的 apparent reversal 与 matched-temperature 后的 residual effect。

推荐形式：

- 每个 domain 一个 bar group；
- left: standard Δ(SC-CoT)；
- right: matched Δ(SC-CoT)；
- 标注 significant / non-significant。

## 主文 Figure 2：Temperature Sweep

- x-axis: temperature；
- y-axis: accuracy；
- CoT vs SC@4；
- Math / GPQA / MMLU-Pro 三个子图。

## 主文 Figure 3：Repetition vs Temperature

展示 CoT repetition rate 随 temperature 变化，支撑 temperature rescues / changes long-trace degeneration 的机制。

## 主文 Figure 4：Vote Efficiency Mechanism

展示 SC@4、Oracle@4、Oracle gap 或 vote efficiency。

## 主文 Figure 5：Runtime Routing Proof-of-Concept

展示 repetition features 相比 surface features 的 AUC / regret reduction。

## 主文 Table 1：Consolidated Strategy Comparison

| Domain | Standard Dominant | Standard Δ | Matched Dominant | Matched Δ | n_discordant | p-value | Verdict |
|---|---|---:|---|---:|---:|---:|---|
| Math | | | | | | | Artifact / Tie |
| GPQA | | | | | | | Artifact / Tie |
| MMLU-Pro | | | | | | | Genuine SC |
| Multi-hop QA | | | | | | | Exploratory |

## Appendix Table：Raw vs Dedup

详见 P0.3。

## Appendix Table：Cross-Model Mini Replication

如果结果强，可以放主文；如果结果 noisy，可以放 appendix，但 Introduction / Discussion 中简要引用。

---

# 7. 4 天执行排期

---

## Day 1：实验管线和 P0 实验启动

### 上午：统一 pipeline

完成：

1. config registry；
2. generation runner；
3. answer extractor；
4. SC aggregator；
5. repetition feature extractor；
6. dedup sensitivity script；
7. result summarizer。

建议每个 generation output 保存为 jsonl：

```json
{
  "model": "Qwen3-32B",
  "domain": "MATH",
  "strategy": "CoT",
  "temperature": 0.8,
  "budget": 16000,
  "seed": 42,
  "problem_id": "...",
  "prompt": "...",
  "raw_output": "...",
  "extracted_answer": "...",
  "gold_answer": "...",
  "correct": true,
  "num_output_tokens": 1234,
  "repetition_onset": 980,
  "dedup_output": "..."
}
```

### 下午：启动 Qwen3-32B temperature sweep

优先跑：

```text
Competition Math / GPQA / MMLU-Pro
CoT / SC@4
τ = 0.2, 0.4, 0.6, 0.8, 1.0
16k budget
```

GPU 分配建议：

- GPU 1：Math CoT sweep；
- GPU 2：Math SC sweep；
- GPU 3：MMLU-Pro CoT/SC；
- GPU 4：GPQA CoT/SC。

### 晚上：启动 cross-model mini replication

```text
Qwen-small + R1-Distill
Math + MMLU-Pro
standard vs matched τ=0.8
100–200 samples/domain/model
```

同时开始从已有 outputs 中计算 runtime repetition features。

---

## Day 2：完成 P0 结果

### 上午：temperature sweep 汇总

生成：

1. accuracy vs τ table；
2. accuracy vs τ plot；
3. repetition rate vs τ plot；
4. effective length vs τ plot。

检查：

- CoT 是否对 temperature 更敏感；
- SC 是否相对稳定；
- MMLU-Pro 是否保留 SC advantage；
- 是否存在 τ=1.0 degeneration。

### 下午：cross-model mini replication 汇总

生成 cross-model table。

如果时间不够，不扩 domain，优先补齐 Math + MMLU-Pro 的样本。

### 晚上：raw vs dedup + threshold sensitivity

完成：

1. raw/dedup table；
2. threshold sensitivity table；
3. correct→wrong / wrong→correct flip list；
4. 50-case audit summary。

---

## Day 3：机制分析与 routing proof-of-concept

### 上午：runtime routing proof-of-concept

完成：

1. CoT failure prediction；
2. discordant-pair strategy preference prediction；
3. early fallback offline simulation；
4. surface features vs repetition features 对比。

### 下午：vote efficiency / error correlation

完成：

1. SC@k curves；
2. Oracle@4；
3. vote entropy；
4. pairwise agreement；
5. correct plurality rate；
6. mechanism table。

### 晚上：routing transfer experiment

完成：

1. always CoT；
2. always SC；
3. static LR router；
4. domain router；
5. runtime repetition router；
6. oracle router。

输出 routing table。

---

## Day 4：写作重构和审稿模拟

### 上午：重写主文

重点重写：

1. title；
2. abstract；
3. introduction；
4. contributions；
5. method；
6. results；
7. discussion；
8. limitations。

### 下午：图表整合

完成主文图表：

1. standard vs matched summary；
2. temperature sweep；
3. repetition vs temperature；
4. vote efficiency；
5. runtime routing proof-of-concept；
6. consolidated comparison table。

Appendix 加入：

1. raw vs dedup；
2. threshold sensitivity；
3. cross-model details；
4. prompt templates；
5. extraction rules；
6. seed-level results。

### 晚上：模拟审稿

让 agent 生成 3 个 reviewer：

1. LLM evaluation / statistics reviewer；
2. inference-time scaling reviewer；
3. routing / systems reviewer。

每个 reviewer 输出：

- summary；
- strengths；
- weaknesses；
- questions；
- score；
- confidence。

然后人工检查：

1. 哪些 claim 仍然过强；
2. 哪些表格不自解释；
3. 哪些实验没有支撑对应结论；
4. 哪些 limitation 应主动承认。

---

# 8. 如果实验结果不符合预期，如何调整叙事

---

## 情况 A：temperature sweep 完全支持主线

表现：

- Math / GPQA 上 CoT τ=0.8 明显恢复；
- standard SC advantage 消失；
- MMLU-Pro across τ 仍然 SC wins。

写法：

> Temperature asymmetry creates spurious width advantages on reasoning-heavy domains, while structured knowledge tasks retain genuine width benefits.

论文评分预期：6–7。

---

## 情况 B：部分模型不复现

表现：

- Qwen3-32B 支持；
- R1-Distill 上 CoT 不太受 temperature 影响。

写法：

> The confound is strongest in models exhibiting long-trace degeneration. Reasoning-tuned models reduce but do not eliminate the need for strategy-temperature controls. This identifies a boundary condition rather than invalidating the finding.

论文评分预期：仍可 6 左右，甚至更 nuanced。

---

## 情况 C：SC 在很多 temperature 下都赢

表现：

- Matched sweep 后 SC 仍然赢多个 domain。

改写：

> Temperature asymmetry inflates and miscalibrates width advantages, but does not always fully explain them.

主贡献从 “reversal is artifact” 改成：

> unmatched protocols overestimate or misattribute strategy effects.

---

## 情况 D：runtime repetition router 效果弱

表现：

- repetition feature AUC 只有小幅提升。

写法：

> Simple repetition signals improve over static surface features but remain far from oracle routing, suggesting that richer runtime states are needed.

不要把它写成主要算法贡献，写成 proof-of-concept。

---

## 情况 E：dedup 对结果影响巨大

表现：

- standard reversal 强依赖 dedup。

写法：

> We do not rely on deduplicated standard-protocol reversals for the main conclusion. The central evidence comes from raw matched-temperature and temperature-sweep comparisons.

主动透明会降低审稿风险。

---

# 9. 最小必做版本

如果 4 天内时间严重不足，只做下面 5 件事：

1. **Qwen3-32B temperature sweep**  
   必须做。它是最能提升核心 claim 的实验。

2. **Runtime repetition-based routing proof-of-concept**  
   必须做。它把论文从 diagnostic 升级为 actionable。

3. **Raw vs dedup + threshold sensitivity**  
   必须做。它封堵一个明显方法学漏洞。

4. **Cross-model mini replication**  
   至少一个额外模型，Math + MMLU-Pro。

5. **Vote efficiency / error correlation analysis**  
   用已有 SC outputs 计算，成本低、收益高。

如果只做这五件，并重写 abstract / intro / contributions / limitations，论文质量会有明显提升。

---

# 10. 预期中稿概率提升

当前版本大概是：

```text
Overall: 5/10 或 6/10
倾向: Weak Reject / Borderline
```

主要原因：idea 好，但 evidence scope 不够，claim 有些过强。

完成本方案后，如果实验结果大体支持主线：

```text
Overall: 6/10 或 7/10
倾向: Borderline Accept / Weak Accept
```

提升最大的原因：

1. temperature sweep 让核心发现从单点 control 变成完整 interaction analysis；
2. cross-model mini replication 缓解单模型质疑；
3. runtime routing proof-of-concept 增强 actionable contribution；
4. dedup sensitivity 提升方法学可信度；
5. vote efficiency 让 domain-dependent explanation 更像机制而不是 speculation。

---

# 11. 最终建议

这篇论文不是差在 idea，而是差在证据链厚度和贡献定位。4 天内不应该追求“更多 benchmark / 更多 budget / 更复杂 router”，而应该集中火力补最能改变审稿判断的实验：

```text
strategy × temperature sweep
+ runtime repetition routing proof-of-concept
+ raw/dedup sensitivity
+ cross-model mini replication
+ vote efficiency mechanism
```

写作上，最关键的是把 claim 从：

```text
routing breaks down
```

改成：

```text
static strategy-only routing is under-specified under temperature and budget shifts;
routing should account for strategy-temperature recipes and runtime generation state.
```

这个叙事更稳、更有建设性，也更符合 NeurIPS 对 methodological empirical papers 的期待。
