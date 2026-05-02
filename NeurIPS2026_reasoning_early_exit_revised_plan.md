# NeurIPS 2026 论文叙事与实验改进方案（修订版）

**论文：** *The Convergence Cost of Reasoning: Why Per-Token Early Exit Underperforms*  
**目标：** 在投稿前三天内，以最高性价比提升论文可信度、叙事清晰度和审稿通过概率。  
**建议定位：** Negative Results / Empirical Analysis / Diagnostic Study，而不是主要的新 early-exit 方法论文。

---

## 0. 总体判断

这篇论文的核心 idea 是有价值的：它研究的是 **per-token early exit 在长 reasoning traces 上为什么会失败**。当前论文已经有大量实验，但最大问题不是实验数量不足，而是：

1. **主叙事不够清楚。** Abstract、Introduction、Contribution 太像实验结果堆叠，而不是围绕一个研究问题逐步展开。
2. **核心因果链条不够干净。** 主实验大量依赖 DeepSeek-R1 distilled models，导致 reasoning effect、distillation effect、teacher style 和 trace distribution 混在一起。
3. **实验角色不清。** 有些实验在证明 early exit 失败，有些在解释机制，有些在做 robustness，有些在补 confound，但没有形成清晰实验树。
4. **部分 claim 过强。** 例如 “distillation-specific”、“reasoning-general”、“minimum-capacity threshold”、“current reasoning models” 等表达超过了当前实验设计能支撑的范围。
5. **指标命名容易误导。** 当前 Q_Acc 容易被误解为最终任务正确率，但很多地方其实是整条 trace 与 full-depth trace 的 exact match。

因此，三天内最重要的不是继续盲目堆模型和 benchmark，而是：

> **修复主线、修复 confound、修复 split、修复指标命名、弱化 claim。**

如果还有余力，再补一个最小 actual rollout 实验；如果没有余力，就把论文明确定位为 diagnostic study，不要过强宣称 deployment-ready 或 deployment benchmark。

---

## 1. 最终推荐主线

建议把论文主线收束为：

> **Standard per-token early exit fails on long reasoning traces because token-level reliability does not translate into trace-level or task-level reliability. Native thinking controls show that this failure is not merely a distillation artifact. Distilled reasoning models further amplify oscillatory convergence, while native reasoning traces still exhibit representation immaturity. Tuned lens reveals a projection bottleneck, and learned gates partially recover performance but do not eliminate the compound-error constraint.**

中文版本：

> **标准 per-token early exit 在长 reasoning traces 上失败，是因为 token-level 可靠性无法直接转化为 trace-level 或 task-level 可靠性。Native thinking 对照说明这种失败不只是 distilled model artifact。Distilled reasoning models 会进一步放大 oscillatory convergence，而 native reasoning traces 中仍然存在 representation immaturity。Tuned lens 显示 raw projection 是重要瓶颈，learned gate 可以部分恢复，但无法消除长 trace 上的 compound error 限制。**

这个主线比原论文更稳健，因为它把结论分成三层：

1. **Reasoning traces 本身使 early exit 更难；**
2. **Distillation 可能额外放大一类 failure mode；**
3. **Projection / learned gating 能部分缓解，但不能解决长 trace compound error。**

---

## 2. 三天内真正应该做什么

### 2.1 执行原则

三天内不要试图做一个完整新论文。应该集中做五件事：

1. **把 native thinking control 提升为主实验；**
2. **修复 tuned lens / learned gate 的 train-val-test split；**
3. **重命名并统一指标；**
4. **弱化所有过强 causal / universal claims；**
5. **重写 Abstract、Introduction、Contributions。**

Actual rollout 是加分项，但不应无条件列为必须完成项。更合理的判断是：

> **如果论文保留 deployment claim，就必须做最小 rollout；如果论文定位为 diagnostic study，则 rollout 可以作为 limitation / future work。**

---

## 3. 优先级总表

| 优先级 | 任务 | 是否三天内必须完成 | 重要程度 | 主要收益 |
|---|---|---:|---:|---|
| P0 | Qwen3 non-thinking / native thinking / R1-Distill-Qwen 三方对照 | 是 | 极高 | 修复 distilled confound，重建主线 |
| P0 | tuned lens / gate nested split 澄清或重跑 | 是 | 极高 | 排除 data leakage 风险 |
| P0 | 指标重命名：Q_Acc → TraceExact 等 | 是 | 极高 | 避免 reviewer 认为指标误导 |
| P0 | 弱化 causal / universal claims | 是 | 极高 | 提升可信度，避免被抓住反驳 |
| P0 | 重写 Abstract / Introduction / Contributions | 是 | 极高 | 改善 clarity 和 paper framing |
| P0-submission | 修复 checklist、float、页数、匿名格式 | 是 | 极高 | 避免格式扣分或 desk rejection |
| Cond-P0 | 最小 actual rollout | 取决于是否保留 deployment claim | 高 | 支撑真实部署结论 |
| P1 | 扩大 softmax CALM n=50 到 n=300–500 | 有余力做 | 中高 | 增强主 baseline 可信度 |
| P1 | final-answer / boxed-answer analysis 前移主文 | 建议做 | 高 | 缓解 TraceExact 过严质疑 |
| P1 | length-bucket analysis | 有余力做 | 中 | 区分 length effect 和 reasoning dynamics |
| P1 | R1-Llama / Phi 补同协议 compact summary | 有余力做 | 中 | 增强跨架构证据 |
| P2 | failure taxonomy 小样本分析 | 可选 | 中 | 增强 mechanism 解释 |
| P2 | QwQ-32B full profiling | 不建议三天内做 | 中低 | 工程成本高，收益有限 |
| P2 | 新增 benchmark / 外部 baseline / wall-clock speedup | 不建议三天内做 | 中低 | 容易拖垮执行，且可能引入新噪声 |

---

## 4. P0-1：把 native thinking control 提升为主实验

### 4.1 为什么这是最重要的

当前论文主实验依赖 distilled reasoning models：

| 维度 | Qwen | Llama |
|---|---|---|
| Reasoning | R1-Distill-Qwen | R1-Distill-Llama |
| Non-reasoning | Qwen3 | Llama-3.1-Instruct |

这个设计的问题是：reasoning 条件同时混入了：

- reasoning behavior；
- DeepSeek-R1 teacher style；
- distillation objective；
- trace distribution；
- post-training / calibration difference。

所以它不能干净地证明 “reasoning 本身导致 early-exit failure”。

最有效的修复方式是把同一家族的 native thinking control 放到主文核心位置。

### 4.2 必须加入的三方对照

建议新增或前移一张主表：

**Table X: Native thinking-mode control separates reasoning effects from distillation effects.**

| Model | Condition | Avg tokens | TokenMatch | TraceExact | BoxedAcc / AnswerAcc | NM rate | Key interpretation |
|---|---|---:|---:|---:|---:|---:|---|
| Qwen3-8B | non-thinking | ... | ... | ... | ... | ... | ordinary generation / instruction baseline |
| Qwen3-8B | native thinking | ... | ... | ... | ... | ... | reasoning-style generation without distillation |
| R1-Distill-Qwen3-8B | distilled reasoning | ... | ... | ... | ... | ... | reasoning + distillation / teacher style |

### 4.3 这张表要回答的问题

| 对比 | 回答的问题 | 预期解释 |
|---|---|---|
| Qwen3 thinking vs Qwen3 non-thinking | thinking behavior 本身是否让 early exit 更难？ | 如果 thinking 更差，说明 failure 不是 distilled-only |
| R1-Distill-Qwen vs Qwen3 thinking | distillation 是否额外放大某些 dynamics？ | 如果 R1-Distill NM 更高，说明 oscillation 可能 distillation-associated |
| R1-Distill-Qwen vs Qwen3 non-thinking | 原始 reasoning vs non-reasoning gap | 作为旧 2×2 结果的桥接 |

### 4.4 应该如何改论文结论

不要说：

> Oscillatory dynamics are distillation-specific.

改成：

> Oscillatory dynamics are strongest in distilled reasoning models and are substantially reduced in the native Qwen thinking-mode control, suggesting an association with distillation or teacher-induced trace style.

不要说：

> Representation immaturity is reasoning-general.

改成：

> Representation immaturity persists in the native Qwen thinking-mode control, suggesting that this failure mode is not solely a distillation artifact.

---

## 5. P0-2：修复 tuned lens / learned gate split

### 5.1 当前风险

当前论文中 tuned lens 和 gate 的数据划分描述不够清楚，甚至可能让 reviewer 认为存在 leakage。尤其需要避免以下情况：

> 先在全部 500 个 MATH-500 questions 上 fit tuned lens，再切分 gate train/test。

如果 tuned lens 在 test questions 上拟合过，那么 learned gate 的 test performance 会被严重质疑。

### 5.2 推荐协议

采用严格 nested split：

```text
Dataset
  ├── Train: fit tuned lens + train gate
  ├── Val: choose thresholds / hyperparameters
  └── Test: report final metrics only
```

如果使用 K-fold CV，则每个 fold 必须重新执行：

```text
For each fold:
  1. Fit tuned lens on fold-train only
  2. Train gate on fold-train only
  3. Choose threshold on fold-val only
  4. Report on fold-test only
```

### 5.3 三天内怎么做

如果时间不够重跑全部实验，建议：

1. **主文只保留 clean nested split 的 R1-Qwen main result；**
2. 旧的 cross-architecture gate transfer 降级为 appendix / preliminary；
3. 所有表格 caption 明确说明：
   - tuned lens 是否在该 dataset 上训练；
   - gate 是否在该 model 上训练；
   - threshold 是否用 validation set 选择；
   - test set 是否完全 held out。

### 5.4 必须新增的表述

建议在方法部分加入：

```text
All tuned-lens transformations, gate parameters, and exit thresholds are fitted without access to test questions. In cross-validation experiments, the tuned lens and gate are re-trained within each fold; no affine transform fitted on held-out questions is used for final reporting.
```

---

## 6. P0-3：统一指标命名

### 6.1 当前问题

当前 Q_Acc 容易被误解为 benchmark final answer accuracy。但很多地方它实际表示：

> 整条 reasoning trace 的所有 token 是否与 full-depth trace 完全一致。

这个指标很严格，作为 diagnostic 是合理的，但不能直接叫 question accuracy，否则 reviewer 会质疑 misleading。

### 6.2 推荐术语

| 新术语 | 定义 | 用途 |
|---|---|---|
| TokenMatch | 某层预测 token 是否等于 final-layer token | token-level diagnostic |
| TraceExact | 整条 trace 所有 token 是否与 full-depth trace 完全一致 | strict trace-level diagnostic |
| BoxedAcc / BoxedExact | boxed answer 或 final answer segment 是否一致/正确 | answer-segment diagnostic |
| AnswerAcc / TaskAcc | benchmark final answer 是否正确 | task-level metric |
| RolloutTaskAcc | actual early-exit rollout 的 final answer accuracy | deployment metric |
| RolloutDivergence | actual rollout 第一次偏离 full-depth 的位置 | error propagation analysis |

### 6.3 建议主文写法

```text
We use TraceExact as a strict diagnostic metric: a trace is counted as correct only if every early-exit token matches the full-depth token. Because this metric is intentionally stringent, we also report answer-level metrics such as BoxedAcc and TaskAcc when evaluating downstream correctness.
```

### 6.4 改名收益

这个改动非常重要，因为它能提前化解一个常见 reviewer objection：

> “0% question accuracy 是不是只是因为 all-token exact match 太严格？”

改名后，论文可以更诚实地说：

> TraceExact collapse 显示 trace-level reliability 很差；BoxedAcc / AnswerAcc 显示这种问题是否真正影响最终答案。

---

## 7. P0-4：弱化 claim

### 7.1 必须替换的强 claim

| 当前可能的表述 | 风险 | 建议改写 |
|---|---|---|
| current reasoning models | 范围过大，且论文有 1.5B reversal | the reasoning models studied here |
| distillation-specific | 只有 Qwen native thinking 支撑 | distillation-associated in our controlled Qwen comparison |
| reasoning-general | 覆盖有限 | persists in both distilled and native Qwen reasoning settings |
| minimum-capacity threshold | 7B vs 8B 不是纯 scale 控制 | suggests a capacity- or architecture-dependent emergence |
| projection quality is the primary bottleneck | tuned lens 是诊断，不是因果证明 | projection quality is a major bottleneck under tuned-lens diagnostics |
| deployment constraint | 如果没有 rollout，过强 | diagnostic evidence suggests a deployment-relevant constraint |
| CALM failure persists at scale | 32B 只测 CALM | CALM-style diagnostics also fail on QwQ-32B |

### 7.2 建议新增 “Scope of claims” 小段

```text
Our strongest claims concern 8B–14B math/code reasoning traces under the evaluated Qwen, Llama, and Phi model families. Distilled models are used as important test cases but introduce teacher- and distillation-specific factors; native Qwen thinking-mode controls reduce but do not eliminate this concern. The 1.5B reversal and the diagnostic-only 32B experiments indicate that scale and training recipe can change the observed dynamics. We therefore frame the proposed mechanisms as empirically supported failure modes rather than universal laws.
```

中文含义：

> 我们最强的结论限于本文评估的 8B–14B 数学/代码 reasoning traces。Distilled models 是重要测试对象，但引入 teacher 和 distillation 因素。Native Qwen thinking control 能缓解但不能完全消除这个问题。1.5B reversal 和 32B diagnostic-only 结果说明 scale 和 training recipe 会改变 dynamics。因此这些机制应被表述为 empirical failure modes，而不是 universal laws。

---

## 8. P0-5：重写 Abstract / Introduction / Contributions

### 8.1 Abstract 当前问题

当前 Abstract 的问题是：

1. 数字太多；
2. 术语太密；
3. 直接上结论；
4. scope 太宽；
5. learned gate 和 diagnostic analysis 的定位混在一起。

### 8.2 建议 Abstract 草稿

```text
Per-token early exit is a natural way to reduce the inference cost of long reasoning traces, but its reliability in this setting remains poorly understood. We show that standard confidence- and similarity-based exit criteria break down on reasoning traces: high token-level agreement can still collapse to near-zero trace-level exactness because small per-token errors compound over thousands of generated tokens.

We identify two representation-level failure modes behind this collapse. Some tokens exhibit oscillatory hidden-state trajectories that create false convergence signals, while others appear confident but remain representationally immature until late layers. Native Qwen thinking-mode controls show that the early-exit deficit is not merely a distillation artifact, while distilled reasoning models amplify the oscillatory failure mode in our experiments.

Using tuned-lens diagnostics, we show that raw intermediate-layer projections substantially underestimate the information available in hidden states, revealing a large projection bottleneck. Learned exit gates partially recover trace-level reliability across reasoning model families, but compound error remains a central obstacle. These results suggest that efficient reasoning requires segment-level, verifier-guided, or training-aware exit mechanisms beyond simple per-token confidence thresholds.
```

### 8.3 Introduction 建议结构

建议扩展为 5–6 段：

1. **Reasoning models are expensive.** 长 CoT / thinking traces 提升能力，但推理成本高。
2. **Per-token early exit is a natural solution.** 解释 early exit 和已有 CALM / LayerSkip 动机。
3. **Reasoning traces are different.** 长、结构化、token role 异质、错误会累积。
4. **Research question.** Standard per-token early exit 在 reasoning traces 上是否可靠？如果失败，为什么？
5. **Main findings.** CALM-style criteria fail；native thinking control 显示不是 distilled-only；distillation 放大 oscillation；tuned lens 显示 projection bottleneck；learned gate 部分恢复。
6. **Contributions.** 简洁列出贡献。

### 8.4 Contributions 建议版本

```text
In summary, this paper makes four contributions:

1. We present a systematic diagnostic evaluation of per-token early exit on long reasoning traces. Standard confidence- and similarity-based criteria fail to maintain trace-level reliability despite high token-level agreement, revealing a sharp mismatch between token-level and trace-level metrics.

2. We identify two representation-level failure modes behind this breakdown. Oscillatory hidden-state trajectories create false convergence signals, while representation immaturity causes tokens to appear confident before they are correct. Native thinking-mode controls show that the deficit is not merely a distillation artifact, while distilled models amplify oscillatory convergence in our experiments.

3. We diagnose a projection bottleneck using tuned lens. By separating intermediate representation availability from raw LM-head readout quality, we show that standard logit-lens projections substantially underestimate early-exit potential.

4. We evaluate learned exit gates as a proof of concept. Learned gates partially recover trace-level reliability across reasoning model families, but residual compound error over long traces remains a major obstacle to practical per-token early exit.
```

---

## 9. Conditional P0：actual autoregressive rollout

### 9.1 是否必须做？

不建议把 full rollout 无条件列为三天内 P0。它应该是 **claim-dependent P0**：

| 如果论文保留这些 claim | rollout 是否必须 |
|---|---:|
| “deployment failure” | 必须至少做最小 rollout |
| “practical deployment constraint” | 强烈建议做最小 rollout |
| “diagnostic evidence suggests deployment risk” | 可以不做，但要写 limitation |
| “we provide a diagnostic study” | 非必须 |

### 9.2 最小 rollout 方案

如果要做，只做最小版，不要扩散。

| Model | Dataset | Sample size | Methods |
|---|---|---:|---|
| R1-Qwen3-8B | GSM8K | 100–200 | full-depth, raw CALM, tuned CALM, learned gate V2a |

报告：

| Metric | 说明 |
|---|---|
| AnswerAcc / TaskAcc | 最终答案正确率 |
| Avg exit layer | 平均退出层 |
| FLOPs savings | 理论计算节省 |
| Generated length | 生成长度是否变化 |
| Divergence position | 第一次偏离 full-depth 的 token 位置 |
| Failure examples | 少量定性例子 |

### 9.3 如果不做 rollout，应如何写

如果三天内不做 rollout，建议明确降级 deployment claim：

```text
Our evaluation is primarily diagnostic: we measure whether intermediate-layer predictions match full-depth predictions on recorded reasoning traces. This setting isolates representation and projection failures, but does not fully capture autoregressive feedback during deployment. Actual optimized early-exit rollout is an important direction for future work.
```

这样论文仍然可以成立，只是定位更清楚。

---

## 10. P1：有余力再做的实验

### 10.1 扩大 softmax CALM n=50

当前 softmax CALM 部分样本较小。如果有余力，把主模型上的 softmax response / gap 从 n=50 扩到 n=300–500。

优先级：中高。  
收益：增强 “standard confidence criteria fail” 的可信度。  
风险：工程成本低，值得做。

### 10.2 final-answer / boxed-answer analysis 前移主文

如果当前已有 boxed answer analysis，不需要大重跑，直接前移到主文。

优先级：高。  
收益：缓解 TraceExact 过严质疑。  
风险：低。

建议主文表：

| Method | TokenMatch | TraceExact | BoxedAcc / AnswerAcc | Savings |
|---|---:|---:|---:|---:|
| Full-depth | — | 100 | ... | 0 |
| Fixed layer | ... | ... | ... | ... |
| Raw CALM | ... | ... | ... | ... |
| Tuned CALM | ... | ... | ... | ... |
| Learned gate | ... | ... | ... | ... |

### 10.3 length-bucket analysis

目的：区分 early-exit failure 是纯 length effect，还是 reasoning dynamics effect。

做法：按照 trace length 分桶，在相似长度范围内比较：

- Qwen3 non-thinking；
- Qwen3 native thinking；
- R1-Distill-Qwen。

优先级：中。  
收益：支持 “不只是长，也有 representation dynamics”。  
风险：低到中。

### 10.4 compact cross-architecture summary

不建议重跑太多新实验，但可以把已有 R1-Llama / Phi / QwQ 结果整理成一张 compact table。

| Model | Native / distilled | Benchmark | CALM fail? | NM pattern? | Gate recovery? | Scope note |
|---|---|---|---|---|---|---|
| R1-Qwen | distilled | GSM8K | yes | numerical oscillation | yes | main |
| Qwen3 thinking | native | GSM8K | yes / partial | reduced oscillation | ... | native control |
| R1-Llama | distilled | GSM8K | ... | different dominant type | yes | cross-arch |
| Phi-4 | reasoning | GSM8K | ... | special tokens | yes | 14B |
| QwQ-32B | reasoning | GSM8K | diagnostic fail | not profiled | not tested | diagnostic-only |

优先级：中。  
收益：让 scope 更清楚。  
风险：低。

---

## 11. 不建议三天内做的事情

下面这些事情长期有价值，但不适合三天内作为主线任务。

### 11.1 大量新增 benchmark

不建议三天内新增：

- GPQA；
- AMC；
- 更多 AIME 年份；
- MBPP；
- 新代码 benchmark。

原因：

1. answer extraction 会引入新噪声；
2. 结果不一定稳定；
3. 主文空间不够；
4. 不能直接修复核心 confound。

### 11.2 外部 baseline

不建议三天内实现：

- LayerSkip；
- speculative decoding；
- self-speculative decoding；
- external learned router；
- verifier-assisted fallback。

原因：

1. 工程成本高；
2. 公平比较难；
3. 容易把论文从 diagnostic study 拉回 method benchmark；
4. 做得不完整反而被 reviewer 攻击。

### 11.3 QwQ-32B full profiling

长期可以做，但三天内不建议作为重点。

原因：

1. 成本高；
2. 只修 scale claim，不修主 confound；
3. 如果结果复杂，会进一步打乱叙事。

### 11.4 wall-clock speedup

如果没有成熟优化实现，不建议强行报告 wall-clock speedup。

原因：

1. naive implementation 可能没有真实 speedup；
2. wall-clock 受 kernel、batching、KV cache、硬件利用率影响很大；
3. 如果数字不好，会拖累本来很强的 diagnostic story。

可以只报告 analytical FLOPs savings，并明确说 wall-clock optimized implementation is future work。

---

## 12. 推荐三天排期

### Day 1：修 native control 主表 + 指标命名

任务：

1. 整理或补跑 Qwen3-8B non-thinking；
2. 整理或补跑 Qwen3-8B native thinking；
3. 整理 R1-Distill-Qwen 同协议结果；
4. 计算 TokenMatch、TraceExact、BoxedAcc / AnswerAcc、NM rate；
5. 全文替换 Q_Acc 命名。

当天必须产出：

- Table X：三方 native/distilled 主表；
- metric definition subsection；
- 初版 claim weakening list。

---

### Day 2：修 split + 重写叙事

任务：

1. 澄清 tuned lens / gate split；
2. 必要时重跑 R1-Qwen main gate with nested split；
3. 重写 Abstract；
4. 重写 Introduction；
5. 重写 Contributions；
6. 把 learned gate 改成 proof-of-concept recovery。

当天必须产出：

- clean split protocol；
- revised Abstract；
- revised Introduction opening；
- revised Contributions；
- updated limitations。

---

### Day 3：补低成本高收益内容 + 格式修复

任务：

1. 把 boxed-answer / final-answer analysis 前移主文；
2. 有余力做 length-bucket analysis；
3. 如果保留 deployment claim，做最小 R1-Qwen GSM8K rollout；
4. 修 checklist、float、页数、匿名；
5. 整理 compact cross-model summary table；
6. 最终全文检查 claim-evidence alignment。

当天必须产出：

- revised main tables / figures；
- clean PDF；
- checklist 正常；
- no overclaim version。

---

## 13. 主文结构建议

建议最终主文结构如下：

```text
1. Introduction
   - Reasoning traces are expensive
   - Per-token early exit is natural
   - Reasoning traces violate standard early-exit assumptions
   - Research question and findings

2. Setup and Metrics
   - Models and datasets
   - Early-exit criteria
   - TokenMatch, TraceExact, AnswerAcc, Savings
   - Native vs distilled comparison design

3. Standard Early Exit Fails on Reasoning Traces
   - CALM / cosine / fixed-layer results
   - Token-level vs TraceExact gap
   - BoxedAcc / AnswerAcc analysis

4. Native Thinking Control and Distillation Effects
   - Qwen3 non-thinking vs native thinking vs R1-Distill-Qwen
   - What is reasoning-associated?
   - What is distillation-associated?

5. Diagnosing the Failure Modes
   - Non-monotonicity
   - Representation immaturity
   - Token-role diagnostics
   - Proxy failure

6. Projection Bottleneck and Partial Recovery
   - Raw vs tuned lens
   - Oracle gap
   - Learned gate as proof of concept
   - Remaining compound-error limit

7. Discussion and Limitations
   - Diagnostic scope
   - Deployment rollout limitation if not done
   - Distillation and scale caveats
   - Future directions: segment-level / verifier-guided / training-aware exits
```

---

## 14. 主文图表建议

### Table 1：Models and comparison design

展示：

- Qwen3 non-thinking；
- Qwen3 native thinking；
- R1-Distill-Qwen；
- R1-Llama；
- Phi；
- QwQ diagnostic-only。

明确 native / distilled / diagnostic-only 状态。

### Table 2：Native thinking control

这是最重要的新主表。

| Model | Condition | Avg tokens | TokenMatch | TraceExact | BoxedAcc / AnswerAcc | NM rate |
|---|---|---:|---:|---:|---:|---:|

### Figure 1：Problem and failure modes

展示：

- per-token early exit；
- long trace compound error；
- oscillatory convergence；
- representation immaturity。

### Figure 2：CALM failure and token-to-trace gap

展示：

- token-level accuracy vs savings；
- TraceExact collapse；
- BoxedAcc / AnswerAcc if available。

### Figure 3：Native vs distilled convergence dynamics

展示：

- Qwen3 non-thinking；
- Qwen3 native thinking；
- R1-Distill-Qwen。

### Figure 4：Projection bottleneck and recovery

展示：

- raw oracle；
- tuned-lens oracle；
- tuned CALM；
- learned gate；
- full-depth reference。

---

## 15. 最终投稿前 checklist

### Scientific checklist

- [ ] Qwen3 non-thinking / native thinking / R1-Distill-Qwen 三方表已在主文出现；
- [ ] tuned lens / gate split 无 test leakage；
- [ ] Q_Acc 已改名为 TraceExact 或其他清晰术语；
- [ ] AnswerAcc / BoxedAcc 已报告或明确说明 limitation；
- [ ] learned gate 被定位为 proof-of-concept，不是主方法；
- [ ] distillation-specific / reasoning-general / capacity-threshold claim 已弱化；
- [ ] 1.5B reversal 和 QwQ diagnostic-only 被作为 scope caveat；
- [ ] oracle 和 deployable method 被清楚区分；
- [ ] 所有 threshold 选择使用 validation set；
- [ ] 所有 final numbers 使用 held-out test set；
- [ ] sample sizes 在正文、表格、caption 中一致。

### Narrative checklist

- [ ] Abstract 先讲问题，再讲发现，不堆过多数字；
- [ ] Introduction 有背景、gap、research question、findings；
- [ ] Contributions 按 research questions 组织；
- [ ] Related Work 明确区分 layer exit 和 sequence-level early stopping；
- [ ] Limitations 诚实说明 diagnostic scope、distillation confound、rollout limitation；
- [ ] 主文每个实验都回答明确问题。

### Submission checklist

- [ ] 使用 NeurIPS 2026 官方 style；
- [ ] 主文页数符合要求；
- [ ] references / appendix / checklist 顺序正确；
- [ ] 没有 appendix float 漂到 checklist 后；
- [ ] checklist 没有污染文本；
- [ ] 匿名信息全部去除；
- [ ] supplementary/code 匿名；
- [ ] PDF 可编译、图表清晰、caption 完整。

---

## 16. 最终建议

这篇论文最值得保留和强化的不是 learned gate，而是下面这个 diagnostic insight：

> **Reasoning traces break the assumptions behind standard per-token early exit.**

因此，最终版本应该避免把自己包装成一个 “我们提出了更好 early-exit gate” 的方法论文，而应该清楚地说：

> **我们系统性分析了为什么 standard per-token early exit 在 reasoning traces 上失败，并识别了 compound error、oscillatory convergence、representation immaturity 和 projection bottleneck 等关键机制。**

三天内的最优策略是：

1. **用 native thinking control 修复 distilled confound；**
2. **用 clean split 修复 methodology risk；**
3. **用指标重命名修复 presentation risk；**
4. **用 claim weakening 修复 overclaim risk；**
5. **用叙事重写修复 clarity risk。**

如果还有时间，再做一个最小 actual rollout。没有时间也不要硬做，只要把 deployment claim 收回到 diagnostic claim，这篇论文仍然可以作为一篇有价值的 NeurIPS empirical / negative-results submission。
