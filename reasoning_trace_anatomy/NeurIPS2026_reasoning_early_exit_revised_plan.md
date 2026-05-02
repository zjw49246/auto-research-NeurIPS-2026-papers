# NeurIPS 2026 论文叙事与实验改进方案（修订版 v2）

**论文：** *The Convergence Cost of Reasoning: Why Per-Token Early Exit Underperforms*  
**目标：** 在投稿前三天内，以最高性价比提升论文可信度、叙事清晰度和审稿通过概率。  
**建议定位：** Negative Results / Empirical Analysis / Diagnostic Study，而不是主要的新 early-exit 方法论文。

---

## 0. 总体判断

这篇论文的核心 idea 是有价值的：它研究的是 **per-token early exit 在长 reasoning traces 上为什么会失败**。当前论文已经有大量实验（正文 9 页 + 附录 34 页），**实验覆盖面相当充分**，包含 native thinking ablation（Section 4.4 + Appendix U）、boxed answer analysis（Section S + Table 37）、length-bucket analysis（Appendix E Tables 16–17 + Appendix W Table 47）、cross-architecture gate transfer（Appendix V）、compound error quantification（Appendix W Table 45 + Figure 8）等。

最大问题不是实验数量不足，而是**已有结果的组织方式和叙事呈现**：

1. **主叙事不够清楚。** Abstract、Introduction、Contribution 太像实验结果堆叠，而不是围绕一个研究问题逐步展开。
2. **核心因果链条不够干净。** 主文 2×2 设计大量依赖 DeepSeek-R1 distilled models，导致 reasoning effect、distillation effect、teacher style 和 trace distribution 混在一起。**关键的 native thinking ablation（Section 4.4）当前只有半段话**，应提升为主文核心。
3. **关键结果埋在附录。** Compound error 量化（Appendix W）、boxed answer analysis（Section S）、cross-architecture summary 等高价值结果散落在 34 页附录中，主文没有充分利用。
4. **部分 claim 过强。** 例如 Abstract 中 "distillation-specific"、"reasoning-general" 等表达超过了当前实验设计能支撑的范围。Section 4.4 的 native thinking 数据实际上显示 oscillation 在 native thinking 中**降低但未消失**（NM 9.1% vs non-reasoning 11.6%），不能支撑 "distillation-specific" 的断言。
5. **指标命名容易误导。** Abstract 中 "yield 0% question accuracy" 是 headline result，但 Q_Acc 实际定义是 trace-level exact match（所有 token 必须完全一致），而非通常理解的 "答对率"。虽然 Section 3.3 有明确定义，但 Abstract 和很多引用处没有定义上下文，reviewer 很容易误解。

因此，三天内最重要的不是跑新实验，而是：

> **重组已有结果、修复叙事主线、修复 split 描述、修复指标命名、弱化 claim。**

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

三天内不要试图做一个完整新论文。**大部分所需数据已存在于当前论文的正文和附录中**，核心工作是重组和改写，而非重跑实验。应该集中做五件事：

1. **把已有的 native thinking control（Section 4.4 + Appendix U Table 41）提升为主文核心实验；**
2. **澄清 tuned lens / learned gate 的 train-val-test split，消除 leakage 质疑；**
3. **重命名 Q_Acc 等易混淆指标，统一指标体系；**
4. **弱化所有过强 causal / universal claims；**
5. **重写 Abstract、Introduction、Contributions。**

Actual rollout 是加分项，但不应无条件列为必须完成项。更合理的判断是：

> **如果论文保留 deployment claim，就必须做最小 rollout；如果论文定位为 diagnostic study，则 rollout 可以作为 limitation / future work。**

### 2.2 已有数据盘点

以下是论文中**已经存在**但被低估或需要前移/重组的关键结果：

| 已有内容 | 当前位置 | 建议操作 |
|---|---|---|
| Native thinking ablation（Qwen3-8B thinking/non-thinking 对比） | Section 4.4（半段话）+ Appendix U Table 41 | 提升为主文独立核心表格 |
| Boxed answer / answer segment analysis | Appendix S + Table 37 | 前移主文，配合 TraceExact 使用 |
| Compound error 量化（p^T 模型 + Figure 8） | Section 5.1（概念提及）+ Appendix W Table 45 | Figure 8 前移主文或在主文增加引用 |
| Length-bucket analysis | Appendix E Tables 16–17 + Appendix W Table 47 | 酌情前移关键数据点 |
| Cross-architecture gate transfer | Appendix V Tables 44, 52–53 | 整理为 compact summary |
| 1.5B reversal | Appendix V Table 42 | 主文 Discussion 明确讨论 |
| Tuned lens cross-difficulty drop | Table 36（GSM8K → MATH-500） | 在 limitation 中明确说明 |

---

## 3. 优先级总表

| 优先级 | 任务 | 工作性质 | 是否三天内必须完成 | 重要程度 | 主要收益 |
|---|---|---|---:|---:|---|
| P0 | 把 Section 4.4 native thinking ablation 提升为主文核心表格 | 重组已有数据 | 是 | 极高 | 修复 distilled confound，重建主线 |
| P0 | tuned lens / gate split 澄清（footnote → 明确表述） | 补充方法描述 | 是 | 极高 | 排除 data leakage 风险 |
| P0 | 指标重命名：Q_Acc → TraceExact 等 | 全文替换 | 是 | 高 | 避免 reviewer 误解 headline result |
| P0 | 弱化 causal / universal claims | 措辞修改 | 是 | 极高 | 提升可信度，避免被抓住反驳 |
| P0 | 重写 Abstract / Introduction / Contributions | 叙事重组 | 是 | 极高 | 改善 clarity 和 paper framing |
| P0-sub | 修复 checklist、float、页数、匿名格式 | 格式修复 | 是 | 极高 | 避免格式扣分或 desk rejection |
| Cond-P0 | 最小 actual rollout | 新实验 | 取决于是否保留 deployment claim | 高 | 支撑真实部署结论 |
| P1-高 | Appendix S boxed-answer analysis 前移主文 | 前移已有数据 | 强烈建议 | 高 | 缓解 TraceExact 过严质疑，**数据已有，成本极低** |
| P1-高 | Appendix W compound error Figure 8 前移或强化引用 | 前移已有数据 | 强烈建议 | 高 | 强化 token-to-trace gap 的核心解释 |
| P1 | 扩大 softmax CALM n=50 到 n=300–500 | 补充实验 | 有余力做 | 中高 | 增强主 baseline 可信度 |
| P1 | length-bucket analysis 前移关键数据 | 前移已有数据 | 有余力做 | 中 | 区分 length effect 和 reasoning dynamics |
| P1 | R1-Llama / Phi 补同协议 compact summary | 整理已有数据 | 有余力做 | 中 | 增强跨架构证据 |
| P2 | failure taxonomy 小样本分析 | 可选 | 可选 | 中 | 增强 mechanism 解释 |
| P2 | QwQ-32B full profiling | 不建议三天内做 | 否 | 中低 | 工程成本高，收益有限 |
| P2 | 新增 benchmark / 外部 baseline / wall-clock speedup | 不建议三天内做 | 否 | 中低 | 容易拖垮执行，且可能引入新噪声 |

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

所以它不能干净地证明 "reasoning 本身导致 early-exit failure"。

### 4.2 论文已有的 native thinking 数据

**关键事实：论文已经做了 native thinking ablation，数据在 Section 4.4 + Appendix U Table 41 中。** 当前问题不是缺数据，而是这个关键对照实验只占了主文半段话（Section 4.4，约 8 行），没有独立的表格，核心数据全在 Appendix U。

已有的关键数据点（来自 Section 4.4 + Table 41）：

| 指标 | Qwen3-8B non-thinking | Qwen3-8B native thinking | R1-Distill-Qwen-8B |
|---|---:|---:|---:|
| NM rate | 11.6% | 9.1% | 20.8% |
| L28 exit accuracy | 47.7% | 34.7% | 32.4% |

解读：
- **Oscillation（NM rate）**：native thinking（9.1%）< non-reasoning（11.6%）<< R1-distilled（20.8%）。说明 oscillation 主要与 distillation 关联，native thinking 甚至低于 non-reasoning。
- **Representation immaturity（exit accuracy）**：native thinking（34.7%）≈ R1-distilled（32.4%），两者均远低于 non-reasoning（47.7%）。说明 representation immaturity 不是 distillation artifact，而是 reasoning 行为本身的属性。

### 4.3 需要做的工作：前移重组，而非新实验

建议把 Section 4.4 + Appendix U 的数据重组为一张主文核心表格：

**Table X: Native thinking-mode control separates reasoning effects from distillation effects.**

| Model | Condition | Avg tokens | TokenMatch | TraceExact | BoxedAcc / AnswerAcc | NM rate | Key interpretation |
|---|---|---:|---:|---:|---:|---:|---|
| Qwen3-8B | non-thinking | ... | ... | ... | ... | 11.6% | ordinary generation / instruction baseline |
| Qwen3-8B | native thinking | ... | ... | ... | ... | 9.1% | reasoning-style generation without distillation |
| R1-Distill-Qwen-8B | distilled reasoning | ... | ... | ... | ... | 20.8% | reasoning + distillation / teacher style |

**具体步骤：**
1. 从 Appendix U Table 41 提取完整数据行，填入上述主文表格；
2. 补充计算 TokenMatch、TraceExact（如 Table 41 中的指标名不同，做对应映射）；
3. 把 Section 4.4 从半段话扩展为独立的 subsection 或合并入主实验讨论；
4. 在 Section 4.4 原位置增加对该表格的分析文字（约 0.5 页）。

### 4.4 这张表要回答的问题

| 对比 | 回答的问题 | 论文已有证据 |
|---|---|---|
| Qwen3 thinking vs Qwen3 non-thinking | thinking behavior 本身是否让 early exit 更难？ | 是：exit accuracy 34.7% vs 47.7%（更差） |
| R1-Distill-Qwen vs Qwen3 thinking | distillation 是否额外放大某些 dynamics？ | 是（NM rate）：20.8% vs 9.1%；否（exit accuracy）：32.4% ≈ 34.7% |
| R1-Distill-Qwen vs Qwen3 non-thinking | 原始 reasoning vs non-reasoning gap | 作为旧 2×2 结果的桥接 |

### 4.5 应该如何改论文结论

不要说：

> Oscillatory dynamics are distillation-specific.

改成：

> Oscillatory dynamics are strongest in distilled reasoning models (NM 20.8%) and are substantially reduced in the native Qwen thinking-mode control (NM 9.1%, below even the non-reasoning baseline of 11.6%), suggesting an association with distillation or teacher-induced trace style rather than reasoning per se.

不要说：

> Representation immaturity is reasoning-general.

改成：

> Representation immaturity persists in the native Qwen thinking-mode control (L28 exit accuracy 34.7% vs 47.7% non-reasoning), suggesting that this failure mode is not solely a distillation artifact but is associated with reasoning-style generation itself. However, this conclusion is currently supported by a single architecture family (Qwen) and requires cross-architecture validation.

---

## 5. P0-2：澄清 tuned lens / learned gate split

### 5.1 当前风险

论文第 8 页脚注明确写到：

> "Per-layer affine transforms fitted on all 500 MATH-500 questions prior to the gate train/test split (2 parameters per layer)."

这意味着 tuned lens 在全部 500 题上拟合后，才进行 gate 的 train/test 划分。虽然每层仅 2 个参数（scale + bias），实际 leakage 风险有限（affine transform 学习的是 per-layer 全局投影，而非 per-question 特征），但 reviewer 一旦注意到这个脚注就可能质疑方法论完整性。

同时，Appendix P（page 21）描述了 "Sequential split and cross-validation"：使用 400/100 的 sequential partition 和 5-fold stratified CV。但 tuned lens 是否在每个 fold 内重新拟合没有明确说明。

### 5.2 推荐协议

理想方案是采用严格 nested split：

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

鉴于 tuned lens 每层仅 2 个参数，实际 leakage 风险低，三天内有两种策略：

**策略 A（低成本优先）：不重跑，通过文字充分说明。**

1. 在方法部分明确解释：tuned lens affine transform 每层仅 2 个参数（scale + bias），学习的是全局层间投影校准，与具体 question 内容无关，不构成实质性 leakage；
2. 补充报告：在 held-out test set 上的 tuned lens 拟合稳定性（如果 5-fold 各 fold 的 affine 参数差异很小，说明 leakage 不显著）；
3. 所有表格 caption 明确说明 split 协议。

**策略 B（如有余力）：在主结果上重跑 nested split。**

1. 主文只保留 clean nested split 的 R1-Qwen main result；
2. 旧的 cross-architecture gate transfer 降级为 appendix / preliminary；
3. 所有表格 caption 明确说明 split 协议。

### 5.4 必须新增的表述

无论采用哪种策略，建议在方法部分加入（根据实际情况选择）：

策略 A 版本：
```text
Tuned-lens affine transforms (2 parameters per layer: scale and bias) are fitted on the full MATH-500 set to calibrate per-layer projection quality. Because these transforms are layer-global and question-agnostic, they do not leak question-specific information. Gate parameters, exit thresholds, and all reported metrics use a strict 400/100 train/test split (or 5-fold stratified CV), with thresholds chosen on a held-out validation partition.
```

策略 B 版本：
```text
All tuned-lens transformations, gate parameters, and exit thresholds are fitted without access to test questions. In cross-validation experiments, the tuned lens and gate are re-trained within each fold; no affine transform fitted on held-out questions is used for final reporting.
```

### 5.5 关联风险：tuned lens cross-difficulty 泛化下降

Table 36 显示 tuned lens oracle 在 GSM8K（in-distribution）上为 58.2%，但在 MATH-500（OOD）上降至 38.4%——降幅显著。这说明 tuned lens 的 affine transform 对数据分布有一定依赖性，需要在 Limitations 中明确提及：

```text
Tuned-lens affine transforms are fitted on MATH-500 and show substantial accuracy degradation when applied to harder out-of-distribution problems (Table 36: 58.2% on GSM8K vs. 38.4% on MATH-500), suggesting that the projection bottleneck is difficulty-sensitive and the tuned-lens calibration may not fully transfer across difficulty levels.
```

---

## 6. P0-3：统一指标命名

### 6.1 当前问题

论文 Abstract 的 headline result 是 "yield 0% question accuracy (Q_Acc)"。Q_Acc 的实际定义（Section 3.3）是：

> "Question-level exit accuracy (Q_Acc) requires all tokens in a question's reasoning trace to match."

这是一个 trace-level exact match 指标——要求一条 trace 中**每一个** token 都与 full-depth 预测一致。对于长度超过 3000 tokens 的 reasoning trace，这个指标极为严格。

问题在于：

1. **Abstract 中 "0% question accuracy" 没有附带定义**，reviewer 第一反应会理解为 "模型对所有问题都答错了"（即 task accuracy = 0%），这与实际含义（trace exact match = 0%）有本质区别；
2. **"question accuracy" 在 ML 社区的约定俗成含义就是 "答对率"**，与论文定义冲突；
3. Section 3.3 中虽有定义，但很多后续引用 Q_Acc 的地方没有重复定义上下文；
4. 论文同时也报告了 boxed answer accuracy（Section S），但不叫 Q_Acc，两个指标的关系容易混淆。

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
We use TraceExact as a strict diagnostic metric: a trace is counted as correct only if every early-exit token matches the full-depth token. Because this metric is intentionally stringent—even 99.2% per-token accuracy yields near-zero TraceExact over 3,000 tokens (Section 5.1)—we also report BoxedAcc (whether the final boxed answer matches) and TaskAcc (benchmark answer correctness) to assess downstream impact.
```

### 6.4 改名收益

这个改动能预防一个几乎必定出现的 reviewer objection：

> "0% question accuracy 是不是只是因为 all-token exact match 太严格？"

改名为 TraceExact 后：

1. **指标含义一目了然**：reviewer 看到 "TraceExact = 0%" 时不会误解为 "模型答不对题"；
2. **自然引出层次化分析**：TraceExact（严格诊断）→ BoxedAcc（答案段一致性）→ TaskAcc（最终正确率），形成清晰的指标层次；
3. **与 boxed answer 分析无缝衔接**：论文 Section S + Table 37 已有 boxed answer 数据，改名后可以在主文中更自然地同时报告两类指标。

---

## 7. P0-4：弱化 claim

### 7.1 必须替换的强 claim

以下替换建议基于论文中的**具体实验证据范围**：

| 当前论文中的表述 | 位置 | 风险 | 建议改写 |
|---|---|---|---|
| oscillatory dynamics (distillation-specific) | Abstract、Figure 1(c) caption | Section 4.4 数据显示 native thinking NM=9.1% < non-reasoning 11.6%，说明 oscillation 在 native thinking 中不存在放大，但 "distillation-specific" 暗示只有 distilled 模型才有 | distillation-associated (strongest in distilled models; reduced below non-reasoning baseline in native thinking mode) |
| representation immaturity (reasoning-general) | Abstract、Contributions | 仅基于 Qwen 家族内部对比，无跨架构 native thinking 验证（Llama 没有 native thinking mode） | persists in both distilled and native Qwen reasoning settings |
| current reasoning models | 多处 | 范围过大，论文有 1.5B reversal（Table 42）且 QwQ-32B 只做了 diagnostic | the reasoning models studied here (8B–14B scale) |
| minimum-capacity threshold | Section 4.2 附近 | Qwen-7B vs Qwen-8B 不是纯 scale 控制，还有架构差异 | suggests a capacity- or architecture-dependent emergence |
| projection quality is the primary bottleneck | Section 5.2 | tuned lens 是诊断工具，不是因果干预；且 Table 36 显示 cross-difficulty 泛化下降显著 | projection quality is a major bottleneck under tuned-lens diagnostics |
| deployment constraint | Conclusion | 如果没有 actual rollout，仅基于 frozen trace 分析 | diagnostic evidence suggests a deployment-relevant constraint |
| CALM failure persists at scale | 涉及 QwQ-32B | QwQ-32B 只测了 CALM，没有 gate 实验，没有 full profiling | CALM-style diagnostics also fail on QwQ-32B in our limited evaluation |

### 7.2 建议新增 "Scope of claims" 小段

```text
Our strongest claims concern 8B–14B math/code reasoning traces under the evaluated Qwen, Llama, and Phi model families. Distilled models are used as important test cases but introduce teacher- and distillation-specific factors; native Qwen thinking-mode controls reduce but do not eliminate this concern. The 1.5B reversal (Appendix V) and the diagnostic-only 32B experiments indicate that scale and training recipe can change the observed dynamics. We therefore frame the proposed mechanisms as empirically supported failure modes rather than universal laws.
```

### 7.3 1.5B reversal 的处理策略

Appendix V Table 42 显示 1.5B 时 reasoning / non-reasoning 的 early-exit 关系反转（R1-distilled 反而更好）。这是一个反直觉结果，当前论文仅在附录提及，但 reviewer 很可能追问。

建议在主文 Discussion / Limitations 中用 1–2 句明确解释：

```text
At the 1.5B scale, the reasoning–non-reasoning early-exit gap reverses (Table 42): the R1-distilled model exhibits higher exit accuracy than its non-reasoning counterpart. This reversal may reflect that at very small scales, base model representation instability dominates over reasoning-specific dynamics, or that distillation from a strong teacher provides a regularizing effect absent in the base model. This observation reinforces our framing of the identified failure modes as scale- and recipe-dependent rather than universal.
```

---

## 8. P0-5：重写 Abstract / Introduction / Contributions

### 8.1 Abstract 当前问题

当前 Abstract 的问题是：

1. 第一句直接抛出 "15.3pp per-token exit accuracy deficit"——不熟悉 early exit 的 reviewer 无法理解这个数字的意义；
2. "yield 0% question accuracy (Q_Acc)" 在没有定义的情况下容易被误解；
3. 结论过于确定（"distillation-specific"、"reasoning-general"）；
4. scope 太宽（"current reasoning models"）；
5. learned gate 和 diagnostic analysis 的定位混在一起。

### 8.2 建议 Abstract 草稿

```text
Per-token early exit is a natural way to reduce the inference cost of long reasoning traces, but its reliability in this setting remains poorly understood. We show that standard confidence- and similarity-based exit criteria break down on reasoning traces: high token-level agreement can still collapse to near-zero trace-level exactness because small per-token errors compound over thousands of generated tokens.

We identify two representation-level failure modes behind this collapse. Some tokens exhibit oscillatory hidden-state trajectories that create false convergence signals, while others appear confident but remain representationally immature until late layers. Native Qwen thinking-mode controls show that the early-exit deficit is not merely a distillation artifact, while distilled reasoning models amplify the oscillatory failure mode in our experiments.

Using tuned-lens diagnostics, we show that raw intermediate-layer projections substantially underestimate the information available in hidden states, revealing a large projection bottleneck. Learned exit gates partially recover trace-level reliability across reasoning model families, but compound error remains a central obstacle. These results suggest that efficient reasoning requires segment-level, verifier-guided, or training-aware exit mechanisms beyond simple per-token confidence thresholds.
```

**与当前 Abstract 的主要差异：**

| 维度 | 当前 Abstract | 建议 Abstract |
|---|---|---|
| 开头 | 直接报数字 "15.3pp deficit" | 先建立问题 "reliability remains poorly understood" |
| Q_Acc 表述 | "0% question accuracy" | "near-zero trace-level exactness" |
| claim 强度 | "distillation-specific"、"reasoning-general" | "not merely a distillation artifact"、"in our experiments" |
| scope | "current reasoning models" | 不使用全称量词 |
| 结构 | 结果堆叠 | 问题→机制→诊断→结论 |

### 8.3 Introduction 建议结构

建议在当前 Introduction 基础上调整为 5–6 段（不需要完全重写，主要是重组和补充过渡）：

1. **Reasoning models are expensive.** 长 CoT / thinking traces 提升能力，但推理成本高。
2. **Per-token early exit is a natural solution.** 解释 early exit 和已有 CALM / LayerSkip 动机。
3. **Reasoning traces are different.** 长、结构化、token role 异质、错误会累积。
4. **Research question.** Standard per-token early exit 在 reasoning traces 上是否可靠？如果失败，为什么？
5. **Main findings.** CALM-style criteria fail；native thinking control 显示不是 distilled-only；distillation 放大 oscillation；tuned lens 显示 projection bottleneck；learned gate 部分恢复。
6. **Contributions.** 简洁列出贡献。

### 8.4 Contributions 建议版本

当前论文 Contributions（page 2）已有合理骨架，但第 1 条中 "distillation-specific" 和 "reasoning-general" 的措辞需修正。建议版本：

```text
In summary, this paper makes four contributions:

1. We present a systematic diagnostic evaluation of per-token early exit on long reasoning traces. Standard confidence- and similarity-based criteria fail to maintain trace-level reliability despite high token-level agreement, revealing a sharp mismatch between token-level and trace-level metrics.

2. We identify two representation-level failure modes behind this breakdown. Oscillatory hidden-state trajectories create false convergence signals, while representation immaturity causes tokens to appear confident before they are correct. Native thinking-mode controls show that the representation-immaturity deficit is not merely a distillation artifact, while distilled models exhibit amplified oscillatory dynamics in our experiments.

3. We diagnose a projection bottleneck using tuned lens. By separating intermediate representation availability from raw LM-head readout quality, we show that standard logit-lens projections substantially underestimate early-exit potential.

4. We evaluate learned exit gates as a proof of concept. Learned gates partially recover trace-level reliability across reasoning model families, but residual compound error over long traces remains a major obstacle to practical per-token early exit.
```

---

## 9. Conditional P0：actual autoregressive rollout

### 9.1 是否必须做？

不建议把 full rollout 无条件列为三天内 P0。它应该是 **claim-dependent P0**：

| 如果论文保留这些 claim | rollout 是否必须 |
|---|---:|
| "deployment failure" | 必须至少做最小 rollout |
| "practical deployment constraint" | 强烈建议做最小 rollout |
| "diagnostic evidence suggests deployment risk" | 可以不做，但要写 limitation |
| "we provide a diagnostic study" | 非必须 |

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

## 10. P1：有余力再做的实验与前移

### 10.1 P1-高：boxed-answer analysis 前移主文

**论文已有数据：** Appendix S + Table 37 详细分析了 per-type exit accuracy（Overall, Reasoning, Answer, Boxed, Ans. Corr.），按 L20–L34 各层报告。Table 37 显示 L32 时 Boxed accuracy 为 27.6%，Answer Corr. 为 8.6%，远低于 full-depth 的 100% 和 84.3%。

**不需要新实验。** 只需把 Table 37 的关键行（或其子集）前移到主文，配合 TraceExact 使用。

建议主文增加一张对比表（从已有数据整理）：

| Method / Layer | TokenMatch | TraceExact | BoxedAcc | AnswerAcc | Savings |
|---|---:|---:|---:|---:|---:|
| Full-depth (L35) | — | 100% | 84.3% | 84.3% | 0% |
| L28 (78% depth) | ... | ... | ... | ... | ~22% |
| L32 (89% depth) | ... | ... | 27.6% | 8.6% | ~11% |
| Learned gate | ... | ... | ... | ... | ... |

优先级：高。数据已有，成本极低，收益极高——直接回答 "TraceExact 过严是否夸大了问题"。

### 10.2 P1-高：compound error Figure 8 前移或强化引用

**论文已有数据：** Appendix W Figure 8 展示了 trace length vs Q_Acc 的衰减曲线，与理论模型 p^T 高度吻合。Table 45 给出了 compound error 量化。Section 5.1 主文已提及概念，但 Figure 8 仅在附录。

建议选择以下之一：
- **方案 A：** 把 Figure 8 前移到主文 Figure 2 或 3 的位置；
- **方案 B：** 在 Section 5.1 增加对 Appendix W Figure 8 的明确引用和一句关键数据（"At the observed operating point p ≈ 0.997, T ≈ 3000, the compound error model predicts TraceExact ≈ 0%, closely matching observed results"）。

优先级：高。这是论文最优雅的理论解释之一，值得更显著的展示。

### 10.3 扩大 softmax CALM n=50

当前 softmax CALM 部分样本较小。如果有余力，把主模型上的 softmax response / gap 从 n=50 扩到 n=300–500。

优先级：中高。  
收益：增强 "standard confidence criteria fail" 的可信度。  
风险：工程成本低，值得做。

### 10.4 length-bucket analysis

**论文已有数据：** Appendix E Tables 16–17 分析了 chain length vs non-monotonicity rate，按 dataset 和 token type 细分；Appendix W Table 47 分析了 trace length vs compound error。

目的：区分 early-exit failure 是纯 length effect，还是 reasoning dynamics effect。如果要前移，建议从已有 Table 16–17 中提取关键数据点，展示**在控制 trace length 后，reasoning traces 仍然比 non-reasoning traces 更难 early exit**。

优先级：中。  
收益：支持 "不只是长，也有 representation dynamics"。  
风险：低——数据已有。

### 10.5 compact cross-architecture summary

不建议重跑新实验，但可以把已有 R1-Llama / Phi / QwQ 结果整理成一张 compact table。

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

**关键认识：大部分实验数据已经存在于论文正文和附录中，三天的核心工作是重组、改写和修复，而非大量新实验。**

### Day 1：前移 native thinking 数据 + 指标命名 + claim 弱化

任务：

1. 从 Appendix U Table 41 提取 native thinking 数据，整理为主文核心表格（Section 4.3 建议的 Table X）；
2. 把 Section 4.4 从半段话扩展为有表格支撑的完整分析（约增加 0.5 页）；
3. 全文替换 Q_Acc → TraceExact（及相关指标重命名）；
4. 撰写指标定义小节（Section 6.3 建议的写法）；
5. 遍历全文，逐条弱化 Section 7.1 表格中列出的过强 claim。

当天必须产出：

- 主文核心三方对照表格（数据来源：Table 41 + Section 4.4）；
- 统一指标命名（全文 Q_Acc → TraceExact）；
- 初版 claim weakening（所有 Section 7.1 列出的替换）。

---

### Day 2：修 split 表述 + 重写叙事

任务：

1. 选择 Section 5.3 的策略 A 或策略 B 澄清 tuned lens / gate split；
2. 在方法部分新增 split 协议说明（Section 5.4）；
3. 重写 Abstract（Section 8.2 草稿）；
4. 调整 Introduction 结构（Section 8.3）；
5. 重写 Contributions（Section 8.4 草稿）；
6. 在 Limitations 中新增 1.5B reversal 说明（Section 7.3）和 tuned lens cross-difficulty 说明（Section 5.5）。

当天必须产出：

- clean split protocol 表述；
- revised Abstract；
- revised Introduction（主要是增加过渡段和重组，不需完全重写）；
- revised Contributions；
- updated Limitations。

---

### Day 3：前移高价值附录内容 + 格式修复

任务：

1. 从 Table 37（Appendix S）前移 boxed-answer 数据到主文（Section 10.1 建议的对比表）；
2. 强化 Section 5.1 对 compound error 的展示（前移 Figure 8 或增强引用）；
3. 有余力整理 compact cross-model summary table（Section 10.5）；
4. 如果保留 deployment claim，做最小 R1-Qwen GSM8K rollout；
5. 修 checklist、float、页数、匿名；
6. 最终全文检查 claim-evidence alignment。

当天必须产出：

- 主文包含 TraceExact + BoxedAcc 双指标对比；
- clean PDF；
- checklist 正常；
- no overclaim version。

---

## 13. 主文结构建议

建议在当前结构基础上做**增量式调整**，而非完全重写。以下是当前结构与建议结构的对照：

| 当前结构 | 建议调整 | 变化幅度 |
|---|---|---|
| 1. Introduction | 增加过渡段（Section 8.3 建议的 6 段结构），修改 Contributions 措辞 | 中 |
| 2. Related Work | 不变 | 无 |
| 3. Methodology (3.1 Experimental Design, 3.2 Token Role Taxonomy, 3.3 Metrics) | 3.3 改指标名；增加 split 协议说明 | 小 |
| 4. Findings (4.1 NM, 4.2 Exit Accuracy, 4.3 Proxy Failure, 4.4 Native Thinking) | **4.4 从半段话扩展为有表格的完整分析**，增加约 0.5 页 | 中 |
| 5. End-to-End (5.1 Compound Error, 5.2 Projection Bottleneck) | 前移 boxed answer 数据到 5.1；强化 Figure 8 引用 | 小 |
| 6. Conclusion | 扩展为 Discussion + Limitations，增加 scope caveat、1.5B reversal、rollout limitation | 中 |

**核心变化只有一个：把 Section 4.4 从半段话扩展为主文核心分析。** 其余都是措辞修正、数据前移和 caveat 补充。

---

## 14. 主文图表建议

### Table 1：Models and comparison design（扩展当前表格）

增加列标明 native / distilled / diagnostic-only 状态：

- Qwen3 non-thinking → non-reasoning baseline；
- Qwen3 native thinking → **native reasoning control**（新增/前移）；
- R1-Distill-Qwen → distilled reasoning（main）；
- R1-Distill-Llama → distilled reasoning（cross-arch）；
- Phi-4 → reasoning（14B）；
- QwQ-32B → reasoning（diagnostic-only）。

### Table 2（新增/前移）：Native thinking control

这是最重要的结构变化——把 Section 4.4 + Appendix U Table 41 的数据整理为主文表格。

| Model | Condition | Avg tokens | TokenMatch | TraceExact | BoxedAcc | NM rate |
|---|---|---:|---:|---:|---:|---:|

### Table 3（前移）：TraceExact + BoxedAcc + AnswerAcc 多层级对比

从 Appendix S Table 37 前移关键行，展示指标层次。

### Figure 1：Problem and failure modes（不变）

保持当前 Figure 1 的三面板设计（token roles / early exit / failure modes）。

### Figure 2：CALM failure and token-to-trace gap（微调）

改 axis label 使用新指标名（TraceExact 替代 Q_Acc）。如有空间，增加 BoxedAcc 曲线。

### Figure 3（建议前移/新增）：Compound error decay

前移 Appendix W Figure 8 或新制作一张图：展示 trace length vs TraceExact 的衰减，叠加理论 p^T 曲线。

### Figure 4：Projection bottleneck and recovery（不变）

保持当前设计：raw oracle / tuned-lens oracle / tuned CALM / learned gate / full-depth reference。

---

## 15. 最终投稿前 checklist

### Scientific checklist

- [ ] Qwen3 non-thinking / native thinking / R1-Distill-Qwen 三方表格已在主文出现（数据来源：Table 41 + Section 4.4）；
- [ ] tuned lens / gate split 协议已在方法部分明确说明，无 test leakage 或有充分的 leakage-minimal 论证；
- [ ] Q_Acc 已改名为 TraceExact 或其他清晰术语，全文一致；
- [ ] BoxedAcc / AnswerAcc 已在主文报告（从 Table 37 前移），而非仅在附录；
- [ ] learned gate 被定位为 proof-of-concept，不是主方法；
- [ ] Abstract 中 "distillation-specific" 已改为 "distillation-associated" 或类似弱化表述；
- [ ] Abstract 中 "reasoning-general" 已改为 "persists in native thinking settings" 或类似限定表述；
- [ ] 1.5B reversal 在 Discussion/Limitations 中有明确解释（Section 7.3 建议文字）；
- [ ] tuned lens cross-difficulty 下降在 Limitations 中有说明（Section 5.5 建议文字）；
- [ ] oracle 和 deployable method 被清楚区分；
- [ ] 所有 threshold 选择使用 validation set；
- [ ] 所有 final numbers 使用 held-out test set；
- [ ] sample sizes 在正文、表格、caption 中一致。

### Narrative checklist

- [ ] Abstract 先讲问题，再讲发现，不堆过多数字；
- [ ] Abstract 中不使用未定义的 "question accuracy"；
- [ ] Introduction 有背景、gap、research question、findings；
- [ ] Contributions 按 research questions 组织，不含过强 claim；
- [ ] Related Work 明确区分 layer exit 和 sequence-level early stopping；
- [ ] Limitations 诚实说明 diagnostic scope、distillation confound、rollout limitation、1.5B reversal、tuned lens cross-difficulty；
- [ ] 主文每个实验都回答明确问题。

### Submission checklist

- [ ] 使用 NeurIPS 2026 官方 style；
- [ ] 主文页数符合要求（当前 9 页 + native thinking 表格可能需要 ~0.5 页，注意不超限）；
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

因此，最终版本应该避免把自己包装成一个 "我们提出了更好 early-exit gate" 的方法论文，而应该清楚地说：

> **我们系统性分析了为什么 standard per-token early exit 在 reasoning traces 上失败，并识别了 compound error、oscillatory convergence、representation immaturity 和 projection bottleneck 等关键机制。**

三天内的最优策略是：

1. **前移已有 native thinking control 数据（Section 4.4 + Table 41）为主文核心，修复 distilled confound；**
2. **澄清 tuned lens split 协议，修复 methodology risk；**
3. **用 TraceExact 替换 Q_Acc，修复 presentation risk；**
4. **逐条弱化 overclaim（distillation-specific → associated, reasoning-general → limited scope），修复 overclaim risk；**
5. **重写 Abstract/Intro/Contributions，修复 clarity risk；**
6. **前移 boxed answer 数据（Table 37）和 compound error 分析（Figure 8），充分利用已有附录。**

**核心认识：论文的实验工作已经非常充分，主要问题是组织和呈现。三天内应聚焦"精准手术式修改"——前移关键数据、弱化过强 claim、统一指标命名、重写叙事框架——而非跑大量新实验。**

如果还有时间，再做一个最小 actual rollout。没有时间也不要硬做，只要把 deployment claim 收回到 diagnostic claim，这篇论文仍然可以作为一篇有价值的 NeurIPS empirical / negative-results submission。
