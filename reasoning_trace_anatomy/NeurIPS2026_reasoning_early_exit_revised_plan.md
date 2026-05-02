# NeurIPS 2026 论文叙事与实验改进方案（修订版 v3）

**论文：** *The Convergence Cost of Reasoning: Why Per-Token Early Exit Underperforms*  
**目标：** 在投稿前五天内，通过写作重组 + 定向 GPU 实验，最大化论文可信度、叙事清晰度和审稿通过概率。  
**可用资源：** 2–4 × RTX 5090（32GB/卡）、agent 驱动的并行 coding 和写作。  
**建议定位：** Diagnostic Study + Constructive Insight（不是主要的新 early-exit 方法论文，但也不只是 negative result）。

---

## 0. 总体判断

这篇论文的核心 idea 是有价值的：它研究的是 **per-token early exit 在长 reasoning traces 上为什么会失败**。当前论文已经有大量实验（正文 9 页 + 附录 34 页），**实验覆盖面相当充分**，包含 native thinking ablation（Section 4.4 + Appendix U）、boxed answer analysis（Section S + Table 37）、length-bucket analysis（Appendix E Tables 16–17 + Appendix W Table 47）、cross-architecture gate transfer（Appendix V）、compound error quantification（Appendix W Table 45 + Figure 8）等。

论文存在两类问题：**叙事组织问题**和**实验缺口**。

### 叙事组织问题（纯写作可修复）

1. **主叙事不够清楚。** Abstract、Introduction、Contribution 太像实验结果堆叠，而不是围绕一个研究问题逐步展开。
2. **核心因果链条不够干净。** 主文 2×2 设计大量依赖 DeepSeek-R1 distilled models，导致 reasoning effect、distillation effect、teacher style 和 trace distribution 混在一起。**关键的 native thinking ablation（Section 4.4）当前只有半段话**，应提升为主文核心。
3. **关键结果埋在附录。** Compound error 量化（Appendix W）、boxed answer analysis（Section S）、cross-architecture summary 等高价值结果散落在 34 页附录中，主文没有充分利用。
4. **部分 claim 过强。** 例如 Abstract 中 "distillation-specific"、"reasoning-general" 等表达超过了当前实验设计能支撑的范围。Section 4.4 的 native thinking 数据实际上显示 oscillation 在 native thinking 中**降低但未消失**（NM 9.1% vs non-reasoning 11.6%），不能支撑 "distillation-specific" 的断言。
5. **指标命名容易误导。** Abstract 中 "yield 0% question accuracy" 是 headline result，但 Q_Acc 实际定义是 trace-level exact match（所有 token 必须完全一致），而非通常理解的 "答对率"。虽然 Section 3.3 有明确定义，但 Abstract 和很多引用处没有定义上下文，reviewer 很容易误解。

### 实验缺口（需要 GPU 修复）

6. **Native thinking 只有 NM rate 和 exit accuracy，没有 CALM/gate 评估。** 论文在 distilled 模型上做了完整的 CALM failure → gate recovery 链条，但在 native thinking 上只做了一半。Reviewer 可以说 "CALM failure 和 gate recovery 可能都是 distillation artifact"。
7. **没有 actual autoregressive rollout。** 论文所有分析基于 frozen teacher traces，不知道实际部署时 early exit 的 error propagation 行为。
8. **没有 segment-level exit 实验。** 论文 Conclusion 明确建议 "segment-level exit mechanisms"，但没有任何实验验证这个方向。这是从 "pure negative result" 变成 "diagnostic + constructive insight" 的关键。
9. **Tuned lens / gate split 可能存在 leakage。** 论文第 8 页脚注显示 tuned lens 在全部 500 题上拟合后才划分 gate train/test。
10. **QwQ-32B 只做了 CALM diagnostic，没有 gate 实验。** 无法判断 gate recovery 在更大 scale 是否 work。

### 五天内的核心策略

> **并行执行写作重组 + 定向 GPU 实验。写作修复叙事（占 ~60% 的质量提升），GPU 实验填补关键缺口（占 ~40% 的质量提升）。最重要的新增实验是 segment-level exit baseline——它能把论文从 pure negative result 变成 diagnostic + constructive insight。**

---

## 1. 最终推荐主线

建议把论文主线收束为（相比 v2 新增了 segment-level exit 的 constructive insight）：

> **Standard per-token early exit fails on long reasoning traces because token-level reliability does not translate into trace-level or task-level reliability. Native thinking controls confirm that these failures are not merely distillation artifacts. We identify compound error over thousands of tokens as the fundamental bottleneck, and show via tuned-lens diagnostics that a large projection gap exists between hidden-state information and raw readout quality. Learned per-token gates partially recover performance but cannot eliminate compound error. A segment-level exit strategy, which makes uniform exit decisions per reasoning step rather than per token, substantially reduces the compound-error burden, validating our diagnostic analysis and pointing toward coarser-grained exit mechanisms.**

中文版本：

> **标准 per-token early exit 在长 reasoning traces 上失败，是因为 token-level 可靠性无法直接转化为 trace-level 或 task-level 可靠性。Native thinking 对照确认这些失败不只是 distilled model artifact。我们识别了长 trace 上的 compound error 作为根本瓶颈，并通过 tuned lens 显示 hidden state 信息与 raw readout 质量之间存在显著的 projection gap。Per-token learned gate 可以部分恢复，但无法消除 compound error。Segment-level exit 策略——按 reasoning step 而非 per token 做统一 exit 决策——显著降低了 compound error，验证了我们的 diagnostic 分析，并指向了粒度更粗的 exit 机制方向。**

这个主线比 v2 更强，因为它增加了第四层：

1. **Reasoning traces 本身使 early exit 更难（native thinking 确认）；**
2. **Compound error 是根本瓶颈（p^T 模型 + rollout 验证）；**
3. **Per-token gate 能部分缓解但不能解决 compound error；**
4. **Segment-level exit 显著降低 compound error，验证 diagnostic insight 并指向解决方向。**

---

## 2. 五天内的执行框架

### 2.1 执行原则

五天内同时推进两条平行线：

- **写作线（agent A）：** 重组已有结果、修复叙事、弱化 claim、前移附录数据。
- **实验线（agent B + GPU）：** 补三个关键实验缺口 + 重跑 split + rollout。

两条线在 Day 3–4 汇合整合。

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

### 2.3 需要新跑的实验清单

| 新实验 | 性质 | GPU 成本 | Agent coding 工时 | 数据依赖 |
|---|---|---|---|---|
| Native thinking 完整协议（CALM + gate on Qwen3-8B thinking） | 填补 confound 缺口 | 4–6h / 1 卡 | ~4h | 需要 Qwen3-8B thinking hidden states |
| Segment-level exit baseline | 新 constructive insight | 0–2h | ~3h | 使用已有 hidden states |
| Divergence position 分析 | 零成本后处理 | 0 | ~1h | 使用已有 per-layer predictions |
| Actual autoregressive rollout | 填补 deployment 缺口 | 8–15h / 2 卡 | ~8h | 需要 early-exit inference pipeline |
| Nested split 重跑（策略 B） | 修复 methodology | 2–4h / 1 卡 | ~2h | 使用已有或重收集 hidden states |
| QwQ-32B gate 实验 | 增强 scale claim | 6–10h / 2–3 卡 | ~3h | 需要 QwQ-32B hidden states |
| CALM n=50 → n=500 扩展 | 增强 baseline 可信度 | 2–4h / 1 卡 | ~1h | 使用已有或重收集 hidden states |

---

## 3. 优先级总表

| 优先级 | 任务 | 工作性质 | 五天内必须完成？ | 重要程度 | 主要收益 | 预估分数提升 |
|---|---|---|---|---|---|---|
| **P0-写作** | 把 Section 4.4 native thinking 提升为主文核心表格 | 重组已有数据 | 是 | 极高 | 修复 distilled confound 叙事 | +0.5 |
| **P0-写作** | 弱化 causal / universal claims | 措辞修改 | 是 | 极高 | 提升可信度 | +0.5 |
| **P0-写作** | 重写 Abstract / Introduction / Contributions | 叙事重组 | 是 | 极高 | 改善 clarity 和 framing | +0.5–1.0 |
| **P0-写作** | 指标重命名：Q_Acc → TraceExact 等 | 全文替换 | 是 | 高 | 避免 reviewer 误解 headline result | +0.5 |
| **P0-实验** | Native thinking 完整协议（CALM + gate） | 新实验 | 是 | 极高 | 彻底堵住 CALM/gate 层面的 distillation confound | +0.5–1.0 |
| **P0-实验** | Segment-level exit baseline | 新实验 | 是 | 极高 | 从 negative result → constructive insight | +0.5–1.0 |
| **P0-实验** | Actual autoregressive rollout | 新实验 | 是 | 极高 | 支撑 deployment claim | +0.5–1.0 |
| **P0-实验** | Nested split 重跑（策略 B） | 重跑实验 | 是 | 极高 | 消除 data leakage 风险 | +0.3 |
| **P0-写作** | Boxed-answer analysis 前移主文 | 前移已有数据 | 是 | 高 | 缓解 TraceExact 过严质疑 | +0.3–0.5 |
| **P0-写作** | Compound error Figure 8 前移 | 前移已有数据 | 是 | 高 | 强化核心解释 | +0.2–0.3 |
| P0-sub | 修复 checklist、float、页数、匿名格式 | 格式修复 | 是 | 极高 | 避免 desk rejection | 防御性 |
| P0-sub | Divergence position 分析 | 零成本后处理 | 是 | 高 | 为 segment exit 提供量化支撑 | +0.2 |
| P1 | QwQ-32B gate 实验 | 新实验 | 强烈建议 | 高 | 增强 scale claim | +0.3–0.5 |
| P1 | CALM n=50 → n=500 扩展 | 补充实验 | 建议做 | 中高 | 增强 baseline 可信度 | +0.1–0.2 |
| P1 | Warm-up 混合策略实验 | 新实验 | 有余力做 | 中 | 进一步验证 constructive direction | +0.1–0.2 |
| P1 | compact cross-architecture summary | 整理已有数据 | 建议做 | 中 | 让 scope 更清楚 | +0.2 |
| P2 | 新增 benchmark / 外部 baseline / wall-clock | 不做 | 否 | 中低 | 容易拖垮执行 | — |

---

## 4. P0-写作-1：把 native thinking control 提升为主实验

### 4.1 为什么这是最重要的

当前论文主实验依赖 distilled reasoning models：

| 维度 | Qwen | Llama |
|---|---|---|
| Reasoning | R1-Distill-Qwen | R1-Distill-Llama |
| Non-reasoning | Qwen3 | Llama-3.1-Instruct |

这个设计的问题是：reasoning 条件同时混入了 reasoning behavior、DeepSeek-R1 teacher style、distillation objective、trace distribution 和 post-training / calibration difference。所以它不能干净地证明 "reasoning 本身导致 early-exit failure"。

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

### 4.3 需要做的工作：前移重组为主文核心表格

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

## 5. P0-实验-1：Native thinking 完整协议（CALM + gate）

### 5.1 为什么必须做这个实验

论文在 distilled 模型上做了完整的分析链条：

```
CALM criteria → failure → gate training → partial recovery
```

但在 native thinking（Qwen3-8B thinking mode）上只做了 NM rate 和 exit accuracy 的对比（Section 4.4），**没有跑 CALM exit criteria，没有训练 gate**。这意味着：

- 不知道 CALM 在非 distilled reasoning traces 上是否也失败
- 不知道 gate 在非 distilled reasoning traces 上是否也能恢复
- Reviewer 可以说 "CALM failure 和 gate recovery 可能都是 distillation artifact"

这是当前最大的方法论盲区。

### 5.2 实验设计

在 Qwen3-8B native thinking traces 上完整复制 distilled 模型的分析流程：

| 步骤 | 内容 | 输出 |
|---|---|---|
| 1. Forward pass | 收集 Qwen3-8B thinking 模式在 GSM8K/MATH-500 上的 per-layer hidden states | hidden states cache |
| 2. CALM evaluation | 计算 cosine similarity / softmax confidence / entropy exit criteria | CALM failure/success 表 |
| 3. Token role analysis | 分类 numerical / special / reasoning KW / other | per-type NM rate 和 exit accuracy |
| 4. Tuned lens | 拟合 per-layer affine transform（在 train split 上） | tuned lens oracle gap |
| 5. Gate training | 训练 learned gate V2a（在 train split 上） | gate recovery 结果 |
| 6. 三方对比表 | 与 Qwen3 non-thinking 和 R1-Distill-Qwen 对比 | 主文核心表格 |

### 5.3 预期结果和解读

| 预期结果 | 如果确认 | 对论文的意义 |
|---|---|---|
| CALM 在 native thinking 上也失败 | 高概率（已知 exit accuracy 很低） | CALM failure 不是 distillation artifact |
| Gate 在 native thinking 上也能部分恢复 | 中等概率 | Gate recovery 是 general mechanism |
| Gate recovery 幅度低于 distilled 模型 | 可能 | Distillation 可能提供额外的 regularity |
| Gate recovery 幅度相近 | 可能 | 两种 reasoning 类型的 exit dynamics 本质相似 |

**无论结果如何，都能增强论文。** 如果 CALM 也失败 + gate 也恢复 → 完全消除 distillation confound。如果 gate 恢复不同 → 提供 distillation vs native reasoning 的有趣对比。

### 5.4 GPU 和时间估算

| 步骤 | GPU 卡数 | 时间 |
|---|---|---|
| Forward pass（8B 模型，500 题 × ~3000 tokens） | 1 × 5090 | 4–6h |
| CALM evaluation | CPU | <1h |
| Tuned lens fitting | CPU | <0.5h |
| Gate training | 1 × 5090 | <1h |
| **总计** | **1 卡** | **~6–8h** |

---

## 6. P0-实验-2：Segment-level early exit baseline

### 6.1 为什么这是性价比最高的新实验

论文的核心发现是 per-token compound error 导致长 trace 上 TraceExact 崩溃。数学上：

- Per-token exit：error ≈ 1 - p^T，其中 p = per-token accuracy，T = trace length（~3000）
- 即使 p = 0.997，p^3000 ≈ 0，TraceExact → 0%

但如果按 reasoning step 做 segment-level exit：

- Segment exit：error ≈ 1 - p_s^S，其中 p_s = per-segment accuracy，S = segment 数量（~10–50）
- 由于 segment 内的 tokens 都用同一层，p_s 高于 per-token p
- S << T，所以 compound error 大幅降低

**如果实验验证了这一点，论文就从 "things are broken" 变成 "things are broken, here's why, and here's evidence for a fix direction"。** 这对 NeurIPS reviewer 来说是质的飞跃：pure negative result 难 accept，diagnostic + constructive insight 容易 accept。

### 6.2 实验设计

```text
Segment-level exit protocol:
1. 把 reasoning trace 按 reasoning step 分段
   - 分段标记：\n\n、step marker、或固定长度 window（如 64/128 tokens）
   - 如果 trace 有 N 个 segment，每个 segment 包含 ~T/N 个 tokens
2. 对每个 segment 的最后一个 token 计算 exit score（gate / CALM criterion）
3. 该 segment 内所有 tokens 使用同一个 exit layer
4. 比较 per-token exit vs segment-level exit：
   - TokenMatch、TraceExact、BoxedAcc、AnswerAcc
   - Average exit layer、FLOPs savings
   - Compound error rate
```

### 6.3 建议的对比表

**Table Y: Per-token vs segment-level exit on R1-Qwen3-8B (GSM8K, 500 questions).**

| Exit strategy | Granularity | TraceExact | BoxedAcc | AnswerAcc | Avg exit layer | Savings |
|---|---|---:|---:|---:|---:|---:|
| Full-depth | — | 100% | 84.3% | 84.3% | L35 | 0% |
| Per-token CALM (cosine, θ=0.99) | per token | ~0% | ... | ... | ... | ... |
| Per-token learned gate | per token | ... | ... | ... | ... | ... |
| **Segment-level CALM** | per step | ... | ... | ... | ... | ... |
| **Segment-level learned gate** | per step | ... | ... | ... | ... | ... |

### 6.4 多种分段策略

建议测试至少两种分段方式，展示 robustness：

| 分段方式 | 说明 | 预期 segment 数量 |
|---|---|---|
| Step-based | 按 reasoning step（\n\n 或 "Step" marker）分段 | ~10–30 |
| Fixed window | 固定 window 大小（如 64 / 128 / 256 tokens） | ~10–50 |
| Sentence-based | 按句号或换行分段 | ~20–60 |

### 6.5 GPU 和时间估算

| 步骤 | GPU 需求 | 时间 |
|---|---|---|
| 分段逻辑实现 | 无 | ~1h (agent coding) |
| 重新计算 segment-level exit decisions | 无（使用已有 hidden states 的 per-token scores） | ~1h |
| 评估 metrics | 无 | ~1h |
| **总计** | **0 GPU** | **~3h** |

**这是零 GPU 成本的实验，但可能是论文质量提升最大的单个改动。**

### 6.6 如何整合到论文

建议在主文新增一个 subsection（约 0.5–1 页）：

```text
Section 5.3: Segment-Level Exit Reduces Compound Error

Our diagnostic analysis identifies per-token compound error as the fundamental bottleneck:
even 99.7% per-token accuracy collapses to near-zero TraceExact over 3,000 tokens. A natural
mitigation is to make exit decisions at coarser granularity. We evaluate a segment-level exit
strategy that assigns a uniform exit layer to each reasoning step, reducing the number of
independent exit decisions from T ≈ 3,000 to S ≈ 20–50.

[Table Y results and analysis]

Segment-level exit substantially improves TraceExact from X% to Y% while maintaining
comparable savings, confirming that compound error—not representation quality per se—is
the primary obstacle to practical early exit on reasoning traces.
```

---

## 7. P0-实验-3：Actual autoregressive rollout

### 7.1 为什么是 P0

有了 2–4 × 5090 和 agent coding，rollout 的实现成本大幅降低。同时，做了 rollout 后论文可以保留 deployment claim，不需要降级为 pure diagnostic study。

### 7.2 实验设计

| 参数 | 值 |
|---|---|
| Model | R1-Qwen3-8B |
| Dataset | GSM8K |
| Sample size | 200 questions |
| Methods | full-depth, raw CALM (cosine θ=0.99), tuned CALM, learned gate V2a, **segment-level gate** |

注意：rollout 中也应包含 segment-level exit 方法，以验证 segment-level exit 在实际 autoregressive 生成中的效果。

### 7.3 报告指标

| Metric | 说明 |
|---|---|
| TaskAcc / AnswerAcc | 最终答案正确率 |
| RolloutDivergence | 第一次偏离 full-depth 的 token 位置 |
| Avg exit layer | 平均退出层 |
| FLOPs savings | 理论计算节省 |
| Generated length | 生成长度是否变化 |
| Failure examples | 少量定性例子 |

### 7.4 GPU 和时间估算

| 步骤 | GPU 卡数 | 时间 |
|---|---|---|
| 实现 early-exit inference pipeline | — | ~8h (agent coding) |
| 运行 5 methods × 200 questions | 2 × 5090 | ~8–12h |
| 结果分析 + 表格生成 | — | ~2h |
| **总计** | **2 卡** | **~10–14h GPU + ~10h coding** |

---

## 8. P0-实验-4：Nested split 重跑

### 8.1 当前风险

论文第 8 页脚注明确写到：

> "Per-layer affine transforms fitted on all 500 MATH-500 questions prior to the gate train/test split (2 parameters per layer)."

虽然每层仅 2 个参数（scale + bias），实际 leakage 风险有限，但有了 GPU 资源，应直接重跑 nested split 消除质疑。

### 8.2 重跑方案

```text
For each of 5 folds:
  1. Fit tuned lens on fold-train only (400 questions)
  2. Train gate on fold-train only
  3. Choose threshold on fold-val only
  4. Report on fold-test only (100 questions)
```

### 8.3 必须新增的表述

```text
All tuned-lens transformations, gate parameters, and exit thresholds are fitted without access to test questions. In 5-fold cross-validation experiments, the tuned lens and gate are re-trained within each fold; no affine transform fitted on held-out questions is used for final reporting.
```

### 8.4 关联风险：tuned lens cross-difficulty 泛化下降

Table 36 显示 tuned lens oracle 在 GSM8K（in-distribution）上为 58.2%，但在 MATH-500（OOD）上降至 38.4%。需要在 Limitations 中明确提及：

```text
Tuned-lens affine transforms show substantial accuracy degradation on harder out-of-distribution problems (Table 36: 58.2% on GSM8K vs. 38.4% on MATH-500), suggesting that the projection bottleneck is difficulty-sensitive.
```

### 8.5 GPU 和时间估算

| 步骤 | GPU 卡数 | 时间 |
|---|---|---|
| 重新收集 hidden states（如未缓存） | 1 × 5090 | 2–4h |
| 5-fold tuned lens + gate 重跑 | CPU / 1 × 5090 | ~1–2h |
| **总计** | **1 卡** | **~3–6h** |

---

## 9. P0-sub：Divergence position 分析

### 9.1 为什么几乎免费但有价值

这个分析零 GPU 成本，但能回答一个关键问题：**early exit 的错误集中在 trace 的什么位置？**

- 如果错误集中在 trace 前段（warm-up 阶段） → 可以用 "前段 full-depth + 后段 early exit" 混合策略
- 如果错误集中在 trace 后段（answer 阶段） → answer 段需要特殊处理
- 如果错误均匀分布 → 问题是系统性的，支持 segment-level exit

### 9.2 实现方案

```text
对每个 question 的每一层 L：
  1. 找到第一个 divergence position：第一个 y_L ≠ y_final 的 token
  2. 记录 divergence position 归一化到 [0, 1]（相对于 trace length）
  3. 按 token type 分组统计
  4. 可视化 divergence position 分布（histogram / CDF）
```

### 9.3 GPU 和时间估算

纯后处理，使用已有的 per-layer predictions。Agent coding ~1 小时，GPU = 0。

---

## 10. P0-写作-2：统一指标命名

### 10.1 当前问题

论文 Abstract 的 headline result 是 "yield 0% question accuracy (Q_Acc)"。Q_Acc 的实际定义（Section 3.3）是 trace-level exact match——要求一条 trace 中**每一个** token 都与 full-depth 预测一致。对于长度超过 3000 tokens 的 reasoning trace，这个指标极为严格。

问题在于：

1. **Abstract 中 "0% question accuracy" 没有附带定义**，reviewer 第一反应会理解为 "模型对所有问题都答错了"；
2. **"question accuracy" 在 ML 社区的约定俗成含义就是 "答对率"**，与论文定义冲突；
3. 论文同时报告了 boxed answer accuracy（Section S），但不叫 Q_Acc，两个指标的关系容易混淆。

### 10.2 推荐术语

| 新术语 | 定义 | 用途 |
|---|---|---|
| TokenMatch | 某层预测 token 是否等于 final-layer token | token-level diagnostic |
| TraceExact | 整条 trace 所有 token 是否与 full-depth trace 完全一致 | strict trace-level diagnostic |
| BoxedAcc / BoxedExact | boxed answer 或 final answer segment 是否一致/正确 | answer-segment diagnostic |
| AnswerAcc / TaskAcc | benchmark final answer 是否正确 | task-level metric |
| RolloutTaskAcc | actual early-exit rollout 的 final answer accuracy | deployment metric |
| RolloutDivergence | actual rollout 第一次偏离 full-depth 的位置 | error propagation analysis |

### 10.3 建议主文写法

```text
We use TraceExact as a strict diagnostic metric: a trace is counted as correct only if every early-exit token matches the full-depth token. Because this metric is intentionally stringent—even 99.2% per-token accuracy yields near-zero TraceExact over 3,000 tokens (Section 5.1)—we also report BoxedAcc (whether the final boxed answer matches) and TaskAcc (benchmark answer correctness) to assess downstream impact.
```

---

## 11. P0-写作-3：弱化 claim

### 11.1 必须替换的强 claim

| 当前论文中的表述 | 位置 | 风险 | 建议改写 |
|---|---|---|---|
| oscillatory dynamics (distillation-specific) | Abstract、Figure 1(c) | NM 9.1% (native thinking) < 11.6% (non-reasoning) | distillation-associated (strongest in distilled models; reduced below non-reasoning baseline in native thinking mode) |
| representation immaturity (reasoning-general) | Abstract、Contributions | 仅基于 Qwen 家族 | persists in both distilled and native Qwen reasoning settings |
| current reasoning models | 多处 | 有 1.5B reversal，QwQ-32B 只 diagnostic | the reasoning models studied here (8B–14B scale) |
| minimum-capacity threshold | Section 4.2 | 7B vs 8B 不是纯 scale 控制 | suggests a capacity- or architecture-dependent emergence |
| projection quality is the primary bottleneck | Section 5.2 | tuned lens 是诊断非因果；Table 36 cross-difficulty 降 | projection quality is a major bottleneck under tuned-lens diagnostics |
| deployment constraint | Conclusion | 如果做了 rollout 可以保留但需限定 scope | deployment-relevant constraint confirmed by autoregressive rollout on R1-Qwen GSM8K |
| CALM failure persists at scale | 涉及 QwQ-32B | 只测了 CALM | CALM-style diagnostics also fail on QwQ-32B in our limited evaluation |

### 11.2 建议新增 "Scope of claims" 小段

```text
Our strongest claims concern 8B–14B math/code reasoning traces under the evaluated Qwen, Llama, and Phi model families. Distilled models are used as important test cases but introduce teacher- and distillation-specific factors; native Qwen thinking-mode controls reduce but do not eliminate this concern. The 1.5B reversal (Appendix V) and the diagnostic-only 32B experiments indicate that scale and training recipe can change the observed dynamics. We therefore frame the proposed mechanisms as empirically supported failure modes rather than universal laws.
```

### 11.3 1.5B reversal 的处理策略

建议在主文 Discussion / Limitations 中明确解释：

```text
At the 1.5B scale, the reasoning–non-reasoning early-exit gap reverses (Table 42): the R1-distilled model exhibits higher exit accuracy than its non-reasoning counterpart. This reversal may reflect that at very small scales, base model representation instability dominates over reasoning-specific dynamics, or that distillation from a strong teacher provides a regularizing effect absent in the base model. This observation reinforces our framing of the identified failure modes as scale- and recipe-dependent rather than universal.
```

---

## 12. P0-写作-4：重写 Abstract / Introduction / Contributions

### 12.1 Abstract 建议草稿

需要更新以包含 segment-level exit 的 constructive result 和 rollout validation：

```text
Per-token early exit is a natural way to reduce the inference cost of long reasoning traces, but its reliability in this setting remains poorly understood. We show that standard confidence- and similarity-based exit criteria break down on reasoning traces: high token-level agreement collapses to near-zero trace-level exactness because small per-token errors compound over thousands of generated tokens.

We identify two representation-level failure modes behind this collapse. Oscillatory hidden-state trajectories create false convergence signals, while representation immaturity causes tokens to appear confident before they are correct. Native Qwen thinking-mode controls—including full CALM and learned-gate evaluation—show that these failures are not merely distillation artifacts, while distilled models amplify oscillatory dynamics.

Using tuned-lens diagnostics, we reveal a large projection bottleneck between hidden-state information and raw readout quality. Learned per-token gates partially recover trace-level reliability, but compound error remains the central obstacle. A segment-level exit strategy that makes uniform exit decisions per reasoning step substantially reduces compound error, confirming our diagnostic analysis and pointing toward coarser-grained exit mechanisms for efficient reasoning.
```

### 12.2 Introduction 建议结构

建议在当前 Introduction 基础上调整为 6 段：

1. **Reasoning models are expensive.** 长 CoT / thinking traces 提升能力，但推理成本高。
2. **Per-token early exit is a natural solution.** 解释 early exit 和已有 CALM / LayerSkip 动机。
3. **Reasoning traces are different.** 长、结构化、token role 异质、错误会累积。
4. **Research question.** Standard per-token early exit 在 reasoning traces 上是否可靠？如果失败，为什么？有没有更好的 exit 粒度？
5. **Main findings.** CALM-style criteria fail → native thinking control confirms not distilled-only → compound error is the bottleneck → segment-level exit substantially mitigates compound error。
6. **Contributions.** 简洁列出贡献。

### 12.3 Contributions 建议版本

```text
In summary, this paper makes five contributions:

1. We present a systematic diagnostic evaluation of per-token early exit on long reasoning traces. Standard confidence- and similarity-based criteria fail to maintain trace-level reliability despite high token-level agreement, revealing a sharp mismatch between token-level and trace-level metrics.

2. We identify two representation-level failure modes behind this breakdown. Oscillatory hidden-state trajectories create false convergence signals, while representation immaturity causes tokens to appear confident before they are correct. Native thinking-mode controls—with full CALM and gate evaluation—show that these failures are not merely distillation artifacts, while distilled models amplify oscillatory dynamics.

3. We diagnose a projection bottleneck using tuned lens. By separating intermediate representation availability from raw LM-head readout quality, we show that standard logit-lens projections substantially underestimate early-exit potential.

4. We evaluate learned exit gates as a proof of concept. Per-token gates partially recover trace-level reliability across reasoning model families, but residual compound error over long traces remains a major obstacle.

5. We show that segment-level exit—making uniform exit decisions per reasoning step rather than per token—substantially reduces compound error, validating our diagnostic analysis and pointing toward coarser-grained exit mechanisms as a promising direction for efficient reasoning.
```

---

## 13. P1：QwQ-32B gate 实验

### 13.1 为什么值得做

当前 QwQ-32B（32B，64 层）只做了 CALM diagnostic，没有 gate 实验。如果 gate 在 32B 上也能 partially recover → 显著增强 scale claim；如果 CALM failure + gate recovery 的 pattern 在 32B 上也成立 → 论文结论更 robust。

### 13.2 GPU 和时间估算

| 步骤 | GPU 卡数 | 时间 |
|---|---|---|
| QwQ-32B forward pass（INT8 量化可单卡，FP16 需 2–3 卡） | 2–3 × 5090 | 6–8h |
| Gate training | 1 × 5090 | ~1h |
| **总计** | **2–3 卡** | **~7–9h** |

---

## 14. P1：其他有余力做的实验与前移

### 14.1 P1-高：Boxed-answer analysis 前移主文

**论文已有数据：** Appendix S + Table 37。不需要新实验。只需把 Table 37 关键行前移主文。

### 14.2 P1-高：Compound error Figure 8 前移

**论文已有数据：** Appendix W Figure 8 + Table 45。前移到主文或增强引用。

### 14.3 Warm-up 混合策略实验

如果 divergence position 分析显示错误集中在 trace 前段，可以测试：

```text
Warm-up hybrid strategy:
  - 前 K tokens（如前 10% 或前 200 tokens）使用 full-depth
  - 后续 tokens 使用 early exit（per-token 或 segment-level）
  - 比较不同 K 值对 TraceExact / BoxedAcc 的影响
```

GPU 成本：0–2h（可用已有 hidden states 后处理）。

### 14.4 CALM n=50 → n=500 扩展

当前 softmax CALM 部分样本较小。扩到 n=300–500 增强可信度。GPU 成本：2–4h / 1 卡。

### 14.5 compact cross-architecture summary

把已有 R1-Llama / Phi / QwQ 结果整理成一张 compact table。不需要 GPU。

---

## 15. 不建议五天内做的事情

### 15.1 大量新增 benchmark

不建议新增 GPQA、AMC、更多 AIME 年份、MBPP 等。原因：answer extraction 引入噪声，不能直接修复核心问题。

### 15.2 外部 baseline

不建议实现 LayerSkip、speculative decoding、self-speculative decoding 等。原因：工程成本高，公平比较难，容易把论文拉回 method benchmark。

### 15.3 wall-clock speedup

如果没有成熟优化实现，不建议报告 wall-clock speedup。可以只报告 analytical FLOPs savings。

---

## 16. 推荐五天排期

### Day 1：并行启动所有 coding + 基础 GPU 实验

```
Agent A (写作):                    Agent B (coding):                    GPU:
├─ 全文 Q_Acc → TraceExact         ├─ 实现 rollout pipeline              ├─ 卡1: Qwen3-8B native thinking
├─ 弱化 claims (Section 11.1)      ├─ 实现 segment-level exit logic      │  forward pass (收集 hidden states)
├─ 从 Table 41 整理三方对照表      ├─ 实现 native thinking CALM/gate     ├─ 卡2: nested split 重跑
├─ 重写 Abstract                   │  evaluation code                    ├─ 卡3: CALM n 扩展
└─ 前移 Table 37 + Figure 8        ├─ 实现 divergence position 分析      └─ 卡4: QwQ-32B forward pass
                                   └─ 准备 QwQ-32B gate training              (如有 4 卡)
```

Day 1 产出：
- 主文写作改动 ~70% 完成（指标、claims、叙事框架）
- Native thinking hidden states 收集完毕
- Nested split 重跑完毕
- Rollout pipeline 基本可用
- Divergence position 分析完毕（零 GPU，当天出结果）

### Day 2：核心 GPU 实验

```
Agent A (整合):                    Agent B (实验):                      GPU:
├─ 整合 nested split 新结果         ├─ 运行 segment-level exit            ├─ 卡1-2: rollout 实验
├─ 撰写 native thinking             │  on all models                     │  (R1-Qwen3-8B × GSM8K × 5 methods)
│  完整协议分析                     ├─ 运行 native thinking              ├─ 卡3: native thinking
├─ 撰写 segment-level exit          │  CALM + gate evaluation            │  CALM + gate evaluation
│  新章节初稿                       ├─ 分析 divergence position           └─ 卡4: QwQ-32B gate training
└─ 更新 Contributions               │  结果，决定是否做 warm-up
                                   └─ 格式化所有结果表格
```

Day 2 产出：
- Native thinking 完整协议结果（CALM failure + gate recovery 数据）
- Segment-level exit 结果
- Rollout 实验结果
- QwQ-32B gate 结果
- 所有新实验的结果表格

### Day 3：结果整合 + 主文定稿

```
Agent A (写作):                    Agent B (补充):                      GPU:
├─ 整合所有实验结果到主文           ├─ warm-up 策略实验                   ├─ 卡1-2: warm-up 验证
├─ 完成 segment-level exit          │  (如 Day 2 结果支持)               │  (如需要)
│  新章节                           ├─ 整理所有补充表格                   └─ 卡3-4: 机动
├─ 重写 Intro/Contributions        └─ 生成所有图表
├─ 完善 Discussion + Limitations
├─ 1.5B reversal 解释
└─ tuned lens cross-diff 说明
```

Day 3 产出：
- 主文包含所有新实验结果
- Segment-level exit 章节完成
- 完善的 Discussion + Limitations

### Day 4：全文整合 + 质量保障

```
Agent A (写作):                    Agent B (验证):
├─ 全文 claim-evidence alignment    ├─ 数字一致性检查
├─ 确保所有新数据的 caption 完整     ├─ 表格交叉引用检查
├─ 调整主文结构和页数               ├─ PDF 编译测试
├─ 统一图表风格                    └─ 确认所有 split 描述一致
└─ 撰写 scope of claims 段落
```

### Day 5：最终 polish + 提交准备

```
Agent A (写作):                    Agent B (格式):
├─ 最终通读全文                     ├─ NeurIPS 格式检查
├─ 摘要/结论最终定稿               ├─ checklist 完善
├─ 确认页数符合限制                 ├─ 匿名检查
└─ appendix 整理排序               └─ supplementary 准备
```

---

## 17. GPU 资源使用估算

| 实验 | GPU 卡数 | 时长 | 运行天 |
|---|---|---|---|
| Native thinking forward pass | 1 × 5090 | 4–6h | Day 1 |
| Nested split 重跑 | 1 × 5090 | 3–6h | Day 1 |
| CALM n 扩展 | 1 × 5090 | 2–4h | Day 1 |
| QwQ-32B forward pass + gate | 2–3 × 5090 | 7–9h | Day 1–2 |
| Rollout（5 methods × 200q） | 2 × 5090 | 8–12h | Day 2 |
| Native thinking CALM + gate eval | 1 × 5090 | 2–3h | Day 2 |
| Segment-level exit | 0（后处理） | ~3h | Day 2 |
| Divergence position analysis | 0（后处理） | ~1h | Day 1 |
| Warm-up 策略（可选） | 1 × 5090 | 2–4h | Day 3 |
| **总计** | **峰值 4 卡** | **~30–50 GPU·h** | **Day 1–3** |

4 卡 5090 在 Day 1–3 内完全够用，Day 4–5 不需要 GPU。

---

## 18. 主文结构建议

建议在当前结构基础上做**增量式调整**，主要增加 segment-level exit 分析：

| 当前结构 | 建议调整 | 变化幅度 |
|---|---|---|
| 1. Introduction | 增加过渡段（Section 12.2 的 6 段结构），修改 Contributions 为 5 条 | 中 |
| 2. Related Work | 不变 | 无 |
| 3. Methodology (3.1–3.3) | 3.3 改指标名；增加 split 协议说明 | 小 |
| 4. Findings (4.1–4.4) | **4.4 扩展为有表格的完整分析**（含 CALM + gate on native thinking） | 中 |
| 5. End-to-End (5.1–5.2) | 前移 boxed answer 到 5.1；前移 Figure 8；**新增 5.3 Segment-Level Exit** | **中-大** |
| 6. Conclusion | 扩展为 Discussion + Limitations，增加 scope caveat、1.5B reversal、rollout 结果 | 中 |

**核心变化有两个：(1) Section 4.4 扩展为完整 native thinking 分析（含 CALM + gate）；(2) 新增 Section 5.3 segment-level exit 分析。** 其余是措辞修正、数据前移和 caveat 补充。

注意：新增内容约 1–1.5 页，当前主文 9 页。NeurIPS 2026 主文上限需确认（通常 9–10 页 + references）。如果空间不够，可以：
- 压缩 Section 4.1–4.3（findings 的详细数字移到附录）
- 或将 segment-level exit 的详细表格放附录，主文只保留关键结论

---

## 19. 主文图表建议

### Table 1：Models and comparison design（扩展当前表格）

增加列标明 native / distilled / diagnostic-only 状态。

### Table 2（新增/前移）：Native thinking 完整对照

从 Table 41 + 新实验结果整合：

| Model | Condition | TokenMatch | TraceExact | BoxedAcc | NM rate | CALM fail? | Gate recovery |
|---|---|---:|---:|---:|---:|---|---|
| Qwen3-8B | non-thinking | ... | ... | ... | 11.6% | ... | ... |
| Qwen3-8B | native thinking | ... | ... | ... | 9.1% | **新数据** | **新数据** |
| R1-Distill-Qwen-8B | distilled | ... | ... | ... | 20.8% | yes | yes |

### Table 3（前移）：TraceExact + BoxedAcc + AnswerAcc 多层级对比

从 Table 37 前移关键行。

### Table 4（新增）：Per-token vs segment-level exit

Section 6.3 的 Table Y。

### Table 5（新增，如做了 rollout）：Actual rollout results

Section 7.3 的 rollout 结果表。

### Figure 1：Problem and failure modes（不变）

### Figure 2：CALM failure and token-to-trace gap（微调 axis label）

### Figure 3（前移）：Compound error decay（Figure 8 前移或新制作）

### Figure 4：Projection bottleneck and recovery（不变）

### Figure 5（新增，可选）：Divergence position 分布

---

## 20. 最终投稿前 checklist

### Scientific checklist

- [ ] Qwen3 non-thinking / native thinking / R1-Distill-Qwen 三方表格已在主文出现，**包含 CALM 和 gate 结果**；
- [ ] Native thinking 上的 CALM failure 和 gate recovery 已验证，确认不是 distillation artifact；
- [ ] Segment-level exit baseline 已在主文出现，证明 coarser-grained exit 降低 compound error；
- [ ] tuned lens / gate 使用 nested split（策略 B），无 test leakage；
- [ ] Q_Acc 已改名为 TraceExact，全文一致；
- [ ] BoxedAcc / AnswerAcc 已在主文报告；
- [ ] Actual rollout 结果已报告（如做了）；
- [ ] learned gate 被定位为 proof-of-concept；
- [ ] Abstract 中 "distillation-specific" 已改为 "distillation-associated"；
- [ ] Abstract 中 "reasoning-general" 已限定 scope；
- [ ] 1.5B reversal 在 Discussion/Limitations 中有明确解释；
- [ ] tuned lens cross-difficulty 下降在 Limitations 中有说明；
- [ ] oracle 和 deployable method 被清楚区分；
- [ ] sample sizes 在正文、表格、caption 中一致。

### Narrative checklist

- [ ] Abstract 包含 segment-level exit 的 constructive result；
- [ ] Abstract 先讲问题，再讲发现，不堆过多数字；
- [ ] Introduction 包含 research question 关于 exit 粒度；
- [ ] Contributions 为 5 条（增加 segment-level exit）；
- [ ] Limitations 诚实说明 diagnostic scope、single-architecture native thinking、1.5B reversal；
- [ ] 主文每个实验都回答明确问题。

### Submission checklist

- [ ] NeurIPS 2026 官方 style；
- [ ] 主文页数符合要求（注意新增内容的空间）；
- [ ] references / appendix / checklist 顺序正确；
- [ ] 匿名信息全部去除；
- [ ] PDF 可编译、图表清晰、caption 完整。

---

## 21. 最终建议

这篇论文经过五天改进后的定位应该是：

> **我们系统性分析了为什么 standard per-token early exit 在 reasoning traces 上失败，识别了 compound error、oscillatory convergence、representation immaturity 和 projection bottleneck 等关键机制，并展示了 segment-level exit 作为更有效替代方案的初步证据。**

五天内的最优策略是并行执行六条线：

1. **前移已有 native thinking control 数据 + 补完 CALM/gate 评估，彻底修复 distilled confound；**
2. **新增 segment-level exit baseline，从 pure negative result 变成 diagnostic + constructive insight；**
3. **做 actual autoregressive rollout，支撑 deployment claim；**
4. **重跑 nested split，消除 methodology risk；**
5. **全文 TraceExact 替换 + claim 弱化 + 叙事重写，修复 presentation 和 overclaim risk；**
6. **前移 boxed answer + compound error + divergence position 数据，充分利用已有附录。**

### 预估质量提升

| 方案 | 预估均分 | 区间 |
|---|---|---|
| 当前论文 | ~4.5–5 | borderline reject |
| 仅 P0 写作改动 | ~6–6.5 | weak accept |
| 写作 + rollout + nested split | ~7–7.5 | accept |
| **写作 + 全部实验（含 segment exit + native thinking full + rollout + QwQ gate）** | **~7.5–8** | **accept** |

**核心认识：写作重组贡献 ~60% 的质量提升（从 ~5 到 ~6.5），GPU 实验贡献 ~40% 的质量提升（从 ~6.5 到 ~7.5–8）。其中 segment-level exit 是 ROI 最高的单个改动——零 GPU 成本，但把论文从 negative result 变成 constructive insight。**
