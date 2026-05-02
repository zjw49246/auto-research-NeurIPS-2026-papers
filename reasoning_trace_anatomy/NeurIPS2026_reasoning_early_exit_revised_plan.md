# NeurIPS 2026 论文叙事与实验改进方案（修订版 v4）

**论文：** *The Convergence Cost of Reasoning: Why Per-Token Early Exit Underperforms*  
**目标：** 在投稿前完成剩余关键实验 + 写作微调，最大化论文可信度和审稿通过概率。  
**可用资源：** 2–4 × RTX 5090（32GB/卡）、agent 驱动的并行 coding 和写作。  
**建议定位：** Diagnostic Study + Constructive Insight。  
**基于论文版本：** 2026-05-02 版（相比 v3 计划基于的 0430 版有大量改进）。

---

## 0. v3 → v4 变更摘要

v4 基于 2026-05-02 新版论文进行了全面对比审计。新版论文已完成 v3 计划中的多个重要任务，同时新增了计划未覆盖的高价值内容。以下是关键变更：

### ✅ 已完成的 v3 计划项目

| 原计划项目 | 新版论文实现 | 完成质量 |
|---|---|---|
| **P0-实验-4：Nested split 重跑** | 脚注2（p.8）：affine transforms fitted on fold's 400 training questions。Section 5.2 报告 nested vs non-nested 对比：savings 20.2%→6.28%, Q_Acc 68.2%→63.0% | 完全完成，高质量 |
| **P1：QwQ-32B gate 实验** | Section 5.2 + Appendix V Table 48：V2a gate 61.0%±2.4% Q_Acc at 4.98%±0.44% savings（5-fold nested CV, n=500） | 完全完成，高质量 |
| **P1：compact cross-architecture summary** | Section 5.2 "Cross-architecture V2a transfer" 整合 R1-Llama (55.0%), Phi-4 (48.0%), QwQ-32B (61.0%) | 完全完成 |
| **P1：CALM n 扩展** | Table 4 cosine-sim 已用 n=500；softmax criteria 仍为 n=50 但有充分说明 | 大部分完成 |

### 🆕 新版论文新增的高价值内容（计划未覆盖）

| 新内容 | 位置 | 意义 |
|---|---|---|
| **Effect decomposition: leakage vs grid density** | Table 5 | 量化 leakage=88%, grid density=12%，出色的方法论分析 |
| **Exit layer grid sensitivity** | Table 6 | 7-layer vs 9-layer grid 的 Q_Acc 差异 + gate val_loss 分析 |
| **R1-Distill-Qwen-7B gate 失败** | Section 5.2 | val_loss≈0.50（chance level），重要 negative result |
| **Phi-4-reasoning-14B cross-architecture gate** | Appendix V Table 49 | 48.0%±7.3% Q_Acc, 12.70% savings |
| **Cross-benchmark V2a transfer (GSM8K)** | Appendix Z | Q_Acc=86.4%，shorter traces 降低 compound error |
| **TIDE concurrent work discussion** | Section 2 | 与 TIDE 明确对比（complementary positioning） |
| **Gap-closing progression** | Section 5.2 | 四层递进：raw → tuned CALM → V2a gate → oracle ceiling |
| **Feature importance analysis** | Appendix P | 确认 tuned lens confidence features 主导 gating |
| **Gate architecture ablation** | Appendix P.1 Table 36 | 2-layer MLP 最优，cross-model 一致性 |

### 📊 优先级调整

| 项目 | v3 优先级 | v4 优先级 | 原因 |
|---|---|---|---|
| Nested split 重跑 | P0-实验 | ✅ 已完成 | 新版已实现 nested CV |
| QwQ-32B gate | P1 | ✅ 已完成 | 新版已有 5-fold nested CV 结果 |
| Q_Acc → TraceExact 重命名 | P0-写作 | **P1-写作** | Section 3.3 已有明确定义（"requires all tokens to match"），主文多处引用 Appendix S boxed-answer analysis，误解风险已降低 |
| Abstract 重写 | P0-写作 | **P1-写作** | 新版 Abstract 已大幅改善，有 gap-closing progression 叙事 |
| Boxed-answer 前移主文 | P0-写作 | **P1-写作** | 关键数据点已在主文 Section 3.3 和 5.1 引用 |
| "distillation-specific" 弱化 | P0-写作 | **P0-minor** | 已部分改进（Contributions 限定 "on one architecture"），仅需局部微调 |
| Native thinking CALM+gate | P0-实验 | **P0-实验（不变）** | 仍是最大方法论盲区 |
| Segment-level exit | P0-实验 | **P0-实验（不变）** | 仍未做，最高 ROI |
| Autoregressive rollout | P0-实验 | **P0-实验（不变）** | 仍未做 |

---

## 1. 总体判断（v4 更新）

论文在 05-02 版中取得了**显著进展**：

**已修复的问题：**
- ✅ Nested CV 完全实现，消除了 data leakage 风险
- ✅ QwQ-32B gate 实验完成，增强了 scale claim
- ✅ Cross-architecture V2a transfer 整合到主文
- ✅ Effect decomposition（Table 5）和 grid sensitivity（Table 6）提供了出色的方法论透明度
- ✅ R1-Distill-Qwen-7B 和 non-reasoning control（Qwen2.5-7B-Instruct）的 gate failure 增强了 reasoning-specificity 论证
- ✅ Contributions 已限定 scope（"on one architecture"）
- ✅ TIDE concurrent work 讨论到位
- ✅ Gap-closing progression（Section 5.2）叙事清晰

**仍存在的核心缺口：**

1. **Native thinking 只有 NM rate 和 exit accuracy，没有 CALM/gate 评估。** Section 4.4 仍然只有约 8 行。论文在 distilled 模型上做了完整的 CALM failure → gate recovery 链条，但在 native thinking 上只做了一半。Reviewer 可以说 "CALM failure 和 gate recovery 可能都是 distillation artifact"。
2. **没有 segment-level exit 实验。** Section 7 Limitations 提到 segment-level methods 作为 promising direction，但没有任何实验验证。这是从 "things are broken" 到 "things are broken, here's why, and here's evidence for a fix direction" 的关键。
3. **没有 actual autoregressive rollout。** 所有分析仍基于 frozen teacher traces。
4. **Q_Acc 命名仍可能引起误解。** 虽然 Section 3.3 定义清晰，但 Abstract 中 "near-zero question accuracy (Q_Acc)" 仍容易被误读为 "模型答对率为零"。

### 核心策略（v4 更新）

> **新版论文的写作和方法论基础已经大幅改善。剩余工作集中在三个关键实验缺口：(1) native thinking CALM+gate 完整协议，(2) segment-level exit baseline，(3) autoregressive rollout。写作方面只需微调而非大改。GPU 资源需求比 v3 计划减少约 40%（nested split 和 QwQ-32B 已完成）。**

---

## 2. 最终推荐主线（不变）

建议把论文主线收束为：

> **Standard per-token early exit fails on long reasoning traces because token-level reliability does not translate into trace-level or task-level reliability. Native thinking controls—including full CALM and learned-gate evaluation—confirm that these failures are not merely distillation artifacts. We identify compound error over thousands of tokens as the fundamental bottleneck, and show via tuned-lens diagnostics that a large projection gap exists between hidden-state information and raw readout quality. Learned per-token gates partially recover performance but cannot eliminate compound error. A segment-level exit strategy, which makes uniform exit decisions per reasoning step rather than per token, substantially reduces the compound-error burden, validating our diagnostic analysis and pointing toward coarser-grained exit mechanisms.**

四层递进：

1. **Reasoning traces 本身使 early exit 更难（native thinking 确认）；**
2. **Compound error 是根本瓶颈（p^T 模型 + rollout 验证）；**
3. **Per-token gate 能部分缓解但不能解决 compound error；**
4. **Segment-level exit 显著降低 compound error，验证 diagnostic insight 并指向解决方向。**

---

## 3. 优先级总表（v4 更新）

| 优先级 | 任务 | 工作性质 | 必须完成？ | 重要程度 | 主要收益 | 预估分数提升 |
|---|---|---|---|---|---|---|
| **P0-实验** | Native thinking 完整协议（CALM + gate on Qwen3-8B thinking） | 新实验 | 是 | 极高 | 彻底堵住 CALM/gate 层面的 distillation confound | +0.5–1.0 |
| **P0-实验** | Segment-level exit baseline | 新实验 | 是 | 极高 | 从 negative result → constructive insight | +0.5–1.0 |
| **P0-实验** | Actual autoregressive rollout | 新实验 | 是 | 极高 | 支撑 deployment claim | +0.5–1.0 |
| **P0-写作** | Section 4.4 扩展（整合 native thinking CALM+gate 新结果） | 重组+新数据 | 是 | 极高 | 修复 distilled confound 叙事 | +0.5 |
| **P0-写作** | 新增 Section 5.3 Segment-Level Exit | 新章节 | 是 | 极高 | 核心 constructive contribution | +0.5 |
| **P0-minor** | 弱化 Section 4.4 中 "distillation-specific" 表述 | 措辞微调 | 是 | 高 | 提升可信度 | +0.2 |
| P0-sub | Divergence position 分析 | 零成本后处理 | 是 | 高 | 为 segment exit 提供量化支撑 | +0.2 |
| P1-写作 | Q_Acc → TraceExact 全文重命名 | 全文替换 | 建议做 | 中高 | 避免 reviewer 误解 headline result | +0.3 |
| P1-写作 | Abstract 进一步精简数字 | 措辞修改 | 建议做 | 中 | 改善 readability | +0.1 |
| P1-写作 | Compound error Figure 8 前移主文 | 前移已有数据 | 建议做 | 中 | 强化核心解释 | +0.1–0.2 |
| P1-写作 | Boxed-answer Table 41 关键行前移 | 前移已有数据 | 建议做 | 中 | 缓解 TraceExact 过严质疑 | +0.1–0.2 |
| P1 | Warm-up 混合策略实验 | 新实验 | 有余力做 | 中 | 进一步验证 constructive direction | +0.1–0.2 |
| ~~P0-实验~~ | ~~Nested split 重跑~~ | ~~重跑实验~~ | — | — | ✅ **已完成** | — |
| ~~P1~~ | ~~QwQ-32B gate 实验~~ | ~~新实验~~ | — | — | ✅ **已完成** | — |
| ~~P1~~ | ~~compact cross-architecture summary~~ | ~~整理已有数据~~ | — | — | ✅ **已完成** | — |

---

## 4. P0-实验-1：Native thinking 完整协议（CALM + gate）

### 4.1 为什么仍是最高优先级

新版论文在 Section 4.4 中仍然只有约 8 行，只报告了 NM rate 和 exit accuracy：

> "A native thinking mode ablation (Qwen3-8B only; Llama-3.1-8B lacks a native thinking mode, n=499) decomposes the two failure modes. Oscillatory dynamics are distillation-specific: thinking NM 9.1% is below non-reasoning 11.6%, compared to R1's 20.8% (Table 2)—the elevated oscillation is absent without distillation. Representation immaturity persists without distillation (L28 exit accuracy 34.7% ≈ R1's 32.4%, both ≪ non-reasoning 47.7%; Appendix U), confirming it as reasoning-general."

**缺失：**
- ❌ 没有在 native thinking traces 上跑 CALM criteria
- ❌ 没有在 native thinking traces 上训练 gate
- ❌ Reviewer 仍然可以说 "CALM failure 和 gate recovery 可能都是 distillation artifact"

新版论文已经在 R1-Distill-Qwen-7B 和 Qwen2.5-7B-Instruct 上证明 gate 失败（val_loss≈0.50），这**更加强调**了在 native thinking 上测试 gate 的重要性——如果 gate 在 native thinking 上也成功恢复，就能完全排除 distillation artifact 的质疑。

### 4.2 实验设计（与 v3 相同）

在 Qwen3-8B native thinking traces 上完整复制 distilled 模型的分析流程：

| 步骤 | 内容 | 输出 |
|---|---|---|
| 1. Forward pass | 收集 Qwen3-8B thinking 模式在 MATH-500 上的 per-layer hidden states | hidden states cache |
| 2. CALM evaluation | 计算 cosine similarity / softmax confidence exit criteria | CALM failure/success 表 |
| 3. Tuned lens | 拟合 per-layer affine transform（在 train split 上） | tuned lens oracle gap |
| 4. Gate training | 训练 learned gate V2a（nested CV） | gate recovery 结果 |
| 5. 三方对比表 | 与 Qwen3 non-thinking 和 R1-Distill-Qwen 对比 | 主文核心表格 |

### 4.3 预期结果

| 预期结果 | 如果确认 | 对论文的意义 |
|---|---|---|
| CALM 在 native thinking 上也失败 | 高概率（已知 exit accuracy 很低） | CALM failure 不是 distillation artifact |
| Gate 在 native thinking 上也能部分恢复 | 中等概率 | Gate recovery 是 general mechanism |
| Gate recovery 幅度与 distilled 模型不同 | 可能 | 提供有趣的 distillation vs native reasoning 对比 |

### 4.4 GPU 和时间估算

| 步骤 | GPU 卡数 | 时间 |
|---|---|---|
| Forward pass（8B 模型，500 题） | 1 × 5090 | 4–6h |
| CALM evaluation | CPU | <1h |
| Tuned lens fitting | CPU | <0.5h |
| Gate training（nested CV） | 1 × 5090 | <1h |
| **总计** | **1 卡** | **~6–8h** |

---

## 5. P0-实验-2：Segment-level early exit baseline

### 5.1 为什么仍是性价比最高的新实验

新版论文 Section 7 Limitations 明确提到：

> segment-level methods 作为 promising direction

但没有任何实验验证。论文的核心发现是 per-token compound error 导致长 trace 上 Q_Acc 崩溃（Table 50: p=0.999 时 T=3000 仍然只有 5.0% Q_Acc）。Segment-level exit 通过减少独立 exit 决策次数（从 T≈3000 到 S≈20–50）来大幅降低 compound error。

**如果实验验证了这一点，论文就从 "things are broken" 变成 "things are broken, here's why, and here's evidence for a fix direction"。**

### 5.2 实验设计（与 v3 相同）

```text
Segment-level exit protocol:
1. 把 reasoning trace 按 reasoning step 分段
   - 分段标记：\n\n、step marker、或固定长度 window（如 64/128 tokens）
2. 对每个 segment 的最后一个 token 计算 exit score（gate / CALM criterion）
3. 该 segment 内所有 tokens 使用同一个 exit layer
4. 比较 per-token exit vs segment-level exit：
   - TokenMatch, Q_Acc, BoxedAcc
   - Average exit layer, FLOPs savings
```

### 5.3 建议的对比表

**Table Y: Per-token vs segment-level exit on R1-Qwen3-8B (MATH-500).**

| Exit strategy | Granularity | Q_Acc | BoxedAcc | Avg exit layer | Savings |
|---|---|---:|---:|---:|---:|
| Full-depth | — | 100% | 84.3% | L35 | 0% |
| Per-token CALM (cosine, θ=0.99) | per token | 29.4% | ... | ... | 11.4% |
| Per-token learned gate (V2a, γ=0.99) | per token | 63.0% | ... | ... | 6.28% |
| **Segment-level CALM** | per step | ... | ... | ... | ... |
| **Segment-level learned gate** | per step | ... | ... | ... | ... |

### 5.4 GPU 和时间估算

| 步骤 | GPU 需求 | 时间 |
|---|---|---|
| 分段逻辑实现 | 无 | ~1h (agent coding) |
| 重新计算 segment-level exit decisions | 无（使用已有 hidden states） | ~1h |
| 评估 metrics | 无 | ~1h |
| **总计** | **0 GPU** | **~3h** |

**零 GPU 成本，但可能是论文质量提升最大的单个改动。**

---

## 6. P0-实验-3：Actual autoregressive rollout

### 6.1 为什么仍是 P0

新版论文所有分析仍基于 frozen teacher traces。做了 rollout 后论文可以保留 deployment claim。

### 6.2 实验设计

| 参数 | 值 |
|---|---|
| Model | R1-Qwen3-8B |
| Dataset | GSM8K |
| Sample size | 200 questions |
| Methods | full-depth, raw CALM (cosine θ=0.99), tuned CALM, learned gate V2a, **segment-level gate** |

### 6.3 报告指标

| Metric | 说明 |
|---|---|
| TaskAcc / AnswerAcc | 最终答案正确率 |
| RolloutDivergence | 第一次偏离 full-depth 的 token 位置 |
| Avg exit layer | 平均退出层 |
| FLOPs savings | 理论计算节省 |

### 6.4 GPU 和时间估算

| 步骤 | GPU 卡数 | 时间 |
|---|---|---|
| 实现 early-exit inference pipeline | — | ~8h (agent coding) |
| 运行 5 methods × 200 questions | 2 × 5090 | ~8–12h |
| 结果分析 | — | ~2h |
| **总计** | **2 卡** | **~10–14h GPU + ~10h coding** |

---

## 7. P0-sub：Divergence position 分析

### 7.1 为什么几乎免费但有价值

零 GPU 成本，回答关键问题：**early exit 的错误集中在 trace 的什么位置？**

- 如果错误集中在 trace 前段 → 支持 warm-up 混合策略
- 如果错误均匀分布 → 支持 segment-level exit
- 无论结果如何都为 segment-level exit 提供量化支撑

### 7.2 实现方案

```text
对每个 question 的每一层 L：
  1. 找到第一个 divergence position：第一个 y_L ≠ y_final 的 token
  2. 记录 divergence position 归一化到 [0, 1]
  3. 可视化 divergence position 分布（histogram / CDF）
```

GPU = 0，Agent coding ~1 小时。

---

## 8. P0-写作：Section 4.4 扩展 + claim 微调

### 8.1 Section 4.4 扩展（依赖 P0-实验-1 结果）

当前 Section 4.4 只有 ~8 行。在 native thinking CALM+gate 实验完成后，应扩展为包含完整三方对比表格的 subsection（约 0.5 页）。

建议新增主文表格：

**Table X: Native thinking-mode control with full CALM and gate evaluation.**

| Model | Condition | NM rate | Exit Acc L28 | CALM Q_Acc | Gate Q_Acc | Gate Savings |
|---|---|---:|---:|---:|---:|---:|
| Qwen3-8B | non-thinking | 11.6% | 47.7% | ... | ... | ... |
| Qwen3-8B | native thinking | 9.1% | 34.7% | **新数据** | **新数据** | **新数据** |
| R1-Distill-Qwen-8B | distilled | 20.8% | 32.4% | 0% | 63.0% | 6.28% |

### 8.2 Claim 微调

新版论文已经在 Contributions 中限定了 scope（"on one architecture"），但 Section 4.4 中仍然使用 "distillation-specific" 这一较强的措辞。建议微调为：

**当前：**
> Oscillatory dynamics are distillation-specific

**建议改为：**
> Oscillatory dynamics are distillation-associated (thinking NM 9.1% is below non-reasoning 11.6%; the elevated oscillation in R1-Qwen at 20.8% is absent without distillation)

这个改动非常小（一个词的替换），但能更准确地反映数据。

### 8.3 新增 Section 5.3: Segment-Level Exit

在当前 Section 5.2 之后新增：

```text
Section 5.3: Segment-Level Exit Reduces Compound Error

Our diagnostic analysis identifies per-token compound error as the fundamental bottleneck:
even 99.7% per-token accuracy collapses to near-zero Q_Acc over 3,000 tokens (Table 50).
A natural mitigation is to make exit decisions at coarser granularity. We evaluate a
segment-level exit strategy that assigns a uniform exit layer to each reasoning step,
reducing the number of independent exit decisions from T ≈ 3,000 to S ≈ 20–50.

[Table Y results and analysis]
```

---

## 9. P1-写作：可选但建议的改进

### 9.1 Q_Acc → TraceExact 重命名

**v4 优先级下调理由：** 新版 Section 3.3 已有非常清晰的定义（"requires all tokens in a question's reasoning trace to match"），且主文 Section 5.1 明确引用了 Appendix S 的 boxed-answer analysis（"evaluating only the final boxed answer, accuracy still drops from 84.3% to <2% at L32"）。误解风险已大幅降低。

**仍建议做的理由：** Abstract 中 "near-zero question accuracy (Q_Acc)" 作为 headline result，reviewer 快速阅读时仍可能误解为 "模型答对率为零"。如果改为 TraceExact，含义自明。

### 9.2 Abstract 进一步精简

新版 Abstract 已经比旧版好很多（有 gap-closing progression），但仍然 number-heavy（Q_Acc, 48-63%, ~5-10%, +45.6pp 等）。如果有时间，可以进一步精简。

### 9.3 Compound error Figure 8 前移

Figure 8（Compound Error vs. Trace Length）非常直观，前移到主文能强化 compound error 作为核心瓶颈的叙事。但新版主文已有 Table 50 的关键数据点引用，所以不是必须。

### 9.4 Boxed-answer Table 41 关键行前移

新版主文 Section 5.1 已引用了 Appendix S 的关键结论。如果空间允许，可以把 Table 41 中 L28/L32 的 reasoning/answer/boxed segment 数据前移。

---

## 10. 不再需要的任务

以下 v3 计划项目在新版论文中已经完成，不再需要执行：

### 10.1 ~~P0-实验-4：Nested split 重跑~~ ✅

新版论文脚注2："For each cross-validation fold, affine transforms are fitted on the fold's 400 training questions, ensuring no test-set information leaks into the tuned lens."

Section 5.2 报告了完整的 nested vs non-nested 对比：
- Savings: 20.2% → 6.28%（-13.92pp）
- Q_Acc: 68.2% → 63.0%（-5.2pp）
- Table 5 进一步分解为 leakage (88%) vs grid density (12%)
- Gate val_loss 从 ~0.35 (leaky) 到 ~0.46 (clean) 确认 pattern

### 10.2 ~~P1：QwQ-32B gate 实验~~ ✅

新版 Section 5.2 + Appendix V Table 48：
- V2a gate: 61.0%±2.4% Q_Acc at 4.98%±0.44% savings（γ=0.99）
- 5-fold nested CV, n=500
- CALM failure at 32B confirmed（τ=0.95: 0% savings; τ=0.90: Q_Acc=2.4%）
- Comparable Q_Acc to 8B R1-Qwen（63.0%），lower savings due to 64-layer grid sparsity

### 10.3 ~~P1：compact cross-architecture summary~~ ✅

新版 Section 5.2 "Cross-architecture V2a transfer" 已整合：
- R1-Llama-8B: 55.0%±8.9% Q_Acc at 9.41% savings
- Phi-4-reasoning-14B: 48.0%±7.3% Q_Acc at 12.70% savings
- QwQ-32B: 61.0%±2.4% Q_Acc at 4.98% savings
- Val_loss 分析区分成功（<0.50）vs 失败（≈0.50）的模型

---

## 11. 推荐排期（v4 更新）

v4 排期比 v3 更集中，因为 nested split 和 QwQ-32B 已完成。

### Day 1：并行启动 coding + GPU 实验

```
Agent A (写作):                    Agent B (coding):                    GPU:
├─ 微调 Section 4.4 claim          ├─ 实现 segment-level exit logic      ├─ 卡1: Qwen3-8B native thinking
│  ("distillation-specific" →      ├─ 实现 native thinking CALM/gate     │  forward pass (收集 hidden states)
│  "distillation-associated")      │  evaluation code                    ├─ 卡2-3: 开始 rollout pipeline
├─ 准备 Section 5.3 框架           ├─ 实现 divergence position 分析          implementation
└─ 可选: Q_Acc → TraceExact        └─ 实现 rollout pipeline
```

Day 1 产出：
- Claim 微调完成
- Native thinking hidden states 收集完毕
- Divergence position 分析完毕（零 GPU，当天出结果）
- Segment-level exit code 可用
- Rollout pipeline 基本可用

### Day 2：核心 GPU 实验

```
Agent A (整合):                    Agent B (实验):                      GPU:
├─ 撰写 native thinking            ├─ 运行 segment-level exit            ├─ 卡1: native thinking
│  完整协议分析                     │  on R1-Qwen3-8B                    │  CALM + gate evaluation
├─ 撰写 Section 5.3 初稿           ├─ 运行 native thinking              ├─ 卡2-3: rollout 实验
├─ 整合 divergence position         │  CALM + gate evaluation            │  (5 methods × 200 questions)
│  结果                            └─ 分析 divergence position
└─ 更新 Contributions                 结果，决定是否做 warm-up
```

Day 2 产出：
- Native thinking 完整协议结果
- Segment-level exit 结果
- Rollout 实验进行中/完成
- 所有新实验的结果表格

### Day 3：结果整合 + 主文定稿

```
Agent A (写作):                    Agent B (补充):
├─ 整合所有实验结果到主文           ├─ warm-up 策略实验（如需要）
├─ 完成 Section 5.3                ├─ rollout 结果分析
├─ 扩展 Section 4.4                └─ 生成图表
├─ 更新 Contributions
└─ 可选: Figure 8 前移
```

### Day 4-5：整合 + polish + 提交

```
├─ 全文 claim-evidence alignment
├─ 确认新数据的 caption 和引用完整
├─ NeurIPS 格式检查
├─ 页数确认（新增 ~1 页内容）
└─ 最终通读
```

---

## 12. GPU 资源使用估算（v4 更新）

| 实验 | GPU 卡数 | 时长 | 运行天 |
|---|---|---|---|
| Native thinking forward pass | 1 × 5090 | 4–6h | Day 1 |
| Native thinking CALM + gate eval | 1 × 5090 | 2–3h | Day 2 |
| Rollout（5 methods × 200q） | 2 × 5090 | 8–12h | Day 1–2 |
| Segment-level exit | 0（后处理） | ~3h | Day 1–2 |
| Divergence position analysis | 0（后处理） | ~1h | Day 1 |
| Warm-up 策略（可选） | 1 × 5090 | 2–4h | Day 3 |
| **总计** | **峰值 3 卡** | **~20–30 GPU·h** | **Day 1–3** |

**比 v3 减少约 40% GPU 需求**（不需要 nested split 重跑和 QwQ-32B forward pass）。

---

## 13. 主文结构建议（v4 更新）

新版论文结构已经很好。建议的增量式调整：

| 当前结构 | 建议调整 | 变化幅度 |
|---|---|---|
| 1. Introduction + Contributions | Contributions 增加第 3 条：segment-level exit | 小 |
| 2. Related Work | 不变 | 无 |
| 3. Methodology (3.1–3.3) | 不变（Q_Acc 重命名为可选） | 无/小 |
| 4. Findings (4.1–4.4) | **4.4 扩展为有表格的完整分析**（含 CALM + gate on native thinking） | 中 |
| 5. End-to-End (5.1–5.2) | **新增 5.3 Segment-Level Exit** | **中-大** |
| 6. Conclusion | 微调 | 小 |
| 7. Limitations | 更新 segment-level 方向从 "promising" 到 "validated" | 小 |

注意页数：新增约 1 页（Section 4.4 扩展 ~0.3 页 + Section 5.3 ~0.7 页）。如果超出 NeurIPS 限制，可压缩 Section 4.1–4.3 的详细数字到附录。

---

## 14. 最终投稿前 checklist（v4 更新）

### 已完成 ✅

- [x] tuned lens / gate 使用 nested split，无 test leakage（脚注 2 + Table 5 effect decomposition）
- [x] QwQ-32B V2a gate 已评估（Table 48: 61.0% Q_Acc）
- [x] Cross-architecture V2a transfer 已验证（R1-Llama, Phi-4, QwQ-32B）
- [x] Contributions 限定 scope（"on one architecture"）
- [x] 1.5B reversal 在 Section 7 / Appendix V 中有解释
- [x] Tuned lens cross-difficulty 下降在 Appendix R/T 中有说明
- [x] learned gate 被定位为 proof-of-concept
- [x] oracle 和 deployable method 被清楚区分（gap-closing progression）
- [x] TIDE concurrent work 讨论到位
- [x] Effect decomposition 和 grid sensitivity 提供方法论透明度

### 仍需完成 ⬜

- [ ] Native thinking 上的 CALM failure 和 gate recovery 已验证
- [ ] Qwen3 non-thinking / native thinking / R1-Distill-Qwen 三方表格已在主文出现，**包含 CALM 和 gate 结果**
- [ ] Segment-level exit baseline 已在主文出现
- [ ] Actual rollout 结果已报告
- [ ] Section 4.4 "distillation-specific" 已改为 "distillation-associated"
- [ ] 可选：Q_Acc 已改名为 TraceExact
- [ ] 可选：Abstract 进一步精简数字
- [ ] 可选：Figure 8 前移主文

### 格式 checklist

- [ ] NeurIPS 2026 官方 style
- [ ] 主文页数符合要求（注意 Section 4.4 扩展 + Section 5.3 新增）
- [ ] references / appendix / checklist 顺序正确
- [ ] 匿名信息全部去除
- [ ] PDF 可编译、图表清晰

---

## 15. 预估质量提升（v4 更新）

| 方案 | 预估均分 | 区间 |
|---|---|---|
| 当前论文（05-02 版） | ~5.5–6 | borderline / weak accept |
| + 全部 P0 实验（native thinking full + segment exit + rollout） | ~7–7.5 | accept |
| **+ P0 实验 + P0/P1 写作改动** | **~7.5–8** | **accept** |

**核心认识更新：05-02 版已经解决了 v3 计划中约 40% 的问题（nested split、QwQ-32B gate、cross-architecture summary、effect decomposition）。剩余质量提升主要来自三个关键实验：native thinking CALM+gate（堵住 confound）、segment-level exit（constructive insight）、rollout（deployment validation）。写作方面只需微调而非大改。**
