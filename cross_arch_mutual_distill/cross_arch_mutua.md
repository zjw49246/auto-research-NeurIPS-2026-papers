# NeurIPS 2026 投稿前论文优化方案（基于修订版 PDF 更新，v2）

**论文题目**：*Symmetric Loss, Asymmetric Outcome: A Diagnostic Study of Cross-Architecture Mutual Distillation*
**目标会议**：NeurIPS 2026
**基于版本**：`cross_arch_mutual_distill_v1_revised_20260503.pdf`（27 页，含 Appendix A–P + NeurIPS Checklist）
**更新日期**：2026-05-03
**文档目的**：基于修订版 PDF 的实际内容，梳理已完成实验全清单、模拟审稿意见、提出可执行的修改方案。已移除对旧版适用但修订版已采纳的建议，并经过二次审视修正了部分建议的优先级和可行性判断。

---

## 第一部分：当前论文内容与已完成实验全清单

### 1.1 核心问题与定位

论文研究 **CNN 与 ViT 之间的跨架构 mutual distillation**，核心发现是：

> 形式上对称的 mutual KL distillation loss，在 CNN↔ViT 共训时产生**不对称结果**：ViT 稳定获益，CNN 侧取决于架构组合、容量差距和 augmentation。

定位是 **diagnostic empirical study**，不是提出新 KD 方法。

### 1.2 叙事线（五步）

1. **指出假设问题**：DML、CSKD 等现有方法默认 mutual training 双方受益，但未在跨架构 pair 中验证。
2. **提出 2×2 factorial design**：交叉 training mode（independent vs. mutual）× augmentation（Mixup vs. no-Mixup），将 distillation effect、Mixup effect、interaction 分离。
3. **证明 asymmetry 存在且非噪声**：同构对照（CNN↔CNN, ViT↔ViT）显示对称结果（|Δ| ≤ 0.82 pp），跨架构 pair 不对称。
4. **分解为 capacity/architecture regime + augmentation confound**：三组 pair 显示 CNN 侧从负到正的 regime 变化；Mixup 会放大 CNN 侧下降。
5. **给出 mitigation**：asymmetric loss weighting（α_A ≈ 0.25）在高 gap 场景下形成 Pareto improvement。

### 1.3 已完成实验全清单（主文 + Appendix）

> **核心原则：在建议"补实验"之前，必须确认 appendix 中没有已完成的版本。**

#### 主文实验

| 实验 | 位置 | Seeds | 关键结果 |
|---|---|---|---|
| R32/DeiT-T 主结果 (CIFAR-100, NoMixup) | Table 1 | 5 | CNN −0.76±0.64, ViT +4.43±0.68 |
| Cross-dataset summary (NoMixup) | Table 2 | 3–5 | 三组 pair 的 Δ_CNN / Δ_ViT / Asym Gap |
| Three-pair CIFAR-100 summary (NoMixup) | Table 3 | 5 each | R32/DT −0.76, MobNet/DT +4.25, R18/DS +1.50 |
| 2×2 factorial CIFAR-100 (R32/DeiT-T) | Table 4 | s42† | CNN interaction −0.43, DeiT −0.01 |
| 2×2 factorial CUB-200 (R18/DeiT-S) | Table 5 | s42† | CNN interaction −2.73, DeiT +2.33 |
| Alpha=0.25 三 pair 比较 | Table 6 | 5 each | R32/DT bilateral✓, R18/DS CNN-neutral, MobNet/DT CNN-harmful |
| Temperature sweep τ∈{1,4,8} | Section 4.1 | 3 | ViT gain ≥+4.0 pp at all τ |
| Factorial interaction plot | Figure 1 | s42 | CIFAR-100 + CUB-200 可视化 |
| Same-arch controls (w/ Mixup) | Table 1 | 3 | |Δ| = 0.21±0.19 (CNN), 0.25±0.09 (ViT) |

> †主文 Tables 4–5 是 single-seed (s42)，但 3-seed matched interaction terms 已在 Appendix G 完成并被主文引用。

#### Appendix 实验（完整清单，含修订版新增内容）

| Appendix | 内容 | Seeds | 状态 |
|---|---|---|---|
| **A** | Temperature details (Table 7) | 3 | ✅ 完整 |
| **B** | CKA-Performance (Table 8 + Figure 2) | s42 | ✅ CKA-accuracy 解耦已证明 |
| **C** | Training Dynamics (Figures 3–4, Table 9: 3-pair CKA) | s42（CKA 方向 3s 验证） | ✅ ΔCKA = +0.22–0.24 across 3 pairs |
| **D** | DML 2×2 Factorial (Table 10) | **3** | ✅ Mixup −2.37 pp > temperature −0.10 pp |
| **E** | Baseline Comparison (Table 11) | 混合 | ✅ CSKD 3-seed, DML/Unidir s42 |
| **F** | MobNet Case Study (Table 12 + Figure 6) | 5 (NoMixup), 3 (Mixup) | ✅ 完整 |
| **G** | 3-Seed Factorial Interaction | **3** | ✅ CIFAR-100 全 3 pairs + CUB-200 |
| **H** | Hyperparameters (Table 13) | — | ✅ 完整 |
| **I** | Asymmetric Loss Weighting + α sweep (Tables 14–16, Figure 7) | 混合 | ✅ CIFAR-100 α=0.25 五 seed, TI-200 α=0.50 三 seed |
| **J** | Tiny-ImageNet Factorial (Table 17–18, Figure 8) | ⚠️ **s42 only** | 独立附录，但 single-seed；ViT interaction = −4.94 pp 无 error bars |
| **K** | TCKD/NCKD Decomposition | s42 (R18/DS 3s) | ✅ Negative result：分解不解释 asymmetry |
| **L** | **Mixup Mechanism: Gradient and Entropy Evidence** (Table 19) | ⚠️ **s42 only** | **修订版新增**：gradient cosine, KD output entropy, layer-wise CKA |
| **M** | Same-Arch NoMixup Controls (Table 20) | 3 | ✅ |Δ| ≤ 0.82 pp |
| **N** | R18/DeiT-S Results (Table 21, Figure 9) | **5** | ✅ 修订版升级到 5-seed，Figure 9 含 error bars |
| **O** | Dataset Details | — | ✅ 完整 |
| **P** | Optimizer Ablation (Table 22) | s42 | ✅ Unified SGD 崩 ViT, Unified AdamW 压 CNN |

#### 关键已有实验总结

论文已完成的实验覆盖度比初看主文要广得多：

- **DML factorial 有 3-seed**：Appendix D Table 10 完整证明 Mixup confound 跨方法成立
- **3-seed factorial interaction 已完成**：Appendix G 覆盖 CIFAR-100 全 3 pairs + CUB-200
- **TCKD/NCKD 分解已做**：Appendix K 覆盖全 3 pairs，结论是 negative result（已被 Discussion 引用）
- **Gradient cosine + entropy 已做**：Appendix L Table 19 有初步分析（但仅 s42, R32/DT 单 pair）
- **Capacity-gap 可视化已有**：Figure 9 已升级为 5-seed，含 error bars
- **CKA 方向性有 3-seed 验证**：Table 9 覆盖 3 pairs × 3 seeds

---

## 第二部分：Review 总结

### 2.1 总体评价

> **Borderline+ (5.5/10)**。修订版在 Discussion 深度和 Appendix 利用上有显著改善，但核心实验缺口仍在。论文定位为 diagnostic study，这本身限制了得分上限——NeurIPS reviewer 对纯诊断性工作通常要求更强的因果证据或更大的实验规模来补偿方法 novelty 的不足。

### 2.2 核心优点

1. **问题有价值**："对称 loss 不保证对称收益"是重要且容易被忽视的现象
2. **Factorial design 是方法贡献**：比普通 KD 论文的单因素 ablation 更严谨
3. **实验覆盖面广**：3 pairs × 3 datasets × temperature/baseline/CKA/α sweep
4. **Asymmetric weighting 有实践意义**：α_A ≈ 0.25 的 Pareto improvement，5-seed 验证
5. **Discussion 机制分析有深度**（修订版改善）：absorption ceiling + gradient/entropy evidence
6. **Appendix 利用充分**（修订版改善）：主文对 D/G/K/L/M/N 都有引用

### 2.3 当前仍存在的核心问题

#### 问题一：capacity sweep 因果验证缺失（最关键）

三个 pair 同时改变了 CNN family、ViT size、baseline accuracy gap、capacity ratio。MobNet（3.4:1）的 CNN gain 反超 R18/DS（2:1），说明参数比例不是唯一因素。

修订版 Discussion 的 absorption ceiling 论证（lines 221–236）是定性推理，引用的是 Table 3 的 3 个 confounded 数据点。修订版 Limitations（lines 311–318）**自己承认了这个问题**：

> "each pair simultaneously varies both the CNN and (in one case) the ViT, the cross-pair comparison cannot isolate a single causal variable."

这等于直接告诉 reviewer "我们知道但没补"——是最容易被 reject 的理由。

#### 问题二：Appendix L 机制分析 under-powered

Appendix L（Table 19）是修订版最有价值的新增分析，但仅限于 s42 + R32/DeiT-Tiny 一个 pair。Discussion 中 "entropy increases ~25%"（line 247）和 "deep-layer CKA drops ~30%"（line 247）引用的就是这个 single-seed 结果。

如果 reviewer 质疑这些数字在其他 pair / seed 上是否稳定，目前无法回答。这个问题的严重性在于：Discussion 的 mechanism story 的定量部分**全部**依赖这一个 single-seed 数据点。

#### 问题三：Tiny-ImageNet Factorial 仅 single-seed

Appendix J Table 17 + Figure 8 全部基于 s42。ViT interaction = −4.94 pp 是全文三个数据集中最大的交互效应，Figure 8 自己标注了 "Tiny-ImageNet bars reflect a single seed (s42) and lack error bars"。

但这个问题的**实际影响需要客观评估**：论文的主结论建立在 CIFAR-100（3-seed factorial, Appendix G）和 CUB-200（3-seed factorial, Appendix G）之上。Tiny-ImageNet 是第三个数据集，起确认性作用。即使 TI-200 的 interaction 最终不显著，也不会动摇主结论。因此这个问题虽然存在，但不是 reject 的主要理由。

#### 问题四：格式仍然偏长

正文 Section 6（Conclusion）包含 Optimizer confound 段落（lines 298–304）和 Limitations（lines 305–319），合计约 0.75 页。NeurIPS 限制 9 页（含图表，不含 references/appendix/checklist）。Related Work 也偏长（约 1.2 页），对于 diagnostic study 而言超出常规比例。

#### 问题五：Abstract 和 teaching tax 表述

Abstract 第 3–4 行以 "the CNN pays a teaching tax of −0.76 pp" 为核心卖点。但论文的实际贡献是 regime-dependence（三个 pair 从 −0.76 到 +4.25）。以最极端的单一数据点为 headline 会误导读者认为 CNN 总是受害。

#### 问题六：缺少实践指导（"so what" 问题）

论文定位为 diagnostic study，但 reviewer 会问"诊断完了然后呢"。目前唯一的 actionable output 是 α_A≈0.25，但这被证明是 pair-specific 的（Table 6：R32/DT 有效，R18/DS 中性，MobNet 有害）。论文没有给出通用的实践决策流程，导致实用价值被削弱。这对于 NeurIPS 这种看重 impact 的会议尤其不利。

#### 问题七：Divergence family 对照不足

当前只有 reverse KL τ=4（主方法）和 forward KL τ=1（DML）。温度不同，无法公平比较。不过论文已有三种不同方法（reverse KL、forward KL、CSKD class-wise soft target）都显示 asymmetry，跨方法一致性已经较强。这个问题可能被个别 reviewer 提出，但不太可能是 reject 的核心原因。

#### 问题八：Unidirectional ablation 逻辑

Table 11（Appendix E）只有 CNN→ViT (α_A=0) 方向的 single-seed (s42)，CNN Δ = −0.54 pp。这个值落在 same-architecture training variance 范围内（|Δ| ≤ 0.82 pp, Table 20），不能作为 "teaching-role effect" 的证据。缺少反方向 ViT→CNN (α_B=0) 的对照。

---

## 第三部分：优化方案

### 设计原则

1. **充分利用 appendix 已有结果**，不建议重复已完成的实验
2. **优先级按"对中稿率的实际影响"排序**，而非理论上的完备性
3. **Claim 强度匹配证据**：不做没有多 seed 支撑的强因果表述
4. **诚实评估每条建议的预期收益**，区分"必须做"和"锦上添花"

---

### 3.1 第一层：写法修改（不需要新实验）

#### 修改 1：格式压回 9 页（必须做）

**对中稿率的影响**：高。NeurIPS 对超页格式要求严格。超过 9 页不一定 desk reject，但会给 reviewer 负面第一印象，且提供了一个低成本的 reject 理由。

**操作**：
- Optimizer confound 段落（Section 6, lines 298–304）→ 移至 Appendix P（Table 22 已在那里），主文仅保留一句："A unified-optimizer ablation (Appendix P) confirms the asymmetry tracks architecture, not optimizer identity."
- Related Work 从 ~1.2 页压缩到 ~0.8 页。当前 Related Work 有五个小节（Knowledge distillation / Mutual and online / Cross-architecture / Augmentation–distillation / Logit-level），其中 "Logit-level KD advances" 小节（lines 82–87）仅 6 行且与论文核心问题关系最弱，可合并入第一段的综述句中。"Cross-architecture distillation" 小节（lines 64–72）中列举了 8 篇 concurrent work，可压缩到 4–5 篇最相关的。
- Limitations 段落从 ~15 行压缩到 ~10 行，删除与 Discussion 正文重复的表述（如 lines 311–314 关于 cross-pair comparison 的讨论与 Section 5 lines 261–265 的 cross-pair variation 段落重叠）
- 预计总共回收 **0.5–0.75 页**，足够压进 9 页

#### 修改 2：调整 Abstract 的核心卖点（必须做）

**对中稿率的影响**：中高。Abstract 是 reviewer 读的第一段文字，决定了他们对论文的 framing 期待。如果 Abstract 强调 teaching tax，reviewer 会期待论文围绕 "CNN 受损" 展开；读到 Section 4.1 后发现三个 pair 中只有一个 CNN 受损，会感觉被 over-sell。

将 Abstract 第 3–4 行：

> "the CNN pays a teaching tax of −0.76 pp under no-Mixup conditions (n=5 seeds, all negative)"

改为：

> "Under the widest capacity gap (≈11:1), the CNN shows a sign-consistent decrease (−0.76 pp, 5/5 seeds negative); narrowing the gap flips the CNN-side effect to bilateral gains in two of three pairs (15/15 seeds directionally consistent; Fisher's combined p=0.011)."

保留 "teaching tax" 作为 Section 4.1 中的操作性定义，但 Abstract 首句卖点改为 regime-dependence。

**注意**：不应完全去掉 teaching tax 概念——它是论文最有记忆点的术语，reader-friendly。调整的目标是把它从"核心发现"降为"一个 regime 下的特定表现"。

#### 修改 3：修正 Unidirectional ablation 解释（建议做）

**对中稿率的影响**：中。不改的话挑剔的 reviewer 会在 Section 3.4 或 Appendix E 处抓住逻辑问题写一个 weakness，但不太可能因此 reject。改了的话堵住一个小洞，且展示作者对实验设计的 self-awareness。

在 Table 11 引用处添加：

> "Under α_A=0 with stop-gradient, the CNN receives no distillation gradient and should train identically to the independent baseline. The observed CNN Δ of −0.54 pp (single seed) falls within the range of same-architecture training variance (|Δ| ≤ 0.82 pp, Table 20) and likely reflects stochastic variation. The ablation's primary value is confirming that ViT gains (+4.31 pp) do not require mutual feedback."

#### 修改 4：标注 Appendix L 的 single-seed 局限性（必须做）

**对中稿率的影响**：中高。如果不标注，reviewer 可能把 Discussion 中引用的 "+25% entropy" 和 "+185% gradient perturbation" 当作 robust finding 来质疑（"你在多少 seeds 上验证了这个？"），然后发现答案是 1 个 seed 1 个 pair。主动标注 limitation 比被动被发现好得多。

Discussion 中引用 Appendix L 结果时（line 247），修改为：

> "Entropy increases by ~25% while deep-layer CKA drops ~30% (Appendix L, Table 19; single seed, R32/DeiT-Tiny). These measurements provide suggestive mechanistic evidence; multi-pair and multi-seed validation is needed to confirm generality."

将 "confirms" 类措辞改为 "suggests" 或 "is consistent with"。

#### 修改 5：添加实践决策指导（强烈建议）

**对中稿率的影响**：中高。Diagnostic study 的最大弱点是 "so what"——reviewer 读完会问"我知道了 asymmetry 存在，那我该怎么办？"。当前论文的唯一 actionable output（α_A≈0.25）被证明是 pair-specific 的，实用价值有限。

在 Discussion 末尾（Practical implications 段落，lines 270–275）或 Conclusion 中，添加一个简明的**实践决策流程**：

> **Practitioner checklist for CNN↔ViT co-training:**
>
> 1. Always include a **no-augmentation control arm** alongside the default Mixup/CutMix pipeline (Section 4.2). Mixed-sample augmentation interacts with distillation in architecture-specific ways that can flip the sign of CNN-side effects.
>
> 2. Compare the CNN's mutual-training accuracy against its independent baseline at the **same seed**. If the CNN shows consistent degradation across seeds, the capacity gap may be too large for symmetric distillation.
>
> 3. In high-gap regimes (CNN baseline substantially exceeds ViT baseline), try **reducing the CNN-side distillation weight** (α_A ∈ [0.25, 0.5]) while keeping the ViT weight at 1.0 (Section 4.3).
>
> 4. If both architectures already show bilateral benefit under symmetric weighting, **retain α_A = 1.0** — asymmetric weighting can harm pairs that are already in the beneficial regime (Table 6, MobNet/DeiT-Tiny).

**为什么这能提升中稿率**：NeurIPS 越来越看重 practical impact。一个 3 行的 checklist 不需要新实验，但把论文从"我们发现了一个现象"变为"我们提供了可执行的诊断方案"。对于 diagnostic study 定位来说，这是最直接的 impact 证明。

---

### 3.2 第二层：低成本分析扩展（利用已有 checkpoints，~3 GPU-hours）

#### 分析 1：Appendix L 扩展到 3 pairs × 3 seeds（强烈建议）

**对中稿率的影响**：高。这是**投入产出比最高的单项改进**。原因：

1. Appendix L (Table 19) 是修订版最重要的新增内容——它是 Discussion 中 mechanism story 的定量基础
2. 但它目前只有 s42 + R32/DeiT-Tiny 一个 pair，是全文最明显的 under-powered 分析
3. 修复它只需要在已有 checkpoints 上做 forward pass，不需要重新训练
4. 扩展后的 3×3 矩阵可以直接验证（或否证）Discussion 的核心 claim

**设计**：

覆盖 3 pairs (R32/DT, MobNet/DT, R18/DS) × 3 seeds (42, 123, 456)，计算：

| 指标 | 当前状态 | 扩展后 |
|---|---|---|
| Gradient cosine (mutual vs. indep, CNN) | s42, R32/DT only: 0.127 → 0.363 | 3 pairs × 3 seeds |
| Gradient cosine (mutual vs. indep, ViT) | s42, R32/DT only: 0.478 → 0.488 | 3 pairs × 3 seeds |
| KD output entropy (Clean vs. Mixup) | s42, R32/DT only: +25% | 3 pairs × 3 seeds |
| Layer-wise CKA | s42, R32/DT only: stem 0.436→0.304 | 至少 2 pairs |

**工作量**：纯 inference，约 **2–3 GPU-hours**。

**关键预期与风险**：

如果跨 pair 结果一致（gradient perturbation 在 CNN 侧远大于 ViT 侧，entropy 增幅稳定）：
- Discussion 的 mechanism story 从 "suggestive" 升级为 "consistent across pairs"
- Table 19 成为论文的 mechanistic centerpiece

如果跨 pair 结果不一致（例如 MobNet pair 的 gradient perturbation 反而很小）：
- 需要修正 Discussion 的叙事，但这是一个有价值的 nuance——说明机制也是 regime-dependent
- **提前发现不一致比被 reviewer 在 rebuttal 期间要求补做好得多**

**产出**：Table 19 扩展为一个 3 pairs × 2 metrics 的矩阵表（含 mean±std across seeds）。如果空间允许，在 Appendix L 中增加一张 gradient cosine vs CNN Δ 的散点图——这自然地回答了"什么决定 CNN-side outcome"的问题。

---

### 3.3 第三层：核心补实验

#### 实验 1：Clean Capacity Sweep（最高优先级，~18 GPU-hours）

**对中稿率的影响**：最高。这是论文 **被 reject 概率最高的单一原因**的直接修复。

**具体论证**：

修订版 Discussion 花了一整段（lines 221–236）论证 absorption ceiling 机制。这个论证的全部定量支撑来自 Table 3 的三个数据点：

```
R32 (72.9%) → CNN Δ = −0.76    "near ceiling, can't absorb"
R18 (79.7%) → CNN Δ = +1.50    "moderate headroom"
MobNet (58.4%) → CNN Δ = +4.25  "most headroom, gains most"
```

问题在于：这三个数据点**同时改变了 CNN family**（ResNet-32 vs ResNet-18 vs MobileNetV3-Small）。MobNet 使用 depthwise-separable convolution，其 inductive bias 与标准 ResNet 完全不同。MobNet 的 CNN Δ = +4.25 可能来自：
- (a) absorption ceiling 更低（论文的 claim）
- (b) depthwise-separable 的 inductive bias 恰好与 ViT 互补（alternative explanation）
- (c) MobNet 的 independent baseline 更弱，从任何 teacher 都能学到更多（confound）

三者无法区分。论文 Limitations 自己承认了 (a) vs (b) 的混淆（line 311–318），但没有解决。

**设计**：

```
固定 ViT：DeiT-Tiny (5.4M)
固定 dataset：CIFAR-100, no-Mixup
变化 CNN（同 ResNet family，仅改深度）：

ResNet-20  (~0.27M, ratio ~20:1, baseline ~68–69%)
ResNet-32  (~0.47M, ratio ~11:1, baseline ~72.87%)  ← 已有
ResNet-56  (~0.86M, ratio ~6:1,  baseline ~74–75%)
ResNet-110 (~1.73M, ratio ~3:1,  baseline ~76–77%)
```

每组 3 seeds, independent + mutual = 6 runs/ratio。新增 3 ratio × 6 runs = **18 runs ≈ 18 GPU-hours**。

**为什么这个实验对中稿率影响最大**：

1. 它直接回应论文自己承认的最大 limitation——这比 reviewer 主动发现问题要好
2. 产出是一张 5–6 个同 family 点的 capacity-gap 曲线，一张图就能回答 "is it capacity or architecture family?"
3. 无论结果是单调还是非单调，都有合理的叙事：
   - 单调：强化 absorption ceiling claim
   - 非单调但与 Table 3 趋势一致：说明 capacity 是 dominant factor，family 是 modulator
   - 与 Table 3 趋势不一致：需要修正叙事，但发现这一点本身就是诊断价值
4. MobNet 的对照点天然存在（已有数据），可以在同一图中标注为 "different CNN family" 来展示 family effect 的大小

**产出**：将 Figure 9 从 3 个 confounded 点扩展为：
- 4 个同 family 点（ResNet-20/32/56/110 + DeiT-Tiny），控制了 architecture family
- 1 个 cross-family 对照点（MobNet/DeiT-Tiny），展示 family effect
- 含 3-seed error bars
- 可选：第二个 x 轴为 baseline gap（仅在有足够数据点时才画，避免 3 点过拟合问题——见下文修改 4 的注释）

#### 关于原方案"修改 4：Baseline accuracy gap 散点图"的修正

原方案建议用 Table 3 的 3 个已有数据点画 baseline gap vs CNN Δ 散点图，称之为"零成本高价值"。

**经重新分析，这个建议的价值被高估了**：

3 个数据点无论用 parameter ratio 还是 baseline gap 做 x 轴，都无法区分谁的解释力更强——任何 3 个点都可以被两个变量各自 fit 出看似合理的趋势。具体来看：

| Pair | Param Ratio | Baseline Gap | CNN Δ |
|---|---|---|---|
| R32/DeiT-T | 11:1 | +8.25 | −0.76 |
| MobNet/DeiT-T | 3.4:1 | −6.09 | +4.25 |
| R18/DeiT-S | 2:1 | +10.29 | +1.50 |

- Parameter ratio 视角：趋势非单调（MobNet 3.4:1 > R18 2:1），看似 ratio 不是好的 predictor
- Baseline gap 视角：**也是非单调**（R18 gap=+10.29 但 Δ=+1.50，R32 gap=+8.25 但 Δ=−0.76）
- 绝对 baseline 视角：MobNet 58.4% → Δ 最大，R32 72.9% → Δ 最小，但 R18 79.7% → Δ=+1.50 不在趋势上

单独画这张图，反而会暴露 3 个数据点无法支撑任何单变量解释的问题，给 reviewer 提供攻击点。

**修正后的建议**：**不单独画 baseline gap 散点图**。而是将 baseline gap 分析并入 capacity sweep 的产出——当数据点从 3 个增加到 5–6 个（且控制了 architecture family）时，baseline gap vs CNN Δ 的趋势才有统计意义。在 capacity sweep 完成后，在 Figure 9 中加一个 secondary x-axis 或并列子图来展示 baseline gap 的解释力。

#### 实验 2：Divergence Family Ablation（建议做，精简版，~3 GPU-hours）

**对中稿率的影响**：中。论文已有三种不同方法（reverse KL τ=4、forward KL τ=1 via DML、CSKD class-wise soft target）都显示 asymmetry，跨方法一致性已经较强。但温度不同（τ=4 vs τ=1）是一个可被挑战的混淆因素。

**原方案建议 forward KL + JSD 各 3 seeds = 6 runs。经重新评估，精简为**：

只做 **Forward KL at τ=4**，3 seeds = **3 runs ≈ 3 GPU-hours**。原因：

1. 核心问题是 "reverse vs forward KL 在同温度下是否不同"，JSD 是额外的
2. 如果 forward KL τ=4 也显示 asymmetry → 与 DML (forward KL τ=1) 和 CSKD 一起，三种方法全部确认
3. 如果 forward KL τ=4 不显示 asymmetry → 这是一个有价值的发现（说明 divergence direction matters），但仍不太可能因此 reject（论文已有两种方法确认 asymmetry）
4. 节省的 3 GPU-hours 可以用于 Appendix L 扩展

**设计**：CIFAR-100 R32/DeiT-T, no-Mixup, τ=4, 3 seeds。Forward KL: KL(p_student || p_teacher)。

#### 实验 3：Tiny-ImageNet Factorial 补 seed（可选，优先级降低，~12 GPU-hours）

**对中稿率的影响**：中低。经重新评估，降低优先级。原因：

1. **主结论不依赖 TI-200**：CIFAR-100 和 CUB-200 的 factorial 都有 3-seed 验证（Appendix G），主结论已经有足够支撑。TI-200 是第三个确认性数据集。
2. **Figure 8 的问题可以用写法解决**：在 Figure 8 caption 中明确标注 TI-200 为 "preliminary, single-seed"，并在 Appendix J 文本中加一句 "Multi-seed validation of the Tiny-ImageNet factorial remains future work"。这比补 12 GPU-hours 的实验成本低得多。
3. **12 GPU-hours 的机会成本**：同样的算力可以做 capacity sweep 的额外 seed 或 Appendix L 扩展。

**如果做**：补 s123, s456 的 4 个 factorial cells，8 runs ≈ 12 GPU-hours。
**如果不做**：在 Figure 8 和 Appendix J 中明确标注 single-seed limitation。

#### 实验 4：Unidirectional 双向 + Multi-seed（可选，~5 GPU-hours）

**对中稿率的影响**：中低。Unidirectional ablation 在论文中的作用是 supporting evidence，不是核心 claim。修改 3（修正解释措辞）已经足以堵住主要逻辑漏洞。

**如果做**：
- CNN→ViT (α_A=0): 补 2 seeds
- ViT→CNN (α_B=0): 3 seeds（新方向）
- 5 runs ≈ 5 GPU-hours

**如果不做**：按修改 3 的建议修正措辞即可。

#### 工作量汇总（按优先级排列）

| 优先级 | 实验 | 新增 runs | GPU-hours | 对中稿率影响 |
|---|---|---|---|---|
| **最高** | Capacity sweep (R20/R56/R110) | ~18 | ~18h | 最高：堵住最大因果漏洞 |
| **高** | Appendix L 扩展 (forward pass only) | 0 | ~3h | 高：机制证据从 1-seed 升级为 9-point 矩阵 |
| **中** | Forward KL τ=4 (精简版 divergence ablation) | 3 | ~3h | 中：排除 divergence artifact |
| **中低** | TI-200 factorial 补 seed | ~8 | ~12h | 中低：确认性，可用写法替代 |
| **中低** | Unidirectional 双向 + 补 seed | ~5 | ~5h | 中低：supporting evidence，可用措辞替代 |

**推荐的最小实验集**（最大化中稿率 / GPU-hour）：

前三项（capacity sweep + Appendix L 扩展 + forward KL）= **~24 GPU-hours ≈ 1 天**。

后两项如果算力充足可做，但不做的话可以通过写法修改弥补。

---

### 3.4 第四层：扩展验证（如果有更多算力）

| 优先级 | 实验 | 作用 | 最低配置 | 对中稿率影响 |
|---|---|---|---|---|
| 高 | ImageNet-100 验证 | 回应 scale concern | R50/DeiT-S, indep+mutual, no-Mixup, 2 seeds | 高：NeurIPS reviewer 对小数据集最常见的质疑 |
| 中高 | 现代架构 pair | 排除 ResNet-DeiT 特例 | ConvNeXt-T/DeiT-S, CIFAR-100, 3 seeds | 中高：排除 "only works on old architectures" |
| 中 | Pretrained fine-tuning | 排除 from-scratch ViT 弱 baseline 解释 | CUB-200 w/ ImageNet-pretrained, 3 seeds | 中 |
| 中 | Asymmetric weighting 预测性验证 | 从 diagnostic 升级到 prescriptive | 用 capacity sweep 规律在新 pair 上测试 | 中 |

**关于 scale concern 的诚实评估**：论文的**根本性天花板**是实验尺度。CIFAR-100/CUB-200/TI-200 + 0.47M–22M 参数的模型，在 2026 年的 NeurIPS 标准下偏小。即使做完所有第三层实验，如果不做 ImageNet-scale 验证，部分 reviewer 仍会认为这是 workshop/journal 水平而非 main track。但 ImageNet-100 验证的 GPU-hour 成本较高（R50/DeiT-S 训练时间远超 CIFAR-100），需要根据实际算力预算决定。

---

### 3.5 叙事升级路线

#### 当前修订版叙事

> "我们观察到 CNN-ViT mutual KD 不对称，并给出了 absorption ceiling 的定性解释和初步 gradient/entropy 证据。"

**卡在什么地方**：mechanism story 的定量基础是 single-seed，capacity 因果是 confounded。

#### 完成第一层 + 第二层后（写法 + Appendix L 扩展）

> "Cross-architecture mutual distillation produces regime-dependent asymmetric outcomes. The ViT reliably gains; the CNN-side effect depends on directional performance imbalance and augmentation pipeline. Gradient and entropy analysis across three pairs shows that Mixup selectively degrades CNN soft-target quality, providing mechanistic evidence consistent with the observed architecture-dependent confound."

**提升在哪里**：mechanism story 有跨 pair 验证（不再是 single-seed anecdote），claim 措辞准确匹配证据强度。

#### 完成第三层后（capacity sweep + divergence ablation）

> "A controlled capacity sweep within the ResNet family shows that CNN-side outcomes track capacity mismatch within an architecture family, while a cross-family comparison reveals additional modulation from CNN inductive biases. The asymmetry persists across reverse KL, forward KL, and class-wise soft targets, confirming it is architecture-driven rather than divergence-specific. A factorial protocol reveals that Mixup compounds the effect through CNN-selective gradient perturbation (+185% gradient cosine shift, confirmed across three pairs). Asymmetric weighting (α_A ≈ 0.25) provides a simple mitigation under the high-gap regime."

**提升在哪里**：capacity 因果有 controlled evidence，divergence artifact 被排除，mechanism story 有 multi-pair 定量支撑。

---

### 3.6 建议的贡献表述

#### C1：Regime-dependent asymmetric outcome

> We show that formally symmetric mutual distillation yields strongly asymmetric outcomes in CNN↔ViT co-training: ViTs consistently gain (+3.3–4.4 pp across three pairs, each n=5), whereas CNN-side effects are regime-dependent, ranging from a sign-consistent teaching tax (−0.76±0.64 pp at ≈11:1 capacity ratio) to bilateral benefit (+4.25±0.80 pp at ≈3.4:1). The direction is governed by baseline performance gap and architecture family, not parameter ratio alone.

#### C2：Factorial protocol reveals augmentation confounds

> A 2×2 factorial evaluation separates distillation effects from Mixup effects, revealing that mixed-sample augmentation selectively harms CNN-side distillation outcomes. This confound is confirmed across both CSKD and DML losses (Appendix D, 3 seeds) and is supported by gradient-alignment and entropy evidence across three architecture pairs (Appendix L).

#### C3：Asymmetric weighting as regime-specific mitigation + practitioner guidance

> Reducing the CNN-side distillation weight to α_A ≈ 0.25 converts the teaching tax into a Pareto improvement (+0.97±0.60 pp CNN, +1.92±0.65 pp ViT, 5 seeds, p=0.022/0.003) under the high-gap regime. The remedy is pair-specific (Table 6), motivating a diagnostic checklist for practitioners selecting co-training configurations.

---

### 3.7 学术规范注意事项

1. **所有新增实验报告完整 seed 结果**，不选择性报告
2. **Single-seed 结果明确标注**（当前 Appendix J/L/P 均为 s42，需在文中标注）
3. **Claim 强度匹配证据**：Appendix L 的 entropy/gradient 数据在扩展到多 pair 前应使用 "suggestive evidence" 而非 "confirms"
4. **保留 negative results**：TCKD/NCKD 不解释 asymmetry（Appendix K）、CKA 不预测 accuracy（Appendix B）都是有价值的 negative result
5. **Paired-seed design**：所有 mutual run 与相同 seed 的 independent baseline 对比
6. **Double-blind 合规**：清理 PDF metadata、文件名、代码链接中的身份信息
7. **提交匿名代码**：至少提供 mutual KL loss、factorial protocol、α sweep 的伪代码

---

## 最终执行清单

### 第一优先级：不需要新实验的修改（1–2 天写作）

- [ ] 格式压回 9 页（移 Optimizer confound 至 appendix，压缩 Related Work 和 Limitations）
- [ ] 调整 Abstract 卖点：regime-dependence 优先于 teaching tax 单一数字
- [ ] 添加实践决策指导 checklist（Discussion 或 Conclusion 末尾）
- [ ] 标注 Appendix L (Table 19) 的 single-seed 局限性（措辞从 "confirms" 改为 "suggests"）
- [ ] 修正 Unidirectional ablation 解释（标注 single-seed + training variance 范围）

### 第二优先级：低成本分析扩展（~3 GPU-hours）

- [ ] Appendix L 的 gradient cosine + entropy 扩展到 3 pairs × 3 seeds（纯 forward pass，投入产出比最高）

### 第三优先级：核心补实验——推荐最小集（~24 GPU-hours，约 1 天）

- [ ] **Capacity sweep**: R20/R56/R110 + DeiT-T, CIFAR-100, no-Mixup, 3 seeds（18h）→ 堵住最大因果漏洞
- [ ] **Forward KL τ=4**: R32/DeiT-T, CIFAR-100, no-Mixup, 3 seeds（3h）→ 排除 divergence artifact
- [ ] （可选）Forward KL τ=4 加做 JSD（额外 3h）→ 锦上添花

### 第四优先级：确认性补充（~17 GPU-hours，如算力充足）

- [ ] TI-200 factorial 补 2 seeds（12h）→ 或通过写法标注 single-seed limitation 替代
- [ ] Unidirectional 双向 + 补 seed（5h）→ 或通过修正措辞替代

### 第五优先级：扩展验证（如有更多算力，最能提升到 7+ 分）

- [ ] **ImageNet-100 验证**（最推荐，直接回应 scale concern）
- [ ] 现代架构 pair（ConvNeXt-T/DeiT-S）
- [ ] Asymmetric weighting 预测性验证

### 预估评分提升

| 阶段 | 预估评分 | 关键变化 |
|---|---|---|
| 修订版当前 | 5.5/10 | Discussion 改善，但核心实验缺口仍在 |
| 完成第一优先级（写法） | ~6/10 | Claim 与证据匹配，格式合规，有 practitioner guidance |
| 完成第二优先级（Appendix L 扩展） | ~6.5/10 | 机制证据从 single-seed 升级为 multi-pair |
| 完成第三优先级最小集（capacity sweep + forward KL） | ~6.5–7/10 | Capacity sweep 堵住最大因果漏洞 |
| 完成第五优先级（ImageNet-100） | ~7–7.5/10 | Scale concern 消除 |

**诚实说明**：论文定位为 diagnostic study，在 NeurIPS 的得分天花板约 7–7.5 分（除非发现了能改变整个 KD 领域实践的 insight）。要冲击 8+ 分，需要从 diagnostic 转型为 prescriptive——提出基于诊断发现的新方法（如 adaptive α weighting across training），这超出了当前论文的 scope。

---

## 参考依据

- NeurIPS 2026 Main Track Handbook
- NeurIPS Paper Checklist Guidelines
- 论文 PDF：`cross_arch_mutual_distill_v1_revised_20260503.pdf`（27 页，Appendix A–P）
