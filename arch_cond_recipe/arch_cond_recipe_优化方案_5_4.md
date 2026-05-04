# Normalization Type Predicts Regularization Response —— 论文优化方案

> **投稿目标**：NeurIPS 2026（截稿 2026-05-06）  
> **当前版本**：arch_cond_recipe_latest.pdf  
> **硬件资源**：2–4× RTX 5090 (32 GB) + RTX PRO 6000 (96 GB)，可并行  
> **执行模式**：Claude Code Agent Auto Research，7×24 不间断

---

# 第一部分：论文现状

## 1.1 核心叙事与 Claim

**标题**：*Normalization Type Predicts Regularization Response: An Optimizer-Conditional Three-Way Interaction in Vision Architectures*

**核心 Claim（原文加粗，第 37–38 行）**：

> **normalization type is the architectural property that determines regularization response, and this effect is conditioned on the optimizer.**

**叙事逻辑链**：
1. 不同 vision architecture 常绑定不同 training recipe（ResNet → SGD, ConvNeXt/ViT → AdamW）
2. 直接跨架构迁移 recipe 会导致不可预测的结果（ResNet-18 +7.63 pp vs. ConvNeXt-Tiny −11.75 pp）
3. 通过 10 architectures × 8 recipes × 3 seeds 的 factorial study 发现：regularization 本身无主效应（p=0.282），但 architecture × optimizer × regularization 三方交互效应极大（η²_p=0.937）
4. 在 SGD 条件下，BN 架构全部从 regularization 中获益，non-BN 架构全部受损（Cliff's δ=1.0, p=0.0095）
5. AdamW 消除了这种差异：9/10 架构的 Δ_reg 在 ±3 pp 以内
6. Norm-swap 因果实验证实 normalization 是因果参与者
7. 实用结论：检查 normalization type 即可零成本砍掉一半 recipe 搜索空间

**论文自述贡献（4 条）**：
1. 系统性 benchmark（10 architectures × 8 recipes × 3 seeds）
2. 发现三方交互（norm type × optimizer × regularization）
3. Norm-swap 因果验证
4. 零成本 recipe selection heuristic

---

## 1.2 已完成实验清单

### 1.2.1 主实验：Factorial Study（主文 Table 2, Section 3.1）

| 维度 | 配置 |
|------|------|
| 架构 | 10 个：CaiT-XXS-24, DeiT-Tiny, Swin-Tiny, EfficientNet-B0, RegNetY-400MF, MLP-Mixer-S/16, PoolFormer-S12, ConvNeXt-Tiny, ResNet-18, WRN-28-10 |
| Normalization | 4 BN + 5 LN + 1 GN |
| Optimizer | SGD (lr=0.1, mom=0.9) / AdamW (lr=1e-3) |
| Augmentation | Light / Strong（+RandAugment, Mixup, CutMix）|
| Regularization | None / Standard（WD + label smoothing + dropout）|
| Seeds | 3（42, 123, 456）|
| 数据集 | CIFAR-100（50k train / 10k test，resize 至 224×224）|
| 训练 | 100 epochs, batch 128, cosine annealing + 5-epoch warmup, FP16, gradient clipping (max_norm=5.0) |
| 实际运行 | 224 / 240 runs（排除项见下）|

**排除/调整项**：
- PoolFormer-S12：SGD 在 lr=0.1 diverge，改用 lr=0.01（12 runs, 3 seeds）
- Swin-Tiny：AdamW 在 lr=1e-3 collapse，改用 lr=5×10⁻⁴
- RegNetY-400MF：受限于 1 seed（8 runs），标记为 preliminary

### 1.2.2 Norm-Swap 因果实验（主文 Table 3, Section 3.3）

| 架构 | Swap 方向 | 训练配置 | Runs |
|------|-----------|----------|------|
| ConvNeXt-Tiny | LN → BN（BatchNorm2d）| 8 recipes × 3 seeds | 24 |
| DeiT-Tiny | LN → BN（BatchNorm2d）| 8 recipes × 3 seeds | 24 |
| ResNet-18 | BN → LN（GroupNorm(1,C)）| 8 recipes × 3 seeds | 24 |
| **合计** | | | **72 runs** |

**关键结果**：
- LN→BN 未能翻转方向：ConvNeXt SGD Δ_reg 从 −11.75 恶化到 −30.64；DeiT 从 −6.67 恶化到 −12.84
- BN→LN 仅衰减不翻转：ResNet SGD Δ_reg 从 +7.63 降到 +2.09，仍为正
- AdamW 在所有 swap 中保持稳定（变化 ≤1.57 pp）

### 1.2.3 Spectral Feature Profiling（主文 Section 2.2, Section 4.2）

| 特征 | 描述 | 结论 |
|------|------|------|
| Power-law exponent β | Hessian 频谱幂律指数 | ❌ 不具区分度（MH1 falsified）|
| Spectral concentration λ_c | 最大特征值 / 均值 | ❌ 弱且被 norm type confound（MH2 scope boundary）|
| Hessian trace tr(H) | 平均曲率 | 用于描述但未作为 predictor |
| Maximum eigenvalue λ_max | 最大 Hessian 特征值 | 用于描述但高 CV |
| Gradient norm ‖∇L‖₂ | 初始梯度大小 | 用于描述 |
| Effective rank (erank) | 激活矩阵有效维度 | ⚠️ 方向性趋势（ρ=−0.527, p=0.117），不显著（MH3 not supported）|

- 10 architectures × 3 seeds = 30 profiling runs
- 每次 ~1–2 GPU-min
- 总成本 ~10 GPU-hours

### 1.2.4 Appendix 完整分析清单（重要：已做的分析不需重复）

| Appendix | 内容 | 状态 |
|----------|------|------|
| **A** | 完整 4-way ANOVA（Table 4），8 架构平衡设计，N=192 | ✅ 已完成 |
| **B** | 阈值敏感性分析（Table 5），BN/non-BN 分离在不同 Δ_reg cutoff 下的鲁棒性 | ✅ 已完成 |
| **C** | 按 optimizer 分层的 regularization 分析（SGD/AdamW 各 n=10） | ✅ 已完成 |
| **D** | Leave-One-Out 稳定性分析（Table 6），10 个 LOO 子集全部保持方向一致 | ✅ 已完成 |
| **E** | Cliff's δ 效应量计算，完整推导和数值 | ✅ 已完成 |
| **F** | Norm-swap 逐配置精细结果（Table 7），8 配置 × 3 swap × 均值 | ✅ 已完成 |
| **G** | 计算资源明细（Table 8），10 架构逐项 GPU 时间 | ✅ 已完成 |

**Appendix 评估**：统计分析层面已非常完备。LOO、阈值敏感性、分层分析等在小样本下能做的 robustness check 基本覆盖。论文的短板不在统计严谨性，而在 **跨数据集验证** 和 **GN 覆盖不足**。

### 1.2.5 计算资源已使用

- 总计 ~271 RTX 5090 GPU-hours（~68 wall-clock hours on 4 GPUs）
- Profiling: ~10 GPU-hours
- Training: ~241 GPU-hours（224 factorial runs + 72 norm-swap runs）

---

## 1.3 当前主要问题诊断

按审稿风险从高到低排列：

### 🔴 致命级（足以导致 reject）

**P1. 核心 claim overclaim："determines" vs. "predicts"**

论文标题用 "Predicts"（可辩护），但正文多处滑向 "determines" 语气。最突出的是第 37–38 行的加粗声明。然而 norm-swap 实验的三个 swap 全部未能实现方向翻转——这直接反驳 "determines"。论文在第 271–279 行承认 "normalization is not the sole factor"，但这个 nuance 未传导到 abstract/introduction/conclusion。

审稿人一句 "Your own norm-swap results contradict your main claim" 即可判 reject。

**P2. 单一数据集（CIFAR-100）**

所有实验在一个数据集上完成。CIFAR-100 是 32×32 低分辨率、50k 小数据集，model capacity-to-data ratio 极高。Regularization 敏感度在小数据集上天然被放大。审稿人会质疑：这些 pattern 在 Tiny-ImageNet / ImageNet-100 上是否复现？

这是 NeurIPS 审稿中最常见的 weakness 之一。

### 🟡 严重级（会被扣分但不一定致命）

**P3. GN 覆盖严重不足（n=1）**

PoolFormer-S12 是唯一 GN 架构，且 SGD 需要特殊 lr 调整。论文基于 n=1 构建 ordinal encoding（BN=2, GN=1, LN=0），但 LOO 显示移除 PoolFormer 反而增大 ρ_s（0.744→0.866），说明二元 BN/non-BN 比三级 ordinal 更适合当前数据。

**P4. 文献背景与新发现边界不清**

"BN implicit regularization synergizes with explicit regularization under SGD" 在第 220–223 行被写成背景事实，实际上是本文的 empirical finding。文献只分别支持 BN 的各项属性，不直接支持三者协同。

**P5. Wightman 2021 引用方向性错误**

第 24–26 行引用 Wightman 2021（ResNet Strikes Back）支持 "ResNet 默认用 minimal augmentation"，但该文献核心发现恰恰是 ResNet 可以用现代 recipe 大幅提升。

**P6. 缺少直接相关工作：BOCB 2024**

Backbone-Optimizer Coupling Bias（2024）直接讨论 vision backbone-optimizer 耦合，是最直接的相关工作。论文引用了跨领域的 Abouzeid 2026（LLM 方向）却遗漏了 vision 领域的直接前作。

### 🟢 改善级（加分项）

**P7. Zero-cost heuristic 隐含前提未说明**

Norm-swap 结果表明重要的是"原生 norm"而非"当前 norm"。Heuristic 隐含假设用户不修改原生 norm，论文未明确提出。

**P8. 缺少 regularization 组分拆解**

当前 regularization 是 bundle（WD + label smoothing + dropout），无法判断哪个组分驱动 BN/non-BN 分离。

**P9. Norm-swap 缺少 GN 方向**

三个 swap 只涉及 LN↔BN，没有 GN→BN 或 GN→LN 的因果验证。

---

# 第二部分：优化方案

## 2.1 叙事优化方案

### 2.1.1 标题与 Abstract

**标题**：保持不变。"Predicts" 用词可辩护，无需改动。

**Abstract 修改要点**：

1. 将 "we show that under SGD, BN and non-BN architectures are perfectly separated" 改为更审慎的表述：
   > Under SGD, BN and non-BN architectures exhibit a complete directional separation in regularization response (Cliff's δ=1.0, p=0.0095): every BN architecture benefits while every non-BN architecture is harmed.

2. "Norm-swap experiments causally confirm" 后面加限定：
   > Norm-swap experiments causally confirm that normalization is involved in this interaction: changing only the normalization layer amplifies or attenuates the regularization response, though direction reversal requires backbone co-design changes beyond normalization alone.

3. 如跨数据集实验完成，在 Abstract 末尾加：
   > These patterns replicate on Tiny-ImageNet, confirming cross-dataset generality.

### 2.1.2 Introduction

**第 24–26 行**（Wightman 引用修正）：

当前：
> ResNet variants are typically trained with SGD, moderate weight decay, and minimal data augmentation [Wightman et al., 2021]

修改为：
> Historically, canonical CNN baselines such as ResNet were trained with SGD-based recipes featuring moderate weight decay and minimal augmentation. While recent work has shown that modern recipes can substantially improve CNN performance [Wightman et al., 2021], the default community practice still associates architecture families with distinct recipe traditions.

**第 37–38 行**（核心 claim 降调）：

当前（加粗）：
> **normalization type is the architectural property that determines regularization response, and this effect is conditioned on the optimizer.**

修改为：
> **normalization type is the strongest architectural predictor of regularization response under SGD. AdamW largely neutralizes this dependence. Norm-swap experiments confirm normalization is causally involved, but backbone-normalization co-design constrains the magnitude of this effect.**

### 2.1.3 Results & Discussion

**Section 3.2（第 220–223 行）**：明确标记为本文假说而非已证结论

当前：
> This separation is consistent with a mechanistic account in which BatchNorm's implicit regularization (batch-level noise injection, scale invariance, loss-landscape smoothing) synergizes with explicit regularization...

修改为：
> We hypothesize that this separation reflects a synergy between BatchNorm's implicit regularization properties (batch-level noise injection [Ioffe & Szegedy, 2015], scale invariance [Arora et al., 2019], loss-landscape smoothing [Santurkar et al., 2018]) and explicit regularization. While each of these BN properties is individually well-established, their joint interaction with explicit regularization under SGD is a new empirical finding of this work, not a previously demonstrated result.

**Section 3.3（Norm-swap 解读）**：强化 "involved but not sufficient" 框架

在第 271–279 行的 "Three-way interaction model" 段落后，增加一段明确的 scope statement：

> **Normalization as necessary but not sufficient.** The consistent absence of sign reversal across all three swaps establishes a clear scope boundary: normalization type is causally involved in the regularization response but is not the sole determinant. Architectures are co-designed with their normalization layers—the convolutional backbone of ResNet, the attention mechanism of DeiT, and the depthwise convolution of ConvNeXt all participate in shaping the optimization landscape. Swapping the normalization layer alone cannot transfer the full suite of architectural properties that underlie a positive regularization response. This finding reframes the practical implication: the normalization layer serves as a reliable *diagnostic signal* for recipe selection, not as a *modifiable knob* for recipe engineering.

**Section 4.1（Discussion）**：

当前第 314–323 行将文献中的各个单独结果串联成因果链，读起来像已被证明的机制。修改为明确的 "mechanistic hypothesis" 框架：

> **Mechanistic hypothesis.** We propose the following account, which is consistent with but not proven by existing literature. [保留现有文献引用，但每一步标注是 "established"、"consistent with" 还是 "our interpretation"]

### 2.1.4 Related Work

**新增引用**：

1. **BOCB 2024**（Backbone-Optimizer Coupling Bias）——vision 领域直接讨论 backbone-optimizer 耦合的工作，应作为最直接的相关工作引用：
   > Concurrent with our regularization-focused analysis, [BOCB 2024] identifies backbone-optimizer coupling bias in vision architectures from the optimizer-selection perspective. Our work complements this by revealing that normalization type mediates the regularization axis of this coupling.

2. **Abouzeid 2026**——保留但降级为跨领域旁证：
   > In the language modeling domain, Abouzeid (2026) observes analogous normalization-optimizer coupling effects, suggesting that the interaction patterns we identify may extend beyond vision architectures.

3. **ResNet Strikes Back**（Wightman et al., 2021）——在 Related Work 中正确定位：
   > Wightman et al. (2021) demonstrated that ResNet architectures can achieve competitive performance with modernized training recipes, challenging the assumption that CNNs inherently require simple recipes. Our findings provide a mechanistic lens for understanding when such recipe transfers succeed or fail.

### 2.1.5 Conclusion

在结论末尾（当前第 535–538 行）加入显式 scope 限定：

> Our empirical patterns are established on CIFAR-100, a small-scale benchmark where the high model capacity-to-data ratio may amplify regularization sensitivity. Whether the BN/non-BN separation persists at ImageNet scale, where this ratio shrinks, remains an important open question. [若 Tiny-ImageNet 实验完成：We provide initial evidence of cross-dataset generality on Tiny-ImageNet (Appendix X), but full-scale validation on ImageNet-1k is needed.]

---

## 2.2 实验优化方案

### 实验总览与优先级

| 优先级 | 实验 | GPU-hours | 时间线 | 影响 |
|--------|------|-----------|--------|------|
| **P0** | Tiny-ImageNet 跨数据集验证 | ~144 | Day 1–2 | 直接回应 #1 limitation |
| **P0** | 增加 GN 架构（ConvNeXt-V2） | ~24 | Day 1 | GN 从 n=1 → n=2 |
| **P1** | GN→BN Norm-Swap（PoolFormer） | ~24 | Day 2 | 因果证据扩展到 GN |
| **P1** | Tiny-ImageNet 完整 grid（补 strong aug） | ~144 | Day 2–3 | 精确复刻主实验 |
| **P1** | ImageNet-100 Pilot | ~48 | Day 2–3 | 更大 scale 验证 |
| **P2** | Regularization 组分拆解 | ~54 | Day 3 | 机制深度 |
| **P2** | Tiny-ImageNet Norm-Swap | ~48 | Day 3–4 | 跨数据集因果验证 |
| **P2** | 额外 GN 架构（ConvNeXt-V2 Norm-Swap） | ~24 | Day 3–4 | GN 因果证据 |
| **P3** | ImageNet-100 完整 grid | ~192 | Day 4+ | 大 scale 完整覆盖 |
| **P3** | Weight Decay 强度 sweep | ~72 | Day 4+ | 剂量-响应关系 |
| **P3** | RMSNorm 架构 | ~48 | Day 4+ | norm type 覆盖扩展 |
| **P3** | Fine-tuning 实验 | ~96 | Day 4+ | 迁移学习场景 |

---

### P0-1：Tiny-ImageNet 跨数据集验证（最高优先级）

**目的**：回应 "单一数据集" 这一最大 weakness。如果 BN/non-BN 分离在 Tiny-ImageNet 上复现，论文的 generalizability 大幅提升。

**数据集**：Tiny-ImageNet（200 类，100k train / 10k val，64×64 native → resize 至 224×224）

**架构选择**（6 个，覆盖三种 norm）：

| 架构 | Norm | 理由 |
|------|------|------|
| ResNet-18 | BN | 经典 BN baseline |
| EfficientNet-B0 | BN | BN + 非传统 conv（MBConv）|
| ConvNeXt-Tiny | LN | 现代 LN + depthwise conv |
| DeiT-Tiny | LN | Transformer + self-attention |
| MLP-Mixer-S/16 | LN | 纯 MLP，无 conv/attention |
| PoolFormer-S12 | GN | 唯一 GN 代表 |

**Recipe 配置**：与 CIFAR-100 完全一致的 2×2×2=8 recipe grid

**训练 protocol**（尽量与 CIFAR-100 一致以保证可比性）：
- 100 epochs, batch 128, cosine annealing + 5-epoch warmup
- SGD: lr=0.1, momentum=0.9 | AdamW: lr=1e-3
- FP16, gradient clipping (max_norm=5.0)
- Images: resize to 224×224
- Light aug: resize + random crop (padding 16) + horizontal flip
- Strong aug: light + RandAugment(N=2,M=9) + Mixup(α=0.2) + CutMix(α=1.0)
- Regularization Standard: SGD WD=5e-4, AdamW WD=0.05, label smoothing ε=0.1, dropout p=0.1
- Seeds: 42, 123, 456
- 注意：若 PoolFormer SGD 在 lr=0.1 diverge，同样降至 lr=0.01

**分阶段执行**：

| 阶段 | 配置 | Runs | GPU-hours | 在哪些 GPU 上跑 |
|------|------|------|-----------|----------------|
| P0 Phase 1 | 6 archs × 4 recipes（light aug, 2 opt × 2 reg）× 3 seeds | 72 | ~144 | GPU 1–3（~48h）|
| P1 Phase 2 | 6 archs × 4 recipes（strong aug, 2 opt × 2 reg）× 3 seeds | 72 | ~144 | GPU 1–3（Day 2–3）|

**预估单 run 时间**：CIFAR-100 均值 ~63 min，Tiny-ImageNet ~2× → ~126 min/run ≈ 2.1h

**Phase 1 总时间**：72 runs × 2.1h / 3 GPUs ≈ 50h → 需要 ~50h 才能完成。**建议分配 4 GPUs**：72 × 2.1 / 4 ≈ 37.8h ✓

**预期结果与判断标准**：
- ✅ **理想结果**（高概率）：SGD 下 BN 架构 Δ_reg > 0，non-BN 架构 Δ_reg < 0，Cliff's δ ≥ 0.8。→ 论文增加一个完整的跨数据集验证 Table + Figure
- ⚠️ **部分复现**：方向一致但分离不完美（δ ~ 0.5–0.8）。→ 仍有价值，说明 pattern 存在但 dataset-dependent 的幅度差异
- ❌ **未复现**（低概率）：BN/non-BN 分离消失。→ 需要在 limitation 中讨论 dataset specificity，但不影响 CIFAR-100 的 empirical contribution

**产出**：
- 新 Table：Tiny-ImageNet Δ_reg（对标主文 Table 2）
- 新 Figure：SGD 条件下 BN vs. non-BN 的 Δ_reg 分布对比（CIFAR-100 vs. Tiny-ImageNet 并排）
- Cliff's δ + permutation p-value（Tiny-ImageNet 的 SGD 层）
- 在 Abstract/Conclusion 中加入 "replicated on Tiny-ImageNet" 的一句话

---

### P0-2：增加 GN 架构 —— ConvNeXt-V2（CIFAR-100）

**目的**：将 GN 组从 n=1 提升到 n=2，验证 GN 是否 consistently 归入 non-BN 阵营。

**架构选择**：ConvNeXt-V2（timm 中可用，原生使用 GroupNorm）

推荐 variant：
- `convnextv2_atto`（3.7M params）或 `convnextv2_femto`（5.2M params）
- 参数量在现有架构范围内（5.3M EfficientNet-B0 ~ 36.5M WRN-28-10）

**配置**：完全复用 CIFAR-100 主实验 protocol
- 8 recipes × 3 seeds = 24 runs
- 预估：~1h/run × 24 = ~24 GPU-hours

**预期结果**：
- ConvNeXt-V2 的 SGD Δ_reg 应为负值（与 PoolFormer-S12 的 −9.01 类似），支持 GN 归入 non-BN 阵营
- AdamW Δ_reg 应在 ±3 pp 以内

**产出**：
- Table 2 扩展为 11 architectures
- 更新 Cliff's δ 和 ANOVA 分析（n=11，n_BN=4, n_non-BN=7，其中 GN=2）
- LOO 分析更新
- 4-way ANOVA 可扩展至 N=216（9 架构平衡）或报告 11 架构非平衡分析

---

### P1-1：GN→BN Norm-Swap（PoolFormer-S12，CIFAR-100）

**目的**：补全因果证据链。现有 norm-swap 只涉及 LN↔BN，缺少 GN 方向。

**配置**：
- PoolFormer-S12 的 GroupNorm 全部替换为 BatchNorm2d
- 8 recipes × 3 seeds = 24 runs
- 使用 PoolFormer 的 SGD lr=0.01 设置（与主实验一致）
- ~24 GPU-hours

**预期结果**：
- 如果 GN→BN 使 SGD Δ_reg 更负（类似 LN→BN 的 amplification 效应），强化 "非 BN norm 架构不适合直接替换为 BN"
- 如果 GN→BN 使 SGD Δ_reg 转正，说明 GN 确实比 LN 更接近 BN → 支持三级 ordinal 结构

**产出**：
- Table 3（Norm-swap）扩展一行
- 讨论 GN 在因果框架中的位置

---

### P1-2：ImageNet-100 Pilot

**目的**：在更大 scale、原生 224×224 分辨率数据集上验证核心 pattern。即使只有 pilot 数据（1 seed），也能大幅增强论文的 scale generalizability 论点。

**数据集**：ImageNet-100（从 ImageNet-1k 中选取 100 类，~130k training images，原生 224×224）
- 使用常见的 100-class 子集划分（如 torchvision 的 ImageFolder 按字母序前 100 类，或使用 published split）

**架构选择**（4 个核心架构）：

| 架构 | Norm | 理由 |
|------|------|------|
| ResNet-18 | BN | 经典 BN |
| EfficientNet-B0 | BN | 现代 BN |
| ConvNeXt-Tiny | LN | 现代 LN |
| DeiT-Tiny | LN | Transformer LN |

**配置**：
- 4 recipes（SGD×{none,standard} + AdamW×{none,standard}），light augmentation
- 1 seed（快速信号检测）
- 16 runs total
- 训练 protocol：100 epochs, batch 128, cosine schedule, 其余同上

**预估时间**：~3h/run × 16 = ~48 GPU-hours → 在 PRO 6000 上跑 48h

**预期结果**：
- 即使 1 seed，4 architectures 也能看到 BN/non-BN 方向性分离
- 不做正式统计检验（n=2 per group 太小），但作为 "directional evidence at larger scale" 报告

**产出**：
- Appendix 新增一个 Table："ImageNet-100 Pilot Results"
- 在 Discussion 中引用："Pilot experiments on ImageNet-100 (Appendix X) suggest the directional pattern persists at larger scale, though formal replication with expanded architecture coverage is needed."

---

### P2-1：Regularization 组分拆解（CIFAR-100）

**目的**：揭示 BN/non-BN 分离究竟由哪个 regularization 组分驱动（weight decay / label smoothing / dropout）。

**配置**：
- 3 key architectures：ResNet-18 (BN), ConvNeXt-Tiny (LN), PoolFormer-S12 (GN)
- 3 组分条件：
  - **WD-only**: weight decay（SGD: 5e-4, AdamW: 0.05），无 LS，无 dropout
  - **LS-only**: label smoothing ε=0.1，无 WD，无 dropout
  - **Dropout-only**: dropout p=0.1，无 WD，无 LS
- 2 optimizers × 2 augmentations × 3 seeds = 12 runs per (arch, component) pair
- 3 archs × 3 components × 12 = 108 runs
- 快速 pilot（1 aug, 1 seed）：3 × 3 × 2 = 18 runs → ~18 GPU-hours

**预期结果**：
- 假说：weight decay 是主要驱动因素（直接与 BN 的 scale invariance 交互；Arora 2019 的理论框架支持这一点）
- Label smoothing 和 dropout 可能独立于 norm type

**产出**：
- 新 Figure："Component-wise regularization response by normalization type"
- 如果 WD 是主驱动因素，强化了 Arora 2019 的理论联系
- 可作为 "fine-grained mechanistic evidence" 放入 Section 3.4 或新增 Section

---

### P2-2：Tiny-ImageNet Norm-Swap

**目的**：在 Tiny-ImageNet 上验证因果关系。如果 norm-swap 效应在第二个数据集上复现，因果 claim 显著增强。

**配置**：
- 2 swap pairs（与 CIFAR-100 对应）：
  - ConvNeXt-Tiny LN→BN
  - ResNet-18 BN→LN
- 4 recipes × 3 seeds = 24 runs per swap
- 总共 48 runs × ~2.1h = ~101 GPU-hours

**预期结果**：
- ConvNeXt LN→BN：SGD Δ_reg 进一步恶化（复现 amplification effect）
- ResNet BN→LN：SGD Δ_reg 减弱但不翻转（复现 attenuation effect）

**产出**：
- Table 3 扩展或新增 Table："Cross-dataset norm-swap results"

---

### P2-3：ConvNeXt-V2 Norm-Swap（CIFAR-100）

**目的**：用新增的 GN 架构做 norm-swap，扩展因果证据。

**配置**：
- ConvNeXt-V2 GN→BN：将所有 GroupNorm 替换为 BatchNorm2d
- 8 recipes × 3 seeds = 24 runs → ~24 GPU-hours

**预期结果**：与 PoolFormer GN→BN（P1-1）对比，验证 GN→BN swap 效应的一致性。

---

### P3-1：ImageNet-100 完整 Grid

**目的**：完整复刻主实验在 ImageNet-100 scale。

**配置**：
- 6 architectures × 8 recipes × 3 seeds = 144 runs × ~3h = ~432 GPU-hours
- 需要 4 GPUs 跑 ~108 小时（~4.5 天）

**注意**：这个实验在 48h deadline 内无法完成，建议在 P0/P1 完成后启动，用于 rebuttal 或 camera-ready。

---

### P3-2：Weight Decay 强度 Sweep

**目的**：展示 regularization response 的剂量-响应关系，而非仅二元 on/off。

**配置**：
- 3 architectures × 4 WD levels（1e-4, 5e-4, 1e-3, 5e-3）× 2 optimizers × 1 aug × 3 seeds = 72 runs
- ~72 GPU-hours

**预期结果**：
- BN 架构：SGD Δ_reg 随 WD 增大单调递增（正向协同）
- LN 架构：SGD Δ_reg 随 WD 增大单调递减（负向干扰）
- AdamW：所有架构对 WD 不敏感

---

### P3-3：RMSNorm 架构

**目的**：扩展 norm type 覆盖到 LLM 常用的 RMSNorm。

**挑战**：timm 中目前没有原生 RMSNorm 的 vision 架构。可能需要：
- 手动将 DeiT-Tiny 或 ConvNeXt-Tiny 的 LayerNorm 替换为 RMSNorm
- 作为 norm-swap 实验的扩展

**配置**：
- 2 architectures × LN→RMSNorm swap × 8 recipes × 3 seeds = 48 runs → ~48 GPU-hours

**预期结果**：RMSNorm 是 LN 的简化版（去掉 re-centering），预计行为更接近 LN 而非 BN。

---

### P3-4：Fine-tuning 实验

**目的**：验证 pattern 在 pretrained → fine-tuning 场景是否成立（更贴近实际使用场景）。

**配置**：
- 使用 ImageNet-1k pretrained 权重
- 在 CIFAR-100 / Tiny-ImageNet 上 fine-tune
- 4 architectures × 4 recipes × 3 seeds = 48 runs × ~30min = ~24 GPU-hours

**意义**：如果 fine-tuning 场景下 BN/non-BN 分离仍然存在，论文的实用价值大幅提升。

---

## 2.3 Idea 优化方案

### 2.3.1 核心 Reframing：从 "determination" 到 "diagnostic signal"

当前 idea 的问题在于 norm type 被定位为 "决定因素"，但 norm-swap 数据不支持这个定位。

**优化后的 idea 框架**：

```
Normalization type 不是 regularization response 的 "开关"，
而是 backbone-normalization-optimizer 联合设计的 "可观测信号"。

它之所以有 predictive power，不是因为它 "决定" 了 response，
而是因为它与 backbone 的其他属性（stem design, residual topology, 
attention mechanism）共同演化，是这些联合设计的 visible proxy。

实用意义：你不需要理解全部的联合设计，
只需要检查 norm type + optimizer choice 就能做出方向正确的 recipe 决策。
```

### 2.3.2 将 Heuristic 扩展为 Decision Tree

当前的 heuristic 是一句话："check norm type to eliminate half the search space"。优化为结构化的 decision tree：

```
Recipe Selection Decision Tree:
─────────────────────────────
1. Identify the model's NATIVE normalization type
   ├── BatchNorm → BN group
   ├── GroupNorm → non-BN group (aligns with LN in regularization response)
   └── LayerNorm → non-BN group

2. Choose optimizer
   ├── SGD:
   │   ├── BN group → USE standard regularization (WD + LS + dropout)
   │   │              Expected benefit: +0.27 to +7.63 pp
   │   └── non-BN group → REDUCE or SKIP regularization
   │                      Expected harm: −1.00 to −11.75 pp
   └── AdamW:
       └── All groups → regularization has minimal impact (±3 pp)
                        Focus tuning on learning rate and augmentation instead

3. Important: This diagnostic applies to the NATIVE norm of the architecture.
   Do NOT swap normalization layers and then apply the swapped group's recipe.
```

### 2.3.3 强化 AdamW Equalization Finding

这个发现其实很有实用价值但论文没有充分强调：

> 如果你不确定一个架构的 regularization 需求，直接用 AdamW。AdamW 的 per-parameter adaptive learning rates 和 decoupled weight decay 内建了 baseline regularization，使得额外 regularization 的边际效应接近零。这消除了 recipe 选择错误的风险。

这可以作为 "safe default recommendation" 写入 conclusion。

### 2.3.4 Norm-Swap 失败的 Positive Framing

当前论文把 norm-swap 的 "未翻转" 视为部分成功。可以更积极地 frame：

> Norm-swap 的 amplification/attenuation（而非 reversal）pattern 本身是一个 informative finding。它揭示了 backbone-normalization co-design 的深度——这些架构不是简单的 "backbone + norm" 组合，而是经过联合优化的整体。这一发现对 NAS（Neural Architecture Search）有直接启示：norm type 不应被视为独立可搜索的维度。

---

## 2.4 执行时间线与 GPU 分配方案

### 硬件资源

| GPU | 显存 | 主要用途 |
|-----|------|---------|
| 5090 #1 | 32 GB | Tiny-ImageNet |
| 5090 #2 | 32 GB | Tiny-ImageNet |
| 5090 #3 | 32 GB | Tiny-ImageNet |
| 5090 #4 | 32 GB | CIFAR-100 补充实验 |
| PRO 6000 | 96 GB | ImageNet-100 / 大模型实验 |

### Day 1（5月4日 12:00 → 5月5日 12:00）

| 时段 | GPU 1–3 | GPU 4 | PRO 6000 | CPU/写作 |
|------|---------|-------|----------|---------|
| 0–4h | 环境搭建：下载 Tiny-ImageNet、验证 timm 架构可用性、ConvNeXt-V2 可用性检查 | 同左 | 下载 ImageNet-100（如有条件） | 叙事修订开始（Section 2.1 的所有修改）|
| 4–24h | Tiny-ImageNet Phase 1 开跑：72 runs（6 archs × 4 recipes × 3 seeds, light aug） | ConvNeXt-V2 CIFAR-100 主实验：24 runs | ImageNet-100 数据准备 & pilot 开跑（4 archs × 4 recipes × 1 seed = 16 runs）| Related Work 更新（补 BOCB 2024）；Introduction 修改 |

**Day 1 预估产出**：
- GPU 4 完成 ConvNeXt-V2 CIFAR-100 全部 24 runs（~24h）
- GPU 1–3 完成 Tiny-ImageNet ~38/72 runs（进度 ~53%）
- PRO 6000 完成 ImageNet-100 ~8/16 runs（进度 ~50%）
- 叙事修订初稿完成

### Day 2（5月5日 12:00 → 5月6日 12:00 ← DEADLINE）

| 时段 | GPU 1–3 | GPU 4 | PRO 6000 | CPU/写作 |
|------|---------|-------|----------|---------|
| 0–14h | 完成 Tiny-ImageNet Phase 1 剩余 ~34 runs | PoolFormer GN→BN norm-swap：24 runs | 完成 ImageNet-100 pilot 剩余 ~8 runs | ConvNeXt-V2 结果分析 → 更新 Table 2, ANOVA |
| 14–24h | Tiny-ImageNet Phase 2 开始（strong aug, 优先跑 1 seed） | 完成 norm-swap → 结果分析 | ImageNet-100 追加 runs（如时间允许） | **Tiny-ImageNet Phase 1 结果分析**：计算 Δ_reg, Cliff's δ, 更新论文 |

**Day 2 关键 checkpoint**（5月6日 00:00，距 deadline ~12h）：
- Tiny-ImageNet Phase 1（light aug）：✅ 应全部完成
- ConvNeXt-V2 CIFAR-100：✅ 应全部完成
- PoolFormer GN→BN norm-swap：✅ 应全部完成
- ImageNet-100 pilot：✅ 应全部完成
- 叙事修改：✅ 应全部完成

**5月6日 00:00–12:00**：
- 整合所有新实验结果到论文
- 更新 Tables, Figures, Appendix
- Final check & 提交

### Day 3（5月6日 12:00 → 5月7日 12:00，post-submission）

| GPU 1–3 | GPU 4 | PRO 6000 |
|---------|-------|----------|
| Tiny-ImageNet Phase 2 续跑 | Regularization 组分拆解（pilot 18 runs） | ImageNet-100 扩展 |

### Day 4（5月7日 12:00 → 5月8日 12:00，rebuttal prep）

| GPU 1–3 | GPU 4 | PRO 6000 |
|---------|-------|----------|
| Tiny-ImageNet norm-swap | ConvNeXt-V2 norm-swap | ImageNet-100 完整 grid |

---

## 2.5 并行执行策略

### GPU 并行

```
Timeline:
        Day 1              Day 2 (→ deadline)     Day 3              Day 4
GPU1-3: [==Tiny-ImageNet Phase 1 (light)==]→[==Tiny-ImageNet Phase 2 (strong)==]→[TI Norm-Swap]
GPU4:   [=ConvNeXt-V2 CIFAR=][=GN→BN Swap=]→[=Reg Component Ablation=]→[=CV2 Swap=]
PRO:    [===ImageNet-100 Pilot===]→[=====ImageNet-100 Expanded=====]→[=========]
```

### CPU 并行（与 GPU 同步进行）

| 任务 | 时间 | 依赖 |
|------|------|------|
| 叙事修订全稿 | Day 1 | 无（独立于实验）|
| Related Work 更新 | Day 1 | 无 |
| ConvNeXt-V2 结果分析 + Table 更新 | Day 1 end | ConvNeXt-V2 完成 |
| Tiny-ImageNet 结果分析 + 新 Table/Figure | Day 2 mid | Phase 1 完成 |
| Norm-swap 结果分析 | Day 2 end | GN→BN 完成 |
| ImageNet-100 pilot 结果整理 | Day 2 end | Pilot 完成 |
| 全文最终整合 | Day 2 末 → deadline | 所有 P0/P1 完成 |

---

## 2.6 风险预案

### 风险 1：Tiny-ImageNet 上 BN/non-BN 分离不复现

**概率**：低（~15%）。机制层面 BN 的 scale invariance 和 batch-level stochasticity 是 dataset-independent 的。

**应对**：
- 如果方向一致但 δ < 1.0：报告为 "directional pattern persists, effect size attenuated at larger scale"
- 如果完全不复现：在 limitation 中显式讨论 dataset specificity，将 contribution scope 收窄至 CIFAR-100

### 风险 2：ConvNeXt-V2 不在 timm 中或行为异常

**概率**：低。ConvNeXt-V2 在 timm ≥ 0.9.x 中可用。

**应对**：
- 备选 GN 架构：HorNet（使用 GN）、或手动将 RegNet 切换到 GN 模式
- 如果所有 GN 备选都不可行，报告 GN 作为 n=1 的限制并在 Future Work 中讨论

### 风险 3：ImageNet-100 训练时间超预期

**概率**：中（~30%）。Data loading 可能成为瓶颈。

**应对**：
- 先在 PRO 6000 上跑 1 个 architecture 确认时间
- 如果超 4h/run，减少到 2 个 architecture（1 BN + 1 LN）
- 极端情况下放弃 ImageNet-100，专注 Tiny-ImageNet

### 风险 4：PoolFormer GN→BN norm-swap 训练崩溃

**概率**：中（~25%）。PoolFormer 的 average pooling token mixing 可能对 BN 的 batch statistics 不兼容。

**应对**：
- 尝试 lr=0.01（与 PoolFormer SGD 调整一致）
- 如果仍然崩溃，记录为 "GN→BN swap triggers collapse, consistent with norm-backbone co-design hypothesis"——这本身也是有价值的结果

### 风险 5：截稿前实验未全部完成

**应对**：
- P0 实验的设计已确保可在 48h 内完成（最保守估计也只需 ~42h）
- Phase 2（strong aug）未完成不影响核心 claim，可先用 Phase 1 结果提交
- ImageNet-100 pilot 如未完成，不影响论文核心价值——Tiny-ImageNet 复现已足够
- 所有 post-deadline 实验可用于 rebuttal 期间补充

---

## 2.7 提交前 Checklist

### 内容检查
- [ ] Abstract 中所有 "determines" 替换为 "predicts" / "is causally involved"
- [ ] Introduction 加粗声明已降调
- [ ] Wightman 2021 引用方向已修正
- [ ] BOCB 2024 已加入 Related Work
- [ ] Section 3.2 的机制叙述标记为 hypothesis
- [ ] Norm-swap 的 "involved but not sufficient" scope statement 已加入
- [ ] Conclusion 的 scale limitation 已显式说明
- [ ] Heuristic 的 "native norm" 前提已明确
- [ ] 新增实验数据已整合到 Tables/Figures
- [ ] NeurIPS Paper Checklist 已更新

### 实验结果检查
- [ ] Tiny-ImageNet Δ_reg Table 完整（6 archs × 2 optimizers）
- [ ] Tiny-ImageNet Cliff's δ 和 p-value 计算完成
- [ ] ConvNeXt-V2 CIFAR-100 结果已加入 Table 2（更新为 11 architectures）
- [ ] GN→BN norm-swap 结果已加入 Table 3
- [ ] ImageNet-100 pilot 结果已加入 Appendix
- [ ] 所有新增 ANOVA / LOO 分析已更新
- [ ] Appendix 编号和交叉引用已更新
- [ ] Computational Resources（Appendix G）已更新新增实验的 GPU 时间
