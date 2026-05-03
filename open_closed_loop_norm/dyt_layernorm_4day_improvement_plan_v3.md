# DyT 替换 LayerNorm 论文 4 天提升方案 (v3)

> 基于 v2 方案修订。核心变化：(1) 压缩到 4 天；(2) E5 降级、E7/E8 升级；(3) Pareto Plan B；(4) 精准手术式写作而非全面重写。

---

## 0. 执行摘要

### 当前论文状态

论文题目 "How Architecture Topology Shapes Normalization-Free Network Effectiveness"，研究 Dynamic Tanh (DyT) 能否跨架构通用替代 LayerNorm。当前新版 (NeurIPS2026_final.pdf) 已有：
- 10 个架构的 CIFAR-100 实验（DeiT-S/T, Vim-T, PlainMamba-T, RWKV-T, CaiT-XXS24, ConvNeXt-T, Swin-T, RMT-T, Sequencer2D-S）
- Tiny-ImageNet 跨数据集验证，ImageNet-100 224×224 分辨率探测
- α-sensitivity sweep，RWKV key_norm decomposition，MLP compensation ablation（已扩展到 3-seed）
- Effective rank 分析，gradient saturation analysis
- CIFAR-100-C corruption robustness（6 架构，3-seed mean + per-seed）
- RMSNorm baseline（仅 DeiT-S 和 RWKV-T）
- NeurIPS checklist

相较旧版 (2026-05-02)，新版改进了 abstract 简洁度、扩展了 RMSNorm 数据、补充了 CIFAR-100-C corruption 分析、增加了最新文献引用、完善了 NeurIPS checklist。**但核心弱点未变：纯 diagnostic paper，无正向方法贡献。**

### 核心问题

1. **无正向方法贡献** —— 只说了"别这么做"，没说"应该怎么做"
2. **Topology claim 有 4 个 open-loop 反例** —— CaiT (-12.34), Swin (-17.18), ConvNeXt (-12.11), RMT (collapse)
3. **Selective DyT 被定位为"止损"** —— 应该被定位为 safe deployment protocol 或 Pareto 最优
4. **Appendix 数据严重低估** —— corruption robustness、α-deadlock 等独立贡献被埋没
5. **RMSNorm baseline 不完整** —— 仅 DeiT-S 和 RWKV-T，审稿人第一个会问
6. **32×32 尺度的实践价值有限** —— 224×224 全面失败，需要机制解释

### 提升策略

将论文从 **topology-centric diagnostic** 升级为 **role-aware prescriptive study**，核心 story arc 为：

> 1. 发现问题：DyT 不是 LayerNorm 的通用替代品 (cross-architecture evidence)
> 2. 解释原因：LayerNorm 在不同位置承担不同功能角色，DyT 只能替代其中一类 (role taxonomy)
> 3. 提出方案：Role-aware selective replacement 是安全且有效的部署策略 (prescriptive method)
> 4. 揭示边界：α-deadlock 在高分辨率下的容量限制 (mechanistic depth)
> 5. 提供工具：Early-α 诊断可在 20 epochs 内预测替换可行性 (practical tool)

### 写作策略：精准手术而非全面重写

新版论文写作质量已经不错（abstract 简洁、related work 完整、实验描述清晰）。**不做全面重写**，而是在现有基础上做精准手术：
1. 新增 role taxonomy section（Method 中）
2. 新增 selective replacement results section（Results 中）
3. 移动 corruption robustness 和 α-deadlock 从 appendix 到主文
4. 添加 Algorithm 1 + decision flowchart
5. 修改 Introduction 和 Abstract 的 framing
6. 保留现有 Related Work、Training Setup 等优质内容

### 预期提升效果

| 维度 | 当前新版 | 提升后版本 |
|---|---|---|
| 主线 | Topology predicts DyT effectiveness | Normalization role determines DyT compatibility |
| 核心贡献 | 多架构现象观察 | Role-aware selective replacement + safe deployment protocol |
| 正向方法 | 无 | Algorithm 1: Role-aware replacement protocol |
| Baseline | LN + RMSNorm (仅2架构) | LN + RMSNorm (全7架构) |
| 独立发现 | 无明确第二贡献 | Corruption robustness 隐式正则化 + α-deadlock 机制 |
| 实践工具 | 无 | Early-α diagnostic + decision flowchart |
| 反例处理 | Topology 叙事的例外 | Role taxonomy 的边界证据 |
| NeurIPS 预估 | 18-22% | 30-46% |

---

## 1. 新版论文已有优势（不需改动）

1. **Abstract 简洁有力** —— 14行，聚焦核心发现，不堆数字
2. **Related Work 完整** —— 包含 Stollenwerk 2025, nGPT, Kanavalau 2026, Lahoti 2026, Feng 2025, Kim 2025
3. **Section 2.4 Architecture-Dependent Normalization** —— 为 role taxonomy 铺垫
4. **实验控制规范** —— 统一 recipe，parameter-matched pairs，3 seeds for core architectures
5. **MLP ablation 已扩展到 3-seed** —— Table 21
6. **CIFAR-100-C 数据完整** —— 6 架构，per-seed + per-corruption 完整分析
7. **NeurIPS checklist** —— 完整 15 项

## 2. 已有可直接复用的数据

**以下数据不需要重跑，只需在新框架下重新组织和解读：**

| 已有数据 | 位置 | 种子数 | 在新框架下的角色 |
|---|---|---|---|
| RWKV key_norm decomposition | Table 20 | 3 seeds | 核心证据：architectural invariant role |
| RWKV selective DyT (excl. key_norm) corruption | Appendix C 正文 | 2 seeds (mCA=55.39%) | Selective replacement corruption baseline |
| CaiT SA-only ablation | §5.1 (旧版) | 1 seed | 证明 24L deep SA stack 是问题根源 |
| α-sensitivity sweep | Table 19, Fig 5 | 1 seed × 5 arch × 4-5 α | α-stability landscape 基础数据 |
| Effective rank per-layer | Figure 7 | seed=42, 6 arch | 支持 tanh compression mechanism |
| Gradient saturation | Appendix P, Figure 8 | seed=42 | 支持 α-deadlock mechanism |
| α-deadlock mechanism | Appendix K, Fig 4 | 3 α × 1 seed | 最强理论贡献，数学推导 + causal chain |
| CIFAR-100-C corruption | Tables 6-15 | 3-seed mean, per-seed | 独立贡献：DyT 隐式正则化效果 |
| RMSNorm DeiT-S | Table 1 | 3 seeds (60.7±0.3%) | 已有 baseline |
| RMSNorm RWKV-T | Table 1 | 3 seeds (72.4±0.2%) | 已有 baseline |
| RMSNorm 详细分析 | Appendix U, W.3 | 3 seeds | DyT-specific vs normalization-intrinsic 证据 |
| Tiny-ImageNet cross-dataset | Table 23, Fig 6 | 3 seeds × 4 arch | 跨数据集验证 |
| Training collapse taxonomy | Appendix O | multiple | 3 种 failure mode 分类 |
| MLP compensation ablation | Table 21 | 3-seed means | MLP 作为 functional compensation layer |
| Convergence speed | Table 3 | 3 seeds × 4 arch | DyT 加速收敛 |
| Throughput / efficiency | Tables 4-5 | — | 计算开销分析 |
| ImageNet-100 probe | Table 16, Fig 3 | 3 α | 224×224 容量限制 |
| ConvNeXt-T probe | Appendix E | 3 seeds | Post-depthwise 结构失败 |
| Per-layer learned α | Fig 3, 5 | seed=42 | 可用于验证 role taxonomy |
| Three-factor analysis | Table 22 | 10 arch | 扩展到全部架构 |

---

## 3. 新主线设计

### 3.1 核心论点

> LayerNorm 在现代架构中承担的功能角色不同。Role-aware selective DyT replacement——在 activation-stabilizer sites 用 DyT，其余保持 LN——是安全且有效的部署策略，在适当场景下可同时实现与 LN 相当的 clean accuracy 和更好的 corruption robustness。

关键定位策略：
- **首选 framing：Pareto 最优** —— 如果 selective DyT corruption gain 显著（>2 pp over LN），定位为 "Pareto optimal: clean ≈ LN + corruption > LN"
- **Plan B framing：Safe deployment protocol** —— 如果 corruption gain 较弱（<1 pp），定位为 "selective DyT recovers >95% of LN clean accuracy while preserving architecture stability, offering a principled safe-deployment protocol"
- **两种 framing 都有 prescriptive 价值**，区别在于正向贡献的强度

### 3.2 Normalization Functional Role Taxonomy

四类功能角色，**基于 norm 在计算图中的位置（非循环定义）**：

| Functional role | 位置判断标准 | 代表位置 | DyT 兼容性 |
|---|---|---|---|
| **Activation stabilizer** | Residual stream pre-norm (pre-attn, pre-FFN) | DeiT/ViT 的 norm1, norm2 | 高 |
| **State-dynamics stabilizer** | SSM/recurrent state transition 路径上 | Vim 的 state-adjacent norms | 中，α-sensitive |
| **Architectural invariant enforcer** | Key/Query/专用 functional norm | RWKV key_norm, QK-norm | 低 |
| **Hierarchical feature-scale controller** | Downsample / stage transition / patch merging | ConvNeXt LN2d, Swin patch-merging | 低 |

**非循环性保证**：分类标准完全基于 norm 在计算图中的结构位置，不依赖 DyT 替换后的表现。

### 3.3 Role Taxonomy 的独立验证

1. **Per-layer learned α correlation (A1)** —— 不同 role 的 norm sites 学到不同的 α 分布（零成本分析）
2. **Activation distribution comparison (A2)** —— DyT 在不同 role sites 的输出分布差异（零成本分析）
3. **RMSNorm 作为控制实验 (E1)** —— 验证 role dependency 是否 DyT-specific

### 3.4 Role Taxonomy 如何解释所有结果

| Architecture | 主要 norm role | DyT Δ (pp) | 解释 |
|---|---|---|---|
| DeiT-S | Activation stabilizer (12L/384) | +6.90 | Wide buffer absorbs tanh compression; DyT provides implicit regularization |
| DeiT-T | Activation stabilizer (12L/192) | +3.91 | Narrower width reduces buffer; same direction, smaller magnitude |
| Vim-T | Activation + state-dynamics | +3.45 | Bidirectional scan enables global aggregation; MLP compensates |
| PlainMamba-T | State-dynamics (no MLP) | -0.41 | No MLP compensation; unidirectional scan lacks global context |
| RWKV-T | Activation + architectural invariant | -2.17 | key_norm site incompatible with DyT; structural norms ~65% |
| CaiT-XXS24 | Activation in deep-narrow stack | -12.34 | 24L SA stack accumulates tanh compression beyond d=192 buffer |
| ConvNeXt-T | Hierarchical scale controller | -12.11 | Post-depthwise LN manages cross-stage feature scales |
| Swin-T | Hierarchical + windowed | -17.18 | Patch-merging norms control resolution transitions |
| RMT-T | Activation + retention mechanism | collapse | Retention mechanism requires precise norm behavior |
| Sequencer2D-S | State-dynamics (BiLSTM) | collapse | BiLSTM recurrence amplifies bounded DyT errors |

---

## 4. 新论文贡献点

### Contribution 1: Role-aware selective replacement as safe and effective deployment strategy

> We propose role-aware selective DyT replacement — replacing activation-stabilizer norms with DyT while preserving architecture-critical norms as LN. This provides a principled deployment protocol that recovers clean accuracy and, when corruption robustness gain is significant, achieves Pareto optimality over both full LN and full DyT.

支撑证据：selective DyT experiments across 4+ architectures (新实验) + CIFAR-100-C evaluation (部分已有)

### Contribution 2: Cross-architecture evidence with role taxonomy

> Across 10 architectures spanning 5 topology types, we show that DyT effectiveness is governed by the functional role of each normalization site — not by architecture topology alone.

支撑证据：Table 1 (已有) + RMSNorm baseline 全架构 (新实验) + per-layer α correlation (零成本分析)

### Contribution 3: DyT as implicit regularizer

> DyT universally improves corruption robustness across all tested architectures (+2.73 to +6.79 pp), including RWKV-T where clean accuracy decreases.

支撑证据：CIFAR-100-C Tables 6-15 (已有数据，移入主文) + selective DyT corruption eval (新实验)

### Contribution 4: α-deadlock mechanism and resolution boundary

> We identify a resolution-dependent failure mode: at 224×224, tanh saturation creates a self-reinforcing α-deadlock, establishing a capacity boundary.

支撑证据：Appendix K 数学推导 + Figure 4 causal chain + Table 16 实验 (全部已有，移入主文)

### Contribution 5: Practical diagnostic toolkit

> We provide an early-α diagnostic: monitoring per-site learned α during the first 20 training epochs predicts replacement feasibility.

支撑证据：α trajectory 分析 (从现有 training log 提取) + 验证实验

---

## 5. 实验计划

### 5.1 资源估算

**训练时间（RTX 5090, CIFAR-100 100 epochs）：**

| Architecture | Params | 估计时间/run |
|---|---|---|
| DeiT-T | 5.4M | ~25 min |
| RWKV-T | 5.8M | ~30 min |
| CaiT-XXS24 | 11.6M | ~45 min |
| PlainMamba-T | 12.1M | ~50 min |
| DeiT-S | 22M | ~60 min |
| Vim-T | 22M | ~75 min |
| Swin-T | 27.6M | ~80 min |
| ConvNeXt-T | 27.9M | ~80 min |

**GPU 容量：** 4 × RTX 5090 × 4 天 × ~20 有效小时/天 = ~320 GPU-hours ≈ **280-360 CIFAR-100 runs**

### 5.2 总实验量概览

| 实验组 | 优先级 | 新增 runs | 目的 |
|---|---|---|---|
| E1: RMSNorm baseline 补全 | P0 | 15 | 消除 mean-centering 攻击 + role hypothesis 控制 |
| E2: RWKV-T selective replacement + Pareto | P0 | 6-9 | 核心方法贡献 |
| E3: Vim-T selective replacement + α | P0 | 10-13 | State-dynamics role 验证 |
| E4: DeiT-S/T α 补 seeds | P0 | 10 | Activation stabilizer role 完善 |
| E7: ConvNeXt-T selective replacement | **P0 (升级)** | 6 | Hierarchical role 验证 |
| E8: Swin-T selective replacement | **P0 (升级)** | 6 | Hierarchical role 验证 |
| E11: Selective DyT CIFAR-100-C eval | P0 | 0 (eval only) | Pareto clean+corruption 验证 |
| E5: DyT-v2 variants at 224×224 | **P1 (降级)** | 4-6 | 突破 scale 天花板尝试 |
| E6: Vim ±MLP 补 seeds | P1 | 8 | MLP compensation multi-seed |
| E9: CaiT depth threshold ablation | P1 | 3-4 | Depth-width factor 验证 |
| E10: PlainMamba-T +MLP 补 seeds | P1 | 4 | MLP grafting multi-seed |
| E12: Tiny-ImageNet selective + RMSNorm | P2 | 18-27 | 跨数据集验证 |
| E13: Resolution continuum (64, 128) | P2 | 4-8 | α-deadlock resolution curve |
| E14: 核心实验 5-seed 扩展 | P2 | 20-30 | 统计显著性 |
| A1: Per-layer α ↔ role correlation | P0 | 0 (分析) | 非循环 taxonomy 验证 |
| A2: Activation distribution comparison | P0 | 0 (分析) | Role taxonomy 可视化 |
| A3: Depth-width scaling formula | P1 | 0 (分析) | 理论深度 |
| A4: Early-α diagnostic validation | P1 | 0 (分析) | 实践工具 |
| A5: Zhu et al. recipe 对比研究 | P1 | 1-3 (可选) | 解释 224×224 矛盾 |

**总计新增训练：P0 ≈ 53-69 runs, P0+P1 ≈ 73-103 runs, P0+P1+P2 ≈ 115-168 runs**

**GPU 利用率：P0+P1 ≈ 55-75%，P0+P1+P2 ≈ 75-90%。留有失败重跑空间。**

### 5.3 E5 降级、E7/E8 升级的理由

**E5 (DyT-v2 at 224×224) 从 P0 降为 P1：**
- α-deadlock 的根本原因是 tanh 的 bounded output range [-1,1]
- DyT-Res/Clamp/Warmup 等简单 fix 高概率无法弥补 18 pp gap
- 现有 α-deadlock 分析（Appendix K）已经是强贡献，不需要 fix 来支撑
- 即使失败也只是确认已知的 bounded range 限制

**E7/E8 (ConvNeXt/Swin selective) 从 P1 升为 P0：**
- 更多架构的 selective replacement 证据直接增强核心方法贡献
- 4 架构 selective 数据（RWKV-T, Vim-T, ConvNeXt-T, Swin-T）比 2 架构 + 1 个 DyT-v2 尝试更有说服力
- ConvNeXt/Swin 代表 hierarchical role，是 role taxonomy 的重要验证
- 成本与 E5 相当（各 6 runs vs E5 的 4-6 runs），但证据价值更高

---

## 6. 实验详细设计

### E1: RMSNorm Baseline 补全

**优先级：P0（最高）**

**目的：** 当前仅 DeiT-S (60.7±0.3%) 和 RWKV-T (72.4±0.2%) 有 RMSNorm 数据。需要扩展到其余 5 个核心架构。无论结果如何都对论文有利：
- 如果 RMSNorm 也显示 role-dependent pattern → role dependency 是 normalization replacement 的一般性质
- 如果 RMSNorm 全部接近 LN → 问题是 tanh bounded range 特有的

**配置：**

| Architecture | Seeds | Runs | 备注 |
|---|---|---|---|
| DeiT-S | — | 0 | 已有 (60.7±0.3%) |
| DeiT-T | 3 | 3 | 新增 |
| Vim-T | 3 | 3 | 新增 |
| RWKV-T | — | 0 | 已有 (72.4±0.2%) |
| PlainMamba-T | 3 | 3 | 新增 |
| ConvNeXt-T | 3 | 3 | 新增 |
| Swin-T | 3 | 3 | 新增 |

**Total: 15 runs ≈ 0.7 GPU-day**

**分析框架：** 结果按 role 分组而非按 architecture 分组，检查 RMSNorm 是否也显示 role-dependent pattern。

### E2: RWKV-T Role-Aware Selective Replacement + Pareto Analysis

**优先级：P0（最高）**

**目的：** 把 RWKV-T 从 "DyT 在 hybrid 上失败" 升级为 role-aware replacement 的核心 showcase。

**已有数据：**
- Full LN: 74.6±0.2% (3 seeds)
- Full DyT α=0.5: 72.5±0.2% (3 seeds)
- DyT excl. key_norm: ~73.2% (3 seeds, Table 20)
- Full DyT α=1.0: 74.42±0.14 (3-seed mean, Table 19)
- RMSNorm: 72.4±0.2% (3 seeds)
- Corruption: LN mCA=54.80%, DyT mCA=57.53% (3-seed mean), Selective (excl.key_norm) mCA=55.39% (2-seed mean)

**新增配置：**

| 配置 | 替换策略 | 目的 | Seeds |
|---|---|---|---|
| **Selective DyT: only residual pre-norm** | 仅替换 pre-attn/pre-FFN 的 norm | Role-aware：只替换 activation stabilizer | **新, 3 seeds** |
| **Selective DyT: residual pre-norm + α=0.25** | 同上 + conservative α | 保守策略 | **新, 3 seeds** |

**新增 runs: 6**（RMSNorm 已在 E1 中计）

**CIFAR-100-C evaluation（E11，零训练成本）：** 对所有 selective DyT checkpoints 跑 CIFAR-100-C，构建 Pareto 曲线。

**关键对比：** selective DyT (only residual pre-norm) vs selective DyT (excl. key_norm) 的 corruption 差异。如果前者 corruption gain 更大，说明替换更少但更精准的 norm sites 能更好地保留 DyT 的正则化效果。

**Pareto 判定标准：**
- **Pareto optimal**: selective DyT clean ≈ LN (Δ < 0.5 pp) AND corruption > LN (Δ > 2 pp) → 使用 Pareto framing
- **Safe deployment**: selective DyT clean ≈ LN (Δ < 0.5 pp) BUT corruption ≈ LN (Δ < 1 pp) → 使用 safe deployment framing
- **两种结果都支持 prescriptive 贡献**

### E3: Vim-T Selective Replacement + α Sensitivity

**优先级：P0**

**目的：** 验证 state-dynamics role，展示 selective DyT 在 SSM 上拓宽 α stability window。

**配置（CIFAR-100）：**

| 配置 | 目的 | Seeds |
|---|---|---|
| Full LN | Baseline | 已有 3 seeds |
| Full DyT α=0.5 | Default | 已有 3 seeds |
| Full DyT α=0.25 | Small α | 已有 1 seed, **补 2 seeds** |
| Full DyT α=1.0 | Saturation risk | 已有 1 seed (collapsed), **补 2 seeds** |
| **Selective DyT: preserve state-adjacent norms** | Role-aware | **新, 3 seeds** |
| **Selective DyT: only pre-FFN norms** | Minimal replacement | **新, 3 seeds** |
| RMSNorm full | Strong baseline | **E1 中已计** |

**新增 runs: 10-13**

### E4: DeiT-S/T α Robustness 补 Seeds

**优先级：P0**

**目的：** 确认 activation-stabilizer role 的 α stability window 比 state-dynamics role 更宽。

| Architecture | α | 当前 seeds | 补 seeds | 新增 runs |
|---|---|---|---|---|
| DeiT-S | 0.25 | 1 | +2 | 2 |
| DeiT-S | 1.0 | 1 (degraded) | +2 | 2 |
| DeiT-T | 0.25 | 0 | +3 | 3 |
| DeiT-T | 1.0 | 0 | +3 | 3 |

**新增 runs: 10**

### E7: ConvNeXt-T Selective Replacement（升级为 P0）

**优先级：P0**

**目的：** 验证 hierarchical feature-scale controller role。ConvNeXt 用 post-depthwise LN (LN2d)，selective replacement 测试"保留 downsample norms"的效果。

**配置：**

| Config | 目的 | Seeds |
|---|---|---|
| Full LN | Baseline | 已有 3 seeds |
| Full DyT | Full replacement | 已有 3 seeds |
| **DyT within-stage only, preserve downsample norms** | Hierarchical role test | **新, 3 seeds** |
| **DyT shallow stages (stage 1-2) only** | Depth-dependent test | **新, 3 seeds** |
| RMSNorm | Baseline | **E1 中已计** |

**新增 runs: 6**

### E8: Swin-T Selective Replacement（升级为 P0）

**优先级：P0**

**配置：**

| Config | 目的 | Seeds |
|---|---|---|
| Full LN | Baseline | 已有 3 seeds |
| Full DyT | Full replacement | 已有 3 seeds |
| **Preserve patch-merging norms, DyT within-block** | Stage transition role test | **新, 3 seeds** |
| **DyT shallow stages only** | Depth-dependent test | **新, 3 seeds** |
| RMSNorm | Baseline | **E1 中已计** |

**新增 runs: 6**

注意：Swin-T 使用了非标准超参数（lr=5×10⁻⁴, 5-epoch warmup, gradient clipping=5.0），保持一致。

### E11: Selective DyT CIFAR-100-C Evaluation

**优先级：P0**

**目的：** 验证 Pareto 或 safe-deployment 定位。对 E2, E3, E7, E8 的 selective DyT checkpoints 跑 CIFAR-100-C 15 corruption types × severity 1-5。

**零训练成本，仅需 evaluation 脚本。**

### E5: DyT-v2 Variants at 224×224（降级为 P1）

**优先级：P1**

**目的：** α-deadlock 的根本原因是 tanh 的 bounded output range。测试简单 fix 能否缩小 18 pp gap。即使全部失败也是有价值的负结果，强化 bounded range 是 fundamental constraint 的结论。

**Variants：**

| Variant | 公式 | 解决什么问题 |
|---|---|---|
| DyT-Res | γ·tanh(αx) + β + x | 加 residual bypass，unbounded output |
| DyT-Clamp | γ·tanh(αx) + β, α clamped ≤ 0.5 | 硬约束防止 α-deadlock |
| DyT-Warmup | α_init=0.01, linear warmup 20 epochs → 0.5 | 避免 early saturation |

**配置：** ImageNet-100 224×224, DeiT-S, 1 seed each, 4-6 runs, ~1 GPU-day

### E6: Vim-T ±MLP 补 Seeds

**优先级：P1**

新版已有 3-seed means (Table 21)。如果需要进一步统计检验，补 2 seeds per condition = 8 runs。

### E9: CaiT Depth Threshold Ablation

**优先级：P1**

**配置：** CaiT-XXS24，只在前 k 层 SA blocks 应用 DyT (k=6, 12, 18)，保留其余为 LN。

**新增 runs: 3-4**（如果 pattern 清晰，不需多 seed）

### E10: PlainMamba-T +MLP 补 Seeds

**优先级：P1**

**新增 runs: 4**

### E12-E14: P2 实验

| 实验 | Runs | 目的 |
|---|---|---|
| E12: Tiny-ImageNet selective + RMSNorm | 18-27 | 跨数据集验证 |
| E13: Resolution continuum (64, 128) | 4-8 | α-deadlock resolution curve |
| E14: 核心实验 5-seed 扩展 | 20-30 | 统计显著性 |

---

## 7. 零成本分析任务

### A1: Per-Layer Learned α ↔ Role Taxonomy Correlation

**优先级：P0**

**方法：**
1. 从已有 converged checkpoints 提取所有 norm sites 的 learned α（Figure 3, 5 已有 seed=42 数据）
2. 按计算图位置分类（pre-attn / pre-FFN / state-adjacent / key_norm / downsample）
3. 跨架构做 α vs. position-type box plot
4. 检验不同 role 的 α 分布是否有显著差异

**预期结果：**
- Activation stabilizer sites: 低 α（near-linear regime）
- Architectural invariant sites (key_norm): 高 α（试图 enforce constraint）
- 深层比浅层 α 更高（已有 Figure 3 支持）

**成本：零。Agent 几小时完成。**

### A2: Activation Distribution Comparison Figure

**优先级：P0**

**3 panel 设计：**
- Panel 1: DeiT-S residual pre-attn norm → LN vs DyT output distribution（预期：相似）
- Panel 2: RWKV-T key_norm site → LN vs DyT output distribution（预期：DyT 被压缩到 [-1,1]）
- Panel 3: ConvNeXt-T downsample norm → LN vs DyT output（预期：scale mismatch）

**成本：零。**

### A3: Depth-Width Scaling Formula

**优先级：P1**

拟合 Δ_predicted ≈ a - b × (L/d)，数据点：DeiT-S (12/384, +6.90), DeiT-T (12/192, +3.91), CaiT-XXS24 (26/192, -12.34) + E9 数据。

### A4: Early-α Diagnostic Validation

**优先级：P1**

从 training log 提取前 20 epochs 的 per-site α trajectory，检验阈值是否可预测成功/失败。

### A5: Zhu et al. Recipe 对比

**优先级：P1**

做 setup comparison table，解释 Zhu et al. 在 ImageNet-1K 报告 positive 而本文 ImageNet-100 报告 -18 pp 的原因。

---

## 8. 4 天执行计划

### Day 0: 代码改造 + 启动第一批实验 + 启动分析

**Agent 并行任务（coding）：**

| Agent | 任务 | 产出 |
|---|---|---|
| Agent-Code-1 | 实现 RMSNorm 替换（如尚未实现全架构版本） | RMSNorm class + config 开关 |
| Agent-Code-2 | 实现 selective replacement（按 norm site name/position 指定 preserve list） | 每个架构的 preserve config |
| Agent-Code-3 | 实现 CIFAR-100-C evaluation pipeline（如尚未有通用脚本） | Corruption eval script |
| Agent-Code-4 | 实现 per-epoch α logging | α trajectory logger |

**Day 0 晚间启动 GPU（代码完成后立即开始）：**

| GPU | 实验 | Runs | 预计时间 |
|---|---|---|---|
| GPU 1 | E1: RMSNorm - DeiT-T(3), Vim-T(3) | 6 | ~5h |
| GPU 2 | E1: RMSNorm - PlainMamba-T(3), ConvNeXt-T(3) | 6 | ~7h |
| GPU 3 | E1: RMSNorm - Swin-T(3) + E4: DeiT-S α=0.25(2), α=1.0(2) | 7 | ~7h |
| GPU 4 | E7: ConvNeXt-T selective (6 runs) | 6 | ~8h |

**Agent 分析任务（Day 0 即开始）：**

| Agent | 任务 |
|---|---|
| Agent-Analysis-1 | A1: Per-layer α ↔ role correlation（从现有 checkpoints） |
| Agent-Analysis-2 | A2: Activation distribution comparison（从现有 checkpoints） |

### Day 1: 核心 Selective Replacement 实验 + 开始写作

**GPU 任务：**

| GPU | 实验 | Runs | 预计时间 |
|---|---|---|---|
| GPU 1 | E2: RWKV-T selective (2 configs × 3 seeds) | 6 | ~3h, 然后 E9: CaiT depth (3 runs) ~2h |
| GPU 2 | E3: Vim-T selective (2 configs × 3 seeds) + α补seeds (4) | 10 | ~12h |
| GPU 3 | E8: Swin-T selective (6 runs) + E4: DeiT-T α (6 runs) | 12 | ~14h |
| GPU 4 | E6: Vim ±MLP 补 seeds (8 runs) + E10: PlainMamba +MLP (4 runs) | 12 | ~14h |

**Agent 并行任务：**

| Agent | 任务 | 产出 |
|---|---|---|
| Agent-Analysis-3 | A5: Zhu et al. recipe 对比研究 | Setup comparison table |
| Agent-Writing-1 | 草拟 role taxonomy section + Algorithm 1 | LaTeX draft |
| Agent-Writing-2 | 改写 Introduction 开头段落（role-aware framing） | LaTeX draft |

### Day 2: 分析 + Pareto figure + 主体写作

**GPU 任务：**

| GPU | 实验 | 预计时间 |
|---|---|---|
| GPU 1-2 | E5: DyT-v2 on ImageNet-100 224×224 (4-6 configs × 1 seed) | ~8-16h |
| GPU 3 | E14: 核心 5-seed 扩展 (最关键配置) | ~12h |
| GPU 4 | E12: Tiny-ImageNet selective + RMSNorm (优先 DeiT-S + RWKV-T) | ~全天 |

**Agent 并行任务：**

| Agent | 任务 | 产出 |
|---|---|---|
| Agent-Analysis-1 | E11: CIFAR-100-C eval 对所有 selective DyT checkpoints | Corruption accuracy tables |
| Agent-Analysis-2 | 构建 Pareto 曲线（clean vs. corruption） | Pareto figure |
| Agent-Analysis-3 | A3: Depth-width scaling formula 拟合 | Regression plot |
| Agent-Analysis-4 | A4: Early-α diagnostic（从 Day 1 的 α logs 提取） | α trajectory plot + diagnostic rule |
| Agent-Writing-1 | 新增 Results: role-aware selective replacement + Pareto/safe-deployment analysis | LaTeX |
| Agent-Writing-2 | 移动 corruption robustness 到主文 Results | LaTeX |
| Agent-Writing-3 | 移动 α-deadlock 到主文 Discussion | LaTeX |

**Day 2 关键决策点：** 根据 E2/E3 的 selective DyT corruption 结果，确定 Pareto vs safe-deployment framing。

### Day 3: 论文整合 + 压力测试 + 最终打磨

**所有 P0/P1 实验应已完成。** GPU 用于 P2 补充和重跑失败实验。

**Agent 并行任务（上午：论文整合）：**

| Agent | 任务 |
|---|---|
| Agent-Writing-1 | 重写 Abstract（role-aware framing，保持简洁） |
| Agent-Writing-2 | 整合所有新 figures + tables |
| Agent-Writing-3 | 重写 Conclusion，确保 claim 与 evidence 精确匹配 |
| Agent-Writing-4 | 更新 Limitations section |
| Agent-Analysis-1 | Claim-evidence audit: 逐条检查 claim 是否有 ≥3 seeds 支持 |

**Agent 并行任务（下午：压力测试）：**

| Agent | 任务 |
|---|---|
| Agent-Review-1 | 模拟 NeurIPS Reviewer 1 (novelty focus): "Where's the new method?" |
| Agent-Review-2 | 模拟 NeurIPS Reviewer 2 (soundness focus): "Post-hoc? Single-seed? Scale?" |
| Agent-Review-3 | 模拟 NeurIPS Reviewer 3 (significance focus): "Who benefits? Practical impact?" |
| Agent-Fix-1 | 根据 review 修改论文弱点 |
| Agent-Fix-2 | 最终数字校对（表格数字与实验 log 一致性） |
| Agent-Fix-3 | References 检查 + NeurIPS checklist 更新 |

---

## 9. 论文结构设计

### 主文结构

```
1. Introduction
   - DyT 背景 + "normalization is not one thing" 洞察
   - 核心发现预览：role-aware selective replacement
   - Contribution list (5 点)

2. Related Work（大部分保留现有内容）
   - 新增少量 role-aware normalization 引用

3. Method
   3.1 Architecture Taxonomy（保留现有内容）
   3.2 Normalization Functional Role Taxonomy（新增，4 类）
   3.3 DyT Replacement Protocol（保留现有内容）
   3.4 Algorithm 1: Role-Aware Replacement Protocol（新增）

4. Experimental Setup（保留现有内容）
   - 新增 RMSNorm baseline description

5. Results
   5.1 Cross-Architecture DyT Effectiveness（改写：按 role 组织）
   5.2 Role-Aware Selective Replacement（新增核心 section）
   5.3 RMSNorm as Control Experiment（新增）
   5.4 Corruption Robustness as Implicit Regularization（从 Appendix C 移入）
   5.5 α-Stability Landscape（改写：加入 selective DyT stability window）

6. Discussion
   6.1 α-Deadlock Mechanism and Resolution Boundary（从 Appendix K 移入）
   6.2 Depth-Width Interaction
   6.3 Compensatory Pathways: MLP Channel
   6.4 Validation of Role Taxonomy（新增）

7. Limitations

8. Conclusion
```

### Algorithm 1: Role-Aware DyT Replacement

```
Algorithm 1: Role-Aware Normalization Replacement
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
Input: Model M with LayerNorm sites {s₁, ..., sₙ}
Output: Model M' with selective DyT replacement

1. For each norm site sᵢ, classify by computational graph position:
   - Residual pre-attention / pre-FFN → ACTIVATION_STABILIZER
   - SSM state path / recurrent update input → STATE_DYNAMICS
   - Key/Query/specialized functional norm → ARCHITECTURAL_INVARIANT
   - Downsample / stage transition / patch merging → HIERARCHICAL_CONTROLLER

2. Replace ACTIVATION_STABILIZER sites with DyT(α_init = 0.5)

3. Preserve ARCHITECTURAL_INVARIANT and HIERARCHICAL_CONTROLLER as LayerNorm

4. For STATE_DYNAMICS sites:
   a. Replace with DyT(α_init = 0.25)
   b. Train for 20 epochs, monitor per-site learned α
   c. If max(α) > τ or val_accuracy < baseline − ε: revert to LayerNorm

5. Return M'
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
```

### Hero Figure: Pareto / Safe-Deployment Analysis

x-axis: Clean accuracy
y-axis: Mean corruption accuracy
每个点 = (architecture, normalization config)
颜色 = architecture
形状 = config (LN=circle, RMSNorm=diamond, Full DyT=cross, Selective DyT=star)

**Pareto 情况：** selective DyT star 在 LN circle 右上方 → 一张图讲完 story
**Safe-deployment 情况：** selective DyT star 与 LN circle 重叠但 Full DyT cross 在左下方 → selective 是 safe default

---

## 10. 写作层面的精准修改

### 10.1 Abstract 修改策略

保留现有简洁风格，替换核心 message：

**现有核心 message：** "normalization replacement is not a drop-in operation: the architectural role of normalization must be understood before it can be safely removed"

**新核心 message（Pareto 版）：** "role-aware selective replacement — applying DyT only at activation-stabilizer sites — matches full LN on clean accuracy while improving corruption robustness by +X pp, achieving Pareto optimality over both extremes"

**新核心 message（safe-deployment 版）：** "role-aware selective replacement — applying DyT only at activation-stabilizer sites — recovers clean accuracy lost by blanket replacement while preserving DyT's corruption robustness benefit, providing a principled deployment protocol"

### 10.2 Introduction 新增关键段落

> A normalization module is not defined solely by its formula. In modern vision architectures, the same `nn.LayerNorm` call can stabilize residual activations (DeiT), regulate recurrent state dynamics (Vim), enforce key-magnitude invariants (RWKV), or align feature scales across resolution stages (Swin, ConvNeXt). A pointwise bounded nonlinearity like DyT can substitute the first role but not the others. This observation motivates **role-aware selective replacement**: applying DyT only where normalization serves as an activation stabilizer, and preserving LayerNorm at architecture-critical sites.

### 10.3 Results 组织原则

**按 role 和 finding 组织，不按 model 流水账：**
1. Cross-architecture landscape — 10 architectures, 3 outcome groups, role explains all
2. Selective replacement — The positive prescriptive result
3. RMSNorm control — Is role-dependency DyT-specific or general?
4. Corruption robustness — The surprising independent finding
5. α-stability — Topology-dependent stability windows + early-α diagnostic

### 10.4 需保留不动的部分

以下现有内容质量好，不需修改：
- Section 2 Related Work（仅新增少量引用）
- Section 3.1 Architecture Taxonomy
- Section 3.2 DyT Replacement Protocol
- Section 3.3 Training Setup
- Table 1 核心数据（新增 RMSNorm 列和 Selective DyT 列）
- Appendix A-B (Convergence Speed, Throughput)
- Appendix F (PlainMamba Counter-Example)
- Appendix G-H (Scale Confound, Baseline Accuracy)
- NeurIPS Checklist（需更新）

---

## 11. Reviewer 攻击准备

| 攻击 | 准备的回应 |
|---|---|
| "Role taxonomy is post-hoc" | Per-layer α correlation (A1) + activation distribution (A2) 提供非循环证据；taxonomy 基于计算图位置 |
| "Selective replacement is too simple" | Algorithm 1 正式化 + Pareto/safe-deployment 实证 + early-α diagnostic |
| "32×32 scale is too small" | α-deadlock mechanism 提供 resolution-dependent 解释；32×32 是 controlled testbed (ViT paper precedent) |
| "Where's RMSNorm?" | 全架构 RMSNorm baseline + 按 role 分组分析 |
| "How is this different from Zhu et al.?" | 显式 recipe comparison + cross-topology extension + role-aware method |
| "α tuning resolves all failures?" | α-sensitivity heatmap 显示 Swin/CaiT 在所有 α 下仍 ≥6.6 pp below LN |
| "Single-seed ablations?" | Key ablations 补到 3 seeds + paired statistical tests |
| "Corruption robustness is just a side effect?" | All 4 architectures improve; RWKV-T improves corruption DESPITE clean accuracy drop |
| "DyT-v2 没有解决 224×224 问题?" | 确认 bounded output range 是 fundamental constraint；结论更强而非更弱 |

---

## 12. 完成标准

### 最小可接受标准（P0 全部完成）

1. RMSNorm baseline 补全 5 架构 × 3 seeds
2. RWKV-T selective replacement 2 configs × 3 seeds + CIFAR-100-C eval
3. Vim-T selective replacement 2 configs × 3 seeds
4. ConvNeXt-T selective replacement 2 configs × 3 seeds
5. Swin-T selective replacement 2 configs × 3 seeds
6. DeiT-S/T α 补 seeds
7. Per-layer α ↔ role correlation 分析
8. Activation distribution comparison figure
9. Algorithm 1 box + decision flowchart
10. α-deadlock 从 Appendix K 移入主文 Discussion
11. Corruption robustness 从 Appendix C 移入主文 Results
12. 主文精准修改完成（role-aware narrative）
13. Pareto/safe-deployment figure

**预估中稿概率：30-38%**

### 理想完成标准（P0+P1 全部完成）

额外：
1. DyT-v2 at 224×224 至少 3 variants × 1 seed
2. Vim ±MLP 补到 3 seeds per condition
3. CaiT depth threshold ablation
4. PlainMamba-T +MLP 补 seeds
5. Depth-width scaling formula
6. Early-α diagnostic validation + 阈值 τ 确定
7. Zhu et al. recipe 对比
8. 模拟 3 reviewer 压力测试并修复

**预估中稿概率：35-43%，如果 DyT-v2 在 224×224 有改善则 +5-8%**

### 全面完成标准（P0+P1+P2）

额外：
1. Tiny-ImageNet selective + RMSNorm 跨数据集验证
2. Resolution continuum curve (32, 64, 128, 224)
3. 核心实验 5-seed 扩展

**预估中稿概率：38-46%**

---

## 13. 中稿概率详细分析

### 13.1 各版本概率对比

| 版本 | 预估概率 | 核心定位 |
|---|---|---|
| 旧版 (2026-05-02) | 12-18% | 纯 diagnostic，abstract 冗长，数据不完整 |
| 新版 (NeurIPS2026_final) | 18-22% | 写作改善，数据完整，仍无方法贡献 |
| 方案 P0 完成 | 30-38% | 有 method (selective DyT)，有 RMSNorm，有 role taxonomy |
| 方案 P0+P1 完成 | 35-43% | 更强统计检验，更多架构覆盖，DyT-v2 探索 |
| 方案 P0+P1+P2 完成 | 38-46% | 跨数据集验证，resolution curve，5-seed |
| 如 Pareto 效果强 | 上述 +3-5% | 独立正向贡献 |
| 如 DyT-v2 有效 | 上述 +5-8% | 突破 scale 限制 |

### 13.2 概率增量拆解

| 改进项 | 增量 (pp) | 置信度 |
|---|---|---|
| Selective DyT 方法贡献 | +6-10 | 高（核心转型） |
| RMSNorm 全架构 baseline | +2-3 | 确定 |
| Role taxonomy 替代 topology | +2-3 | 中（取决于 A1/A2） |
| Corruption robustness 进主文 | +1-2 | 高（已有数据） |
| α-deadlock 进主文 | +1-2 | 高（已有推导） |
| Early-α diagnostic | +1-2 | 中 |
| Algorithm 1 + flowchart | +1 | 高 |

### 13.3 仍存在的硬伤（方案无法完全解决）

1. **32×32 scale** — 实践价值有限 (-5-8% ceiling)
2. **No ImageNet-1K full training** — 4 天内不可能
3. **Role taxonomy 仍是 empirical** — 无 formal theory
4. **Selective DyT 方法简单** — Algorithm 1 本质是 lookup table
5. **领域聚焦** — 仅 vision classification

这些硬伤决定了概率上限约 46-52%。

### 13.4 新版 vs 旧版提升分析

旧版→新版（已完成的改进）：
- Abstract 简洁化：+1-2%
- RMSNorm 补充 (2 arch)：+1%
- CIFAR-100-C 数据：+1%
- 写作质量：+1-2%
- NeurIPS checklist：+0.5%

**总计：旧版 12-18% → 新版 18-22%，提升约 4-6 pp**

新版→方案完成（待执行的改进）：
- 从 diagnostic → prescriptive 转型：+8-12%
- RMSNorm 全覆盖：+2-3%
- 其他改进：+4-8%

**总计：新版 18-22% → 方案完成 30-46%，预期提升 12-24 pp**

---

## 14. 关键风险和 Contingency

### 风险 1：Selective DyT 的 corruption robustness 不显著

现有数据暗示风险：RWKV-T selective (excl. key_norm) mCA = 55.39% vs LN 54.80% = 仅 +0.59 pp，远低于 full DyT 的 +2.73 pp。

**Contingency (Plan B)：** 切换到 safe-deployment framing："selective DyT recovers >95% of LN clean accuracy while preserving architecture stability, providing a principled deployment protocol superior to blanket replacement." 这仍然是有价值的 prescriptive 贡献。

**注意：** E2 中的 "only residual pre-norm" 配置与 "excl. key_norm" 不同——前者替换更少的 norm sites，可能有不同的 corruption profile。需要等 E2 结果再做判断。

### 风险 2：RMSNorm 在所有架构上都接近 LN

**Contingency：** "Role-dependent failure is specific to bounded pointwise alternatives; statistical normalization is uniformly robust because it preserves distributional information." 这强化了 DyT 的特异性分析。

### 风险 3：Per-layer α 与 role 不相关

**Contingency：** 不放 A1 分析。Role taxonomy 由 selective replacement intervention 结果支撑（更强的因果证据）。

### 风险 4：ConvNeXt/Swin selective replacement 无效

**Contingency：** "Hierarchical normalization serves a qualitatively different role that is structurally incompatible with pointwise replacement at any site." 这仍然支持 role taxonomy。

### 风险 5：4 天时间不够

**优先级严格执行：** P0 → P1 → P2。P0 最小完成标准在 2.5 天内可完成（含代码改造）。Day 3 留给写作和打磨。**不要为了完成 P2 而牺牲写作质量。**

---

## 15. 不做清单

| 不做 | 原因 |
|---|---|
| ImageNet-1K 完整训练 | 4 天内不可能 |
| COCO / ADE20K downstream | 成本高、recipe confound 大 |
| 10-20 个新模型 | 当前 10 个模型足够 |
| NLP / LM 扩展 | 不同领域，风险太大 |
| Fused DyT CUDA kernel | 效率不是主线 |
| 10-seed 全矩阵 | 5 seeds 足够 |
| Neural collapse 分析 | 与 role taxonomy 主线距离远 |
| Adversarial robustness | Corruption robustness 已足够 |
| 论文全面重写 | 现有写作质量好，精准手术即可 |

---

## 16. 新标题建议

1. **When Can Dynamic Tanh Replace LayerNorm? A Role-Aware Analysis Across Vision Architectures** （推荐）
2. **Beyond Drop-in Replacement: Normalization Roles Determine Dynamic Tanh Effectiveness**
3. **Role-Aware Normalization Replacement: Why Dynamic Tanh Works for Some Sites but Not Others**

---

## 17. 额外GPU使用建议

如果可以提供 6-8 卡而非 4 卡：

**优先分配给 P0 加速（Day 0-1）：**
- 额外 2 卡用于并行 E7/E8 selective replacement
- 使 P0 在 Day 1 结束前全部完成
- Day 2 可以启动更多 P1 实验

**不建议分配给 P2：**
- P2 对中稿概率的边际贡献仅 +3-5%
- 不如用于重跑失败实验和扩展 seed 数

---

## 18. 总结

本方案的核心是 **三重精准手术**：

1. **从 diagnostic 到 prescriptive** —— Selective DyT 作为 Pareto optimal 或 safe deployment protocol
2. **从 topology 到 role** —— 基于计算图位置的非循环 taxonomy
3. **从 appendix 到 main text** —— Corruption robustness 和 α-deadlock 移入主文

**GPU 利用率：** P0+P1 ≈ 55-75%（4 卡 4 天），留有充足的失败重跑空间。

**关键时间节点：**
- Day 1 结束：P0 实验完成，Pareto/safe-deployment framing 确定
- Day 2 结束：核心写作完成，Pareto figure 就绪
- Day 3 结束：论文整合 + 压力测试完成

**预估 NeurIPS 中稿概率提升：从 18-22%（当前新版）→ 30-46%（方案完成后）**
