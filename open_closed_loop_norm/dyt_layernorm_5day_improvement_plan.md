# DyT 替换 LayerNorm 论文 5 天提升方案 (v2)

## 0. 执行摘要

### 当前论文状态

论文题目 "How Architecture Topology Shapes Normalization-Free Network Effectiveness"，研究 Dynamic Tanh (DyT) 能否跨架构通用替代 LayerNorm。已有 10 个架构的 CIFAR-100 实验（DeiT-S/T, Vim-T, PlainMamba-T, RWKV-T, CaiT-XXS24, ConvNeXt-T, Swin-T, RMT-T, Sequencer2D-S），Tiny-ImageNet 跨数据集验证，ImageNet-100 224×224 分辨率探测，α-sensitivity sweep，RWKV key\_norm decomposition，MLP compensation ablation，effective rank 分析，CIFAR-100-C corruption robustness 等。论文数据丰富，但主线 (topology predicts DyT effectiveness) 有 4 个 open-loop 反例，claim-evidence 错位。

### 核心问题

当前论文是一篇 **diagnostic paper**（"DyT 在这些架构上不 work"），但 NeurIPS 更偏好 **prescriptive paper**（"这是正确的做法，且比现有方案更好"）。论文最大的弱点不是实验不够，而是：

1. **无正向方法贡献** —— 只说了"别这么做"，没说"应该怎么做"
2. **Topology claim 有 4 个 open-loop 反例** —— 框架解释力不够
3. **Selective DyT 被定位为"止损"** —— 应该被定位为"Pareto 最优"
4. **已有 appendix 数据严重低估** —— corruption robustness、α-deadlock 等独立贡献被埋没
5. **缺 RMSNorm baseline** —— 审稿人第一个会问
6. **32×32 尺度的实践价值有限** —— 224×224 全面失败，需要 fix 或机制解释

### 提升策略

将论文从 **topology-centric diagnostic** 升级为 **role-aware prescriptive study**，核心 story arc 为：

> 1. 发现问题：DyT 不是 LayerNorm 的通用替代品 (cross-architecture evidence)
> 2. 解释原因：LayerNorm 在不同位置承担不同功能角色，DyT 只能替代其中一类 (role taxonomy)
> 3. 提出方案：Role-aware selective replacement 是 Pareto 最优——保持 clean accuracy + 提升 corruption robustness (prescriptive method)
> 4. 揭示边界：α-deadlock 在高分辨率下的容量限制 (mechanistic depth)
> 5. 提供工具：Early-α 诊断可在 20 epochs 内预测替换可行性 (practical tool)

### 预期提升效果

| 维度 | 当前版本 | 提升后版本 |
|---|---|---|
| 主线 | Topology predicts DyT effectiveness | Normalization role determines DyT compatibility |
| 核心贡献 | 多架构现象观察 | Role-aware selective replacement + Pareto optimality |
| 正向方法 | 无 | Algorithm 1: Role-aware replacement protocol |
| Baseline | 仅 LN | LN + RMSNorm |
| 独立发现 | 无明确第二贡献 | Corruption robustness 隐式正则化 + α-deadlock 机制 |
| 实践工具 | 无 | Early-α diagnostic + decision flowchart |
| 反例处理 | Topology 叙事的例外 | Role taxonomy 的边界证据 |
| NeurIPS 预估 | 15-20% | 35-45% |

---

## 1. 当前论文分析

### 1.1 已有优点

1. **问题重要且自然** —— DyT (Zhu et al., 2025) 原始工作验证标准 Transformer / ViT / LLM，检查跨 topology 迁移是自然延伸
2. **实验现象丰富** —— 10 个架构，3 个结果方向（positive / near-zero / negative-or-collapse），非 trivial binary outcome
3. **已有丰富的机制线索** —— key\_norm decomposition, α-sensitivity sweep, effective rank, MLP ablation, gradient saturation analysis, convergence speed, throughput benchmarks
4. **Appendix 数据金矿未充分利用** —— CIFAR-100-C corruption robustness (Table 9-10), α-deadlock mechanism (Appendix K), training collapse taxonomy (Appendix O), per-layer learned α (Figure 5-6), gradient saturation (Appendix P) 都是高价值数据
5. **实验控制规范** —— 统一 recipe，parameter-matched pairs，3 seeds for core architectures

### 1.2 已有可直接复用的 Appendix 数据

**以下数据不需要重跑，只需在新框架下重新组织和解读：**

| 已有数据 | 位置 | 种子数 | 在新框架下的角色 |
|---|---|---|---|
| RWKV key\_norm decomposition | Table 2 | 3 seeds | 核心证据：architectural invariant role，structural ~65% + key\_norm ~35% |
| CaiT SA-only ablation | §5.1 | 1 seed | 证明 24L deep SA stack 是问题根源，支持 depth-width factor |
| α-sensitivity sweep | Table 14, Fig 8 | 1 seed × 5 arch × 4-5 α | α-stability landscape 的基础数据 |
| Effective rank per-layer | Figure 3 | seed=42, 6 arch | 支持 tanh compression mechanism |
| Gradient saturation | Appendix P | seed=42 | 支持 α-deadlock mechanism |
| α-deadlock mechanism | Appendix K, Fig 7 | 3 α × 1 seed | 最强理论贡献，有数学推导 + causal chain + 实验验证 |
| CIFAR-100-C corruption | Tables 9-10 | 3-seed mean, 2-seed per-corruption | 独立贡献：DyT 隐式正则化效果 |
| PlainMamba-T per-seed | Table 12 | 3 seeds | Clean negative result for closed-loop |
| Tiny-ImageNet cross-dataset | Table 4, 13 | 3 seeds × 4 arch | 跨数据集验证 |
| Training collapse taxonomy | Appendix O | multiple | 3 种 failure mode 分类 |
| PlainMamba-Wide collapse | §5.3 | 3 seeds | MLP compensation 必要不充分 |
| Convergence speed | Table 6 | 3 seeds | DyT 加速收敛但不影响 peak |
| Throughput / efficiency | Tables 7-8 | — | 计算开销分析 |
| ImageNet-100 probe | Table 11, Fig 6 | 3 α | 224×224 容量限制 |
| ConvNeXt-T probe | Appendix E | 3 seeds | Pre-norm pattern 外架构失败 |
| Per-layer learned α | Figure 5 | seed=42 | 可用于验证 role taxonomy 的非循环证据 |
| MLP ablation | Table 3 | 1 seed | Vim-T ±MLP, PlainMamba-T +MLP |

### 1.3 主要风险

#### 风险 1：正向贡献不足（最严重）

审稿人核心评价会是："Interesting observations but no method contribution." 仅报告负结果不够 NeurIPS，必须有 prescriptive takeaway。

#### 风险 2：Topology claim 与数据不符

4 个 open-loop 反例 (CaiT -12.34, Swin -17.18, ConvNeXt -12.11, RMT collapse) 直接驳斥 "open-loop ⇒ DyT works"。Topology 只能作为 organizing lens，不能作为 predictor。

#### 风险 3：32×32 尺度限制实践价值

ImageNet-100 224×224 上 DyT 全面失败 (-18 pp)。审稿人会问："谁会在 32×32 上部署？" 必须提供机制解释或 fix。

#### 风险 4：机制解释偏 post-hoc

Depth-width, MLP compensation, α saturation 等解释是事后归纳。需要 intervention evidence 或 non-circular diagnostic。

#### 风险 5：缺 RMSNorm baseline

审稿人第一个问题："DyT 的好处是否只是因为 LN 的 mean-centering 不必要？" RMSNorm 是最自然的控制条件。

#### 风险 6：关键 ablation seed 数不足

Vim-T ±MLP, PlainMamba-T +MLP, α-sensitivity sweep, CaiT SA-only ablation 等均为 single-seed。

---

## 2. 新主线设计

### 2.1 核心论点

> LayerNorm 在现代架构中承担的功能角色不同。Role-aware selective DyT replacement——在 activation-stabilizer sites 用 DyT，其余保持 LN——不仅比 full DyT 更安全，还能实现比 full LN 更好的 clean-robustness trade-off，因为 DyT 在适当位置提供隐式正则化。

关键转变：Selective DyT 不是"止损"（recover from full DyT degradation），而是 **"Pareto 最优"**（同时优于 full LN 和 full DyT）。

### 2.2 Normalization Functional Role Taxonomy

四类功能角色，**基于 norm 在计算图中的位置（非循环定义）**：

| Functional role | 位置判断标准 | 代表位置 | DyT 兼容性 |
|---|---|---|---|
| **Activation stabilizer** | Residual stream pre-norm (pre-attn, pre-FFN) | DeiT/ViT 的 norm1, norm2 | 高 |
| **State-dynamics stabilizer** | SSM/recurrent state transition 路径上 | Vim 的 state-adjacent norms | 中，α-sensitive |
| **Architectural invariant enforcer** | Key/Query/专用 functional norm | RWKV key\_norm, QK-norm | 低 |
| **Hierarchical feature-scale controller** | Downsample / stage transition / patch merging | ConvNeXt LN2d, Swin patch-merging | 低 |

**非循环性保证**：分类标准完全基于 norm 在计算图中的结构位置，不依赖 DyT 替换后的表现。

### 2.3 Role Taxonomy 的独立验证

除了 selective replacement intervention，还通过以下非循环证据验证：

1. **Per-layer learned α correlation** —— 不同 role 的 norm sites 是否学到不同的 α 分布（从现有 checkpoint 分析，零成本）
2. **Activation distribution comparison** —— DyT 在不同 role sites 的输出分布是否近似 LN（从现有 checkpoint 分析，零成本）
3. **RMSNorm 作为控制实验** —— 如果 RMSNorm 在所有 sites 都接近 LN，说明问题是 tanh 特有的；如果 RMSNorm 也显示 role-dependent pattern，说明 role dependency 是 normalization replacement 的一般性质

### 2.4 新主线如何解释所有结果

| Architecture | 主要 norm role | DyT Δ (pp) | 解释 |
|---|---|---|---|
| DeiT-S | Activation stabilizer (12L/384) | +6.90 | Wide buffer absorbs tanh compression; DyT provides implicit regularization |
| DeiT-T | Activation stabilizer (12L/192) | +3.91 | Narrower width reduces buffer; same direction, smaller magnitude |
| Vim-T | Activation + state-dynamics | +3.45 | Bidirectional scan enables global aggregation; MLP compensates; but α-fragile |
| PlainMamba-T | State-dynamics (no MLP) | -0.41 | No MLP compensation; unidirectional scan lacks global context |
| RWKV-T | Activation + architectural invariant | -2.17 | key\_norm site incompatible with DyT; structural norms account for ~65% |
| CaiT-XXS24 | Activation in deep-narrow stack | -12.34 | 24L SA stack accumulates tanh compression beyond d=192 buffer |
| ConvNeXt-T | Hierarchical scale controller | -12.11 | Post-depthwise LN manages cross-stage feature scales |
| Swin-T | Hierarchical + windowed | -17.18 | Patch-merging norms control resolution transitions; windowed attn lacks global stats |
| RMT-T | Activation + retention mechanism | collapse | Retention mechanism requires precise norm behavior |
| Sequencer2D-S | State-dynamics (BiLSTM) | collapse | BiLSTM recurrence amplifies bounded DyT errors |

---

## 3. 新论文贡献点

### Contribution 1: Role-aware selective replacement as Pareto-optimal strategy

> We propose role-aware selective DyT replacement — replacing activation-stabilizer norms with DyT while preserving architecture-critical norms as LN. This achieves **Pareto optimality**: matching full LN on clean accuracy while surpassing it on corruption robustness.

支撑证据：selective DyT Pareto 分析 (新实验) + CIFAR-100-C evaluation (部分已有)

### Contribution 2: Cross-architecture evidence with role taxonomy

> Across 10 architectures spanning 5 topology types, we show that DyT effectiveness is governed by the functional role of each normalization site — not by architecture topology alone. The role taxonomy, defined by computational graph position, provides non-circular classification.

支撑证据：Table 1 (已有) + RMSNorm baseline (新实验) + per-layer α correlation (零成本分析)

### Contribution 3: DyT as implicit regularizer

> DyT universally improves corruption robustness across all tested architectures (+2.73 to +6.79 pp), including RWKV-T where clean accuracy decreases. DyT's bounded activation range provides distribution-shift regularization independent of in-distribution performance.

支撑证据：CIFAR-100-C Tables 9-10 (已有数据) + selective DyT corruption eval (新实验)

### Contribution 4: α-deadlock mechanism and resolution boundary

> We identify a resolution-dependent failure mode: at 224×224, tanh saturation creates a self-reinforcing α-deadlock (sech²(αx) → 0 suppresses both signal and parameter gradients), establishing a capacity boundary that no α tuning can resolve.

支撑证据：Appendix K 数学推导 + Figure 7 causal chain + Table 11 实验 (全部已有)

### Contribution 5: Practical diagnostic toolkit

> We provide an early-α diagnostic: monitoring per-site learned α during the first 20 training epochs predicts replacement feasibility, enabling practitioners to apply DyT safely without exhaustive validation.

支撑证据：α trajectory 分析 (从现有 training log 提取) + 验证实验

---

## 4. 新标题建议

保留 topology 作为 organizing lens，加入 role 作为核心：

1. **When Can Dynamic Tanh Replace LayerNorm? A Role-Aware Analysis Across Vision Architectures** （推荐）
2. **Beyond Drop-in Replacement: Normalization Roles Determine Dynamic Tanh Effectiveness**
3. **Role-Aware Normalization Replacement: Why Dynamic Tanh Works for Some Sites but Not Others**

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

**GPU 容量：** 4 × RTX 5090 × 5 天 × ~20 有效小时/天 = ~400 GPU-hours ≈ **350-450 CIFAR-100 runs**

### 5.2 总实验量概览

| 实验组 | 优先级 | 新增 runs | 目的 |
|---|---|---|---|
| E1: RMSNorm baseline 全架构 | P0 | 21-24 | 消除 mean-centering 攻击 + role hypothesis 控制 |
| E2: RWKV-T selective replacement + Pareto | P0 | 12-15 | 核心方法贡献 |
| E3: Vim-T selective replacement + α | P0 | 12-15 | State-dynamics role 验证 |
| E4: DeiT-S/T α 补 seeds | P0 | 8-12 | Activation stabilizer role 完善 |
| E5: DyT-v2 variants at 224×224 | P0 | 6-12 | 突破 scale 天花板尝试 |
| E6: Vim ±MLP 补 seeds | P1 | 8 | MLP compensation multi-seed |
| E7: ConvNeXt-T selective replacement | P1 | 9-12 | Hierarchical role 验证 |
| E8: Swin-T selective replacement | P1 | 9-12 | Hierarchical role 验证 |
| E9: CaiT depth threshold ablation | P1 | 4-8 | Depth-width factor 验证 |
| E10: PlainMamba-T +MLP 补 seeds | P1 | 4 | MLP grafting multi-seed |
| E11: Selective DyT CIFAR-100-C eval | P1 | 0 (eval only) | Pareto clean+corruption 验证 |
| E12: Tiny-ImageNet selective + RMSNorm | P2 | 18-27 | 跨数据集验证 |
| E13: Resolution continuum (64, 128) | P2 | 4-8 | α-deadlock resolution curve |
| E14: 核心实验 5-seed 扩展 | P2 | 30-40 | 统计显著性 |
| A1: Per-layer α ↔ role correlation | P0 | 0 (分析) | 非循环 taxonomy 验证 |
| A2: Activation distribution comparison | P0 | 0 (分析) | Role taxonomy 可视化 |
| A3: Depth-width scaling formula | P1 | 0 (分析) | 理论深度 |
| A4: Early-α diagnostic validation | P1 | 0 (分析) | 实践工具 |
| A5: Zhu et al. recipe 对比研究 | P1 | 1-3 (可选) | 解释 224×224 矛盾 |

**总计新增训练：P0 ≈ 60-80 runs, P0+P1 ≈ 110-150 runs, P0+P1+P2 ≈ 160-225 runs**

**GPU 利用率：P0+P1+P2 ≈ 45-65%，留有充足的失败重跑和追加空间**

---

## 6. 实验详细设计

### E1: RMSNorm Baseline 全架构

**优先级：P0（最高）**

**目的：** 不仅是 baseline，更是 role hypothesis 的控制实验。RMSNorm 去掉了 LN 的 mean-centering 但保留了统计 normalization 的 rescaling 能力。如果 RMSNorm 在所有架构上都接近 LN，说明问题是 tanh 的 bounded output range 特有的。如果 RMSNorm 也显示 role-dependent pattern，说明 role dependency 是 normalization replacement 的一般性质。无论哪种结果都对论文有利。

**配置：**

| Architecture | Seeds | Runs |
|---|---|---|
| DeiT-S | 3 | 3 |
| DeiT-T | 3 | 3 |
| Vim-T | 3 | 3 |
| RWKV-T | 3 | 3 |
| PlainMamba-T | 3 | 3 |
| ConvNeXt-T | 3 | 3 |
| Swin-T | 3 | 3 |

**Total: 21 runs ≈ 1 GPU-day**

**分析框架：** 结果按 role 分组而非按 architecture 分组，检查 RMSNorm 是否也显示 role-dependent pattern。

### E2: RWKV-T Role-Aware Selective Replacement + Pareto Analysis

**优先级：P0（最高）**

**目的：** 把 RWKV-T 从 "DyT 在 hybrid 上失败" 升级为 role-aware replacement 的核心 showcase。关键目标是展示 **Pareto optimality**：selective DyT 的 clean accuracy ≈ LN 且 corruption robustness > LN。

**配置（CIFAR-100）：**

| 配置 | 替换比例 | 目的 | Seeds |
|---|---|---|---|
| Full LN | 0% DyT | Baseline | 已有 3 seeds |
| Full DyT α=0.5 | 100% DyT | Default full replacement | 已有 3 seeds |
| Full DyT α=1.0 | 100% DyT | Best α full replacement | 已有 (alpha-rwkv-a100) |
| DyT (excl. key\_norm) | ~80% DyT | Key\_norm preserved | 已有 3 seeds (Table 2) |
| **Selective DyT: only residual pre-norm** | ~50% DyT | Role-aware replacement | **新, 3 seeds** |
| **Selective DyT: residual pre-norm + α=0.25** | ~50% DyT | Role-aware + conservative α | **新, 3 seeds** |
| **RMSNorm full** | 0% DyT | Strong baseline | **新, 3 seeds (E1)** |

已有数据：Full LN (3 seeds), Full DyT α=0.5 (3 seeds), DyT excl. key\_norm (3 seeds), Full DyT α=1.0 (1 seed)
**新增 runs: 6-9**（selective DyT 两种配置 × 3 seeds，RMSNorm 已在 E1 中计）

**CIFAR-100-C evaluation（E11，零训练成本）：**
对所有 converged checkpoints 跑 CIFAR-100-C evaluation，构建 Pareto 曲线：
- x-axis: Clean accuracy
- y-axis: Mean corruption accuracy
- 每个点对应一种 normalization 配置

**预期结论：**

最佳情况：
> Selective DyT (only residual pre-norms replaced) matches full LN on clean accuracy while surpassing it on corruption robustness by +X pp, achieving Pareto optimality.

次佳情况：
> Selective DyT substantially recovers the full-DyT degradation and preserves corruption robustness gains, demonstrating that role-aware replacement is strictly superior to blanket replacement.

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
| **RMSNorm full** | Strong baseline | **E1 中已计** |

**新增 runs: 10-13**

**CIFAR-100-C evaluation：** 对 selective DyT checkpoints 跑 corruption eval，检查 robustness gain 是否保留。

### E4: DeiT-S/T α Robustness 补 Seeds

**优先级：P0**

**目的：** 确认 activation-stabilizer role 的 α stability window 比 state-dynamics role 更宽。

**配置：**

| Architecture | α | 当前 seeds | 补 seeds | 新增 runs |
|---|---|---|---|---|
| DeiT-S | 0.25 | 1 | +2 | 2 |
| DeiT-S | 1.0 | 1 (degraded) | +2 | 2 |
| DeiT-T | 0.25 | 0 | +3 | 3 |
| DeiT-T | 1.0 | 0 | +3 | 3 |

**新增 runs: 10**（RMSNorm 已在 E1 中计）

### E5: DyT-v2 Variants at 224×224（Scale 天花板突破尝试）

**优先级：P0（成本极低，upside 最大）**

**目的：** α-deadlock 的根本原因是 tanh 的 bounded output range [-1, 1]。测试简单 fix 能否在 224×224 上缩小 18 pp gap。即使失败也是有价值的负结果。

**Variants：**

| Variant | 公式 | 解决什么问题 |
|---|---|---|
| DyT-Res | γ·tanh(αx) + β + x | 加 residual bypass，unbounded output |
| DyT-Clamp | γ·tanh(αx) + β, α clamped ≤ 0.5 | 硬约束防止 α-deadlock |
| DyT-Warmup | α\_init=0.01, linear warmup 20 epochs → 0.5 | 避免 early saturation |

**配置：** ImageNet-100 224×224, DeiT-S, 1 seed each

| Config | LN baseline | 对比 |
|---|---|---|
| LN | 70.6% | — |
| DyT α=0.1 (best current) | 52.3% | Δ = -18.3 |
| DyT-Res α=0.1 | ? | 目标：缩小 gap |
| DyT-Res α=0.5 | ? | 目标：缩小 gap |
| DyT-Clamp α=0.5 | ? | 目标：避免 deadlock |
| DyT-Warmup α→0.5 | ? | 目标：避免 early saturation |

**新增 runs: 4-6**（每 run ~2-4 小时，总计 ~1 GPU-day）

**预期分析：**

如果任何 variant 显著改善（例如 >60%）：
> We identify the failure mode (α-deadlock) and show that a simple fix (e.g., residual bypass) partially addresses it, opening a path for DyT at higher resolutions.

如果全部失败：
> The capacity limitation persists across multiple mitigation strategies, confirming that the bounded tanh output range is a fundamental constraint of pointwise normalization substitution.

### E6: Vim-T ±MLP 补 Seeds

**优先级：P1**

**配置：**

| Config | 当前 seeds | 补 seeds | 新增 runs |
|---|---|---|---|
| Vim-T full (with MLP) + DyT | 1 | +2 | 2 |
| Vim-T full (with MLP) + LN | 1 | +2 | 2 |
| Vim-T -MLP + DyT | 1 | +2 | 2 |
| Vim-T -MLP + LN | 1 | +2 | 2 |

**新增 runs: 8**

### E7: ConvNeXt-T Selective Replacement

**优先级：P1**

**目的：** 验证 hierarchical feature-scale controller role。ConvNeXt 用 post-depthwise LN (LN2d)，不是 pre-norm residual pattern，但 selective replacement 仍可测试 "保留 downsample norms" 的效果。

**配置：**

| Config | 目的 | Seeds |
|---|---|---|
| Full LN | Baseline | 已有 3 seeds |
| Full DyT | Full replacement | 已有 3 seeds |
| **DyT within-stage only, preserve downsample norms** | Hierarchical role test | **新, 3 seeds** |
| **DyT shallow stages (stage 1-2) only** | Depth-dependent test | **新, 3 seeds** |
| **RMSNorm** | Baseline | **E1 中已计** |

**新增 runs: 6**

### E8: Swin-T Selective Replacement

**优先级：P1**

**配置：**

| Config | 目的 | Seeds |
|---|---|---|
| Full LN | Baseline | 已有 3 seeds |
| Full DyT | Full replacement | 已有 3 seeds |
| **Preserve patch-merging norms, DyT within-block** | Stage transition role test | **新, 3 seeds** |
| **DyT shallow stages only** | Depth-dependent test | **新, 3 seeds** |
| **RMSNorm** | Baseline | **E1 中已计** |

**新增 runs: 6**

注意：Swin-T 使用了非标准超参数（lr=5×10⁻⁴, 5-epoch warmup, gradient clipping=5.0），保持一致。

### E9: CaiT Depth Threshold Ablation

**优先级：P1**

**目的：** 直接验证 depth-width factor。DeiT-T (12L/192) 成功 (+3.91) 而 CaiT-XXS24 (24+2L/192) 失败 (-12.34)。中间有没有阈值？

**配置：** 对 CaiT-XXS24，只在前 k 层 SA blocks 应用 DyT，保留其余为 LN：

| k (DyT layers) | 总 SA layers | 目的 | Seeds |
|---|---|---|---|
| 6 | 24 | 浅层 only | 1 |
| 12 | 24 | 半数 | 1 |
| 18 | 24 | 深层也替换 | 1 |
| 24 (full SA) | 24 | 全替换（≈已有 SA-only ablation） | 已有 1 seed |

**新增 runs: 3-4**（如果结果 pattern 清晰，不需多 seed）

**预期结论：** k=12 或以下时 DyT 仍 positive → depth threshold 在 12-18 层之间。配合 DeiT-T (12L positive) 的数据点，画出连续的 depth vs. DyT Δ 曲线。

### E10: PlainMamba-T +MLP 补 Seeds

**优先级：P1**

**新增 runs: 4**（+MLP DyT/LN × 2 extra seeds）

### E11: Selective DyT CIFAR-100-C Evaluation

**优先级：P1**

**目的：** 验证 Pareto optimality——selective DyT 是否保留 DyT 的 corruption robustness gain。

**方法：** 对 E2, E3, E7, E8 的 selective DyT checkpoints 跑 CIFAR-100-C 15 corruption types × severity 1-5。

**零训练成本，仅需 evaluation 脚本。**

**预期结论：** 如果 selective DyT 在 RWKV-T 上 clean ≈ LN 且 corruption > LN，这是论文最有说服力的正向结果。

### E12: Tiny-ImageNet Selective + RMSNorm

**优先级：P2**

**配置：** DeiT-S, Vim-T, RWKV-T 的 LN / RMSNorm / Full DyT / Selective DyT × 3 seeds = ~36 runs

**训练时间：** Tiny-ImageNet (64→32) ~2× CIFAR-100，总计 ~2 GPU-days

### E13: Resolution Continuum Curve

**优先级：P2**

**目的：** 在 DeiT-S 上跑 64×64 和 128×128，连接 32×32 (+6.90) 和 224×224 (-18.3) 两个端点，画出连续的 resolution vs. DyT Δ 曲线。

**配置：** DeiT-S, LN/DyT, 64×64 和 128×128, 1 seed each = 4 runs

**配合 α-deadlock 的 αx₀ ∝ p² 分析，精确定位 breakpoint resolution。**

### E14: 核心实验 5-Seed 扩展

**优先级：P2**

对 P0 实验中最关键的配置从 3 seeds 扩展到 5 seeds，支撑 paired t-test 统计检验：
- RWKV-T selective DyT: +2 seeds = 2 runs
- Vim-T selective DyT: +2 seeds = 2 runs
- DeiT-S RMSNorm: +2 seeds = 2 runs
- DeiT-S DyT α=0.5: +2 seeds = 2 runs
- 对应 LN baselines: +2 seeds each where needed

**新增 ~20-30 runs**

---

## 7. 零成本分析任务

### A1: Per-Layer Learned α ↔ Role Taxonomy Correlation

**优先级：P0**

**目的：** 为 role taxonomy 提供非循环证据。如果不同 role 的 norm sites 学到不同的 α 分布，taxonomy 就有独立于 DyT 结果的验证。

**方法：**
1. 从已有 converged checkpoints 提取所有 norm sites 的 learned α（论文 Figure 5 已有 seed=42 数据）
2. 按计算图位置分类（pre-attn / pre-FFN / state-adjacent / key\_norm）
3. 跨架构做 α vs. position-type box plot
4. 检验不同 role 的 α 分布是否有显著差异

**预期结果：**
- Activation stabilizer sites（pre-attn, pre-FFN）：低 α（near-linear regime）
- State-dynamics sites：中等 α
- Architectural invariant sites（key\_norm）：高 α（试图 enforce constraint）
- 深层比浅层 α 更高（已有 Figure 5 支持）

**成本：零新训练。Agent 几小时完成分析 + 图表。**

### A2: Activation Distribution Comparison Figure

**优先级：P0**

**目的：** 最直观的 role taxonomy 可视化证据。

**方法：** 从 converged checkpoints 提取不同 norm sites 的 pre-activation 和 post-normalization 分布，对比 LN vs. DyT。

**设计 3 个 panel：**

Panel 1: DeiT-S residual pre-attn norm
- LN output distribution vs. DyT output distribution → 预期：非常相似

Panel 2: RWKV-T key\_norm site
- LN output distribution vs. DyT output distribution → 预期：明显不同（DyT 被 tanh 压缩到 [-1,1]，key magnitude 结构被破坏）

Panel 3: ConvNeXt-T downsample norm site
- LN output vs. DyT output → 预期：scale mismatch

**成本：零新训练。Agent 几小时。**

### A3: Depth-Width Scaling Formula

**优先级：P1**

**目的：** 拟合简单公式预测 DyT Δ，从 post-hoc 升级为 predictive。

**数据点（open-loop architectures with pre-norm pattern）：**

| Arch | L | d | L×(1/d) | Δ |
|---|---|---|---|---|
| DeiT-S | 12 | 384 | 0.031 | +6.90 |
| DeiT-T | 12 | 192 | 0.063 | +3.91 |
| CaiT-XXS24 | 26 | 192 | 0.135 | -12.34 |

加上 E9 的 CaiT depth ablation 数据点，尝试拟合：
```
Δ_predicted ≈ a - b × (L/d)
```

**即使 R² 只有 0.7，能预测 Δ 符号就有价值。**

**配合 closed-loop 数据（Vim-T L=10/d=384 Δ=+3.45, PlainMamba-T L=12/d=384 Δ=-0.41），检验公式是否需要 topology 修正项。**

### A4: Early-α Diagnostic Validation

**优先级：P1**

**目的：** 从训练前 20 epochs 的 per-site α trajectory 预测最终 DyT 成功/失败。

**方法：**
1. 修改训练代码，每 epoch 记录所有 DyT sites 的 learned α 值
2. 对已有实验（或 rerun 前 20 epochs）提取 α trajectory
3. 定义 diagnostic metric: max α across sites at epoch 20, 或 α growth rate
4. 检验：成功架构 (DeiT) 的 early-α 是否 < 阈值，失败架构 (CaiT, Swin) 的 early-α 是否 > 阈值

**如果阈值存在，提出实用规则：**
> Monitor per-site α after 20 epochs. If any site's α > τ, preserve that site as LN.

**成本：** 需要修改代码记录 α trajectory（agent 几分钟），可能需要对部分实验 rerun 前 20 epochs（每 run ~5-15 min）。

### A5: Zhu et al. 2025 Recipe 对比

**优先级：P1**

**目的：** 审稿人 100% 会问为什么 Zhu et al. 在 ImageNet-1K 224×224 上 DeiT 报告 positive，本文 ImageNet-100 224×224 报告 -18 pp。

**方法：**
1. 做一张 setup comparison table (Zhu et al. vs. ours)
2. 如果 Zhu et al. 开源了代码/config，尝试用他们的 recipe 在 ImageNet-100 上复现
3. 识别关键差异（pre-trained init? longer training? different α? per-channel γ,β?）

**可能的解释：**
- Zhu et al. 使用了 pre-trained weights（DyT 在 fine-tune 时 ok，from scratch 时 fail）
- Zhu et al. 训练 300 epochs vs. 本文 100 epochs（更长训练可能让 α 有时间逃离 deadlock）
- Zhu et al. 可能使用了不同的 α initialization 或 per-channel scaling

**成本：** 1-3 runs (如果尝试 recipe 复现) + 文献研究。

---

## 8. 5 天执行计划

### Day 0: 代码改造 + 启动第一批实验

**Agent 并行任务（coding）：**

| Agent | 任务 | 产出 |
|---|---|---|
| Agent-Code-1 | 实现 RMSNorm 替换 | RMSNorm class + config 开关 |
| Agent-Code-2 | 实现 selective replacement（按 norm site name/position 指定 preserve list） | 每个架构的 preserve config |
| Agent-Code-3 | 实现 DyT-v2 variants（DyT-Res, DyT-Clamp, DyT-Warmup） | 3 个新 DyT class |
| Agent-Code-4 | 实现 per-epoch α logging + CIFAR-100-C evaluation pipeline | α trajectory logger + corruption eval script |

**Day 0 晚间启动 GPU（代码完成后立即开始）：**

| GPU | 实验 | Runs | 预计时间 |
|---|---|---|---|
| GPU 1 | E1: RMSNorm - DeiT-S(3), DeiT-T(3), RWKV-T(3) | 9 | ~5h |
| GPU 2 | E1: RMSNorm - Vim-T(3), PlainMamba-T(3) | 6 | ~6h |
| GPU 3 | E1: RMSNorm - ConvNeXt-T(3), Swin-T(3) | 6 | ~8h |
| GPU 4 | E5: DyT-v2 on ImageNet-100 224×224 (4-6 configs × 1 seed) | 4-6 | ~8-16h |

### Day 1: 核心 Selective Replacement 实验

**GPU 任务：**

| GPU | 实验 | Runs | 预计时间 |
|---|---|---|---|
| GPU 1 | E2: RWKV-T selective (2 configs × 3 seeds) + α=1.0 补 2 seeds | 8 | ~4h |
| GPU 2 | E3: Vim-T selective (2 configs × 3 seeds) + α补seeds | 10-13 | ~12h |
| GPU 3 | E4: DeiT-S/T α补seeds (10 runs) | 10 | ~6h, 然后 E6: Vim ±MLP (8 runs) ~8h |
| GPU 4 | E9: CaiT depth threshold (3-4 runs) + E10: PlainMamba +MLP (4 runs) | 7-8 | ~6h, 然后 E7: ConvNeXt selective (6 runs) ~8h |

**Agent 并行任务（分析）：**

| Agent | 任务 | 产出 |
|---|---|---|
| Agent-Analysis-1 | A1: Per-layer α ↔ role correlation（从现有 checkpoints） | α vs. role-type box plot |
| Agent-Analysis-2 | A2: Activation distribution comparison（从现有 checkpoints） | 3-panel distribution figure |
| Agent-Analysis-3 | A5: Zhu et al. recipe 对比研究 | Setup comparison table |
| Agent-Writing-1 | 草拟 role taxonomy section + Algorithm 1 | LaTeX draft |

### Day 2: 补充实验 + 大规模分析

**GPU 任务：**

| GPU | 实验 | 预计时间 |
|---|---|---|
| GPU 1 | E8: Swin-T selective (6 runs) | ~8h |
| GPU 2 | E14: 核心 5-seed 扩展 (10-15 runs) | ~12h |
| GPU 3 | E12: Tiny-ImageNet selective + RMSNorm (18-27 runs，优先 DeiT-S + RWKV-T) | ~全天 |
| GPU 4 | E13: Resolution continuum (4 runs) + 重跑失败实验 | ~8h |

**Agent 并行任务：**

| Agent | 任务 | 产出 |
|---|---|---|
| Agent-Analysis-1 | E11: CIFAR-100-C eval 对所有 selective DyT checkpoints | Corruption accuracy tables |
| Agent-Analysis-2 | 构建 Pareto 曲线（clean vs. corruption，每个点是一种 config） | Pareto figure |
| Agent-Analysis-3 | A3: Depth-width scaling formula 拟合 | Regression plot |
| Agent-Analysis-4 | A4: Early-α diagnostic（从 Day 1 的 α logs 提取） | α trajectory plot + diagnostic rule |
| Agent-Analysis-5 | E5 结果分析：DyT-v2 在 224×224 效果评估 | DyT-v2 analysis |
| Agent-Writing-1 | 重写 Introduction + Method | LaTeX draft |
| Agent-Writing-2 | 重写 Results: cross-architecture outcomes | LaTeX draft |

### Day 3: 论文重写日

**所有实验应已完成。** GPU 用于重跑失败实验和 P2 补充。

**Agent 并行任务：**

| Agent | 任务 | 产出 |
|---|---|---|
| Agent-Writing-1 | 重写 Abstract | LaTeX |
| Agent-Writing-2 | 重写 Results: role-aware selective replacement + Pareto analysis | LaTeX |
| Agent-Writing-3 | 重写 Results: α-stability + α-deadlock mechanism (从 Appendix K 移入) | LaTeX |
| Agent-Writing-4 | 重写 Results: corruption robustness as implicit regularization | LaTeX |
| Agent-Writing-5 | 重写 Discussion + Limitations | LaTeX |
| Agent-Writing-6 | 整理所有新 figures + tables | LaTeX |
| Agent-Analysis-1 | Claim-evidence audit: 逐条检查 claim 是否有 ≥3 seeds 支持 | Audit report |

### Day 4: 压力测试 + 最终打磨

**Agent 并行任务：**

| Agent | 任务 |
|---|---|
| Agent-Review-1 | 模拟 NeurIPS Reviewer 1 (novelty focus): "Where's the new method?" |
| Agent-Review-2 | 模拟 NeurIPS Reviewer 2 (soundness focus): "Post-hoc? Single-seed? Scale?" |
| Agent-Review-3 | 模拟 NeurIPS Reviewer 3 (significance focus): "Who benefits? Practical impact?" |
| Agent-Fix-1 | 根据 review 修改论文中的弱点 |
| Agent-Fix-2 | 最终数字校对（表格数字与实验 log 一致性） |
| Agent-Fix-3 | References 检查 + NeurIPS checklist 更新 |
| Agent-Fix-4 | 重写 Conclusion，确保 claim 与 evidence 精确匹配 |

**Reviewer Attack List + 准备好的回应：**

| 攻击 | 准备的回应 |
|---|---|
| "Role taxonomy is post-hoc" | Per-layer α correlation (A1) + activation distribution (A2) 提供非循环证据；taxonomy 基于计算图位置，不依赖 DyT 结果 |
| "Selective replacement is too simple" | Algorithm 1 正式化 + Pareto optimality 展示它不只是简单保留，而是 optimal trade-off |
| "32×32 scale is too small" | α-deadlock mechanism 提供 resolution-dependent 解释；DyT-v2 探索 fix；32×32 是 controlled testbed (precedent: ViT paper) |
| "Where's RMSNorm?" | 全架构 RMSNorm baseline + 按 role 分组分析 |
| "How is this different from Zhu et al.?" | 显式 recipe comparison + cross-topology extension + role-aware method |
| "α tuning resolves all failures?" | α-sensitivity heatmap (Figure 8) 显示 Swin/CaiT 在所有 α 下仍 ≥6.6 pp below LN |
| "Single-seed ablations?" | Key ablations 补到 3-5 seeds + paired statistical tests |
| "Corruption robustness is just a side effect?" | All 4 architectures improve; RWKV-T improves corruption DESPITE clean accuracy drop; selective DyT preserves this gain |

---

## 9. 论文结构设计

### 主文结构

```
1. Introduction
   - DyT 背景 + "normalization is not one thing" 洞察
   - 核心发现预览：role-aware selective > full LN > full DyT (on Pareto frontier)
   - Contribution list (5 点)

2. Related Work
   - Normalization in deep learning
   - Normalization-free networks (Fixup, ReZero, NFNet, DyT, Chen et al.)
   - Architecture-dependent normalization

3. Method
   3.1 Architecture Taxonomy (保留 topology 作为 organizing lens)
   3.2 Normalization Functional Role Taxonomy (4 类，基于计算图位置)
   3.3 DyT Replacement Protocol (universal + selective)
   3.4 Algorithm 1: Role-Aware Replacement Protocol

4. Experimental Setup
   - Datasets, training recipe, evaluation protocol
   - RMSNorm baseline description

5. Results
   5.1 Cross-Architecture DyT Effectiveness (Table 1 + Figure 1)
       - 按 role 组织，不按 model 流水账
   5.2 Role-Aware Selective Replacement (Table 2 + Pareto figure)
       - Selective DyT Pareto optimality: clean ≈ LN + corruption > LN
   5.3 RMSNorm as Control Experiment
       - RMSNorm 是否也显示 role-dependent pattern
   5.4 Corruption Robustness as Implicit Regularization (Table 3)
       - DyT universally improves corruption robustness
   5.5 α-Stability Landscape (Figure + heatmap)
       - Topology-dependent stability windows
       - Early-α diagnostic

6. Discussion
   6.1 α-Deadlock Mechanism and Resolution Boundary
       - 数学推导 (Eq. 2-3)
       - Causal chain (Figure)
       - Resolution-dependent capacity ceiling
       - DyT-v2 exploration results
   6.2 Depth-Width Interaction
       - Scaling formula
       - CaiT depth threshold
   6.3 Compensatory Pathways: MLP Channel
   6.4 Validation of Role Taxonomy
       - Per-layer α correlation
       - Activation distribution comparison

7. Limitations

8. Conclusion
```

### 主文 Figure/Table 设计

**Table 1: Cross-Architecture Results**

| Architecture | Topology | Role | LN | RMSNorm | Full DyT | Selective DyT | Δ Full | Δ Selective |
|---|---|---|---|---|---|---|---|---|
| DeiT-S | Open-loop | Act. stab. | 58.0 | ? | 64.9 | 64.9* | +6.90 | +6.90* |
| DeiT-T | Open-loop | Act. stab. | 56.9 | ? | 60.8 | — | +3.91 | — |
| Vim-T | Closed-loop | State-dyn. | 61.1 | ? | 64.5 | ? | +3.45 | ? |
| PlainMamba-T | Closed-loop | State-dyn. | 74.2 | ? | 73.8 | — | -0.41 | — |
| RWKV-T | Hybrid | Arch. inv. | 74.6 | ? | 72.5 | ? | -2.17 | ? |
| ConvNeXt-T | Open (CNN) | Hier. scale | 55.1 | ? | 43.0 | ? | -12.11 | ? |
| Swin-T | Open (win.) | Hier. scale | 64.0 | ? | 46.8 | ? | -17.18 | ? |

*DeiT-S 全部 norm 都是 activation stabilizer，所以 selective = full DyT

**Figure 1: Pareto Analysis (Hero Figure)**

x-axis: Clean accuracy
y-axis: Mean corruption accuracy
每个点 = (architecture, normalization config)
颜色 = architecture
形状 = config (LN=circle, RMSNorm=diamond, Full DyT=cross, Selective DyT=star)

**如果 selective DyT 的 star 点在 LN circle 和 Full DyT cross 的右上方（clean ≈ LN, corruption > LN），这张图一张就能讲完整个 story。**

**Figure 2: Role Taxonomy + Decision Flowchart**

左半：4 类 role 的示意图（标注在不同架构的计算图上）
右半：Algorithm 1 的 decision flowchart

**Figure 3: α-Stability Heatmap**

已有 Figure 8 数据，加入 selective DyT 的 stability window 对比。

**Figure 4: α-Deadlock Mechanism**

已有 Figure 7 的 causal chain，加入 DyT-v2 的 resolution curve（如果做了 E13）。

**Figure 5: Mechanistic Evidence for Role Taxonomy**

3 panel: (a) per-layer α vs. role-type, (b) activation distribution comparison, (c) effective rank

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

---

## 10. 写作层面的具体修改

### 10.1 Abstract 结构

```
1. Background: DyT replaces LN with learnable bounded nonlinearity
2. Gap: Whether DyT transfers beyond standard self-attention is unknown
3. Key insight: Same LayerNorm formula serves different functional roles
4. Core finding: Role-aware selective replacement achieves Pareto optimality
   (clean ≈ LN + corruption > LN)
5. Mechanistic contribution: α-deadlock at high resolution
6. Practical contribution: Early-α diagnostic for safe deployment
```

不堆数字。只在最关键处给 1-2 个数字（e.g., "selective DyT matches LN clean accuracy while improving corruption robustness by +X pp across Y architectures"）。

### 10.2 Introduction 核心段落

> A normalization module is not defined solely by its formula. In modern vision architectures, the same `nn.LayerNorm` call can stabilize residual activations (DeiT), regulate recurrent state dynamics (Vim), enforce key-magnitude invariants (RWKV), or align feature scales across resolution stages (Swin, ConvNeXt). A pointwise bounded nonlinearity like DyT can substitute the first role but not the others. This observation motivates **role-aware selective replacement**: applying DyT only where normalization serves as an activation stabilizer, and preserving LayerNorm at architecture-critical sites.

> Surprisingly, this selective strategy is not merely a compromise — it achieves **Pareto-optimal** performance: matching full LayerNorm on clean accuracy while surpassing it on corruption robustness. The bounded activation range of DyT at appropriate sites provides implicit distribution-shift regularization without disrupting architecture-critical normalization functions.

### 10.3 Results 组织原则

**按 role 和 finding 组织，不按 model 流水账：**

1. **Cross-architecture landscape** —— 10 architectures, 3 outcome groups, role explains all
2. **Selective replacement as Pareto optimum** —— The positive headline result
3. **RMSNorm control** —— Is role-dependency DyT-specific or general?
4. **Corruption robustness** —— The surprising finding
5. **α-stability** —— Topology-dependent stability windows + early-α diagnostic

### 10.4 Discussion 要点

不要写 "topology predicts DyT effectiveness"。改成：

> Topology provides a useful organizing lens for understanding normalization diversity, but the replacement outcome is determined by the functional role of each normalization site and its interaction with depth, width, α, and resolution. The role taxonomy — defined by computational graph position — offers a principled basis for selective replacement decisions.

### 10.5 Limitations（诚实但不自毁）

1. Role taxonomy is an empirical framework, not a formal theory
2. Selective replacement protocol tested on 4 architectures with selective configs; generalization to untested architectures requires the early-α diagnostic
3. Primary experiments at 32×32; the α-deadlock analysis provides resolution-dependent mechanistic explanation rather than large-scale empirical validation
4. Statistical power limited by 3-5 seeds; we report paired effect sizes and note where results are marginal
5. DyT-v2 exploration is preliminary; a complete solution to the resolution boundary remains open

但强调：

> These limitations do not undermine the core finding: blanket LayerNorm-to-DyT replacement is unsafe, while role-aware selective replacement consistently achieves a better clean-robustness trade-off than either extreme.

---

## 11. 完成标准

### 最小可接受标准（P0 全部完成）

1. RMSNorm baseline 7 架构 × 3 seeds
2. RWKV-T selective replacement 2 configs × 3 seeds + CIFAR-100-C eval
3. Vim-T selective replacement 2 configs × 3 seeds
4. DeiT-S/T α 补 seeds
5. DyT-v2 at 224×224 至少 3 variants × 1 seed
6. Per-layer α ↔ role correlation 分析
7. Activation distribution comparison figure
8. Algorithm 1 box + decision flowchart
9. α-deadlock 从 Appendix K 移入主文 Discussion
10. Corruption robustness 从 Appendix C 移入主文 Results
11. 主文重写完成（role-aware narrative + Pareto framing）
12. Pareto 曲线 (clean vs. corruption) 作为 hero figure

**预估中稿概率：35-42%**

### 理想完成标准（P0+P1 全部完成）

在最小标准基础上，额外：

1. ConvNeXt-T selective replacement 3 seeds
2. Swin-T selective replacement 3 seeds
3. CaiT depth threshold ablation
4. Vim ±MLP 补到 3 seeds
5. PlainMamba-T +MLP 补到 3 seeds
6. Depth-width scaling formula
7. Early-α diagnostic validation + 阈值 τ 确定
8. Zhu et al. recipe 对比
9. CIFAR-100-C eval 覆盖所有 selective DyT configs
10. 统计检验（paired t-test / Wilcoxon on 5-seed data）
11. 模拟 3 reviewer 压力测试并修复

**预估中稿概率：38-45%，如果 DyT-v2 在 224×224 有改善则 42-50%**

### 全面完成标准（P0+P1+P2）

额外：

1. Tiny-ImageNet selective + RMSNorm 跨数据集验证
2. Resolution continuum curve (32, 64, 128, 224)
3. 核心实验 5-seed 扩展
4. ViT-B/ViT-Tiny 额外架构验证

**预估中稿概率：40-48%，如果 DyT-v2 work 则 45-52%**

---

## 12. 关键风险和 Contingency

### 风险 1：Selective DyT 的 corruption robustness 不显著

如果 selective DyT 在 RWKV-T 上 corruption ≈ LN（没有 gain），Pareto 论点弱化。

**Contingency：** 改为强调 "selective DyT recovers clean accuracy while maintaining full DyT's corruption benefit partially"——仍然比 full DyT 好。

### 风险 2：RMSNorm 在所有架构上都接近 LN

如果 RMSNorm 完全 role-independent（所有架构都 ≈ LN），说明 mean-centering 不重要，但 role taxonomy 失去 generality（只对 DyT 有效）。

**Contingency：** 改为 "the role-dependent failure is specific to bounded pointwise alternatives; statistical normalization (LN, RMSNorm) is uniformly robust because it preserves distributional information." 这仍然是有价值的 finding。

### 风险 3：DyT-v2 全部失败

**Contingency：** 放 appendix 作为 negative result，强调 "the bounded output range is a fundamental constraint; unbounded alternatives (RMSNorm) work because they preserve scale information." 这强化了 role taxonomy 的解释力。

### 风险 4：Per-layer α 与 role 不相关

如果不同 role sites 的 learned α 没有显著差异。

**Contingency：** 不放这个分析。Role taxonomy 仍然由 selective replacement intervention 结果支撑（这是更强的因果证据）。

### 风险 5：ConvNeXt/Swin selective replacement 无效

ConvNeXt 的 LN2d (post-depthwise) 不是 pre-norm pattern，selective 可能不产生有意义的结果。

**Contingency：** 如果失败，强调 "ConvNeXt's normalization serves a qualitatively different role (post-depthwise scale management) that is structurally incompatible with pointwise replacement at any site." 这仍然支持 role taxonomy。

---

## 13. 不做清单

以下明确不在 5 天范围内：

| 不做 | 原因 |
|---|---|
| ImageNet-1K 完整训练 | 2-4 卡 5 天内无法完成多模型多 seed |
| COCO / ADE20K downstream | 成本高、recipe confound 大 |
| 10-20 个新模型 | 当前 10 个模型足够，问题是证据组织 |
| NLP / LM 扩展 | 不同领域、不同 recipe、5 天风险太大 |
| Fused DyT CUDA kernel | 效率不是主线 |
| 10-seed 全矩阵 | 5 seeds 足够支撑统计检验 |
| Neural collapse 分析 | 有趣但与 role taxonomy 主线距离较远 |
| Adversarial robustness | Corruption robustness 已经足够支撑正则化发现 |

---

## 14. 预期 Story Arc（最终版本）

**Opening hook：**
> Dynamic Tanh has been proposed as a normalization-free alternative to LayerNorm, but we find that blindly replacing all LayerNorm sites degrades or collapses most architectures. The key insight is that "LayerNorm" is not one thing — the same operator serves four distinct functional roles across modern vision architectures.

**Core method：**
> We propose role-aware selective replacement (Algorithm 1): replace activation-stabilizer norms with DyT, preserve architecture-critical norms as LayerNorm. This is not a compromise but a Pareto-optimal strategy — matching full LN on clean accuracy while surpassing it on corruption robustness.

**Surprising finding：**
> DyT universally improves corruption robustness across all tested architectures, even where it hurts clean accuracy. Role-aware selective replacement preserves this regularization benefit while recovering clean accuracy.

**Mechanistic depth：**
> We identify a resolution-dependent failure mode: at 224×224, tanh saturation creates a self-reinforcing α-deadlock, establishing a fundamental capacity boundary for pointwise normalization alternatives.

**Practical tool：**
> We provide a simple early-α diagnostic: monitoring per-site α growth during the first 20 training epochs predicts whether DyT replacement will succeed, enabling safe deployment without exhaustive architecture-specific validation.

---

## 15. 总结

本方案的核心不是"多跑实验"，而是 **三重升级**：

1. **从 diagnostic 到 prescriptive** —— Selective DyT 不是止损而是 Pareto 最优
2. **从 phenomenon 到 mechanism** —— α-deadlock 数学分析 + role taxonomy 非循环验证
3. **从 observation 到 tool** —— Algorithm 1 + early-α diagnostic 可供实践者直接使用

**GPU 利用率：** P0+P1+P2 ≈ 160-225 runs / 350-450 capacity ≈ 45-65%。剩余留作失败重跑。

**Agent 利用率：** Day 0-1 以 coding + 实验启动为主，Day 2-4 以分析 + 写作为主，每天 4-6 agents 并行。

**预估 NeurIPS 中稿概率提升：从 15-20% → 35-50%**（取决于 Pareto 结果和 DyT-v2 结果）。
