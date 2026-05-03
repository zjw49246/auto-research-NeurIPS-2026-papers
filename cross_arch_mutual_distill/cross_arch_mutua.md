# NeurIPS 2026 投稿前论文优化方案（基于修订版 PDF，v4 饱和执行版）

**论文题目**：*Symmetric Loss, Asymmetric Outcome: A Diagnostic Study of Cross-Architecture Mutual Distillation*
**目标会议**：NeurIPS 2026
**基于版本**：`cross_arch_mutual_distill_v1_revised_20260503.pdf`（27 页，含 Appendix A–P + NeurIPS Checklist）
**更新日期**：2026-05-03

---

**硬件配置**：4 × RTX 5090 (32 GB)
**执行模式**：Agent Auto Research（全自动实验调度 + 结果分析 + 条件分支决策）
**计划时长**：4 天（~96 小时墙钟时间）
**计算预算**：4 GPU × 96 h = **384 GPU-hours**（论文原始实验总计 ~150 GPU-hours on RTX PRO 6000）

---

## 第一部分：当前论文内容与已完成实验全清单

### 1.1 核心问题与定位

论文研究 **CNN 与 ViT 之间的跨架构 mutual distillation**，核心发现是：

> 形式上对称的 mutual KL distillation loss，在 CNN↔ViT 共训时产生**不对称结果**：ViT 稳定获益，CNN 侧取决于架构组合、容量差距和 augmentation。

定位是 **diagnostic empirical study**，不是提出新 KD 方法。

### 1.2 已完成实验全清单（主文 + Appendix）

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
| Same-arch controls (w/ Mixup) | Table 1 | 3 | |Δ| = 0.21±0.19 (CNN), 0.25±0.09 (ViT) |

> †主文 Tables 4–5 是 single-seed (s42)，但 3-seed matched interaction terms 已在 Appendix G 完成并被主文引用。

#### Appendix 实验

| Appendix | 内容 | Seeds | 状态 |
|---|---|---|---|
| **A** | Temperature details (Table 7) | 3 | ✅ |
| **B** | CKA-Performance (Table 8 + Figure 2) | s42 | ✅ |
| **C** | Training Dynamics (Figures 3–4, Table 9) | s42（CKA 3s） | ✅ |
| **D** | DML 2×2 Factorial (Table 10) | **3** | ✅ |
| **E** | Baseline Comparison (Table 11) | 混合 | ✅ |
| **F** | MobNet Case Study (Table 12 + Figure 6) | 5/3 | ✅ |
| **G** | 3-Seed Factorial Interaction | **3** | ✅ |
| **H** | Hyperparameters (Table 13) | — | ✅ |
| **I** | α sweep (Tables 14–16, Figure 7) | 混合 | ✅ |
| **J** | Tiny-ImageNet Factorial (Table 17–18, Figure 8) | ⚠️ **s42** | single-seed |
| **K** | TCKD/NCKD Decomposition | s42 (R18/DS 3s) | ✅ |
| **L** | Mixup Mechanism: Gradient/Entropy (Table 19) | ⚠️ **s42** | **修订版新增**，single-seed |
| **M** | Same-Arch NoMixup Controls (Table 20) | 3 | ✅ |
| **N** | R18/DeiT-S Results (Table 21, Figure 9) | **5** | ✅ |
| **O** | Dataset Details | — | ✅ |
| **P** | Optimizer Ablation (Table 22) | s42 | ✅ |

---

## 第二部分：Review 总结

### 2.1 总体评价

> **Borderline+ (5.5/10)**。修订版在 Discussion 深度和 Appendix 利用上有显著改善，但核心实验缺口仍在。

### 2.2 当前仍存在的核心问题（按严重程度排列）

| 严重度 | 问题 | 说明 |
|---|---|---|
| **致命** | Capacity sweep 因果验证缺失 | 3 个 pair 同时改变 CNN family/ViT size/baseline，论文自己承认了但未解决 |
| **高** | 实验尺度偏小 | CIFAR-100/CUB-200/TI-200 + 小模型，NeurIPS reviewer 期望至少 ImageNet-scale |
| **高** | Appendix L 机制分析 under-powered | Discussion 核心定量 claim 依赖 single-seed single-pair |
| **中高** | 缺少现代架构验证 | 只有 ResNet + DeiT，reviewer 质疑 "only works on old architectures" |
| **中** | TI-200 Factorial single-seed | Figure 8 最大交互效应无 error bars |
| **中** | Abstract teaching tax over-sell | 三个 pair 中只有一个 CNN 受损，但 Abstract 以此为核心 |
| **中** | 缺少实践指导 | "So what" 问题未充分回答 |
| **中** | 格式偏长 | 正文超 9 页 |
| **中低** | Divergence family 对照温度不同 | reverse KL τ=4 vs forward KL τ=1 |
| **中低** | Unidirectional 仅单方向 single-seed | 逻辑可用措辞修正 |

---

## 第三部分：4 天饱和执行方案

### 设计原则

1. **前两天解决"致命"和"高"级问题**，确保论文不会因单一原因被 reject
2. **第三天解决"中高"和"中"级问题**，消除常见 weakness 点
3. **第四天整合分析 + 写作修改**，将实验结果转化为论文提升
4. **每天结束有 Agent 分析检查点**，基于中间结果调整后续优先级
5. **每个实验标注依赖关系**，Agent 可按 DAG 自动调度

### 运行时间估算基准

基于论文 Appendix H 的训练配置和 RTX 5090 性能：

| 数据集 | 单 run 时长 | 备注 |
|---|---|---|
| CIFAR-100 (200 epochs, batch 128) | ~50 min | R32/DeiT-T pair |
| CUB-200 (200 epochs, batch 64) | ~1.2 h | R18/DeiT-S pair |
| TI-200 (200 epochs, batch 64) | ~1.2 h | R18/DeiT-S pair |
| ImageNet-100 (200 epochs, batch 256) | ~3.5 h | R50/DeiT-S pair（估算） |
| Forward pass (分析) | ~10–20 min | 已有 checkpoint |

---

### Day 1：因果验证 + 机制扩展（目标：解决致命级和高级问题）

**总计：~92 GPU-hours across 4 GPUs**

#### GPU-A：Capacity Sweep — NoMixup 条件（~23h）

**目的**：堵住论文最大因果漏洞。在同一 ResNet family 内，仅改变深度/容量，控制 architecture family。

```
固定：DeiT-Tiny (5.4M), CIFAR-100, NoMixup, τ=4, α=1.0
每组：independent baseline + mutual cross = 2 runs/seed

ResNet-20  (~0.27M): indep × 3 seeds + mutual × 3 seeds = 6 runs (~5h)
ResNet-44  (~0.66M): indep × 3 seeds + mutual × 3 seeds = 6 runs (~5h)
ResNet-56  (~0.86M): indep × 3 seeds + mutual × 3 seeds = 6 runs (~5h)
ResNet-110 (~1.73M): indep × 3 seeds + mutual × 3 seeds = 6 runs (~5h)
ResNet-32  已有（复用 Table 1 数据）

尾部时间（~3h）：R20 补 s789, s1011 (4 runs) → 最极端 capacity gap 点提前升 5-seed
```

**产出**：Figure 9 从 3 个 confounded 点 → 6 个同 family controlled 点（R20/R32/R44/R56/R110 + MobNet 对照），R20 达 5-seed
**依赖**：无，Day 1 即可启动

#### GPU-B：Capacity Sweep — Mixup 条件（~23h）

**目的**：为 capacity sweep 的新 pair 生成 2×2 factorial 数据。与 GPU-A 的 NoMixup 条件配对，可对每个新 ResNet 变体计算 Mixup×KD interaction。

```
固定：DeiT-Tiny, CIFAR-100, Mixup (α_mix=0.8, α_cut=1.0), τ=4, α=1.0

ResNet-20:  indep × 3 seeds + mutual × 3 seeds = 6 runs (~5h)
ResNet-44:  indep × 3 seeds + mutual × 3 seeds = 6 runs (~5h)
ResNet-56:  indep × 3 seeds + mutual × 3 seeds = 6 runs (~5h)
ResNet-110: indep × 3 seeds + mutual × 3 seeds = 6 runs (~5h)

尾部时间（~3h）：R110 补 s789, s1011 Mixup (4 runs) → 最大 capacity 点提前升 5-seed
```

**产出**：每个新 ResNet 变体的完整 2×2 factorial（3-seed），R110 达 5-seed Mixup。验证 Mixup confound 是否在所有 capacity regime 下都存在
**依赖**：无，与 GPU-A 并行启动

#### GPU-C：TI-200 Factorial 补 seed + Divergence Ablation + Unidirectional（~23h）

**目的**：消除所有 single-seed 和 missing-control 的 weakness 点。

```
Block 1: TI-200 factorial 补 seed (~10h)
  s123 × 4 cells (indep/mutual × Mixup/NoMixup) = 4 runs × 1.2h = ~5h
  s456 × 4 cells = 4 runs × 1.2h = ~5h

Block 2: Divergence family ablation (~5h)
  CIFAR-100, R32/DeiT-T, NoMixup, τ=4
  Forward KL: s42, s123, s456 = 3 runs × ~50min = ~2.5h
  JSD (0.5 forward + 0.5 reverse): s42, s123, s456 = 3 runs = ~2.5h

Block 3: Unidirectional 补全 (~5h)
  CIFAR-100, R32/DeiT-T, NoMixup
  CNN→ViT (α_A=0): 补 s123, s456 = 2 runs = ~1.5h
  ViT→CNN (α_B=0): s42, s123, s456 = 3 runs = ~2.5h
  Forward KL τ=4 + Unidirectional CNN→ViT: s42 = ~1h（验证 forward KL 下的方向控制）

Block 4: CUB-200 factorial 补 seed (~3h)
  R18/DeiT-S CUB-200, s123 × 2 cells (indep/mutual NoMixup) = 2 runs × 1.2h = ~2.4h
  → Tables 4-5 single-seed 问题进一步缓解
```

**产出**：
- TI-200 factorial 升级为 3-seed，Figure 8 加上 error bars
- 三种 divergence (reverse KL / forward KL / JSD) 在同温度下的 asymmetry 对比
- 双向 unidirectional 的 3-seed 对比
- CUB-200 factorial 额外 seed 验证
**依赖**：无

#### GPU-D：Appendix L 机制分析扩展 + ImageNet-100 启动（~23h）

**目的**：(1) 将 Discussion 的机制证据从 single-seed 升级；(2) 启动 ImageNet-100 实验。

```
Block 1: Appendix L 扩展 (~3h，纯 forward pass)
  3 pairs × 3 seeds × 2 conditions (Mixup/NoMixup) = 18 个 checkpoint
  每个 checkpoint 计算：
    - Gradient cosine similarity (CE vs KD gradients)
    - KD output entropy
    - Layer-wise CKA (stem → final block)
    - Top-1 confidence of soft targets
    - ECE (Expected Calibration Error)

Block 2: ImageNet-100 数据准备 + 首批实验 (~20h)
  数据准备：从 ImageNet-1K 抽取 100 类子集 (~30 min)
  R50 independent NoMixup: s42 (~3.5h), s123 (~3.5h)
  DeiT-S independent NoMixup: s42 (~3.5h), s123 (~3.5h)
  R50/DeiT-S mutual NoMixup: s42 (~3.5h), s123 (~3.5h)
```

**产出**：
- Table 19 从 1×1 扩展为 3 pairs × 3 seeds 矩阵，含 mean±std
- ImageNet-100 的 2 个 baseline + 2 个 mutual 初步结果
**依赖**：Appendix L 扩展无依赖；ImageNet-100 需要数据准备

#### Day 1 Agent 检查点（~第 20 小时）

当 capacity sweep 前 3 个 ResNet 变体跑完后（GPU-A/B 各完成 ~15h），Agent 执行：

1. **分析 capacity sweep 中间结果**：
   - 画 ResNet-20/32/44/56/110 的 CNN Δ vs capacity ratio 曲线
   - 判断趋势是单调还是非单调
   - 如果强烈非单调 → 考虑 Day 2 补一个中间点
   - 如果趋势清晰 → Day 2 资源投向 ImageNet-100 和架构扩展

2. **分析 Appendix L 扩展结果**：
   - 检查 gradient cosine 变化在跨 pair 上是否一致
   - 如果一致 → Discussion mechanism story 可以加强
   - 如果不一致 → 标记为 "regime-dependent mechanism"，仍有价值但需调整叙事

3. **决定 Day 2 优先级排序**

---

### Day 2：Scale 验证 + 架构扩展（目标：解决实验尺度问题）

**总计：~92 GPU-hours across 4 GPUs**

Day 2 的具体分配取决于 Day 1 检查点的结果。以下是默认计划（Day 1 结果无重大意外时）：

#### GPU-A：ImageNet-100 续跑（~24h）

```
R50/DeiT-S mutual NoMixup: s456 (~3.5h)
R50 independent Mixup: s42 (~3.5h)
DeiT-S independent Mixup: s42 (~3.5h)
R50/DeiT-S mutual Mixup: s42 (~3.5h)
R50 independent NoMixup: s456 (~3.5h)
DeiT-S independent NoMixup: s456 (~3.5h)
R50/DeiT-S mutual Mixup: s123 (~3.5h)
```

**产出**：ImageNet-100 上的 3-seed NoMixup 完整 + Mixup 条件 2 seeds，含 baseline + mutual + Δ

#### GPU-B：现代 CNN 架构 — ConvNeXt-Tiny + Swin-Tiny（~23h）

**目的**：排除 "asymmetry is a ResNet/DeiT artifact" 的质疑。ConvNeXt-Tiny (~28.6M) 和 Swin-Tiny (~28.3M) 分别代表现代 CNN 和现代 ViT。

```
Block 1: ConvNeXt-T/DeiT-S CIFAR-100 (~15h)
  τ=4:
  ConvNeXt-T independent NoMixup: s42, s123, s456 (~2.5h)
  DeiT-S independent NoMixup: 复用已有 R18/DS 实验中的 DeiT-S baselines
  ConvNeXt-T/DeiT-S mutual NoMixup: s42, s123, s456 (~2.5h)
  ConvNeXt-T independent Mixup: s42, s123, s456 (~2.5h)
  ConvNeXt-T/DeiT-S mutual Mixup: s42, s123, s456 (~2.5h)
  ConvNeXt-T/DeiT-S 同构控制 (CNN↔CNN): s42, s123 (~2h)

Block 2: Swin-Tiny 基线 + 对照 (~8h)
  Swin-T independent NoMixup: s42, s123, s456 (~2.5h)
  Swin-T/ResNet-32 mutual NoMixup: s42, s123, s456 (~2.5h)
  → ViT↔CNN 反向 capacity 配置（Swin-T 远大于 R32）
  Swin-T independent Mixup: s42 (~1h)
```

**产出**：
- ConvNeXt-T/DeiT-S 的完整 2×2 factorial（3-seed）+ 同构控制
- Swin-Tiny/R32 对照（现代 ViT 作为大容量端）
- 直接回答 "does the finding generalize beyond ResNet/DeiT?"

#### GPU-C：Capacity Sweep α 验证 + 5-seed 升级（~22h）

**目的**：(1) 在 capacity sweep 新 pair 上测试 α_A=0.25 是否有效；(2) 关键实验升级到 5-seed。

```
Block 1: Capacity sweep α=0.25 验证 (~10h)
  R20/DeiT-T mutual, α_A=0.25, NoMixup: s42, s123, s456 (~2.5h)
  R56/DeiT-T mutual, α_A=0.25, NoMixup: s42, s123, s456 (~2.5h)
  R110/DeiT-T mutual, α_A=0.25, NoMixup: s42, s123, s456 (~2.5h)
  R44/DeiT-T mutual, α_A=0.25, NoMixup: s42, s123, s456 (~2.5h)
  → 对比 Day 1 的 α=1.0 结果，看 Pareto improvement 是否随 capacity gap 系统性变化

Block 2: Capacity sweep 升级到 5 seeds (~6h)
  选择 Day 1 结果中趋势最关键的 2 个新 ResNet 变体，补 s789, s1011：
  R56 indep+mutual NoMixup: 4 runs (~3h)
  Rxxx indep+mutual NoMixup: 4 runs (~3h)

Block 3: EfficientNet-B0/DeiT-T 对照 (~6h)
  CIFAR-100, NoMixup:
  EfficientNet-B0 (~5.3M) indep: s42, s123, s456 (~2.5h)
  EfficientNet-B0/DeiT-T mutual: s42, s123, s456 (~2.5h)
  → 第二个 non-ResNet CNN family（与 MobNet 不同的高效架构），进一步分离 family effect
```

**产出**：
- α_A=0.25 在 4 个新 pair 上的效果 → 验证 practitioner checklist 的泛化性
- 关键 capacity sweep 点从 3-seed 升级到 5-seed → 更强的统计检验
- EfficientNet 数据点 → Figure 9 增加第三个 CNN family

#### GPU-D：Soft Target Quality 系统性分析 + 新 pair 的 Appendix L 扩展（~23h）

**目的**：把 Appendix L 从 "suggestive single-pair evidence" 升级为 "systematic cross-pair analysis"。

```
Block 1: Day 1 capacity sweep checkpoints 的 forward-pass 分析 (~4h)
  新增 4 个 ResNet 变体 × 3 seeds × 2 conditions = 24 checkpoints
  每个计算：gradient cosine, entropy, CKA, ECE, confidence
  → 产出：capacity vs mechanism metric 的连续曲线

Block 2: Per-class 分析 (~4h，所有已有 checkpoints)
  在 CIFAR-100 上计算 per-class soft target quality：
  - 哪些类 CNN soft target 优于 ViT？哪些类反转？
  - CNN-side 下降是否集中在特定类别？
  - → 如果发现 CNN 在 ViT-dominant classes 上下降更多，支撑 "directional absorption" 论述

Block 3: Training dynamics 提取 (~6h)
  如果 Day 1 训练时保存了 per-epoch checkpoints（建议每 10 epochs 保存），提取：
  - Per-epoch Δ 曲线（如 Figure 4）for capacity sweep 4 个新 pair
  - CKA dip timing vs capacity ratio 的关系
  → 验证 "transient representational reorganization" 是否也在新 pair 上出现

Block 4: ImageNet-100 首批结果的 forward-pass 分析 (~3h)
  对 Day 1-2 完成的 ImageNet-100 checkpoints 做 gradient/entropy 分析
  → 验证 mechanism 在 larger scale 上是否保持

Block 5: CUB-200 ConvNeXt 验证 (~4h)
  ConvNeXt-T/DeiT-S mutual NoMixup on CUB-200: s42, s123, s456 (~4h)
  → 跨数据集验证现代架构结果一致性

Block 6: 统计检验自动化 (~2h)
  对所有完成的实验，自动运行：
  - Sign test (directional consistency)
  - Paired t-test (effect size with CI)
  - Fisher's combined p-value (cross-pair)
  → 产出：complete statistical summary table
```

**产出**：
- 完整的 mechanism metric vs capacity ratio 关系曲线
- Per-class 分析（新 insight，文献中未见系统性研究）
- ImageNet-100 的初步 mechanism 验证
- CUB-200 上的 ConvNeXt 交叉验证
- 自动化统计检验结果

#### Day 2 Agent 检查点（~第 44 小时）

1. **Capacity sweep 完整结果分析**：
   - 6 个同 family 点（R20/R32/R44/R56/R110 + R20 5-seed）的趋势确认
   - 与 MobNet/EfficientNet 对照点对比
   - 决定 Figure 9 的最终呈现方式

2. **ImageNet-100 中间结果**：
   - 如果 NoMixup 3-seed 已完成，检查 asymmetry 方向
   - 如果方向一致 → Day 3 继续 Mixup 条件和 α sweep
   - 如果方向不同 → 标记为重要发现，调整 claim

3. **ConvNeXt + Swin 结果分析**：
   - ConvNeXt asymmetry 方向和幅度
   - Swin-T/R32 的"反向 capacity"结果 → 是否印证 capacity 假说
   - 如果 asymmetry 存在 → 强有力的 generality 证据
   - 如果不存在 → 需要在 Discussion 中加 nuance

4. **决定 Day 3 资源分配**

---

### Day 3：深化 + 整合（目标：Novel 贡献 + 消除中级问题）

**总计：~84 GPU-hours across 4 GPUs**

#### GPU-A：ImageNet-100 扩展（~24h）

```
条件分支：基于 Day 2 ImageNet-100 结果

If asymmetry 方向一致（高概率）:
  ImageNet-100 factorial Mixup 条件补全:
    R50 indep Mixup: s123, s456 (~7h)
    DeiT-S indep Mixup: s123, s456 (~7h)
    R50/DeiT-S mutual Mixup: s456 (~3.5h)
  ImageNet-100 α=0.25 验证:
    R50/DeiT-S mutual NoMixup α_A=0.25: s42, s123 (~7h)
  → 产出：ImageNet-100 完整 2×2 factorial 3-seed + α=0.25 对照

If asymmetry 方向不一致:
  补充 seeds 确认异常:
    R50/DeiT-S mutual NoMixup: s789, s1011 (~7h)
  + 不同 ImageNet-100 子集验证 (~14h):
    重新采样 100 类，重跑 baseline + mutual key conditions
```

#### GPU-B：α Weighting 预测性验证（~20h）

**目的**：将 α_A≈0.25 从 "pair-specific observation" 升级为 "predictable prescription"。这是把论文从 diagnostic 推向 prescriptive 的关键。

```
设计思路：
  用 capacity sweep 结果拟合一个 simple heuristic:
    α_A_optimal ≈ f(capacity_ratio) 或 f(baseline_gap)

  在 "held-out" pair 上验证预测：
  1. ConvNeXt-T/DeiT-S (Day 2 已有 α=1.0 结果)
     → 用 heuristic 预测 optimal α → 跑 3 seeds 验证 (~3h)
  2. EfficientNet-B0/DeiT-T (Day 2 已有 α=1.0 结果)
     → 同上 (~2.5h)
  3. ImageNet-100 R50/DeiT-S (Day 2 已有 α=1.0 结果)
     → 用 heuristic 预测 → 跑 2 seeds 验证 (~7h)
  4. Day 1 capacity sweep 中的极端 pair (R20/DeiT-T, ratio ~20:1)
     → α_A=0.25 已有(Day 2)，补测 α_A=0.10 和 α_A=0.50 (~5h)
  5. Swin-T/R32 α_A=0.25 对照 (~2.5h)
     → "反向 capacity" 配置下的 α 效果
```

**产出**：
- Heuristic function α*(gap) 或 α*(ratio) 的拟合与验证
- "Predict-then-validate" 实验设计（NeurIPS reviewer 非常认可这种 methodology）
- 如果预测成功 → C3 贡献从 "pair-specific remedy" 升级为 "principled prescription"

#### GPU-C：Pretrained Fine-tuning 对照 + Swin-T 扩展（~22h）

**目的**：排除 "asymmetry only happens with from-scratch weak ViT" 的质疑。论文当前所有 ViT 都是 from-scratch，baseline 较弱。Reviewer 可能认为如果 ViT 预训练过，asymmetry 会消失。

```
Block 1: Pretrained Fine-tuning (~16h)
  CUB-200, ImageNet-pretrained:
  R18 pretrained indep: s42, s123, s456 (~4h)
  DeiT-S pretrained indep: s42, s123, s456 (~4h)
  R18/DeiT-S pretrained mutual NoMixup: s42, s123, s456 (~4h)
  R18/DeiT-S pretrained mutual Mixup: s42, s123, s456 (~4h)

Block 2: Swin-T 扩展 (~6h)
  Swin-T/R32 mutual Mixup: s42, s123, s456 (~2.5h)
  Swin-T/R32 同构控制 (ViT↔ViT): s42, s123 (~2h)
  Pretrained α sweep (α_A ∈ {0.25, 0.50}): s42 only (~1.5h)
```

**产出**：
- Pretrained 条件下 asymmetry 方向和幅度
- Swin-T/R32 完整 2×2 factorial + 同构控制
- 如果 pretrained 下 asymmetry 减弱 → "The asymmetry tracks the baseline competence gap"
- 如果 asymmetry 仍存在 → "The finding generalizes to pretrained fine-tuning regimes"

#### GPU-D：Novel 分析 + 图表生成自动化（~18h）

```
Block 1: Gradient conflict 细粒度分析 (~5h)
  对 capacity sweep 所有 pair 的 checkpoints:
  - 逐 epoch 计算 gradient cosine (每 20 epochs 一个采样点)
  - 与 per-epoch accuracy Δ 对照
  - → 如果 gradient conflict 的 timing 与 CNN dip (epoch 70-100) 吻合，直接支撑因果链

Block 2: Capacity-aware Mixup interaction 曲线 (~3h analysis)
  用 Day 1 GPU-A + GPU-B 的数据，画：
  - x = capacity ratio, y = Mixup×KD interaction term
  - 看 Mixup confound 是否随 capacity gap 系统性变化
  - → 如果是，这是全新的定量发现

Block 3: Knowledge transfer efficiency 分析 (~2h)
  计算每个 pair 的 "transfer efficiency" = Δ_student / teacher_baseline
  - 画 transfer efficiency vs capacity ratio 曲线
  - 与 CKA 曲线对照，看是否有统一的 capacity-dependent 解释

Block 4: 自动化图表生成 (~4h)
  用 matplotlib/seaborn 自动生成所有新增 figure：
  - Figure 9 扩展版（capacity sweep + family markers + error bars）
  - Appendix L 扩展版（3×3+ mechanism metrics 矩阵）
  - ImageNet-100 主结果表
  - ConvNeXt + EfficientNet + Swin 结果表
  - α prediction vs validation 图
  - Cross-dataset factorial 汇总图（Figure 8 更新版，全部有 error bars）

Block 5: 自动化统计检验 (~2h)
  对所有新增实验运行完整统计检验套件：
  - 每个 pair: sign test + paired t-test + Cohen's d + 95% CI
  - 跨 pair: Fisher's combined test
  - Factorial: interaction F-test（如果 3+ seeds）
  - 产出：LaTeX-ready 统计摘要表

Block 6: 论文修改建议草稿 (~2h)
  基于所有实验结果，自动生成：
  - 新增 Appendix 的 LaTeX 草稿骨架
  - 主文修改建议（标注具体行号 + 新旧对照）
```

#### Day 3 Agent 检查点（~第 68 小时）

1. **所有核心实验应已完成**，检查是否有 failed/anomalous runs
2. **ImageNet-100 结果分析**：方向、幅度、统计显著性
3. **ConvNeXt + EfficientNet + Swin 结果分析**：generality 的最终判断
4. **α prediction 验证结果**：heuristic 是否泛化
5. **Pretrained 结果分析**：baseline competence vs inductive bias 的判断
6. **决定 Day 4 的重点**：哪些 seed 需要补跑、哪些图表需要修正

---

### Day 4：收尾 + 写作（目标：将实验转化为论文提升）

**总计：~56 GPU-hours 实验 + Agent 写作时间**

#### GPU-A/B：缺失 seed 补跑 + 关键实验加固（~22h each, 共 ~44h）

```
基于 Day 3 检查点确定的补跑列表（优先级排序）：

Priority 1: 任何 failed/anomalous runs 的重跑（预估 ~8h）
Priority 2: ImageNet-100 关键条件升级到 3-seed（如 Mixup 条件补 s456）(~7h)
Priority 3: ConvNeXt/Swin 关键发现的 additional seed 验证（3→5 seeds for key pair）(~6h)
Priority 4: α prediction 验证中偏差最大的 pair，补 additional α 值 (~5h)
Priority 5: Pretrained fine-tuning α=0.25 验证，3 seeds (~4h)
Priority 6: Capacity sweep 全部点升 5-seed（补跑剩余 s789, s1011）(~10h)
Priority 7: 如果以上完成仍有剩余时间 → 探索性实验（如 DeiT-Base 配对，300 epoch 训练）(~4h)
```

#### GPU-C：Sanity Checks + 鲁棒性验证（~14h）

```
Block 1: 超参鲁棒性 (~6h)
  Capacity sweep 中选 2 个关键点，用不同 learning rate (0.5× 和 2× default) 重跑：
  R20/DeiT-T mutual NoMixup, lr=0.05: s42 (~50min)
  R20/DeiT-T mutual NoMixup, lr=0.2: s42 (~50min)
  R110/DeiT-T mutual NoMixup, lr=0.05: s42 (~50min)
  R110/DeiT-T mutual NoMixup, lr=0.2: s42 (~50min)
  ConvNeXt-T/DeiT-S mutual NoMixup, lr=0.05: s42 (~50min)
  ConvNeXt-T/DeiT-S mutual NoMixup, lr=0.2: s42 (~50min)
  → 确认结果不依赖于特定超参选择

Block 2: ImageNet-100 子集鲁棒性 (~4h)
  用不同的 100 类子集（random seed 2）重跑：
  R50/DeiT-S mutual NoMixup: s42 (~3.5h)
  → 确认不依赖于类别选择

Block 3: Training length 鲁棒性 (~4h)
  R32/DeiT-T mutual NoMixup CIFAR-100, 300 epochs: s42, s123 (~1.7h × 2)
  → 确认 200 epoch 结果不是 early stopping artifact
```

#### GPU-D：最终分析 + 图表 + 写作（~8h compute + Agent 写作）

```
Block 1: 最终版所有图表生成 (~3h)
  - 整合 Day 1-4 所有结果
  - 生成 publication-quality figures (PDF/SVG format)
  - 确保所有图表风格一致（统一配色、字体大小、error bar 格式）

Block 2: 完整统计检验报告 (~2h)
  - 汇总所有实验的统计检验结果
  - 生成 LaTeX-ready 表格
  - 标注统计显著性水平 (*p<0.05, **p<0.01, ***p<0.001)

Block 3: 论文修改完整建议文档 (~3h)
  基于所有实验结果，自动生成：
  1. 更新后的 Abstract（具体文本，teaching tax → regime-specific）
  2. 新增 Section/Appendix 的 LaTeX 草稿
  3. 修改后的 Contribution 表述（C1/C2/C3 根据实际结果调整）
  4. 更新后的 Limitations（删除已修复的 limitation）
  5. 更新后的 Practitioner Checklist（4 步决策流程）
  6. 需要删除或修改的现有文本（标注行号 + 新旧对照）
  7. 格式压缩建议（正文回到 ≤9 页）
  8. 更新后的 Conclusion（加入 scale + arch generality 新证据）
```

---

### 完整实验清单汇总

| 实验组 | Runs | GPU-hours | Day | 对中稿率影响 |
|---|---|---|---|---|
| **Capacity sweep NoMixup** (R20/R44/R56/R110, 3s + R20 升 5s) | ~28 | ~23h | 1 | 最高：因果验证 |
| **Capacity sweep Mixup** (同上 + R110 升 5s) | ~28 | ~23h | 1 | 高：新 pair factorial |
| **TI-200 factorial 补 seed** | 8 | ~10h | 1 | 中：去除 single-seed 标签 |
| **Divergence ablation** (FwdKL + JSD, τ=4) | 6 | ~5h | 1 | 中：排除 KL artifact |
| **Unidirectional 补全** (双向 + 多 seed) | 5 | ~4h | 1 | 中低 |
| **CUB-200 factorial 补 seed** | 2 | ~3h | 1 | 中：交叉验证 |
| **Appendix L 扩展** (forward pass) | 0 | ~3h | 1 | 高：机制证据 |
| **ImageNet-100** (R50/DeiT-S, factorial + α) | ~24 | ~80h | 1–3 | 高：scale concern |
| **ConvNeXt-T/DeiT-S** (CIFAR-100 factorial + CUB-200) | ~16 | ~19h | 2 | 中高：modern arch |
| **Swin-Tiny/R32** (CIFAR-100, factorial + 控制) | ~10 | ~11h | 2–3 | 中高：modern ViT |
| **EfficientNet-B0/DeiT-T** (CIFAR-100) | ~6 | ~5h | 2 | 中：family 对照 |
| **Capacity sweep α=0.25** (4 个新 pair) | ~12 | ~10h | 2 | 中高：α 泛化 |
| **Capacity sweep 5-seed 升级** | ~8 | ~6h | 2 | 中：统计力度 |
| **α prediction validation** (held-out pairs) | ~12 | ~20h | 3 | 高：prescriptive 升级 |
| **Pretrained fine-tuning** (CUB-200) | ~12 | ~16h | 3 | 中高：排除 from-scratch 解释 |
| **Gradient dynamics 细粒度分析** | 0 | ~5h | 3 | 中：novel insight |
| **Per-class + Transfer efficiency 分析** | 0 | ~6h | 2–3 | 中：novel insight |
| **补跑 + 关键实验加固** | ~20 | ~44h | 4 | 中高：结果可信度 |
| **Sanity checks + 鲁棒性** | ~8 | ~14h | 4 | 中：reviewer 信任度 |
| **自动化分析 + 图表 + 写作** | 0 | ~16h | 2–4 | — |
| **总计** | **~205** | **~323h** | | |

利用率：323 / 384 ≈ **84%**（剩余 ~61 GPU-hours 作为 buffer，用于 failed runs 和条件分支中的额外实验）。

> **注**：84% 利用率在 Agent Auto Research 模式下已接近上限——需要预留 ~16% buffer 用于 failed run 重试、条件分支决策后的追加实验、以及 Agent 分析/调度的等待时间。如果 Day 1-2 进展顺利无异常，Day 4 的 buffer 可进一步用于 additional architecture pairs 或 longer training 实验。

---

### 写法修改清单（与实验并行，Day 1–4 持续执行）

以下写法修改不依赖新实验，可在 Day 1 实验启动后立即开始：

| 修改 | 描述 | 对中稿率影响 |
|---|---|---|
| **格式压回 9 页** | Optimizer confound → Appendix P，压缩 Related Work（1.2→0.8 页）和 Limitations | 高 |
| **Abstract 调整** | Teaching tax 从 headline 降为 regime-specific example，regime-dependence 为核心 | 中高 |
| **Practitioner checklist** | Discussion 末尾加 4 步决策流程 | 中高 |
| **Appendix L 标注** | Single-seed limitation 明确标注，"confirms" → "suggests" | 中高 |
| **Unidirectional 解释修正** | 标注 training variance 范围，重新定义 ablation 价值 | 中 |

实验完成后的写法更新（Day 3-4）：

| 修改 | 描述 |
|---|---|
| **新增 Appendix Q** | Capacity Sweep 完整结果（Figure 9 扩展版 + 表格） |
| **新增 Appendix R** | ImageNet-100 Validation Results |
| **新增 Appendix S** | Modern Architecture (ConvNeXt / Swin-T / EfficientNet) Results |
| **新增 Appendix T** | Pretrained Fine-tuning Results |
| **更新 Figure 9** | 从 3 个 confounded 点 → 6+ 个 controlled 点 + family markers |
| **更新 Figure 8** | 所有数据集加 error bars |
| **更新 Table 19** | 从 1×1 → 3+ pairs × 3 seeds 矩阵 |
| **更新 Discussion** | 基于新实验结果修正 mechanism story |
| **更新 Limitations** | 删除已修复的 limitation（capacity confound），更新尺度相关表述 |
| **更新 Contribution** | 根据实际结果调整 C1/C2/C3 表述 |
| **更新 Conclusion** | 加入 scale validation 和 architecture generality 的新证据 |

---

### 条件分支决策树

Agent Auto Research 模式下，以下决策点需要根据中间结果自动判断：

```
Day 1 检查点:
├── Capacity sweep 趋势单调？
│   ├── 是 → Day 2 升级关键点到 5-seed，确认趋势
│   └── 否/非单调 → 检查是否有 outlier seed，补跑确认，考虑增加中间点
│
├── Appendix L 跨 pair 一致？
│   ├── 是 → Discussion 可用 "confirmed across pairs" 措辞
│   └── 否 → 改为 "regime-dependent mechanism"，仍有价值
│
└── ImageNet-100 Day 1 首批结果
    ├── Asymmetry 方向一致 → 继续按计划 Day 2 补跑
    └── 方向不同 → 优先补 seed 确认，推迟 Mixup 条件

Day 2 检查点:
├── ConvNeXt 结果
│   ├── Asymmetry 存在 → 强 generality claim
│   ├── Asymmetry 不存在 → 重要 nuance，检查是否因 capacity ratio ~1:1
│   └── 方向反转 → 非常有趣的发现，需要额外分析
│
├── Swin-T/R32 结果
│   ├── R32 侧获益 → 与主发现一致（小容量端获益），支持 capacity 假说
│   └── R32 侧受损 → 需要区分 capacity effect 和 architecture effect
│
├── α prediction heuristic 拟合质量
│   ├── R² > 0.7 → Day 3 验证可以继续
│   └── R² < 0.5 → 改为定性分析，不做定量预测
│
└── EfficientNet 结果
    └── 与 MobNet 趋势一致？→ 支持 "family effect" 论述

Day 3 检查点:
├── α prediction 验证结果
│   ├── 3/5 held-out pairs 预测成功 → C3 升级为 prescriptive
│   └── ≤1/5 成功 → 保持 pair-specific 表述
│
├── Pretrained fine-tuning
│   ├── Asymmetry 减弱 → 支持 "baseline gap drives asymmetry"
│   └── Asymmetry 不变 → 支持 "architectural inductive bias drives asymmetry"
│
└── 所有实验完成度检查
    ├── >90% 完成 → Day 4 focus on writing + sanity checks
    └── <90% 完成 → Day 4 priority on completing core experiments
```

---

### 预估评分提升

| 阶段 | 预估评分 | 关键变化 |
|---|---|---|
| 修订版当前 | 5.5/10 | Discussion 改善，但核心缺口仍在 |
| Day 1 完成 | ~6.5/10 | Capacity sweep 堵住因果漏洞，Appendix L 扩展，single-seed 问题消除 |
| Day 2 完成 | ~7/10 | ImageNet-100 scale validation，ConvNeXt + Swin modern arch |
| Day 3 完成 | ~7–7.5/10 | α prediction validation，pretrained 对照，systematic mechanism analysis |
| Day 4 完成 | ~7.5/10 | 完整写作整合，所有图表更新，统计检验完备，鲁棒性验证 |

**相比 v3 方案的提升**：v4 方案通过增加 Swin-Tiny 验证、CUB-200 交叉验证、更全面的 sanity checks、以及更充分的 Day 4 补跑计划，将 GPU 利用率从 63% 提升到 84%，总实验量从 ~242h 增加到 ~323h。新增的 ~81 GPU-hours 主要投向：架构覆盖（Swin +11h）、鲁棒性检验（+14h）、Day 4 加固补跑（+14h）、跨数据集验证（+7h）和更深入的分析（+6h）。

**诚实说明**：7.5 分接近 diagnostic study 在 NeurIPS 的得分天花板。要冲击 8+ 分，需要提出一个基于诊断发现的 **新方法**（如 adaptive α scheduling across training epochs, or architecture-aware distillation loss），这超出了当前论文 scope 但可以作为 future work 指向下一篇论文。

---

## 参考依据

- NeurIPS 2026 Main Track Handbook
- NeurIPS Paper Checklist Guidelines
- 论文 PDF：`cross_arch_mutual_distill_v1_revised_20260503.pdf`（27 页，Appendix A–P）
