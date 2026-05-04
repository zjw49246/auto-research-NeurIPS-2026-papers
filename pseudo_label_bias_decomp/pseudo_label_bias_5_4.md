# NeurIPS 2026 投稿内部评审与优化方案（v3，经 Appendix 精确审计 + 饱和执行修订版）

**论文标题**：*Decomposing Pseudo-Label Bias: How Class Imbalance Amplifies Domain–Class Interaction in Semi-Supervised Learning*  
**目标会议**：NeurIPS 2026 Main Track  
**截稿日期**：2026 年 6 月 6 日  
**当前判断**：idea 有价值，当前约 5/10 → 目标 7.5–8/10；定位为"诊断型/分析型论文 + 可操作改进"，通过补强实验、统计建模、baseline 和叙事来最大化论文分数。  
**可用资源**：2–4 张 RTX 5090（32 GB）+ RTX PRO 6000（96 GB），24/7 Agent Auto Research 模式运行，GPU 饱和执行 3–4 天。

---

## 第一部分：当前论文内容、叙事与已完成实验

### 1.1 当前论文试图解决的问题

这篇论文关注半监督学习（Semi-Supervised Learning, SSL）中的 pseudo-label bias。文章的出发点是：在真实应用中，未标注数据往往同时存在两类偏差：

1. **类别不均衡**：某些类别样本很多，某些类别样本很少，形成 long-tail 分布。
2. **域偏移**：未标注数据来自不同视觉域，例如 Art、Clipart、Product、Real World 等。

已有 SSL 或 imbalanced SSL 工作通常主要处理类别边际分布上的偏差；domain adaptation 工作通常主要处理域偏移。但本文认为，两者并不是独立作用的。类别不均衡可能会放大某些"类别 × 域"组合上的 pseudo-label 错误，而这种结构化错误无法通过单独的 class-only 或 domain-only 分析看出来。

因此，论文的核心问题可以概括为：

> 在半监督学习中，当 class imbalance 和 domain shift 同时存在时，pseudo-label error 是否存在可量化的 class × domain interaction？如果存在，这种 interaction 会如何随 imbalance ratio、模型架构和 normalization 类型变化？

### 1.2 当前论文的主要方法

论文提出了一个基于 two-way ANOVA 的 pseudo-label bias decomposition 框架，用来把 pseudo-label precision 的方差分解为三个部分：

- **Class / class-group main effect**：不同类别或不同频率组之间的 pseudo-label precision 差异。
- **Domain main effect**：不同 domain 之间的 pseudo-label precision 差异。
- **Class × domain interaction effect**：某些类别在某些 domain 中特别容易或特别不容易出错，且这种偏差不能被单独的 class effect 或 domain effect 解释。

论文使用 `eta-squared` η² 作为 effect size，衡量每个因素解释了总方差的比例。

当前设计有两个 resolution：

| 分析粒度 | 设计 | 目的 |
|---|---|---|
| Cell-level | 3 个频率组 × 3 个 source domains | 可解释性强，便于展示 head/mid/tail 与 domain 的宏观结构 |
| Per-class | 65 个类别 × 3 个 source domains | 统计功效更强，可以发现 cell-level 聚合后被隐藏的 interaction |

这个设计是论文最有价值的部分之一：它不是提出一个新的 SSL 算法，而是提出一个诊断框架，用来解释 pseudo-label 偏差的结构来源。

### 1.3 当前论文的叙事主线

当前论文的叙事可以拆成四步：

#### Step 1：现有方法忽略 class 与 domain 的交互

文章首先指出，class-imbalanced SSL 主要看类别边际偏差，domain adaptation 主要看域边际偏差，二者通常没有定量分析 class × domain interaction。因此，现有方法可能无法判断某个 pseudo-label 错误到底来自 class imbalance、domain shift，还是二者的交互。

#### Step 2：提出 ANOVA decomposition 作为诊断工具

论文将 pseudo-label precision 作为响应变量，将 class group / class 与 domain 作为两个因素，做 two-way ANOVA decomposition。通过 η² 轨迹，论文试图观察不同 imbalance ratio 下，class effect、domain effect 和 interaction effect 如何变化。

#### Step 3：在 FixMatch + OfficeHome 上发现非单调 interaction 轨迹

在 ResNet-18 上，cell-level interaction 呈现 inverted-U：

- ρ = 1：interaction 已经存在，但较弱。
- ρ = 10：interaction 被放大，达到峰值。
- ρ = 20：interaction 开始下降。
- ρ = 50：interaction 基本消失或不显著。

论文将其解释为：中等 class imbalance 会放大 class × domain interaction；极端 imbalance 下，class effect 过强，domain-dependent structure 被压制或掩盖。

#### Step 4：进一步提出 architecture-dependent response modes

论文比较了三种模型：

| 模型 | Normalization | Macro-architecture | 当前论文给出的模式 |
|---|---|---|---|
| ResNet-18 | BatchNorm | CNN | early HEAD amplification，峰值在 ρ=10 |
| ConvNeXt-Tiny | LayerNorm | CNN | delayed HEAD amplification，峰值在 ρ=20 |
| ViT-S/16 | LayerNorm | Attention | HEAD-stable，没有明显 inverted-U |

由此，论文提出一个更强的解释：

- normalization type 与 interaction amplification 的时间位置有关；
- macro-architecture 与 amplification 是否出现有关；
- hierarchical CNN 更容易产生 HEAD amplification；
- ViT 的 isotropic attention 使 interaction 更稳定。

这个叙事有趣，但也是当前论文最容易被 reviewer 攻击的地方，因为三种模型不足以支撑强 causal claim。

---

### 1.4 当前已经完成的实验（含 Appendix 精确审计）

> **重要提示**：本节经过对 PDF Appendix A–N 的逐页核对。以下标注 ✅ 表示已完成、⚠️ 表示部分完成、❌ 表示缺失。

#### 1.4.1 主实验设置

| 项目 | 当前设置 |
|---|---|
| Dataset | OfficeHome |
| Target domain | Art 为主，Clipart 做部分 robustness |
| Source domains | Art 以外的 3 个 domains，例如 Clipart、Product、Real World |
| SSL 方法 | FixMatch |
| Confidence threshold | 主实验 τ=0.95 |
| Imbalance ratios | ρ ∈ {1, 10, 20, 50} |
| Labeled set | class-balanced，10 samples/class |
| Unlabeled pool | 按指数衰减构造 long-tail distribution |
| Metrics | pseudo-label precision、mask rate、test accuracy、ANOVA η²、F-statistic、p-value、domain spread |
| Seeds | {42, 123, 456, 789, 1024}，目标每条件 5 seeds |

#### 1.4.2 Appendix 内容审计汇总（精确版）

| Appendix | 内容摘要 | 精确状态 | 关键细节 |
|---|---|---|---|
| **A** | Per-class ANOVA: 3 arch × 4 ρ, Table 4 (overall + HEAD/MID/TAIL) | ✅ **完整** | R18: n=5(ρ=1,10,20), n=4(ρ=50); ViT: n=5 all ρ; ConvNeXt: n=5 all ρ |
| **B** | Threshold sensitivity: τ∈{0.80,0.95,0.99}, Tables 5-6 | ✅ **完整** | matched-seed comparison 消除 seed-count confound；τ∈[0.80,0.95] robust |
| **C** | ConvNeXt seed count: n=5 验证, |Δ|≤0.017 | ✅ **完整** | **所有 ρ 均 n=5** |
| **D** | Domain spread: R18, Figure 3 + Table 7 | ⚠️ **4-seed** | seed 1024 JSONL truncated at epoch 43。4-seed pooled estimates |
| **E** | Per-class interaction details: stratum comparison | ✅ **完整** | 跨架构 HEAD/MID/TAIL 分析。MID dominance documented |
| **F** | Cell-level cross-architecture: Table 8 | ⚠️ **部分缺失** | R18 全 ρ ✅; ViT 全 ρ ✅; ConvNeXt: ρ=1 缺失, η²_class/η²_domain 全"—", n=4 |
| **G** | ConvNeXt detailed: MID-dominant, delayed peak | ✅ **完整** | n=5, all ρ per-class 分析完整 |
| **H** | Main-effect divergence | ✅ **完整** | class/domain η² trajectory across architectures |
| **I** | Three-mode description | ✅ **完整** | 3 modes 详细数值 |
| **J** | Mechanistic hypotheses | ✅ **完整** | BN coupling + hierarchical pyramid，已标 speculative |
| **K** | Optimizer confound: 3 lines of evidence | ✅ **完整** | R18+AdamW: cell η²=0.082, per-class η²=0.224 vs SGD 0.214 |
| **L** | Broader impact | ✅ **完整** | fairness implications |
| **M** | Problem setup + notation | ✅ **完整** | Table 9 notation summary |
| **N** | Cross-condition robustness: Table 10 | ✅ **完整** | 4 conditions (SGD vs AdamW, Art vs Clipart), cell+per-class |

**关于 Table 8 ConvNeXt 的精确说明**：Table 8 显示 ConvNeXt 在 ρ={10,20,50} 的 cell-level 分析仅用 n=4 seeds，而 Appendix C 确认 per-class 使用了 n=5。这说明 cell-level ANOVA 脚本在第 5 个 seed 加入前就跑过了。**修复方式：从已有的 5 个 seed 的 pseudo-label logs 重新计算 cell-level ANOVA，无需补跑 GPU 实验。**

#### 1.4.3 实际缺失清单（经精确审计）

| 缺失项 | 类型 | 修复方式 | 成本 |
|---|---|---|---|
| ConvNeXt cell-level Table 8: ρ=1 行缺失 | 统计 | 检查 pseudo-label logs → 重新计算；若 logs 不存在则补跑 | **CPU-only 或 ~30min GPU** |
| ConvNeXt cell-level Table 8: η²_class, η²_domain 全"—" | 统计 | 从现有 logs 重新计算 cell-level ANOVA 的全部分量 | **CPU-only** |
| ConvNeXt cell-level Table 8: n=4 → n=5 更新 | 统计 | 加入第 5 个 seed 重算 | **CPU-only** |
| R18 ρ=50 seed 1024 | 训练 | 补跑 1 run | **~20min GPU** |
| Domain spread seed 1024 (JSONL truncated) | 训练 | 补跑或恢复 log | **~20min GPU** |
| "rerun pending" / "interim estimate" 语言 | 文字 | 修改或删除 | **0 cost** |

**关键结论：真正需要 GPU 的补跑工作 ≤ 1 GPU-hour。ConvNeXt cell-level 问题 90% 概率是纯 CPU 重新计算即可解决。**

#### 1.4.4 ResNet-18 / BN+CNN 实验 ✅

ResNet-18 是当前论文中最完整、最核心的实验。

主要发现：

- cell-level interaction 在 ρ=10 达到峰值（η²=0.101, p<.001）；
- class-group effect 从 0.295 上升到 0.657；
- domain effect 从 0.559 下降到 0.154；
- 在 ρ≈10 附近出现 dominance crossover；
- test accuracy 从 0.432（ρ=1）下降到 0.402（ρ=50）；
- coverage 从 0.68 上升到 0.76：模型更自信但 quality 变差。

唯一不完整：ρ=50 为 n=4（seed 1024 excluded）。

#### 1.4.5 ConvNeXt-Tiny / LN+CNN 实验 ✅（per-class）/ ⚠️（cell-level）

Appendix C 明确验证了所有 ρ 下 n=5 seeds，且第 5 个 seed 带来的 HEAD η² 变化 |Δ|≤0.017。

Per-class 分析已完成（Appendix G）：
- MID-dominant baseline at ρ=1（MID 0.363 > TAIL 0.255 > HEAD 0.194）
- Systematic stratum redistribution across ρ
- Delayed inverted-U: HEAD peak at ρ=20 (0.266)
- Overall η²_interact 从 0.269 到 0.134

Cell-level 不完整：Table 8 中 ConvNeXt 的 η²_class 和 η²_domain 列为 "—"，ρ=1 的 cell-level 整行缺失。

#### 1.4.6 ViT-S/16 / LN+Attention 实验 ✅

ViT 实验完整（n=5 all ρ）。

核心发现：
- cell-level interaction 较弱但 per-class 极其显著（ρ=1: p<10⁻⁷⁶）
- HEAD η² 稳定在 0.166–0.198 的窄带内
- MID interaction 在多数 ρ 下最强
- domain effect 始终占主导，即使 ρ=50（η²_domain=0.596 vs η²_class=0.297）

**这个 cell-level vs per-class resolution gap 是论文最强的单一贡献之一**，应该更大力推。

#### 1.4.7 Threshold sensitivity 实验 ✅

完整（Appendix B, Table 5-6）：
- τ∈[0.80, 0.95] robust
- τ=0.99 因 coverage 压缩丢失显著性
- matched-seed comparison 消除了 seed-count confound

#### 1.4.8 Cross-condition robustness ✅

完整（Appendix N, Table 10）：
- Optimizer: R18+AdamW，interaction 保持显著（per-class η²=0.224 vs SGD 0.214）
- Target domain: R18+Clipart，interaction 显著（per-class η²=0.237, p=2.5×10⁻²⁷）

#### 1.4.9 Domain spread ⚠️

4-seed interim（Appendix D, Table 7）。Seed 1024 JSONL truncated at epoch 43。需要补跑或恢复日志后重新计算。

---

## 第二部分：Review 总结

### 2.1 总体评价

当前论文的 idea 是有价值的：用 two-factor ANOVA 将 pseudo-label bias 分解为 class、domain 和 class × domain interaction，是一个清晰、可解释、容易推广的诊断框架。对于 SSL、long-tail learning 和 domain adaptation 的交叉问题，这个分析角度是合理的，也有潜力形成一篇不错的 NeurIPS analysis paper。

但是，从当前版本来看，论文更像一个 promising case study，而不是证据充分的 NeurIPS 强稿。主要问题不是写作完全不清楚，而是：

- 实验范围偏窄（单数据集、单 SSL 方法、单主 target domain）；
- architecture claim 过强（3 个模型不足以支撑 2×2 factorial causal claim）；
- 统计分析假设需要加强（未报告 retained counts，缺 weighted ANOVA）；
- 少量实验不完整（ConvNeXt cell-level main effects, seed 1024）；
- **practical significance 尚未展示——这是最大的弱点**；
- NeurIPS 格式和 checklist 存在风险。

如果按正式 review 给分，当前版本大约是：

> **Weak Reject / Borderline below acceptance，约 5/10。**

### 2.2 当前论文的主要优点

#### 优点 1：问题重要

Pseudo-labeling 在 class imbalance 和 domain shift 下同时失效，是一个真实且重要的问题。文章选择分析 pseudo-label precision 的结构化偏差，比只报告最终 accuracy 更有解释力。

#### 优点 2：方法简单且可解释

ANOVA decomposition 本身不复杂，但它能把 pseudo-label bias 分成 class effect、domain effect 和 interaction effect，具有较好的解释性。对于 analysis paper，这种简单而有效的诊断工具是加分项。

#### 优点 3：双粒度分析有价值——论文最强的单一贡献

cell-level design 可解释性强，per-class design 统计功效强。尤其是 ViT 在 ρ=1 的 cell-level 不显著（p=.267）但 per-class 极显著（p<10⁻⁷⁶）的结果，说明 aggregation 可能掩盖细粒度 interaction。**这一点应该作为论文核心卖点更大力推，比 architecture modes 更安全。**

#### 优点 4：architecture-dependent pattern 有趣

ResNet-18、ConvNeXt 和 ViT 呈现不同 interaction trajectory，这个观察有潜力引出"architecture-aware pseudo-label auditing"的新方向。

#### 优点 5：robustness 基础较好

Appendix B（threshold sensitivity + matched-seed comparison）、K（optimizer confound 3 lines of evidence）、N（target domain + optimizer switching）已经形成了较好的 robustness 链条。这些在此前的方案中被低估了。

### 2.3 当前论文的主要弱点

#### 弱点 1（最关键）：缺少 practical payoff

论文证明了 interaction 存在，但还没有证明这个 diagnostic 能改善训练、校准或最终 accuracy。纯诊断论文在 NeurIPS 天然不利——reviewer 会问 "interesting, but so what?"

加入一个简单的 class × domain-aware correction method，即使不是 SOTA，也会把论文从"仅诊断"变成"诊断 + 可操作改进"，这是**全方案中 ROI 最高的单项改动**。

#### 弱点 2：实验范围太窄

当前核心实验集中在 OfficeHome → Art + FixMatch。虽然 Appendix N 有 Clipart robustness，但只做了 ρ=10 的 R18。Reviewer 很可能会问：

- 换其他 target domain 全 ρ sweep 是否成立？
- 换 DomainNet 是否成立？
- 换 SSL 方法是否成立？
- 换 imbalanced SSL baseline 后 interaction 是否仍存在？

#### 弱点 3：architecture causal claim 证据不足

当前论文声称 normalization type 影响 amplification timing，macro-architecture 影响 amplification presence。但只有三个模型，且 Table 3 的 2×2 matrix 中 BN+Attention cell 为空。三个模型之间差异太多（normalization、optimizer、capacity、inductive bias 等），无法支撑因果推断。

#### 弱点 4：统计分析需要更严谨

- **retained pseudo-label counts 未报告**——这是 ANOVA on proportions 的根本性问题，precision=1.0 但 n=2 与 precision=0.95 但 n=200 完全不同
- 缺少 weighted ANOVA 作为 robustness
- η² 有 positive bias，应补充 ω²
- 大量 p-values 未做 multiple-testing correction

#### 弱点 5：主张"imbalance amplifies interaction"需要更精确

per-class overall interaction 在某些情况下随 ρ 增大反而下降。真正更稳定的现象是 architecture- and stratum-dependent 的 reshaping，不是普适的 amplification。

#### 弱点 6：格式与 checklist 问题

- footer "Preprint. Under review." 不适合正式匿名提交
- 缺 line numbers
- Type 3 font 风险
- Checklist 声称 code open access 但实际未发布
- 个别位置出现 "pending rerun" 语言

---

## 第三部分：优化方案（v3 饱和执行版）

### 3.0 核心策略

沿三条线同时推进：

1. **扩大证据面**：多 target domain + 多 SSL method + 第二 benchmark → 消除"cherry-picking"/"method-specific"质疑
2. **增加实用价值**：class-domain-aware correction method → 从"诊断"到"可操作"
3. **加固统计基础**：retained counts + weighted ANOVA + effect size variants → 消除统计质疑

**并行原则**：GPU 实验、CPU 统计分析、论文写作三条线并行执行。

---

### 3.1 优先级 P0：投稿前必须修复（Day 0，4–6 小时）

> 这些不修会直接导致 desk reject 或 reviewer 信任崩溃。

#### P0.1 补齐 missing 实验结果 [~1 GPU-hour]

| 任务 | 步骤 |
|---|---|
| 检查 ConvNeXt ρ=1 pseudo-label logs | 若存在 → CPU 重算 cell-level ANOVA；若不存在 → 补跑 1 条件（5 seeds, ~2.5h） |
| ConvNeXt cell-level η²_class, η²_domain | 从已有 n=5 logs 重算 → 填充 Table 8 的 "—" |
| R18 ρ=50 seed 1024 | 补跑 1 run（~20min） |
| Domain spread seed 1024 | 补跑 1 run 或恢复 JSONL（~20min） |

#### P0.2 报告 retained pseudo-label counts [CPU-only]

**硬性要求**。从已有 pseudo-label logs 提取每个 (ρ, arch, domain, class/group) cell 的：
- total unlabeled samples, retained samples, correct pseudo-labels, precision, coverage

主文：group-level summary table。Appendix：full heatmap。

#### P0.3 Weighted ANOVA [CPU-only]

用 retained count 作为权重，重跑 ANOVA。比较 weighted vs unweighted 的 interaction trajectory 一致性。

#### P0.4 NeurIPS 格式修复 [文字]

- 官方 submission mode（去掉 "Preprint. Under review."）
- 加 line numbers
- 检查 Type 3 fonts（`pdf.fonttype=42`）
- PDF metadata 匿名
- main text ≤ 9 pages

#### P0.5 Checklist + "pending" 语言修复 [文字]

- code: "will be released upon acceptance"
- 删除所有 "pending rerun"、"interim estimate"
- seed 排除规则必须明确

---

### 3.2 优先级 P1-S：最高优先级实验——"杀手级"改进（Day 0.5–2）

> 这些是对论文分数影响最大的改动，每一项都直接回应 reviewer 的核心质疑。

#### P1-S.1 ★★★ Correction Method（CDAT）[~45 GPU-hours]

**分数影响**：+1.0–1.5 分。将论文从"纯诊断"变为"诊断 + 可操作"。

**执行策略（分两阶段）**：
- **Day 0 晚间**：实现最简版 CDAT——per-(predicted-class, domain) 的 EMA confidence threshold
  - 仅维护一个 running mean confidence per cell
  - threshold = max(global_τ × scale_factor, cell_ema × adjustment)
  - 不超过 100 行代码修改
- **Day 1 上午**：在 R18 × ρ=10 × 2 seeds 上做 quick validation（~1h）
  - 若 pseudo-label precision 有改善 → 继续全量实验
  - 若无改善 → 切换到 CDRW（reweighting 方案）或调整 threshold logic
- **Day 1–2**：全量实验

**备选方案 B：CDRW（Class-Domain Reweighted Pseudo-Label Loss）**
- 对每个 (predicted class, domain) cell 估计 reliability proxy（confidence variance, augmentation consistency）
- unsupervised loss 中对不同 cells 加权

**关键约束**：训练时不能使用 unlabeled ground-truth label。所有 proxy 必须基于训练时可获得的信息（predicted class, domain id, confidence, entropy, consistency）。

**实验矩阵**：

| 阶段 | 条件 | Runs | GPU-hours |
|---|---|---|---|
| Quick validation | R18 × ρ=10 × 2 seeds | 2 | ~0.5 |
| Core | Art, R18+ViT, 4ρ × 5seeds, FixMatch+CDAT vs FixMatch | 80 | ~40 |
| Extension | Clipart, R18, ρ=10, 5seeds | 10 | ~2.5 |
| **总计** | | **92** | **~43** |

**报告指标**：pseudo-label precision Δ、coverage Δ、test accuracy Δ、class-balanced accuracy Δ、worst-cell accuracy Δ、interaction η² Δ

#### P1-S.2 ★★★ 多 Target Domain（分层执行）[~50–90 GPU-hours]

**分数影响**：+0.5–1.0 分。消除"cherry-picking single target"质疑。

**分层执行，避免盲目铺量**：

**阶段 1：Quick Probe（Day 0.5，与 P0 并行）**
- R18 only × 3 missing targets (Clipart/Product/Real World) × ρ={1,10,50} × 2 seeds
- = 18 runs, ~4.5 GPU-hours
- 目的：确认 interaction trajectory 的方向性（inverted-U 是否大致出现）
- **决策点**：如果 3/3 targets 都有 interaction → 全量铺开；如果 1/3 没有 → 调整叙事后铺开；如果 0/3 → 重大叙事调整

**阶段 2：R18 Full Sweep（Day 1）**
- R18 × 3 targets × 4ρ × 5 seeds（扣除 Clipart ρ=10 已有 + probe 已跑）
- ≈ 48 runs，~12 GPU-hours

**阶段 3：ConvNeXt + ViT（Day 2，在 R18 确认后）**
- 2 arch × 3 targets × 4ρ × 5 seeds = 120 runs, ~60 GPU-hours
- **可按 target 优先级排序**：先跑 probe 中 interaction 最强的 target

**总计**：~166 runs, ~77 GPU-hours

**主文报告方式（新增 summary table）**：

| Target | R18 interaction peak ρ | ConvNeXt peak ρ | ViT pattern | 支持主结论？ |
|---|---|---|---|---|
| Art | 10 | 20 | flat/stable | ✅ |
| Clipart | 待补 | 待补 | 待补 | ? |
| Product | 待补 | 待补 | 待补 | ? |
| Real World | 待补 | 待补 | 待补 | ? |

#### P1-S.3 ★★ FlexMatch（adaptive threshold SSL）[~40 GPU-hours]

**分数影响**：+0.5 分。回答"是不是 FixMatch 特有的？"

FlexMatch 使用 per-class adaptive threshold，是最自然的对照：
- 如果 interaction 在 adaptive threshold 下仍然存在 → 这是结构性问题，论文 motivation 大幅增强
- 如果 interaction 减弱但不消失 → 更有趣，说明 thresholding 调制但不消除 interaction
- 如果 interaction 消失 → 论文需要调整 claim

**实验矩阵**：Art target, 3 arch × 4ρ × 5 seeds = 60 runs

**依赖**：FlexMatch 脚本准备（基于 FixMatch 代码修改，~2–3h 开发）

#### P1-S.4 ★★ DARP + AdaMatch 对照 [~25 GPU-hours]

**分数影响**：+0.5 分。直接证明"class-only 和 domain-only correction 都不够"。

| 方法 | 类型 | 核心问题 | Runs |
|---|---|---|---|
| **DARP** | class-imbalanced SSL (distribution alignment) | class-only rebalancing 后 interaction 还存在吗？ | 20 |
| **AdaMatch** | SSL + domain adaptation | domain alignment 后 interaction 还存在吗？ | 20 |

**实验矩阵**：Art target, R18, 4ρ × 5 seeds × 2 methods = 40 runs

**叙事价值极高**：如果两者都无法消除 interaction → "class×domain decomposition is necessary" 被直接证明。

#### P1-S.5 ★★ Swin Transformer（填充 Table 3）[~15 GPU-hours]

**分数影响**：+0.5 分。填补 2×2 矩阵的关键空缺。

Swin 是 hierarchical + attention + LN，是当前 Table 3 中空缺 cell 的最佳候选：

| | BN | LN |
|---|---|---|
| Hierarchical CNN | R18: peak ρ=10 ✅ | ConvNeXt: peak ρ=20 ✅ |
| Isotropic attention | — (空缺) | ViT: flat-band ✅ |
| **Hierarchical attention** | — | **Swin: ?** |

**提前到 Day 1 启动**。原因：只有 20 runs，是所有 P1 中最轻量的；结果直接影响 Table 3 叙事的写法，越早知道越好。

**实验矩阵**：Art target, 4ρ × 5 seeds = 20 runs

---

### 3.3 优先级 P1-C：统计加强（全部 CPU 并行）[0 GPU-hours]

> 与 GPU 实验完全并行执行，不占用 GPU 资源。

#### P1-C.1 Effect size variants
- ω²（减少 η² 的 positive bias）
- partial η²、generalized η²
- bootstrap CI over seeds

#### P1-C.2 Binomial mixed-effects model (GLMM)
- logit(p_{c,d,s}) = μ + α_c + β_d + (αβ)_{c,d} + u_s
- seed 作为 random effect
- **风险管理**：先在现有数据上跑，确认方向与 ANOVA 一致后再写入论文

#### P1-C.3 Multiple testing correction
- Benjamini-Hochberg FDR correction
- 区分 primary tests（少数核心 claim）vs exploratory

#### P1-C.4 Nonparametric permutation test
- 对 class-domain assignment 做 1000+ permutations
- 计算 null distribution 下 η² 的分布

#### P1-C.5 ★ Training dynamics 分析 [CPU-only]

论文声明 "At each epoch's end we record every unlabeled sample's pseudo-label alongside ground truth"。这意味着已有 epoch-level pseudo-label logs。

分析内容：
- interaction η² 随 epoch 的演变轨迹（e.g., η²_interact at epoch {10, 20, 30, 40, 50}）
- interaction 是早期就出现还是训练后期才浮现？
- 不同 architecture 下 interaction 的出现时间点是否不同？

**价值**：
1. 回答 "results depend on epoch choice" 的质疑
2. 揭示 self-reinforcing bias amplification 的时序动态
3. 为 correction method 的 early intervention 提供依据
4. **完全免费——只需从已有 logs 提取**

---

### 3.4 优先级 P1-B：高优先级扩展实验（Day 2–3）

#### P1-B.1 DomainNet subset [~30 GPU-hours]

第二 benchmark，消除 "single-dataset" 质疑。

**class selection rule（锁定，不可更改）**：
- 规则：DomainNet 官方 class 列表，按字母序取前 65 个
- 4 domains：Clipart, Painting, Real, Sketch
- Target domain：Real（与 OfficeHome Art 的"高多样性"类似）

**实验矩阵**：
- Phase 1：R18, FixMatch, 4ρ × 5 seeds = 20 runs（~30 GPU-hours）
- Phase 2（P2 级别）：ViT, 4ρ × 5 seeds = 20 runs（~30 GPU-hours）

#### P1-B.2 Oracle diagnostic analysis [CPU-only]

用真实 unlabeled labels：
- 识别 high-error class-domain cells
- 模拟修复后 accuracy 上限
- 量化 interaction 的"危害上界"

**必须标注：oracle 仅用于分析，不用于训练或模型选择。**

---

### 3.5 优先级 P2：扩展实验（Day 3–4，资源允许时执行）

> 这些把论文从 7.5 推向 8/10。时间紧张时可降级或放弃。

| 编号 | 实验 | Runs | GPU-hours | 价值 |
|---|---|---|---|---|
| P2.1 | FlexMatch × 多 target domains (R18) | 60 | ~30 | 证明 FlexMatch 结论在多 target 下 generalize |
| P2.2 | ResNet-18 + GroupNorm | 20 | ~10 | 直接测试 normalization 对 interaction 的因果影响 |
| P2.3 | DomainNet + ViT | 20 | ~30 | 第二 benchmark 的 architecture generalization |
| P2.4 | SoftMatch（ICLR 2023） | 20 | ~15 | soft weighting vs hard thresholding 的对照 |
| P2.5 | DARP/AdaMatch × 其他 target | 40 | ~20 | baseline comparison generalization |
| P2.6 | PACS lightweight validation | 10 | ~3 | 第三 benchmark（类别少，只做 cell-level） |
| P2.7 | Naturally imbalanced split | 20 | ~10 | 非 synthetic imbalance 的验证 |
| P2.8 | Correction × FlexMatch | 40 | ~20 | CDAT 在 adaptive threshold 下的效果 |
| **总计** | | **~230** | **~138** | |

---

### 3.6 论文叙事优化

#### 3.6.1 推荐标题方向

**最推荐（取决于 Swin 结果）**：

- 若 Swin 呈现 inverted-U：Decomposing Pseudo-Label Bias: Class–Domain Interaction Across Architectures in Semi-Supervised Learning
- 若 Swin 呈现 flat-band：Decomposing Pseudo-Label Bias: How Architecture Shapes Class–Domain Interaction in Semi-Supervised Learning

**最安全选择**：
> Auditing Pseudo-Label Bias at the Class–Domain Intersection in Semi-Supervised Learning

#### 3.6.2 修订后的 4 个 Contributions

1. **Two-resolution diagnostic framework**：ANOVA decomposition at cell-level and per-class resolution. Per-class design detects interaction invisible to coarse aggregation (ViT ρ=1: cell p=.267 vs per-class p<10⁻⁷⁶).

2. **Structural phenomenon across conditions**：Class×domain interaction is significant across 4 target domains (OfficeHome), 2 datasets (OfficeHome + DomainNet), 2+ SSL methods (FixMatch + FlexMatch), and 4 architectures (R18, ConvNeXt, ViT, Swin). Neither class-only (DARP) nor domain-only (AdaMatch) correction eliminates it.

3. **Architecture-dependent redistribution**：Imbalance does not simply amplify interaction; it redistributes it across frequency strata in architecture-dependent ways. Hierarchical CNNs exhibit inverted-U HEAD amplification; isotropic attention maintains stable interaction.

4. **Actionable correction**：A simple class-domain-aware thresholding method reduces interaction η² and improves worst-cell pseudo-label quality, demonstrating the diagnostic's practical value.

#### 3.6.3 需要弱化的 Claims

- "normalization determines timing" → "normalization is associated with timing in our observations"
- "macro-architecture determines presence" → "macro-architecture modulates whether HEAD amplification appears"
- "class imbalance amplifies interaction" → "moderate imbalance can amplify interaction in specific strata/backbones, while extreme imbalance suppresses it"
- "three cells suffice to disentangle factors" → 删除此表述

#### 3.6.4 需要强化的 Claims

- **per-class resolution reveals interaction missed by coarse aggregation**——最安全、最强的 contribution
- pseudo-label coverage can increase while quality decreases（coverage-quality paradox）
- class-only and domain-only corrections are insufficient
- interaction decomposition provides actionable diagnostics（配合 correction method）
- the phenomenon generalizes across target domains, SSL methods, and datasets
- **Training dynamics**：interaction 在训练过程中如何演变（新增）

---

### 3.7 建议新增图表

#### 主文

| 图 | 内容 | 作用 |
|---|---|---|
| Figure 1 | η² trajectory across ρ（保留） | 核心 inverted-U |
| Figure 2 | HEAD η² across architectures（保留，加 Swin） | 3+1 modes |
| Figure 3 (**新**) | 4 target domains: per-class interaction η² across ρ, R18 | generalization |
| Figure 4 (**新**) | SSL method comparison: FixMatch vs FlexMatch vs DARP vs AdaMatch, η²_interact | "neither class-only nor domain-only" |
| Figure 5 (**新**) | CDAT correction: before/after comparison (η² + accuracy + worst-cell) | practical payoff |

#### Appendix 新增

| 表/图 | 内容 |
|---|---|
| Table: Retained counts | 每个 cell 的 sample counts + precision（P0.2） |
| Table: Weighted ANOVA | weighted vs unweighted η² comparison（P0.3） |
| Table: 4 targets full | 每个 target × arch 的 cell-level + per-class ANOVA |
| Table: Swin | 填充 Table 3 + per-class ANOVA |
| Table: FlexMatch | FixMatch vs FlexMatch full ANOVA |
| Table: DARP + AdaMatch | class-only / domain-only ANOVA |
| Table: DomainNet | 第二 benchmark ANOVA |
| Table: GLMM | binomial mixed-effects model |
| Table: ω² | effect size robustness |
| Figure: Training dynamics | η²_interact across epochs（P1-C.5） |
| Figure: Correction details | CDAT 实现 + full results |
| Figure: Oracle analysis | 修复 high-error cells 的理论 accuracy 上限 |

---

## 第四部分：饱和执行计划（3.5 天）

### 4.1 资源预算

| 资源 | 规格 | 备注 |
|---|---|---|
| GPU 1–3 | RTX 5090 (32GB) × 3 | 每卡 1 个 OfficeHome 实验 |
| GPU 4 | RTX PRO 6000 (96GB) | 可同时跑 2 个 OfficeHome 实验（每个 ~6-8GB） |
| 等效 GPU slots | **4–5 并行** | RTX PRO 6000 可当 1.5-2 张用 |
| 总 GPU-hours（3.5 天） | **336–420** | 4 slots × 24h × 3.5 days |
| OfficeHome 单 run | R18 ~15min, ConvNeXt/ViT/Swin ~25min | |
| DomainNet 单 run | ~40–60min | 数据集更大 |

### 4.2 实验总量

| 优先级 | 实验组 | Runs | GPU-hours | Day |
|---|---|---|---|---|
| P0.1 | 补齐 missing seeds + ConvNeXt cell-level | ~3 | ~1 | Day 0 |
| P1-S.1 | Correction (CDAT) quick val + core | 92 | ~43 | Day 1–2 |
| P1-S.2 | 多 target: probe + R18 sweep + ConvNeXt/ViT | 166 | ~77 | Day 0.5–2 |
| P1-S.3 | FlexMatch (Art, 3 arch) | 60 | ~40 | Day 1–2 |
| P1-S.4 | DARP + AdaMatch (Art, R18) | 40 | ~25 | Day 1 |
| P1-S.5 | Swin Transformer (Art) | 20 | ~15 | Day 1 |
| P1-B.1 | DomainNet R18 | 20 | ~30 | Day 2–3 |
| P2.1 | FlexMatch × 多 targets | 60 | ~30 | Day 3 |
| P2.2 | R18 + GroupNorm | 20 | ~10 | Day 3 |
| P2.3 | DomainNet + ViT | 20 | ~30 | Day 3 |
| P2.4 | SoftMatch | 20 | ~15 | Day 3 |
| P2.5–P2.8 | 其他扩展 | ~110 | ~53 | Day 3.5 |
| **总计** | | **~631** | **~369** | |

**覆盖结论**：4 GPU slots × 3.5 天 = ~336–420 GPU-hours。P0 + P1-S + P1-B（核心）≈ 231 GPU-hours，在 Day 2 结束前即可完成。Day 3–3.5 执行 P2 扩展。

### 4.3 Day-by-Day 详细执行计划

---

#### Day 0（准备 + 立即修复，~6 小时）

**代码开发（并行进行）**：

| 任务 | 预计时间 | 优先级 |
|---|---|---|
| CDAT 最简版实现 | 2–3h | 最高 |
| FlexMatch 适配（基于现有 FixMatch 脚本） | 1–2h | 高 |
| DARP/AdaMatch 脚本准备 | 2h | 高 |
| Swin Transformer backbone 集成（timm） | 1h | 高 |
| DomainNet 数据 split 生成（字母序前 65 classes） | 1h | 中 |
| OfficeHome 多 target config 文件 | 0.5h | 高 |

**GPU 立即启动**：

| GPU | 任务 | 时间 |
|---|---|---|
| GPU 1 | R18 ρ=50 seed 1024 补跑 | ~20min |
| GPU 1 | Domain spread seed 1024 补跑 | ~20min |
| GPU 1 | ConvNeXt ρ=1 补跑（仅在 logs 不存在时） | 0 或 ~2.5h |

**CPU 立即启动**：

| 任务 | 预计时间 |
|---|---|
| ConvNeXt cell-level ANOVA 重算（Table 8 填充） | 1h |
| P0.2 Retained counts 全量提取 | 2–3h |
| P0.3 Weighted ANOVA 计算 | 1–2h |
| P1-C.5 Training dynamics：interaction η² across epochs | 3–4h |
| P1-C.1 ω², partial η², generalized η² | 2h |

**Day 0 晚间（代码就绪后，启动 probe）**：

| GPU | 任务 | Runs | 时间 |
|---|---|---|---|
| GPU 1 | Target probe: Clipart, R18, ρ={1,10,50}, 2 seeds | 6 | ~1.5h |
| GPU 2 | Target probe: Product, R18, ρ={1,10,50}, 2 seeds | 6 | ~1.5h |
| GPU 3 | Target probe: Real World, R18, ρ={1,10,50}, 2 seeds | 6 | ~1.5h |
| GPU 4 | CDAT quick validation: R18, ρ=10, 2 seeds | 2 | ~0.5h |

**Day 0 结束时的交付物**：
- ✅ Table 8 ConvNeXt 完整填充
- ✅ Retained counts 报告
- ✅ Weighted ANOVA 结果
- ✅ Training dynamics 初步结果
- ✅ ω²/partial η² 结果
- ✅ 3 个 target domain 的 probe 结果 → **决策：是否全量铺开**
- ✅ CDAT quick validation → **决策：继续 CDAT 还是切换 CDRW**

---

#### Day 1（核心实验第一波，GPU 全程饱和）

**GPU 分配（4–5 slots 饱和）**：

| GPU | 任务 | Runs | 预计时间 |
|---|---|---|---|
| GPU 1 | R18 多 target full sweep: Clipart + Product + Real World, 4ρ × 5seeds（扣除已有+probe） | ~43 | ~11h → **接续** FlexMatch, Art, R18, 4ρ × 5seeds | 20 | ~5h |
| GPU 2 | DARP, Art, R18, 4ρ × 5seeds → **接续** AdaMatch, Art, R18, 4ρ × 5seeds | 40 | ~20h |
| GPU 3 | Swin Transformer, Art, 4ρ × 5seeds | 20 | ~8h → **接续** FlexMatch, Art, R18 剩余 or ConvNeXt 开始 | 20 | ~8h |
| GPU 4 (PRO 6000) | FlexMatch, Art, ConvNeXt + ViT, 4ρ × 5seeds（并行 2 experiments） | 40 | ~20h |

**Day 1 GPU 利用率**：4 GPU × 20–24h ≈ **80–96 GPU-hours**

**CPU 并行**：

| 任务 | 预计时间 |
|---|---|
| P1-C.3 Multiple testing correction（BH FDR） | 1h |
| P1-C.4 Permutation test（1000 permutations） | 4–6h |
| 分析 Day 0 probe 结果 | 1h |
| NeurIPS 格式修复 | 2h |
| Checklist + "pending" 语言修复 | 1h |
| 论文叙事初步修订（弱化 causal claim, 调整 contribution） | 3–4h |

**Day 1 中午检查点**：
- Swin 首批结果 → inverted-U 还是 flat-band？→ 更新 Table 3 叙事
- DARP 首批结果 → class-only rebalancing 后 interaction 还在吗？

**Day 1 结束时的交付物**：
- ✅ R18 × 3 target domains 全量完成
- ✅ Swin Transformer 完整结果
- ✅ DARP + AdaMatch 完整结果
- ✅ FlexMatch (Art) R18 完成或接近完成
- ✅ FlexMatch (Art) ConvNeXt + ViT 进行中

---

#### Day 2（核心实验第二波 + Correction + DomainNet）

**GPU 分配**：

| GPU | 任务 | Runs | 预计时间 |
|---|---|---|---|
| GPU 1 | OfficeHome → 3 targets, ConvNeXt, 4ρ × 5seeds × 3 targets | 60 | ~24h |
| GPU 2 | OfficeHome → 3 targets, ViT, 4ρ × 5seeds × 3 targets | 60 | ~24h |
| GPU 3 | CDAT correction, Art, R18 + ViT, 4ρ × 5seeds | 80 | ~20h |
| GPU 4 (PRO 6000) | DomainNet, R18, FixMatch, 4ρ × 5seeds + overflow | 20+20 | ~20h |

**Day 2 GPU 利用率**：4 GPU × 20–24h ≈ **80–96 GPU-hours**

**CPU 并行**：

| 任务 | 预计时间 |
|---|---|
| 分析 Swin → 更新 Table 3 | 2h |
| 分析 DARP + AdaMatch | 2h |
| 分析 FlexMatch vs FixMatch | 2h |
| 分析 R18 × 4 targets：制作 summary figure | 3h |
| P1-C.2 Binomial mixed-effects model (GLMM) | 4–6h |
| P1-B.2 Oracle diagnostic analysis | 2–3h |
| 更新论文 Figure 3–5 | 3–4h |
| 更新 Appendix tables | 2–3h |

**Day 2 结束时的交付物**：
- ✅ OfficeHome 4 targets × 3 arch 基本完成
- ✅ FlexMatch (Art, 3 arch) 完整
- ✅ CDAT correction 完成或接近完成
- ✅ DomainNet R18 进行中
- ✅ GLMM + Oracle analysis 完成
- ✅ 论文主文修订初稿

---

#### Day 3（扩展实验 + 论文精修）

**GPU 分配（P2 + 收尾）**：

| GPU | 任务 | Runs | 预计时间 |
|---|---|---|---|
| GPU 1 | 多 target 收尾 + P2.2 R18+GroupNorm | ~40 | ~13h → **接续** P2.6 PACS | 10 | ~2h |
| GPU 2 | P2.1 FlexMatch × 2 targets → P2.4 SoftMatch | 60+20 | ~20h |
| GPU 3 | CDAT 补充 + P2.5 扩展 | ~70 | ~20h |
| GPU 4 (PRO 6000) | DomainNet 收尾 + P2.3 DomainNet ViT | 40 | ~20h |

**CPU 并行**：

| 任务 | 预计时间 |
|---|---|
| 分析 CDAT correction 结果 | 3h |
| 分析 DomainNet ANOVA | 2h |
| 论文主文精修（§4.2–4.6 全部更新） | 6–8h |
| 全部新图表制作 | 4h |
| Appendix 更新 | 3–4h |

---

#### Day 3.5（收尾 + 论文定稿，半天）

**GPU 任务**：补跑任何失败/异常的 seeds，P2.7–P2.8 扩展。

**写作任务**：
- 论文 main text 终稿（≤9 pages）
- Appendix 终稿
- Abstract 重写
- NeurIPS 格式终检
- Checklist 最终核对
- PDF 编译 + 匿名检查

---

### 4.4 关键决策点与应急预案

| 时间 | 决策点 | 可能结果 | 应对策略 |
|---|---|---|---|
| Day 0 晚 | 3 target probe | A) 3/3 有 interaction → 全量铺开 | 按计划执行 |
| | | B) 2/3 有 → 报告异质性 | 铺开但叙事加 "strength varies across targets" |
| | | C) 0-1/3 有 → 不 generalize | 聚焦 Art+Clipart，加强其他维度 |
| Day 0 晚 | CDAT quick val | A) precision 改善 > 2% | 全量实验 |
| | | B) 改善 < 1% | 切换 CDRW 或调整 |
| | | C) 性能下降 | 降级为 oracle diagnostic |
| Day 1 中午 | Swin 结果 | A) inverted-U | 强化 "hierarchy drives amplification" |
| | | B) flat-band | "attention prevents amplification" |
| | | C) 介于两者 | Table 3 改为 "spectrum" |
| Day 1 中午 | DARP 结果 | A) interaction 仍存在 | 写入主文 §4.4 |
| | | B) 消失 | 调整叙事，仍有分析价值 |
| Day 2 中午 | FlexMatch | A) interaction 存在 | 强化 motivation |
| | | B) 减弱但不消失 | "thresholding modulates but doesn't eliminate" |
| | | C) 消失 | 调整 claim scope |
| Day 2 晚 | GLMM vs ANOVA | A) 一致 | 加强统计可信度 |
| | | B) 不一致 | 分析原因，讨论差异 |
| Day 3 | R18+GroupNorm | A) 收敛 + trajectory 改变 | 写入 appendix |
| | | B) 不收敛 | 放弃 |

---

## 第五部分：预期论文结构与分数

### 5.1 预期论文结构（补强后）

| Section | 内容 | Pages |
|---|---|---|
| §1 Introduction | 问题定义 + resolution gap motivation + correction motivation | ~1.5 |
| §2 Related Work | SSL + imbalanced SSL + DA + variance decomposition | ~1 |
| §3 Method | Two-resolution ANOVA + domain spread + cross-arch protocol + CDAT | ~1.5 |
| §4.1 Setup | OfficeHome/DomainNet, FixMatch/FlexMatch, 4 arch, 4 targets | ~0.5 |
| §4.2 Non-Monotonic Interaction | R18 inverted-U + 4 target generalization + DomainNet | ~1 |
| §4.3 Architecture Modes | 4 arch, updated Table 3 with Swin | ~1 |
| §4.4 Neither Class-Only Nor Domain-Only | DARP vs AdaMatch vs FlexMatch | ~0.5 |
| §4.5 Correction | CDAT results, η² Δ + accuracy Δ + worst-cell Δ | ~0.5 |
| §4.6 Robustness | threshold, optimizer, training dynamics, DomainNet | ~0.5 |
| §5 Discussion + Limitations | 弱化 causal, acknowledge scope, future work | ~1 |
| **Total** | | **~9** |

### 5.2 预期分数提升

| 版本 | 分数 | 状态 | 核心改变 |
|---|---|---|---|
| 当前版本 | 5/10 | Weak Reject | 单 dataset/method/target, 无 practical payoff |
| + P0 | 5.5/10 | Borderline | 格式/统计修复，消除信任风险 |
| + P1-S.2 多 target + P1-S.5 Swin | 6.0–6.5/10 | Borderline | generalization + Table 3 填充 |
| + P1-S.3 FlexMatch + P1-S.4 DARP/AdaMatch | 6.5–7.0/10 | Borderline Accept | "structural phenomenon" claim 有支撑 |
| + P1-S.1 Correction (CDAT) | **7.0–7.5/10** | **Accept** | practical payoff = 论文定位质变 |
| + P1-B DomainNet + P1-C stats | **7.5/10** | Accept | 第二 benchmark + 统计严谨性 |
| + P2 扩展 | **7.5–8.0/10** | Solid Accept | 全方位覆盖 |

**最关键的分数跳跃**：
- 5 → 6.5：multi-target + Swin + FlexMatch + DARP/AdaMatch
- 6.5 → 7.5：**Correction method**（纯诊断 → 可操作）
- 7.5 → 8.0：DomainNet + 统计加强 + P2 扩展

### 5.3 建议的最终论文定位

> 本文提出一个系统的 pseudo-label bias auditing framework，基于 two-resolution ANOVA decomposition，揭示 class imbalance 与 domain shift 的交互结构。通过 OfficeHome（4 target domains）和 DomainNet 上的多 SSL 方法、多架构验证，论文证明 class × domain interaction 是结构性的、普遍存在的，且不能被 class-only 或 domain-only correction 消除。论文进一步展示一个简单的 class-domain-aware correction 可以降低 interaction 并改善 worst-cell performance，证明 diagnostic 的实用价值。

---

## 第六部分：学术规范注意事项

1. **不要选择性报告**：所有跑过的 target、method、seed 都应记录；失败结果放 appendix，不能删除。
2. **不要使用 test labels 做模型选择或 training**：CDAT 只用 predicted class + domain + confidence。
3. **明确 primary vs exploratory analysis**：主结论对应少数预先定义的 tests，其他标为 exploratory + BH FDR correction。
4. **所有排除必须透明**：seed 排除、日志截断、OOM、训练失败都要说明原因和排除规则。
5. **不要过度解释机制**：BN batch-coupling 和 hierarchical feature pyramid 作为 hypothesis 而非 causal conclusion。
6. **代码与数据可复现**：至少提供 split generation、training scripts、pseudo-label logging、ANOVA analysis scripts。
7. **统计显著性 ≠ 实际重要性**：报告 effect size、CI、accuracy impact 和 worst-cell performance，避免只堆 p-values。
8. **Correction method 不泄露 ground truth**：所有训练时的决策必须基于 predicted class / domain / confidence 等可观测量。
9. **DomainNet class selection 规则不可事后修改**——锁定字母序前 65 个 classes。
10. **负面结果必须报告**——如果某 target domain 不支持结论，放 appendix 作为 honest negative，反而增加可信度。

---

## 第七部分：最短时间内最应补充的实验（优先级排序）

如果时间极其紧张（只有 2 天），按以下顺序取舍：

| 优先级 | 实验 | 理由 | 不做的后果 |
|---|---|---|---|
| **#1** | P0 全部 | 不修就会被 reject | desk reject 风险 |
| **#2** | P1-S.1 CDAT Correction | 论文定位质变 | "so what?" 质疑无法回应 |
| **#3** | P1-S.2 多 Target（至少 R18 × 3 targets） | 消除 cherry-picking | "single-target" 直接被打 |
| **#4** | P1-S.4 DARP + AdaMatch | 证明 decomposition 的必要性 | "class-only/domain-only 够了" 无法反驳 |
| **#5** | P1-S.3 FlexMatch（至少 Art × R18） | 证明非 FixMatch 特有 | "method-specific" 质疑 |
| **#6** | P1-S.5 Swin | 填充 Table 3 | architecture claim 停留在推测 |
| **#7** | P1-B.1 DomainNet | 第二 benchmark | "single-dataset" 质疑 |
| **#8** | P1-C 统计加强 | 统计严谨性 | 被统计 reviewer 质疑 |
| **#9+** | P2 扩展 | 锦上添花 | 分数上限从 8 降到 7.5 |

**最低可行方案（2 天）**：P0 + #2–#5 ≈ ~133 GPU-hours，2 张 5090 可完成。  
**推荐方案（3.5 天）**：P0 + #2–#8 ≈ ~265 GPU-hours，3 张 5090 + 1 张 PRO 6000 可覆盖。  
**饱和方案（4 天）**：全部 ≈ ~369 GPU-hours，4 GPU slots 可覆盖。
