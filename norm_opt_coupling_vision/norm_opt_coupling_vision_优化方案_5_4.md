# 论文优化方案：Normalization Modulates the Spectral Advantage of Gradient Orthogonalization

**目标会议：** NeurIPS 2026  
**执行周期：** 4 天（96 小时），不间断运行  
**硬件资源：** 4 × RTX 5090，全程饱和利用  
**运行模式：** Agent Auto Research，Claude Code 全程代理  
**版本：** 终版

---

# 第一部分：当前论文现状

## 1.1 论文核心内容

本文研究归一化层（BatchNorm / LayerNorm / GroupNorm）与优化器（Muon / AdamW / SGD）在谱域（spectral domain）的交互作用。作者提出了一个新度量——**谱耦合分数（Spectral Coupling Score, SCS）**，用于量化归一化层方向向量与相邻权重矩阵奇异向量之间的对齐程度，并通过置换检验进行零假设校准。

**当前叙事框架：** "归一化调制（modulate）了梯度正交化的谱优势"——偏被动观察性质，将BN谱屏障定位为一种阻碍。

**三大核心发现：**

1. **BN谱屏障**：在BatchNorm下，非正交化优化器（AdamW、SGD）的权重主奇异向量主动回避BN归一化方向（SCS低于null baseline），而Muon通过梯度正交化维持正耦合
2. **Muon跨归一化一致耦合提升**：Muon在BN和LN下均提供约+0.0035的SCS excess优势（交互检验 p=0.895），提升不依赖归一化类型
3. **耦合-精度链接的归一化依赖性**：在可学习参数归一化（LN, GN）下，耦合差异直接转化为精度差异（+5.49pp, +6.39pp）；在BN下，同样的耦合提升仅带来+0.89pp——约6倍的不对称压缩

## 1.2 已完成的全部实验清单（含附录）

以下逐一梳理正文 + Appendix A-K 中**所有**已完成的实验，确保后续规划不产生重复。

### 主实验（正文 §4）

| 编号 | 实验内容 | 位置 | 详情 |
|------|---------|------|------|
| E1 | 核心2×2 factorial精度矩阵 | Table 1, §4.2 | ResNet-18(BN) × {AdamW, Muon} + DeiT-Tiny(LN) × {AdamW, Muon}，CIFAR-100，3 seeds {42,137,2024}，含LR grid search |
| E2 | SCS excess-above-null分析 | Table 2, §4.3 | 上述4条件的per-seed z-scores + 描述性cross-seed t-test，N=1000置换，k=10 |
| E3 | 交互检验（primary test） | §4.3 | Muon SCS advantage 在BN vs LN之间的 paired t-test (t=0.149, p=0.895, df=2) |
| E4 | SGD扩展（BN下第三优化器） | §4.4, Figure 1 | ResNet-18(BN) + SGD，3 seeds，揭示BN下三种耦合模式 |
| E5 | BN vs GN(1,C) 解耦实验 | Table 3, §4.6, Figure 2 | 同一ResNet-18上将BN替换为GN(1,C)，{Muon, AdamW, SGD} × 3 seeds（GN+Muon为4 seeds），CIFAR-100 |
| E6 | 耦合-精度传导分析 | §4.5 | BN作为"均场均衡器"的6倍不对称量化 |
| E7 | 标准谱指标对比 | §5 | 有效秩、条件数、梯度熵均无法检测BN谱屏障（seed=42, 所有条件）。注：这里对比的是**优化器之间**的差异，不是归一化之间 |
| E8 | 跨数据集验证 | §4.7, Table 4 | Tiny-ImageNet上 ResNet-18 × {BN, GN} × {Muon, AdamW, SGD}，2 seeds {42, 137} |

### 附录实验（Appendix A-K，逐项列出）

| 编号 | 实验内容 | 位置 | 详情 |
|------|---------|------|------|
| A1 | 完整超参数汇总 | Appendix A, Table 9 | 所有条件的LR/WD/momentum/epochs等完整记录 |
| A2 | 统计框架细节 | Appendix B | per-seed z-score公式(Eq.8)、交互检验公式(Eq.9)、cross-norm不可比约束 |
| A3 | **权重衰减消融** | Appendix C, Table 5 | DeiT-Tiny(LN): AdamW wd=0.01 vs wd=0.0 vs Muon wd=0，3 seeds。去除wd仅影响-0.11pp，Muon优势+5.62pp保持 |
| A4 | **BN三模式SCS轨迹** | Appendix D, Table 6 | ResNet-18(BN) × {SGD, AdamW, Muon}，epochs {10,50,100,200} SCS excess ± SE，3-seed均值 |
| A5 | **LN轨迹详情** | Appendix E, Table 7 | DeiT-Tiny(LN) × {AdamW, Muon}，epochs {10,50,100,200} SCS excess，**seed=42 单种子** |
| A6 | **GN SCS轨迹** | Appendix F, Table 8 | ResNet-18 GN(1,C) × {Muon, AdamW, SGD}，epochs {10,50,100,200}，seed=42 |
| A7 | **GN+Muon晚期反转** | Appendix G | 4 seeds确认反转（z = -3.05, -1.91, -4.41, -3.10），aggregate t=-5.81, p=0.010。含wd=0控制实验 |
| A8 | **Per-layer SCS分解** | Appendix H | GN+Muon反转由layer4.1.conv2（512×512, 21%参数）主导 |
| A9 | **Tiny-ImageNet精度矩阵** | Appendix K, Table 10 | ResNet-18 × {BN, GN} × {SGD, Muon, AdamW}，2 seeds |
| A10 | **跨数据集pattern一致性** | Table 4, Table 11 | 6个key pattern在CIFAR-100 vs Tiny-ImageNet的方向一致性 |
| A11 | **跨数据集SCS z-score** | Appendix K, Table 12 | 6个条件在两个数据集上epoch-200 z-score对比 |
| A12 | **Tiny-ImageNet SCS轨迹** | Appendix K | 6条件×epochs {10,50,100,200}，含cross-seed复制 |

### 关键已有结果：GN(1,C) on ResNet-18 ≈ LN on ResNet-18

**论文明确指出**（第220-221行）：GN(1,C) 等价于 per-sample LayerNorm over the channel dimension。因此 Table 3 中 ResNet-18 × GN(1,C) 的结果**已经实质性地等同于"LN on ResNet-18"**，这组实验不需要重做。后续实验规划应基于此避免重复。

### 已完成实验资源统计

| 维度 | 已覆盖 | 缺失 |
|------|--------|------|
| **数据集** | CIFAR-100, Tiny-ImageNet | ImageNet-100, ImageNet-1K |
| **架构** | ResNet-18, DeiT-Tiny | ResNet-50, ViT-Small, ConvNeXt-T |
| **优化器** | AdamW, Muon, SGD | SOAP, Shampoo, Lion |
| **归一化** | BN, LN, GN(1,C) | RMSNorm, GN多group, InstanceNorm |
| **Seeds** | 3（CIFAR-100）/ 2（Tiny-IN） | 需≥5 for robust statistics |
| **总runs** | ~72 | — |

## 1.3 当前论文的核心弱点（按严重程度排序）

| 弱点 | 严重程度 | 说明 |
|------|---------|------|
| **架构-归一化混淆** | 致命 | BN↔ResNet-18, LN↔DeiT-Tiny, 6倍差距可能是架构效应。GN解耦(§4.6)部分缓解但仍不充分 |
| **叙事定位偏被动，且SGD悖论未解决** | 高 | SGD+BN精度最高(78.05%)但屏障最深，直接矛盾"屏障=坏事"的隐含假设 |
| **统计效力不足** | 高 | CIFAR-100仅3 seeds (df=2), Tiny-IN仅2 seeds (df=1), LN/DeiT轨迹为单seed |
| **GN学习率公平性缺失** | 中高 | SGD+GN(1,C)沿用BN最优LR=0.01，论文自承"a broader GN-specific LR sweep may narrow the gap"——GN精度偏低可能部分是调参不充分，而非归一化本身。不扫LR则6.39pp差距不可信 |
| **规模局限** | 中高 | 仅小规模数据集和小模型，缺少ImageNet标准验证 |
| **训练配方混淆（recipe confound）** | 中 | ResNet-18和DeiT-Tiny使用完全不同的数据增强、schedule、正则化配方，跨架构精度差距可能部分归因于recipe而非norm-opt交互 |
| **优化器/归一化覆盖不足** | 中 | 仅测Muon一种谱优化器；缺少RMSNorm |
| **SCS因果方向未确立** | 高 | 论文仅建立SCS pattern与精度的相关性，未区分因果方向：是SCS→精度、共因、还是精度→SCS？缺少干预实验 |
| **SCS指标设计局限** | 中高 | (1) 绝对值\|⟨d̂,sᵢ⟩\|遮蔽对齐/反对齐方向，BN屏障靠间接推断而非直接测量；(2) k=10覆盖谱的2%（512×512层），选择缺乏论证，结论对k的鲁棒性未知 |
| **缺乏理论支撑** | 中 | BN谱屏障仅直觉几何解释，无形式化证明；未利用BN scale invariance这一现成理论工具 |
| **无actionable贡献** | 中 | SCS纯诊断工具，不指导设计改进；GN+Muon晚期反转暗示了可干预的方向但未探索 |

---

# 第二部分：优化方案

## 2.1 叙事优化

### 2.1.1 核心叙事重构：从"调制/屏障"到"传导路径的归一化依赖性"

**当前叙事（弱）：**
> "归一化调制了优化器的谱优势"——BN谱屏障是一种"阻碍"，Muon"突破"了它

**问题：** 
1. SGD+BN有最深屏障却有最高精度，反驳"屏障=坏事"
2. 交互检验(p=0.895)表明Muon的耦合提升在BN和LN下**完全相同**(+0.0035)，说明谱耦合本身不受归一化影响——不同的是**耦合转化为精度的效率**

**优化后叙事（精确版）：**

> "Muon的梯度正交化在所有归一化环境下提供**同等**的谱耦合提升（+0.0035，交互检验 p=0.895），但这个提升转化为精度改善的效率是**归一化依赖的**：LN/GN环境下高效传导（+5.49pp / +6.39pp），BN环境下被压缩（+0.89pp）。这揭示了BN的batch统计平滑作为性能保障机制（performance floor）的角色——它独立于谱耦合，为精度提供下限保障，使得优化器谱质量差异的影响被大幅衰减。"

**关键区分——之前方案的错误修正：**

之前曾有"BN反对齐 = 隐式谱正则化"的假设，暗示反对齐深度单调预测泛化。但BN下数据显示：

| 优化器 | SCS excess | 精度 | "反对齐→正则化"预测 |
|--------|-----------|------|-------------------|
| SGD | 最负 (-0.0067) | **最高** (78.05%) | ✓ 预测正确 |
| AdamW | 负 (-0.0009) | 最低 (76.90%) | ✗ **预测错误**——应比Muon好 |
| Muon | **正** (+0.0026) | 中间 (77.79%) | ✗ 不符合 |

AdamW比Muon更反对齐但精度更低，说明**反对齐深度不单调决定泛化**。正确的解释框架是：

> BN的batch统计平滑**压缩了精度差异的范围**（76.9-78.1%，仅1.2pp跨度），使得所有优化器都落入一个"performance floor"附近。这个compression effect独立于SCS的方向性——SCS揭示的是谱结构差异，而BN抹平了这些差异对精度的影响。LN/GN没有这种压缩效应，因此SCS差异（谱耦合质量）直接映射为精度差异。

### 2.1.1b 定量化工具：耦合传导率 η

上述叙事可通过一个定量指标锐化——**耦合传导率（Coupling Conductance）**：

> η = Δaccuracy / ΔSCS_excess

即单位SCS提升转化为多少精度提升。η 高的归一化环境意味着优化器谱质量的改进能高效传导为精度收益；η 低意味着被归一化"衰减"了。

| 归一化 | ΔSCS (Muon − AdamW) | Δaccuracy | η (pp/unit) |
|--------|---------------------|-----------|-------------|
| BN (ResNet-18) | +0.0035 | +0.89pp | **254** |
| LN (DeiT-Tiny) | +0.0035 | +5.49pp | **1569** |
| GN(1,C) (ResNet-18) | ~+0.0035 (估计) | +6.39pp | **~1826** |

η 本身可以成为归一化环境的分类工具：
- η ≫ 1000 → "透明型"归一化（LN, GN），值得投入谱优化器
- η ≪ 500 → "压缩型"归一化（BN），优化器选型收益有限

**GN连续谱实验的延伸价值：** 如果P0-A做完，可画"group数 vs η"的连续曲线。如果η随group数单调变化，这就从定性分类升级为定量预测工具——比"BN压缩/LN透明"的二元框架更有力。

### 2.1.2 SGD悖论的正面整合

**定位：** SGD+BN最高精度不是"悖论"，而是**SCS作为mechanistic（非predictive）工具的最佳论证**。

具体论述：
> "三个优化器在BN下达到几乎相同的精度（77-78%），但SCS揭示了它们在谱结构上截然不同的行为模式——SGD形成深层持续屏障，AdamW形成浅层衰减屏障，Muon维持正耦合。SCS的价值在于发现**精度无法区分的机制差异**，而非预测精度本身。这种mechanistic diversity在精度一致的表面下暗示了不同的泛化路径——一个仅凭test accuracy无法观察到的发现。"

### 2.1.3 增加Practical Implications小节

| 使用场景 | 建议 | 依据 | 置信度 |
|---------|------|------|--------|
| BN-based CNN | 优化器选型收益有限，SGD/AdamW即可 | BN提供performance floor（η≈254），精度对优化器谱质量不敏感 | 中——仅CIFAR-100/Tiny-IN小规模验证，ImageNet上SGD vs AdamW差距可能更大 |
| LN-based Transformer | 应优先考虑谱优化器（Muon/SOAP） | LN无performance floor（η≈1569），优化器谱质量直接决定精度 | 中高——DeiT-Tiny验证，需更大模型确认 |
| GN-based检测/分割模型 | 谱优化器收益最大 | GN放大优化器差异（η≈1826） | 中——仅分类任务验证，检测/分割可能不同 |
| 评估新归一化方案 | 用η（耦合传导率）评估其对优化器差异的传导特性 | η可定量度量归一化是"压缩型"还是"透明型" | 需GN连续谱实验确认η连续性 |

**⚠ 注意：** 上述建议均需标注"基于小规模实验的初步证据"，避免过度泛化。η的具体数值依赖于实验规模，跨规模的绝对值可能变化，但相对排序（BN ≪ LN ≈ GN）的鲁棒性更强。

### 2.1.4 建议的新论文结构

```
1. Introduction（重写：谱解耦视角 + 耦合传导率η + 实践意义）
2. Related Work（扩充：SOAP/Shampoo/RMSNorm + BOCB + BN scale invariance文献）
3. Method
   3.1 Spectral Coupling Score（保留 + 新增SCS_signed双指标）
   3.2 Permutation Null Baseline（保留）
   3.3 Coupling Conductance η（新增：定量化传导效率）
   3.4 Geometric Analysis: Scale Invariance and Anti-Alignment（新增：基于BN尺度不变性的几何论证）
4. Experiments
   4.1 Setup（扩充模型/数据/优化器）
   4.2 Main Results: Norm-Optimizer Accuracy（扩充新模型）
   4.3 Spectral Coupling Analysis（扩充 + SCS_signed结果）
   4.4 Same-Architecture Norm Factorial（新增核心实验）
   4.5 Normalization as Spectral Group Continuum（新增：GN多group连续谱 + η连续曲线）
   4.6 Causal Intervention: Does SCS Drive Accuracy?（因果干预实验）
   4.7 Per-Channel Coupling Decomposition（通道级分析验证理论）
   4.8 GN+Muon Late-Stage Reversal and Orthogonalization Scheduling
   4.9 Cross-Scale Validation（ImageNet-100/1K）
   4.10 Generalization to Other Spectral Optimizers（SOAP/Shampoo）
   4.11 Representation Analysis（验证BN performance floor机制）
   4.12 Ablations（扩充：含k敏感性、SCS变体对比）
5. Discussion
   5.1 BN as Spectral Decoupler: Performance Floor via Scale Invariance（新叙事核心 + 理论支撑）
   5.2 Coupling Conductance η as Normalization Taxonomy（η作为分类工具）
   5.3 SCS as Mechanistic (not Predictive) Diagnostic（正面解答SGD悖论）
   5.4 Practical Guidelines
   5.5 Comparison with Standard Spectral Metrics（已有E7，保留）
6. Limitations（更新）
7. Conclusion
```

---

## 2.2 实验优化方案

### 2.2.0 计算资源预算

| 参数 | 值 |
|------|-----|
| GPU数量 | 4 × RTX 5090 |
| 可用时间 | 96 小时（4天） |
| **总GPU-hours** | **384 GPU-hours** |
| 安全系数 | ×1.3（考虑调参、失败重跑、排队开销） |
| **有效GPU-hours** | **~295 GPU-hours** |

**训练时间估算（含1.3x安全系数）：**

| 任务 | 原始估算 | 安全估算 |
|------|---------|---------|
| CIFAR-100 ResNet-18 (200ep) | 20 min/run | **26 min/run** |
| CIFAR-100 DeiT-Tiny (200ep) | 45 min/run | **58 min/run** |
| CIFAR-100 ViT-Small (200ep) | 60 min/run | **90 min/run**（ViT-S 参数量约22M，远大于DeiT-T的5M） |
| CIFAR-100 ConvNeXt-T (200ep) | 40 min/run | **52 min/run** |
| Tiny-ImageNet ResNet-18 (200ep) | 60 min/run | **78 min/run** |
| ImageNet-100 ResNet-18 (200ep) | 2.5 h/run | **3.3 h/run** |
| ImageNet-100 ResNet-50 (200ep) | 4 h/run | **5.2 h/run** |
| ImageNet-1K ResNet-50 (90ep) | 8 h/run | **12 h/run** |
| SCS计算 per checkpoint | 3 min | CPU并行，不占GPU |

### 2.2.1 实验设计的关键修正

**修正1：去除"LN on ResNet-18"——已被GN(1,C)覆盖**

论文明确指出GN(1,C) ≡ per-sample LN over channels（第220-221行）。ResNet-18 + GN(1,C) 的结果已在Table 3中。标准 `nn.LayerNorm([C,H,W])` 归一化所有维度（C+H+W），与GN(1,C)是不同操作，但其SCS方向向量的定义需要从头构建（当前论文未定义），且在CNN上的语义不清晰。**放弃此条件**，改为GN多group连续谱实验（更有价值）。

**修正2：ViT-Small + BN 标记为高风险**

BN在Transformer中的语义不明确（ViT处理 batch × seq_len × hidden 的序列，BN沿batch归一化每个position-feature对，需足够大batch且位置语义被混淆）。此条件可能训练不稳定。Plan B是放弃此格子并在论文中讨论BN不适用于Transformer的原因（本身是有价值的observation）。

**修正3：提升ConvNeXt-T优先级**

ConvNeXt-T是CNN架构但天然使用LN，是"架构-归一化解耦"的理想测试对象——如果BN→LN的替换在ConvNeXt上也展现类似的SCS pattern变化，则进一步支持归一化是主因而非架构。从P2提升至P1。

**修正4：表征分析指标多元化**

论文已发现有效秩在优化器之间无区分度（§5）。P1-C不能把赌注全押在有效秩上，需要多元化指标组合，并明确区分"跨优化器对比"和"跨归一化对比"两个分析维度。

**修正5：SCS_signed提升为P0级主分析双指标，k敏感性提升至P0**

SCS_signed和k敏感性是SCS指标的**根基验证**，如果它们失败，整篇论文的claim都需要修正。应在第一时间验证根基，再构建上层实验。两者均为零GPU成本。

**修正6：新增epoch-0 SCS基线**

论文从未检查训练前的SCS。这个对照区分了"训练动态产物"vs"初始化固有"两种截然不同的解释。零GPU成本，高信息价值。

**修正7：新增因果干预实验（P1-F）**

当前所有分析都是相关性层面。因果方向不明是§1.3中"高"严重度弱点。P1-F通过正交旋转权重矩阵强制改变SCS，观察训练动态和精度的响应。仅2.8 GPU-hours，可能是论文最大的档次提升。

**修正8：新增Per-channel SCS分解（P1-G）**

BN是per-channel操作，SCS是per-layer聚合，缺少关键分析粒度。P1-G按channel方差分组分析SCS贡献，直接验证scale invariance理论猜想的核心机制。零GPU成本。

**修正9：GN反转从"调查"升级为核心分析**

除延长训练（300/400ep）确认趋势外，新增层大小分析（反转是否只在大层出现？）、精度关联（反转是否影响精度？）、正交化衰减schedule的连接。这可能成为论文最有深度的独立发现。

**修正10：正交化衰减schedule替代Norm-Aware Muon**

原Norm-Aware Muon方向有问题——BN下反对齐可能是有益的（SGD最反对齐但精度最高），"保留归一化方向"可能反而降低精度。改为针对GN+Muon晚期反转的正交化衰减——动机更明确（解决已观察到的问题）、验证标准更清晰（反转是否被阻止？精度是否维持？）。

**修正11：Norm-Aware Muon伪代码修正（降级为备选方案）**

BN方向d ∈ R^m在输出空间，对应**左**奇异向量；LN方向d ∈ R^n在输入空间，对应**右**奇异向量。投影方向不同：

```python
# ---- BN-type (d在输出空间, m维) ----
# G: (m, n), d_hat: (m,) 
coeff = d_hat @ G                               # (n,) 
G_parallel = d_hat.unsqueeze(1) * coeff.unsqueeze(0)  # (m, n)
G_perp = G - G_parallel
update = G_parallel + newton_schulz(G_perp)

# ---- LN-type (d在输入空间, n维) ----
# G: (m, n), d_hat: (n,)
coeff = G @ d_hat                               # (m,)
G_parallel = coeff.unsqueeze(1) * d_hat.unsqueeze(0)  # (m, n)
G_perp = G - G_parallel
update = G_parallel + newton_schulz(G_perp)
```

---

### 2.2.2 全部实验按优先级展开

#### P0：消除架构混淆 + 增强统计效力（最高优先级）

**P0-A0: GN学习率公平性补救——SGD+GN(1,C) LR sweep（最高优先级，阻塞性）**

论文承认SGD+GN(1,C)使用了BN最优LR=0.01，未做GN专属LR搜索。这直接威胁Table 3中GN精度差距的可信度。**必须在所有GN相关实验之前完成。**

| 序号 | 条件 | LR范围 | Seeds | 单run | 总时间 | 目的 |
|------|------|--------|-------|------|--------|------|
| P0-A0 | ResNet-18 + GN(1,C) + SGD | {0.003, 0.01, 0.03, 0.1, 0.3} | 1 | 26min | 2.2h | 确认GN最优LR |

如果最优LR ≠ 0.01，需用新LR重跑SGD+GN的正式5-seed实验（额外2.2h）。此结果直接影响P0-A以及论文Table 3的结论。

小计：**2.2h**（最坏4.4h）

---

**P0-A: ResNet-18 归一化扩展（CIFAR-100）——补RMSNorm + GN多group连续谱**

已有（直接利用）：
- ResNet-18 + BN × {Muon, AdamW, SGD} × 3 seeds ✅
- ResNet-18 + GN(1,C) × {Muon, AdamW, SGD} × 3-4 seeds ✅（≡ LN over channels）

需补充：

| 序号 | 归一化 | 优化器 | Seeds | 单run | 总时间 | 目的 |
|------|--------|--------|-------|------|--------|------|
| P0-A1 | **RMSNorm** | Muon, AdamW, SGD | 5 | 26min | 6.5h | LLM主流归一化 |
| P0-A2 | **GN(G=8)** | Muon, AdamW, SGD | 5 | 26min | 6.5h | group连续谱 |
| P0-A3 | **GN(G=32)** | Muon, AdamW, SGD | 5 | 26min | 6.5h | group连续谱 |
| P0-A4 | **InstanceNorm(G=C)** | Muon, AdamW, SGD | 5 | 26min | 6.5h | group连续谱极端 |

小计：60 runs × 26min = **26 GPU-hours**

**关键产出：** ResNet-18上 {BN, GN(1), GN(8), GN(32), IN(G=C), RMSNorm} × 3 opts 的完整连续谱。这比简单添加LN更有说服力——可以画出"Group数 vs SCS"的连续变化图，直观展示归一化粒度对谱耦合的系统性影响。

**⚠ RMSNorm SCS方向向量注意事项：** RMSNorm无mean centering，其方向向量 d^RMS = (1/RMS(x))·**1** 为近似均匀向量（所有分量相等）。论文当前未定义此情形下的SCS direction vector。需在实现前明确：(1) 在Method §3中补充RMSNorm方向向量的定义；(2) 均匀方向向量可能导致SCS信号极弱（与所有奇异向量几乎等投影）——如果实验显示RMSNorm条件下SCS excess ≈ 0，这本身是有价值的finding，说明RMSNorm的"方向中立性"。

**P0-B: DeiT-Tiny 归一化扩展（CIFAR-100）——在Transformer上验证**

已有：DeiT-Tiny + LN × {Muon, AdamW} × 3 seeds ✅，SGD+LN已知failure ✅

需补充：

| 序号 | 归一化 | 优化器 | Seeds | 单run | 总时间 | 说明 |
|------|--------|--------|-------|------|--------|------|
| P0-B1 | **RMSNorm** | Muon, AdamW | 5 | 58min | 9.7h | LN的简化版（去mean centering） |
| P0-B2 | **GN(1,C)** | Muon, AdamW | 5 | 58min | 9.7h | 等效LN但不同实现 |
| P0-B3 | **BN** | Muon, AdamW | 2 | 58min | 3.9h | **高风险**——BN在Transformer可能不稳定，探索性仅2 seeds |

注意：SGD + DeiT 已知failure（<13%），不做SGD变体。BN条件标为高风险，如不收敛则放弃并作为negative result讨论。P0-B3仅2 seeds作为可行性探测，收敛再补seeds。

小计：24 runs × 58min = **~23 GPU-hours**（P0-B3从5 seeds减至2 seeds，节省5.8h）

**P0-C: 已有条件补充seeds至5**

| 条件 | 现有seeds | 补充 | runs | 时间 |
|------|----------|------|------|------|
| ResNet-18+BN × {Muon,AdamW,SGD} | 3 | +2 | 6 | 2.6h |
| ResNet-18+GN(1,C) × {Muon,AdamW,SGD} | 3-4 | +1~2 | 6 | 2.6h |
| DeiT-Tiny+LN × {Muon,AdamW} | 3 | +2 | 4 | 3.9h |
| DeiT-Tiny+LN 轨迹分析（multi-seed） | 1 (seed=42) | +4 | 4 | 3.9h |

最后一行特别重要：**LN/DeiT轨迹目前只有单seed**（Appendix E），这是论文的重大统计弱点。

小计：20 runs → **~13 GPU-hours**

**P0-D: SCS指标根基验证——零GPU成本的关键补充（最高优先级）**

这组分析全部基于已有checkpoint的CPU重算，零GPU成本，但直接决定SCS指标的可信度。**必须在论文提交前完成，否则审稿人会质疑整个指标的根基。**

**P0-D1: SCS_signed作为主分析双指标**

当前SCS使用 |⟨d̂, sᵢ⟩|（绝对值），对齐和反对齐在单个奇异向量层面不可区分。BN spectral barrier是通过"SCS低于null baseline"间接推断的。SCS_signed = Σ (σᵢ/Σσ) × ⟨d̂, sᵢ⟩（去掉绝对值）可以直接区分：

- SCS_abs < null 且 SCS_signed 显著为负 → **直接证明**主动反对齐（而非随机低耦合）
- SCS_abs < null 但 SCS_signed ≈ 0 → 仅为随机低耦合，"barrier"叙事需修正

| 分析 | 条件 | 计算量 | 产出 |
|------|------|--------|------|
| SCS_signed重算 | 所有已有checkpoint（~72 runs × 4 epochs） | ~4 CPU-hours | 所有条件的signed excess + z-scores |
| 双指标对比表 | — | 分析时间 | Table: SCS_abs vs SCS_signed per condition |

**论文呈现：** 在Method §3.1中引入SCS_abs和SCS_signed两种定义，讨论差异。在§4.3中同时报告两种指标，用SCS_signed直接论证BN反对齐。

**P0-D2: k敏感性验证（阻塞性）**

论文所有结论建立在k=10上。如果k=20或k=50时结论翻转，整篇论文就塌了。

| 分析 | k值 | 条件 | 计算量 |
|------|-----|------|--------|
| k敏感性 | {1, 3, 5, 10, 20, 50} | 核心4条件（BN/LN × Muon/AdamW）× 3 seeds × epoch 200 | ~3 CPU-hours |

**关键判断：**
- BN屏障方向（SCS excess正/负）对k鲁棒 → 强化结论
- 某个k范围内方向翻转 → 必须在论文中讨论并解释k选择的依据
- 如果k=1时结论最清晰 → 可能SCS应默认使用top-1 singular vector

**P0-D3: 随机初始化SCS基线（epoch 0）**

论文从未检查训练前的SCS。这个对照区分了两种截然不同的解释：

- 如果epoch 0时所有条件SCS ≈ null → BN barrier是**训练动态的产物**
- 如果epoch 0时BN条件已展现反对齐 → barrier是**初始化几何固有的**，训练仅维持/加深

| 分析 | 条件 | 计算量 |
|------|------|--------|
| Epoch 0 SCS | 所有已有条件（使用训练前的初始化权重重新计算） | ~2 CPU-hours |
| Epoch 1 SCS（可选） | 1个batch更新后的SCS，看barrier形成速度 | ~1 CPU-hour |

**⚠ 这是最高性价比的分析**——零成本但能区分两种完全不同的因果机制。无论结果如何都有价值。

小计：**~10 CPU-hours，零GPU**

**P0总计：~64 GPU-hours + 10 CPU-hours**（含P0-A0 2.2h，P0-B3减至2 seeds节省5.8h，P0-D零GPU）

---

#### P1：新架构验证 + SOAP + 因果干预 + 表征分析 + GN反转深化（强烈建议）

**P1-A: ConvNeXt-Tiny同架构factorial（CIFAR-100）——最安全的桥接架构**

ConvNeXt是CNN但天然使用LN，不存在ViT+BN的兼容性风险。在ConvNeXt上替换LN为BN/GN，比在ViT上做更可靠。

| 归一化 | 优化器 | Seeds | 单run | 总时间 |
|--------|--------|-------|------|--------|
| LN (原生) | Muon, AdamW, SGD | 5 | 52min | 13h |
| BN (替换) | Muon, AdamW, SGD | 5 | 52min | 13h |
| GN(1,C) | Muon, AdamW, SGD | 5 | 52min | 13h |
| RMSNorm | Muon, AdamW, SGD | 5 | 52min | 13h |

含LR grid search（4 norms × 3 opts × 4 LR × 50ep ≈ 48 runs × 13min = 10.4h）

小计：**~62 GPU-hours**

**价值：** 如果ConvNeXt上也出现"BN压缩精度差异 + LN放大精度差异"的pattern，则在CNN架构内就隔离了归一化效应，不再需要依赖DeiT vs ResNet的跨架构对比。

**P1-B: ViT-Small factorial（CIFAR-100）——Transformer端验证（精简版）**

ConvNeXt-T (P1-A) 已是主力桥接架构。ViT-Small定位为Transformer端的**辅助验证**，不需全覆盖。GN(1,C)在ViT上与LN高度相似（ViT无空间维度可对比），故略去。

| 归一化 | 优化器 | Seeds | 单run | 总时间 | 风险 |
|--------|--------|-------|------|--------|------|
| LN (原生) | Muon, AdamW, SGD | 3 | 90min | 13.5h | 低 |
| RMSNorm | Muon, AdamW, SGD | 3 | 90min | 13.5h | 低 |
| **BN** | **Muon, AdamW** | **2** | **90min** | **6h** | **高——可能不收敛** |

BN条件仅2 seeds × 2 opts探测可行性。SGD+ViT-Small如果失败则记录为negative result。

含LR grid search：2 norms × 3 opts × 4 LR × 50ep + BN × 2 opts × 4 LR = 32 runs × 23min ≈ **~12h**

小计：**~45 GPU-hours**

**P1-C: SOAP优化器（CIFAR-100）——验证谱优化器普遍性**

SOAP (Vyas et al., NeurIPS 2024) 是另一种谱预条件化优化器。如果SOAP也展现类似Muon的耦合pattern，论文发现可从"Muon特有"泛化为"谱优化器的普遍性质"。

| 架构 | 归一化 | Seeds | 总时间 |
|------|--------|-------|--------|
| ResNet-18 | BN, GN(1,C), RMSNorm | 3 | 9 runs × 26min = 3.9h |
| DeiT-Tiny | LN, RMSNorm | 3 | 6 runs × 58min = 5.8h |
| ConvNeXt-T | LN, BN | 3 | 6 runs × 52min = 5.2h |

含grid search：~8h

小计：**~23 GPU-hours**

**P1-D: 表征分析——验证BN performance floor机制（CPU，不占GPU）**

**关键修正：** 论文§5已发现有效秩在**优化器之间**无区分度。P1-D的分析应聚焦两个维度：
1. **跨归一化对比**（BN vs GN vs RMSNorm）——有效秩等指标可能有区分度
2. **跨优化器对比**——需要更敏感的指标

| 指标 | 计算方式 | 分析维度 | 预期结果 |
|------|---------|---------|---------|
| **有效秩** | exp(H(σ/Σσ)) per layer | 跨归一化 | BN条件下可能更高（尚未验证） |
| **stable rank** | ‖W‖²_F / ‖W‖²_op | 跨归一化 | BN条件下可能更高 |
| **feature decorrelation** | 特征协方差矩阵off-diagonal Frobenius范数 | 跨归一化+跨优化器 | BN条件下off-diagonal更小（BN促进去相关） |
| **nuclear norm ratio** | ‖W‖_* / (‖W‖_F × √min(m,n)) | 跨优化器 | Muon可能更高（更均衡的奇异值分布） |
| **泛化gap** | train acc - test acc | 两者 | BN条件下gap更小 |
| **精度方差 across optimizers** | std(acc) over 3 optimizers | 跨归一化 | BN下更小（performance floor） |

最后一个指标最直接——如果BN下三个优化器精度方差确实比LN下小，这就是"performance floor"最直观的证据。**可以从已有训练log直接提取，零成本。**

**CPU总计：~8 CPU-hours**，完全不占GPU。

**P1-E: GN+Muon晚期反转深化研究（CIFAR-100，核心分析）**

Appendix G报告GN(1,C)+Muon的SCS在epoch 200反转为负（4 seeds, t=-5.81, p=0.010），由layer4.1.conv2（512×512, 21%参数）主导。**这可能是整篇论文最有深度的发现**——Muon的正交化在训练前期有利（建立谱耦合），但后期反而有害（谱退化）。需作为核心分析维度展开。

**P1-E1: 延长训练确认趋势**

| 条件 | Epochs | Seeds | 单run | 总时间 |
|------|--------|-------|------|--------|
| ResNet-18 + GN(1,C) + Muon | 300 | 3 | 33min | 1.7h |
| ResNet-18 + GN(1,C) + Muon | 400 | 3 | 44min | 2.2h |

**P1-E2: 反转 vs 层大小分析（零GPU，已有checkpoint）**

| 分析 | 方法 | 计算量 |
|------|------|--------|
| 所有层SCS轨迹热图 | 对ResNet-18所有conv层分别追踪per-layer SCS × epoch | ~2 CPU-hours |
| 层参数量 vs 反转epoch | 散点图：每层参数量 vs 首次SCS < 0的epoch | 分析时间 |

**预期：** 如果反转只在大层（layer4, 512×512）出现而小层（layer1, 64×64）不反转 → 反转是过参数化效应，Muon的正交化在大层上"过度修正"了已适应的谱结构。

**P1-E3: 反转 vs 精度关联（零GPU，已有训练log）**

GN+Muon的精度在epoch 200是否也有下降？从已有log提取epoch {100, 150, 200, 300, 400}的test accuracy：
- 如果SCS反转但精度稳定 → 反转是无害的谱重组
- 如果精度也同步下降 → 反转有实际影响，可引出actionable建议

**P1-E4: 正交化衰减schedule（见P4-A）**

小计：**~3.9 GPU-hours + 2 CPU-hours**

---

**P1-F: 因果干预实验——SCS是否驱动精度？（论文档次关键提升）**

**动机：** 论文当前只建立了SCS与精度的相关性。因果方向不明是高严重度弱点。一个简单的干预实验可以将论文从"correlation"升级为"causal evidence"。

**实验设计：**

1. 从epoch 100取一个BN+AdamW的checkpoint（此时SCS为负，屏障存在）
2. 对权重矩阵W做正交旋转：W' = R·W，其中R选择使W'的top-k左奇异向量与d^BN对齐（强制提升SCS）
3. 从W'继续训练到epoch 200
4. 对照组：从相同checkpoint不做旋转，继续训练到epoch 200

**观察指标：**
- SCS轨迹：干预后SCS是否被训练动态恢复到原来的负值？
- 精度变化：干预后精度是否改善/下降？

**两种可能的结果及其意义：**

| 结果 | SCS轨迹 | 精度 | 解释 | 论文价值 |
|------|---------|------|------|---------|
| A | 干预后SCS迅速恢复到负值 | 无显著变化 | BN训练动态**主动维护**反对齐，SCS是动态平衡的产物而非因果驱动因素 | 强力支持"mechanistic diagnostic"叙事 |
| B | 干预后SCS保持高位 | 精度改善 | SCS有因果效力，但BN阻止模型自发达到高SCS | 支持更强的claim："SCS不仅诊断，还预测干预效果" |
| C | 干预后SCS保持高位 | 精度下降 | 高SCS在BN下反而有害——BN反对齐是有益的隐式正则化 | 回到"反对齐=正则化"叙事（但有因果证据支撑） |

**三种结果都有高价值**——最坏情况是修正叙事（结果C），最好情况是建立因果链（结果B）。

| 条件 | 操作 | Seeds | 单run | 总时间 |
|------|------|-------|------|--------|
| BN+AdamW，从epoch 100旋转 | 强制SCS提升 + 继续训练100ep | 3 | 13min | 0.7h |
| BN+AdamW，从epoch 100对照 | 不旋转 + 继续训练100ep | 3 | 13min | 0.7h |
| BN+Muon，从epoch 100旋转 | 强制SCS降低（反向旋转） + 继续训练100ep | 3 | 13min | 0.7h |
| GN+AdamW，从epoch 100旋转 | 强制SCS提升 + 继续训练100ep | 3 | 13min | 0.7h |

小计：**~2.8 GPU-hours**（极低成本，极高回报）

**⚠ 实现注意：** 旋转矩阵R的构造需要仔细设计——不能破坏权重矩阵的scale（否则引入新的混淆因素）。建议使用Givens旋转只调整top-k奇异向量方向，保持奇异值不变。

---

**P1-G: Per-Channel SCS分解（零GPU）**

BN是per-channel操作，但SCS是per-layer聚合。当前的per-layer分解（Appendix H）已有价值，但还少了一层粒度：**per-channel分解**。

**分析目标：** 直接验证理论猜想的核心机制——"BN方向集中在低方差channel，奇异向量集中在高方差channel"。

**方法：** 对关键层（layer4.1.conv2等），将d^BN的channel按方差分组：

| 分组 | Channel定义 | 分析 |
|------|-----------|------|
| 高方差组 | σ²_c > median(σ²) 的channel | d^BN在这些channel上权重低；top奇异向量在这些channel上投影高？ |
| 低方差组 | σ²_c < median(σ²) 的channel | d^BN在这些channel上权重高；top奇异向量在这些channel上投影低？ |

**具体计算：**
1. 对每个channel c，计算其对SCS的贡献：contribution_c = (σ₁/Σσ) × |d̂_c × u₁_c|
2. 按channel方差排序，画"channel方差 vs SCS贡献"的散点图
3. 比较Muon vs AdamW vs SGD在高/低方差channel组的贡献差异

**预期结果：**
- BN下，SGD/AdamW的奇异向量能量集中在高方差channel → 与d^BN（集中在低方差channel）正交 → 反对齐
- Muon的Newton-Schulz正交化均衡了奇异值 → 能量更均匀分布 → 减轻反对齐
- GN下，方向向量由learned γ决定（非方差驱动）→ 无此结构性反对齐

小计：**~4 CPU-hours，零GPU**

---

**P1总计：~143 GPU-hours + 14 CPU-hours**

---

#### P2：规模扩展 + Tiny-ImageNet补充（建议）

**P2-A: ImageNet-100验证**

| 架构 | 归一化 | 优化器 | Seeds | 单run | 总时间 |
|------|--------|--------|-------|------|--------|
| ResNet-18 | BN | Muon, AdamW, SGD | 3 | 3.3h | 29.7h |
| ResNet-18 | GN(1,C) | Muon, AdamW, SGD | 3 | 3.3h | 29.7h |
| ResNet-18 | RMSNorm | Muon, AdamW | 3 | 3.3h | 19.8h |
| ResNet-50 | BN | Muon, AdamW | 3 | 5.2h | 31.2h |

含grid search：~15h

小计：**~126 GPU-hours**

**P2-B: Tiny-ImageNet补充**

| 条件 | 补充内容 | 时间 |
|------|---------|------|
| 现有6条件 +1 seed | 6 runs × 78min | 7.8h |
| RMSNorm × {Muon, AdamW} × 3 seeds | 6 runs × 78min | 7.8h |

小计：**~16 GPU-hours**

**P2-C: 训练配方消融（Training Recipe Ablation）**

跨架构精度对比（ResNet-18 vs DeiT-Tiny）受训练配方差异干扰——ResNet使用standard augmentation + step schedule，DeiT使用RandAugment + Mixup + cosine schedule。需验证recipe差异不是6倍不对称的主因。

| 条件 | 配方 | Seeds | 单run | 总时间 |
|------|------|-------|------|--------|
| ResNet-18 + BN + Muon | DeiT-style强增强 | 3 | 26min | 1.3h |
| ResNet-18 + GN(1,C) + Muon | DeiT-style强增强 | 3 | 26min | 1.3h |

**判断逻辑：** 在强增强recipe下，如果BN vs GN的精度跨度（across optimizers）仍然保持~1pp vs ~6pp的比例，则recipe confound可排除。如果跨度显著收窄，需在论文中作为covariate讨论。

小计：**~2.6 GPU-hours**

**P2总计：~145 GPU-hours**

---

#### P3：消融 + ImageNet-1K + 更多优化器（加分项）

**P3-A: SCS指标鲁棒性消融**

已有消融（直接利用，不需重做）：
- ✅ 权重衰减消融（Appendix C）
- ✅ Per-layer SCS分解（Appendix H）
- ✅ 标准谱指标对比（§5）

需补充（注：top-k敏感性已提升至P0-D2，SCS变体对比中SCS_signed已提升至P0-D1）：

| 消融变量 | 设置 | 资源 | 时间 |
|---------|------|------|------|
| 置换次数N | N ∈ {100,500,1000,5000,10000} | CPU | 5h |
| Batch size | BS ∈ {32,64,128,256,512} | GPU | 15 runs × 26min = 6.5h |
| LR schedule | cosine/linear/step | GPU | 9 runs × 26min = 3.9h |
| 模型宽度 | width ×{0.25,0.5,1,2} | GPU | 12 runs × 20min = 4h |

小计：**~14.5 GPU-hours + 5 CPU-hours**

**P3-B: ImageNet-1K（如有时间）**

| 条件 | Seeds | 总时间 |
|------|-------|--------|
| ResNet-50+BN × {Muon, AdamW} | 1 | 24h |
| ResNet-50+GN × Muon | 1 | 12h |

小计：**~36 GPU-hours**

**P3-C: Shampoo + Lion**

| 优化器 | 架构×归一化 | Seeds | 总时间 |
|--------|-----------|-------|--------|
| Shampoo | ResNet-18 × {BN, GN(1,C)}, DeiT-Tiny × LN | 3 | ~10h |
| Lion | ResNet-18 × {BN, GN(1,C)}, DeiT-Tiny × LN | 3 | ~8h |

小计（含grid search）：**~24 GPU-hours**

**P3-D: ResNet深度扩展**

| 架构 | 归一化 | 优化器 | Seeds | 总时间 |
|------|--------|--------|-------|--------|
| ResNet-34 | BN | Muon, AdamW | 3 | 6 runs × 33min = 3.3h |
| ResNet-50 | BN | Muon, AdamW | 3 | 6 runs × 39min = 3.9h |
| ResNet-101 | BN | Muon, AdamW | 3 | 6 runs × 58min = 5.8h |

小计：**~13 GPU-hours**

**P3总计：~88 GPU-hours + 8 CPU-hours**

---

#### P4：正交化衰减Schedule + SCS变体 + Norm-Aware Muon备选（探索性）

**P4-A: 正交化衰减Schedule——针对GN+Muon晚期反转的actionable干预**

**设计理由：** 原方案中的Norm-Aware Muon（保留归一化方向分量不做正交化）设计方向有问题——BN下反对齐可能是有益的（SGD最反对齐但精度最高），"保留归一化方向"可能反而降低精度。更合理的干预目标是**GN+Muon的晚期反转**——这是一个已观察到的具体问题，有明确的动机和验证标准。Norm-Aware Muon降级为备选方案。

**核心思路：** 在训练后期逐渐减弱正交化强度，防止Newton-Schulz"过度修正"已经适应的谱结构。

```python
def scheduled_orthogonalization(G, epoch, warmdown_start=100, max_epoch=400):
    """渐进式正交化衰减：epoch < warmdown_start时完全正交化，之后线性衰减"""
    if epoch < warmdown_start:
        alpha = 1.0
    else:
        alpha = 1.0 - (epoch - warmdown_start) / (max_epoch - warmdown_start)
        alpha = max(alpha, 0.0)
    
    G_orth = newton_schulz(G)
    return alpha * G_orth + (1 - alpha) * G
```

**实验设计：**

| 阶段 | 条件 | warmdown_start | Seeds | 单run | 总时间 |
|------|------|----------------|-------|------|--------|
| 快速验证 | ResNet-18 + GN(1,C) + Muon, 400ep | {100, 150, 200} | 3 | 44min | 6.6h |
| 对照 | ResNet-18 + GN(1,C) + Muon, 400ep（标准Muon） | — | 3 | 44min | 2.2h |

**判断标准：**
- 如果衰减schedule阻止了SCS反转**且**维持/提升精度 → **actionable贡献**，从observational论文升级为有方法贡献的论文
- 如果SCS反转被阻止但精度无变化 → 反转确实是无害的谱重组，但schedule仍有诊断价值
- 如果无效 → 附录中讨论

**论文价值：** 如果成功，形成完整闭环——"SCS诊断问题（晚期反转）→ 提出干预（正交化衰减）→ 验证效果"。这是审稿人最喜欢的"诊断→治疗"叙事，直接解决"无actionable贡献"的弱点。

小计：**~8.8 GPU-hours**

**P4-A-backup: Norm-Aware Muon（降级为备选方案）**

如果P4-A的正交化衰减无效，可尝试原方案的Norm-Aware Muon。但注意：在BN下"保留归一化方向"可能有害，应主要在GN/LN下测试。

```python
# 备选方案伪代码，仅在P4-A失败后考虑
# ---- BN-type (d在输出空间, m维) ----
def norm_aware_muon_bn(G, d_norm):
    d_hat = d_norm / d_norm.norm()
    coeff = d_hat @ G                                # (n,)
    G_parallel = d_hat.unsqueeze(1) * coeff.unsqueeze(0)  # (m,n)
    G_perp = G - G_parallel
    return G_parallel + newton_schulz(G_perp)
```

小计：如P4-A失败后启用，**~7 GPU-hours**

**P4-B: SCS变体对比（CPU，不占GPU）**

| 变体 | 公式 | 目的 |
|------|------|------|
| SCS_squared | Σ (σᵢ/Σσ) × ⟨d̂,sᵢ⟩² | 强调极端对齐，对大/小角度更敏感 |
| SCS_unweighted | (1/k)Σ \|⟨d̂,sᵢ⟩\| | 消除奇异值加权影响，检查权重方案的role |

注：SCS_abs（现有）和SCS_signed（新增）已在P0-D1中作为主分析指标处理，此处仅测试辅助变体。

小计：**2 CPU-hours**

**P4总计：~8.8 GPU-hours + 2 CPU-hours**

---

### 2.2.3 资源汇总与可行性评估

| 优先级 | GPU-hours | CPU-hours | 新增runs | 核心价值 |
|--------|-----------|-----------|---------|---------|
| **P0** | 64 | **10** | ~115 | 消除架构混淆 + 统计效力 + **GN LR公平性** + **SCS根基验证（SCS_signed/k敏感性/epoch-0基线）** |
| **P1** | **143** | **14** | ~112 | 新架构 + SOAP + **因果干预实验** + 表征分析 + **GN反转深化 + per-channel分解** |
| **P2** | 145 | 0 | ~50 | 规模扩展 + **训练配方消融** |
| **P3** | 88 | **5** | ~66 | 消融 + ImageNet-1K + 更多优化器 |
| **P4** | **8.8** | **2** | ~15 | **正交化衰减schedule** + SCS变体 |
| **合计** | **~449** | **~31** | **~358** | — |

**可行性评估：**
- 有效预算295 GPU-hours → **P0 + P1全量 + P2大部分 = 64 + 143 + 88 = 295h**
- P0-D（SCS根基验证）和P1-G（per-channel分解）全部在CPU上，完全不影响GPU调度
- P1-F（因果干预）仅2.8 GPU-hours但可能是论文最大的档次提升
- P0-A0（GN LR sweep）在Day 1第一时段完成，结果决定后续GN实验是否需要调整LR
- P2剩余 + P3 + P4 作为"空闲填充"
- 最坏情况：至少P0+P1可以完成 → 论文核心弱点被消除 + SCS根基被验证 + 因果证据建立
- CPU任务全程与GPU并行执行

---

### 2.2.4 96小时饱和执行计划（4卡并行）

```
==========================================================================
                  96小时饱和执行计划（4 × RTX 5090）
==========================================================================

=== 第1天（0-24h）：P0 全部完成 + P1 启动 ===

【0-2.5h】P0-A0 阻塞性LR sweep（最高优先级）
┌──────────┬────────────────────────────────────────────────────────┐
│ GPU 0    │ **P0-A0: GN(1,C)+SGD LR sweep** (5 runs × 26min=2.2h)│
│          │ ★ 结果决定后续GN实验的SGD学习率                         │
├──────────┼────────────────────────────────────────────────────────┤
│ GPU 1-3  │ P0-A1/A2/B1 提前启动（不依赖P0-A0结果的条件）          │
├──────────┼────────────────────────────────────────────────────────┤
│ CPU      │ P1-D 表征分析启动 + P4-B SCS变体对比                    │
└──────────┴────────────────────────────────────────────────────────┘

【2.5-8h】P0-A + P0-B 主体
┌──────────┬────────────────────────────────────────────────────────┐
│ GPU 0    │ P0-A1: ResNet-18 + RMSNorm × {Muon,AdamW,SGD} × 5s   │
│          │ (15 runs × 26min = 6.5h)                              │
├──────────┼────────────────────────────────────────────────────────┤
│ GPU 1    │ P0-A2: ResNet-18 + GN(8) × {Muon,AdamW,SGD} × 5s     │
│          │ (15 runs × 26min = 6.5h)                              │
├──────────┼────────────────────────────────────────────────────────┤
│ GPU 2    │ P0-B1: DeiT-Tiny + RMSNorm × {Muon,AdamW} × 5s       │
│          │ (10 runs × 58min = 9.7h) → 溢出到下一时段              │
├──────────┼────────────────────────────────────────────────────────┤
│ GPU 3    │ P0-B2: DeiT-Tiny + GN(1,C) × {Muon,AdamW} × 5s       │
│          │ (10 runs × 58min = 9.7h) → 溢出到下一时段              │
├──────────┼────────────────────────────────────────────────────────┤
│ CPU      │ P1-D 表征分析：有效秩/stable rank/feature decorrelation│
│          │ P4-B: SCS变体对比（已有checkpoint）                      │
│          │ 写作：叙事重构 Introduction + Discussion 草稿           │
└──────────┴────────────────────────────────────────────────────────┘

【8-16h】P0-A续 + P0-C
┌──────────┬────────────────────────────────────────────────────────┐
│ GPU 0    │ P0-A3: ResNet-18 + GN(32) × {Muon,AdamW,SGD} × 5s    │
│          │ (15 runs × 26min = 6.5h)                              │
├──────────┼────────────────────────────────────────────────────────┤
│ GPU 1    │ P0-A4: ResNet-18 + IN(G=C) × {Muon,AdamW,SGD} × 5s   │
│          │ (15 runs × 26min = 6.5h)                              │
├──────────┼────────────────────────────────────────────────────────┤
│ GPU 2    │ P0-B1 续完成 → P0-C: ResNet-18+BN 补2 seeds (2.6h)    │
│          │ → P0-C: ResNet-18+GN(1,C) 补seeds (2.6h)              │
├──────────┼────────────────────────────────────────────────────────┤
│ GPU 3    │ P0-B2 续完成 → P0-C: DeiT-Tiny+LN 补2 seeds (3.9h)   │
│          │ → P0-C: DeiT-Tiny+LN 轨迹multi-seed (3.9h)            │
├──────────┼────────────────────────────────────────────────────────┤
│ CPU      │ SCS计算：P0-A1完成的checkpoint                          │
│          │ P1-D 表征分析续：CKA/Neural Collapse                    │
│          │ 精度方差分析（从已有log直接提取）                        │
└──────────┴────────────────────────────────────────────────────────┘

【16-24h】P0完成 → P1启动
┌──────────┬────────────────────────────────────────────────────────┐
│ GPU 0    │ P0-B3: DeiT-Tiny+BN × {Muon,AdamW} × 2s（高风险）     │
│          │ (4 runs × 58min = 3.9h)                               │
│          │ → P1-E: GN+Muon反转调查 300ep+400ep×3s (3.9h)         │
├──────────┼────────────────────────────────────────────────────────┤
│ GPU 1    │ P1-A LR grid search: ConvNeXt-T                       │
│          │ 4 norms × 3 opts × 4 LR × 50ep (48 runs × 13min=10h) │
├──────────┼────────────────────────────────────────────────────────┤
│ GPU 2    │ P1-B LR grid search: ViT-Small                        │
│          │ 2 norms × 3 opts × 4 LR + BN×2 opts×4 LR (~12h)      │
├──────────┼────────────────────────────────────────────────────────┤
│ GPU 3    │ P1-C: SOAP grid search + ResNet-18 runs                │
│          │ (grid: 6h → formal: 3.9h)                             │
├──────────┼────────────────────────────────────────────────────────┤
│ CPU      │ SCS计算pipeline（持续处理P0所有完成的checkpoint）         │
│          │ P0结果汇总 + group连续谱图绘制                           │
│          │ 理论推导：BN反对齐猜想直觉论证                            │
└──────────┴────────────────────────────────────────────────────────┘

=== 第2天（24-48h）：P1 主体完成 + P2提前启动 ===

【24-36h】ConvNeXt-T + ViT-Small 正式训练
┌──────────┬────────────────────────────────────────────────────────┐
│ GPU 0    │ P1-A: ConvNeXt-T LN × {Muon,AdamW,SGD} × 5s (13h)    │
├──────────┼────────────────────────────────────────────────────────┤
│ GPU 1    │ P1-A: ConvNeXt-T BN × {Muon,AdamW,SGD} × 5s (13h)    │
├──────────┼────────────────────────────────────────────────────────┤
│ GPU 2    │ P1-B: ViT-Small LN × {Muon,AdamW,SGD} × 3s (13.5h)   │
│          │ → P2-C: 训练配方消融 (2.6h) ← 释放时间填充              │
├──────────┼────────────────────────────────────────────────────────┤
│ GPU 3    │ P1-C: SOAP on DeiT-Tiny × {LN,RMSNorm} × 3s (5.8h)   │
│          │ → P1-C: SOAP on ConvNeXt-T × {LN,BN} × 3s (5.2h)     │
├──────────┼────────────────────────────────────────────────────────┤
│ CPU      │ SCS计算 + 结果分析                                      │
│          │ 写作：P0实验结果整合、group连续谱section                  │
│          │ 理论推导续                                              │
└──────────┴────────────────────────────────────────────────────────┘

【36-48h】ConvNeXt-T 续 + ViT-Small续 + P2提前启动
┌──────────┬────────────────────────────────────────────────────────┐
│ GPU 0    │ P1-A: ConvNeXt-T GN(1,C) × {Muon,AdamW,SGD} × 5s     │
│          │ (13h)                                                  │
├──────────┼────────────────────────────────────────────────────────┤
│ GPU 1    │ P1-A: ConvNeXt-T RMSNorm × {Muon,AdamW,SGD} × 5s     │
│          │ (13h)                                                  │
├──────────┼────────────────────────────────────────────────────────┤
│ GPU 2    │ P1-B: ViT-Small RMSNorm × {Muon,AdamW,SGD} × 3s      │
│          │ (13.5h)                                                │
├──────────┼────────────────────────────────────────────────────────┤
│ GPU 3    │ P2-A提前: ImageNet-100 ResNet-18+BN 部分启动            │
│          │ ← P1-B精简释放的GPU时间提前给ImageNet-100               │
├──────────┼────────────────────────────────────────────────────────┤
│ CPU      │ SCS计算（ConvNeXt-T完成的checkpoint）                    │
│          │ 写作：ConvNeXt-T结果section                              │
│          │ 理论推导：力争完成猜想初稿                                │
└──────────┴────────────────────────────────────────────────────────┘

=== 第3天（48-72h）：P1收尾 + P2+P3 ===

【48-60h】ViT-Small BN探测 + ImageNet-100主力 + 消融
┌──────────┬────────────────────────────────────────────────────────┐
│ GPU 0    │ P1-B: ViT-Small BN × {Muon,AdamW} × 2s               │
│          │ (高风险探索, 6h)                                       │
│          │ → P3-A: batch size消融 (6.5h)                          │
├──────────┼────────────────────────────────────────────────────────┤
│ GPU 1    │ P2-A: ImageNet-100 ResNet-18+BN × {Muon,AdamW,SGD}×3s │
│          │ (9 runs × 3.3h = 29.7h) → 溢出到Day4                  │
├──────────┼────────────────────────────────────────────────────────┤
│ GPU 2    │ P2-A: ImageNet-100 ResNet-18+RMSNorm × {Muon,AdamW}×3s │
│          │ (19.8h) ← P1-B精简释放的时间提前给ImageNet              │
├──────────┼────────────────────────────────────────────────────────┤
│ GPU 3    │ P2-B: Tiny-ImageNet补充 (15.6h)                        │
├──────────┼────────────────────────────────────────────────────────┤
│ CPU      │ SCS计算（ViT-Small checkpoint）                         │
│          │ P3-A: k敏感性 + N收敛性消融（CPU, 8h）                   │
│          │ 全面结果分析 + 图表制作                                  │
└──────────┴────────────────────────────────────────────────────────┘

【60-72h】ImageNet-100续 + P3+P4
┌──────────┬────────────────────────────────────────────────────────┐
│ GPU 0    │ P3-D: ResNet-{34,50,101} 深度扩展 (13h)                │
├──────────┼────────────────────────────────────────────────────────┤
│ GPU 1    │ P2-A续: ImageNet-100                                   │
├──────────┼────────────────────────────────────────────────────────┤
│ GPU 2    │ P3-C: Lion on ResNet-18+DeiT-Tiny (8h)                │
│          │ → P4-A: Norm-Aware Muon快速验证 (1.3h)                 │
│          │ → P4-A: 如positive→扩展 (6h)                           │
├──────────┼────────────────────────────────────────────────────────┤
│ GPU 3    │ P2-A: ImageNet-100 ResNet-18+GN×{Muon,AdamW,SGD}×3s   │
│          │ (29.7h) → 溢出到Day4                                   │
├──────────┼────────────────────────────────────────────────────────┤
│ CPU      │ 全面结果汇总                                            │
│          │ 写作：完整论文修订进行中                                  │
└──────────┴────────────────────────────────────────────────────────┘

=== 第4天（72-96h）：收尾 + 写作冲刺 ===

【72-84h】长训练收尾 + 补缺
┌──────────┬────────────────────────────────────────────────────────┐
│ GPU 0    │ P3-A: LR schedule + 模型宽度消融 (~8h)                  │
│          │ → P3-B: ImageNet-1K ResNet-50+BN Muon (如有时间, 12h)  │
├──────────┼────────────────────────────────────────────────────────┤
│ GPU 1    │ P2-A续完成 → P2-A: ImageNet-100 ResNet-50 (31.2h部分)  │
├──────────┼────────────────────────────────────────────────────────┤
│ GPU 2    │ 补缺实验 / 失败重跑                                     │
├──────────┼────────────────────────────────────────────────────────┤
│ GPU 3    │ P2-A续完成 → 补缺实验                                   │
├──────────┼────────────────────────────────────────────────────────┤
│ CPU      │ 最终SCS计算batch                                        │
│          │ 写作冲刺                                                │
└──────────┴────────────────────────────────────────────────────────┘

【84-96h】写作冲刺 + 提交
┌──────────┬────────────────────────────────────────────────────────┐
│ GPU 0-3  │ 仅处理最后的SCS补充计算和必要重跑                       │
├──────────┼────────────────────────────────────────────────────────┤
│ 全力写作  │ - 整合所有新实验到论文                                   │
│          │ - 更新所有Table和Figure                                  │
│          │ - 新增：group连续谱图 / ConvNeXt-T结果表 / ViT-Small表   │
│          │ - 新增：SOAP对比表 / 表征分析散点图 / 消融附录            │
│          │ - 重写Abstract/Introduction/Conclusion                  │
│          │ - 完善Appendix                                          │
│          │ - 最终校对 + 提交                                        │
└──────────┴────────────────────────────────────────────────────────┘
```

---

## 2.3 Idea层面的优化

### 2.3.1 核心概念升级：Performance Floor vs. Transparent Transmission

这是叙事层面的核心概念，替代论文原稿中隐含的"屏障=阻碍"定位：

**BN = Performance Floor（性能保障机制）：**
- BN的batch统计平滑将所有优化器压缩到一个窄精度带（如CIFAR-100上77-78%）
- 在这个floor之上，谱耦合差异的边际影响被大幅衰减
- **SCS在BN下揭示的不是性能差异，而是被floor掩盖的mechanistic diversity**

**LN/GN = Transparent Transmission（透明传导）：**
- LN/GN不提供batch统计平滑，优化器的谱耦合质量直接映射为精度差异
- Muon的+0.0035耦合提升在LN下放大为+5.49pp精度提升

**核心数据支持——precision floor的直接证据：**

| 归一化 | 精度跨度（across 3 optimizers） | 解释 |
|--------|-------------------------------|------|
| BN (ResNet-18) | 76.90-78.05 = **1.15pp** | BN压缩 → floor |
| GN(1,C) (ResNet-18) | 69.80-76.19 = **6.39pp** | GN透明 → 差异放大 |
| LN (DeiT-Tiny) | 70.52-76.01 = **5.49pp**（仅2 opts） | LN透明 → 差异放大 |

这组数据（已在论文中但未被这样分析）是"performance floor"最直接的证据。

### 2.3.2 理论贡献：基于Scale Invariance的BN反对齐几何论证

利用BN的**尺度不变性（scale invariance）**这一已知理论性质，将反对齐的论证从纯直觉升级为几何推论。

**BN Scale Invariance的核心性质（已有文献：Arora et al. 2019, 已被论文引用）：**

BN的输出对输入权重的缩放不变：BN(αWx) = BN(Wx)。这意味着：
1. BN将优化有效地约束到超球面 ‖W‖ = const 上
2. 在超球面上，梯度被投影为 g_proj = g − (g·w/‖w‖²)w（去除径向分量）
3. 优化轨迹沿球面切方向移动，不改变权重范数

**几何论证：**

> **Proposition（非形式化）：** 在BN的scale-invariant优化下：
>
> 1. BN方向向量 d^BN ∝ 1/σ_c 偏向低方差通道
> 2. 超球面约束下，梯度流倾向于沿高方差方向（信息量大的方向）分配权重能量
> 3. 因此W的主奇异向量 u₁ 倾向于对齐高方差通道
> 4. d^BN 和 u₁ 的能量分布在频谱的"相反端"→ 反对齐
>
> 关键推论：反对齐不是偶然的训练artifact，而是BN scale invariance的**几何推论**。
> 任何保持scale invariance的优化器（SGD, AdamW）都会产生反对齐；
> Muon的Newton-Schulz正交化操作在**梯度空间**而非**权重空间**，不受scale invariance约束，因此可以打破反对齐。

**形式化猜想：**

> **Conjecture:** 考虑单层线性模型 y = W · BN(x)，输入 x ∈ R^n 各通道独立且方差非均匀 (σ²₁ > σ²₂ > ... > σ²ₘ > 0)。设 W* = argmin E[‖y - t‖²] 为最小MSE解，则：
>
> ⟨d̂^BN, û₁⟩ = O(σ_min / σ_max)
>
> 即通道方差异质性越强，反对齐越强。

**经验验证：**
1. P0-A的GN连续谱：SCS随group数的连续变化应与猜想预测一致（GN不具有BN的scale invariance）
2. P0-D3的epoch-0基线：如果初始化时反对齐已存在，说明不仅是训练动态产物
3. P1-G的per-channel分解：直接验证"高方差channel vs 低方差channel"的能量分布预测
4. P1-F的因果干预：如果强制对齐后训练恢复反对齐，说明scale invariance约束在主动维护反对齐

**论文呈现策略：** Method §3.4 "Geometric Analysis: Scale Invariance and Anti-Alignment"——以1页篇幅给出几何论证 + Conjecture + 指向实验验证。不承诺严格证明，但有理论根基的几何论证足以将审稿人的攻击面从"没有理论"缩小为"理论不够严格"——后者在NeurIPS经验性论文中是可接受的。

### 2.3.3 Group连续谱——新的分析视角

P0-A的GN多group实验可以产生一个全新的分析角度：

```
BN ←→ GN(1,C) ←→ GN(8) ←→ GN(32) ←→ IN(G=C)
 ↑         ↑          ↑          ↑          ↑
batch统计  per-sample  中间态      中间态     per-channel
全通道LN   通道LN                             实例归一化
```

**预期：** SCS behavior应随group数连续变化，形成一条从"BN模式"到"learned-parameter模式"的连续曲线。这比离散的BN vs LN vs GN对比更有说服力，因为它展示的是**系统性趋势**而非个别比较。

**关键图：** "Group数 vs SCS excess"的连续变化图（每种优化器一条曲线），预期展示：
- Muon: 全range正耦合，随group数变化较小
- AdamW: 从BN端的负耦合逐渐过渡到LN端的正耦合
- SGD: 从BN端的深负耦合到GN端的强正耦合

### 2.3.4 因果分析框架

P1-F因果干预实验使论文可以建立一个完整的因果分析框架：

**因果问题层次：**

| 层次 | 问题 | 回答工具 | 状态 |
|------|------|---------|------|
| L1: 描述 | SCS在不同条件下有什么pattern？ | 现有SCS分析 | ✅ 论文已有 |
| L2: 关联 | SCS pattern与精度有什么关联？ | η（耦合传导率） | 本方案新增 |
| L3: 机制 | 为什么BN产生反对齐？ | Scale invariance几何论证 + per-channel分解 | 本方案新增 |
| L4: 因果 | SCS是否驱动精度？ | P1-F因果干预实验 | 本方案新增 |
| L5: 干预 | 能否通过操纵SCS改善训练？ | P4-A正交化衰减schedule | 本方案新增 |

论文原稿只覆盖L1，本方案将论文推进到L1-L5全覆盖。这是从"observation paper"到"analysis paper"的质变。

### 2.3.5 正交化衰减——从观察到行动的完整闭环

如果P1-E确认GN+Muon反转是持续现象，P4-A的正交化衰减schedule成功阻止反转并维持精度，论文就完成了一个完整闭环：

```
SCS诊断发现问题（GN+Muon晚期反转）
        ↓
Per-layer分解定位根因（layer4.1.conv2过度正交化）
        ↓
Scale invariance理论解释机制（正交化强度与谱结构演化的冲突）
        ↓
正交化衰减schedule解决问题（阻止反转 + 维持精度）
```

这个闭环直接回应"无actionable贡献"的弱点，且每一步都有实验支撑。

---

## 2.4 预期效果总结

| 维度 | 论文现状 | 优化后 | 关键改动 |
|------|---------|--------|---------|
| **叙事** | "屏障/调制"（自相矛盾） | "Performance Floor vs Transparent Transmission" + 耦合传导率η定量化 | 从定性标签→定量工具 |
| **SCS指标根基** | 未验证（k=10无论证，无方向区分） | SCS_signed双指标 + k敏感性 + epoch-0基线 | P0-D，零GPU成本 |
| **因果关系** | 无（仅相关性） | 因果干预实验（旋转权重→观察恢复） | P1-F，2.8 GPU-hours |
| **SGD悖论** | 回避 | 正面整合为"mechanistic diversity" | 叙事重构 |
| **架构混淆** | 致命弱点 | 大幅缓解 | ConvNeXt-T factorial + ViT-Small + GN连续谱 |
| **统计效力** | 弱(3s, df=2) | 强(5s, df=4) | P0-C |
| **理论支撑** | 直觉描述 | Scale invariance几何论证 + Conjecture | 从handwaving→几何推论 |
| **Actionable贡献** | 无 | 正交化衰减schedule（针对GN反转） | P4-A，动机明确、可验证 |
| **分析粒度** | Layer级SCS | + Per-channel SCS分解 | P1-G，零GPU成本 |
| **GN反转理解** | 未解释 | 层大小分析 + 精度关联 + 干预 | P1-E |
| **架构覆盖** | 2个 | 4-5个 | +ConvNeXt-T, ViT-Small, ResNet-{34,50,101} |
| **优化器覆盖** | 3个 | 5-6个 | +SOAP, Shampoo, Lion |
| **归一化覆盖** | 3个(BN,LN,GN) | 6个(+RMSNorm,GN多group,IN) | P0-A |
| **数据规模** | CIFAR-100+Tiny-IN | +ImageNet-100(+ImageNet-1K部分) | P2-A, P3-B |
| **总runs** | ~72 | ~450+ | ~6倍增长 |
| **因果分析层次** | L1（描述） | L1-L5全覆盖 | 从observation→analysis |
| **预估分数** | 5-6/10 | **7.5-8.5/10** | 因果证据+理论升级是核心加分项 |

## 2.5 风险与应急预案

| 风险 | 概率 | 影响 | 应急方案 |
|------|------|------|---------|
| **P0-A0：GN最优LR远偏离0.01，Table 3结论被动摇** | 中 | **致命** | 用新LR重跑SGD+GN 5-seed（+2.2h），更新Table 3数据。如果GN精度跨度缩小，调整"6倍不对称"的具体数值，但performance floor叙事仍可成立（只要BN跨度远小于GN跨度） |
| ConvNeXt-T上BN→LN替换不展示预期pattern | 中 | 高 | 转为"归一化效应依赖架构-归一化的原生匹配度"的更nuanced叙事 |
| ViT-Small + BN 训练不稳定 | 高 | 低 | 放弃此条件，在论文中讨论为何BN不适用于Transformer（有价值的observation） |
| SOAP不展示类似Muon的pattern | 中 | 中 | 限定为"Newton-Schulz正交化特有"，仍有价值但普适性claim减弱 |
| 表征分析（有效秩等）不支持performance floor | 中 | 中 | 使用"精度方差 across optimizers"作为替代证据（这个从已有数据可以直接算） |
| ImageNet-100/1K跑不完 | 高 | 低 | ImageNet-100优先，1K作为附录级补充，在Limitations中说明 |
| Norm-Aware Muon不work | 高 | 低 | 不放入正文，附录中brief discussion |
| Group连续谱不连续 | 低 | 中 | 作为interesting negative result讨论，可能反映group归一化的非线性行为 |
| **GN+Muon反转在300/400ep持续加深** | 中 | 中 | 整合入叙事作为"GN下过度正交化的谱代价"——反而增强论文的分析深度。需检查精度是否也同步下降 |
| **训练配方消融显示recipe是主因** | 低 | 高 | 弱化跨架构对比的claim，转而强调同架构内（ResNet-18, ConvNeXt-T）的norm变体对比 |
| LR grid search空间不够导致某些条件精度偏低 | 中 | 中 | 为新条件使用更宽的LR grid（如6点而非4点）；报告时标注grid范围 |
| 某些归一化替换后NaN/不收敛 | 中 | 低 | 记录为negative result，本身有分析价值（说明归一化兼容性） |
| **RMSNorm SCS方向向量定义不清** | 低 | 中 | RMSNorm无mean centering，d^RMS = (1/RMS(x))·1 为均匀向量。需在Method §3明确定义。如方向向量退化（全1向量），则SCS信号可能很弱——作为finding讨论。**注意：如果RMSNorm下SCS无信号，不要硬凑"6种归一化"的故事，改为诚实讨论"SCS对无mean-centering归一化不适用"** |
| **P0-D2: k敏感性显示结论不鲁棒** | 低 | **致命** | 如果k=20时BN屏障方向翻转 → 整篇论文的核心claim被动摇。应急：在论文中讨论k选择依据（与有效秩的关系），并报告k-robust的结论子集 |
| **P1-F: 因果干预的旋转破坏了权重结构** | 中 | 中 | Givens旋转理论上保持奇异值，但可能破坏层间的特征兼容性（downstream层期望特定的特征分布）。应急：使用小角度旋转（如10°而非90°），验证旋转前后loss的连续性 |
| **P1-F: 因果干预结果C（高SCS反而降低精度）** | 低 | 高 | 需要回退到"反对齐=有益"叙事，但这次有因果证据支撑而非仅相关性。实际上结果C反而是最publishable的——挑战直觉的因果证据 |
| **P4-A: 正交化衰减无效** | 中 | 低 | 不放入正文，附录中brief discussion。"诊断→治疗"闭环不成立，但诊断部分仍完整 |
| **Per-channel分解不支持理论预测** | 中 | 中 | 如果高/低方差channel的SCS贡献没有预期的分离 → 弱化scale invariance论证中的per-channel部分，保留超球面约束的宏观论证 |

## 2.6 关键设计决策清单

以下列出本方案相对于论文原稿的所有关键设计决策及其理由：

| 决策 | 理由 | 影响 | GPU成本 |
|------|------|------|---------|
| 叙事从"屏障/调制"改为"performance floor vs transparent transmission" | BN下AdamW比Muon更反对齐但精度更低，"屏障=坏事"的暗示不成立 | 消除内部矛盾 | 零 |
| 新增耦合传导率η | "Performance floor"叙事偏定性，需定量化工具 | 从定性标签→可计算的分类指标 | 零 |
| SCS_signed提升为主分析双指标（P0-D1） | SCS_abs的绝对值遮蔽对齐/反对齐方向，BN屏障靠间接推断 | 反对齐论证从间接变为直接 | 零（CPU） |
| k敏感性提升至P0（P0-D2） | 论文所有结论建立在k=10上，如果换k结论翻转论文就塌了 | 确保SCS指标根基牢固 | 零（CPU） |
| 新增epoch-0 SCS基线（P0-D3） | 未检查训练前SCS，无法区分"训练产物"vs"初始化固有" | 区分两种截然不同的因果机制 | 零（CPU） |
| 新增因果干预实验（P1-F） | 所有分析都是相关性，因果方向是高严重度盲区 | 从correlation升级为causal evidence | 2.8h |
| 新增per-channel SCS分解（P1-G） | BN是per-channel操作但SCS是per-layer聚合 | 直接验证理论猜想核心机制 | 零（CPU） |
| GN反转定位为核心分析（P1-E） | 反转可能是论文最有深度的发现，不应仅做简单调查 | 新增层大小分析、精度关联、正交化衰减连接 | 零（新增分析全CPU） |
| 正交化衰减schedule替代Norm-Aware Muon（P4-A） | Norm-Aware Muon方向有问题（BN下反对齐可能有益）；正交化衰减针对已观察到的GN反转 | actionable贡献动机正确、可验证 | +1.8h |
| 理论论证基于scale invariance | BN的scale invariance是已知性质，可将论证从handwaving升级为几何推论 | 审稿人攻击面缩小 | 零 |
| 去除"LN on ResNet-18" | GN(1,C)已等价于LN over channels，重复 | 改为更有价值的GN多group连续谱 | — |
| ViT-Small+BN标为高风险，降低seeds | BN在Transformer语义不清，可能不收敛 | 避免浪费GPU在不稳定实验上 | 节省 |
| ConvNeXt-T作为P1核心实验 | CNN+LN桥接架构，比ViT+BN更安全 | 保证至少一个安全的同架构多归一化实验 | — |
| 表征分析指标多元化 | 有效秩在优化器间已知无区分度（§5） | 降低单一指标失效风险 | — |
| Practical Implications增加置信度标注 | 原建议偏武断，未标注泛化限制 | 避免审稿人质疑过度泛化 | 零 |
| 所有时间估算×1.3安全系数 | ViT-Small参数量远大于DeiT-Tiny等 | 确保高优先级实验不被挤压 | — |

**方案核心理念：**
1. **加固SCS指标根基**（P0-D，零GPU）——不在沙滩上盖楼
2. **建立因果证据**（P1-F，极低GPU）——从observation到analysis的质变
3. **深化分析粒度**（P1-G per-channel，零GPU）——理论和实验的桥梁
4. **修正方法贡献方向**（P4-A正交化衰减）——动机正确、有闭环
5. **升级理论论证**（scale invariance几何推论）——利用已有理论工具

这五项核心改进中有三项零GPU成本，总GPU新增仅~4.6h，但每一项都直接回应了一个高/致命严重度的弱点。
