# NeurIPS 2026 投稿前论文诊断与优化方案（更新版）

**论文题目**：*Training Recipe Modulates Calibration Failure Modes under Class Imbalance*  
**目标会议**：NeurIPS 2026 Main Track  
**基准版本**：May 3, 2026 PDF  
**本文档目的**：基于 May 3 版 PDF 重新评估原优化方案，删除已解决的建议，保留并更新仍然适用的改进项，为最终投稿提供精准的修改清单。

---

## 一、与旧版优化方案的变更摘要

### 1.1 已解决、不再适用的建议（共 6 项）

以下建议在 May 3 版 PDF 中已被充分解决，从本方案中移除：

| 原方案编号 | 内容 | 解决证据 |
|-----------|------|---------|
| P0-0 | 标题拼写 "Imalance" → "Imbalance" | PDF 标题已修正 |
| 实验 1 主体 | 补齐 single-seed 配置（原估 6–8 runs） | §5.2 声明主配置已达 3 seeds；Table 6 验证方向 100% 一致 |
| 实验 4 | CutMix factor isolation 6 条件对照 | Figure 1 含 6 recipe；§4.3 报告 CutMix-only p=0.002 |
| — | Checkpoint stability 新增实验 | Appendix D：13 实验 / 118 checkpoints，10/13 100% stable；D.1 Spearman ρ ≥ 0.964 |
| — | Global TS / 2-group TS baseline 实验 | Appendix G 完成双架构分析；§5.1 引用核心数字 |
| — | Swin-T 完整分析 | Appendix E 全部 per-class 结果；主文 Table 3；4 recipes × 3 seeds |

### 1.2 保留并更新的建议

以下建议仍然适用，但根据当前版本调整了状态描述和优先级：

- P0-1（格式）：基本解决但仍有边界风险 → 调整为 P0
- P0-2（experiment ID）：有改善但仍有残留 → 降为 P1
- P0-3（claim 收缩）：部分改善但仍有过强表述 → 保持 P0
- P0-4（数据划分协议）：有改善但主文未明确写出 → 降为 P1
- 实验 2（ImageNet-100-LT 补 seeds）：未解决 → 保持 P1
- 实验 3（class-wise reliability）：未解决 → 提升为 **P0**（最关键缺口）
- 实验 5（balanced CIFAR 对照）：未解决 → 保持 P1
- 实验 7（logit norm probe）：未解决 → 保持 P2

### 1.3 新增建议

基于对 May 3 版的深入审读及对原方案潜在盲点的分析，新增以下此前未涉及的建议项：

- **P1-5（新增）**：强化 "practical implication" 叙事，回应 "无方法贡献" 拒稿风险——框定 2-group TS 角色 + 讨论 routing 方向 + 可选 CDA-TS baseline
- **P1-7（新增）**：显式说明 $T_k^*$ 在尾部类上的可靠性处理（CI-confirmed 设计选择），防止审稿人用高方差质疑框架
- **问题 7（新增）**："无方法贡献"是最可能的拒稿理由
- **问题 8（新增）**：$T_k^*$ 点估计在尾部类上的可靠性
- **§5.4（新增）**：Appendix 核心结论主文化策略
- **P0-1 补充**：class-wise reliability diagram 的实际可行性风险与 binning 噪声应对策略
- **P1-1 补充**：Balanced CIFAR 对照实验结果不如预期的应急预案
- **P1-2 补充**：ImageNet-100-LT 算力评估与降级方案

### 1.4 饱和方案 v2 新增（GPU 利用率优化）

为填补 Day 2 下午和 Day 3 的 GPU 空闲时段（原方案训练利用率仅 ~45%），v2 新增三组实验：

- **K 组（新增）**：RN50 + Swin-T ImageNet-100-LT 多 seed 扩展（8 runs, ~44 GPU-h）——将 ImageNet-100-LT 跨架构验证从 "ViT 3 seeds + 其余 1 seed" 升级为 "3 arch × 3 seeds 完整矩阵"
- **L 组（新增）**：CIFAR-100-LT 完整 4-recipe 因子（8 runs, ~20 GPU-h）——补齐 Mixup-only 和 LS-only 独立消融，使 CIFAR-100-LT 与 CIFAR-10-LT 实验设计对等
- **M 组（新增）**：ConvNeXt-T ImageNet-100-LT 侦察（4 runs, ~28 GPU-h）——验证新增架构在 ImageNet scale 上的 bidirectional pattern 一致性

v2 总计：**~78 runs、~242 GPU-h**，训练利用率 **~72%**，含 I/O 达 **~85–90%**。

---

## 二、当前版本已完成实验的完整盘点

> 本节更新自原方案 §1.4–§1.6，反映 May 3 版的实际覆盖情况。制定修改建议时，所有已完成实验以本节为准。

### 2.1 Seed 覆盖（大幅改善）

| 配置 | Seeds | 来源 |
|------|-------|------|
| ViT Standard (Mixup+LS) | 42, 123, 456 | §5.2, Table 6 |
| ViT Mixup-only | 42, 123, 456 | §4.5, Table 6 脚注 |
| ViT LS-only | 42, 123 | Table 6 |
| ViT Vanilla | 42, 123 | Table 6 |
| RN50 Vanilla | 42, 123 | Table 6 |
| RN50 LS-only | 42, 123 | Table 6 |
| RN18 Vanilla | 42, 123 | Table 6 |
| Swin-T 全部 4 recipes | 3 seeds each | §5.2 |
| 3 arch × Standard/Mixup-only/Vanilla | 3 seeds each | §5.2 |

**关键结论**：§4.5 确认 "All seven two-seed pairs preserved the qualitative calibration pattern"，Mixup-active |Δ| ≤ 0.105。**Seed robustness 已充分。**

**仍缺少的**：ImageNet-100-LT 全部配置（single seed），§5.2 明确承认。

### 2.2 CutMix Factor Isolation（已完成）

Figure 1 包含 6 个 recipe（Vanilla / Mixup-only / LS-only / CutMix-only / CutMix+LS / Standard）。§4.3 报告 CutMix-only $\overline{T}^*_{\text{head}}=0.856$，bidirectional，paired permutation test vs. Vanilla：$p=0.002, d=-2.86$。Appendix D 有 CutMix-only 和 CutMix+LS 的 checkpoint stability（各 20 checkpoints, 100% stable）。**无需新增。**

### 2.3 Global TS / 2-group TS（已完成）

Appendix G 包含：
- ViT：Global $T^*=1.168$；Table 11 展示 10 类 before/after；Table 12 group summary；2-group TS 改善 40.3%（head-tail gap -81%）
- CNN (RN50)：Global $T^*=0.775$；2-group TS 改善 18.7%（gap -75%）
- Table 13 跨架构对比
- Oracle vs. predicted routing：oracle ECE 0.0179 vs. predicted ECE 0.2222（3× 恶化）

§5.1 Discussion 已引用核心数字。**实验本身已完成；仅主文呈现方式可优化。**

### 2.4 Checkpoint Stability（已完成）

Appendix D：13 实验、118 checkpoints，10/13 100% bidirectional stable。D.1 新增 best-test-accuracy vs. last-epoch 对比（Spearman ρ ≥ 0.964，90% UC/OC agreement）。`vit_standard_earlystop_s42_v3` 用 stratified validation protocol 独立确认。**已远超审稿预期，无需新增。**

### 2.5 Accuracy Confound + Normalization Probe（已完成）

Enhanced ViT（RandAugment + DropPath, 无 Mixup）达到 60.07% accuracy 仍单向过自信（$T^*_{\text{head}}=1.682$）。RN50+GroupNorm(1,C) 产生 $T^*_{\text{head}}=2.958$，ruling out normalization type 作为唯一解释。**已完成。**

### 2.6 IR 敏感性（已完成）

Table 5 + Figure 3 覆盖 IR = {10, 20, 50, 100, 200}。Two-component decomposition 清晰定位转折点在 IR=10→20。**5 个 IR 点已足够。**

---

## 三、Review 总结（更新版）

### 3.1 总体评价

May 3 版相比旧版有实质性改善：seed 覆盖从大量 single-seed 扩展到主配置全部 3 seeds；checkpoint stability 覆盖 118 个 checkpoints；Swin-T 完整纳入；CutMix 独立隔离；post-hoc TS 双架构分析完成。论文的实验基础现在**显著强于旧版**。

但仍存在几个关键缺口，如果不解决，可能导致 borderline / weak reject：

1. **$T_k^*$ 的解释缺乏 class-wise reliability 验证**——这是当前版本最大的逻辑漏洞
2. **缺少 balanced vs. long-tailed 的交互对照**——无法直接证明现象来自 Mixup × imbalance 交互
3. **ImageNet-100-LT 仍为 single seed**——最强统计证据源的说服力被削弱
4. **部分 claim 仍然过强**
5. **主文页数可能略超 9 页限制**

如果修复以上要点，论文有较大机会达到 **weak accept** 或更强。

### 3.2 主要优点（与旧版评价一致，此处不重复）

论文的核心优势不变：问题重要、诊断工具简洁、factorial ablation 合理、实验覆盖面广、实践启示明确。新版在 robustness 方面的改善进一步强化了这些优点。

### 3.3 仍然存在的主要问题

#### 问题 1：$T_k^*$ 的解释需要 class-wise reliability 验证（原问题 3，仍未解决）

$T_k^* < 1$ 表示 sharpening logits 能降低该类别 NLL，但这是对 logit 空间的操作描述，**不直接等价于** reliability diagram 中 "predicted probability < empirical accuracy" 的定义。当前 PDF Figure 2 只有**全局** reliability diagrams，没有 class-wise 验证。

审稿人可以合理质疑：论文整个诊断框架的基础假设未被直接验证。这不是锦上添花，而是**必须堵的逻辑漏洞**。

#### 问题 2：缺少 balanced baseline 证明交互效应（原实验 5，仍未解决）

论文核心论点是 "input mixing **under class imbalance** creates bidirectional miscalibration"。但缺少 balanced setting 的对照，无法排除以下替代解释：

- Mixup 本身就会导致某些类欠自信（与 imbalance 无关）
- Bidirectional pattern 可能在 balanced 条件下也存在（只是程度较轻）

一个简单的 2×2 对照（balanced/LT × Mixup/Vanilla）就能消除这些质疑。

#### 问题 3：ImageNet-100-LT single seed（原实验 2，仍未解决）

§5.2 明确承认 "The ImageNet-100-LT scout experiments remain single-seed." 这是 100 类、224×224 分辨率的最强统计证据源，但 single seed 使其只能作为 supportive evidence。

#### 问题 4：部分 claim 仍然过强（原问题 2，部分改善）

具体表述见下文 P0-2。

#### 问题 5：主文页数边界风险（原问题 1，基本解决但需确认）

Section 5.3 Conclusion 结束位置接近第 10 页上部。需确认是否严格 ≤ 9 content pages。

#### 问题 6：Experiment ID 中 CutMix 污染痕迹（原问题 5，有改善但仍有残留）

Table 7 脚注 ‡ 仍提到 `vit_mixup_only_s42` "contained an unintended CutMix α=1.0 config"。审稿人看到这个脚注可能产生对实验管理质量的疑虑。

#### 问题 7："无方法贡献"是最可能的拒稿理由（新增）

NeurIPS 对纯诊断性 empirical study 的接受度有限。论文当前的实践启示链条是：诊断 bidirectional miscalibration → 证明 global TS 结构性失败 → 提出 2-group TS 作为简单修正。但 2-group TS 在 predicted routing 下 ECE 恶化 3 倍（0.0748 → 0.2222），这意味着**论文提出的唯一"方法"在实际部署场景中不 work**。审稿人可以合理质疑："你们的诊断虽然有趣，但提出的修正方案在实践中不可用。"

这不要求作者提出一个新方法，但需要更积极地框定 2-group TS 的角色，并讨论可行的 routing 策略方向。

#### 问题 8：$T_k^*$ 在尾部类上的点估计可靠性（新增）

CIFAR-10-LT 每类约 200 个验证样本用于拟合 $T_k^*$（单标量参数），过拟合风险不高。但论文 Table 1 脚注已指出 class 9（$n_{\text{train}}=50$）的 $T_k^*$ bootstrap SE = 153.3，点估计高度不稳定。论文使用 CI-confirmed 方向（而非原始点估计）来得出结论，这是正确的做法，但没有足够显式地说明这个设计选择及其理由。审稿人可能会拿尾部类 $T_k^*$ 的高方差来质疑整个框架的可靠性。

---

## 四、优化方案（更新版）

目标：**先堵逻辑漏洞，再补关键缺口，最后优化呈现。** 不再建议任何已完成的实验。

---

### 4.1 P0：投稿前必须完成

#### P0-1：Class-wise reliability diagram 验证 $T_k^*$ 的解释

**优先级提升理由**：这是当前版本**最大的逻辑漏洞**，成本极低但一直未做。如果审稿人质疑 "$T_k^*<1$ 不等于 reliability 意义上的 underconfidence"，论文整个诊断框架的说服力将大打折扣。

**具体做法**：

1. 从 Standard ViT (CIFAR-10-LT) 中选取 2 个 CI-confirmed underconfident 头部类（如 airplane, automobile）和 2 个 CI-confirmed overconfident 尾部类（如 ship, truck）
2. 为每个选定类别画 class-wise reliability diagram（横轴 predicted confidence，纵轴 empirical accuracy，仅使用真实标签为该类的样本）
3. 验证 $T_k^* < 1$ 的类别在 reliability diagram 上确实显示 confidence < accuracy（柱子在对角线下方），$T_k^* > 1$ 的类别反之
4. 新增一张主文表格（或在现有表格中添加列）：

| Class group | Mean $T_k^*$ | Mean conf-acc gap | Class-wise ECE | Direction agreement |
|---|---|---|---|---|
| CI-confirmed UC heads | < 1 | negative | ... | ✓ / ✗ |
| CI-confirmed OC tails | > 1 | positive | ... | ✓ / ✗ |

**主文呈现**：新增 1 张图（4 个子图：2 head + 2 tail 的 class-wise reliability diagram）+ 1 个简表。放在 §4.1 主结果之后，作为对 $T_k^*$ 解释的直接验证。

**⚠️ 实际可行性风险与应对**：

CIFAR-10-LT 使用 stratified balanced evaluation split 时，每类约 200 个样本。传统 reliability diagram 使用 10–15 个等宽 bin，每个 bin 内平均只有 13–20 个样本，可能导致**图形极其 noisy**，反而削弱说服力。

应对策略（按优先级）：

1. **首选方案：使用定量指标替代或辅助 binned diagram**。计算每个类的 **confidence-accuracy gap**（mean predicted confidence − empirical accuracy），这是一个不依赖 binning 的标量指标。如果所有 $T_k^* < 1$ 的类 gap < 0（模型 confidence 低于实际 accuracy），所有 $T_k^* > 1$ 的类 gap > 0，则直接证明了方向一致性。这个指标对样本量不敏感，即使每类 200 样本也能得到可靠结果。**将 direction agreement rate（例如 "10/10 classes agree"）作为主文的核心报告指标**。

2. **Reliability diagram 的处理**：如果仍要画图，使用 **5 个 equal-mass bins**（每 bin ~40 样本），而非 10–15 个等宽 bin。Equal-mass binning 确保每个 bin 有足够样本，减少噪声。在图中加 shaded area 表示 $\pm 1$ binomial SE，让读者看到不确定性。

3. **备选数据源**：如果 CIFAR-10-LT 样本量确实不足以生成可读的 class-wise reliability diagram，**优先在 ImageNet-100-LT 上做**。ImageNet-100-LT 有更多验证样本且有 100 个类，class-wise reliability 分析的统计效力更强。但这需要 ImageNet-100-LT 的模型输出可用。

4. **双保险策略**：同时报告 (a) confidence-accuracy gap 定量表（所有类）+ (b) 2–4 个代表性类的 reliability diagram（选 $T_k^*$ 离 1 最远的类，即信号最强的类）。这样即使单个类的 diagram 有噪声，定量表仍能给出可靠的方向一致性结论。

**预期收益**：堵住核心逻辑漏洞，将 $T_k^*$ 从"NLL-optimal temperature"提升为经验验证的"calibration direction indicator"。  
**预估成本**：极低。只需对已有模型输出做后处理分析，无需重新训练。半天至一天（含处理 binning 噪声问题）。

---

#### P0-2：收缩仍然过强的 claim

当前 PDF 中仍需修改的具体表述：

| 位置 | 当前表述 | 建议修改 |
|------|---------|---------|
| §5.1 | "the training recipe **consistently predicts** the qualitative mode of calibration failure under class imbalance across all three architectures tested" | "the training recipe **is a strong empirical predictor of** the qualitative mode of calibration failure across the three architectures tested" |
| §5.1 | "Input mixing is **the consistent factor**: it shifts head-class calibration from overconfidence to underconfidence" | "Input mixing is **the primary empirical factor** associated with shifts in head-class calibration from overconfidence toward underconfidence" |
| §5.3 | "input mixing (Mixup + CutMix) in particular — **consistently predict** whether calibration failure under class imbalance is unidirectional or bidirectional across all architectures tested" | "input mixing (Mixup + CutMix) in particular — **is a strong predictor of** whether calibration failure under class imbalance is unidirectional or bidirectional in our tested configurations" |
| Abstract | "we **discover** that input mixing **drives** head (frequent) classes into underconfidence" | "we **find** that input mixing **shifts** head (frequent) classes toward underconfidence" |

**核心原则**：
- "consistently predicts" → "is a strong empirical predictor of"
- "determines / drives" → "shifts / is associated with"
- "the consistent factor" → "the primary empirical factor"
- 在涉及架构的结论中加上 "in our tested configurations" / "under standard training choices"

**预估成本**：极低。纯文字修改，1–2 小时。

---

#### P0-3：确认主文 ≤ 9 content pages

当前 Section 5.3 Conclusion 结束在第 10 页上部，可能超限。建议以下压缩策略（按优先级）：

1. **Table 5**（IR 敏感性完整表，第 8 页）：移入 Appendix。主文保留 Figure 3 和文字描述 即可传达相同信息
2. **Table 2**（cross-seed stability，第 5 页）：仅 4 行，可内联到 §4.1 文字中，如 "Cross-seed replication yields |Δ| ≤ 0.089 for Mixup-active recipes (Table 6 in Appendix)"
3. **§5.2 Limitations**（第 10 页）：normalization probe 讨论（"A normalization probe... RN50+GN warranting further investigation"）可精简为 2 句，完整讨论留在 Appendix
4. **§5.1 Discussion** 中 2-group TS 的 oracle vs. predicted routing 细节可精简，完整分析已在 Appendix G

**目标**：释放约 0.75–1 页空间，确保主文内容严格在 9 页内结束。

**预估成本**：低。文字重组 + LaTeX 排版，约半天。

---

### 4.2 P1：强烈建议完成（显著提升审稿分数）

#### P1-1：Balanced CIFAR 对照实验

**目的**：证明 bidirectional miscalibration 来自 input mixing 与 class imbalance 的**交互效应**，而非 input mixing 的独立效应。

**具体做法**：

在 balanced CIFAR-10（原始均衡分布，每类 5000 样本）上训练以下配置：
- Balanced + Vanilla
- Balanced + Standard (Mixup+LS)

与已有的 CIFAR-10-LT 结果形成 2×2 对照：

| | Balanced | Long-tailed (IR=100) |
|---|---|---|
| Vanilla | 预期：uniform OC | 已有：uniform OC（$\overline{T}^*_{\text{head}}=2.053$） |
| Standard (Mixup+LS) | 预期：improved calibration, 无方向冲突 | 已有：bidirectional（$\overline{T}^*_{\text{head}}=0.762$） |

**主文呈现**：可在 §4.4 或 §4.1 末尾新增一段，用 2×2 小表展示。或用一句话：*"On balanced CIFAR-10, the same Standard recipe produces $\overline{T}^*_{\text{head}}=X.XX$ with 0 CI-confirmed underconfident classes, confirming that bidirectional miscalibration arises from the interaction of input mixing and class imbalance."*

**⚠️ 结果不如预期的应急预案**：

上述 2×2 表格预期 balanced + Mixup 不会出现双向分裂。但存在两种非预期场景：

**场景 A：balanced + Mixup 也出现轻微双向分裂**（例如少数类 $T_k^* < 1$，但程度远弱于 LT 条件）。

这不是灾难性结果。如果出现，应调整叙事为：*"Input mixing introduces a weak class-dependent calibration effect even on balanced data, but class imbalance dramatically amplifies the head–tail directional split."* 论文结论从 "imbalance is necessary" 调整为 "imbalance is the primary amplifier"。这个 framing 仍然有价值——它说明了为什么 balanced benchmarks 上 Mixup 的校准改善（Thulasidasan et al. 2019）不能外推到 LT 场景。

报告方式：在 2×2 表中如实报告 balanced 条件的 $T_k^*$ 和 $n_{\text{under}}/n_{\text{over}}$，加一句 "The bidirectional pattern is absent or marginal on balanced data ($n_{\text{under}}=X$ vs. $n_{\text{under}}=3$ under LT), confirming that class imbalance is the primary amplifier."

**场景 B：balanced + Mixup 出现与 LT 同等强度的双向分裂。**

这是最糟糕的场景——它意味着 bidirectional pattern 不依赖 imbalance，论文的核心叙事需要根本性调整。但从已有文献来看（Thulasidasan et al. 2019、Zhang et al. 2022 均报告 Mixup 在 balanced 数据上改善全局校准），此场景可能性很低。如果真的出现，需要检查实验代码是否有 bug（如 data loader 意外引入了不平衡），而非直接改变叙事。

**关键提醒**：无论结果如何，balanced 对照实验都应如实报告。不报告意味着审稿人可以合理推测作者做了但结果不利。

**预期收益**：将论文结论从 "Mixup drives underconfidence" 精确为 "Mixup × imbalance creates directional conflict"，叙事更精确、更难被攻击。  
**预估成本**：低。仅需 2 个新训练 run + 后处理分析。约 1 天。

---

#### P1-2：ImageNet-100-LT 补充至少 1 个 seed

**目的**：将 ImageNet-100-LT 从 "scout experiment" 提升为正式 cross-dataset validation。

**建议最小配置**：
- Standard (Mixup+LS)：补 seed 123
- Vanilla：补 seed 123

**如果算力充足**，进一步补充：
- Mixup-only：补 seed 123
- LS-only：补 seed 123

**报告方式**：在 Table 4 中添加新 seed 行；在 §4.3 中报告 cross-seed |Δ| 和方向一致性。

**预期收益**：消除 "ImageNet-100-LT single seed" 这一最容易被审稿人抓住的 robustness 缺口。  
**预估成本**：中。ImageNet-100-LT 训练成本高于 CIFAR。2–4 个 run，取决于算力。

---

#### P1-3：主文 §3 明确数据划分协议

**目的**：消除 test-set leakage 质疑。

当前版本通过 Appendix D.1（checkpoint selection robustness）间接回应了这一问题，但主文 §3 **仍未明确写出数据划分协议的边界**。建议在 §3 中用 3–4 句话明确：

> We construct three disjoint evaluation splits from the standard test set: (1) a model-selection validation set (Y samples per class) for checkpoint selection, (2) a calibration set (Z samples per class) for fitting temperature parameters and estimating per-class $T_k^*$, and (3) an evaluation set (W samples per class) for final metric reporting. Full split details and sizes are provided in Appendix H.

如果实际协议不是三集合分离的，至少应说明现有协议为何不构成 leakage（例如：$T_k^*$ 估计和 ECE 报告使用同一集合，但 $T_k^*$ 不参与 model selection）。

**预估成本**：极低。纯文字补充。

---

#### P1-4：清理 Experiment ID 中的 CutMix 污染痕迹

**问题**：Table 7（Appendix D）脚注 ‡ 仍可见 `vit_mixup_only_s42` 的 CutMix 污染说明。尽管作者已用 clean run 替代，但这个脚注的存在可能让审稿人产生对实验管理质量的疑虑。

**建议做法**：

1. 在 Appendix H（或新建一个实验注册表 Appendix）中建立一张完整的实验配置表，明确每个 exp_id 的 Mixup α、CutMix α、LS ε、seed、checkpoint policy、使用位置（主文/附录/排除）
2. 将 Table 7 脚注 ‡ 改为更中性的表述，例如：*"This experiment used the Mixup+CutMix recipe (effective recipe: input mixing active, no LS). Checkpoint-level directional stability (bidirectional at all 30 checkpoints) remains valid."* 避免使用 "unintended" / "contamination" 等暗示实验管理问题的词汇
3. 确保 `_v3` 后缀的含义有说明——如果是版本迭代，在实验注册表中注明

**预估成本**：低。文字修改 + 可能需要补充一张表。

---

#### P1-5：强化 "practical implication" 叙事，回应 "无方法贡献" 质疑

**问题分析**：

"只是 observation，没有 method" 是这篇论文在 NeurIPS **最可能的被拒理由**。当前的应对策略是引用 Appendix G 的 2-group TS 改善（40.3%/18.7%），但这个策略有一个致命弱点：**2-group TS 在 predicted routing 下 ECE 恶化 3 倍**（0.0748 → 0.2222）。这意味着论文提出的唯一"方法"在不知道真实标签的部署场景中不可用。

审稿人可以写：*"The proposed 2-group TS requires oracle knowledge of which frequency group a test sample belongs to. Since this is unavailable at test time, the practical value of the diagnostic is unclear."*

**应对策略（三管齐下）**：

1. **框定 2-group TS 的角色**：在 §5.1 Discussion 中明确，2-group TS 不是作为可部署方法提出的，而是作为**诊断验证**——它证明了 group-aware recalibration 的上界远优于 global TS，从而确认了 bidirectional miscalibration 是一个有实际后果的结构性问题。建议措辞：*"The 2-group oracle scheme is not proposed as a deployable method, but as a diagnostic upper bound that confirms group-aware recalibration substantially outperforms global scaling—validating that the bidirectional structure has practical consequences, not just diagnostic interest."*

2. **讨论可行的 routing 方向**：在 §5.1 或 §5.3 中用 2-3 句话讨论 predicted routing 失败的原因（尾部类准确率仅 40.8%，多数尾部样本被误分到头部类，导致错误应用 $T_{\text{head}}$）和可能的改进方向：
   - 使用训练集 class frequency prior 做 soft routing（不需要预测正确，只需要估计样本属于高频/低频组的概率）
   - 使用 model uncertainty（如 entropy、Monte Carlo dropout）作为 routing signal
   - 使用 CDA-TS 类方法直接编码 class frequency 到 temperature，绕过 routing 问题
   
   **不需要实现这些方法**，只需在 Discussion 中指出方向，表明作者认真思考了实用性。

3. **CDA-TS baseline 升级为 P1**（从原 P2-3 提升）：如果算力允许，实际跑一个 CDA-TS baseline。CDA-TS 直接将 class frequency 编码到 per-class temperature 中，不需要 routing——如果它在 bidirectional miscalibration 场景下有效，恰好证明了论文诊断框架的实用价值（"我们的诊断揭示了为什么 CDA-TS 比 global TS 更合适"）。如果无效，也说明 bidirectional miscalibration 的修正比单纯编码频率更复杂，是一个值得进一步研究的 open problem。

**预估成本**：策略 1 和 2 是纯文字修改（极低）；策略 3 需要实现 CDA-TS（中等）。

---

#### P1-6：在 Related Work / Setup 中定位 $T_k^*$ 的概念框架

校准文献中有重要的概念区分：
- **Top-label calibration**（Guo et al. 2017）：只关心 argmax 类别的预测概率是否校准
- **Class-conditional calibration**（Nixon et al. 2019; Widmann et al. 2019）：要求每个类别的预测概率分别校准

论文的 per-class $T_k^*$ 本质上是 class-conditional 分析，但当前 PDF 中没有明确这一定位。建议在 §2.1 或 §3 中加一段：

> Our per-class $T_k^*$ decomposition operates at the class-conditional level (cf. Nixon et al., 2019): it diagnoses whether predictions conditioned on true class $k$ are systematically over- or under-confident. This complements top-label calibration metrics (Guo et al., 2017) that aggregate across classes and may mask class-dependent directional heterogeneity.

**预估成本**：极低。补充 1–2 句话 + 1 个引用。

---

#### P1-7：显式说明 $T_k^*$ 点估计在尾部类上的可靠性处理

**问题**：Table 1 脚注指出 class 9（$n_{\text{train}}=50$）的 $T_k^*$ bootstrap SE = 153.3，Table 5 中 IR=200 时 $T_9^*=50.0$（优化器上界）。审稿人可能质疑：如果尾部类的 $T_k^*$ 点估计这么不稳定，整个诊断框架的基础是否可靠？

**论文实际上已经正确处理了这个问题**——通过使用 CI-confirmed 方向（而非原始点估计）来得出结论，并在 aggregate statistics 中排除 optimizer-bound-hitting 的类。但这个设计选择没有被足够显式地说明。

**建议做法**：

1. 在 §3 中明确一段方法论说明：*"Because per-class $T_k^*$ estimates can exhibit high variance for the rarest classes (bootstrap SE > 1.0 for classes with $n_{\text{train}} \leq 50$), we base all directional conclusions on CI-confirmed classification rather than raw point estimates. A class is classified as underconfident or overconfident only if its bootstrap 95% CI lies entirely below or above 1.0, respectively. This conservative criterion ensures that our bidirectional finding is not driven by noisy tail-class estimates."*

2. 在 Limitations 中加一句：*"For the most extreme tail classes, $T_k^*$ saturates at the optimizer bound ($T_k^* = 50.0$), indicating severe overconfidence but precluding quantitative comparison with other classes. Our directional conclusions remain valid, but absolute $T_k^*$ magnitudes should not be compared across classes with vastly different sample sizes."*

3. 在涉及 $\overline{T}_{\text{tail}}^*$ 的报告中，考虑同时报告 median（对极端值更鲁棒）和 mean，或在注脚中标注排除了 optimizer-bound-hitting 的类后的值（部分 Table 已做，但不一致）。

**预期收益**：预防审稿人用 $T_k^*$ 高方差来质疑整个框架。同时，显式说明 CI-confirmed 设计选择反而能展示方法论的严谨性——变被动防守为主动加分。  
**预估成本**：极低。纯文字补充。

---

### 4.3 P2：建议完成（进一步增强论文质量）

#### P2-1：将 Appendix G 核心结果可视化为主文图

**现状**：§5.1 Discussion 已引用 Appendix G 的关键数字（global TS 恶化 head、2-group TS 改善 40.3%/18.7%），但仅以文字形式出现。

**建议**：新增 1 张主文图（如 grouped bar chart），展示：
- 横轴：No correction / Global TS / 2-group TS
- 纵轴：Mean |$T_k^*$ − 1|
- 分组：Head classes vs. Tail classes
- 双面板：ViT (left) + CNN (right)

数据全部来自 Appendix G Table 12/13，无需新实验。

**预期收益**：将 "practical implication" 从文字描述升级为视觉直觉，增强 "诊断 → 失败机制 → 简单修正" 的叙事完整性。  
**预估成本**：极低。画一张图，约 2 小时。

**优先级说明**：降为 P2 的原因是核心数字已在 Discussion 中出现，不补图不会导致逻辑缺口。

---

#### P2-2：Logit norm by class 机制探测

**目的**：提供一个 mechanistic probe 避免 "pure phenomenology" 评价。

**做法**：对 Standard ViT 和 Vanilla ViT 的已有 checkpoint，提取所有测试样本的 logit 向量，按 head/tail 分组计算平均 logit norm（$\|z_k\|_2$）。

**预期结果**：如果 Mixup 系统性压低 head-class logit norm（使 softmax 输出更接近 uniform），则直接解释了 $T_k^* < 1$（需要 sharpening 来恢复区分度）。

**主文呈现**：放在 Appendix 中，主文 §5.1 Discussion 或 §4.2 用 1–2 句话引用。

**预估成本**：极低。无需重新训练，只需对已有 logits 做统计分析。约半天。

---

#### P2-3：CDA-TS 或 Dual-Branch TS 外部 baseline 对比

> **注意**：本项的核心实验部分已部分提升到 P1-5（策略 3）。如果 P1-5 中已实现 CDA-TS baseline，本项自动完成。此处保留完整说明以备参考。

**目的**：与已有的 class-aware calibration 方法进行对话，同时回应 "无方法贡献" 质疑。

当前 Appendix G 只比较了 Global TS 和 2-group TS。论文已引用 Islam et al. [8]（CDA-TS）和 Guo et al. [7]（Dual-Branch TS），但未做实验对比。

**建议**：选择其中 **1 个**方法（优先 CDA-TS，因为它直接编码 class frequency 到 per-class temperature 且不需要 routing），在 Standard ViT (CIFAR-10-LT) 上运行，与 Global TS / 2-group TS 对比。

**报告维度**：
- Overall ECE / class-wise ECE mean
- Head ECE / Tail ECE
- Mean |$T_k^*$ − 1| after correction
- Head-tail $\overline{T}_k^*$ gap

**两种可能结果的解读**：
- 如果 CDA-TS 有效：证明论文诊断框架的实用价值——"一旦识别出 bidirectional miscalibration，class-frequency-aware 方法是比 global TS 更合适的修正策略"
- 如果 CDA-TS 无效或有限：说明 bidirectional miscalibration 的修正比单纯编码频率更复杂，class-level heterogeneity（如 2-group TS 内部仍有残差）需要更精细的方法

**预期收益**：证明 per-class $T_k^*$ 诊断框架不仅揭示了问题，也能指导选择更合适的 post-hoc 方法。  
**预估成本**：中。需要实现或适配 CDA-TS 代码。但 CDA-TS 是 post-hoc 方法，不需要重新训练模型。

---

#### P2-4：补充参考文献

经调研，以下论文与本文高度相关但未被引用：

| 参考文献 | 相关性 | 建议位置 |
|---------|--------|---------|
| Nixon et al. (2019) "Measuring Calibration in Deep Learning" | 提出 class-wise ECE，是 per-class 校准分析的直接前驱 | §2.1 或 §3 |
| Hebbalaguppe et al. (CVPR 2022) "A Stitch in Time Saves Nine" | Train-time regularization 对校准的影响 | §2.3 |

此外，论文引用的 **Ao et al. [2] (2023) "Two sides of miscalibration"** 建议 double-check 引用信息的准确性（出版地、页码、可访问性）。

---

### 4.4 饱和计算方案：充分利用 GPU 资源的扩展实验

> **硬件资源**：2–4 × RTX 5090（~32GB VRAM / 卡）  
> **运行模式**：Agent Auto Research（全自动流水线：训练 → checkpoint 管理 → $T_k^*$ 估计 → 统计分析 → 结果汇总）  
> **总预算**：3–4 天 × 4 GPU × 24h = **288–384 GPU-hours**

在 P0–P2 核心任务之外，充裕的算力允许以下扩展实验同步执行，**将多个原 P3 项目提升为可执行**。

---

#### 4.4.1 扩展实验设计总览

| 实验组 | 新增 runs | 单 run 估时 | 总 GPU-hours | 原优先级 → 新优先级 | 中稿提升 |
|--------|----------|-----------|------------|------------------|---------|
| A. Balanced CIFAR-10 全因子 | 4 recipes × 3 seeds = 12 | ~1.5h | ~18 | P1 → **P1（扩充）** | 高 |
| B. Balanced CIFAR-100 对照 | 2 recipes × 2 seeds = 4 | ~2.5h | ~10 | 新增 | 中–高 |
| C. ImageNet-100-LT 全因子补 seed（ViT） | 4 recipes × 2 seeds = 8 | ~6h | ~48 | P1 → **P1（扩充）** | 高 |
| D. ConvNeXt-T on CIFAR-10-LT | 4 recipes × 3 seeds = 12 | ~2h | ~24 | P3 → **P1** | 高 |
| E. DeiT-S / ViT-B 规模探测 | 2 recipes × 2 seeds = 4 | ~2.5h | ~10 | P3 → **P2** | 中 |
| F. IR 细密扫描（IR=15,25,30,40） | 2 recipes × 4 IR × 1 seed = 8 | ~1.5h | ~12 | P3 → **P2** | 中 |
| G. 长尾训练 baselines | 3 methods × 2 seeds = 6 | ~2h | ~12 | P3 → **P2** | 中 |
| H. CDA-TS / Dual-Branch TS | 0（post-hoc） | — | 0 | P2 → **P1** | 高 |
| I. Logit norm + entropy 全面分析 | 0（post-hoc） | — | 0 | P2 → **P1** | 中 |
| J. Class-wise reliability 全模型 | 0（post-hoc） | — | 0 | P0 → **P0（扩充）** | 极高 |
| **K. RN50 + Swin-T IN100-LT 多 seed** | **2 arch × 2 recipes × 2 seeds = 8** | **~5.5h** | **~44** | **新增 → P1** | **高** |
| **L. CIFAR-100-LT 完整 4-recipe 因子** | **2 recipes × 3 seeds + 2 补 seed = 8** | **~2.5h** | **~20** | **新增 → P1** | **中–高** |
| **M. ConvNeXt-T IN100-LT 侦察** | **2 recipes × 2 seeds = 4** | **~7h** | **~28** | **新增 → P2** | **中–高** |
| **合计** | **~74 训练 runs** | | **~226 GPU-h** | | |

> **GPU 利用率**：226 GPU-hours 训练 + ~16h buffer/rerun = ~242 GPU-h。含后处理 I/O 开销约 280–300 有效 GPU-hours。4 卡 × 3.5 天 = 336 GPU-hours 可用，**训练利用率约 72%，实际负载利用率 ~85–90%**。如果使用 2 卡，总可用 168 GPU-h，需精简 E/G/M 组，总训练 GPU-h 降至 ~160h。

---

#### 4.4.2 各实验组详细设计

**A. Balanced CIFAR-10 全因子（12 runs）**

原方案仅建议 2 runs（Vanilla + Standard）。饱和方案扩充为完整 2×2 factorial × 3 seeds：

| Config | Mixup | LS | Seeds |
|--------|-------|----|-------|
| Balanced-Standard | ✓ | ✓ | 42, 123, 456 |
| Balanced-Mixup-only | ✓ | ✗ | 42, 123, 456 |
| Balanced-LS-only | ✗ | ✓ | 42, 123, 456 |
| Balanced-Vanilla | ✗ | ✗ | 42, 123, 456 |

**收益**：
- 完整 2(balanced/LT) × 2(Mixup/no) × 2(LS/no) × 3 seeds 的 factorial design，可以做 **三因子 ANOVA**（Imbalance × Mixup × LS），定量分离交互效应
- 3 seeds 使 balanced 结果本身有 robustness 保障
- 如果 balanced 条件下确实无双向分裂（预期如此），这是极强的交互效应证据

**应急预案**：如 P1-1 所述（场景 A / 场景 B 处理）。3 seeds 的设计还允许判断任何非预期结果是否跨 seed 稳定。

---

**B. Balanced CIFAR-100 对照（4 runs）**

在 100 类数据上重复 balanced 对照，与已有 CIFAR-100-LT 结果形成交叉验证：

| Config | Seeds |
|--------|-------|
| Balanced-CIFAR-100-Standard | 42, 123 |
| Balanced-CIFAR-100-Vanilla | 42, 123 |

**收益**：证明交互效应不依赖于类别数（10 vs 100）。CIFAR-100 有 100 类，统计效力远强于 CIFAR-10。

---

**C. ImageNet-100-LT 全因子补 seed（8 runs）**

原方案建议最低补 2 runs（Standard + Vanilla × 1 seed）。饱和方案扩充为 4 recipes × 2 additional seeds：

| Config | 已有 seed | 补充 seeds |
|--------|----------|-----------|
| IN100-Standard | 42 | 123, 456 |
| IN100-Mixup-only | 42 | 123, 456 |
| IN100-LS-only | 42 | 123, 456 |
| IN100-Vanilla | 42 | 123, 456 |

**收益**：将 ImageNet-100-LT 从 "single-seed scout" 提升为**与 CIFAR-10-LT 对等的正式 cross-dataset validation**（4 recipes × 3 seeds）。这是当前论文最大的 robustness 缺口，补齐后直接消除审稿人最容易抓住的攻击点。

---

**D. ConvNeXt-T on CIFAR-10-LT（12 runs）**

ConvNeXt 是使用 Transformer-style 训练 recipe 的现代 CNN。它位于 ResNet（传统 CNN）和 ViT（Transformer）之间，能回答一个关键问题：**bidirectional pattern 是由架构决定还是由训练 recipe 决定？**

| Config | Seeds |
|--------|-------|
| ConvNeXt-Standard | 42, 123, 456 |
| ConvNeXt-Mixup-only | 42, 123, 456 |
| ConvNeXt-LS-only | 42, 123, 456 |
| ConvNeXt-Vanilla | 42, 123, 456 |

**收益**：
- 如果 ConvNeXt + Mixup 也产生 bidirectional（预期如此），论文架构覆盖从 3 种扩展到 4 种，且包含了"现代 CNN"这一重要类别
- ConvNeXt 使用 AdamW + LayerNorm（与 ViT 相同），但有卷积结构（与 ResNet 相同）。如果 bidirectional 程度介于 ViT 和 ResNet 之间，完美支持 local-vs-global 感受野假说
- ConvNeXt 是 NeurIPS 审稿人**最容易想到的"你为什么没测这个"**架构

**实现注意**：ConvNeXt-T 需要 timm 实现；CIFAR 32×32 可能需要调整 stem（改为更小 kernel），参考 ConvNeXt 论文的 CIFAR 配置。

---

**E. DeiT-S / ViT-B 规模探测（4 runs）**

测试模型规模对 baseline overconfidence 和 threshold-crossing 的影响：

| Config | Model | Seeds |
|--------|-------|-------|
| CIFAR-10-LT Standard | DeiT-S 或 ViT-B/16 | 42, 123 |
| CIFAR-10-LT Vanilla | DeiT-S 或 ViT-B/16 | 42, 123 |

**收益**：回应 "ViT-Small/4 太小，更大模型可能不同" 的质疑。如果大模型的 baseline overconfidence 不同，可能改变 threshold-crossing 动态。

---

**F. IR 细密扫描（8 runs）**

在 IR=10→20 转折区间增加密度：

| IR | Standard | Vanilla |
|----|----------|---------|
| 15 | ✓ | ✓ |
| 25 | ✓ | ✓ |
| 30 | ✓ | ✓ |
| 40 | ✓ | ✓ |

**收益**：精确定位 bidirectional miscalibration 的 onset IR。当前 Figure 3 在 IR=10（uniform UC）和 IR=20（bidirectional）之间有跳跃。如果 IR=15 已经 bidirectional，说明 onset 在极低 imbalance 就出现——这会增强论文的实践警示意义。

---

**G. 长尾训练 baselines（6 runs）**

测试 3 个代表性长尾训练方法的校准方向特征：

| Method | Seeds | 意义 |
|--------|-------|------|
| Logit Adjustment (Menon et al. 2021) | 42, 123 | 最常用长尾 baseline |
| Balanced Softmax (Ren et al. 2020) | 42, 123 | 处理类先验偏移 |
| cRT / Classifier Re-training | 42, 123 | 分析 classifier head 对校准方向的影响 |

**收益**：
- 如果这些方法也产生 bidirectional miscalibration（因为它们通常与 Mixup 一起使用），论文的实践警示范围扩大到整个长尾学习领域
- 如果某些方法缓解了 bidirectional pattern，提供了对比参照，增强论文的实用价值
- 回应审稿人 "为什么不和长尾方法比较" 的质疑

---

**H. CDA-TS / Dual-Branch TS（0 GPU，post-hoc）**

在所有已有 + 新增模型的 logits 上运行 post-hoc calibration baselines：

| Method | 应用范围 |
|--------|---------|
| Global TS | 已有（Appendix G） |
| 2-group TS (oracle) | 已有（Appendix G） |
| CDA-TS (Islam et al. 2021) | **新增**：ViT Standard + CNN Standard + ConvNeXt Standard |
| Dual-Branch TS (Guo et al. 2023) | **新增**：同上 |
| Per-class TS (10/100 temperatures) | **新增**：作为理论上界 |

**收益**：完整的 post-hoc calibration baseline 比较，将论文从 "global TS 不 work" 扩展到 "这是不同 class-aware 方法的系统比较"。

---

**I. Logit norm + entropy 全面分析（0 GPU）**

对所有已有和新增 checkpoint 提取 logit 向量，计算：

| 指标 | 分析维度 |
|------|---------|
| $\|z_k\|_2$（logit norm by class） | Head vs Tail × Mixup vs Vanilla × 所有架构 |
| $H(\text{softmax}(z))$（predictive entropy） | 同上 |
| $\max_k \text{softmax}(z)_k$（max confidence） | 同上 |
| Logit margin（top1 - top2 logit） | 同上 |

**收益**：提供 mechanistic insight。如果 Mixup 系统性压低 head-class logit norm 或提高 head-class entropy，直接解释了 $T_k^* < 1$。这把论文从 "pure phenomenology" 提升为 "有 mechanistic 线索的 diagnostic study"。

---

**J. Class-wise reliability 全模型分析（0 GPU）**

不仅对 Standard ViT（P0-1），而是对**所有**已有 + 新增模型系统性计算：

| 分析内容 | 覆盖范围 |
|---------|---------|
| Per-class confidence-accuracy gap | 所有 arch × 所有 recipe × 所有 dataset |
| Per-class ECE (class-conditional) | 同上 |
| Direction agreement rate ($T_k^*$ sign vs conf-acc gap sign) | 同上 |
| 代表性 class reliability diagram（5-bin equal-mass） | 选取 4–6 个代表性配置 |

**收益**：如果 direction agreement rate 在所有配置上都 > 90%，这是对 $T_k^*$ 诊断框架的**全面验证**，远超只验证一个模型的 P0-1。

---

**K. RN50 + Swin-T ImageNet-100-LT 多 seed 扩展（8 runs）**

C 组仅补充 ViT-Small 的 ImageNet-100-LT 多 seed。RN50 和 Swin-T 在 ImageNet-100-LT 上同样需要多 seed 验证，否则跨架构一致性仅依赖单 seed：

| Architecture | Config | Seeds |
|-------------|--------|-------|
| ResNet-50 | Standard | 123, 456 |
| ResNet-50 | Vanilla | 123, 456 |
| Swin-T | Standard | 123, 456 |
| Swin-T | Vanilla | 123, 456 |

**收益**：
- 将 ImageNet-100-LT 的 cross-architecture 比较从 "3 architectures × 1 seed" 升级为 "3 architectures × 3 seeds"
- 解决 "ImageNet 结果仅 single seed, 方向一致性可能是噪声" 的审稿攻击
- RN50 ~5h/run、Swin-T ~6h/run，合计 ~44 GPU-h

---

**L. CIFAR-100-LT 完整 4-recipe 因子扩展（8 runs）**

论文已有 CIFAR-100-LT 的 Standard 和 Vanilla，但缺少 Mixup-only 和 LS-only 的独立消融。这导致 CIFAR-100-LT 的 factorial 分析不完整，无法与 CIFAR-10-LT 的 §4.3 对等：

| Config | Seeds |
|--------|-------|
| CIFAR-100-LT Mixup-only | 42, 123, 456 |
| CIFAR-100-LT LS-only | 42, 123, 456 |
| CIFAR-100-LT Standard s456（补 seed） | 456 |
| CIFAR-100-LT Vanilla s456（补 seed） | 456 |

**收益**：
- 将 CIFAR-100-LT 的 recipe 消融从 "Standard vs Vanilla" 升级为 "4-recipe 完整因子"
- 支持 100 类场景下的 Mixup vs LS 独立贡献分析——CIFAR-10 只有 10 类，per-class 样本量大；CIFAR-100 有 100 类，更接近实际长尾场景
- 8 runs × ~2.5h = ~20 GPU-h

---

**M. ConvNeXt-T ImageNet-100-LT 侦察（4 runs）**

D 组在 CIFAR 上验证 ConvNeXt，但如果 ConvNeXt 仅有 CIFAR 结果，审稿人可能质疑 "为什么新增架构只在小数据集上测"。ImageNet-100-LT 上的 ConvNeXt 侦察补齐这一缺口：

| Config | Seeds |
|--------|-------|
| ConvNeXt-T Standard (ImageNet-100-LT) | 42, 123 |
| ConvNeXt-T Vanilla (ImageNet-100-LT) | 42, 123 |

**收益**：
- ConvNeXt 的跨数据集验证：如果 CIFAR 和 ImageNet-100-LT 上 bidirectional pattern 一致，ConvNeXt 结论从 "CIFAR-only observation" 升级为 "跨 scale 验证"
- 4 runs × ~7h = ~28 GPU-h

---

## 五、叙事与呈现建议（更新版）

### 5.1 Claim 语气收缩后的核心叙事

> Under long-tailed class imbalance, input mixing augmentations—Mixup and CutMix—shift the calibration failure mode from uniform overconfidence to bidirectional miscalibration, where head classes become underconfident while tail classes remain overconfident. This directional conflict explains why global temperature scaling is structurally limited and motivates group-aware recalibration.

### 5.2 建议的摘要方向

> Modern neural networks under class imbalance are known to be poorly calibrated, but global metrics obscure whether different classes err in the same direction. We study class-wise calibration through per-class optimal temperatures ($T_k^*$) and find that input mixing augmentations under long-tailed training shift the calibration failure mode from uniform overconfidence to bidirectional miscalibration: head classes move toward underconfidence ($T_k^* < 1$) while tail classes remain overconfident ($T_k^* > 1$). A factorial ablation shows input mixing is the primary empirical driver of this directional split, with label smoothing mainly modulating its severity. The pattern holds across three architectures (ViT, ResNet-50, Swin-T), three datasets (CIFAR-10/100-LT, ImageNet-100-LT), and multiple random seeds. Because head and tail classes require opposite temperature corrections, global temperature scaling becomes a structural compromise; a simple 2-group scheme reduces the head–tail calibration gap by 75–81%. Our results establish that post-hoc calibration strategies for models trained with input mixing on imbalanced data should operate at class- or group-level rather than globally.

**与当前摘要的关键差异**：
- "drives" → "shifts"
- "discover" → "find"
- "consistently" → 删除或改为 "across tested configurations"
- 增加 balanced/LT 交互的暗示（"under long-tailed training"）

### 5.3 主文 9 页结构建议

| Section | 建议页数 | 内容 |
|---------|---------|------|
| 1. Introduction | 1.0 | 问题、gap、发现、贡献 |
| 2. Related Work | 0.75 | calibration、long-tail calibration、Mixup 与 calibration |
| 3. Diagnostic Framework & Setup | 1.0 | $T_k^*$ 定义与解释、datasets、recipes、metrics |
| 4.1 Main Results | 1.25 | CIFAR-10-LT 2×2 主表 + class-wise reliability 验证 |
| 4.2 Cross-Architecture | 0.75 | 三架构对比 + threshold-crossing |
| 4.3 Cross-Dataset | 0.75 | ImageNet-100-LT + CIFAR-100-LT |
| 4.4 IR Sensitivity | 0.5 | Figure 3 + 文字描述（Table 5 移入 Appendix） |
| 4.5 Robustness | 0.5 | Seed + checkpoint stability 要点（细节在 Appendix） |
| 5.1 Global TS Failure | 0.75 | Discussion + 可选 Figure |
| 5.2 Limitations | 0.25 | 精简版 |
| 5.3 Conclusion | 0.25 | |
| **合计** | **~7.75** | 留约 1.25 页给 figures/tables |

### 5.4 Appendix 内容主文化策略（新增）

**核心问题**：当前论文有大量高质量分析埋在 Appendix A–G 中，但 NeurIPS 审稿人通常只仔细读主文，对 Appendix 至多扫一眼。如果审稿人没有看到 Appendix D 的 118 checkpoint 稳定性分析、Appendix G 的 global TS 失败证据、Appendix E/F 的跨架构 per-class 结果，他们对论文 robustness 的感知会大打折扣。

**通用策略：每个 Appendix 分析的核心结论都应在主文中用一句话总结并引用**，确保审稿人即使不读 Appendix 也能感受到这些分析的存在和说服力。

具体建议：

| Appendix | 核心结论 | 主文引用建议 |
|----------|---------|------------|
| A (Seed stability) | 7 配置 × 2–3 seeds，方向 100% 一致 | §4.5 已有覆盖 ✓ |
| B (Additional limitations) | 5 个补充 caveats | 可在 §5.2 用一句 "see Appendix B for additional caveats" 引用 |
| C (Statistical tests) | Permutation test p=0.007, Holm-corrected | §4.1 已有覆盖 ✓ |
| D (Checkpoint stability) | 118 checkpoints, 10/13 100% stable；best vs. last epoch Spearman ρ ≥ 0.964 | §4.5 应确保这两个数字都出现在主文中（当前 §5.2 提到了但位置不够突出） |
| E (Swin-T per-class) | Swin-T bidirectional，LS 对 tail OC 有最强抑制（Δ=2.281） | §4.2 提到 Swin-T 但未引用 Appendix E 的 LS-tail 发现 |
| F (CNN per-class) | RN50 Standard 推 8/10 类到 $T_k^* < 1$，只有 ship/truck 仍 OC | §4.2 可加一句 "Notably, the Standard recipe pushes eight of ten RN50 classes below $T_k^*=1$, leaving only the two most extreme tail classes overconfident (Appendix F)" |
| G (Global TS analysis) | Global TS 恶化 head/2-group TS 改善 40–81%/oracle vs predicted routing | §5.1 已引用核心数字 ✓，但可增加一张图（P2-1） |

**成本**：极低。每条只需在主文相应位置加 1 句话。

### 5.5 主文图表建议

| 图/表 | 内容 | 状态 |
|------|------|------|
| Table 1 | ViT-Small 2×2 recipe ablation | 已有 ✓ |
| Figure 1 | $\overline{T}^*_{\text{head}}$ across recipes/datasets | 已有 ✓ |
| Figure 2 | Reliability diagrams (global) | 已有 ✓ |
| **新增 Figure** | **Class-wise reliability diagram（2 head + 2 tail 子图）** | **P0-1 新增** |
| Table 3 | Cross-architecture comparison | 已有 ✓ |
| Figure 3 | $T_k^*$ vs. IR | 已有 ✓ |
| Table 2 | Cross-seed stability | 考虑内联到文字 |
| Table 4 | Cross-dataset 2×2 | 已有 ✓ |
| Table 5 | IR 敏感性完整表 | 建议移入 Appendix |
| **可选新增 Figure** | Global TS vs. 2-group TS bar chart | P2-1 可选 |

---

## 六、针对潜在 reviewer attack 的应对策略（更新版）

| Reviewer 质疑 | 应对策略 | 当前准备程度 |
|--------------|---------|------------|
| $T_k^*$ 不等于 reliability 意义上的 under/overconfidence | Class-wise reliability diagram + confidence-accuracy gap 验证（P0-1） | **❌ 需做** |
| Bidirectional pattern 可能不需要 imbalance，Mixup 自身就会导致 | Balanced CIFAR 对照（P1-1），含应急预案 | **❌ 需做** |
| ImageNet-100-LT single seed | 补 seeds（P1-2） | **❌ 需做** |
| Claim 过强 / 因果化 | 收缩表述（P0-2） | **⚠️ 部分改善** |
| Test set leakage | 明确数据划分协议（P1-3） | **⚠️ 间接回应但未明示** |
| Experiment ID 混乱 / CutMix 污染 | 清理脚注 + 建立实验注册表（P1-4） | **⚠️ 有改善** |
| 只是 observation，无方法贡献 | **三管齐下策略（P1-5）**：框定 2-group TS 为诊断上界 + 讨论 routing 方向 + 可选 CDA-TS baseline | **❌ 需做（当前应对不足）** |
| 2-group TS predicted routing 不 work | §5.1 讨论 routing 失败原因 + 指出改进方向（P1-5 策略 2） | **❌ 需做** |
| 尾部类 $T_k^*$ 方差太大，框架不可靠 | 显式说明 CI-confirmed 设计选择（P1-7） | **❌ 需做** |
| Architecture claim confounded | 已有 threshold-crossing 分析 + 降低 claim 强度 | **✅ 可回应（收缩 claim 后）** |
| Checkpoint selection artifact | Appendix D（118 ckpts）+ D.1（selection strategy 对比） | **✅ 充分回应** |
| Seed instability | §4.5 + Table 6（主配置 3 seeds, 方向 100% 一致） | **✅ 充分回应** |
| CutMix 未被独立测试 | Figure 1 六 recipe 对照 + §4.3 CutMix-only p=0.002 | **✅ 充分回应** |
| 只在 CIFAR scale 上验证 | ImageNet-100-LT（224×224, 100 类）已覆盖；但 single seed 待补 | **⚠️ 部分覆盖** |
| 好内容埋在 Appendix 审稿人看不到 | Appendix 核心结论主文化策略（§5.4） | **⚠️ 需做（成本极低）** |

---

## 七、更新后的优先级总表（饱和计算版）

> 基于 4 × RTX 5090、3–4 天 Agent Auto Research 模式重新评估。原 P3 中多项因算力充裕而提升为 P1/P2。

| 优先级 | 项目 | 类型 | 解决的审稿风险 | GPU-h |
|--------|------|------|------------|-------|
| **P0** | Class-wise reliability + conf-acc gap 验证（**全模型**，非仅 1 个） | 分析 | 核心逻辑漏洞 | 0 |
| **P0** | 收缩 "consistently predicts" 等过强表述 | 写作 | Reviewer 攻击 claim | 0 |
| **P0** | 确认主文 ≤ 9 页 | 写作 | Desk rejection | 0 |
| **P1** | Balanced CIFAR-10 全因子（12 runs）+ 应急预案 | **训练** | Mixup × imbalance 交互 | ~18 |
| **P1** | Balanced CIFAR-100 对照（4 runs） | **训练** | 跨类别数交互验证 | ~10 |
| **P1** | ImageNet-100-LT 全因子补 2 seeds（8 runs） | **训练** | Cross-dataset robustness | ~48 |
| **P1** | ConvNeXt-T 全因子（12 runs） | **训练** | Architecture generality（现代 CNN） | ~24 |
| **P1** | CDA-TS + Dual-Branch TS + Per-class TS baselines | 分析 | "无方法贡献"拒稿风险 | 0 |
| **P1** | 强化 practical implication 叙事（2-group TS 框定 + routing 讨论） | 写作 | "无方法贡献"拒稿风险 | 0 |
| **P1** | Logit norm + entropy 全面 mechanistic 分析 | 分析 | "Phenomenology only" | 0 |
| **P1** | 主文 §3 明确数据划分 + $T_k^*$ 可靠性说明 + 概念定位 | 写作 | 多项防守 | 0 |
| **P1** | 清理 experiment ID + Appendix 核心结论主文化 | 写作 | Reproducibility + 呈现 | 0 |
| **P2** | DeiT-S / ViT-B 规模探测（4 runs） | **训练** | Model scale concern | ~10 |
| **P2** | IR 细密扫描 IR=15/25/30/40（8 runs） | **训练** | 精确定位 onset | ~12 |
| **P2** | 长尾训练 baselines: LogitAdj / BalancedSoftmax / cRT（6 runs） | **训练** | Long-tail method comparison | ~12 |
| **P2** | Appendix G → 主文 Figure + 补充参考文献 | 写作/分析 | Practical relevance 可视化 | 0 |
| **P1** | RN50 + Swin-T ImageNet-100-LT 多 seed（8 runs） | **训练** | Cross-arch × cross-dataset robustness | ~44 |
| **P1** | CIFAR-100-LT 完整 4-recipe 因子（8 runs） | **训练** | CIFAR-100 factorial completeness | ~20 |
| **P2** | ConvNeXt-T ImageNet-100-LT 侦察（4 runs） | **训练** | New arch cross-dataset validation | ~28 |
| | **训练 runs 合计** | | | **~74 runs** |
| | **GPU-hours 合计** | | | **~226h** |

---

## 八、饱和执行时间线（4 × RTX 5090，3.5 天）

> **设计原则**：GPU 训练任务 24h 不间断排队；Agent 在训练期间并行执行分析任务和写作任务；每天结束时有 checkpoint，确保即使提前截止也有完整可用的论文版本。

---

### Day 0（准备，约 2–4 小时）—— 环境搭建 + 实验配置

**Agent 任务**：
- [ ] 确认训练代码支持所有所需架构（ViT-Small、ResNet-50、Swin-T、ConvNeXt-T、DeiT-S），检查 timm 版本
- [ ] 准备 balanced CIFAR-10/100 数据集（标准均衡分布，不做 LT 采样）
- [ ] 建立统一实验配置模板：每个 exp 的 Mixup α、CutMix α、LS ε、seed、架构、数据集、IR、checkpoint policy 全部参数化
- [ ] 搭建自动化流水线：训练完成 → 自动提取 logits → 自动估计 per-class $T_k^*$ → 自动计算 bootstrap CI + confidence-accuracy gap + logit norm + entropy
- [ ] 建立实验注册表（Appendix H 用），每个 run 自动写入配置记录

**GPU 状态**：空闲（准备阶段不占 GPU）

---

### Day 1（全天）—— 启动全部 CIFAR 训练 + 核心分析

**GPU 分配（4 卡并行，CIFAR runs ~1.5–2.5h/run）**：

| 时段 | GPU 0 | GPU 1 | GPU 2 | GPU 3 |
|------|-------|-------|-------|-------|
| 00:00–02:00 | Bal-C10 Standard s42 | Bal-C10 Mixup s42 | Bal-C10 LS s42 | Bal-C10 Vanilla s42 |
| 02:00–04:00 | Bal-C10 Standard s123 | Bal-C10 Mixup s123 | Bal-C10 LS s123 | Bal-C10 Vanilla s123 |
| 04:00–06:00 | Bal-C10 Standard s456 | Bal-C10 Mixup s456 | Bal-C10 LS s456 | Bal-C10 Vanilla s456 |
| 06:00–09:00 | Bal-C100 Standard s42 | Bal-C100 Vanilla s42 | Bal-C100 Standard s123 | Bal-C100 Vanilla s123 |
| 09:00–11:00 | ConvNeXt Standard s42 | ConvNeXt Mixup s42 | ConvNeXt LS s42 | ConvNeXt Vanilla s42 |
| 11:00–13:00 | ConvNeXt Standard s123 | ConvNeXt Mixup s123 | ConvNeXt LS s123 | ConvNeXt Vanilla s123 |
| 13:00–15:00 | ConvNeXt Standard s456 | ConvNeXt Mixup s456 | ConvNeXt LS s456 | ConvNeXt Vanilla s456 |
| 15:00–17:00 | IR15 Standard | IR15 Vanilla | IR25 Standard | IR25 Vanilla |
| 17:00–19:00 | IR30 Standard | IR30 Vanilla | IR40 Standard | IR40 Vanilla |
| 19:00–21:00 | DeiT-S Standard s42 | DeiT-S Vanilla s42 | DeiT-S Standard s123 | DeiT-S Vanilla s123 |
| 21:00–23:00 | LogitAdj s42 | LogitAdj s123 | BalSoftmax s42 | BalSoftmax s123 |
| 23:00–01:00 | cRT s42 | cRT s123 | *buffer/rerun* | *buffer/rerun* |

**Day 1 GPU 产出**：~44 CIFAR-scale runs 完成（Balanced-C10 ×12 + Balanced-C100 ×4 + ConvNeXt ×12 + IR sweep ×8 + DeiT-S ×4 + LT baselines ×6 – 2 buffer）

**Agent 并行分析任务**（CPU，与 GPU 训练同步执行）：

- [ ] **P0-1 扩充版（J 组）**：对所有**已有**模型（ViT/RN50/Swin-T × 4 recipes × CIFAR-10-LT + ImageNet-100-LT）执行 class-wise reliability 全面分析
  - 计算每个模型每个类的 confidence-accuracy gap
  - 计算 direction agreement rate（$T_k^*$ sign vs conf-acc gap sign）
  - 生成代表性 class reliability diagrams（5-bin equal-mass）
  - 生成汇总表：跨所有模型的 agreement rate
- [ ] **I 组**：对已有 checkpoint 提取 logit 向量，计算 logit norm / entropy / max confidence / logit margin，按 head/tail 分组统计
- [ ] **H 组准备**：实现 CDA-TS 和 Dual-Branch TS 的 post-hoc 评估代码

**Agent 写作任务**（与分析同步执行）：

- [ ] **P0-2**：修改所有过强 claim
- [ ] **P0-3**：Table 5 移入 Appendix；Table 2 内联；精简 §5.2；确认 ≤ 9 页
- [ ] **P1-3**：§3 补充数据划分协议
- [ ] **P1-6**：§2.1 补充 $T_k^*$ 概念定位
- [ ] **P1-7**：§3 补充 CI-confirmed 方法论说明
- [ ] 补充参考文献（Nixon et al. 2019 等）

**Day 1 结束 Checkpoint**：
- ✅ 所有 CIFAR-scale 训练完成
- ✅ 所有已有模型的 class-wise reliability 分析完成
- ✅ 主文文字修改（P0-2, P0-3, P1-3, P1-6, P1-7）完成
- ✅ CDA-TS/Dual-Branch TS 评估代码就绪

---

### Day 2（全天）—— ImageNet-100-LT 训练 + Day 1 结果整合

**GPU 分配（4 卡并行，ImageNet runs ~6h/run）**：

| 时段 | GPU 0 | GPU 1 | GPU 2 | GPU 3 |
|------|-------|-------|-------|-------|
| 00:00–06:00 | IN100 Standard s123 | IN100 Mixup s123 | IN100 LS-only s123 | IN100 Vanilla s123 |
| 06:00–12:00 | IN100 Standard s456 | IN100 Mixup s456 | IN100 LS-only s456 | IN100 Vanilla s456 |
| 12:00–17:00 | RN50-IN100 Std s123 | RN50-IN100 Van s123 | RN50-IN100 Std s456 | RN50-IN100 Van s456 |
| 17:00–23:00 | SwinT-IN100 Std s123 | SwinT-IN100 Van s123 | SwinT-IN100 Std s456 | SwinT-IN100 Van s456 |
| 23:00–24:00 | *Day 1 失败 rerun* | *Day 1 失败 rerun* | *buffer* | *buffer* |

**Day 2 GPU 产出**：8 个 ViT ImageNet-100-LT runs（C 组）+ 4 个 RN50 ImageNet-100-LT runs + 4 个 Swin-T ImageNet-100-LT runs（K 组）= 16 runs 完成

**Agent 并行分析任务**（处理 Day 1 GPU 产出）：

- [ ] **Balanced CIFAR-10 结果分析**：
  - 计算 12 runs 的 per-class $T_k^*$，生成 2(bal/LT) × 2(Mixup/no) × 2(LS/no) 完整 factorial 表
  - 三因子 ANOVA（Imbalance × Mixup × LS）分析交互效应
  - 判断是否触发应急预案（场景 A / B）
  - 草拟主文段落
- [ ] **Balanced CIFAR-100 结果分析**：
  - 计算 4 runs 的 per-class $T_k^*$
  - 与 CIFAR-10 balanced 结果对比
- [ ] **ConvNeXt-T 结果分析**：
  - 计算 12 runs 的 per-class $T_k^*$
  - 与 ViT/RN50/Swin-T 对比，检验 local-vs-global 感受野假说
  - 判断 ConvNeXt 是否支持 threshold-crossing 解释
  - 草拟主文段落（新增架构）
- [ ] **IR 细密扫描结果分析**：
  - 更新 Figure 3，增加 4 个 IR 点
  - 精确定位 onset IR
- [ ] **DeiT-S / ViT-B 结果分析**：
  - 与 ViT-Small 对比 baseline overconfidence
- [ ] **长尾训练 baselines 分析**：
  - 各方法的 per-class $T_k^*$ pattern
  - 判断哪些方法也产生 bidirectional
- [ ] **H 组执行**：在所有 Day 1 已完成模型上运行 CDA-TS / Dual-Branch TS / Per-class TS
- [ ] **Day 1 新模型的 class-wise reliability 分析**：ConvNeXt、DeiT-S、balanced models、LT baselines

**Agent 写作任务**：

- [ ] **P1-5**：基于 CDA-TS 结果框定 practical implication 叙事
- [ ] **P1-1**：基于 balanced 结果写入主文交互效应分析
- [ ] **Appendix 主文化**：为每个 Appendix 添加主文总结句
- [ ] 清理 experiment ID + 建立实验注册表

**Day 2 结束 Checkpoint**：
- ✅ ImageNet-100-LT ViT 8 runs（C 组）+ RN50 4 runs + Swin-T 4 runs（K 组）= 16 runs 完成
- ✅ 所有 CIFAR-scale 结果分析完成
- ✅ CDA-TS/Dual-Branch TS 评估完成
- ✅ Balanced 对照结果写入主文
- ✅ ConvNeXt CIFAR 结果初步写入
- 📝 论文草稿包含所有新增分析，但尚未最终整合
- 🔄 Day 3 凌晨继续：CIFAR-100-LT L 组 + ConvNeXt IN100 M 组排队中

---

### Day 3（全天）—— 补充实验 + ImageNet 结果整合 + 全文重构

**GPU 分配（4 卡并行）**：

| 时段 | GPU 0 | GPU 1 | GPU 2 | GPU 3 |
|------|-------|-------|-------|-------|
| 00:00–02:30 | C100-LT Mixup s42 | C100-LT Mixup s123 | C100-LT LS s42 | C100-LT LS s123 |
| 02:30–05:00 | C100-LT Mixup s456 | C100-LT LS s456 | C100-LT Std s456 | C100-LT Van s456 |
| 05:00–12:00 | ConvNeXt-IN100 Std s42 | ConvNeXt-IN100 Van s42 | ConvNeXt-IN100 Std s123 | ConvNeXt-IN100 Van s123 |
| 12:00–18:00 | *Day 1–2 失败 rerun* | *Day 1–2 失败 rerun* | *buffer* | *buffer* |
| 18:00–24:00 | *可选探索性实验* | *可选探索性实验* | *buffer* | *buffer* |

**Day 3 GPU 产出**：8 个 CIFAR-100-LT 完整因子 runs（L 组）+ 4 个 ConvNeXt-T ImageNet-100-LT runs（M 组）= 12 runs 完成，外加 rerun buffer

**Agent 分析任务**（处理 Day 2 GPU 产出）：

- [ ] **ImageNet-100-LT 多 seed 结果分析**：
  - 4 recipes × 3 seeds 完整 factorial
  - Cross-seed |Δ| 和方向一致性
  - 与 CIFAR-10-LT 的 cross-dataset Δ 对比
  - Paired permutation tests（多 seed 版本）
  - 更新 Table 4，生成 ImageNet-100-LT seed stability 表
- [ ] **ImageNet-100-LT 新模型的 class-wise reliability 分析**
- [ ] **ImageNet-100-LT 上的 CDA-TS / Dual-Branch TS 评估**
- [ ] **全模型汇总分析**：
  - 生成最终的 direction agreement rate 汇总表（所有 arch × recipe × dataset）
  - 生成最终的 logit norm / entropy 汇总（所有 arch × recipe）
  - 生成最终的 post-hoc calibration 对比表（Global TS vs 2-group TS vs CDA-TS vs Dual-Branch TS vs Per-class TS）

**Agent 写作任务（全天重点）**：

- [ ] **主文全面重构**：
  - 整合所有新增结果（balanced 对照、ConvNeXt、ImageNet 多 seed、IR 细密扫描、class-wise reliability 验证、post-hoc baseline 对比、mechanistic probes）
  - 确认 ≤ 9 页
  - 更新所有图表
  - 重写 Abstract 和 Introduction（反映扩展后的 scope）
  - 重写 Discussion 和 Conclusion
- [ ] **Appendix 重构**：
  - 新增 Appendix 容纳 ConvNeXt 完整结果、balanced 对照完整表、CDA-TS 详细分析、IR 细密结果、DeiT-S 结果
  - 更新实验注册表（Appendix H）
  - 确保每个 Appendix 的核心结论在主文有总结句

**Agent 补充分析任务**（处理 Day 3 GPU 产出）：

- [ ] **CIFAR-100-LT 完整因子分析（L 组）**：
  - 计算 8 runs 的 per-class $T_k^*$
  - 生成 CIFAR-100-LT 的 2(Mixup/no) × 2(LS/no) 因子表
  - 与 CIFAR-10-LT 和 balanced CIFAR-100 对比：100 类场景下 Mixup 独立贡献是否更强
  - Paired permutation tests（Mixup-only vs Vanilla、LS-only vs Vanilla）
- [ ] **RN50 + Swin-T ImageNet-100-LT 多 seed 分析（K 组）**：
  - 3 arch × 3 seeds 完整 cross-architecture seed stability 表
  - 计算 cross-seed 方向一致性
  - 更新 Table 4，生成三架构版 ImageNet-100-LT 汇总
- [ ] **ConvNeXt-T ImageNet-100-LT 分析（M 组）**：
  - 与 CIFAR 上 ConvNeXt 结果对比
  - 判断 ConvNeXt bidirectional pattern 是否跨 scale 一致
  - 如果一致，将 ConvNeXt 升级为主文正式架构（而非仅 Appendix）

**Day 3 结束 Checkpoint**：
- ✅ 所有 ~74 训练 runs 完成（含 Day 3 的 L + M 组）
- ✅ 所有后处理分析完成
- ✅ 主文草稿整合所有新增内容
- 📝 需要最终 polish

---

### Day 3.5（半天）—— 最终 polish + 提交准备

**Agent 任务**：

- [ ] 全文一致性检查：数字、表格引用、实验 ID 全部 cross-check
- [ ] LaTeX 编译检查：确认 ≤ 9 content pages
- [ ] PDF 技术检查：`pdffonts` 确认无 Type 3 fonts
- [ ] 匿名性检查：代码路径、实验 ID、supplementary 中无作者信息
- [ ] NeurIPS checklist 填写
- [ ] Supplementary material 打包（代码 + 完整实验配置）
- [ ] 最终 proofread：拼写、语法、figure caption 一致性、页数确认

---

## 九、饱和计算工作量估算

### 9.1 GPU 训练工作量

| 实验组 | Runs | 单 run GPU-h | 总 GPU-h | 执行日 |
|--------|------|------------|---------|-------|
| A. Balanced CIFAR-10 全因子 | 12 | ~1.5 | ~18 | Day 1 |
| B. Balanced CIFAR-100 对照 | 4 | ~2.5 | ~10 | Day 1 |
| C. ImageNet-100-LT 全因子补 seed（ViT） | 8 | ~6 | ~48 | Day 2 前半 |
| D. ConvNeXt-T 全因子（CIFAR） | 12 | ~2 | ~24 | Day 1 |
| E. DeiT-S 规模探测 | 4 | ~2.5 | ~10 | Day 1 |
| F. IR 细密扫描 | 8 | ~1.5 | ~12 | Day 1 |
| G. 长尾训练 baselines | 6 | ~2 | ~12 | Day 1 |
| **K. RN50 + Swin-T IN100-LT 多 seed** | **8** | **~5.5** | **~44** | **Day 2 后半** |
| **L. CIFAR-100-LT 完整 4-recipe 因子** | **8** | **~2.5** | **~20** | **Day 3 凌晨** |
| **M. ConvNeXt-T IN100-LT 侦察** | **4** | **~7** | **~28** | **Day 3 上午** |
| Buffer / rerun | ~4 | ~2–6 | ~16 | Day 3 下午 |
| **合计** | **~78** | | **~242** | |

**可用 GPU-hours**：4 GPU × 3.5 天 × 24h = **336 GPU-h**  
**训练利用率**：242 / 336 ≈ **72%**（训练占用），剩余时间为 I/O、数据预处理、logit 提取、buffer/rerun  
**实际总负载**：含后处理 I/O 约 280–300 有效 GPU-h，**利用率 ~85–90%**

> **2 卡方案**：如果只有 2 张 GPU，总可用 168 GPU-h。需要精简 E 组（DeiT-S → 砍掉）、G 组（LT baselines 只保留 LogitAdj 2 runs）、M 组（ConvNeXt IN100 → 砍掉），总 GPU-h 降至 ~160h，刚好在 3.5 天内完成。

### 9.2 后处理分析工作量（Agent 自动化，CPU 并行）

| 分析任务 | 覆盖范围 | 估时 |
|---------|---------|------|
| Per-class $T_k^*$ 估计 + bootstrap CI | 所有 ~78 新 runs + 已有 ~30 runs | 自动化后约 3h |
| Class-wise confidence-accuracy gap | 所有 ~108 runs | 自动化后约 1.5h |
| Class-wise ECE + reliability diagrams | 选取 ~20 个代表性配置 | 自动化后约 1.5h |
| Logit norm / entropy / margin 统计 | 所有 ~108 runs | 自动化后约 1.5h |
| CDA-TS / Dual-Branch TS / Per-class TS 评估 | ~10 个关键配置 | 实现代码约 4h，运行约 1h |
| 三因子 ANOVA（Balanced 实验） | 24 runs | 约 0.5h |
| Permutation tests 更新 | 关键对比组 | 约 0.5h |
| 实验注册表自动生成 | 所有 runs | 自动化后约 0.5h |
| CIFAR-100-LT 因子分析 + RN50/Swin-T IN100 多 seed 分析 + ConvNeXt IN100 分析 | K/L/M 组 ~20 runs | 约 2h |
| **合计** | | **~16h Agent 时间** |

### 9.3 写作工作量

| 写作任务 | 估时 |
|---------|------|
| P0-2 claim 收缩 | 1h |
| P0-3 页数压缩 + 表格重组 | 2h |
| P1-3 数据划分协议 | 0.5h |
| P1-5 practical implication 叙事 | 2h |
| P1-6 概念定位 | 0.5h |
| P1-7 $T_k^*$ 可靠性说明 | 0.5h |
| P1-4 experiment ID 清理 | 1h |
| Appendix 核心结论主文化 | 1h |
| 整合新增结果到主文（balanced / ConvNeXt / ImageNet 多 seed×多 arch / IR 细密 / CIFAR-100 因子 / baselines） | 5h |
| 更新 Abstract / Introduction / Discussion / Conclusion | 3h |
| 新增 / 更新 Appendix 内容 | 3h |
| 最终 polish + 一致性检查 | 2h |
| **合计** | **~21h Agent 时间** |

### 9.4 总工作量对比

| 方案版本 | 新增训练 runs | GPU-hours | 分析/写作时间 | 总人天 |
|---------|-------------|----------|------------|-------|
| 原始方案（旧版 PDF） | ~50+ | ~200+ | ~5 天 | ~7 天 |
| 首版更新方案（May 3 版） | 4 | ~18 | ~2–3 天 | ~3 天 |
| 饱和方案 v1 | ~58 | ~150 | ~32h Agent | 3.5 天 |
| **饱和方案 v2（本版）** | **~78** | **~242** | **~36h Agent** | **3.5 天** |

首版更新方案的工作量不饱和（4 runs 用 4 张 5090 半天即完成）。饱和方案 v1 将 GPU 利用率提升到 ~45%，但 Day 2 下午和 Day 3 仍有大量空闲。**v2 通过新增 K/L/M 三组实验填满 Day 2 下午至 Day 3 上午**，将训练利用率从 45% 提升到 **72%**（含 I/O 达 ~85–90%），同时进一步强化了 ImageNet-100-LT 跨架构 seed 覆盖（K 组）、CIFAR-100-LT 因子完整性（L 组）、和 ConvNeXt 跨 scale 验证（M 组）。

---

## 十、最终建议

### 10.1 饱和方案的核心收益

相比首版更新方案（4 runs），饱和方案 v2（~78 runs）在 3.5 天内将论文从**以下六个维度全面升级**：

| 维度 | 首版方案（4 runs） | 饱和方案 v2（~78 runs） | 审稿感知提升 |
|------|-------------------|---------------------|------------|
| **交互效应证据** | Balanced CIFAR-10 × 2 runs | Balanced CIFAR-10 全因子（12）+ CIFAR-100（4）= 16 runs → 三因子 ANOVA | 从 "单次对照" 到 "统计可靠的交互分析" |
| **跨数据集鲁棒性** | ImageNet-100-LT 补 2 runs | ViT 全因子 3 seeds + **RN50/Swin-T 各补 2 seeds** = 20 runs（含已有 4） | 从 "scout" 到 "3 arch × 3 seeds 完整矩阵" |
| **架构覆盖** | 3 架构不变 | + ConvNeXt-T（12 CIFAR + **4 IN100**）+ DeiT-S（4）= 4–5 种架构，**ConvNeXt 跨 scale** | 从 "ViT/CNN/Swin" 到 "含现代 CNN + 跨 scale 验证" |
| **CIFAR-100 因子完整性** | 仅 Standard/Vanilla | **+ Mixup-only + LS-only 共 8 runs → 完整 4-recipe 因子** | 从 "100 类仅 2 recipe" 到 "与 CIFAR-10 对等" |
| **Post-hoc baseline** | 仅 Global TS + 2-group TS | + CDA-TS + Dual-Branch TS + Per-class TS | 从 "一个不 work 的 remedy" 到 "系统性 baseline 比较" |
| **Mechanistic insight** | 仅 logit norm | Logit norm + entropy + max confidence + logit margin，跨所有架构 | 从 "pure phenomenology" 到 "有 mechanistic 线索" |

### 10.2 执行后论文的预期状态

如果饱和方案 v2 完整执行，论文将具备以下实验覆盖：

- **5 种架构**：ViT-Small、ResNet-50、Swin-T、ConvNeXt-T、DeiT-S
- **4 个数据集**：CIFAR-10-LT、CIFAR-100-LT（**完整 4-recipe 因子**）、ImageNet-100-LT（**3 架构 × 3 seeds**）+ balanced CIFAR-10/100 对照
- **9 个 IR 点**：10, 15, 20, 25, 30, 40, 50, 100, 200
- **6+ 种 post-hoc calibration 方法比较**：Global TS、2-group TS、CDA-TS、Dual-Branch TS、Per-class TS
- **3 种长尾训练方法校准分析**：Logit Adjustment、Balanced Softmax、cRT
- **ConvNeXt-T 跨 scale 验证**：CIFAR（12 runs）+ ImageNet-100-LT（4 runs）
- **全模型 class-wise reliability 验证**：direction agreement rate 汇总
- **全模型 mechanistic 分析**：logit norm + entropy 分组统计
- **GPU 训练利用率**：~242h / 336h ≈ **72%**，含 I/O 达 **~85–90%**

这个覆盖面在 NeurIPS diagnostic study 中属于**非常充分**的水平，且 4 × RTX 5090 在 3.5 天内的计算资源得到高效利用。

### 10.3 中稿概率评估

| 状态 | 预估审稿分数区间 | 主要风险 |
|------|----------------|---------|
| 当前 May 3 版（不修改） | Borderline → Weak Reject | $T_k^*$ 逻辑漏洞 + 无方法贡献 + ImageNet 单 seed |
| 首版更新方案（4 runs） | Weak Reject → Weak Accept | 逻辑漏洞堵住但实验覆盖偏薄 |
| **饱和方案 v2（~78 runs）** | **Weak Accept → Accept** | 剩余风险主要是 "diagnostic vs method" 的 framing |

### 10.4 最大的剩余风险

即使饱和方案完整执行，论文最大的单点风险仍然是 **"无方法贡献" 的审稿印象**。应对策略：

1. **不要勉强提出新方法**——这会偏离论文的 diagnostic 定位，且仓促的方法容易被更挑剔地审视
2. **通过 CDA-TS baseline 对比展示诊断的实用价值**——如果 CDA-TS 在 bidirectional 场景下显著优于 Global TS，论文的 takeaway 变为"我们的诊断揭示了为什么 class-frequency-aware 方法是必要的"
3. **通过 2-group TS 的 oracle/predicted 对比激发 future work**——明确指出 routing 是一个 open problem，为后续方法研究打开空间
4. **在 Introduction 和 Conclusion 中强调**：这篇论文改变了 practitioners 应该如何审计和校准长尾模型——不是"只看 ECE"，而是"必须做 per-class $T_k^*$ 诊断"

### 10.5 关键执行提醒

1. **Day 0 的自动化流水线搭建是成功的前提**。如果每个 run 的后处理（logit 提取 → $T_k^*$ 估计 → bootstrap CI → reliability 分析）需要手动操作，3.5 天内无法消化 78 runs 的结果。必须在 Day 0 完成端到端自动化。

2. **Balanced CIFAR 结果的应急预案必须提前准备**（见 P1-1）。Day 1 晚间即可获得 balanced 结果；如果出现非预期场景，需要 Day 2 调整叙事方向，不能等到 Day 3。

3. **ConvNeXt-T 的 CIFAR 配置可能需要调整**（stem kernel size、learning rate 等）。建议 Day 0 先跑一个 quick sanity check（5 epochs），确认训练不 diverge。

4. **主文 ≤ 9 页是硬约束**。即使新增大量结果，主文应保持精简——新增实验的详细数据进 Appendix，主文只报告核心数字和结论。选择性而非穷尽性地报告是关键。

5. **Agent Auto Research 模式下的质量检查**：自动化流水线应在每个 run 完成后自动检查：(a) 训练 loss 是否收敛；(b) test accuracy 是否在合理范围内；(c) $T_k^*$ 是否有 NaN 或全部 hit optimizer bound。异常 run 自动标记并安排 rerun（Day 2–3 buffer 时间）。
