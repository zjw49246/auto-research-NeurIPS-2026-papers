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

### 4.4 P3：如有余力可做（边际收益）

#### P3-1：IR 细密扫描（IR=15, 30）

当前 5 个 IR 点已足够。如有余力加 IR=15 和 IR=30 可精确定位转折点，但审稿人不会因此给负面评价。

#### P3-2：ConvNeXt-T 或 DeiT-S

增加现代架构可提升 generality 声明，但当前 3 架构（ViT / CNN / Swin-T）已形成完整的 local-vs-global 梯度。除非审稿人明确质疑，否则属于 nice-to-have。

#### P3-3：Full ImageNet-LT / iNaturalist-LT

从 ImageNet-100-LT 扩展到完整 ImageNet-LT 或 iNaturalist-LT 能显著增强 benchmark 说服力，但计算成本高。论文定位为 diagnostic study，当前 3 个 LT 数据集已足够。

#### P3-4：长尾训练 baseline（Logit Adjustment / Balanced Softmax 等）

论文不是提出新方法，而是揭示训练 recipe 对校准方向的影响。与长尾训练方法的对比有价值，但不是当前 framing 的核心需求。如果篇幅允许，可在 Appendix 中简要分析 1–2 个。

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

## 七、更新后的优先级总表

| 优先级 | 项目 | 状态 | 解决的审稿风险 | 预估成本 |
|--------|------|------|------------|---------|
| **P0** | Class-wise reliability diagram + confidence-accuracy gap 验证 $T_k^*$（含 binning 噪声应对策略） | **未做** | 核心逻辑漏洞 | **极低（半天–1天）** |
| **P0** | 收缩 "consistently predicts" 等过强表述 | 部分改善 | Reviewer 攻击 claim | 极低（2h） |
| **P0** | 确认主文 ≤ 9 页（Table 5 移入 Appendix 等） | 边界风险 | Desk rejection | 低（半天） |
| **P1** | Balanced CIFAR 对照（2 runs）+ 结果不如预期的应急预案 | **未做** | Mixup × imbalance 交互 | 低（1 天） |
| **P1** | ImageNet-100-LT 补 1–2 seeds（Standard + Vanilla） | **未做** | Cross-dataset robustness | 中（依赖算力） |
| **P1** | 强化 "practical implication" 叙事：框定 2-group TS 角色 + 讨论 routing 方向 + 可选 CDA-TS | **未做** | **"无方法贡献"拒稿风险** | 低–中 |
| **P1** | 主文 §3 明确数据划分协议 | 未明示 | Test-set leakage | 极低 |
| **P1** | 清理 experiment ID 污染痕迹 | 有改善 | Reproducibility 信任 | 低 |
| **P1** | Related Work 中定位 $T_k^*$ 概念框架 | 未做 | 校准领域审稿人 | 极低 |
| **P1** | 显式说明 $T_k^*$ 点估计在尾部类的可靠性处理（CI-confirmed 设计选择） | 未显式说明 | 框架可靠性质疑 | 极低 |
| **P1** | Appendix 核心结论主文化（每个 Appendix 加 1 句主文总结） | 部分覆盖 | 审稿人不读 Appendix | 极低 |
| **P2** | Appendix G → 主文 Figure（bar chart） | 数字已引用 | Practical relevance 可视化 | 极低（2h） |
| **P2** | Logit norm by class probe | **未做** | "Phenomenology only" | 极低（半天） |
| **P2** | 补充参考文献（Nixon et al. 等） | 未做 | 引用完整性 | 极低 |
| **P3** | IR 细密扫描 | 已有 5 点 | 边际 | 低 |
| **P3** | ConvNeXt-T / DeiT-S | 未做 | Architecture generality | 中–高 |
| **P3** | Full ImageNet-LT / iNaturalist-LT | 未做 | Benchmark strength | 高 |

---

## 八、建议执行时间线

### 第 1 天（约 6–8 小时）—— 堵漏洞 + 文字修改
- [ ] **P0-1**：制作 class-wise confidence-accuracy gap 定量表（所有 10 类）+ 2–4 个代表性类的 reliability diagram（使用 5-bin equal-mass binning）。如果 CIFAR-10-LT 噪声过大，标记为"待 ImageNet-100-LT 结果替换"
- [ ] **P0-2**：修改所有过强 claim（全文搜索 "consistently predicts" / "determines" / "drives" / "the consistent factor"）
- [ ] **P0-3**：将 Table 5 移入 Appendix；Table 2 内联到文字；精简 §5.2；确认 ≤ 9 页
- [ ] **P1-5 策略 1+2**：在 §5.1 中框定 2-group TS 为诊断上界；添加 2–3 句 routing 方向讨论
- [ ] **P1-6**：在 §2.1 中补充 $T_k^*$ 概念定位段落（top-label vs class-conditional）
- [ ] **P1-7**：在 §3 中添加 CI-confirmed 设计选择的方法论说明

### 第 2 天（约 4–6 小时）—— 启动实验 + 文字完善
- [ ] **P1-1**：启动 balanced CIFAR-10 的 2 个训练 run（Vanilla + Standard）
- [ ] **P1-3**：在 §3 中补充数据划分协议说明
- [ ] **P1-4**：清理 Table 7 脚注措辞；建立实验注册表
- [ ] **§5.4 策略**：为每个 Appendix 的核心结论在主文中添加 1 句话总结

### 第 3–4 天 —— 实验分析 + 可视化
- [ ] **P1-2**：启动 ImageNet-100-LT 补充 seed 训练（Standard + Vanilla，seed 123）
- [ ] **P1-1 完成**：分析 balanced CIFAR 结果。如果结果符合预期，写入主文；如果出现场景 A，按应急预案调整叙事
- [ ] **P2-1**：制作 Global TS vs. 2-group TS bar chart（数据来自 Appendix G Table 12/13）
- [ ] **P2-2**：提取 logit norms，分析 head/tail 差异

> **⚠️ ImageNet-100-LT 算力说明**：ViT-Small 在 ImageNet-100-LT (224×224) 上的训练通常需要 10–20 GPU-hours per run（取决于 GPU 型号）。如果只有单卡，2 个 run 可能需要 2–3 天。**算力不足时的降级方案**：优先只补 Standard 的 1 个 seed（最低限度验证 bidirectional 方向一致性），Vanilla 留到 rebuttal 期间补充。

### 第 5 天 —— 收尾
- [ ] **P1-2 完成**：分析 ImageNet-100-LT 新 seed 结果，更新 Table 4，报告 cross-seed |Δ| 和方向一致性
- [ ] **P1-5 策略 3**：如有余力，实现 CDA-TS baseline 并与 Global TS / 2-group TS 对比
- [ ] **P2-4**：补充参考文献（Nixon et al. 2019 等）
- [ ] 最终排版检查、PDF fonts、匿名性检查、页数确认

---

## 九、新增实验与分析工作量估算

| 项目 | 训练 runs | 后处理/写作 | 说明 |
|------|----------|----------|------|
| Class-wise reliability + confidence-accuracy gap | 0 | 对已有模型输出分析 | 含 binning 策略调优 |
| Balanced CIFAR-10 × 2 recipes | 2 | Per-class $T_k^*$ 估计 + 应急预案判断 | CIFAR 训练快，约数小时 |
| ImageNet-100-LT × 2 recipes × 1 seed | 2 | Per-class $T_k^*$ 估计 | 10–20 GPU-hours/run |
| Logit norm analysis | 0 | 对已有 logits 统计 | |
| CDA-TS baseline（可选 P1-5） | 0（post-hoc） | 实现 + 评估 | 需适配代码 |
| "Practical implication" 叙事强化 | 0 | §5.1 文字修改 | 框定 2-group TS + routing 讨论 |
| $T_k^*$ 可靠性说明 | 0 | §3 + Limitations 文字补充 | |
| Appendix 核心结论主文化 | 0 | 6–7 句话分布在各 section | |
| Claim 收缩 + 页数压缩 | 0 | 全文文字修改 | |
| **合计新增训练 runs** | **4** | | |
| **合计纯文字/分析工作** | — | **约 2–3 人天** | |

对比原方案暗示的 ~50+ runs 和修正前方案的 ~10-15 runs，**当前实际只需 4 个新训练 run + 约 2–3 人天的分析和文字工作**。实验工作量极小是因为 May 3 版已完成了绝大部分实验；剩余缺口主要集中在**分析层**（class-wise reliability 验证、logit norm probe）、**叙事层**（claim 收缩、practical implication 框定、$T_k^*$ 可靠性说明）和**呈现层**（Appendix 核心结论主文化、页数压缩）。

---

## 十、最终建议

May 3 版论文的实验基础已经**显著强于旧版**。剩余的改进空间主要在**四个维度**：

1. **逻辑完整性**（P0-1 + P1-7）：class-wise reliability 验证 + $T_k^*$ 可靠性说明——堵住诊断框架的两个逻辑漏洞。其中 P0-1 是全方案**成本最低、价值最高**的单项改进，但需注意 CIFAR-10-LT 每类仅 200 样本的 binning 噪声风险，建议以 confidence-accuracy gap 定量指标为主、reliability diagram 为辅的双保险策略。

2. **叙事精确性**（P0-2 + P1-1 + P1-5）：claim 收缩 + balanced 对照 + practical implication 强化。三者共同将论文叙事从 "Mixup drives underconfidence" 精确为 "Mixup × imbalance creates directional conflict, with practical consequences for post-hoc calibration"。其中 P1-1（balanced 对照）需要准备结果不如预期的应急预案；P1-5（practical implication）是应对 "无方法贡献" 拒稿风险的关键防线——当前 2-group TS 在 predicted routing 下不可用的问题必须在 Discussion 中主动讨论。

3. **鲁棒性收尾**（P1-2 + P1-3 + P1-4）：ImageNet 补 seed、明确协议、清理痕迹。其中 ImageNet-100-LT 补 seed 依赖算力，应提前评估可行性并准备降级方案。

4. **呈现效能**（§5.4 Appendix 主文化 + P0-3 页数压缩）：论文有大量高质量分析埋在 Appendix 中（118 checkpoint stability、跨架构 per-class 结果、global TS 失败证据），审稿人大概率不会仔细阅读。每个 Appendix 的核心结论都应在主文中用一句话总结，确保审稿人感受到分析的深度和广度。

**关键判断**：如果 P0 全部完成 + P1 完成 5 项以上（特别是 P1-1 balanced 对照、P1-5 practical implication 强化、P1-7 $T_k^*$ 可靠性说明），论文的接收概率将从当前的 borderline 区间显著提升到 **weak accept 或更强**。

**最大的单点风险**不是实验不够，而是 **"无方法贡献" 的审稿印象**。应对此风险的核心策略不是勉强提出一个新方法（这会偏离论文的 diagnostic 定位），而是通过三管齐下（框定 2-group TS 角色 + 讨论 routing 方向 + 可选 CDA-TS baseline）让审稿人认识到：这个诊断发现**改变了 practitioners 应该如何选择 post-hoc calibration 策略**——这本身就是一个有实践影响的贡献。
