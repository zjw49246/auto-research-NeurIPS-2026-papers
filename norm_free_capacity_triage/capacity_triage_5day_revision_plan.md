# Capacity Triage 论文 5 天提升方案（NeurIPS 2026 冲刺版）

日期：2026-05-02  
截止时间：2026-05-06  
资源假设：2–4 张 RTX 5090 / RTX PRO 6000，可用 agent auto research 持续跑实验、写代码、画图和改论文。  
目标：把论文从“有趣但证据偏薄的 empirical observation”提升为“叙事清晰、实验扎实、带有初步机制解释的系统性经验研究”。

---

## 0. 总体判断

当前论文最有价值的点不是“BN 比 GN 准”，而是：

> Normalization choice may shape how limited model capacity is allocated across classes, not merely how high the aggregate accuracy becomes.

也就是说，论文应该避免被写成一篇普通的 BN-vs-GN 性能比较，而应升级成：

> 一篇研究 normalization 如何影响 class-level performance allocation / class-learning trajectory 的系统性经验论文。

目前最危险的 claim 是：

> We discovered a sharp d/K threshold around 0.8.

这个说法过强。更稳妥、更容易被 reviewer 接受的主线是：

> In width-varying CNNs, the reproducibility of normalization-induced class-level allocation effects increases as representation capacity grows relative to class count. We use d/K as a lightweight empirical proxy for this capacity-to-class-count ratio.

换句话说，`d/K` 不应被包装成普适定律，而应被定义为一个经验 proxy。

---

## 1. 修改后的核心叙事

### 1.1 当前叙事的问题

当前论文把四件不同事情混在了一起：

1. BN 提升 overall accuracy；
2. BN final raw Gini 更低，说明最终模型 per-class accuracy 更均匀；
3. BN matched-accuracy Gini 更高，说明在相同 overall accuracy 的训练阶段，BN 的 class gains 更集中；
4. matched-accuracy effect 在高 `d/K` 下更稳定。

这些结果可以共存，但必须明确区分：

- **Final outcome question**：训练结束后，BN/GN 谁的 per-class accuracy 更均匀？
- **Trajectory question**：在相同 overall accuracy 的训练阶段，BN/GN 是否通过不同的 class-learning path 达到该 accuracy？
- **Reproducibility question**：这种 class-level path difference 是否跨 seed 稳定？

如果不区分，读者会困惑：

> BN 到底是让 per-class accuracy 更均匀，还是让收益更集中？

### 1.2 修改后的统一解释

建议全文采用如下解释：

> BN 最终模型通常更强，也更均匀；但在达到相同 aggregate accuracy 的过程中，BN 和 GN 的 class-learning trajectory 不同。BN 可能先更集中地提升一部分类别，然后在训练后期让弱类别追上来。因此 final raw Gini 和 matched-accuracy Gini 不矛盾，它们分别描述最终结果和学习路径。

英文可写成：

> Final uniformity and matched-accuracy triage are not contradictory: they describe different stages of learning. BN may allocate gains unevenly along the learning trajectory, while its stronger convergence eventually allows weaker classes to catch up, producing a more uniform final profile.

### 1.3 修改后的贡献表述

建议 contributions 改成四点：

1. **Phenomenon**：展示 normalization choice 不仅影响 aggregate accuracy，也影响 class-level performance profiles。
2. **Methodology**：提出 accuracy-conditioned / matched-accuracy Gini curve analysis，用于区分 final outcome 和 learning trajectory。
3. **Capacity dependence**：通过 width sweeps 展示 class-level allocation effect 的 reproducibility 随 capacity-to-class-count proxy `d/K` 增强。
4. **Mechanistic probes**：用 class-learning curves、margin distribution、feature geometry、batch-size controls 提供初步机制证据。

不要把贡献写成：

> We find a universal threshold at d/K≈0.8.

应写成：

> We observe a reproducibility crossover around d/K≈0.7–1.0 in our controlled small-CNN setting.

---

## 2. 5 天内实验优先级

下面按重要性排序。P0 必须完成；P1 强烈建议完成；P2 有余力再做。

---

# P0：必须完成

P0 的目标是解决 reviewer 最可能抓住的硬伤：

- seed 太少；
- threshold 附近证据不稳；
- learning rate confound；
- matched-accuracy 方法太脆；
- BN/GN 机制差异没有 batch-statistics 对照。

---

## P0.1 CIFAR-100 ScaleCNN dense width + 10 seeds

### 目的

验证 `d/K` reproducibility crossover 不是 3 seeds 的偶然结果。

当前完整 width curve 主要依赖 3 seeds，而核心 transition 出现在 c=8/10/12 附近。这是最脆弱的地方。

### 实验设置

Dataset：CIFAR-100  
Model：ScaleCNN  
Norm：BN vs GN  
Seeds：10  
Widths：

| c | d=8c | d/K |
|---:|---:|---:|
| 4 | 32 | 0.32 |
| 6 | 48 | 0.48 |
| 7 | 56 | 0.56 |
| 8 | 64 | 0.64 |
| 9 | 72 | 0.72 |
| 10 | 80 | 0.80 |
| 11 | 88 | 0.88 |
| 12 | 96 | 0.96 |
| 14 | 112 | 1.12 |
| 16 | 128 | 1.28 |
| 24 | 192 | 1.92 |
| 32 | 256 | 2.56 |
| 48 | 384 | 3.84 |

如果时间紧，至少做：

`c = 6, 7, 8, 9, 10, 11, 12, 14, 16`

因为这些点直接决定 crossover 是否可信。

### 指标

每个 run 保存：

- final overall accuracy；
- final raw Gini；
- per-class accuracy by epoch；
- matched-accuracy Gini；
- accuracy-conditioned Gini AUC；
- class-wise BN-GN delta vector；
- seed-wise sign consistency；
- confidence interval；
- SNR。

### 预期产出

新增主文图：

> Dense validation of the reproducibility crossover.

图中至少包含：

- x-axis：`d/K`
- y-axis：class-wise allocation stability 或 Gini delta SNR
- error bar / bootstrap CI
- 不写 sharp threshold，写 crossover region。

### 推荐结论表述

如果结果稳定：

> The transition is better described as a crossover region rather than a sharp threshold. In CIFAR-100 ScaleCNN, reproducibility increases rapidly around d/K≈0.7–1.0.

如果结果不够稳定：

> The exact crossover location is seed-sensitive, but the broader trend remains: higher d/K produces more reproducible class-level allocation effects.

---

## P0.2 Learning-rate control

### 目的

消除 BN 用 `lr=0.1`、GN 用 `lr=0.05` 的 confound。

这是最容易被严格 reviewer 攻击的点。因为 matched-accuracy Gini 本身依赖 training trajectory，而 learning rate 会直接改变 trajectory。

### 实验设置

Dataset：CIFAR-100  
Model：ScaleCNN  
Widths：

`c = 8, 10, 12, 16, 32`

Norm：BN vs GN  
LR setting：

1. BN lr=0.05, GN lr=0.05；
2. BN lr=0.1, GN lr=0.1；
3. BN best LR, GN best LR；
4. 可选：BN/GN lr ∈ {0.025, 0.05, 0.1, 0.2} full sweep。

Seeds：至少 5。

### 指标

- final accuracy gap；
- final raw Gini gap；
- matched-accuracy Gini delta；
- accuracy-conditioned Gini AUC；
- class-wise delta vector stability；
- crossover location under each LR setting。

### 主文写法

新增小节：

> Learning-rate-controlled validation around the crossover

要明确回答：

1. 同 LR 下 BN/GN accuracy gap 是否还在？
2. 同 LR 下 matched-accuracy allocation effect 是否还在？
3. crossover 是否依然存在？
4. 如果 exact crossover location 变化，是否影响主结论？

### 推荐结论表述

理想结果：

> Although the exact magnitude of the matched-accuracy Gini delta depends on learning rate, the high-d/K reproducibility of BN-induced allocation remains under matched-LR controls.

如果低 d/K 结果受影响：

> Learning rate affects quantitative estimates near the low-d/K regime, so we restrict strong quantitative claims to widths where the effect remains robust under matched-LR controls.

---

## P0.3 Accuracy-conditioned Gini curve analysis

### 目的

把原来的“单点 matched-accuracy Gini”升级为更稳健的“曲线级分析”。

当前方法用 GN final accuracy 去找 BN 的中间 checkpoint。这个设计有价值，但容易被质疑：

> 你比较的是 early BN 和 converged GN，不公平。

### 改进方法

对每个 seed、每个 norm，构建：

\[
G(a)
\]

即 Gini as a function of overall accuracy。

在 BN 和 GN 的 shared accuracy interval 内，采样 20 或 50 个 accuracy bins：

\[
A_{\text{shared}} = [\max(A_{\min}^{BN}, A_{\min}^{GN}), \min(A_{\max}^{BN}, A_{\max}^{GN})]
\]

计算：

\[
\Delta G(a) = G_{BN}(a) - G_{GN}(a)
\]

报告：

1. mean \(\Delta G(a)\)；
2. area-under-\(\Delta G\)；
3. fraction of bins where \(\Delta G(a)>0\)；
4. seed-wise consistency；
5. bootstrap CI。

### 新指标命名

建议叫：

> Accuracy-conditioned Gini AUC

或者：

> Trajectory-level Gini difference

### 主文图

新增一张关键图：

- x-axis：overall accuracy；
- y-axis：Gini；
- BN/GN 两条曲线；
- shared accuracy range shaded；
- 显示 \(\Delta G(a)\) 的方向。

这张图应替代或弱化原来的 epoch-vs-Gini 图。

### 推荐结论表述

> Single-point matched-accuracy estimates can be sensitive to checkpoint choice. We therefore compare BN and GN over the full shared accuracy interval. The trajectory-level Gini difference remains directionally consistent at high d/K, indicating that the effect is not an artifact of one matched epoch.

---

## P0.4 Batch size sweep

### 目的

验证 BN 的 class-level effect 是否和 batch statistics 有关。

因为 BN 和 GN 的核心差异是 BN 使用 batch dimension statistics，缺少 batch-size sweep 会让机制解释很弱。

### 实验设置

Dataset：CIFAR-100  
Model：ScaleCNN  
Widths：

`c = 8, 10, 16, 32`

Norm：BN vs GN  
Batch size：

`32, 64, 128, 256`

Seeds：3–5。

学习率建议：

- 主实验使用 batch-size-scaled LR；
- 另做一个固定 LR sanity check，如果资源允许。

### 指标

- final accuracy；
- raw Gini；
- accuracy-conditioned Gini AUC；
- class-wise delta vector stability；
- margin delta stability；
- BN-GN effect magnitude vs batch size。

### 可能结论

如果 effect 随 batch size 系统变化：

> This supports the hypothesis that batch-level statistics contribute to class-level allocation dynamics.

如果 effect 不随 batch size 明显变化：

> This suggests that the observed effect is not merely a small-batch noise artifact, although it does not rule out a role for batch statistics.

二者都能写，但要避免过强因果 claim。

---

## P0.5 Tiny-ImageNet 补低 d/K

### 目的

验证 `d/K` crossover 不是 CIFAR-100 特有。

当前 Tiny-ImageNet 只测了 c=32、48，两个点都在高 `d/K`。这只能说明 high-d/K direction consistency，不能验证 crossover。

### 实验设置

Dataset：Tiny-ImageNet  
Model：ScaleCNN  
Norm：BN vs GN  
Widths：

| c | d=8c | K | d/K |
|---:|---:|---:|---:|
| 8 | 64 | 200 | 0.32 |
| 16 | 128 | 200 | 0.64 |
| 24 | 192 | 200 | 0.96 |
| 32 | 256 | 200 | 1.28 |
| 48 | 384 | 200 | 1.92 |

Seeds：3–5。

### 指标

同 CIFAR-100：

- overall accuracy；
- final raw Gini；
- matched / accuracy-conditioned Gini；
- class-wise delta stability；
- margin stability。

### 结果解释

如果 Tiny-ImageNet 也在 d/K≈0.7–1.0 附近出现 reproducibility 增强：

> Strong support for d/K as a cross-K empirical proxy.

如果 Tiny-ImageNet 不复现：

> d/K alone is insufficient; dataset difficulty and class granularity also modulate the effect.

后者也可以接受，但需要把主 claim 降调。

---

# P1：强烈建议完成

P1 的目标是把论文从“ScaleCNN 单模型现象”提升到“系统性经验研究”。

---

## P1.1 Class-wise delta vector stability

### 目的

直接证明什么叫 deterministic triage。

Gini 是一个 scalar。Reviewer 会问：

> 你说 deterministic，到底哪些东西 deterministic？

应该直接看每个 seed 下 BN-GN 的 per-class delta vector：

\[
v_s = [Acc^{BN}_{1,s}-Acc^{GN}_{1,s}, ..., Acc^{BN}_{K,s}-Acc^{GN}_{K,s}]
\]

计算不同 seeds 之间的：

- Pearson correlation；
- Spearman correlation；
- cosine similarity；
- top-k helped classes overlap；
- bottom-k hurt classes overlap。

### 预期图

新增主文图：

> Class-wise allocation signatures become seed-stable above the crossover.

可以画两种：

1. x-axis：`d/K`，y-axis：mean pairwise Spearman correlation of \(v_s\)；
2. heatmap：rows 为 seeds，columns 为 classes，values 为 BN-GN per-class delta，左边 low d/K，右边 high d/K。

### 推荐结论表述

> Above the crossover, not only the scalar Gini delta but also the identity of helped and hurt classes becomes more reproducible across seeds.

这会显著增强 “deterministic triage” 的可信度。

---

## P1.2 Per-class learning curves and time-to-threshold

### 目的

解释 final raw Gini 和 matched-accuracy Gini 为什么不矛盾。

核心假设：

> BN changes class learning order. At matched aggregate accuracy, it may learn some classes earlier or more aggressively; by convergence, weaker classes catch up, producing lower final raw Gini.

### 分析方法

对每个 class 计算：

- accuracy-over-epoch curve；
- time-to-30% accuracy；
- time-to-50% accuracy；
- time-to-70% accuracy；
- time-to-final-80% relative performance；
- area under per-class learning curve。

定义 class difficulty：

- GN final accuracy；
- None final accuracy；
- early epoch accuracy；
- class confusion rate。

看：

- BN 是否更早学会 easy classes；
- BN 后期是否补 hard classes；
- class difficulty 是否预测 BN gain；
- high d/K 下 learning order 是否更稳定。

### 推荐主文结论

> The matched-accuracy concentration arises because BN and GN reach the same aggregate accuracy through different class-learning orders. BN's trajectory initially concentrates gains on a more reproducible subset of classes, while late training improves weaker classes and yields a more uniform final model.

---

## P1.3 Margin distribution and margin stability

### 目的

提供最有说服力的初步机制解释。

分类准确率和 margin 直接相关。建议重点做 margin，而不是过多堆复杂 feature 指标。

### 定义

对样本 \(x\)：

\[
m(x)=z_y(x)-\max_{j\neq y}z_j(x)
\]

对每个 class：

\[
M_k = \mathbb{E}_{x\in k}[m(x)]
\]

BN-GN margin delta：

\[
\Delta M_k = M^{BN}_k - M^{GN}_k
\]

### 分析

1. \(\Delta M_k\) 与 \(\Delta Acc_k\) 的相关性；
2. \(\Delta M\) vector 的 seed-wise correlation；
3. margin Gini / margin variance；
4. low-margin classes 是否在 BN 下后期 catch up；
5. margin stability 是否和 accuracy stability 同步出现。

### 推荐主文结论

> The output-level allocation transition is mirrored by a margin-level transition: above the crossover, BN-GN class-wise margin deltas become more seed-stable and correlate with class-wise accuracy deltas.

注意措辞：

不要说：

> This proves BN causes triage through margins.

应说：

> This suggests that the allocation effect is reflected in decision-boundary geometry.

---

## P1.4 Feature geometry diagnostics

### 目的

把机制解释从输出层扩展到 representation 层。

### 指标

取最后一层 feature \(h(x)\)，计算：

1. feature effective rank；
2. class mean norm；
3. class mean pairwise distance；
4. nearest class center distance；
5. intra-class variance；
6. inter-class / intra-class ratio；
7. optional neural collapse proxy。

重点推荐：

\[
D_k = \min_{j\neq k} ||\mu_k-\mu_j||
\]

即每个 class 到最近 class center 的距离。

然后看：

\[
\Delta D_k = D^{BN}_k - D^{GN}_k
\]

与：

\[
\Delta Acc_k
\]

是否相关。

### 推荐主文结论

> Feature-level diagnostics show that high-d/K allocation stability is accompanied by more stable class prototype separation and margin geometry.

同样不要 claim causality。

---

## P1.5 标准架构扩展

### 目的

缓解 ScaleCNN 是 custom architecture 的质疑。

### 优先级

1. ResNet-small：增加 seeds、补完整 matched-accuracy / accuracy-conditioned analysis；
2. CIFAR ResNet-18；
3. WideResNet-16-2 或 WideResNet-28-2。

### 最小设置

Dataset：CIFAR-100  
Norm：BN vs GN vs None  
Seeds：3–5  
Capacity points：至少 2 个，如果能做 3 个更好。

### 指标

- final accuracy；
- raw Gini；
- accuracy-conditioned Gini AUC；
- class-wise delta stability；
- margin stability。

### 推荐结论

> The direction of the allocation effect is not specific to ScaleCNN, although the exact crossover location and magnitude are architecture-dependent.

这句话很重要：承认 architecture dependence，避免 overclaim。

---

## P1.6 多指标 robustness

### 目的

避免 reviewer 说 Gini 是单指标 artifact。

### 指标

除 Gini 外，补充：

- per-class accuracy standard deviation；
- worst-class accuracy；
- worst-10% class accuracy；
- bottom-decile mean accuracy；
- top-bottom class accuracy gap；
- Theil index；
- Jain’s fairness index；
- class-rank stability。

### 主文表格

| Metric | Final BN more uniform? | Trajectory effect detected? | Reproducibility increases with d/K? |
|---|---|---|---|
| Gini | Yes | Yes | Yes |
| Std | Yes/partial | Yes | Yes |
| Worst-10% | Yes/partial | Partial | Yes/partial |
| Theil | Yes | Yes | Yes |
| Rank stability | N/A | Yes | Yes |

### 推荐结论

> The observed trend is not specific to the Gini coefficient; alternative class-level disparity metrics show directionally consistent patterns.

---

# P2：有余力再做

---

## P2.1 CIFAR-100-C / robustness difficulty axis

### 目的

回答测试集难度是否影响 class-level disparity。

### 设置

训练仍用 CIFAR-100 clean。  
评估用 CIFAR-100-C。

Corruptions 可选：

- Gaussian noise；
- motion blur；
- fog；
- contrast；
- JPEG compression。

Severity：1–5。

### 指标

- overall corruption accuracy；
- per-class Gini；
- worst-10% accuracy；
- BN-GN class-wise gain；
- clean class difficulty vs corruption BN gain。

### 可能结论

> Under distribution shift, class-level disparity increases, and normalization-induced allocation differences become more pronounced.

如果结果不明显，放 appendix。

---

## P2.2 Synthetic K control

### 目的

区分 `d/K` 和 width 本身。

### 设置

在 CIFAR-100 上构造 class subsets：

- K=25；
- K=50；
- K=100。

固定 width c，改变 K：

| c | K | d/K |
|---:|---:|---:|
| 8 | 100 | 0.64 |
| 8 | 50 | 1.28 |
| 8 | 25 | 2.56 |
| 16 | 100 | 1.28 |
| 16 | 50 | 2.56 |

每个 K 用 3 个 random subsets，每个 subset 3 seeds。

### 风险

class subset 本身改变任务难度，会引入新 confound。建议只作为 appendix exploratory。

---

## P2.3 更多 normalization baselines

### 可选 baselines

- Ghost BatchNorm；
- SyncBatchNorm；
- Batch Renormalization；
- Weight Standardization + GN；
- RMSNorm；
- EvoNorm。

### 价值

帮助回答 BN 的 effect 来自 batch statistics、affine transformation、还是 normalization scope。

### 建议

若时间有限，只做 GhostBN / Batch Renorm，因为它们最贴近 BN 机制。

---

## P2.4 长尾设置

### 设置

- CIFAR-100-LT；
- Tiny-ImageNet-LT；
- imbalance factor 10 / 50 / 100。

### 价值

连接 fairness / long-tail literature。

### 风险

会打开新问题：class imbalance 与 capacity triage 的关系。建议不放主文，除非结果特别漂亮。

---

# 3. 机制解释应该如何写

新增机制分析后，论文可以做到“初步机制解释”，但不要声称完整因果机制。

---

## 3.1 推荐机制链条

最推荐的机制链条是：

> BN changes class-level learning order and margin geometry. At matched accuracy, this produces a more concentrated learning trajectory. As capacity relative to class count increases, the margin and class-wise accuracy deltas become more seed-stable. By convergence, BN's stronger optimization allows weaker classes to catch up, yielding a more uniform final profile.

中文解释：

> BN 改变了类别学习顺序和 margin 几何。在相同总准确率时，这会表现为更集中的类别收益路径；当 capacity-to-class ratio 增加时，这种 margin 和 accuracy 的类别级变化变得跨 seed 稳定；训练到最后，BN 的整体优化优势让弱类别追上来，因此最终模型更均匀。

---

## 3.2 机制 section 建议结构

建议新增：

## 4.X Mechanistic Probes: Learning Order and Margin Geometry

结构：

### Paragraph 1：为什么需要机制分析

> The Gini transition could be a metric artifact. We therefore examine whether it is reflected in class-learning order, margins, and representation geometry.

### Paragraph 2：learning order

报告：

- per-class time-to-threshold；
- BN 是否先学会某些类别；
- late training 是否降低 final Gini。

### Paragraph 3：margin geometry

报告：

- \(\Delta Margin_k\) 与 \(\Delta Acc_k\) 相关；
- high d/K 时 margin delta vector seed-stable；
- margin stability 与 accuracy stability 同步出现。

### Paragraph 4：feature geometry

报告：

- class prototype separation；
- effective rank；
- between/within ratio。

### Paragraph 5：保守结论

> These diagnostics do not establish causality, but they show that the output-level capacity-allocation transition is accompanied by parallel changes in decision-boundary and representation geometry.

---

## 3.3 不应写的机制 claim

不要写：

> We reveal the causal mechanism.

不要写：

> BN causes deterministic triage through margin redistribution.

不要写：

> d/K governs normalization-induced class allocation.

不要写：

> d/K≈0.8 is a universal threshold.

应写：

> These results are consistent with the hypothesis that BN alters class-level learning dynamics through changes in margin and feature geometry.

---

# 4. 论文结构重排建议

建议把 Results 重排成如下结构。

---

## 4.1 BN improves final accuracy and final class-level uniformity

讲 final outcome：

- BN final accuracy > GN；
- BN final raw Gini < GN；
- worst-10% class accuracy 更好；
- 多 normalization spectrum。

这里不要讲 matched Gini。

---

## 4.2 Accuracy-conditioned analysis reveals a distinct class-learning trajectory

讲 trajectory：

- matched-accuracy Gini；
- accuracy-conditioned Gini curve；
- BN 在同 aggregate accuracy 下 class gains 更集中；
- 解释这和 final raw Gini 不矛盾。

---

## 4.3 Reproducibility increases with capacity-to-class ratio

讲 reproducibility：

- dense width；
- 10 seeds；
- `d/K` as empirical proxy；
- crossover region；
- class-wise delta vector stability。

---

## 4.4 Controls: learning rate and batch size

讲 confound control：

- same-LR；
- best-LR；
- batch size sweep；
- 结论如何变化。

---

## 4.5 Mechanistic probes: learning order, margins, and feature geometry

讲机制：

- per-class learning order；
- margin；
- feature geometry；
- 不 claim causality。

---

## 4.6 Cross-dataset and cross-architecture validation

讲外部有效性：

- Tiny-ImageNet low/high d/K；
- CIFAR-10 high d/K；
- ResNet-small；
- ResNet-18 / WRN if available。

---

# 5. 需要新增或替换的主文图表

---

## Figure 1：Conceptual illustration

内容：

- 两个模型 aggregate accuracy 相同；
- per-class profile 不同；
- 引出 final outcome vs trajectory。

目的：

帮助非 normalization 专家理解论文。

---

## Figure 2：Accuracy-conditioned Gini curves

内容：

- x-axis：overall accuracy；
- y-axis：Gini；
- BN/GN curves；
- shared accuracy interval；
- area-under-\(\Delta G\)。

目的：

支撑 matched-accuracy 方法。

---

## Figure 3：Dense reproducibility crossover

内容：

- x-axis：`d/K`；
- y-axis：Gini delta SNR 或 accuracy-conditioned Gini AUC stability；
- 10-seed dense widths；
- shaded crossover region。

目的：

支撑 capacity dependence。

---

## Figure 4：Class-wise delta stability

内容：

- low d/K heatmap；
- high d/K heatmap；
- rows：seeds；
- columns：classes；
- values：BN-GN per-class accuracy delta。

目的：

直观展示 deterministic triage。

---

## Figure 5：Learning order

内容：

- class difficulty bins；
- BN/GN time-to-threshold；
- early vs late per-class gains。

目的：

解释 raw vs matched Gini。

---

## Figure 6：Margin mechanism

内容：

- x-axis：\(\Delta Margin_k\)；
- y-axis：\(\Delta Acc_k\)；
- low/high d/K 对比；
- 或 margin delta seed correlation vs d/K。

目的：

提供机制证据。

---

## Table 1：Main dense width results

列：

- c；
- d/K；
- BN acc；
- GN acc；
- final raw Gini gap；
- accuracy-conditioned Gini AUC；
- class-wise delta stability；
- seeds。

---

## Table 2：LR and batch-size controls

列：

- width；
- LR setting；
- batch size；
- acc gap；
- Gini trajectory delta；
- stability；
- conclusion。

---

## Table 3：Metric robustness

列：

- metric；
- final outcome direction；
- trajectory direction；
- reproducibility trend；
- notes。

---

# 6. 5 天执行排期

---

## Day 1：实验调度与核心 seed 扩展

### 实验

启动：

1. CIFAR-100 ScaleCNN dense width 10 seeds；
2. Tiny-ImageNet low/high d/K；
3. LR control；
4. batch size sweep。

### 工程任务

统一日志格式，每个 run 保存：

```json
{
  "dataset": "cifar100",
  "arch": "scalecnn",
  "width_c": 10,
  "d": 80,
  "K": 100,
  "d_over_K": 0.8,
  "norm": "BN",
  "lr": 0.1,
  "batch_size": 128,
  "seed": 42,
  "epoch_metrics": [
    {
      "epoch": 1,
      "test_acc": 0.0,
      "gini": 0.0,
      "per_class_acc": []
    }
  ],
  "final_acc": 0.0,
  "final_gini": 0.0
}
```

必须保存 per-class accuracy by epoch，否则 trajectory analysis 做不了。

---

## Day 2：LR / batch / Tiny-ImageNet 继续跑，开始分析

### 实验

继续：

- LR control；
- batch size sweep；
- Tiny-ImageNet；
- optional ResNet-small extra seeds。

### 分析

实现：

1. accuracy-conditioned Gini curve；
2. class-wise delta vector stability；
3. bootstrap crossover；
4. metric robustness。

### 写作

开始重写：

- Abstract；
- Introduction；
- Contributions；
- Section 4 skeleton。

---

## Day 3：机制分析

### 分析

从 checkpoints / logits / features 计算：

1. per-class learning curves；
2. time-to-threshold；
3. per-class margin；
4. margin delta seed stability；
5. feature effective rank；
6. class center distance；
7. intra/inter class variance。

### 图表

生成：

- Figure 2：Gini-vs-accuracy curve；
- Figure 4：class delta heatmap；
- Figure 5：learning order；
- Figure 6：margin mechanism。

---

## Day 4：标准架构和外部验证

### 实验

完成或补充：

- ResNet-small matched-accuracy analysis；
- ResNet-18 / WRN-16-2；
- CIFAR-100-C evaluation；
- optional synthetic K control。

### 写作

完成 Results 主体：

- final outcome；
- trajectory；
- reproducibility；
- controls；
- mechanism；
- validation。

---

## Day 5：收敛论文

### 任务

1. 所有 claim 对应图表；
2. 所有 overclaim 降调；
3. figure/table caption 检查；
4. limitations 重写；
5. checklist / LLM usage 声明修正；
6. appendix 整理；
7. final sanity check。

---

# 7. Agent 任务分解

---

## Experiment Agent

负责持续跑实验、监控失败、自动重启。

### Batch 1：Dense width

```bash
for c in 4 6 7 8 9 10 11 12 14 16 24 32 48; do
  for norm in bn gn; do
    for seed in 42 43 44 45 46 47 48 49 50 51; do
      python train.py \
        --dataset cifar100 \
        --arch scalecnn \
        --width $c \
        --norm $norm \
        --lr default \
        --batch_size 128 \
        --seed $seed \
        --save_epoch_metrics \
        --save_logits \
        --save_features final
    done
  done
done
```

### Batch 2：LR control

```bash
for c in 8 10 12 16 32; do
  for norm in bn gn; do
    for lr in 0.05 0.1; do
      for seed in 42 43 44 45 46; do
        python train.py \
          --dataset cifar100 \
          --arch scalecnn \
          --width $c \
          --norm $norm \
          --lr $lr \
          --batch_size 128 \
          --seed $seed \
          --save_epoch_metrics \
          --save_logits \
          --save_features final
      done
    done
  done
done
```

### Batch 3：Batch size sweep

```bash
for c in 8 10 16 32; do
  for norm in bn gn; do
    for bs in 32 64 128 256; do
      for seed in 42 43 44 45 46; do
        python train.py \
          --dataset cifar100 \
          --arch scalecnn \
          --width $c \
          --norm $norm \
          --lr scaled \
          --batch_size $bs \
          --seed $seed \
          --save_epoch_metrics \
          --save_logits \
          --save_features final
      done
    done
  done
done
```

### Batch 4：Tiny-ImageNet

```bash
for c in 8 16 24 32 48; do
  for norm in bn gn; do
    for seed in 42 43 44 45 46; do
      python train.py \
        --dataset tiny_imagenet \
        --arch scalecnn \
        --width $c \
        --norm $norm \
        --lr default \
        --batch_size 128 \
        --seed $seed \
        --save_epoch_metrics \
        --save_logits \
        --save_features final
    done
  done
done
```

---

## Analysis Agent

实现以下脚本：

```bash
python analyze_accuracy_conditioned_gini.py
python analyze_class_delta_stability.py
python analyze_learning_order.py
python analyze_margin_geometry.py
python analyze_feature_geometry.py
python analyze_metric_robustness.py
python bootstrap_crossover.py
```

每个脚本输出：

- `.csv`；
- `.json`；
- `.tex` table；
- `.pdf` figure；
- sanity-check log。

---

## Writing Agent

负责根据实验结果替换论文文字。

### 必须检查

每个 claim 必须属于以下三类之一：

1. directly supported by main experiment；
2. supported by appendix；
3. explicitly framed as hypothesis / limitation。

禁止出现 unsupported universal claim。

---

# 8. Abstract 建议版本

```text
Normalization layers are usually selected for optimization and convergence, but their role in shaping class-level outcomes remains less understood. We study this question in width-varying CNNs by asking whether normalization choice affects not only aggregate accuracy, but also how accuracy is allocated across classes. Across CIFAR-100, Tiny-ImageNet, and multiple small CNN architectures, Batch Normalization improves final accuracy and typically yields more uniform final per-class performance than Group Normalization. However, accuracy-conditioned analyses reveal a distinct learning trajectory: at matched aggregate accuracy, BN and GN allocate class-level gains differently. By sweeping model width, we find that the reproducibility of this class-level allocation effect increases as representation dimension grows relative to the number of classes, with a crossover region around d/K≈0.7–1.0 in our controlled ScaleCNN setting. Learning-rate and batch-size controls, alternative class-level disparity metrics, and representation diagnostics based on learning order, margins, and feature geometry support the view that normalization can act as a class-level capacity-allocation mechanism. Our results suggest that aggregate accuracy alone can hide systematic differences in how architecture choices distribute performance across classes.
```

注意：如果 Tiny-ImageNet 或机制实验结果不够强，abstract 中要删掉对应强 claim。

---

# 9. Introduction 建议结构

## Paragraph 1：aggregate accuracy hides class-level behavior

普通 benchmark 只报告平均准确率，但同样的 average accuracy 可能对应很不一样的 per-class profile。

## Paragraph 2：normalization usually treated as optimization choice

BN/GN/LN 通常被视为收敛、稳定性、batch-size 相关的选择，但很少被研究是否改变 class-level performance allocation。

## Paragraph 3：capacity-limited models make allocation visible

在小模型里，容量有限，不同类别竞争表示空间，因此 normalization 的 allocation effect 可能更明显。

## Paragraph 4：our study

我们用 width-varying CNN，改变 `d/K`，系统研究 final outcome、learning trajectory、reproducibility。

## Paragraph 5：findings

- BN final accuracy higher；
- BN final per-class profile more uniform；
- matched/accuracy-conditioned trajectory shows different allocation path；
- reproducibility increases with capacity-to-class ratio；
- margins/features provide initial mechanism evidence。

---

# 10. Limitations 建议写法

应诚实但不要自毁。

建议包括：

1. **Empirical scope**：主要是 small CNNs，不声称适用于 ViT/LLM/large-scale models。
2. **d/K as proxy**：`d/K` 是 lightweight proxy，不是真实容量的完整描述。
3. **Crossover not universal**：crossover location 可能受 architecture、dataset、optimizer、batch size 影响。
4. **Mechanism incomplete**：margin/feature/batch-size results 是 mechanistic probes，不是 causal proof。
5. **Dataset scale**：ImageNet-scale validation 尚未完成。
6. **Normalization design space**：没有穷尽所有 normalization variants。

推荐句子：

> We therefore interpret d/K as an empirical proxy that tracks reproducibility in our controlled width sweeps, not as a universal law governing normalization-induced allocation.

> Our margin and feature analyses provide mechanistic evidence but do not establish a causal pathway from batch statistics to class-level allocation.

---

# 11. LLM usage 声明

如果使用 agent auto research 参与写作、代码、实验调度、分析，原来的声明不能写成“LLM only for writing assistance”。

建议根据真实情况写：

```text
We used LLM-based tools to assist with drafting, code generation, experiment orchestration, and analysis scripting. All experimental results, plots, and paper claims were inspected and verified by the authors. The authors take full responsibility for the correctness and integrity of the submission.
```

如果会议要求更具体，可以加：

```text
No LLM-generated result was included without author verification.
```

---

# 12. 最小可交付版本

如果 5 天内实验无法全部完成，最低限度必须完成以下 6 件：

1. CIFAR-100 ScaleCNN dense width 10 seeds，尤其 c=6–16；
2. LR control：c=8/10/12/16/32，same LR + best LR；
3. accuracy-conditioned Gini curve AUC；
4. class-wise delta vector stability；
5. margin delta 与 accuracy delta 相关性；
6. 重写 abstract/introduction/results，明确 final outcome vs trajectory。

只要这 6 件做好，论文会明显比当前版本更强。

---

# 13. 最终推荐主线一句话

最终论文应该用这一句话统领：

> Normalization choice changes not only how well a capacity-limited classifier performs, but also how it distributes performance across classes; BN yields stronger and more uniform final models, while accuracy-conditioned analyses reveal a distinct class-learning trajectory whose reproducibility increases with capacity relative to class count.

中文版本：

> Normalization 不只是优化技巧；在容量受限的分类模型中，它还会影响模型如何把性能分配到不同类别上。BN 最终通常得到更强、更均匀的模型，但在相同总准确率下，它沿着不同的类别学习路径达到该性能；这种路径差异的可复现性会随着相对类别数的容量增加而增强。
