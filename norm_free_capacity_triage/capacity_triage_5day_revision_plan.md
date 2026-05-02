# Capacity Triage 论文 5 天提升方案（NeurIPS 2026 冲刺版 v2）

日期：2026-05-02
截止时间：2026-05-06
资源假设：2–4 张 RTX 5090（32 GB），可用 agent auto research 持续跑实验、写代码、画图和改论文。
目标：把论文从 "有趣但证据偏薄的 empirical observation"（预估 4–5/10）提升为 "叙事清晰、实验扎实、带有初步机制解释和因果检验的系统性经验研究"（目标 6.5–7/10）。

---

## 0. 总体判断

### 0.1 当前论文最有价值的点

不是 "BN 比 GN 准"，而是：

> Normalization choice may shape how limited model capacity is allocated across classes, not merely how high the aggregate accuracy becomes.

论文应避免被写成普通的 BN-vs-GN 性能比较，而应升级成：

> 一篇研究 normalization 如何影响 class-level performance allocation / class-learning trajectory 的系统性经验论文，带有初步因果检验和理论联系。

### 0.2 当前最危险的 claim

> We discovered a sharp d/K threshold around 0.8.

这个说法过强。更稳妥、更容易被 reviewer 接受的主线是：

> In width-varying CNNs, the reproducibility of normalization-induced class-level allocation effects increases as representation capacity grows relative to class count. We use d/K as a lightweight empirical proxy for this capacity-to-class-count ratio.

`d/K` 不应被包装成普适定律，而应被定义为一个经验 proxy。

### 0.3 当前论文的四个致命弱点

按 reviewer 攻击概率排序：

1. **"So what?"** — 观察到 BN 影响 per-class 分布，但论文从未分析 BN 到底帮了哪些类、为什么帮这些类。"Capacity triage" 这个概念缺乏具体内容。
2. **d/K 因果方向未验证** — 现有实验只改变 d（width），K 始终是 100。无法区分 d/K 是真正的控制变量，还是 d 本身就够了。
3. **Crossover 区域证据稀疏** — c=8 (SNR=0.50) 到 c=10 (SNR=4.98) 之间只有一跳，没有中间点。
4. **ScaleCNN 是自定义架构** — 虽然有 ResNet-small，但只有 3 个 width 点，无法验证 crossover。

### 0.4 论文 v16 已有数据完整清单

在设计新实验之前，必须先盘清家底，避免重复劳动：

| 已有数据 | 位置 | 详情 | 可直接复用 |
|---|---|---|---|
| ScaleCNN 8 widths × 3 seeds | Table 1 | c=4,8,10,12,16,24,32,48, BN/GN, ΔG+SNR, per-epoch per-class accuracy | 基线数据 |
| ScaleCNN 5 widths × 10 seeds | Table 2 | c=8,10,12,16,32, Cohen's d, per-seed 数据 | P0.1 已有 5 点 |
| ResNet-small 3 widths × 3/10 seeds | Table 7, 8 | c=8,16,32, BN/GN/None, c=32 有 10 seeds per-seed | 标准架构基线 |
| Tiny-ImageNet 2 widths × 3 seeds | Table 5 | c=32,48 (d/K=1.28, 1.92)，仅高 d/K | 高 d/K 参照 |
| CIFAR-10 c=32 × 3 seeds | Table 6 | d/K=25.6, extreme high d/K | 直接可用 |
| 5-norm spectrum | Table 3 | BN/LN/GN/IN/None, c=8 和 c=32 | 直接可用 |
| GN group-size dose-response | Table 9 (App C) | ScaleCNN c=32, G∈{1,2,4,8,16,32}, 3 seeds | categorical 结论已有 |
| LR confound analysis | Table 10 (App D) | 7 widths, BN@0.05 vs BN@0.1, confound ratio r | c≥16 clean, c≤12 confounded |
| Optimizer ablation | Table 11 (App E) | SGD/Adam/AdamW, ScaleCNN+ResNet c=32 | 直接可用 |
| Batch size sensitivity | Table 12 (App F) | B=32 vs B=128, ScaleCNN c=32, 3 seeds | 直接可用 |
| Convergence crossover | Table 4, Figure 6 | 10 seeds, epoch 50/100/150/200 | 核心证据，未被充分利用 |
| Per-class delta bars | Figure 4 | c=32 (82 helped/0 hurt) vs c=8 (27/3/70) | 直接可用 |
| Bootstrap crossover | App A | 10000 resamples, 95% CI=[0.16,0.72] | 数据可用，图需重生成 |
| Experimental transparency | App G | 失败 run、negative results、dropped experiments | 直接可用 |

**关键发现**：论文 Section 3.2 说 "recording per-class accuracy and overall accuracy at every epoch"，意味着所有现有 run 的 per-epoch per-class accuracy 数据已经存在。以下分析完全不需要新实验：
- Accuracy-conditioned Gini curve
- Class-wise delta vector stability
- Per-class learning curves + time-to-threshold
- Class difficulty vs BN gain 分析
- 多指标 robustness

---

## 1. 修改后的核心叙事

### 1.1 当前叙事的问题

当前论文把四件不同的事情混在一起：

1. BN 提升 overall accuracy；
2. BN final raw Gini 更低（最终模型 per-class accuracy 更均匀）；
3. BN matched-accuracy Gini 更高（相同 overall accuracy 阶段，class gains 更集中）；
4. matched-accuracy effect 在高 d/K 下更稳定。

这些结果可以共存，但必须明确区分：

- **Final outcome question**：训练结束后，BN/GN 谁的 per-class accuracy 更均匀？
- **Trajectory question**：在相同 overall accuracy 的训练阶段，BN/GN 是否通过不同的 class-learning path 达到该 accuracy？
- **Reproducibility question**：这种 class-level path difference 是否跨 seed 稳定？

如果不区分，读者会困惑：

> BN 到底是让 per-class accuracy 更均匀，还是让收益更集中？

### 1.2 修改后的统一解释

全文采用如下解释：

> BN 最终模型通常更强，也更均匀；但在达到相同 aggregate accuracy 的过程中，BN 和 GN 的 class-learning trajectory 不同。BN 可能先更集中地提升一部分类别，然后在训练后期让弱类别追上来。因此 final raw Gini 和 matched-accuracy Gini 不矛盾，它们分别描述最终结果和学习路径。

英文版：

> Final uniformity and matched-accuracy triage are not contradictory: they describe different stages of learning. BN may allocate gains unevenly along the learning trajectory, while its stronger convergence eventually allows weaker classes to catch up, producing a more uniform final profile.

**Table 4 的 convergence crossover 是这个叙事的关键证据**：GN 在 epoch 15–141 领先，BN 仅在最后 ~60 epochs 反超并拉开差距。这说明 BN 的 capacity redistribution 是 late-training phenomenon。

### 1.3 修改后的贡献表述

建议 contributions 改成五点：

1. **Phenomenon**：展示 normalization choice 不仅影响 aggregate accuracy，也影响 class-level performance profiles——具体分析哪些类获益、与 class difficulty 的关系。
2. **Methodology**：提出 accuracy-conditioned Gini curve analysis，用于区分 final outcome 和 learning trajectory。
3. **Causal test**：通过固定 width 变 class count 的 synthetic K control 实验，直接验证 d/K（而非 d 本身）是 allocation reproducibility 的控制变量。
4. **Capacity dependence**：通过 dense width sweeps（含标准架构）展示 class-level allocation effect 的 reproducibility 随 d/K 增强，crossover region 约在 d/K ≈ 0.7–1.0。
5. **Mechanistic probes**：用 class-learning order、convergence crossover、margin distribution、极端 batch-size control 提供初步机制证据，并联系 neural collapse 理论。

不要把贡献写成：

> We find a universal threshold at d/K ≈ 0.8.

应写成：

> We observe a reproducibility crossover around d/K ≈ 0.7–1.0 in our controlled small-CNN setting.

---

## 2. 5 天内实验与分析优先级

按重要性分三档。Tier 0 必须完成；Tier 1 强烈建议完成；Tier 2 有余力再做。

GPU 预算：4 张 5090 × 5 天 = 480 GPU-hours。

---

# Tier 0：必须完成（决定论文命运）

Tier 0 解决 reviewer 最可能给出的致命攻击：
- crossover 区域证据稀疏 → P0.1
- d/K 因果方向未验证 → P0.2
- "triage 了谁" 没有回答 → P0.3
- accuracy-conditioned 方法不够 robust → P0.4
- class-wise stability 未量化 → P0.5
- 叙事混乱 → 写作任务

---

## P0.1 CIFAR-100 ScaleCNN dense width + 10 seeds

### 目的

填充 crossover 区域的空白。当前 c=8 (SNR=0.50) 到 c=10 (SNR=4.98) 之间只有一跳，这是最脆弱的地方。

### 已有 vs 需要新跑

已有 10-seed 数据：c = 8, 10, 12, 16, 32
已有 3-seed 数据：c = 4, 24, 48

**需要新跑**：c = 6, 7, 9, 11, 14，每个 10 seeds × BN + GN = 100 runs

### 实验设置

Dataset：CIFAR-100
Model：ScaleCNN
Norm：BN vs GN
Seeds：10（42–51）
新 Widths：

| c | d=8c | d/K |
|---:|---:|---:|
| 6 | 48 | 0.48 |
| 7 | 56 | 0.56 |
| 9 | 72 | 0.72 |
| 11 | 88 | 0.88 |
| 14 | 112 | 1.12 |

GN group count：g=2 for c=7,9,11,14; c=6 用 g=2 或 g=3（取最近可整除值）。

### 估算时间

小 width 训练快（c=6 约 2–3 min/run），4 卡并行总计 ~1.5h。

### 指标

每个 run 保存：
- final overall accuracy
- final raw Gini
- per-class accuracy by epoch（必须保存，后续分析依赖）
- matched-accuracy Gini
- accuracy-conditioned Gini AUC
- class-wise BN-GN delta vector

### 预期产出

新增主文图（替换 Figure 3a）：

> Dense validation of the reproducibility crossover (13 widths × 10 seeds).

图中包含：
- x-axis：d/K
- y-axis：SNR 或 class-wise delta Spearman correlation
- bootstrap CI
- 标注 crossover region，不写 "sharp threshold"

### Contingency

如果 c=9 的 SNR 远低于 c=10（如 SNR < 2），crossover 是渐变而非跃变。此时写：

> The transition is gradual rather than sharp; reproducibility increases steadily across d/K ≈ 0.5–1.0 rather than exhibiting a step change.

这也是可以接受的结论——只是叙事从 "threshold" 变成 "gradient"。

---

## P0.2 Synthetic K control：直接验证 d/K 因果方向

### 目的

这是论文 d/K 理论的**最干净的因果检验**。

当前所有实验只改变 d（通过改 width），K 始终为 100。无法区分：
- 假说 A：d/K 是真正的控制变量
- 假说 B：d 本身就是控制变量，K 无关

通过固定 width 变 K，直接区分两个假说。

### 实验设置

在 CIFAR-100 上构造 class subsets：

| c | K (subset) | d=8c | d/K | 期望行为 |
|---:|---:|---:|---:|---|
| 8 | 100 | 64 | 0.64 | 已有：stochastic |
| 8 | 50 | 64 | 1.28 | 如果 d/K 理论正确：reproducible |
| 8 | 25 | 64 | 2.56 | 如果 d/K 理论正确：highly reproducible |
| 16 | 100 | 128 | 1.28 | 已有：reliable |
| 16 | 50 | 128 | 2.56 | sanity check |

每个 K 用 3 个 random class subsets × 5 seeds × BN + GN = 30 runs per K。
新 runs 总计：3 个新 K 条件 × 30 = 90 runs。

### 估算时间

K=50 和 K=25 训练更快（更少的类，更容易）。4 卡并行 ~2h。

### 控制变量

- 每个 K 使用 3 个 random class subsets 取平均，控制 subset selection 偏差
- 报告 baseline accuracy 变化（K=25 会比 K=100 容易）
- 报告 matched-accuracy ΔG 和 SNR

### 核心检验

如果 c=8/K=50 (d/K=1.28) 的 SNR 接近 c=16/K=100 (d/K=1.28) 的 SNR → **d/K 理论的强因果证据**。
如果 c=8/K=50 的 SNR 仍然很低 → d/K 不够，还有其他因素（如绝对 capacity）。

### 推荐结论表述

理想结果：

> Fixing width and halving K produces a reproducibility increase comparable to doubling width at fixed K, confirming that d/K—not d alone—governs allocation reproducibility.

非理想结果：

> Reducing K at fixed width improves reproducibility but not to the level predicted by d/K matching, suggesting that absolute capacity and d/K jointly determine the transition.

**两种结果都是有价值的 contribution。**

---

## P0.3 分析 BN 到底帮了哪些类（零 GPU 成本）

### 目的

回答论文最核心的 "so what" 问题。

当前论文 Figure 4 显示 c=32 时 BN 帮了 82 个类、0 个受损，但**从未分析这 82 个类有什么共性**。"Capacity triage" 这个概念需要具体内容。

### 分析内容

使用现有 10-seed c=32 数据（BN/GN/None per-class accuracy），进行以下分析：

#### 1. BN gain vs class difficulty

以 GN final accuracy 定义 class difficulty（GN accuracy 低 = hard class）。

计算：
- Spearman correlation between GN per-class accuracy and BN gain (BN_acc - GN_acc)
- 按 difficulty 分 5 个 quintile，每个 quintile 的 mean BN gain
- 散点图：x = GN per-class accuracy, y = BN per-class gain

如果 BN 更多帮助 hard classes → "BN redistributes capacity toward underperforming classes"（均等化）
如果 BN 更多帮助 easy classes → "BN amplifies existing advantages"（马太效应）
如果无明显关系 → "BN's allocation is class-identity-dependent but not difficulty-dependent"

#### 2. BN gain vs class confusability

从 confusion matrix 计算每个 class 的 confusability：
- max off-diagonal rate（最容易被错分为哪个类）
- total off-diagonal rate
- number of "confused neighbors"

看 BN gain 是否与 confusability 相关。

#### 3. CIFAR-100 superclass 结构

CIFAR-100 有 20 个 superclass（每个包含 5 个 fine-grained class）。分析：
- BN gain 在同 superclass 内的一致性
- superclass-level mean BN gain 的排名
- 特别关注 fine-grained distinction 密集的 superclass（如 "flowers" 包含 orchid/poppy/rose/sunflower/tulip）

#### 4. Low vs high d/K 下的类别对比

在 c=8（stochastic）和 c=32（deterministic）下：
- c=32 下 "consistently helped" 的 82 个类，在 c=8 下的 BN gain 分布
- 是否是同一组类在不同 width 下都被帮助，还是完全不同
- 如果是同一组 → class identity matters（intrinsic class property）
- 如果不同 → width-dependent allocation（capacity-dependent）

### 预期产出

新增主文图：

> Which classes does BN help? BN gain correlates with class difficulty / confusability.

新增一段 Results 分析（~0.5 页）。

### 推荐结论表述

> The capacity triage effect is not random: BN's per-class gains correlate with [class difficulty / confusability / superclass structure]. This suggests that normalization-induced allocation reflects systematic properties of the class structure, not merely seed-level noise.

---

## P0.4 Accuracy-conditioned Gini curve analysis

### 目的

把原来的 "单点 matched-accuracy Gini" 升级为更稳健的 "曲线级分析"。

当前方法用 GN final accuracy 去找 BN 的中间 checkpoint。这个设计有价值，但容易被质疑：

> 你比较的是 early BN 和 converged GN，不公平。

### 前置验证

**在构建 G(a) 曲线之前，必须先验证单调性**：

- 由于 cosine annealing，accuracy 不是 epoch 的单调函数
- 同一 accuracy 可能对应多个 epoch，因此多个 Gini 值
- 如果 Gini-vs-accuracy 曲线不单调，需要用 Pareto front 或 envelope 处理

建议：先在 c=32 的现有数据上画 Gini-vs-accuracy 散点图（每个 epoch 一个点），确认形状后再决定插值方法。

### 改进方法

对每个 seed、每个 norm，构建：

G(a) = Gini as a function of overall accuracy

在 BN 和 GN 的 shared accuracy interval 内，采样 20–50 个 accuracy bins：

A_shared = [max(A_min^BN, A_min^GN), min(A_max^BN, A_max^GN)]

计算：

ΔG(a) = G_BN(a) - G_GN(a)

报告：
1. mean ΔG(a)
2. area-under-ΔG（accuracy-conditioned Gini AUC）
3. fraction of bins where ΔG(a) > 0
4. seed-wise consistency
5. bootstrap CI

### 新指标命名

> Accuracy-conditioned Gini AUC

### 主文图

新增一张关键图：
- x-axis：overall accuracy
- y-axis：Gini
- BN/GN 两条曲线
- shared accuracy range shaded
- 显示 ΔG(a) 的方向

### 推荐结论表述

> Single-point matched-accuracy estimates can be sensitive to checkpoint choice. We therefore compare BN and GN over the full shared accuracy interval. The trajectory-level Gini difference remains directionally consistent at high d/K, indicating that the effect is not an artifact of one matched epoch.

---

## P0.5 Class-wise delta vector stability（零 GPU 成本）

### 目的

直接量化证明什么叫 "deterministic triage"。

Gini 是一个 scalar。Reviewer 会问：

> 你说 deterministic，到底哪些东西 deterministic？

必须直接看每个 seed 下 BN-GN 的 per-class delta vector 的跨 seed 稳定性。

### 分析方法

使用现有 10-seed 数据（c=8, 10, 12, 16, 32），计算：

v_s = [Acc^BN_{1,s} - Acc^GN_{1,s}, ..., Acc^BN_{K,s} - Acc^GN_{K,s}]

不同 seeds 之间的：
- Pearson correlation
- Spearman correlation
- Cosine similarity
- Top-10 helped classes overlap（Jaccard index）
- Bottom-10 hurt classes overlap

### 预期图

新增主文图：

> Class-wise allocation signatures become seed-stable above the crossover.

推荐两种可视化：

1. **x-axis：d/K，y-axis：mean pairwise Spearman correlation**。预期在 crossover 处从 ~0.3 跳到 ~0.8+。

2. **Heatmap**：rows = seeds，columns = classes（按 mean delta 排序），values = BN-GN per-class delta。左面板 low d/K（噪声），右面板 high d/K（一致的 pattern）。

### 推荐结论表述

> Above the crossover, not only the scalar Gini delta but also the identity of helped and hurt classes becomes more reproducible across seeds.

---

# Tier 1：强烈建议完成（显著提分）

Tier 1 把论文从 "ScaleCNN-only observation" 提升到 "general phenomenon with mechanism"。

---

## P1.1 标准架构 width sweep

### 目的

解决 "ScaleCNN is custom architecture" 的质疑。当前 ResNet-small 只有 3 个 width 点，无法验证 crossover。

### 实验设置

**ResNet-18 width sweep on CIFAR-100**：

| Width multiplier | 最终 feature dim d | d/K |
|---:|---:|---:|
| 0.25 | 128 | 1.28 |
| 0.5 | 256 | 2.56 |
| 1.0 | 512 | 5.12 |

Norm：BN vs GN
Seeds：5
Runs：3 × 2 × 5 = 30 runs

**WideResNet-16 width sweep on CIFAR-100**（如果 GPU 余量充足）：

| Widen factor | 最终 feature dim d | d/K（估算） |
|---:|---:|---:|
| 1 | 64 | 0.64 |
| 2 | 128 | 1.28 |
| 4 | 256 | 2.56 |

Norm：BN vs GN
Seeds：5
Runs：3 × 2 × 5 = 30 runs

### 估算时间

ResNet-18 on CIFAR-100：~10–15 min/run。4 卡并行 ~2h。
WRN-16：~15–20 min/run。4 卡并行 ~3h。

### 推荐结论

> The direction of the allocation effect is not specific to ScaleCNN; [ResNet-18 / WideResNet] shows directionally consistent BN advantage in accuracy and per-class uniformity, although the exact crossover location and magnitude are architecture-dependent.

---

## P1.2 Per-class learning curves 与 convergence crossover 深入分析（零 GPU 成本）

### 目的

解释 final raw Gini 和 matched-accuracy Gini 为什么不矛盾，核心利用 Table 4 的 convergence crossover 发现。

Table 4 已经展示了 aggregate 级别的 convergence crossover。现在需要做 **per-class 版本**。

### 分析方法

使用现有 c=32 10-seed 数据：

1. **Per-class BN-GN crossover epoch**：对每个 class，找 BN accuracy 超过 GN accuracy 的最早 epoch。

2. **Crossover epoch vs class difficulty**：
   - 如果 easy classes 先反超、hard classes 后反超 → BN 的 capacity redistribution 有序进行
   - 如果 hard classes 先反超 → BN 优先帮助弱者（均等化机制）
   - 如果随机 → 无序

3. **Time-to-threshold per class**：
   - time-to-30%, time-to-50%, time-to-70%
   - BN vs GN 的 rank correlation

4. **Late-training catch-up**：
   - 最后 50 epochs 的 per-class accuracy gain
   - BN 是否在后期帮助了前期落后的类

### 主文结论

> The matched-accuracy concentration arises because BN and GN reach the same aggregate accuracy through different class-learning orders. BN's trajectory initially concentrates gains on a reproducible subset of classes, while late training redistributes capacity to weaker classes, yielding a more uniform final model.

> This explains the apparent contradiction between lower final Gini (more uniform outcome) and higher matched-accuracy Gini (more concentrated trajectory): they describe different stages of the same learning process.

---

## P1.3 Neural Collapse 理论联系（零 GPU 成本）

### 目的

给 d/K crossover 一个理论解释框架，把论文从纯 empirical 提升到 "empirical + theoretical grounding"。

### 理论联系

Neural collapse（Papyan et al. 2020, Jiang et al. 2024）说：当 d ≫ K 时，last-layer features 收敛到 simplex equiangular tight frame (ETF)，每个 class 获得等量的 feature space。

这恰好解释了为什么高 d/K 下 per-class allocation 更 deterministic：
- d < K（underdetermined）：解空间大，seed 对 allocation 影响大 → stochastic regime
- d > K（overdetermined）：解几何受 ETF 约束，allocation 更确定 → deterministic regime
- d/K ≈ 1 是转变点，恰好和我们观察到的 crossover region 吻合

### 写作

在 Discussion 中新增一段（约 150 词）：

> Our d/K crossover may connect to neural collapse theory. When d ≫ K, the solution geometry becomes increasingly constrained toward a simplex ETF, leaving less room for seed-dependent allocation variation. The crossover around d/K ≈ 0.7–1.0 may mark the transition from the underdetermined regime (d < K, many valid class allocations) to the overdetermined regime (d > K, geometry constrains allocation). This geometric perspective also suggests why BN—which standardizes features along the batch dimension—might influence allocation differently from GN: by coupling class representations through shared batch statistics, BN may accelerate convergence toward the constrained solution manifold. We note this connection is speculative and leave rigorous verification to future work.

### 可选实验验证

如果有余力，在 high/low d/K 的 final models 上计算 ETF metric（equiangularity + equal norm），看是否和 allocation stability 相关。只需一次 forward pass。

---

## P1.4 极端 batch size 实验

### 目的

提供最有信息量的 batch-statistics 机制证据。

当前 App F 有 B=32 vs B=128，方向一致。但从机制角度，**极端 batch size 才有鉴别力**。

### 实验设置

| Batch size | BN statistics 特性 | 预期 |
|---:|---|---|
| 8 | 接近纯噪声 | 如果 effect 消失 → batch statistics 是关键机制 |
| 16 | 较大噪声 | 过渡 |
| 128 | 标准（已有） | baseline |

Dataset：CIFAR-100
Model：ScaleCNN c=32
Norm：BN vs GN
LR：按 linear scaling rule 调整
Seeds：3–5
Runs：B=8,16 × BN/GN × 5 seeds = 20 runs

### 估算时间

B=8 训练更慢（更多 iterations），但 c=32 模型小。~3h 4 卡并行。

### 可能结论

如果 BN effect 在 B=8 下显著减弱或消失：

> Batch-level statistics are necessary for BN's allocation effect: reducing batch size to 8—where batch means and variances are dominated by sampling noise—eliminates the reproducible class-level allocation pattern.

如果 effect 仍在（可能性也不小）：

> The allocation effect persists even at batch size 8, suggesting it arises from BN's architectural structure (affine parameters, computation graph) rather than accurate batch statistics alone.

**两种结果都是有价值的机制证据。**

---

## P1.5 Margin 分析（需要检查是否有 logits 数据）

### 前置检查

先确认训练时是否保存了 logits 或 predictions。如果没有，只对 final model 做 forward pass（~10 min）。

### 分析

对 final model，每个样本 x 计算：

m(x) = z_y(x) - max_{j≠y} z_j(x)

对每个 class：

M_k = E_{x∈k}[m(x)]

BN-GN margin delta：

ΔM_k = M^BN_k - M^GN_k

计算：
1. ΔM_k 与 ΔAcc_k 的 Spearman correlation
2. ΔM vector 的 seed-wise Spearman correlation
3. margin stability 是否和 accuracy stability 同步出现（across d/K）

### 推荐结论

> The output-level allocation transition is mirrored by a margin-level transition: above the crossover, BN-GN class-wise margin deltas become more seed-stable and correlate strongly with class-wise accuracy deltas (Spearman ρ = [X]).

注意措辞——不说 "proves causality"，说 "mirrored" / "consistent with"。

---

## P1.6 将 LR confound 和 batch-size 现有结论移入主文

### 目的

App D (LR confound) 和 App F (batch size) 已有足够数据，但关键结论藏在 appendix 里，reviewer 可能忽略。

### 做法

在 Results 中新增一个简短的 Controls subsection（~0.5 页），直接引用 App D 和 App F 的结论：

> **Learning rate control.** BN and non-BN variants use different optimized learning rates (0.1 vs. 0.05). Our LR confound analysis (Appendix D) shows that the within-BN Gini shift from switching LR is negligible at c ≥ 16 (confound ratio r ≤ 7.5%), while at c ≤ 12, LR and normalization effects are entangled. We therefore anchor all quantitative Gini claims to widths with d/K ≥ 1.28.
>
> **Batch size.** BN outperforms GN at both B=32 and B=128 (Appendix F), ruling out a batch-size artifact at the default setting.
>
> **Optimizer.** The direction of BN's advantage is consistent across SGD, Adam, and AdamW (Appendix E).

这比重新跑 100 runs LR sweep 的 ROI 高 100 倍。

---

# Tier 2：有余力再做

---

## P2.1 Tiny-ImageNet 补低 d/K

### 前置验证

**在投入之前，先跑 c=8 Tiny-ImageNet 一个试探 run**。如果 accuracy < 15%，放弃。

### 设置

Dataset：Tiny-ImageNet (32×32)
Model：ScaleCNN
Widths：c = 8, 16, 24
Norm：BN vs GN
Seeds：3–5

### 风险

K=200，c=8 时 d=64，模型极弱。per-class accuracy 可能是纯噪声。

### 可接受的降级结论

> Cross-dataset validation is limited to the high-d/K regime; low-d/K Tiny-ImageNet models are too weak for meaningful per-class analysis.

这是一个合理的 limitation。

---

## P2.2 Worst-class / worst-10% accuracy

### 目的

连接 fairness literature。只计算两个简单指标：

- worst-class accuracy
- worst-10% mean accuracy

对现有所有数据。如果 BN 在 worst-class 上也更好 → 连接 fairness。

---

## P2.3 Feature geometry ETF metric

### 做法

在 final model 上做 forward pass，取 last-layer features h(x)，计算：

- class mean vectors μ_k
- equiangularity metric（Jiang et al. 2024）
- equal norm metric

看是否和 d/K 以及 allocation stability 相关。

---

# 3. 机制解释应该如何写

### 3.1 推荐机制链条

> BN changes class-level learning order and margin geometry. At matched accuracy, this produces a more concentrated learning trajectory. As capacity relative to class count increases (approaching the neural-collapse-constrained regime), the margin and class-wise accuracy deltas become more seed-stable. By convergence, BN's stronger optimization allows weaker classes to catch up, yielding a more uniform final profile.

### 3.2 机制 section 建议结构

新增：

> ## 4.X Mechanistic Probes

#### Paragraph 1：为什么需要机制分析

> The Gini transition could be a metric artifact. We therefore examine whether it is reflected in class-learning order, margins, and batch-size sensitivity.

#### Paragraph 2：convergence crossover 和 learning order

- Table 4 的 aggregate convergence crossover
- per-class 版本：哪些 class 先 crossover
- class difficulty vs crossover epoch 的关系

#### Paragraph 3：batch-size sensitivity

- 极端 B=8 的结果
- batch statistics 的作用

#### Paragraph 4：margin geometry（如果有数据）

- ΔMargin_k 与 ΔAcc_k 相关
- margin stability 与 accuracy stability 同步

#### Paragraph 5：neural collapse 连接

- d/K 与解几何的关系
- speculative but grounded

#### Paragraph 6：保守结论

> These diagnostics do not establish causality, but they show that the output-level capacity-allocation transition is accompanied by parallel changes in learning dynamics, decision-boundary geometry, and batch-statistics sensitivity.

### 3.3 不应写的机制 claim

禁止：
- "We reveal the causal mechanism."
- "BN causes deterministic triage through margin redistribution."
- "d/K governs normalization-induced class allocation."
- "d/K ≈ 0.8 is a universal threshold."

应写：
- "These results are consistent with the hypothesis that..."
- "This suggests that..."
- "We observe an association between..."

---

# 4. 论文结构重排建议

Results 重排为如下结构：

## 4.1 BN improves final accuracy and per-class uniformity

讲 final outcome：
- BN final accuracy > GN（width-persistent）
- BN final raw Gini < GN
- worst-class / worst-10% accuracy（如果做了 P2.2）
- normalization spectrum

**不讲 matched Gini。**

## 4.2 Accuracy-conditioned analysis reveals a distinct class-learning trajectory

讲 trajectory：
- accuracy-conditioned Gini curve
- BN 在同 aggregate accuracy 下 class gains 更集中
- convergence crossover（Table 4）解释 final vs trajectory 不矛盾

## 4.3 Which classes does BN help?

讲 class-level analysis（新增）：
- BN gain vs class difficulty
- BN gain vs confusability
- superclass 结构
- 回答 "triage 了谁"

## 4.4 Reproducibility increases with d/K

讲 reproducibility：
- dense width + 10 seeds
- d/K as empirical proxy
- crossover region
- class-wise delta vector stability

## 4.5 d/K causal validation: synthetic K control

讲因果检验（新增）：
- 固定 width 变 K
- d/K matching across different (d, K) pairs

## 4.6 Controls: learning rate, batch size, optimizer

讲 confound control：
- LR confound ratio（从 App D 提取）
- batch size（从 App F 提取）
- optimizer（从 App E 提取）

## 4.7 Mechanistic probes

讲机制：
- learning order + convergence crossover
- extreme batch size
- margin（如果有）
- neural collapse 联系

## 4.8 Cross-dataset and cross-architecture validation

讲外部有效性：
- Tiny-ImageNet
- CIFAR-10
- ResNet-small（现有）
- ResNet-18 / WRN（新增）

---

# 5. 需要新增或替换的主文图表

### 主文精选 4–5 个核心 Figure（避免图表堆砌）

**Figure 1：Conceptual illustration（保留或重画）**

保持当前 Gini-vs-epoch 图，但加标注 final outcome vs trajectory 区间。

**Figure 2：Accuracy-conditioned Gini curves（新增，替代当前 Figure 2 的 accuracy trajectory）**

- x-axis：overall accuracy
- y-axis：Gini
- BN/GN curves
- shared accuracy interval shaded
- 展示 ΔG(a)

**Figure 3：Dense reproducibility crossover（替换当前 Figure 3a）**

- x-axis：d/K
- y-axis：class-wise delta Spearman correlation 或 SNR
- 13 widths × 10 seeds
- shaded crossover region
- 可选：叠加 synthetic K control 的 matched d/K 点

**Figure 4：Class-wise delta heatmap + class analysis（升级当前 Figure 4）**

- 左面板：low d/K heatmap（rows=seeds, cols=classes）
- 右面板：high d/K heatmap
- 下方面板：BN gain vs class difficulty 散点图

**Figure 5：标准架构验证（如果做了 P1.1）**

- ResNet-18 / WRN 的 BN-GN gap 和 Gini delta
- 与 ScaleCNN 对比

### 主文 Table

**Table 1：Main dense width results（扩展当前 Table 1）**

列：c, d/K, BN acc, GN acc, ΔG, acc-conditioned Gini AUC, class-delta Spearman, seeds

**Table 2：Synthetic K control results（新增）**

列：c, K, d/K, SNR, class-delta Spearman, conclusion

**Table 3：Controls summary（新增，精简）**

列：control type, finding, appendix reference

其余 Table 移入 Appendix。

---

# 6. 5 天执行排期

---

## Day 1：实验启动 + 核心分析 + 叙事重构

### 实验（后台跑，GPU 全负荷）

1. **ScaleCNN dense width** c={6,7,9,11,14} × BN/GN × 10 seeds = 100 runs（~1.5h）
2. **Synthetic K control** c=8 K={50,25} + c=16 K=50 × 3 subsets × 5 seeds × 2 norms = 90 runs（~2h）
3. **ResNet-18 width sweep** × BN/GN × 5 seeds = 30 runs（~2h）
4. **极端 batch size** B=8,16 × BN/GN × c=32 × 5 seeds = 20 runs（~3h）
5. **Tiny-ImageNet 试探** c=8 × BN/GN × 1 seed = 2 runs（~15min，看 accuracy 是否可用）

Day 1 总计：~242 runs，4 卡并行 ~6–8h。GPU 利用率 ~30%。

### 分析 Agent（同时工作，使用现有数据）

1. 构建 accuracy-conditioned Gini curve（先验证单调性）
2. 计算 class-wise delta vector Spearman correlation across d/K
3. 分析 BN 帮了哪些类 + class difficulty 关系
4. Per-class learning curves + time-to-threshold
5. Per-class convergence crossover epoch vs class difficulty
6. 修复 Figure 8 bootstrap 图

### 写作 Agent（同时工作）

1. 重写叙事框架（Section 1.2 统一解释）
2. 重写 contributions（五点）
3. 全局替换 "threshold" → "crossover region"
4. 开始 Introduction 重写

---

## Day 2：处理 Day 1 实验结果 + 核心分析完成

### 实验结果处理

- Dense width 结果 → 生成新 crossover 图
- Synthetic K 结果 → 生成 d/K 因果验证表
- ResNet-18 结果 → 标准架构验证
- 极端 batch size 结果 → 机制证据
- Tiny-ImageNet 试探 → 决定是否继续投入

### 补充实验（如果 GPU 空闲）

- WRN-16 width sweep（30 runs, ~3h）
- Tiny-ImageNet 低 d/K full runs（如果试探 ok）
- Margin forward pass（~10 min per model）

### 分析 Agent

- 生成所有核心 Figure（Figure 2–5）
- 多指标 robustness（worst-class, worst-10%）
- Margin 分析（如果有 logits）

### 写作 Agent

- 重写 Section 4.2（accuracy-conditioned）
- 新写 Section 4.3（which classes does BN help）
- 重写 Section 4.4（dense crossover）
- 新写 Section 4.5（synthetic K control）

---

## Day 3：机制分析 + 论文结构完成

### 分析

- 完善所有图表
- Convergence crossover per-class 版本
- Neural collapse ETF metric（如果有余力）
- Synthetic K control 深入分析

### 写作

- 完成所有 Results sections
- 新写 Controls subsection（P1.6，从 Appendix 提取）
- 新写 Mechanistic Probes section
- 新写 Neural Collapse Discussion 段落

---

## Day 4：标准架构整合 + 论文完善

### 写作

- 整合 ResNet-18 / WRN 结果到 Section 4.8
- 整合 Tiny-ImageNet 结果
- 重写 Abstract（根据实际结果，准备两个版本：有/无机制证据）
- 重写 Introduction
- 重写 Limitations（见 Section 8）
- 重写 Conclusion
- 更新 LLM usage 声明

### 审查

- 每个 claim 对照 evidence 逐条检查
- 检查是否有残留的 overclaim

---

## Day 5：收敛论文

### 任务

1. 所有 claim 对应图表——逐条验证
2. 所有 overclaim 降调——特别检查 abstract 和 conclusion
3. Figure/Table caption 检查——确保 caption 自包含
4. Figure 8 bootstrap 图重新渲染
5. Limitations 完整性检查
6. NeurIPS checklist 更新
7. LLM usage 声明修正
8. Appendix 整理
9. 考虑是否需要修改标题
10. Final proofreading

---

# 7. Agent 任务分解

---

## Experiment Agent

### Batch 1：Dense width（Day 1 第一批）

```bash
for c in 6 7 9 11 14; do
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

### Batch 2：Synthetic K control（Day 1 第二批）

```bash
for c in 8 16; do
  for K in 25 50; do
    for subset_id in 0 1 2; do
      for norm in bn gn; do
        for seed in 42 43 44 45 46; do
          python train.py \
            --dataset cifar100 \
            --arch scalecnn \
            --width $c \
            --norm $norm \
            --num_classes $K \
            --class_subset_id $subset_id \
            --lr default \
            --batch_size 128 \
            --seed $seed \
            --save_epoch_metrics
        done
      done
    done
  done
done
```

### Batch 3：ResNet-18 width sweep（Day 1 第三批）

```bash
for width_mult in 0.25 0.5 1.0; do
  for norm in bn gn; do
    for seed in 42 43 44 45 46; do
      python train.py \
        --dataset cifar100 \
        --arch resnet18 \
        --width_multiplier $width_mult \
        --norm $norm \
        --lr default \
        --batch_size 128 \
        --seed $seed \
        --save_epoch_metrics
    done
  done
done
```

### Batch 4：Extreme batch size（Day 1 第四批）

```bash
for bs in 8 16; do
  for norm in bn gn; do
    for seed in 42 43 44 45 46; do
      python train.py \
        --dataset cifar100 \
        --arch scalecnn \
        --width 32 \
        --norm $norm \
        --lr $(echo "0.1 * $bs / 128" | bc -l) \
        --batch_size $bs \
        --seed $seed \
        --save_epoch_metrics
    done
  done
done
```

### Batch 5：Tiny-ImageNet 试探（Day 1 快速检查）

```bash
python train.py \
  --dataset tiny_imagenet \
  --arch scalecnn \
  --width 8 \
  --norm bn \
  --lr 0.1 \
  --batch_size 128 \
  --seed 42 \
  --save_epoch_metrics
```

如果 final accuracy > 15%，继续 full Tiny-ImageNet runs。

---

## Analysis Agent

Day 1–2 实现以下脚本：

```bash
python analyze_accuracy_conditioned_gini.py      # P0.4
python analyze_class_delta_stability.py           # P0.5
python analyze_which_classes_helped.py            # P0.3
python analyze_learning_order.py                  # P1.2
python analyze_convergence_crossover_perclass.py  # P1.2
python analyze_margin_geometry.py                 # P1.5
python analyze_synthetic_k.py                     # P0.2
python analyze_standard_arch.py                   # P1.1
python compute_etf_metric.py                      # P2.3
```

每个脚本输出：
- `.csv`
- `.json`
- `.pdf` figure
- sanity-check log

---

## Writing Agent

负责根据分析和实验结果替换论文文字。

### 必须检查

每个 claim 必须属于以下三类之一：
1. directly supported by main experiment
2. supported by appendix
3. explicitly framed as hypothesis / limitation

禁止出现 unsupported universal claim。

---

# 8. 关键段落建议写法

## 8.1 Abstract 建议版本

准备两个版本，根据实际结果选择：

### Version A（完整版，如果所有实验成功）

```text
Normalization layers are usually selected for optimization and convergence,
but their role in shaping class-level outcomes remains less understood. We
study this question in width-varying CNNs by asking whether normalization
choice affects not only aggregate accuracy, but also how accuracy is
allocated across classes. We find a dual-layer structure: Batch
Normalization improves final accuracy and per-class uniformity at all
tested widths, while accuracy-conditioned analyses reveal a distinct
class-learning trajectory whose per-class allocation pattern becomes
seed-reproducible above a d/K crossover region around 0.7-1.0. A synthetic
class-count control confirms that d/K—not width alone—governs this
transition. Class-level analysis shows that BN's gains correlate
systematically with class difficulty, and mechanistic probes based on
learning order, margins, and extreme batch-size controls provide initial
evidence that batch-level statistics contribute to allocation dynamics.
Results replicate on ResNet-18 and across datasets (CIFAR-100, Tiny-
ImageNet, CIFAR-10), suggesting that normalization can act as a class-level
capacity-allocation mechanism in capacity-limited CNNs.
```

### Version B（保守版，如果部分实验不理想）

```text
Normalization layers are usually selected for optimization and convergence,
but their role in shaping class-level outcomes remains less understood. We
study this question in width-varying CNNs. Across CIFAR-100, Tiny-ImageNet,
and multiple architectures, Batch Normalization improves final accuracy and
per-class uniformity. Accuracy-conditioned Gini curve analysis reveals that
BN and GN reach similar aggregate accuracy through different class-learning
trajectories. The reproducibility of this class-level allocation effect
increases with the ratio of representation dimension to class count (d/K),
with a crossover region around 0.7-1.0 in our controlled settings. We
analyze which classes benefit from BN and provide initial mechanistic
evidence through learning-order analysis and batch-size controls. Our
results suggest that aggregate accuracy alone can hide systematic
differences in how normalization distributes performance across classes.
```

## 8.2 Introduction 建议结构

### Paragraph 1：aggregate accuracy hides class-level behavior

普通 benchmark 只报告平均准确率，但同样的 average accuracy 可能对应很不一样的 per-class profile。在 fairness-sensitive 应用中（医疗诊断、物种识别），某些类的 accuracy 远低于平均值是不可接受的。

### Paragraph 2：normalization usually treated as optimization choice

BN/GN/LN 通常被视为收敛、稳定性、batch-size 相关的选择，但很少被研究是否改变 class-level performance allocation。

### Paragraph 3：capacity-limited models make allocation visible

在小模型里，容量有限，不同类别竞争表示空间，因此 normalization 的 allocation effect 可能更明显。Neural collapse 理论预测当 d ≫ K 时解几何受约束，这暗示 d/K ratio 可能是 allocation behavior 的关键参数。

### Paragraph 4：our study

我们用 width-varying CNN，系统研究 final outcome、learning trajectory、reproducibility。关键创新：(1) accuracy-conditioned Gini curve 方法论，(2) synthetic K control 验证 d/K 因果方向，(3) 分析 BN 帮了哪些类。

### Paragraph 5：findings

- BN final accuracy higher and more uniform
- accuracy-conditioned trajectory shows different allocation path
- BN systematically helps [hard / confusable] classes
- reproducibility increases with d/K, confirmed by synthetic K control
- convergence crossover + batch-size sensitivity provide mechanism evidence

## 8.3 Limitations 建议写法

应诚实但不自毁。

1. **Empirical scope**：主要是 small CNNs，不声称适用于 ViT/LLM/large-scale models。
2. **d/K as proxy**：`d/K` 是 lightweight proxy，不是真实容量的完整描述。synthetic K control 支持其因果角色，但绝对 capacity 也可能有贡献。
3. **Crossover not universal**：crossover location 可能受 architecture、dataset、optimizer 影响。
4. **Mechanism incomplete**：margin/learning-order/batch-size results 是 mechanistic probes，不是 causal proof。
5. **Dataset scale**：ImageNet-scale validation 尚未完成。

推荐句子：

> We interpret d/K as an empirical proxy supported by synthetic-K causal controls in our settings, not as a universal law governing normalization-induced allocation.

> Our mechanistic probes provide convergent evidence but do not establish a complete causal pathway from batch statistics to class-level allocation.

---

# 9. LLM usage 声明

```text
We used LLM-based tools (Claude Code agents) to assist with experiment
orchestration, analysis scripting, code generation, and manuscript drafting.
All experimental designs, results, plots, and paper claims were inspected
and verified by the authors. The authors take full responsibility for the
correctness and integrity of the submission. No LLM-generated result was
included without author verification.
```

---

# 10. 最小可交付版本

如果 5 天内实验无法全部完成，最低限度必须完成以下 8 件：

1. **ScaleCNN dense width** c=6–14 × 10 seeds（填充 crossover）
2. **Synthetic K control** c=8 K=50/25（验证 d/K 因果方向）
3. **分析 BN 帮了哪些类**（回答 "so what"）
4. **Accuracy-conditioned Gini curve AUC**（方法论升级）
5. **Class-wise delta vector stability**（量化 "deterministic triage"）
6. **Per-class convergence crossover analysis**（解释 final vs trajectory）
7. **LR/batch-size/optimizer confound 移入主文**（利用现有 Appendix D/E/F）
8. **重写 abstract/introduction/results 叙事**（final outcome vs trajectory 区分）

只要这 8 件做好，论文会从 4–5/10 提升到 6–7/10。

---

# 11. 最终推荐主线一句话

> Normalization choice changes not only how well a capacity-limited classifier performs, but also how it distributes performance across classes; BN yields stronger and more uniform final models, while accuracy-conditioned analyses reveal a distinct class-learning trajectory whose reproducibility—and the identity of helped classes—increases with capacity relative to class count, as confirmed by synthetic class-count controls.

---

# 12. 预估分数与风险

| 场景 | 预估分数 | NeurIPS 通过概率 |
|---|---|---|
| 当前 v16 | 4–5/10 | ~5–10% |
| 只做 dense width + 叙事修复 | 5–6/10 | ~15–25% |
| 完成 Tier 0 全部 | 6–6.5/10 | ~30–35% |
| 完成 Tier 0 + Tier 1 | 6.5–7/10 | ~40–50% |

**最大 swing factor**：synthetic K control 和 class-level analysis 的结果质量。如果这两个分析产生 compelling 结果，论文有可能到 7+/10。

**最大风险**：synthetic K control 不支持 d/K 理论。如果 c=8/K=50 的 SNR 远低于 c=16/K=100，需要降调 d/K claim。Contingency：报告 "d/K is necessary but not sufficient; absolute capacity also matters"。
