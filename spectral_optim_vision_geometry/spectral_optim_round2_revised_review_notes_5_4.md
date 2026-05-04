# Spectral Optimization Paper: Review Notes & Improvement Plan

> 更新日期：2026-05-04
> 基于论文全文（含 Appendix A-I）的完整审读
> 资源假设：2-4x RTX 5090，3-4 天连续执行，Agent Auto Research 模式

## Part 1. 论文主体内容

### 1.1 核心结论

这篇文章的核心主张是：

> 在作者设定的 controlled、from-scratch、short-to-medium schedule 视觉训练范式下，显式利用权重矩阵谱结构的优化器家族（Muon / SOAP / Shampoo）比 AdamW 和 SGD 具有更强的跨架构鲁棒性，而且这种优势会随着架构类型而显著变化。

作者想说明的主要是三件事：

1. 优化器表现不是纯超参数问题，而是和架构设计深度耦合。
2. 谱优化器的优势不是 Muon 一个方法的偶然，而是 spectral family 的共同性质。
3. 这种优势和层级谱结构有关，不同架构中甚至呈现出相反的 layer-wise 相关性，因此背后可能存在不同机制。

### 1.2 主要证据

论文的证据链大致如下：

1. 在 3 个数据集、3 个架构的 20-epoch 对比里，Muon 全面优于 AdamW，差距约为 +5.9 到 +24.3 个点。
2. 在 CIFAR-100 上再加入 SOAP 和 Shampoo 后，三种 spectral methods 都明显优于 AdamW，且 family 内部差距远小于它们相对 AdamW 的优势。
3. 20 epoch 扩展到 50 epoch 后，ResNet-50 和 ConvNeXt-T 上的优势会缩小但不会消失，而 DeiT-S 上几乎不缩小，作者据此区分"收敛速度优势"和"解质量优势"。
4. SGD 在 layer-normalized 架构上几乎失效，而在 ResNet-50 上还算可用，作者据此强调 normalization / architecture 对 optimizer viability 的决定性作用。
5. 通过 layer-wise weight swap，作者声称 Muon 在 ResNet-50 上几乎所有层都是 beneficial，在 DeiT-S 上则更 selective。
6. 通过 spectral response ratio 与 per-layer benefit 的 Spearman correlation，作者给出 CNN 为正、transformer 为负的 sign reversal，并据此提出"不同架构上谱优化器通过不同机制起作用"的解释。

### 1.3 有价值的地方

这篇文章真正有价值的点，不只是"某个新优化器在某些表上更强"，而是它抓住了一个更值得讨论的问题：

> optimizer selection 在现代视觉模型里，可能首先是一个 architecture-coupled structural decision，而不是一个靠统一 recipe 就能抹平的次要训练细节。

如果这个故事成立，它的意义会比单纯报新 SOTA 或报一个 optimizer win-rate 更大，因为它把"优化器表现为何跨 backbone 不稳定"这件事从经验现象往结构性解释推进了一步。

### 1.4 文章的 story telling 和主线

这篇文章当前的主线是比较清楚的：

1. 先用 SGD 在 ResNet-50 和 ConvNeXt-T 上的巨大反差，建立 optimizer-architecture coupling 的问题意识。
2. 再用 Muon vs AdamW 的 3x3 结果说明，谱优化器具有更稳定的跨架构转移性。
3. 然后把 SOAP / Shampoo 拉进来，试图把结论从"Muon 有效"升级到"spectral family 有效"。
4. 接着用 20 vs 50 epoch 对比说明，这种优势有时是更快到达高点，有时是到达了更好的解。
5. 最后用 weight swap 和谱相关分析，把全文从 benchmark comparison 往 mechanism paper 推一步。

前半段是强的，尤其是：

- 主结果矩阵整齐；
- gap 幅度大；
- spectral family comparison 至少部分支持"不是 Muon 偶然"；
- 50-epoch 补充实验帮助回应"只是收敛快一点吗"这个自然质疑。

但后半段要明显更谨慎：

- "主要由 normalization 决定"这个解释目前还没有被干净隔离出来；
- sign reversal 很有趣，但离"机制成立"还有距离；
- 从有限受控实验直接推到"layer norm 架构默认应使用 spectral optimizer"还偏强。

## Part 2. 当前版本的优点和问题

### 2.1 优点

#### 1. 主结果很醒目，而且不是单次偶然

Muon 相对 AdamW 的差距在 9 个配置上全为正，而且不少配置是双位数大幅领先。这种结果在 optimizer paper 里是有吸引力的，不是那种只能靠统计显著性硬撑的小增益。

#### 2. 文章不是只停留在换个 optimizer 分数更高

作者尝试回答三个更深的问题：这种现象是否跨架构稳定、是否只是收敛速度、以及优势主要来自哪些层。这比很多只报最终 accuracy 的优化器论文更有研究味道。

#### 3. spectral family 的视角独特

把 SOAP 和 Shampoo 纳入比较非常重要，因为这让文章有机会把 claim 从"Muon 有效"提升到"利用谱结构这一类方法有效"。

#### 4. 控制实验矩阵

三种架构、三种数据集、统一 batch size / cosine schedule / from-scratch 训练，使得 paper 读起来是一个设计过的 controlled study，而不是杂乱地拼 benchmark。

#### 5. 50-epoch 扩展实验是必要且有价值的

这部分至少让作者没有把所有 gap 都偷换成"就是谱方法更好"，而是承认有些 gap 更像早期收敛收益，有些才像 persistent solution quality gap。

### 2.2 主要问题

#### 问题 1: baseline protocol 过于特殊，导致 external validity 很弱

全文最危险的地方不是结果不显著，而是实验 recipe 太"非主流"，从而让 reviewer 很容易怀疑大 gap 主要来自 baseline 被压低了。

主要风险包括：

- 全部模型 from scratch；
- 数据集较小，且 CIFAR-100 被 resize 到 224x224；
- 没有 warmup；
- 没有 Mixup / CutMix / RandAugment / label smoothing；
- batch size 固定为 128；
- 单卡训练。

对 ResNet-50 来说，这种 recipe 可能还算能跑；但对 DeiT-S、ConvNeXt-T 这类现代架构，AdamW 往往高度依赖更完整的训练 recipe。于是当前论文里的巨大 gap，可能混合了两部分：

1. spectral optimizer 的真实优势；
2. baseline 在一个对其并不友好的 recipe 下被系统性低估。

#### 问题 2: 学习率调优协议不够对称

- sweep 只做 5 epoch；
- 27 个配置里有 7 个在扫到的范围内仍是 monotonic；
- 其中 2 个甚至直接用了 literature defaults；
- Shampoo 还做了额外 refined sweep，且这个 refined sweep 明显改写了其排名（+4.4pp）。

这说明 tuning budget 本身就可能深刻影响结论。

#### 问题 3: "family-level advantage" 的 claim 证据范围偏窄

SOAP 和 Shampoo 只在 CIFAR-100 上被系统纳入主比较，而 Tiny-ImageNet 与 ImageNet-100 实际上只有 Muon 和 AdamW 的主结果。family-level claim 的证据范围小于正文里给人的 impression。

#### 问题 4: "normalization 是 primary mediator" 有严重混杂

当前三个 backbone 同时改变了 normalization 类型、convolution vs attention、standard conv vs depthwise separable conv、token mixing 方式、layer composition。因此是"architecture bundle 差异"而不是"normalization causal effect"的识别实验。

#### 问题 5: 机制部分证据偏薄

- 机制指标是自定义 spectral response ratio；
- 证据主体是相关性分析而非干预性证据；
- ConvNeXt-T 缺少同等强度的 layer-wise analysis；
- ResNet-50 weight-swap 有 BatchNorm running stats 未重算的 caveat。

#### 问题 6: 统计与评估协议不一致

- 混用 3 seed、4 seed 和 single seed；
- IN100 部分 50-epoch 结果是 single-seed estimate；
- 用 best validation accuracy 而非 last-epoch accuracy；
- Welch t-test 基于极小 n。

#### 问题 7: practical recommendation 过强

"任何使用 layer normalization 的架构都应默认用 spectral optimizers"在当前证据下太满——只有 2 个 LN 架构、数据集不大、light recipe、未和近年非谱 optimizer 比较。

#### 问题 8: 缺少与近年非谱 optimizer 的对比

论文只和 AdamW 与 SGD 比较，但 2024 年已有 Lion（NeurIPS 2024）、Prodigy（ICML 2024）等广泛使用的新 optimizer。reviewer 会问：spectral 方法是否只是比"老 baseline"强？

---

## Part 2.5 Appendix 已有资产盘点（执行前必读）

**在设计补充实验前，必须先核对 appendix 已经做了哪些工作，避免重复劳动。**

| Appendix | 已有内容 | 价值 |
|----------|--------|------|
| **A**: Hyperparameter Configurations | 全部 27 个配置的完整超参数，含 Muon hybrid parameter routing（2D 参数走 Newton-Schulz，1D 走 companion AdamW lr=3×10⁻⁴） | 直接可用，无需重做 |
| **B**: Learning Rate Sweep Results | 含 DeiT-S SGD Inverted-U Response（B.1），Tables 6-8 全部 sweep 曲线。**已证明 SGD 在 DeiT-S 上不是没调到位而是结构性失败** | 重要——说明 SGD 退化有 LR sweep 支撑，不需要重新验证 |
| **C**: Multi-Seed Reproducibility | Table 10 多种子结果 | 直接可用 |
| **D**: Extended Training (50 Epochs) + Weight Decay | **已做** ConvNeXt-T SGD wd=0.01 vs wd=10⁻⁴ 的对照（Δ=-0.09pp，不显著），确认 SGD 退化不是 weight decay artifact | 重要——不需要再补 weight decay 实验 |
| **E**: ConvNeXt-T SGD Degradation Analysis | 5-point LR sweep（lr=0.001 到 0.1）+ wd sanity check，**已充分证明 SGD 在 ConvNeXt-T 上的退化不是超参数问题** | 重要——不需要重做 SGD degradation 分析 |
| **F**: Muon Algorithm Background | 完整 Newton-Schulz 推导 + 与 Shampoo 关系 | 直接可用 |
| **G**: Wall-Clock Efficiency | 全部配置 per-epoch wall-clock times，Muon overhead 5-22% | 直接可用 |
| **H**: Alternative Spectral Optimizers | SOAP/Shampoo 在 CIFAR-100 上完整对比（Table 15）| 直接可用，但**仅限 CIFAR-100** |
| **I**: Layer-wise Analysis | Weight swap（ResNet-50 + DeiT-S）+ Spectral divergence correlation + Cross-seed variance (Figure 3) | 可用但有缺口：缺 ConvNeXt-T、缺 BN stats 重算 |

**结论**：Appendix D+E 的 sanity check 已经做得不错，不需要重做。但以下确实是缺口：
- Enhanced recipe 对照（appendix 里完全没有）
- SOAP/Shampoo 的非 CIFAR 数据集结果
- 近年非谱 optimizer baseline
- Normalization swap 因果实验
- ConvNeXt-T layer-wise 机制分析

---

## Part 3. 总体策略：从"防守性修补"升级为"主动补强"

既然时间和资源充裕，策略不应该是"收紧 claim 让证据对齐"（防守），而应该是"补到证据能支撑更强 claim"（进攻）：

| 维度 | 最低限度策略 | 资源充裕时的最优策略 |
|------|-----------|-----------------|
| Baseline recipe | 降 claim 到"under our protocol" | 补 enhanced recipe 实验，证明 gap 在标准 recipe 下仍存在 |
| Family claim | 降 claim 到"仅 CIFAR-100" | SOAP/Shampoo 扩展到全矩阵 |
| Normalization | 降 claim 到"correlated factor" | 补 norm swap 因果实验，坐实 causal claim |
| 机制 | 降 claim 到"suggests" | 补强到能写"provides evidence for" |
| Optimizer coverage | 保持只和 AdamW/SGD 比 | 加 Lion + Prodigy |
| Architecture coverage | 保持 3 个 | 加 Swin-T 到 4 个 |

**核心目标**：让论文从 "interesting but protocol-specific" 升级为 "comprehensive controlled study with causal evidence"。

---

## Part 4. 完整实验方案（按优先级排序）

### P0 级 —— 生死攸关，Day 0-1 完成

#### 实验 1: Enhanced Recipe 对照实验 ★★★★★

**解决问题**：问题 1（baseline 被压低）——全文最大的质疑点

**Enhanced recipe 定义**（对所有 optimizer 统一施加）：
- 5-epoch linear warmup
- Mixup (α=0.8) + CutMix (α=1.0)
- RandAugment (n=2, m=9)
- Label smoothing (ε=0.1)
- 其他不变（batch 128, cosine annealing, from scratch, 20 epoch）

**配置矩阵**：

| 配置 | Optimizer | Epoch | Seeds | 估算时间/run |
|------|-----------|-------|-------|-------------|
| DeiT-S / CIFAR-100 | Muon, AdamW, SOAP | 20 | 3 | ~15 min |
| DeiT-S / ImageNet-100 | Muon, AdamW | 20 | 3 | ~45 min |
| ConvNeXt-T / CIFAR-100 | Muon, AdamW, SOAP | 20 | 3 | ~15 min |
| ConvNeXt-T / ImageNet-100 | Muon, AdamW | 20 | 3 | ~45 min |
| ResNet-50 / CIFAR-100 | Muon, AdamW | 20 | 3 | ~15 min |

**前置**：Enhanced recipe 的 LR 需要重新 sweep
- 5 配置 × ~3 opt × 7 LR × 5 epoch ≈ 105 runs × ~4 min = ~7 GPU-hours

**正式训练**：
- (3+2+3+2+2) × 3 seeds = 36 runs × ~15-45 min = ~15 GPU-hours

**总计 ≈ 22 GPU-hours（2 GPU 并行 ≈ 11 小时）**

**放入论文**：新增 Section 5.X "Robustness Under Enhanced Training Recipe" + Table（正文给关键数字，完整矩阵放 appendix）

**预期结果场景**：
- 场景 A（gap ≥ 5pp）：最佳。直接写 "spectral advantage persists under standard training recipes"
- 场景 B（gap 2-5pp）：仍然好。写 "gap narrows but remains consistent and statistically significant"
- 场景 C（gap ≈ 0）：需调整叙事为 "spectral optimizers excel in simplified/resource-constrained settings"

---

#### 实验 2: SOAP/Shampoo 扩展到 Tiny-ImageNet 和 ImageNet-100 ★★★★★

**解决问题**：问题 3（family-level claim 仅限 CIFAR-100）

| 配置 | Optimizer | Epoch | Seeds |
|------|-----------|-------|-------|
| ResNet-50 / Tiny-ImageNet | SOAP, Shampoo | 20 | 3 |
| DeiT-S / Tiny-ImageNet | SOAP, Shampoo | 20 | 3 |
| ConvNeXt-T / Tiny-ImageNet | SOAP, Shampoo | 20 | 3 |
| ResNet-50 / ImageNet-100 | SOAP, Shampoo | 20 | 3 |
| DeiT-S / ImageNet-100 | SOAP, Shampoo | 20 | 3 |
| ConvNeXt-T / ImageNet-100 | SOAP, Shampoo | 20 | 3 |

**前置**：LR sweep ≈ 8.4 GPU-hours
**正式训练**：18 runs × ~20-45 min ≈ 10 GPU-hours

**总计 ≈ 18 GPU-hours（2 GPU 并行 ≈ 9 小时）**

**放入论文**：升级 Table 1 为完整 5 optimizer × 3 arch × 3 dataset 矩阵。正文从"CIFAR-100 上 family advantage"升级为"全矩阵 family advantage"

---

### P1 级 —— 高优先级，Day 1-2 完成

#### 实验 3: 现代非谱 Optimizer Baseline（Lion + Prodigy）★★★★

**解决问题**：问题 8（未和近年 optimizer 比较）
**选择理由**：
- **Lion**（Chen et al., NeurIPS 2024）：symbolic search 发现的 optimizer，非谱，代表"自动发现的优化算法"路线
- **Prodigy**（Mishchenko & Defazio, ICML 2024）：parameter-free adaptive method，代表"不需要调 LR"路线

| 配置 | Optimizer | Epoch | Seeds |
|------|-----------|-------|-------|
| 3 arch × CIFAR-100 | Lion, Prodigy | 20 | 3 |
| DeiT-S / ImageNet-100 | Lion, Prodigy | 20 | 3 |
| ConvNeXt-T / ImageNet-100 | Lion, Prodigy | 20 | 3 |

**总计 ≈ 16 GPU-hours（2 GPU 并行 ≈ 8 小时）**

**放入论文**：Section 5.1 新增段落 "Comparison with Recent Non-Spectral Optimizers"。如果 spectral 方法仍然领先 Lion/Prodigy → 大幅加强 "not just beating old baselines" 的叙事

---

#### 实验 4: Normalization Swap 因果实验 ★★★★

**解决问题**：问题 4（normalization claim 无因果证据）
**具体做法**：构造 ResNet-50-LN（把 BatchNorm 全部替换为 LayerNorm，其他完全不变）

timm 的 ResNet 实现支持 `norm_layer` 参数，可以直接传入 `nn.LayerNorm`。

**因果逻辑**：
- **预测**：如果 normalization 是关键因素 → ResNet-50-LN 上 SGD 应该退化，spectral gap 应该放大（向 ConvNeXt-T 的模式靠拢）
- **反预测**：如果不是 → SGD 仍然可用，spectral gap 不变

| 配置 | Optimizer | Epoch | Seeds |
|------|-----------|-------|-------|
| ResNet-50-LN / CIFAR-100 | Muon, AdamW, SGD | 20 | 3 |
| ResNet-50-LN / Tiny-ImageNet | Muon, AdamW | 20 | 3 |

**总计 ≈ 8 GPU-hours（1 GPU ≈ 8 小时）**

**放入论文**：新增 Section 5.X "Isolating the Effect of Normalization"（正文 1-2 段）。如果预测成立 → 这将是全文最有力的因果证据，可以在 Discussion 里显著加强 normalization mechanism 的叙事

---

#### 实验 5: Extended LR Tuning（修复 monotonic 和不对称问题）★★★★

**解决问题**：问题 2（LR 调优不对称）

**具体做法**：
1. 对 7 个 monotonic 配置在原 sweep 范围之外向两个方向各扩 2-3 个 LR 点
2. 对 2 个 literature default 配置（Muon/ResNet-50/C100 lr=0.02, SGD/ResNet-50 lr=0.1）做 formal sweep
3. 用 20 epoch（非 5 epoch）验证，确保选出的 LR 不是 short-horizon artifact
4. 如果发现新的最优 LR → 用新 LR 重跑 3-seed 正式实验

**总计 ≈ 20 GPU-hours（2 GPU 并行 ≈ 10 小时）**

**放入论文**：更新 Appendix B sweep 表格，正文 Section 4.2 新增 1-2 句 "Extended tuning confirmed that..."

---

### P2 级 —— 强烈建议，Day 2-3 完成

#### 实验 6: 100-Epoch 长训练实验 ★★★

**解决问题**：50-epoch 已显示 CNN 上 gap 在缩小，reviewer 必问"100 epoch 呢？"

| 配置 | Optimizer | Epoch | Seeds |
|------|-----------|-------|-------|
| ResNet-50 / CIFAR-100 | Muon, AdamW | 100 | 3 |
| DeiT-S / CIFAR-100 | Muon, AdamW | 100 | 3 |
| ConvNeXt-T / CIFAR-100 | Muon, AdamW | 100 | 3 |

**总计 ≈ 12 GPU-hours（2 GPU 并行 ≈ 6 小时）**

**放入论文**：扩展 Table 2 和 Figure 2，新增 100-epoch 数据点。预期：ResNet-50 gap 可能接近 0（纯收敛速度）；DeiT-S gap 仍持续（解质量差异）→ 强化双机制叙事

---

#### 实验 7: 第四架构 Swin-T ★★★

**解决问题**：当前只有 2 个 LN 架构，reviewer 质疑样本量不够

**选择理由**：Swin-T 是另一个广泛使用的 LN 架构，用 shifted window attention（区别于 DeiT-S 的 global attention），~28M 参数，与其他架构可比。增加它可以：
- 把 LN 架构从 2 个增加到 3 个
- 测试 window attention vs global attention 的差异
- 让 "LN 架构上 spectral 更强" 的论点基于 3 个而非 2 个数据点

| 配置 | Optimizer | Epoch | Seeds |
|------|-----------|-------|-------|
| Swin-T / CIFAR-100 | Muon, AdamW, SGD, SOAP | 20 | 3 |
| Swin-T / Tiny-ImageNet | Muon, AdamW | 20 | 3 |
| Swin-T / ImageNet-100 | Muon, AdamW | 20 | 3 |

**总计 ≈ 20 GPU-hours（2 GPU 并行 ≈ 10 小时）**

**放入论文**：Table 1 扩展为 4×3 矩阵。如果 Swin-T spectral gap 也大 → "layer-normalized architectures benefit from spectral methods" 基于 3 个 LN 架构，说服力大增

---

#### 实验 8: 机制分析补强（4 个子任务）★★★

**解决问题**：问题 5（机制证据偏薄）

**8a. ResNet-50 Weight Swap 重算 BatchNorm Stats**（≈ 2 GPU-hours）
- 每次 swap 一层后，在训练集跑一遍 forward pass 重算 BN running mean/var
- 然后再评估 validation accuracy
- 消除当前 caveat，如果结论不变 → 论文可以去掉"we do not recalculate"的 disclaimer

**8b. ConvNeXt-T 完整 Layer-wise Analysis**（≈ 2 GPU-hours）
- Weight swap（同 ResNet-50/DeiT-S 协议）
- Spectral divergence correlation
- 从已有 CIFAR-100 checkpoints 提取即可，无需重新训练

**8c. 对照谱指标**（≈ 2 GPU-hours）
- 除 spectral response ratio 外，额外计算：
  - Effective rank（entropy-based）
  - Condition number (σ₁/σ_min)
  - Nuclear norm ratio
- 对 3 个架构分别计算 Spearman correlation
- 证明 sign reversal 不是 metric-specific artifact

**8d. Per-layer Correlation 按 Layer Type 分组**（< 1 hour）
- 检查 spectral divergence correlation 在 conv / attention / linear 分组后是否仍成立
- 排除 layer type 作为混杂变量

**总计 ≈ 7 GPU-hours（1 GPU ≈ 7 小时）**

**放入论文**：
- 8a → Section 5.5 更新，移除或转化 caveat
- 8b → Appendix I 新增 ConvNeXt-T 子节
- 8c → Appendix 新增对比表，正文 1 句话提及
- 8d → 正文或 appendix 新增分组 correlation 表

---

### P3 级 —— 锦上添花，Day 3-4 或有剩余资源时做

#### 实验 9: 50-Epoch Enhanced Recipe（≈ 12 GPU-hours）
选 3 个最关键配置（DeiT-S/C100, ConvNeXt-T/C100, ResNet-50/C100），在 enhanced recipe 下延长到 50 epoch，观察 gap 演化。

#### 实验 10: SOAP/Shampoo Enhanced Recipe（≈ 4.5 GPU-hours）
在 enhanced recipe 下也跑 SOAP/Shampoo（限 CIFAR-100 × 3 arch），验证 family advantage 在强 recipe 下是否持续。

#### 实验 11: Batch Size Sensitivity（≈ 4 GPU-hours）
测试 batch size 256 和 512 对 spectral gap 的影响。选 DeiT-S + ConvNeXt-T / CIFAR-100，Muon + AdamW。

#### 实验 12: Gradient Spectral Analysis（≈ 8 GPU-hours）
在训练中每隔 N 步保存梯度矩阵 SVD，比较不同 optimizer 下梯度谱演化。实现复杂度较高，有余力再做。

---

## Part 5. GPU 调度时间线

### 资源：4x RTX 5090，连续 4 天

### Day 0（小时 0-12）：基础设施 + LR Sweep

| 时段 | GPU-0 | GPU-1 | GPU-2 | GPU-3 |
|------|-------|-------|-------|-------|
| 0-4h | Enhanced recipe LR sweep (DeiT-S, ConvNeXt-T) | SOAP/Shampoo LR sweep (TinyIN) | SOAP/Shampoo LR sweep (IN100) | Lion/Prodigy LR sweep (C100) |
| 4-8h | Enhanced recipe LR sweep (ResNet-50) | Norm swap (ResNet-50-LN) LR sweep | Swin-T LR sweep (C100) | Extended LR sweep (monotonic configs) |
| 8-12h | 汇总 LR sweep → 确定最优 LR | Extended LR sweep 续 | Swin-T LR sweep (TinyIN, IN100) | **CPU: 修改 abstract/intro claim 初稿** |

### Day 1（小时 12-36）：P0 + P1 核心实验

| 时段 | GPU-0 | GPU-1 | GPU-2 | GPU-3 |
|------|-------|-------|-------|-------|
| 12-20h | Enhanced recipe 正式 (DeiT-S) | Enhanced recipe 正式 (ConvNeXt-T) | SOAP/Shampoo TinyIN 正式 | SOAP/Shampoo IN100 正式 |
| 20-28h | Enhanced recipe 正式 (ResNet-50) | SOAP/Shampoo 续 | Lion/Prodigy C100 正式 | Swin-T C100 正式 |
| 28-36h | **数据分析** → 确认 enhanced gap 方向 | Norm swap 正式 (ResNet-50-LN) | Lion/Prodigy IN100 正式 | Swin-T TinyIN + IN100 正式 |

### Day 2（小时 36-60）：P2 实验 + 机制补强

| 时段 | GPU-0 | GPU-1 | GPU-2 | GPU-3 |
|------|-------|-------|-------|-------|
| 36-44h | Extended LR 重跑 (正式 20ep) | 100-epoch (ResNet-50) | 100-epoch (DeiT-S) | 100-epoch (ConvNeXt-T) |
| 44-52h | 机制 8a: BN stats 重算 | 100-epoch 续 | 100-epoch 续 | 机制 8b: ConvNeXt-T layer-wise |
| 52-60h | 机制 8c+8d: 对照谱指标 | 100-epoch 续 | P3: 50ep enhanced recipe | P3: SOAP/Shampoo enhanced |

### Day 3（小时 60-84）：写作 + P3 + 收尾

| 时段 | GPU-0 | GPU-1 | GPU-2 | GPU-3 |
|------|-------|-------|-------|-------|
| 60-68h | P3: Batch size sensitivity | P3: 50ep enhanced 续 | P3: Gradient spectral | 查漏补缺 / 重跑失败 |
| 68-76h | **全面写作**：整合新结果到论文 | 查漏补缺 | 查漏补缺 | 空闲 |
| 76-84h | **论文定稿** + proofread + figure polish | 数据 double-check | 数据 double-check | 备用 |

### GPU-hours 汇总

| 优先级 | 实验 | GPU-hours | 状态 |
|--------|------|-----------|------|
| P0 | Enhanced recipe 对照 | 22h | 必做 |
| P0 | SOAP/Shampoo 全矩阵扩展 | 18h | 必做 |
| P1 | Lion + Prodigy baseline | 16h | 必做 |
| P1 | Normalization swap 因果实验 | 8h | 必做 |
| P1 | Extended LR tuning | 20h | 必做 |
| P2 | 100-epoch 长训练 | 12h | 强烈建议 |
| P2 | Swin-T 第四架构 | 20h | 强烈建议 |
| P2 | 机制分析补强 (8a-8d) | 7h | 强烈建议 |
| P3 | 50ep enhanced / batch size / gradient spectral 等 | ~28h | 如有余量 |
| **总计** | | **~151h** | 4 GPU × 4 天 = 384h 预算，利用率 ~39% |

剩余 ~233 GPU-hours 用于：LR sweep 返工、失败重跑、unexpected 需求、seed 补充。

---

## Part 6. 写作修改方案（与实验并行在 CPU 上执行）

### 6.1 Abstract 重写

当前问题："fundamentally different optimization mechanisms"、"sign reversal exposes" 等 claim 过满。

修改方向：
- "exposes fundamentally different optimization mechanisms" → "provides evidence for architecture-dependent optimization mechanisms"
- 新增 enhanced recipe 结果概括
- 如果加了 Swin-T → "four architectures"
- 如果加了 Lion/Prodigy → "outperforming both classical (AdamW, SGD) and recent (Lion, Prodigy) baselines"

### 6.2 Introduction 收紧 + 扩展

- 保留核心问题（optimizer-architecture coupling），这是好问题
- Findings 列表加限定："Under our controlled protocol..."
- 新增 finding 关于 enhanced recipe / norm swap / 新 baseline

### 6.3 Section 5 结构调整

新增小节（根据实验结果按需添加）：
- **5.X Robustness Under Enhanced Training Recipe**（实验 1）
- **5.X Comparison with Recent Non-Spectral Methods**（实验 3）
- **5.X Isolating Normalization Effects**（实验 4）
- 更新 5.1 和 5.2 加入 Swin-T（实验 7）

### 6.4 Section 6 Discussion 重写

- 如果 norm swap 成功 → 保持 "normalization as primary mediator" 但加因果证据支撑
- 如果 norm swap 不成功 → 降级为 "one important contributing factor"
- 新增 Lion/Prodigy 讨论段落
- practical recommendation 基于证据强度校准

### 6.5 Appendix 更新

- Appendix A: 新增所有新 optimizer 配置
- Appendix B: 新增 extended sweep 结果
- Appendix D: 新增 100-epoch 结果
- Appendix H: 从仅 CIFAR-100 扩展到完整矩阵
- 新增 Appendix: Enhanced Recipe Details
- 新增 Appendix: Swin-T Results
- 新增 Appendix: Lion/Prodigy Configuration
- Appendix I: 更新 BN-recalculated weight swap + ConvNeXt-T layer-wise + 对照谱指标

### 6.6 Figure 更新

- Figure 1 (heatmap): 从 3×3 扩展为 4×3（如果加 Swin-T）
- Figure 2 (gap evolution): 新增 100-epoch 数据点
- 新增 Figure: Enhanced recipe vs light recipe 的 gap 对比（并排 bar chart）
- 新增 Figure: Norm swap 结果（ResNet-50 vs ResNet-50-LN 的 optimizer ranking 变化）

---

## Part 7. 风险与应对

| 风险 | 概率 | 应对 |
|------|------|------|
| Enhanced recipe 下 gap 消失 | 中 | 不是坏结果。调整叙事为"regime-dependent advantage"，仍有论文价值 |
| Lion/Prodigy 实现有 bug | 中 | 先用 C100/ResNet-50 做 smoke test；实在不行只做 Lion |
| ResNet-50-LN 在 timm 中实现困难 | 低 | timm ResNet 支持 `norm_layer` 参数；退路是 torchvision 手动替换 |
| Swin-T 训练超时 | 低 | ~28M params，与 ConvNeXt-T 相当；如超时只做 CIFAR-100 |
| Norm swap 不支持因果假说 | 中 | 诚实报告，降级为 "correlated factor"。reviewer 尊重诚实 |
| 某 GPU 故障 | 低 | 3 GPU 可完成 P0+P1；P3 可砍 |
| LR sweep 范围仍不够 | 低 | monotonic 情况继续外扩直到找到峰值 |

---

## Part 8. 执行检查清单

### Day 0 结束检查
- [ ] 全部 LR sweep 完成，最优 LR 确定
- [ ] Abstract/Introduction claim 初稿修改完成
- [ ] 训练代码改造完成：enhanced recipe、Lion/Prodigy 集成、ResNet-50-LN 变体、Swin-T 配置

### Day 1 结束检查
- [ ] P0 实验 1（Enhanced recipe）全部正式训练完成
- [ ] P0 实验 2（SOAP/Shampoo 扩展）≥ 80% 完成
- [ ] P1 实验 3（Lion/Prodigy）CIFAR-100 完成
- [ ] P2 实验 7（Swin-T）CIFAR-100 完成
- [ ] **关键决策点**：Enhanced recipe 下 gap 方向和量级已知 → 确定论文叙事走向

### Day 2 结束检查
- [ ] 全部 P0 + P1 实验完成
- [ ] P2 实验 6（100-epoch）≥ 50% 完成
- [ ] P2 实验 8（机制补强）全部完成
- [ ] 新 Table 1（完整矩阵）和 enhanced recipe Table 数据填充完成
- [ ] Section 5 新增小节初稿完成

### Day 3 结束检查
- [ ] 全部 P2 实验完成，P3 尽可能多做
- [ ] 论文全文修改完成（正文 + appendix + figures）
- [ ] 所有数字 double-checked
- [ ] NeurIPS checklist 更新

### 提交前最终检查
- [ ] 所有 claim 与证据对齐
- [ ] 所有 figure/table 编号正确
- [ ] 参考文献完整
- [ ] 匿名化无遗漏
- [ ] 页数符合要求
