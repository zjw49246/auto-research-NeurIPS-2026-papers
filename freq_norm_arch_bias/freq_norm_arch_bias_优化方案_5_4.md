# 审稿与优化方案：Normalization Layers as Implicit Spectral Filters

## 论文信息
- **标题**: Normalization Layers as Implicit Spectral Filters: Optimizer Regime Shapes Frequency Bias Across Vision Architectures
- **投稿目标**: NeurIPS 2026
- **截稿日期**: 2026-05-06
- **审稿日期**: 2026-05-04
- **可用硬件**: 2-4 × RTX 5090 (32GB) 或 RTX PRO 6000 (96GB)
- **执行模式**: Agent Auto Research，不间断运行

---

## 第一部分：论文审稿与现状分析

### 1.1 核心叙事

论文提出将归一化层（BN/LN）视为**隐式频谱滤波器**，其滤波方向不是归一化类型本身决定的，而是由**优化器制度**（具体是有效更新幅度）决定的。

核心论证链条：
1. 定义 Δ_{H/L}（BN 与 LN 之间的高低频能量比中位数差异）作为诊断指标
2. 在 6 个架构（4 个家族）上测量 Δ_{H/L}，发现所有 AdamW 配置产生非正值，仅 SGD+CNN 产生正值
3. LINCHPIN 实验：同一 ResNet-50 仅换优化器，Δ_{H/L} 从 +0.454 骤降到 −0.016
4. 学习率梯度实验：SGD 下 lr 从 0.1→0.01→0.001，Δ_{H/L} 单调衰减至零
5. 结论：有效更新幅度是 BN-LN 频谱差距方向的主要决定因素

### 1.2 已完成的实验清单（含附录）

**主实验（正文）：**
- 6 架构 × BN/LN × SGD/AdamW 的 Δ_{H/L} 测量（Table 1）
- LINCHPIN 实验：ResNet-50 双优化器对比
- 跨架构 sign test（m=5, p=0.031）
- Epoch-resolved Δ_{H/L} 轨迹（Figure 1c, Figure 8, 9）
- Block collapse 分析（DeiT-S+SGD, Figure 1d）
- ImageNet-100 scale-up 验证（ResNet-50 SGD/AdamW, DeiT-S AdamW）
- 深度分解分析（Figure 1f, Figure 7）

**附录已有实验（务必利用，不要重复）：**

| 附录 | 内容 | 关键结果 |
|------|------|---------|
| B | 完整超参数配置 (Table 2)、数据增强 (Table 3)、训练时长 (Table 4) | 基准配置完备 |
| C.1 | BH 校正 (Table 5, m=13) | 10/13 配置显著 |
| C.2 | Wilcoxon 检验 (Table 6) | 与 bootstrap 结论一致 |
| C.3 | Sign test 敏感性 (Table 8, m=4→7) | 所有标准下方向一致 |
| C.4 | ConvNeXt-T trimmed bootstrap | 去异常值后更显著 |
| D | 归一化效应量 (Table 9)、逐 seed Δ_{H/L} (Table 10, 11) | 跨架构归一化后量级一致 |
| E.1 | EfficientNet-B0 详细分析 + LINCHPIN 复制 | 第二个 CNN 上复制 LINCHPIN |
| E.2 | MambaVision-T+SGD | 不确定性结论，n=6 |
| E.3 | VMamba-S+SGD | 边缘显著，p_adj=0.049 |
| F.1 | ImageNet-100 scale-up (224×224) | ResNet-50 SGD: +0.357, AdamW: +0.010 |
| F.2 | CIFAR-100 (32×32) 全因子设计 (Table 12, 13) | 32×32 分辨率打破 pattern |
| G.1 | Lion 优化器（ResNet-50 失败） | ResNet-50+Lion 训练崩溃 |
| G.2 | Lion 完整分析（DeiT-S+Lion） | Δ_{H/L} ≈ 0，Lion 中和频谱差距 |
| G.3 | Constant-LR 控制 (Figure 3) | 排除 cosine schedule 干扰 |
| G.4 | Weight decay 敏感性 W1a (Table 14) | BN 对 WD 鲁棒，LN 受 WD 影响 |
| H | LN+SGD 训练不稳定性 (Figure 4) | 1/4 seed 崩溃 |
| I.1 | Block collapse 扩展分析 | SGD 在非 CNN 上均崩溃 |
| I.2 | 随机初始化基线 (Table 15) | 训练前即有 BN-LN 差异，但训练改变方向 |
| I.3 | 学习率对齐控制 C1 (Table 16, 17) | lr=0.001 时 SGD≈AdamW |
| I.4 | AdamW lr 梯度 MF3 (Table 18) | lr 与 Δ_{H/L} 的非单调关系 |
| J | 腐蚀鲁棒性 (Table 19, 20) | 16/16 条件下 sign 保持；HF-LF gap 与 Δ_{H/L} 一致 |
| K.1 | 空间截断敏感性 (Table 21) | c=4→16 方向一致 |
| K.2 | 崩溃阈值敏感性 | ε=0.01→0.20 方向不变 |
| K.3 | 有效样本量修正（block bootstrap + n_eff） | CI 扩展 1.0-2.7× |
| K.4 | BN leave-one-out 敏感性 (Table 22) | 去除任一 seed 结果稳定 |
| K.5 | 频率截断敏感性 (Table 23) | 9/10 配置跨 4 个 cutoff 方向一致 |
| K.6 | **Pre-normalization 分析 (Table 24)** | SGD 下 BN 放大 3×，AdamW 下擦除 |
| K.7 | 空间分辨率滤波器详情 (Table 25) | 保留率 21%-100% |
| K.8 | 完整数学推导 | FFT + radial binning 公式 |
| L | 附加图表 (Figure 5-9) | per-layer 频谱、epoch 轨迹等 |
| M.1 | 扩展讨论：纹理偏置假说、NAS 启示、迁移学习预测 | 仅为假说，无实验验证 |
| M.2-M.6 | CIFAR-100 过拟合、跨分辨率、架构敏感性等 | 多角度分析 |
| N | 额外局限性 | LN+SGD seed 敏感性等 |

### 1.3 当前论文的核心弱点

#### 1.3.1 引用问题

**严重问题：**

1. **Rippel et al. 2015 — 错误引用**
   - 位置：第 104-105 行
   - 问题：论文将其引用为 "connects high-frequency content to robustness"，但 Rippel et al. 2015 ("Spectral Representations for Convolutional Neural Networks", NeurIPS 2015) 实际是关于 spectral pooling 和频域参数化的效率方法，与鲁棒性无关。全文不包含 "adversarial" 和 "robustness"。
   - 修复建议：删除该引用，或替换为 Tsipras et al. 2019 "Robustness May Be at Odds with Accuracy"

2. **Andriushchenko et al. 2024 — 作者顺序错误**
   - 位置：参考文献第 543 行
   - 问题：写为 "Maksym Andriushchenko, Francesco D'Angelo, ..."，但 arXiv、NeurIPS 会议录、ACM DL 三方确认第一作者为 Francesco D'Angelo
   - 修复建议：改为 "Francesco D'Angelo, Maksym Andriushchenko, Aditya Varre, and Nicolas Flammarion"，正文引用改为 "D'Angelo et al. [2024]"

**中等问题：**

3. **Soudry et al. 2018 — 表述不精确**
   - 位置：第 117 行 "SGD selects max-margin solutions [Soudry et al., 2018]"
   - 问题：原论文证明的是 GD（梯度下降），不是 SGD
   - 修复建议：改为 "Gradient descent selects max-margin solutions" 或补充后续 SGD 扩展工作的引用

4. **Raghu et al. 2021 — 描述略有拉伸**
   - 位置：第 108-109 行 "more uniform spatial attention across frequency bands"
   - 问题：Raghu 原文用 CKA 分析表征相似性，未做频域分析。"frequency bands" 是本文加的框架
   - 修复建议：改为 "more uniform representations across layers"，或明确说明是本文对 Raghu 发现的频域解读

5. **Huang et al. 2025 (InceptionMamba) — 年份和级别**
   - 问题：实际发表于 2026 年 Neural Processing Letters (volume 58)，而非标注的 2025 年
   - 修复建议：更新年份为 2026，标注期刊名

**轻微问题：**

6. **Patro et al. 2025 (SpectFormer) — 标题小错**
   - 实际标题用 "What" 不是 "all"："Frequency and Attention is What You Need"
   - 修复建议：核对并修正标题

7. **Wang et al. 2020 — 引用语境略偏**
   - 被引为 "connecting high-frequency content to robustness"，但原论文核心贡献是关于泛化性
   - 修复建议：改为 "connects high-frequency exploitation to generalization and robustness"

8. **Piao et al. 2024 (FredNormer) — 未正式发表**
   - 仅为 arXiv 预印本，ICLR 2025 投稿似乎被撤回
   - 修复建议：标注为 arXiv preprint

9. **Frankle et al. 2020 — 年份约定**
   - 基于 arXiv 日期标注为 2020，正式发表于 ICLR 2021
   - 修复建议：建议标注正式会议 ICLR 2021

#### 1.3.2 实验设计弱点

1. **机制解释只停留在相关性**："有效更新幅度"与频谱差距相关，但缺乏因果推导或理论分析
2. **单 seed 配置过多**：Table 1 中 DeiT-S+AdamW、EfficientNet-B0+AdamW 等仅用 1 个 seed，需补充至少 3 个 seed
3. **SGD 跨架构覆盖不足**：SGD 只在 ResNet-50 + EfficientNet-B0（需 warmup）上产生可度量正偏置
4. **仅 BN vs LN**：未测试 GroupNorm、RMSNorm 等现代归一化方式
5. **无下游因果验证**：M.1 提出了纹理偏置假说，但未实验验证 Δ_{H/L} 的变化是否导致下游行为变化
6. **Sign test 功效低**：m=5, p=0.031，需增加更多架构配置提升统计功效
7. **MambaVision-T 对 cutoff 敏感**：需讨论或排除该不稳定配置
8. **分辨率局限**：主要实验在 Tiny ImageNet (64×64)，需更多 full ImageNet (224×224) 验证

#### 1.3.3 核心 Idea 价值评估

大方向合理且有新颖性，但机制解释深度不足。补齐实验后可达到中等偏上的会议论文水平。具体中稿率评估见第二部分 §2.8。

---

## 第二部分：优化方案

### 2.1 叙事优化

**当前叙事问题**：论文定位为纯诊断性研究（"we measure and characterize rather than prescribe"），但未能给出机制解释或下游因果证据，导致审稿人的核心质疑为 "so what?"。

**优化方向**：将叙事从"诊断性发现"升级为"带有机制洞察和因果验证的诊断性发现"

**具体调整**：

1. **Introduction 增加机制预告**：在贡献列表中增加 toy model 分析的贡献点（如果方案 C 成功），或增加下游因果验证的贡献点（方案 B 回退）

2. **Section 5 (Discussion) 大幅加强**：
   - 当前 Discussion 主要是现象描述和局限性
   - 加入：(a) 简化模型的机制分析结果，(b) 纹理偏置因果验证结果
   - 将 M.1 中的"假说"升级为"经验证的预测"

3. **Related Work 补充**：
   - 加入并发工作 arXiv:2604.01563 的讨论（"Does Your Optimizer Care How You Normalize?"），作为独立验证的佐证
   - 加入 Yunis et al. 2024（"Approaching deep learning through the spectral dynamics of weights"）的讨论，连接权重频谱与表征频谱

4. **Conclusion 增强**：从"我们描述了一个现象"升级为"我们描述了现象、初步揭示了机制、并验证了下游影响"

### 2.2 引用修复（CPU 工作，Day 1 完成）

| 问题 | 修复操作 |
|------|---------|
| Rippel et al. 2015 错引为 robustness | 删除或替换为 Tsipras et al. 2019 |
| D'Angelo et al. 2024 作者顺序 | 正文改为 D'Angelo et al.，参考文献修正作者序 |
| Soudry 2018 "SGD" | 改为 "gradient descent" 或补充 SGD 扩展引用 |
| Raghu 2021 "frequency bands" | 改为 "more uniform representations across layers" |
| Huang 2025 年份 | 改为 2026，标注 Neural Processing Letters |
| Patro 2025 标题 | 修正为 "What You Need" |

### 2.3 实验补充方案（按优先级排序）

---

#### 【P0 — 必须完成 | Day 1-2 | 直接提升统计可信度和机制深度】

##### P0-1：单 seed 配置补种（GPU 工作）
**目的**：消除"单 seed 不可靠"的审稿质疑，这是 Table 1 最大的统计弱点  
**具体操作**：为所有标记 ^{1s} 的配置补充至 3 个 seed

| 配置 | 当前 seeds | 需补 seeds | 每 run 时间 | 总 GPU 时间 |
|------|-----------|-----------|-----------|-----------|
| DeiT-S+AdamW (TinyIN) | s42 | s0, s123 → BN+LN 各 2 runs = 4 runs | 4-5h | ~18h |
| EfficientNet-B0+AdamW (TinyIN) | s42, s123 | s0 → BN+LN = 2 runs | ~3h | ~6h |
| EfficientNet-B0+SGD (TinyIN) | s42 | s0, s123 → BN+LN = 4 runs | ~3h | ~12h |
| VMamba-S+SGD (TinyIN) | s42 | s123 → BN+LN = 2 runs（NaN 风险，需监控） | 5-6h | ~11h |
| DeiT-S+AdamW (IN100) | s42 | s123 → BN+LN = 2 runs | 取决于 IN100 训练时间 | ~10h |

**总计**：~14 runs × ~4h avg = ~56 GPU-hours  
**4 卡并行**：~14h 完成  
**分配**：GPU 0 + GPU 1 同时开跑，Day 1 早上启动，Day 1 晚间完成

##### P0-2：Toy Model 理论分析（CPU 工作，与 GPU 实验并行）
**目的**：给"有效更新幅度驱动频谱差距"提供定性的理论支撑  
**方法**：在简化设定下推导 BN 的频域增益与学习率的关系

**具体方案**：
1. **设定**：单层线性网络 + BN/LN，输入为合成信号（已知频率成分）
2. **分析**：
   - 将 BN 的 batch 统计量（均值 μ_B、方差 σ²_B）分解到频域
   - 推导一步梯度更新后，BN 和 LN 各自在不同频率带上的增益 G(ω)
   - 证明当 lr 足够大时，BN 的 G(ω_high)/G(ω_low) > LN 的对应比值
3. **验证**：用合成实验验证 toy model 的预测与真实网络的行为定性一致
4. **产出**：一个新的小节（如 §3.5 或 §5.1），配合 1-2 个图，论述机制

**关键洞察方向**（基于已有 K.6 pre-normalization 分析）：
- Table 24 已经显示：SGD 下 BN 将 pre-existing gap 从 +0.151 放大到 +0.454（3×），而 AdamW 下从 +0.180 擦除到 −0.016
- 这说明 BN 在 SGD 下是**频率选择性放大器**，在 AdamW 下是**频率选择性衰减器**
- Toy model 只需解释这个不对称性即可

**风险**：理论可能推不出干净的闭式解  
**回退**：如果闭式推导失败，改为数值模拟的 toy model（见 P0-2b 回退方案）

##### P0-2b：回退方案 — 数值 Toy Model（如果解析推导失败）
**目的**：即使无闭式解，也能给出 toy model 级别的机制证据  
**方法**：
1. 构建 3 层 CNN + BN/LN 的小网络（~10K 参数）
2. 在合成频率数据上训练，控制输入频率成分
3. 逐层测量 BN/LN 对不同频率带的增益，扫描 lr
4. 绘制 lr vs 频率增益不对称性 的图，验证与真实网络的定性一致性

**GPU 时间**：极少（toy model 几分钟一个 run），可在 CPU 上完成  
**产出**：1-2 个图 + 机制讨论段落

##### P0-3：纹理偏置因果验证（CPU/GPU 混合，利用现有 checkpoint）
**目的**：验证 M.1 中的核心假说——"SGD+BN 的正 Δ_{H/L} 导致更强的纹理偏置"  
**方法**：使用 Geirhos et al. 的 cue-conflict stimuli（texture-shape 冲突刺激图像）

**具体操作**：
1. 收集/下载 cue-conflict 数据集（公开可用）
2. 在 4 个现有 ResNet-50 checkpoint 上评估（SGD+BN, SGD+LN, AdamW+BN, AdamW+LN）
3. 测量各配置的 texture accuracy vs shape accuracy
4. 预测：SGD+BN（Δ_{H/L}=+0.454）应该有最高的 texture accuracy
5. 对比 ImageNet-100 (224×224) 的 checkpoint 以确认在标准分辨率上也成立

**但注意**：论文 §5 第 468-469 行指出"64×64 of Tiny ImageNet precludes use of 224×224 cue-conflict stimuli"。因此这个实验**只能在 ImageNet-100 (224×224) 的 checkpoint 上做**。已有 ResNet-50+SGD (IN100) 和 ResNet-50+AdamW (IN100) 的 checkpoint。

**额外需求**：需要训练 ResNet-50+SGD+LN 和 ResNet-50+AdamW+LN 在 IN100 上的模型（如果还没有的话）  
**GPU 时间**：如果需要补训 IN100 模型 ~2 runs × ~5h = ~10h；cue-conflict 评估本身 <1h  
**产出**：一个新表 + 一段因果验证讨论

---

#### 【P1 — 高优先级 | Day 1-3 | 扩展归一化和架构覆盖面】

##### P1-1：GroupNorm 实验
**目的**：扩展到第三种归一化类型，回应 Limitations 中"only BN/LN tested"  
**方法**：在 ResNet-50 上同时训练 GroupNorm (G=32) 变体

| 配置 | 数据集 | Seeds | Runs | GPU 时间 |
|------|--------|-------|------|---------|
| ResNet-50+GN+SGD | TinyIN | s42, s123 | 2 | ~7h |
| ResNet-50+GN+AdamW | TinyIN | s42, s123 | 2 | ~7h |

**分析**：计算 Δ_{BN/GN} 和 Δ_{GN/LN}，检验 GN 是否在频域上表现为 BN 和 LN 的中间态  
**预测**：GN 的 group-wise 标准化应该产生介于 BN 和 LN 之间的频谱效应  
**总 GPU 时间**：~14h（2 卡并行 ~7h）

##### P1-2：RMSNorm 实验
**目的**：RMSNorm 是 LLM/现代 ViT 中越来越常用的归一化方式，覆盖它增加论文的时效性  
**方法**：在 DeiT-S 上训练 RMSNorm 变体（DeiT-S 默认用 LN，RMSNorm 是 LN 的去均值版本）

| 配置 | 数据集 | Seeds | Runs | GPU 时间 |
|------|--------|-------|------|---------|
| DeiT-S+RMSNorm+AdamW | TinyIN | s42, s123 | 2 | ~9h |

**分析**：比较 LN vs RMSNorm 在 AdamW 下的频谱差异  
**总 GPU 时间**：~9h

##### P1-3：增加 SGD 下的 CNN 覆盖（VGG-16 或 WideResNet-28）
**目的**：目前 SGD 产生正 Δ_{H/L} 只在 ResNet-50 和 EfficientNet-B0 (with warmup) 上确认。增加一个经典 CNN 可加强 "SGD+CNN→正偏置" 的普适性  
**方法**：选择 VGG-16-BN（经典无 skip connection 的 CNN），在 TinyIN 上训练 BN 和 LN 变体

| 配置 | Seeds | Runs | GPU 时间 |
|------|-------|------|---------|
| VGG-16+BN+SGD | s42, s123 | 2 | ~6h |
| VGG-16+LN+SGD | s42, s123 | 2 | ~6h |
| VGG-16+BN+AdamW | s42, s123 | 2 | ~6h |
| VGG-16+LN+AdamW | s42, s123 | 2 | ~6h |

**总 GPU 时间**：~24h（4 卡并行 ~6h）  
**预测**：VGG-16+SGD 应产生正 Δ_{H/L}，VGG-16+AdamW 应产生非正值  
**额外价值**：VGG 无 residual connection，如果 pattern 一致，排除 skip connection 作为混淆因子

---

#### 【P2 — 中优先级 | Day 2-4 | 深化机制和扩展验证】

##### P2-1：Per-layer 梯度频域分解（CPU/GPU 混合）
**目的**：直接测量不同层/不同频率带的梯度幅度，验证"大 lr 下高频梯度更大"的机制假说  
**方法**：
1. 在 ResNet-50+SGD (lr=0.1) 和 ResNet-50+AdamW 训练过程中，记录每层梯度
2. 对梯度做 2D FFT，分高低频带计算梯度能量
3. 比较 SGD vs AdamW 在高频带 vs 低频带的梯度幅度比
4. 关联 K.6 的 pre-normalization 结果

**GPU 时间**：需要重新训练 2 个 ResNet-50 runs with gradient logging，~8h  
**产出**：直接的梯度频域证据，支撑机制解释

##### P2-2：迁移学习验证
**目的**：验证 M.1 中的第二个预测——"SGD+BN 预训练的 ResNet 用 AdamW 微调，Δ_{H/L} 应向零收敛"  
**方法**：
1. 取已有的 ResNet-50+SGD+BN (IN100) checkpoint
2. 用 AdamW 微调 20-50 epochs
3. 测量微调前后的 Δ_{H/L} 变化轨迹

**GPU 时间**：~3-5h  
**产出**：验证论文的可测试预测，增加实用价值

##### P2-3：额外优化器（LAMB）
**目的**：LAMB 是介于 SGD 和 Adam 之间的优化器（layer-wise adaptive），测试它是否产生中间态的频谱效应  
**方法**：在 ResNet-50 上训练 LAMB

| 配置 | Seeds | Runs | GPU 时间 |
|------|-------|------|---------|
| ResNet-50+BN+LAMB | s42 | 1 | ~4h |
| ResNet-50+LN+LAMB | s42 | 1 | ~4h |

**总 GPU 时间**：~8h

##### P2-4：完整的 AdamW lr-gradient（修复 MF3 不一致性）
**目的**：当前 MF3（Table 18）的 warmup 策略不一致（lr≥0.003 用 5-epoch linear warmup，低 lr 用 standard cosine），导致无法做 clean lr regression  
**方法**：统一所有 lr 点的 warmup 为 5-epoch linear warmup，重跑 lr ∈ {0.0005, 0.001, 0.002, 0.003}

| 配置 | Runs | GPU 时间 |
|------|------|---------|
| ResNet-50 BN+LN × 4 lr points | 8 | ~28h |

**分配**：如果 GPU 有空闲时段再做，优先级低于 P0 和 P1

---

#### 【P3 — 低优先级 | Day 3-4 | 锦上添花】

##### P3-1：Swin Transformer（新架构）
**目的**：Swin-T 是 hierarchical ViT，带有 shifted window attention，与 DeiT 的 global attention 不同  
**方法**：在 TinyIN 上训练 Swin-T BN/LN + AdamW

##### P3-2：Full ImageNet-1K（如有 96GB 卡）
**目的**：从 ImageNet-100 扩展到 Full ImageNet，彻底消除"toy dataset"质疑  
**方法**：至少训练 ResNet-50+SGD+BN/LN 在 Full ImageNet 上（90 epochs）  
**GPU 时间**：每 run ~12-15h on 96GB card  
**注意**：如果用 5090 (32GB)，batch size 可能需要调整

##### P3-3：SignSGD 优化器
**目的**：SignSGD 只用梯度的符号更新（类似于 Adam 的签名部分），可以测试"是梯度幅度还是方向决定频谱偏置"  
**方法**：ResNet-50+BN/LN+SignSGD

##### P3-4：跨架构 corruption robustness 扩展
**目的**：当前 Appendix J 的 corruption 分析只在 ResNet-50 上做了 4 配置；扩展到 DeiT-S  
**已有基础**：Table 20 已经有 DeiT-S+AdamW 和 EfficientNet-B0+AdamW 的 corruption 结果

---

### 2.4 方案 C 失败的回退策略（回退到方案 B）

**判断标准**：如果 Day 2 结束时，Toy Model 分析出现以下情况之一，则放弃方案 C：
1. 解析推导无法给出与实验定性一致的结论
2. 数值 toy model 的行为与真实网络的行为不一致
3. 分析过于复杂，无法在截稿前写成清晰的论文段落

**回退操作**：
1. 删除机制分析相关内容
2. 保留 P0-3 的纹理偏置因果验证（方案 B 的核心）
3. 加强 P2-1 的梯度频域分解作为"towards mechanism"的初步证据
4. 在 Discussion 中将机制分析改为"open question + preliminary evidence"
5. 贡献列表调整为：(1) 跨架构诊断 (2) LINCHPIN 因果证据 (3) 下游因果验证 (4) 实践建议

**方案 B 的最低要求**（必须在截稿前完成）：
- P0-1 多 seed 复制 ✓
- P0-3 纹理偏置验证 ✓
- P1-1 或 P1-3 至少一个（扩展覆盖面）✓
- 引用修复 ✓

---

### 2.5 四天饱和执行计划

#### GPU 资源分配总览

**假设 4 × RTX 5090，96 小时连续运行**

```
GPU 总预算：4 卡 × 96h = 384 GPU-hours
Table 4 单 run 时间：3-6h（取决于架构）
预计可完成：~70-90 training runs
```

---

#### Day 1（5 月 4 日）— 启动关键实验 + 理论工作

**GPU 调度（4 卡全部启动）：**

| GPU | 时间段 | 任务 | 优先级 |
|-----|--------|------|--------|
| GPU 0 | 0-5h | DeiT-S+AdamW (TinyIN) BN s0 | P0-1 |
| GPU 0 | 5-10h | DeiT-S+AdamW (TinyIN) LN s0 | P0-1 |
| GPU 0 | 10-15h | DeiT-S+AdamW (TinyIN) BN s123 | P0-1 |
| GPU 0 | 15-20h | DeiT-S+AdamW (TinyIN) LN s123 | P0-1 |
| GPU 0 | 20-24h | DeiT-S+AdamW (IN100) BN s123 | P0-1 |
| GPU 1 | 0-3h | EfficientNet-B0+SGD (TinyIN) BN s0 | P0-1 |
| GPU 1 | 3-6h | EfficientNet-B0+SGD (TinyIN) LN s0 | P0-1 |
| GPU 1 | 6-9h | EfficientNet-B0+SGD (TinyIN) BN s123 | P0-1 |
| GPU 1 | 9-12h | EfficientNet-B0+SGD (TinyIN) LN s123 | P0-1 |
| GPU 1 | 12-15h | EfficientNet-B0+AdamW (TinyIN) BN s0 | P0-1 |
| GPU 1 | 15-18h | EfficientNet-B0+AdamW (TinyIN) LN s0 | P0-1 |
| GPU 1 | 18-24h | VMamba-S+SGD (TinyIN) BN s123 | P0-1 |
| GPU 2 | 0-4h | VGG-16+BN+SGD (TinyIN) s42 | P1-3 |
| GPU 2 | 4-8h | VGG-16+LN+SGD (TinyIN) s42 | P1-3 |
| GPU 2 | 8-12h | VGG-16+BN+AdamW (TinyIN) s42 | P1-3 |
| GPU 2 | 12-16h | VGG-16+LN+AdamW (TinyIN) s42 | P1-3 |
| GPU 2 | 16-20h | ResNet-50+GN+SGD (TinyIN) s42 | P1-1 |
| GPU 2 | 20-24h | ResNet-50+GN+AdamW (TinyIN) s42 | P1-1 |
| GPU 3 | 0-5h | DeiT-S+RMSNorm+AdamW (TinyIN) s42 | P1-2 |
| GPU 3 | 5-10h | DeiT-S+RMSNorm+AdamW (TinyIN) s123 | P1-2 |
| GPU 3 | 10-14h | ResNet-50+SGD+LN (IN100) s42（如果没有） | P0-3 前置 |
| GPU 3 | 14-18h | ResNet-50+AdamW+LN (IN100) s42（如果没有） | P0-3 前置 |
| GPU 3 | 18-24h | VMamba-S+SGD (TinyIN) LN s123 | P0-1 |

**CPU 并行工作（由另一个 Agent 执行）：**
- Toy Model 理论推导/数值模拟 (P0-2)
- 引用修复（P0 级 CPU 工作）
- 下载 cue-conflict 数据集（为 P0-3 做准备）
- 对已完成 checkpoint 运行 spectral extraction pipeline

**Day 1 产出目标**：
- ~24 training runs 启动/完成
- Toy model 初步结果
- 引用修复完成

---

#### Day 2（5 月 5 日）— 完成核心实验 + 分析

**GPU 调度：**

| GPU | 时间段 | 任务 | 优先级 |
|-----|--------|------|--------|
| GPU 0 | 0-5h | DeiT-S+AdamW (IN100) LN s123 | P0-1 |
| GPU 0 | 5-9h | VGG-16+BN+SGD (TinyIN) s123 | P1-3 |
| GPU 0 | 9-13h | VGG-16+LN+SGD (TinyIN) s123 | P1-3 |
| GPU 0 | 13-17h | VGG-16+BN+AdamW (TinyIN) s123 | P1-3 |
| GPU 0 | 17-21h | VGG-16+LN+AdamW (TinyIN) s123 | P1-3 |
| GPU 0 | 21-24h | 空闲 / 重跑失败 runs | — |
| GPU 1 | 0-4h | ResNet-50+GN+SGD (TinyIN) s123 | P1-1 |
| GPU 1 | 4-8h | ResNet-50+GN+AdamW (TinyIN) s123 | P1-1 |
| GPU 1 | 8-12h | ResNet-50+BN+LAMB (TinyIN) s42 | P2-3 |
| GPU 1 | 12-16h | ResNet-50+LN+LAMB (TinyIN) s42 | P2-3 |
| GPU 1 | 16-20h | ResNet-50+BN+SignSGD (TinyIN) s42 | P3-3 |
| GPU 1 | 20-24h | ResNet-50+LN+SignSGD (TinyIN) s42 | P3-3 |
| GPU 2 | 0-5h | 迁移学习验证：finetune SGD+BN→AdamW | P2-2 |
| GPU 2 | 5-9h | Per-layer 梯度频域分解 run 1 | P2-1 |
| GPU 2 | 9-13h | Per-layer 梯度频域分解 run 2 | P2-1 |
| GPU 2 | 13-17h | ResNet-50 AdamW lr-gradient 修复 lr=0.0005 | P2-4 |
| GPU 2 | 17-21h | ResNet-50 AdamW lr-gradient 修复 lr=0.001 | P2-4 |
| GPU 2 | 21-24h | 空闲 / 重跑 | — |
| GPU 3 | 0-5h | Cue-conflict 纹理偏置评估（利用 IN100 checkpoints） | P0-3 |
| GPU 3 | 5-10h | Swin-T+BN+AdamW (TinyIN) s42 | P3-1 |
| GPU 3 | 10-15h | Swin-T+LN+AdamW (TinyIN) s42 | P3-1 |
| GPU 3 | 15-20h | ResNet-50 AdamW lr-gradient 修复 lr=0.002 | P2-4 |
| GPU 3 | 20-24h | ResNet-50 AdamW lr-gradient 修复 lr=0.003 | P2-4 |

**CPU 并行工作：**
- 对 Day 1 完成的 checkpoint 批量运行 spectral extraction
- Toy model 完善（或回退到数值模拟）
- 开始撰写新增内容的论文文本
- 判断方案 C 是否成功（**关键决策点**）

**Day 2 产出目标**：
- P0-1 多 seed 补种全部完成
- P0-3 纹理偏置验证完成
- 方案 C vs B 的决策做出
- ~24 additional runs 完成

---

#### Day 3（5 月 6 日，截稿日）— 收尾分析 + 论文写作

**GPU 调度（处理剩余和重跑）：**

| GPU | 任务 | 优先级 |
|-----|------|--------|
| GPU 0-1 | 任何 Day 1-2 失败/需重跑的 runs | P0-P1 |
| GPU 2 | Full ImageNet ResNet-50+SGD BN/LN（如有 96GB 卡） | P3-2 |
| GPU 3 | DeiT-S corruption robustness 扩展 | P3-4 |

**CPU 核心工作（全力写作）：**
1. 更新 Table 1（加入新 seed 的结果，重算 bootstrap CI）
2. 新建 Table/Figure：纹理偏置结果
3. 新建 Table/Figure：GroupNorm、RMSNorm 结果
4. 新建 Table/Figure：VGG-16 结果
5. 更新 sign test（m 值增加）
6. 写入 toy model 分析（方案 C）或下游因果验证（方案 B）
7. 更新 Discussion 和 Conclusion
8. 修复所有引用问题

---

#### Day 4（5 月 7 日）— 后续完善（如有延期空间）

| 任务 | 优先级 |
|------|--------|
| Full ImageNet runs 完成 + 分析 | P3-2 |
| 论文整体 polish | — |
| 补充任何 reviewer 可能质疑的额外消融 | — |
| 更新 NeurIPS checklist | — |

---

### 2.6 实验完成后的新论文结构预览

**如果方案 C 成功：**

```
§1 Introduction — 增加机制贡献点
§2 Related Work — 补充并发工作 + Yunis 2024
§3 Method — 不变
  §3.5 (新) Theoretical Analysis: Toy Model
§4 Experiments — 更新 Table 1 + 增加 GroupNorm/VGG 结果
  §4.3 (新) Downstream Causal Validation: Texture Bias
§5 Discussion — 大幅加强机制讨论
§6 Conclusion — 升级
```

**如果方案 C 失败，回退到方案 B：**

```
§1 Introduction — 增加下游因果验证贡献点
§2 Related Work — 补充并发工作
§3 Method — 不变
§4 Experiments — 更新 Table 1 + 增加新架构/归一化结果
  §4.3 (新) Downstream Causal Validation: Texture Bias
  §4.4 (新) Gradient Frequency Decomposition (初步机制证据)
§5 Discussion — 加强，但不含理论分析
§6 Conclusion — 中等升级
```

---

### 2.7 GPU 资源供需精算

#### 供给侧

| 资源配置 | 可用 GPU-hours |
|---------|--------------|
| 4 卡 × 48h（2 天） | **192 GPU-hours** |
| 4 卡 × 72h（3 天） | **288 GPU-hours** |
| 3 卡 × 72h（3 天） | 216 GPU-hours |

#### 需求侧（精确计算每个实验的 GPU-hours）

基于 Table 4 的实测训练时间（Tiny ImageNet, 200 epochs, 单卡）：

| 优先级 | 实验 | runs 数 | 单 run 时间 | GPU-hours | 累计 |
|--------|------|---------|-----------|-----------|------|
| **P0-1** | DeiT-S+AdamW 补种 (TinyIN) | 4 | 4.5h | 18 | 18 |
| **P0-1** | EfficientNet-B0+SGD 补种 (TinyIN) | 4 | 3h | 12 | 30 |
| **P0-1** | EfficientNet-B0+AdamW 补种 (TinyIN) | 2 | 3h | 6 | 36 |
| **P0-1** | VMamba-S+SGD 补种 (TinyIN) | 2 | 5.5h | 11 | 47 |
| **P0-1** | DeiT-S+AdamW 补种 (IN100) | 2 | 7h | 14 | 61 |
| **P0-3** | IN100 LN 补训（如需） | 2 | 5h | 10 | 71 |
| **P0-3** | Cue-conflict 评估 | — | <1h | 1 | 72 |
| | | | **P0 小计** | **72** | |
| **P1-1** | ResNet-50+GN (SGD+AdamW, 2 seeds) | 4 | 3.5h | 14 | 86 |
| **P1-2** | DeiT-S+RMSNorm (AdamW, 2 seeds) | 2 | 4.5h | 9 | 95 |
| **P1-3** | VGG-16 全因子 (BN/LN × SGD/AdamW × 2 seeds) | 8 | 4h | 32 | 127 |
| | | | **P1 小计** | **55** | |
| **P2-1** | 梯度频域分解 (gradient logging) | 2 | 4h | 8 | 135 |
| **P2-2** | 迁移学习验证 | 1 | 5h | 5 | 140 |
| **P2-3** | LAMB 优化器 | 2 | 4h | 8 | 148 |
| **P2-4** | AdamW lr-gradient 修复 (4 lr × BN/LN) | 8 | 3.5h | 28 | 176 |
| | | | **P2 小计** | **49** | |
| **P3-1** | Swin-T (AdamW, 2 seeds) | 2 | 5h | 10 | 186 |
| **P3-3** | SignSGD | 2 | 4h | 8 | 194 |
| | | | **P3 小计** | **18** | |
| | | | **总计** | **194** | |

#### 供需对比结论

```
当前方案总需求：                194 GPU-hours
4 卡 × 48h（2 天）可用：       192 GPU-hours  ← 刚好覆盖 P0+P1+P2+P3
4 卡 × 72h（3 天）可用：       288 GPU-hours  ← 剩余 94 GPU-hours
```

**结论：4 卡 2 天刚好能跑完全部方案，3 天有 94h 富余。**

#### 富余 GPU-hours 的追加实验（饱和利用）

3 天方案下有 ~94 GPU-hours 空闲，建议追加以下实验：

| 追加实验 | runs | GPU-hours | 价值 |
|---------|------|-----------|------|
| **ResNet-18** BN/LN × SGD/AdamW (TinyIN, 2 seeds) | 8 | ~16h | 测试小模型是否保持 pattern |
| **ConvNeXt-T+SGD with warmup** BN/LN (TinyIN) | 2 | ~8h | 尝试让 ConvNeXt 在 SGD 下不崩溃 |
| **ResNet-50+GN on IN100** (SGD+AdamW) | 2 | ~10h | GroupNorm 在高分辨率上的验证 |
| **VGG-16 on IN100** (SGD BN/LN) | 2 | ~10h | VGG 在标准分辨率上的验证 |
| **DeiT-S+GN+AdamW** (TinyIN) | 2 | ~9h | GroupNorm 在 ViT 上的效果 |
| **ResNet-50+BN+Adagrad** (TinyIN) | 2 | ~7h | 第四种优化器 |
| **重跑/失败恢复缓冲** | — | ~15h | 约 10% 的失败率缓冲 |
| **Spectral extraction 批量运行** | — | ~5h | 所有新 checkpoint 的频谱提取 |
| | | **追加小计 ~80h** | |

**追加后总需求：274 GPU-hours**，在 3 天 4 卡（288h）内完成，利用率 **95%**。

#### 最终饱和方案汇总

| 类别 | runs 数 | GPU-hours | 占比 |
|------|---------|-----------|------|
| P0（必须） | 16 | 72h | 26% |
| P1（高优先级） | 14 | 55h | 20% |
| P2（中优先级） | 13 | 49h | 18% |
| P3（低优先级） | 4 | 18h | 7% |
| 追加饱和实验 | 18 | 65h | 24% |
| 缓冲/频谱提取 | — | 20h | 7% |
| **总计** | **~65 runs** | **~279h** | **97%** |

**结论：3 天 4 卡，可跑 ~65 training runs，GPU 利用率 97%，完全饱和。**

---

### 2.8 预期收益总结

| 优化项 | 对论文评分的预期影响 | 回应的审稿质疑 |
|--------|---------------------|-------------|
| P0-1 多 seed | ↑ Soundness +1 | "single-seed results unreliable" |
| P0-2 Toy Model | ↑ Novelty +1, Significance +1 | "no mechanistic explanation" |
| P0-3 纹理偏置验证 | ↑ Significance +1.5 | "so what? no downstream impact" |
| P1-1 GroupNorm | ↑ Completeness +0.5 | "only BN vs LN tested" |
| P1-2 RMSNorm | ↑ Relevance +0.5 | "not covering modern normalizations" |
| P1-3 VGG-16 | ↑ Soundness +0.5 | "SGD positive bias only on ResNet" |
| P2-1 梯度频域 | ↑ Depth +0.5 | "mechanism is correlational" |
| P2-2 迁移学习 | ↑ Practical value +0.5 | "no actionable predictions verified" |
| 引用修复 | ↑ Rigor +0.5 | "citation errors" |

**方案 C 成功后预期中稿率**：NeurIPS/ICML **40-55%**  
**方案 B 回退后预期中稿率**：NeurIPS/ICML **35-45%**  
**仅做 P0（最低要求）后预期中稿率**：NeurIPS/ICML **30-40%**

---

### 2.8 Idea 层面的优化建议

#### 当前 idea 的核心定位调整

**从**：
> "我们发现归一化层的频谱效应取决于优化器"（纯现象描述）

**升级为（方案 C 成功时）**：
> "我们发现归一化层是隐式频谱滤波器，其滤波特性由有效更新幅度塑造；我们给出了一个简化理论模型解释这一机制，并实验验证了该效应直接导致了模型的纹理/形状偏好差异"

**或升级为（方案 B 回退时）**：
> "我们发现归一化层是隐式频谱滤波器，其滤波特性由有效更新幅度塑造；我们验证了该效应直接影响下游的纹理偏置和鲁棒性特性，为优化器-归一化联合设计提供了频域视角的实证指导"

#### 新增的 takeaway message（面向实践者）

在结论中加入具体的实践建议：
1. **优化器-归一化匹配建议**：需要高频敏感性（如医学影像边缘检测）→ SGD+BN；需要鲁棒性 → AdamW+LN
2. **预训练模型选择**：标准 ImageNet 预训练用 SGD+BN，已内嵌高频偏置；下游微调切换 AdamW 可自然衰减
3. **频域分析规范**：报告 Δ_{H/L} 或类似频域指标时，必须声明优化器和学习率

这些建议将论文从"学术发现"延伸到"实践指导"，显著提升 Significance 评分。
