# 优化方案 v2：Correlation Sweeps Reveal That Architectural Robustness to Spurious Correlations Is Dataset-Dependent

**目标会议**：NeurIPS 2026  
**截稿日期**：2026-06-06  
**当前日期**：2026-05-04（可用 ~32 天，密集执行窗口 3.5–4 天）  
**当前评估分数**：~6.5–7.0/10  
**目标分数**：8.0–8.5/10  
**可用资源**：2–4 × RTX 5090 (32 GB) + 1 × RTX PRO 6000 (96 GB)，24/7 Agent Auto Research 模式

---

## 第一部分：已有实验完整审计（含 Appendix A–P 精确盘点）

### 1.1 主实验（已完成）

| 实验组 | 详情 | 状态 |
|---|---|---|
| **Waterbirds 主评估** | 12 架构 × 10 个 r levels {0.70–0.999} × S∈{3,5} seeds | ✅ 完整（~500 runs） |
| **Spawrious 跨数据集** | 11 架构（排除 Vim-S）× 3 levels {0.70, 0.90, 0.95} × 3 seeds | ✅ 完整（99 runs） |
| **CelebA 主文** | DeiT-S + Swin-T，完整 3×3；ConvMixer 4/9 runs | ⚠️ ConvMixer 待完成 |
| **ConvNeXt 2×2 factorial** | {V1,V2}×{IN-1k,IN-22k}，r=0.95 | ✅ 完整（~20 runs） |
| **统计检验** | Mann-Whitney U, Cohen's d, Cliff's δ, bootstrap rank stability, permutation test, Bonferroni | ✅ 完整 |

### 1.2 Appendix 实验精确审计

| Appendix | 内容 | 状态 | 关键数据 |
|---|---|---|---|
| **A** | CelebA boundary condition（tipping-point + class-imbalance confound + extended coverage Table 2） | ⚠️ Table 2 多架构 in-progress | MambaOut-T 0 runs; PoolFormer/Twins/PVTv2 各 1 run; FocalNet/Vim-S/DeiT-B 未纳入 |
| **B** | Background Attention Ratio (BAR) | ✅ 完整 | 9 架构，11,788 张测试图，**负面结果**：A+ BAR 反而更高 |
| **C** | Attention Windowing Ablation | ✅ 完整 | DeiT-S 3 窗口大小，3 seeds。局部化降低 WGA，否证 locality 假说 |
| **D** | Architecture Details | ✅ 完整 | 12+4 架构的 timm ID、参数量、FLOPs、预训练 |
| **E** | DFR Ablation | ✅ 完整 | 12 架构，2 seeds (42,43)，r=0.95。Gap 缩小到 3.9–4.8pp |
| **F** | Grad-CAM 定性观察 | ✅ 完整 | illustrative，BAR 已否定系统性 |
| **G** | CKA 表征相似度 | ✅ 完整 | **负面结果**：tier label 不预测 CKA。但 within-A+ CKA(0.726) > within-A-(0.549) |
| **H** | Linear Probe | ✅ 完整 | 12 架构，seed 42。A+ 同时更好编码 spurious 和 core features |
| **I** | GroupDRO | ⚠️ 只有 4 架构 | PVTv2, MambaOut, PoolFormer, DeiT-S；3 seeds × 3 LR。Gap 5.2→2.4pp |
| **J** | Spawrious 扩展分析 | ✅ 完整 | 曲线形状、跨数据集分歧、解释因素、r=0.99 pilot(n=1)、caveats |
| **K** | Spawrious Multi-Seed 分析 | ✅ 完整 | 数据集选择理由、per-seed WGA/degradation、DeiT-S outlier 分析、ConvMixer 方差分析 |
| **L** | 补充表格与程序 | ✅ 完整 | Correlation grid、bootstrap 细节、ConvNeXt 2×2、sensitivity analysis、LR validation |
| **M** | 统计 vs 实际显著性 | ✅ 完整 | Cohen's d=3.44、regime dependence、73% strict bootstrap 讨论 |
| **N** | AUC 与 τ_p 灵敏度 | ✅ 完整 | Normalized AUC 完全 rank separation；τ_p 三阶段分析 |
| **O** | ConvNeXt Factorial | ✅ 完整 | V1+IN-1k 退化；GRN + IN-22k 各自可补偿。回归分析 ISOTROPIC×GLOBAL 显著 |
| **P** | Learning Rate 控制 | ✅ 完整 | 3 arch × 3 LR × 3 corr = 27 runs。DeiT-S 在所有 LR 下都更脆弱 |

### 1.3 关键缺口总结

| 缺口 | 审稿人攻击点 | 影响 |
|---|---|---|
| **Dataset 与 spurious modality 绑定** | "差异来自 dataset artifact 还是 cue modality？" | **最关键** |
| CelebA Table 2 多架构不完整 | "three-dataset comparison 的第三条腿站不稳" | 高 |
| Spawrious 只有 3 个 r levels | 无法画 degradation curve / AUC | 高 |
| GroupDRO 只有 4/12 架构 | "gap narrows but doesn't vanish" 证据不充分 | 中高 |
| Twins-SVT-S 只在 r=0.95 评估 | 从 AUC 分析中排除 | 中 |
| r=0.99 只是 pilot (n=1) | "极端区域怎么样？" | 中 |
| 12 架构 → n 偏小 | 统计检验功效受限 | 中 |
| 只有 ERM 训练（+ GroupDRO/DFR post-hoc） | "换训练方法还有这个 gap 吗？" | 中 |

---

## 第二部分：核心诊断与论文当前分数评估

### 2.1 论文的真正强项（优化方案不应忽视）

1. **Tipping-point methodology** 本身是方法论贡献，清晰、可复现、有新意
2. **Architecture-dataset interaction** 是最原创的发现——同样的架构在不同 spurious modality 下排名完全不同
3. **Narrower invariant**（isotropic global mixing vulnerability）跨三个数据集都成立，是最稳的结论
4. **Robustness analysis 远超平均水平**：LOO、permutation test、Bonferroni、bootstrap、GroupDRO、DFR、LR control、scale ablation、attention windowing——共 5 个 mechanistic probes + 多重统计验证
5. **诚实的负面结果报告**：BAR 反向、CKA 不显著、windowing 降低 WGA 都完整报告

### 2.2 论文最大的方法论软肋

**Dataset 和 spurious modality 完全混淆。**

| Dataset | Foreground | Spurious cue 位置 | Cue 类型 | 图像来源 | 类别数 | Minority n |
|---|---|---|---|---|---|---|
| Waterbirds | 自然鸟图 | 背景（空间分离） | 自然场景 | CUB+Places 合成 | 2 | 642 |
| Spawrious | AI 生成狗图 | 背景（上下文） | AI 生成场景 | 全合成 | 2 (subset) | ≥1000 |
| CelebA | 人脸 | 前景共址 | 面部属性 | 自然照片 | 2 | 180 |

审稿人可以质疑：观察到的 architecture ranking 变化到底来自：
- spurious cue 的视觉形态（这是论文想说的）
- 还是 dataset size / 图像生成方式 / 类别难度 / foreground segmentation 质量 / pretraining alignment / 完全不同的 foreground domain？

**这三个 dataset 之间差异太多，无法归因到 modality 这一个因素。**

### 2.3 当前分数与瓶颈

| 维度 | 当前得分 | 上限原因 |
|---|---|---|
| Novelty/Idea | 7.5 | Methodology + interaction finding 有新意 |
| Empirical evidence | 6.0 | dataset-modality confound 是硬伤 |
| Statistical rigor | 7.0 | 已有大量检验，但 n=12 是先天限制 |
| Writing/presentation | 7.0 | 清晰，但叙事略过强 |
| Significance/impact | 6.5 | 纯诊断型，无 actionable guideline |
| **Overall** | **~6.5–7.0** | **empirical evidence 和 confound 是瓶颈** |

---

## 第三部分：优化策略

### 3.0 核心思路

三条主线并行推进：

1. **拆开 dataset-modality confound**：Waterbirds-Cue-Modality Factorial（★★★★★ 全方案最高优先级）
2. **加厚证据层**：完善 CelebA/Spawrious 覆盖、增加架构、扩大 GroupDRO
3. **强化叙事**：从 "dataset-dependent" 升级到 "cue-modality-dependent"，把 narrower invariant 推到 C 位

---

## 第四部分：详细实验方案

### P0：投稿前必须修复（Day 0，~4 GPU-hours + 文字工作）

#### P0.1 CelebA Table 2 收尾

Table 2 中多架构数据不完整。需要：

| 架构 | 当前状态 | 需补 runs | GPU-hours |
|---|---|---|---|
| MambaOut-T | 0 runs | 3 corr × 3 seeds = 9 | ~3h |
| PoolFormer-S24 | r=0.70 1 run | 8 | ~3h |
| PVTv2-B2 | r=0.70 1 run | 8 | ~3h |
| Twins-SVT-S | r=0.70 1-2 runs | 7 | ~2.5h |
| ConvMixer (ext.) | 0 runs | 9 | ~3h |
| ConvMixer 主文 | 4/9 runs | 5 | ~2h |
| **小计** | | **~46** | **~16h** |

可选追加（建议做）：FocalNet-T, DeiT-B, Vim-S 各 3×3=9 runs → +27 runs, ~10h

**CelebA 总计：~73 runs, ~26 GPU-hours**

#### P0.2 Twins-SVT-S 全 Waterbirds sweep

当前只在 r=0.95 评估。补齐 9 个 r levels × 3 seeds = 27 runs, ~7 GPU-hours。  
完成后可纳入 AUC 分析。

#### P0.3 格式/语言检查（文字工作）

- 确认 NeurIPS 2026 submission 格式（匿名、line numbers、Type 3 font）
- 确认所有 "pending" / "in progress" 语言已更新
- Checklist code access: "will be released upon acceptance"

---

### P1-S：最高优先级——Waterbirds Cue-Modality Factorial（★★★★★）

> **这是全方案中 ROI 最高的单一实验。它直接拆开 dataset-modality confound，将论文主张从可质疑的跨数据集观察升级为控制变量的因果推断。**

#### P1-S.1 实验设计

**核心思想**：在同一个 base dataset (Waterbirds/CUB) 上，保持 foreground、label、train/test split、group-balanced test 不变，仅改变 spurious cue 的视觉形态。

**四种 Cue Modality**：

| Modality | 缩写 | Cue 位置 | Cue 性质 | 与现有实验的关系 |
|---|---|---|---|---|
| **Natural Background** | NAT | 背景（空间分离） | 自然场景（Places） | = 现有 Waterbirds 主实验 |
| **Texture Background** | TEX | 背景（空间分离） | 随机纹理 / Fourier noise | 测试 A+ 优势是否需要自然场景 |
| **Patch/Border Cue** | PAT | 图像角落 / 边框 | 小色块或纹理块 | 低语义、局部 shortcut |
| **Foreground-Co-Located** | COL | 鸟体内部（用 CUB mask） | 颜色 / 纹理 overlay | 模拟 CelebA 的空间共址，无 class-imbalance confound |

**Correlation strength sweep**：r ∈ {0.70, 0.90, 0.95}（3 levels，与 Spawrious 对齐）。  
如果时间允许，追加 r=0.80 和 r=0.99。

**Architecture selection**：6 个代表性架构，覆盖 taxonomy 四个 cell：

| 架构 | Tier | P1 | P2 | 选择理由 |
|---|---|---|---|---|
| Swin-T | A+ | ✓ | ✓ | hierarchical local attention，Waterbirds 表现稳定 |
| PVTv2-B2 | A+ | ✓ | ✓ | hierarchical global attention，Waterbirds 最强 |
| PoolFormer-S24 | A- | ✗ | ✓ | P2-only，A- 中最高 |
| ResNet-50 | A- | ✗ | ✓ | 经典 CNN baseline |
| DeiT-S | A- | ✓ | ✗ | isotropic global attention，Spawrious outlier |
| ResMLP-24 | A- | ✗ | ✗ | isotropic global MLP，一直最弱 |

**最小实验量**：6 arch × 3 新 modalities × 3 r levels × 3 seeds = **162 runs**  
（NAT modality 已有数据，直接复用。）

**扩展实验量**（推荐）：补 2 架构 (ConvMixer + MambaOut-T) + 2 个 r levels → 8 arch × 3 新 mod × 5 r × 3 seeds = **360 runs**

#### P1-S.2 数据构造细节

**TEX (Texture Background)**：
- 用 CUB segmentation masks 提取鸟前景
- 生成两类纹理背景（如 Perlin noise 或 DTD 纹理子集）
- 按 correlation r 控制 bird_class → texture_class 的对齐概率
- 保持 224×224 分辨率，与原 Waterbirds 一致

**PAT (Patch/Border Cue)**：
- 保留原 Waterbirds 的随机（或中性）背景
- 在图像固定位置（如左下角 32×32 区域）贴一个彩色块
- 两类色块与 label 按 r 对齐
- 不修改 foreground 或背景其余部分

**COL (Foreground-Co-Located Cue)**：
- 保留原 Waterbirds 的随机背景
- 用 CUB segmentation mask，在鸟体区域内叠加轻微色调偏移（如 hue shift ±15°）或透明纹理 overlay
- 两类 overlay 与 label 按 r 对齐
- overlay 强度需要 pilot 测试确保可学习但不过于明显

**关键约束**：
- 所有 modality 共享相同的 train/test split、foreground images、label 定义
- group-balanced test set 结构一致
- 只有 cue 的视觉形态和位置不同

#### P1-S.3 GPU 成本估算

| 配置 | Runs | GPU-hours |
|---|---|---|
| 最小版：6 arch × 3 mod × 3 r × 3 seeds | 162 | ~40h |
| 推荐版：8 arch × 3 mod × 5 r × 3 seeds | 360 | ~90h |
| 数据构造代码开发 | — | ~6h 人工（Day 0） |
| Pilot 测试 (COL overlay 强度) | ~12 | ~3h |
| **总计（推荐版）** | **~372** | **~93h** |

#### P1-S.4 预期结果与应对

| 结果 | 可能性 | 论文处理方式 |
|---|---|---|
| NAT 有 tier，TEX 有弱 tier，PAT/COL tier collapse | **最可能** | 主结论升级为 cue-modality-dependent，isotropic global 跨 modality 脆弱 |
| 所有 modality 都有 tier（P1∧P2 普遍有效） | 中 | P1∧P2 比预期更通用；Spawrious collapse 归因于 AI-generated images 或任务太容易 |
| 只有 NAT 有 tier，其余全 collapse | 低 | Waterbirds tier 是 natural-habitat-specific；但 methodology 仍有价值，强化"需要 per-domain 诊断" |
| DeiT-S/ResMLP 在所有 modality 下都偏弱 | **极可能** | 进一步支撑 narrower invariant |

#### P1-S.5 新增图表

**Figure（主文）：Architecture × Cue Modality heatmap**
- 行 = architecture，列 = cue modality
- 颜色 = WGA@r=0.95 或 ΔWGA(r=0.70→0.95)
- 一眼看出 ranking 是否随 cue 变化

**Figure（主文）：A+ advantage across cue modalities**
- 横轴 = cue modality（NAT → TEX → PAT → COL）
- 纵轴 = mean(A+) - mean(A-)
- 如果从左到右递减 → 论文核心图

**Table（Appendix）：Interaction regression**
- `degradation ~ P1 * P2 * cue_modality + isotropic_global + seed`
- 报告 P1∧P2 × cue_modality 交互项是否显著

---

### P1-A：高优先级实验（Day 1–2）

#### P1-A.1 Spawrious 扩展到更多 correlation levels（~70 GPU-hours）

当前只有 {0.70, 0.90, 0.95}，无法画 degradation curve 或计算 AUC。

**补充 levels**：{0.75, 0.80, 0.85, 0.93}（4 个新 level）

| 配置 | Runs | GPU-hours |
|---|---|---|
| 11 arch × 4 new levels × 3 seeds | 132 | ~55h |
| 追加 r=0.97 (if time) | 33 | ~14h |
| **总计** | **~165** | **~69h** |

**价值**：
- 可以为 Spawrious 画完整 degradation curve
- 可以计算 Spawrious AUC，与 Waterbirds AUC 直接比较
- 可以看 isotropic global mixing outlier 在什么 r 开始分离

#### P1-A.2 新增 2–3 个架构到主评估（~60 GPU-hours）

从 12 扩到 14–15 个架构，增强统计功效，测试 taxonomy 边界。

**推荐新增**：

| 架构 | timm ID | P1 | P2 | Params | 选择理由 |
|---|---|---|---|---|---|
| **ConvNeXt-T V1** | `convnext_tiny.fb_in1k` | ✗ | ✓ | 28.6M | 已有 factorial 数据，P2-only，现代 CNN |
| **CAFormer-S18** | `caformer_s18.sail_in1k` | ✓ | ✓ | ~26M | MetaFormer 家族（已引用），conv+attention 混合 → A+ |
| **EfficientNetV2-S** | `tf_efficientnetv2_s.in1k` | ✗* | ✓ | ~22M | SE 是 channel-wise 非 spatial → P2-only |

*SE blocks 是 input-dependent 但作用于 channel 维度，不是 spatial token mixing → 不满足 P1。这本身是一个有意义的边界讨论。

**实验量**：
- Waterbirds: 3 arch × 10 levels × mixed seeds ≈ 96 runs (~24h)
- Spawrious: 3 arch × 7 levels × 3 seeds = 63 runs (~26h)  
- CelebA: 3 arch × 3 levels × 3 seeds = 27 runs (~10h)
- **总计：~186 runs, ~60 GPU-hours**

**更新后的 taxonomy**：
- A+ (P1∧P2): PVTv2, FocalNet-T, MambaOut-T, Swin-T, Twins-SVT-S, **CAFormer-S18** → n₁=6
- A- (¬P1∨¬P2): PoolFormer, DeiT-B, ResNet-50, Vim-S, ConvMixer, DeiT-S, ResMLP, **ConvNeXt-T**, **EfficientNetV2-S** → n₂=9
- Mann-Whitney 6-vs-9 比 5-vs-7 更有功效

#### P1-A.3 扩展 GroupDRO 到全部/大部分架构（~30 GPU-hours）

当前只有 4/12 架构。"Gap narrows but doesn't vanish" 是重要的 claim，但证据不足。

**方案**：补充 8 个架构（若含新增则 11 个），每个 3 LR × 3 seeds：

| 配置 | Runs | GPU-hours |
|---|---|---|
| 8 missing arch × 3 LR × 3 seeds (LR search) | 72 | ~18h |
| 选定 best LR 后的正式评估 | 已包含在上面 | — |
| 新增 3 arch × 3 LR × 3 seeds | 27 | ~7h |
| **总计** | **~99** | **~25h** |

**主文报告**：full 12–15 arch GroupDRO table，群体均值 gap + per-architecture 排名与 ERM 的 rank correlation。

#### P1-A.4 r=0.99 完整实验（~15 GPU-hours）

Appendix J.6 pilot (n=1) 显示 cluster 结构在 r=0.99 溶解。扩展到正式实验：

| 配置 | Runs | GPU-hours |
|---|---|---|
| 12 arch × 4 additional seeds at r=0.99 | 48 | ~12h |
| r=0.999 追加 2 seeds（当前 n=3 for most） | 24 | ~6h |
| **总计** | **~72** | **~18h** |

**价值**：填充 Appendix J.6 pilot → 正式结果，展示极端 correlation 下的架构行为。

---

### P1-B：中优先级实验（Day 2–3）

#### P1-B.1 JTT (Just-Train-Twice) 训练算法（~25 GPU-hours）

论文只用 ERM 训练。GroupDRO 和 DFR 是 post-hoc 方法。JTT 是训练时介入但不需要 group labels 的代表性方法。

**JTT 流程**：
1. Stage 1：标准 ERM 训练 T₁ epochs
2. 识别训练集中被错误预测的样本
3. Stage 2：对错误样本 upweight λ 倍后重新训练

**实验量**：12 arch × r=0.95 × 3 seeds × ~2× 训练时间 = ~72 effective runs → ~25 GPU-hours

**如果 A+/A- gap 在 JTT 下仍存在**：证明架构差异不是 ERM 特有的 artifact。  
**如果 gap 消失**：说明 JTT 的 error-based reweighting 可以补偿架构差异，但这本身也是有意义的发现。

#### P1-B.2 DFR 扩展（+3 seeds）（~10 GPU-hours）

当前 DFR (Appendix E) 只有 2 seeds，seed 42 显著 (p=0.0025) 但 seed 43 不显著 (p=0.075)。补 3 个 seeds (44, 45, 46)：

12 arch × 3 new seeds × CPU-heavy (feature extraction + logistic regression) → ~10 GPU-hours for feature extraction

#### P1-B.3 多层 Linear Probe（CPU-dominated）

当前 Appendix H 只在 penultimate layer 做 probe。扩展到每个 stage/block 的输出：

- 对 hierarchical 架构：每个 stage 输出
- 对 isotropic 架构：每隔 3-4 层

**目的**：找到 A+ 和 A- 的 representation 在哪一层开始分化。这可能提供 mechanistic insight，补充现有 5 个 negative probes。

**成本**：feature extraction ~5 GPU-hours；probe training 全 CPU。

---

### P1-C：CPU 并行任务（0 GPU-hours）

#### P1-C.1 Cue-Modality Factorial 的统计分析框架

预先准备好分析代码：
- Architecture × Cue Modality × r 的 three-way interaction analysis
- `degradation ~ P1 * P2 * cue_modality + isotropic_global`
- Per-modality Mann-Whitney U test
- Cross-modality rank correlation (Kendall's τ)

#### P1-C.2 Spawrious AUC 计算

一旦 P1-A.1 数据到位，立即计算：
- Per-architecture normalized AUC
- A+ vs A- AUC comparison
- 与 Waterbirds AUC (Appendix N) 直接比较

#### P1-C.3 更新统计检验（含新增架构）

- Mann-Whitney U with n₁=6 vs n₂=9
- Updated bootstrap rank stability
- Updated permutation test
- Bonferroni correction for expanded partition

#### P1-C.4 Spawrious OLS 回归扩展

Appendix O.1 已有 ISOTROPIC × GLOBAL regression。  
扩展：加入 cue_modality 作为 moderator（如果 Cue-Modality Factorial 完成）。

---

### P2：扩展实验（Day 3–3.5，时间允许时执行）

| 编号 | 实验 | Runs | GPU-hours | 价值 |
|---|---|---|---|---|
| P2.1 | Cue-Modality Factorial 全量版（12+ arch × 4 mod × 5 r × 5 seeds） | ~900 | ~225 | 把最小版结论 generalize |
| P2.2 | Pre-training diversity（DINOv2/CLIP weights on 4 arch） | ~36 | ~15 | 测试 IN-1k 特异性 |
| P2.3 | CnC (Correct-n-Contrast) 训练算法 | ~72 | ~25 | 另一种 non-group-label 方法 |
| P2.4 | Scale ablation 扩展（PVTv2-B2 vs PVTv2-B5；DeiT-S vs DeiT-B 对比已有） | ~60 | ~20 | 测试规模效应 |
| P2.5 | SpuCo-Animals（如果可访问） | ~150 | ~60 | 第四数据集，自然图像+可控 correlation |
| P2.6 | Waterbirds r=0.99 Cue-Modality Factorial | ~48 | ~12 | 极端 regime 下的 cue-modality interaction |
| **总计** | | **~1266** | **~357** | |

---

## 第五部分：论文叙事优化

### 5.1 当前叙事（需要调整）

> "Waterbirds 发现 P1∧P2 tier → Spawrious tier collapse → CelebA gap within noise → architecture robustness is dataset-dependent"

**问题**：dataset-dependent 是描述性的，不是机制性的。审稿人会问 "dependent on what exactly?"

### 5.2 升级后叙事（配合 Cue-Modality Factorial）

> "Correlation sweeps reveal architecture-dependent degradation curves. A controlled cue-modality sweep on the same base dataset shows that the protective value of P1∧P2 depends on the **visual geometry of the spurious cue**: when the shortcut is a spatially extended, separable scene feature (natural backgrounds), hierarchy + input-dependent mixing provides a 4.6 pp advantage; this advantage diminishes when the shortcut becomes local (patch), synthetic (texture), or co-located with the target (foreground overlay). Cross-dataset validation (Spawrious, CelebA) confirms this modality dependence. However, one narrower structural vulnerability persists across **all** cue modalities and datasets: isotropic architectures with global token mixing and no locality bias (DeiT-S, DeiT-B, ResMLP-24) are consistently the most fragile."

### 5.3 修订后的 Contributions

1. **Tipping-point methodology**：correlation sweep 替代单点 WGA，揭示退化曲线和 tipping point
2. **Controlled cue-modality evidence**（★新增核心贡献）：在同一 dataset 上控制 spurious cue 的视觉形态，证明 architecture robustness 由 cue modality 调制，而非 dataset artifact
3. **Cross-dataset validation**：Spawrious + CelebA 外部验证 cue-modality dependence
4. **Narrower invariant**：isotropic global mixing vulnerability 跨所有 cue modality 和 dataset 都成立
5. **Negative mechanistic evidence**：5 个 probes 排除空间注意力等简单解释

### 5.4 需要弱化的 Claims

- "P1∧P2 separates tiers" → "P1∧P2 separates tiers under spatially separable background cues; the separation is cue-modality-dependent"
- "architecture robustness is dataset-dependent" → "architecture robustness is **cue-modality-dependent**: the same architecture can be robust to one visual form of shortcut and vulnerable to another"
- "the viable explanation lies in the classifier's decision boundary" → "the mechanism operates beyond spatial attention patterns or representational geometry; the classifier readout is a candidate for future investigation"

### 5.5 需要强化的 Claims

- **Narrower invariant** 推到 abstract 和 introduction 的核心位置
- **Cue-modality dependence** 作为主要发现（替代 dataset-dependence 的模糊表述）
- **Tipping-point methodology** 的诊断价值——不是每个 architecture 都需要全 sweep，但至少需要多于一个 r level

### 5.6 建议的论文结构调整

| Section | 内容 | 变化 |
|---|---|---|
| §1 Introduction | 问题 + 现有评估的局限 + 本文方法 | 微调：强调 cue-modality 而非 dataset |
| §2 Related Work | 不变 | — |
| §3 Method | Tipping-point methodology + P1/P2 定义 + Cue-Modality Factorial 设计 | **新增 3.3 Cue-Modality Factorial** |
| §4.1 Setup | 数据集、架构、训练 | 扩展：新增架构 + 4 cue modalities |
| §4.2 Waterbirds Tipping Points | 不变（+ 新增架构） | 更新统计数据 |
| §4.3 Sensitivity Analysis | 不变 | 更新统计数据 |
| §4.4 Necessity, not Sufficiency | 不变 | — |
| **§4.5 Cue-Modality Factorial** | **新增核心 section** | **新** |
| §4.6 Cross-Dataset Validation | Spawrious + CelebA（降级为 validation） | 重写：CelebA 为 boundary condition |
| §4.7 Pre-training Confound | ConvNeXt factorial | 不变 |
| §5 Discussion | 强化 narrower invariant + cue-modality narrative | 重写 |

---

## 第六部分：3.5 天饱和执行计划

### 6.1 资源预算

| 资源 | 配置 | 并行能力 |
|---|---|---|
| GPU 1–3 | RTX 5090 (32GB) × 3 | 每卡 1 个实验 |
| GPU 4 | RTX PRO 6000 (96GB) | 可并行 2 个 Waterbirds/CelebA 实验 |
| 等效 GPU slots | **4–5 并行** | |
| 3.5 天总预算 | **336–420 GPU-hours** | |

### 6.2 实验总量与预算

| 优先级 | 实验组 | Runs | GPU-hours | Day |
|---|---|---|---|---|
| P0.1 | CelebA Table 2 收尾 | 73 | 26 | Day 0 |
| P0.2 | Twins-SVT-S full sweep | 27 | 7 | Day 0 |
| **P1-S** | **Cue-Modality Factorial（推荐版）** | **372** | **93** | **Day 0.5–2** |
| P1-A.1 | Spawrious 扩展 levels | 165 | 69 | Day 1–2 |
| P1-A.2 | 新增 3 架构全评估 | 186 | 60 | Day 1–2 |
| P1-A.3 | GroupDRO 扩展全架构 | 99 | 25 | Day 2 |
| P1-A.4 | r=0.99/0.999 扩展 | 72 | 18 | Day 2 |
| P1-B.1 | JTT 训练算法 | 72 | 25 | Day 2–3 |
| P1-B.2 | DFR +3 seeds | — | 10 | Day 2 |
| P1-B.3 | 多层 Linear Probe | — | 5 | Day 2 |
| P2（选做） | 扩展实验 | ~300 | ~80 | Day 3–3.5 |
| **核心总计（P0+P1）** | | **~1066** | **~338** | |
| **含 P2 总计** | | **~1366** | **~418** | |

**结论**：4 GPU slots × 3.5 天 = 336–420 GPU-hours。P0 + P1 全部 = 338h，正好在预算内。P2 可在 Day 3.5 选做高优先级部分。

### 6.3 Day-by-Day 执行计划

---

#### Day 0（准备 + 立即修复 + 开始数据构造）

**代码开发（并行进行）**：

| 任务 | 预计时间 | 优先级 |
|---|---|---|
| Cue-Modality Factorial 数据构造脚本 | 4–5h | **最高** |
| ├─ TEX: CUB mask 提取 + 纹理背景生成 | 1.5h | |
| ├─ PAT: 角落色块注入 | 1h | |
| ├─ COL: foreground overlay (需 pilot) | 1.5h | |
| └─ 统一 data loader + correlation control | 1h | |
| 新增架构集成（ConvNeXt-T, CAFormer-S18, EfficientNetV2-S） | 1h | 高 |
| GroupDRO 脚本适配 | 1h | 高 |
| JTT 脚本实现 | 2h | 中 |

**GPU 立即启动**（用 Day 0 做 P0）：

| GPU | 任务 | Runs | 时间 |
|---|---|---|---|
| GPU 1 | CelebA 收尾 batch 1 (MambaOut-T + PoolFormer) | ~17 | ~6h |
| GPU 2 | CelebA 收尾 batch 2 (PVTv2 + Twins-SVT-S + ConvMixer) | ~22 | ~8h |
| GPU 3 | CelebA 收尾 batch 3 (FocalNet + DeiT-B + Vim-S) | ~27 | ~10h |
| GPU 4 (PRO) | Twins-SVT-S Waterbirds full sweep (27 runs) | 27 | ~7h |

**Day 0 晚间**（代码就绪后启动 pilot）：

| GPU | 任务 | Runs | 时间 |
|---|---|---|---|
| GPU 1 | COL overlay intensity pilot: Swin-T × 3 intensities × r=0.95 × 1 seed | 3 | ~1h |
| GPU 2 | TEX pilot: Swin-T × r=0.95 × 1 seed | 1 | ~0.3h |
| GPU 3 | PAT pilot: Swin-T × r=0.95 × 1 seed | 1 | ~0.3h |
| GPU 4 | CelebA 收尾 overflow | — | — |

**CPU 并行**：

| 任务 | 时间 |
|---|---|
| 准备 Cue-Modality 统计分析框架 | 2h |
| 新增架构 LR validation search 准备 | 1h |
| NeurIPS 格式检查 | 1h |

**Day 0 交付物**：
- ✅ CelebA Table 2 全部完成
- ✅ Twins-SVT-S full Waterbirds sweep
- ✅ Cue-Modality 数据构造 + pilot 完成 → **决策：COL overlay 强度确定**
- ✅ 代码准备完毕

---

#### Day 1（Cue-Modality Factorial 主战场 + 新增架构启动）

**GPU 分配（全程饱和）**：

| GPU | 任务 | Runs | 时间 |
|---|---|---|---|
| GPU 1 | Cue-Modality: TEX, 6 arch × 3 r × 3 seeds | 54 | ~14h → 接续 PAT 部分 |
| GPU 2 | Cue-Modality: PAT, 6 arch × 3 r × 3 seeds | 54 | ~14h → 接续 COL 部分 |
| GPU 3 | Cue-Modality: COL, 6 arch × 3 r × 3 seeds | 54 | ~14h → 接续扩展 (8 arch) |
| GPU 4 (PRO) | 新增架构: ConvNeXt-T + CAFormer + EfficientNetV2 Waterbirds LR search + main sweep | ~96 | ~24h |

**如果 Cue-Modality 6-arch × 3-mod × 3-r 在 14h 完成**（Day 1 下午即有初步结果）：

**Day 1 中午检查点**：
- TEX 首批结果 → A+ 优势还在吗？
- PAT 首批结果 → patch cue 下 tier 是否 collapse？
- **关键决策**：如果 3 modality 全部 collapse → 确认 modality-dependence；如果都保持 tier → 需要调整叙事

**Day 1 晚间**：Cue-Modality 6-arch 基础版完成，开始扩展到 8 arch + 更多 r levels

**CPU 并行**：

| 任务 | 时间 |
|---|---|
| 分析 Cue-Modality pilot/首批结果 | 2h |
| Spawrious AUC 代码准备 | 1h |
| 论文 §4.5 (Cue-Modality) 初稿 | 3h |
| 更新 CelebA Table 2 到论文 | 1h |

**Day 1 交付物**：
- ✅ Cue-Modality Factorial 6-arch 基础版完成 (162 runs)
- ✅ 新增架构 Waterbirds sweep 进行中
- ✅ 初步 Cue-Modality 结果分析
- ✅ 论文 §4.5 初稿

---

#### Day 2（Spawrious 扩展 + GroupDRO + r=0.99 + Cue-Modality 扩展）

**GPU 分配**：

| GPU | 任务 | Runs | 时间 |
|---|---|---|---|
| GPU 1 | Spawrious 4 new levels, batch 1 (6 arch × 4 levels × 3 seeds) | 72 | ~14h → 接续 r=0.99 expansion |
| GPU 2 | Spawrious batch 2 (5 arch × 4 levels × 3 seeds) | 60 | ~12h → 接续 GroupDRO batch 1 |
| GPU 3 | GroupDRO expansion (8 arch × 3 LR × 3 seeds) | 72 | ~18h → 接续 r=0.999 expansion |
| GPU 4 (PRO) | 新增架构 Spawrious + CelebA + Cue-Modality 扩展版 | ~140 | ~24h |

**CPU 并行**：

| 任务 | 时间 |
|---|---|
| Cue-Modality 全量统计分析（三路 interaction） | 4h |
| Spawrious AUC 计算（一旦数据到位） | 2h |
| 更新 Mann-Whitney / bootstrap / permutation（含新架构） | 3h |
| 论文叙事全面修订 | 4h |
| 新增 Figure 制作 | 3h |
| DFR feature extraction for +3 seeds | 2h |
| 多层 Linear Probe feature extraction | 3h |

**Day 2 交付物**：
- ✅ Spawrious 7-level 完整 degradation curves
- ✅ GroupDRO 12+ 架构完整结果
- ✅ r=0.99 完整实验
- ✅ 新增架构全评估基本完成
- ✅ Cue-Modality 扩展版进行中
- ✅ 论文叙事修订稿

---

#### Day 3（JTT + P2 扩展 + 论文精修）

**GPU 分配**：

| GPU | 任务 | Runs | 时间 |
|---|---|---|---|
| GPU 1 | JTT: 12 arch × r=0.95 × 3 seeds | ~72 | ~20h |
| GPU 2 | Cue-Modality 扩展版收尾 + r=0.80/0.99 补充 | ~100 | ~20h |
| GPU 3 | P2 优先项：Pre-training diversity (DINOv2/CLIP) | ~36 | ~12h → 接续 P2 |
| GPU 4 | P2 优先项：scale ablation / additional seed replications | ~60 | ~18h |

**CPU 并行**：

| 任务 | 时间 |
|---|---|
| JTT 结果分析 | 2h |
| 全部图表终稿 | 4h |
| Appendix 全面更新 | 4h |
| 论文 main text 精修 | 6h |

---

#### Day 3.5（收尾 + 定稿，半天）

**GPU**：补跑任何失败/异常 seeds，P2 收尾。

**写作**：
- Abstract 重写（突出 cue-modality-dependent + narrower invariant）
- Introduction 精修
- NeurIPS 格式终检
- Checklist 最终核对
- PDF 编译 + 匿名检查

---

## 第七部分：关键决策点与应急预案

| 时间 | 决策点 | 可能结果 | 应对 |
|---|---|---|---|
| Day 0 晚 | COL pilot | A) overlay 可学习 → 继续 | B) 太弱 → 调整强度 | C) 太强 → 降低强度 |
| Day 1 下午 | Cue-Modality 首批结果 | A) NAT 有 tier, 其他 collapse → 最理想，直接写 | B) 全部有 tier → 调整叙事为 "P1∧P2 more universal than expected" | C) NAT 也 collapse → 检查数据/代码 |
| Day 1 下午 | TEX 下 DeiT-S 表现 | A) 仍然最弱 → narrower invariant 强化 | B) 不再最弱 → 需要更谨慎讨论 |
| Day 2 上午 | Spawrious degradation curves | A) isotropic global 明确分离 → 支持 narrower invariant | B) 分离不清 → 降低 claim 强度 |
| Day 2 下午 | GroupDRO 全量结果 | A) Gap 缩小但不消失 → 强化 "architectural, not just training" | B) Gap 消失 → 调整讨论 |
| Day 3 | JTT 结果 | A) Gap 存在 → "not ERM-specific" | B) Gap 消失 → 有意义的 negative result，放 appendix |

---

## 第八部分：预期论文分数提升

| 版本 | 分数 | 关键变化 |
|---|---|---|
| 当前版本 | 6.5–7.0 | dataset-modality confound 未解决 |
| + P0（CelebA完整 + Twins full sweep） | 7.0 | 三数据集证据链完整 |
| + P1-S（Cue-Modality Factorial 6-arch） | **7.5–8.0** | **核心 confound 解决，叙事升级** |
| + P1-A（Spawrious 扩展 + 新增架构 + GroupDRO + r=0.99） | **8.0** | 证据厚度 + 统计功效 + 训练 robustness |
| + P1-B（JTT + DFR + multi-layer probe） | **8.0–8.5** | 全方位覆盖 |
| + P2（pre-training diversity + scale + SpuCo） | **8.5** | 锦上添花 |

**最关键的分数跳跃**：  
6.5–7.0 → **7.5–8.0**：Cue-Modality Factorial。  
这一个实验就能把论文从 "interesting empirical observation" 变成 "controlled evidence for a mechanistic claim"。

---

## 第九部分：学术规范注意事项

1. **Cue-Modality Factorial 的设计必须在数据构造前锁定**，不能看了结果再调 cue 类型。四种 modality 的定义和 overlay 参数在 pilot 后固定，后续不可修改。
2. **所有跑过的实验都要报告**。如果某个 cue modality 结果不符合预期，放 appendix 作为 honest result。
3. **新增架构的 P1/P2 分类必须事先确定**，不能看了结果再分。CAFormer → P1∧P2，ConvNeXt-T → P2-only，EfficientNetV2 → P2-only，这些分类基于架构定义，不基于实验结果。
4. **GroupDRO 和 JTT 不使用 test labels 做模型选择**。GroupDRO 用 group-balanced validation set 选 LR。JTT 的 upweight ratio λ 用 validation set 选。
5. **CelebA 仍然是 boundary condition**。即使 Table 2 完成，CelebA 的 class-imbalance confound 不变。不要因为数据更完整就把它升级为强证据。
6. **Negative results 继续完整报告**。如果 Cue-Modality Factorial 中某个 modality 没有差异，这本身也是重要发现。
7. **pre-registration**：虽然 Cue-Modality Factorial 是事后设计的，但应在论文中明确说明它是 confirmatory（验证 modality-dependence 假说），不是 exploratory。P1∧P2 分类仍然是 exploratory。

---

## 第十部分：最短时间优先级排序

如果时间极其紧张（只有 2 天），按以下顺序取舍：

| 排名 | 实验 | GPU-hours | 理由 | 不做的后果 |
|---|---|---|---|---|
| **#1** | P0 全部（CelebA + Twins） | 33 | 不修就有 incomplete data | 审稿信任风险 |
| **#2** | **P1-S Cue-Modality Factorial 最小版** | **43** | **解决核心 confound** | **论文最大软肋无法回应** |
| **#3** | P1-A.2 新增 2-3 架构 | 60 | 增强统计功效 | n=12 被质疑功效不足 |
| **#4** | P1-A.1 Spawrious 扩展 levels | 69 | 对称 degradation curves | Spawrious 比较不对等 |
| **#5** | P1-A.3 GroupDRO 扩展 | 25 | "gap narrows" claim 证据不足 | 只有 4/12 架构的 GroupDRO |
| **#6** | P1-A.4 r=0.99 完整实验 | 18 | 补充极端 regime | pilot 停留为 pilot |
| **#7** | P1-B.1 JTT | 25 | non-ERM validation | "only ERM" 质疑 |
| **#8** | P1-B.2 DFR +3 seeds | 10 | 统计显著性 | DFR 结论 unstable (1/2 seeds 显著) |
| **#9** | Cue-Modality 扩展版 | 50 | generalize 到更多架构/r levels | 最小版已足够 |
| **#10+** | P2 扩展 | ~80 | 锦上添花 | 分数上限从 8.5 降到 8.0 |

**最低可行方案（2 天）**：#1–#4 = 205 GPU-hours，3 张 5090 可完成。  
**推荐方案（3.5 天）**：#1–#8 = 283 GPU-hours，3 张 5090 + PRO 6000 可覆盖。  
**饱和方案（4 天）**：#1–#9 + P2 部分 = ~400 GPU-hours，4 张 5090 + PRO 6000 可覆盖。
