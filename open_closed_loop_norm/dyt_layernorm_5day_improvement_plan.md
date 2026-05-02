# DyT 替换 LayerNorm 论文 5 天提升方案

## 0. 执行摘要

当前论文已经有一个有价值的问题：**Dynamic Tanh（DyT）是否可以作为 LayerNorm 的通用替代品，跨视觉架构直接使用？** 现有实验也观察到了清晰现象：DyT 在 DeiT-S/T 这类标准 global self-attention 架构上提升明显，在 Vim-T 上有收益但不稳定，在 RWKV-T、CaiT、Swin-T、ConvNeXt-T、RMT-T、Sequencer2D-S 等架构上下降或 collapse。

当前最大问题不是实验完全不够，而是论文主线容易被审稿人认为过强：如果主张是“architecture topology predicts DyT effectiveness”，那么 CaiT、Swin、ConvNeXt、RMT 这些 open-loop 或非 recurrent 反例会削弱这个说法；Vim-T 作为 closed-loop 却有提升，RWKV-T 作为 hybrid 又可以通过 α tuning 接近 LN，也会让 topology taxonomy 显得不够稳定。

因此，建议将论文从 **topology-centric narrative** 升级为 **normalization functional role-centric narrative**：

> DyT 不能通用替代 LayerNorm，因为同样叫 LayerNorm，它在不同架构和不同位置承担的功能角色不同。DyT 比较适合替代普通 residual stream 中的 activation stabilizer，但当 normalization 承担 state-dynamics stabilizer、architectural invariant enforcer、hierarchical feature-scale controller 等角色时，直接 full replacement 会变得脆弱或失败。

短期 5 天内，不建议盲目扩展到大量新模型、ImageNet-1K、COCO/ADE20K 或 NLP/LM。最现实、最高 ROI 的做法是：围绕现有模型，补强 **role-aware selective replacement、RMSNorm baseline、α sensitivity、关键 ablation 多 seed、机制诊断**，把论文从“多架构现象报告”提升为“LayerNorm 功能角色决定 DyT 可替代性的系统诊断研究”。

---

## 1. 当前论文的主要问题

### 1.1 当前论文已经有的优点

当前论文的优点包括：

1. **问题重要且自然**  
   DyT 原始工作主要验证标准 Transformer / ViT / LLM 等 global self-attention 架构，检查它能否迁移到 SSM、RWKV、hierarchical Transformer、CNN-like 视觉架构，是一个自然且有价值的问题。

2. **实验现象丰富**  
   论文不是只报告 DyT 好或不好，而是得到了一组有区分度的结果：DeiT-S/T 提升、Vim-T 部分提升、PlainMamba-T 近似持平、RWKV-T 下降、CaiT/Swin/ConvNeXt/RMT/Sequencer 明显失败。

3. **已有一些有价值的机制线索**  
   RWKV-T 的 key_norm decomposition、α sensitivity、effective rank、saturation fraction、MLP ablation 等分析都指向一个事实：LayerNorm 在不同架构中的作用并不相同。

4. **负结果有实践价值**  
   如果社区把 DyT 当成 universal drop-in replacement，这篇论文可以提醒大家：不能看到 LayerNorm 就全部替换。

### 1.2 当前论文的主要风险

当前论文的风险主要来自 claim 和 evidence 之间的错位。

#### 风险 1：Topology claim 太强

如果论文主线是：

> DyT effectiveness is shaped or predicted by information-flow topology.

那么审稿人会立刻看到一些反例：

- DeiT 是 open-loop，DyT 有效；
- CaiT、Swin、ConvNeXt、RMT 也不是普通 closed-loop SSM，但 DyT 失败；
- Vim 是 closed-loop，但 DyT 在 CIFAR-100 上有效；
- RWKV 是 hybrid，但 α=1.0 时 full DyT 可接近 LN。

因此，topology 只能作为 organizing lens，不能作为单独的 causal explanation。

#### 风险 2：机制解释偏 post-hoc

论文中 depth-width、MLP compensation、α saturation、rank collapse、key_norm functional role 等解释都很有启发，但目前很多证据仍然是相关性或跨模型比较。审稿人可能会问：

> 这些解释是事后归纳，还是经过 intervention 验证的机制？

#### 风险 3：关键 ablation seed 数不足

一些最支持机制 claim 的实验目前是 single-seed，例如 Vim-T ±MLP、PlainMamba-T +MLP、RMT/Sequencer collapse 等。由于 Vim-T 本身在 Tiny-ImageNet 上出现 seed-dependent sign reversal，所以 single-seed mechanism claim 很容易被质疑。

#### 风险 4：α tuning 可能改变部分结论

RWKV-T 在默认 α=0.5 下下降，但 α=1.0 时接近 LN。这会让审稿人质疑：

> 你们观察到的失败是否只是 α 没调好？

因此，α sensitivity 不能只放在附录，而应该作为主线的一部分。

#### 风险 5：低分辨率 setting 的外推性不足

主实验集中在 CIFAR-100 32×32 和 Tiny-ImageNet resized to 32×32，而 ImageNet-100 224×224 上 DyT 大幅失败。这个结果如果处理不好，会被视为论文主结论的弱点；如果处理得好，可以变成一个贡献：**DyT has a resolution-dependent failure mode**。

---

## 2. 建议的新主线：从 Topology-aware 到 Role-aware

### 2.1 新核心观点

建议将论文核心观点改为：

> Dynamic Tanh is not a universal LayerNorm substitute. It works when LayerNorm primarily acts as an activation stabilizer, but becomes fragile or harmful when LayerNorm plays state-dynamics, architectural-invariant, or hierarchical feature-scale roles.

中文表述：

> DyT 不是 LayerNorm 的通用替代品。它比较适合替代普通 residual stream 里的激活尺度稳定功能；但当 LayerNorm 承担状态动力学稳定、结构性不变量约束、跨 stage 特征尺度控制等功能时，直接替换会变得脆弱甚至失败。

### 2.2 新概念框架：Normalization Functional Roles

建议将 LayerNorm 的功能角色分成四类：

| Functional role | 含义 | 代表模型/位置 | DyT 兼容性 | 主要风险 |
|---|---|---|---|---|
| Activation stabilizer | 稳定 residual stream 的激活尺度 | DeiT / ViT pre-norm | 高 | α 过大时 tanh saturation |
| State-dynamics stabilizer | 稳定 SSM / recurrent state update 的输入和状态传播 | Vim / Mamba state-adjacent norms | 中等，α 敏感 | 状态递推放大尺度误差 |
| Architectural invariant enforcer | 维持结构性约束，如 key magnitude | RWKV key_norm / QK norm-like sites | 低 | 破坏 attention/key scale 机制 |
| Hierarchical feature-scale controller | 管理多 stage / 多分辨率特征尺度 | ConvNeXt LN2d、Swin stage/downsample norms | 低 | 跨 stage 尺度漂移、rank collapse |

### 2.3 新主线如何解释现有结果

#### DeiT-S/T：DyT 成功案例

DeiT 中的 LayerNorm 主要是 standard residual pre-norm activation stabilizer。DyT 的 tanh 压缩可以在一定范围内替代这种尺度稳定作用，并可能带来 regularization，因此 DeiT-S/T 获得明显提升。

#### Vim-T：DyT 有效但脆弱

Vim-T 里的 norm 不只是 activation stabilizer，还靠近 SSM state update。DyT 可以部分替代，但对 α 和初始化非常敏感，稳定窗口窄。因此 CIFAR-100 上有收益，但 Tiny-ImageNet 上出现 seed-dependent sign reversal。

#### PlainMamba-T：接近中性或无收益

PlainMamba-T 缺少 MLP compensation，并且 scan / state update 机制不同。DyT 的有界压缩难以被其他通道补偿，因此收益不明显。

#### RWKV-T：functional norm mismatch

RWKV 的 key_norm 不是普通 activation norm，而是 architecture-specific invariant，用于控制 key magnitude。DyT 作为 pointwise tanh 不能等价替代这个约束，因此 full replacement 下降。保留 key_norm 后性能恢复一部分，正好支持 functional role 框架。

#### CaiT / Swin / ConvNeXt / RMT：边界和失败模式

这些模型不是简单的 DeiT-like global self-attention。CaiT 的深窄 attention stack 容易积累 tanh compression；Swin 和 ConvNeXt 的 normalization 与 hierarchical stage / feature-scale control 强绑定；RMT / Sequencer 的 retention 或 recurrent 机制对 bounded pointwise replacement 更敏感。因此这些失败不是 topology 叙事的例外，而是 role-aware 框架下的边界证据。

---

## 3. 新论文贡献点建议

建议将贡献点重写为：

1. **Functional role perspective for normalization replacement**  
   We show that LayerNorm modules with the same mathematical form can play different architectural roles, including activation stabilization, recurrent-state stabilization, architectural invariant enforcement, and hierarchical feature-scale control.

2. **Cross-architecture evidence that DyT is not a universal replacement**  
   Across standard self-attention, SSM, RWKV-like hybrid, hierarchical Transformer, and CNN-like architectures, DyT succeeds, becomes fragile, or fails depending on the normalization role.

3. **Role-aware selective replacement**  
   We propose and evaluate a simple role-aware replacement principle: replace generic residual pre-norm activation stabilizers, but preserve state-critical, key/query-critical, and stage-transition normalization sites.

4. **Mechanistic diagnostics and α-stability landscape**  
   Through α sweeps, key_norm decomposition, state/activation norm analysis, saturation ratio, and effective-rank diagnostics, we identify why full DyT replacement fails in specific architectures.

---

## 4. 新标题建议

不建议继续使用过强的 topology-centric 标题。推荐以下标题之一：

1. **When Can Dynamic Tanh Replace LayerNorm? A Role-Aware Study Across Vision Architectures**
2. **Beyond Drop-in Replacement: Functional Roles of LayerNorm Shape Dynamic Tanh Effectiveness**
3. **Dynamic Tanh Is Not a Universal LayerNorm Substitute: Evidence from State-Space, Hybrid, and Hierarchical Vision Models**
4. **The Functional Role of Normalization Determines Whether Dynamic Tanh Works**

最推荐第 1 个或第 2 个。它们表达了问题意识，又避免过度承诺 topology 可以 predict 一切。

---

## 5. 5 天内最优实验策略

### 5.1 基本原则

在 5 天 + 2–4 张 RTX 5090 的约束下，实验目标不是做成终极 benchmark，而是补上最容易被审稿人攻击的短板：

1. full DyT replacement 没有正向方案；
2. 缺 RMSNorm 等强 baseline；
3. α tuning 可能改变结论；
4. 关键 mechanism ablation seed 数不足；
5. 机制分析仍偏 post-hoc；
6. 低分辨率结果外推性不足。

因此，短期最应该做的是：

> Role-aware selective replacement + RMSNorm baseline + α sensitivity + key mechanism ablation 3 seeds + targeted diagnostics.

不建议短期主攻：

- ImageNet-1K 完整训练；
- COCO / ADE20K；
- 10–20 个新模型；
- NLP / LM 扩展；
- fused DyT kernel；
- 10 seeds 全矩阵。

---

## 6. 短期实验优先级

## S1. RWKV-T role-aware selective replacement

### 优先级

最高。

### 目的

把 RWKV 从“DyT 在 hybrid 上失败”的个例，升级成“architectural invariant norm 不能被 pointwise DyT 替代”的核心证据。

### 实验配置

建议跑以下配置：

| 配置 | 目的 |
|---|---|
| Full LN | 原始 baseline |
| Full DyT α=0.5 | 默认 full replacement |
| Full DyT best α | 回应 α tuning 质疑 |
| DyT preserving key_norm | 已有 key_norm 证据，建议补齐 seeds |
| Role-aware DyT preserving key_norm + other functional norms | 验证 role-aware replacement |
| RMSNorm full replacement | 强 baseline |

### Seed 数

- 最低：3 seeds；
- 理想：5 seeds；
- 如果时间紧，优先保证 Full LN / Full DyT / key_norm-preserved / role-aware DyT / RMSNorm 的 3 seeds。

### 预期论文结论

如果 role-aware DyT 明显优于 full DyT，可以写：

> Preserving architecture-functional normalization sites recovers a substantial fraction of the full-DyT degradation, showing that DyT should not be applied by module name alone.

如果 role-aware DyT 仍然不如 LN，也仍然有价值：

> The remaining gap indicates that RWKV’s incompatibility is not solely due to key_norm, but preserving functional norms consistently improves robustness over blanket replacement.

---

## S2. Vim-T α sensitivity + selective replacement

### 优先级

最高。

### 目的

修复 closed-loop 结论不稳的问题，把 Vim-T 从“closed-loop 也能提升”的单点结果，改写成“state-dynamics normalization makes DyT fragile and α-sensitive”。

### 实验配置

| 配置 | 目的 |
|---|---|
| LN | baseline |
| Full DyT α=0.25 | 小 α，检查 near-linear regime |
| Full DyT α=0.5 | 默认 DyT |
| Full DyT α=1.0 | 检查 saturation / collapse |
| Role-aware DyT preserving state-adjacent norms | 验证 state-critical norms |
| DyT only non-state / MLP-side norms | 检查哪些位置可替 |
| RMSNorm | 强 baseline |

### Seed 数

- CIFAR-100：3 seeds；
- Tiny-ImageNet：如果时间充裕，只补 LN / Full DyT / Role-aware DyT / RMSNorm 的 3 seeds。

### 可选但很有价值

如果代码支持，补 Vim-T ±MLP 的 3 seeds：

| 配置 | 目的 |
|---|---|
| Vim-T full + LN/DyT | 对照 |
| Vim-T -MLP + LN/DyT | 验证 MLP compensation |

### 预期论文结论

> In SSM-style architectures, DyT can sometimes replace LayerNorm, but the replacement is narrowly conditioned on α and on whether state-critical normalization sites are preserved.

---

## S3. DeiT-S/T RMSNorm baseline + α robustness

### 优先级

很高。

### 目的

DeiT 是 DyT 成功案例。需要证明 DyT 不只是比 LayerNorm 好，还要和 RMSNorm 这种强替代 baseline 对比。

### 实验配置

对 DeiT-S 和 DeiT-T 跑：

| 配置 | 目的 |
|---|---|
| LN | baseline |
| Full DyT α=0.25 | 小 α |
| Full DyT α=0.5 | 默认 DyT |
| Full DyT α=1.0 | saturation risk |
| RMSNorm | 强 baseline |

### Seed 数

3 seeds 即可。

### 预期论文结论

> Standard pre-norm global self-attention is the cleanest case where DyT can replace activation-stabilizing LayerNorm. Its α-stability range is wider than that of state-dynamics architectures.

---

## S4. ConvNeXt-T 或 Swin-T selective replacement

### 优先级

中高。

### 目的

证明 open-loop 不是充分条件，hierarchical / stage-related normalization 不能 blanket replacement。

### ConvNeXt-T 配置

| 配置 | 目的 |
|---|---|
| Full LN | baseline |
| Full DyT | 复现大幅失败 |
| DyT shallow stages only | 检查浅层是否可替 |
| DyT within-stage only | 检查 stage 内替换 |
| Preserve downsampling/stage-transition norms | 验证 hierarchical feature-scale role |
| RMSNorm | 可选 baseline |

### Swin-T 配置

| 配置 | 目的 |
|---|---|
| Full LN | baseline |
| Full DyT | 复现失败 |
| Preserve patch-merging / stage-boundary norms | 验证 stage transition role |
| Block-internal DyT only | 检查普通 block norm 可替性 |
| RMSNorm | 可选 baseline |

### Seed 数

- 最低：1 seed diagnostic；
- 更好：3 seeds。

### 预期论文结论

> Hierarchical architectures contain normalization sites that control feature-scale transitions across stages. Full DyT replacement disrupts these roles, while preserving stage-critical norms can partially recover the gap.

---

## S5. 机制诊断分析

### 优先级

高。

### 目的

让 functional role taxonomy 不是纯叙事，而有机制证据支撑。

### 必做诊断 1：RWKV key magnitude / WKV scale

分析：

- key vector norm distribution；
- WKV attention weights / logits scale；
- key_norm preserved vs replaced 的差异；
- Full DyT 与 role-aware DyT 的 key magnitude drift。

支持结论：

> key_norm enforces an architectural invariant that pointwise tanh does not preserve.

### 必做诊断 2：Vim state norm / state update input norm

分析：

- SSM state input norm；
- hidden state norm across depth/tokens；
- 不同 α 下 state norm 是否发散或塌缩；
- role-aware preserving state-adjacent norm 是否更稳定。

支持结论：

> State-dynamics stabilizing norms are more α-sensitive than ordinary activation stabilizers.

### 必做诊断 3：DyT saturation ratio / effective rank

分析：

- tanh saturation fraction；
- per-layer effective rank；
- α 与 saturation / rank / accuracy 的关系；
- 成功模型和失败模型的对比。

支持结论：

> DyT failure often coincides with tanh saturation and representation compression, especially in deep/narrow or hierarchical settings.

---

## S6. 可选实验：ViT-Tiny 或 resolution diagnostic

### ViT-Tiny

如果 timm pipeline 已经成熟，可以加一个低风险模型：

| 配置 | 目的 |
|---|---|
| LN | baseline |
| Full DyT | 检查 DeiT 是否个例 |
| RMSNorm | baseline |

3 seeds，CIFAR-100 即可。

### Resolution diagnostic

如果 ImageNet-100 或 resize pipeline 已经 ready，可以对 DeiT-S/T 做：

| Resolution | 配置 |
|---|---|
| 32×32 | LN / DyT |
| 64×64 | LN / DyT |
| 128×128 | LN / DyT |
| 224×224 | LN / DyT |

记录：

- DyT - LN accuracy；
- saturation ratio；
- effective rank；
- activation norm。

预期结论：

> DyT benefit decreases as resolution increases, suggesting a resolution-dependent saturation failure mode.

---

## 7. 5 天执行计划

### Day 0：重构叙事与实验代码

#### 写作任务

1. 重写 abstract skeleton；
2. 重写 introduction skeleton；
3. 新增 “Normalization Functional Roles” 小节；
4. 新增 “Role-aware Replacement Protocol” 小节；
5. 设计新主表模板。

#### 代码任务

1. 支持按 norm site selective replacement；
2. 支持 RMSNorm 替换；
3. 支持 α grid；
4. 支持自动记录 saturation ratio / activation norm；
5. 支持自动导出 CSV 和图表。

---

### Day 1：启动最高优先级实验

#### GPU 1：RWKV-T selective replacement

- Full LN；
- Full DyT α=0.5；
- Full DyT best α；
- DyT preserving key_norm；
- Role-aware DyT；
- RMSNorm。

#### GPU 2：Vim-T α sensitivity

- LN；
- DyT α=0.25/0.5/1.0；
- Role-aware DyT；
- RMSNorm。

#### GPU 3：DeiT-S/T RMSNorm + α robustness

- DeiT-S：LN / DyT α=0.25/0.5/1.0 / RMSNorm；
- DeiT-T：LN / DyT α=0.25/0.5/1.0 / RMSNorm。

#### GPU 4：ConvNeXt/Swin selective or Vim ±MLP

优先级：

1. Vim ±MLP 3 seeds；
2. ConvNeXt selective replacement；
3. Swin selective replacement。

---

### Day 2：继续多 seed + 开始机制分析

#### 训练任务

1. 补齐 RWKV-T 关键配置 3 seeds；
2. 补齐 Vim-T 关键配置 3 seeds；
3. 补齐 DeiT RMSNorm / α 的 3 seeds；
4. 若有余力，开始 ConvNeXt/Swin selective。

#### 分析任务

1. 提取 RWKV key magnitude；
2. 提取 Vim state norm；
3. 提取 DyT saturation ratio；
4. 初步生成 alpha stability plot。

---

### Day 3：补强边界实验和诊断图

#### 实验任务

1. ConvNeXt-T selective replacement；
2. Swin-T selective replacement，如果 ConvNeXt 已完成；
3. ViT-Tiny 或 resolution diagnostic，可选；
4. 对 collapse 或异常结果重跑。

#### 写作任务

1. 写 Results 1：Cross-architecture outcomes；
2. 写 Results 2：Role-aware selective replacement；
3. 写 Results 3：α-stability landscape；
4. 写 Results 4：Mechanistic diagnostics。

---

### Day 4：论文重写日

#### 主文重构

建议主文结构改为：

1. Introduction；
2. Related Work；
3. Functional Roles of Normalization；
4. Role-aware DyT Replacement Protocol；
5. Experimental Setup；
6. Cross-Architecture Results；
7. Role-aware Selective Replacement；
8. α-stability and Failure Modes；
9. Mechanistic Diagnostics；
10. Limitations；
11. Conclusion。

#### 图表整理

主文建议保留 4–5 张关键图表：

1. 主结果表：LN / RMSNorm / Full DyT / Role-aware DyT；
2. Functional role taxonomy table；
3. Selective replacement recovery figure；
4. α stability landscape；
5. Mechanism figure：key norm / state norm / saturation。

---

### Day 5：压力测试和最终润色

#### Reviewer attack list

让 agent 模拟 NeurIPS reviewer，重点攻击：

1. topology claim 是否过强；
2. role taxonomy 是否 post-hoc；
3. selective replacement 是否 cherry-pick；
4. α tuning 是否解释了所有结果；
5. RMSNorm baseline 是否充分；
6. low-resolution setting 是否限制结论；
7. seed 数是否足够；
8. 是否存在 best test leakage。

#### Claim-evidence audit

逐条检查论文中的强 claim 是否有表/图支持：

| Claim | 需要支持的证据 |
|---|---|
| DyT works for activation stabilizer role | DeiT + RMSNorm + α robustness |
| State-dynamics norms are α-sensitive | Vim α sweep + state norm |
| key_norm is architectural invariant | RWKV key_norm / role-aware ablation |
| hierarchical stage norms should be preserved | ConvNeXt/Swin selective |
| full replacement is inferior to role-aware replacement | selective replacement table |
| saturation/rank compression explains failures | saturation + effective rank |

没有充分证据的句子，全部改成：

- “suggests”；
- “is consistent with”；
- “provides evidence that”；
- “within the tested settings”。

---

## 8. 新主表和新图表设计

### Table 1：Main results with stronger baselines

| Model | Role class | LN | RMSNorm | Full DyT | Role-aware DyT | Best α DyT | Δ Full DyT | Δ Role-aware |
|---|---|---:|---:|---:|---:|---:|---:|---:|
| DeiT-S | Activation stabilizer | | | | | | | |
| DeiT-T | Activation stabilizer | | | | | | | |
| Vim-T | State-dynamics stabilizer | | | | | | | |
| RWKV-T | Architectural invariant | | | | | | | |
| ConvNeXt-T | Hierarchical scale controller | | | | | | | |

### Table 2：Functional role taxonomy

| Functional role | Norm sites | Representative models | Full DyT behavior | Role-aware recommendation |
|---|---|---|---|---|
| Activation stabilizer | residual pre-norm | DeiT/ViT | often works | replace |
| State-dynamics stabilizer | SSM/state-adjacent norms | Vim/Mamba | fragile | tune α or preserve |
| Architectural invariant | key_norm/QK-like norms | RWKV | harmful | preserve |
| Hierarchical scale controller | stage/downsample norms | Swin/ConvNeXt | harmful | preserve stage-critical norms |

### Figure 1：Outcome map

展示不同模型的 DyT - LN accuracy，颜色表示 functional role，形状表示 full DyT / role-aware DyT。

### Figure 2：Selective replacement recovery

对 RWKV、Vim、ConvNeXt/Swin：

```text
LN baseline
Full DyT
Role-aware DyT
RMSNorm
```

用 grouped bar 展示 recovery。

### Figure 3：α stability landscape

x-axis：α  
y-axis：accuracy 或 DyT-LN delta  
颜色：模型  
标记 collapse 区间。

### Figure 4：Mechanism diagnostics

建议分三个 panel：

1. RWKV key norm distribution；
2. Vim state norm across layers/tokens；
3. DyT saturation ratio vs effective rank / accuracy。

---

## 9. 写作层面的具体修改

### 9.1 Abstract 修改方向

不要堆太多数值。建议结构：

1. DyT 背景；
2. 现有 gap：是否跨架构通用未知；
3. 本文 thesis：LayerNorm 的功能角色决定可替代性；
4. 主要结果：DeiT 成功，SSM 脆弱，RWKV/hierarchical 失败；
5. 方法：role-aware selective replacement；
6. takeaway：不要 blanket replacement。

### 9.2 Introduction 修改方向

Introduction 需要更清楚地区分：

- module name：都叫 LayerNorm；
- functional role：实际功能不同；
- replacement risk：DyT 只替代 bounded activation rescaling，不能替代所有 norm 功能。

建议 introduction 的核心段落：

> A normalization module is not defined solely by its formula. In modern architectures, the same LayerNorm operator can stabilize residual activations, regulate recurrent state dynamics, enforce key/query magnitude constraints, or align feature scales across resolution stages. A pointwise bounded nonlinearity such as DyT can plausibly replace the first role, but not necessarily the others.

### 9.3 Results 修改方向

Results 不要按模型流水账写，而要按 role/failure mode 写：

1. Activation stabilizer：DeiT success；
2. State-dynamics stabilizer：Vim fragility；
3. Architectural invariant：RWKV key_norm；
4. Hierarchical feature-scale controller：ConvNeXt/Swin failure；
5. α stability；
6. role-aware replacement。

### 9.4 Discussion 修改方向

Discussion 要主动弱化 claim：

不要写：

> topology predicts DyT effectiveness.

改成：

> topology provides useful cues for identifying normalization roles, but the replacement outcome is determined by the functional role of each normalization site and its interaction with α, depth, width, and resolution.

### 9.5 Limitations 修改方向

Limitations 要诚实但不自毁：

1. role taxonomy 是 empirical framework，不是 formal theory；
2. selective replacement policy 仍需更多架构验证；
3. high-resolution ImageNet-scale training 仍未充分覆盖；
4. 部分 auxiliary probes 仍是 single-seed；
5. α tuning 与 recipe sensitivity 说明 DyT 不是 plug-and-play。

但要强调：

> These limitations do not undermine the main conclusion that blanket LayerNorm-to-DyT replacement is unsafe; rather, they motivate the role-aware replacement principle.

---

## 10. 哪些建议不要短期做

### 10.1 不建议短期做 ImageNet-1K 完整训练

原因：

- 2–4 张 5090，5 天内很难完成多模型、多 baseline、多 seed；
- 即使跑出单 seed，也未必能说服审稿人；
- 可能拖垮主线写作。

长期可以作为 rebuttal / camera-ready / 后续版本。

### 10.2 不建议短期做 COCO/ADE20K

原因：

- 接入和训练成本高；
- 容易引入新的 recipe confound；
- 与当前 DyT/LN 替换主线距离较远。

### 10.3 不建议短期接大量新模型

原因：

- 当前已有 10 个模型，问题不是绝对模型数，而是证据组织和机制验证；
- 新模型容易引入工程风险；
- 5 天内最重要的是把现有结果变成 role-aware story。

### 10.4 不建议短期做 fused DyT kernel

原因：

- 当前吞吐实验显示 DyT 不一定比 LN 快；
- 效率不是这篇论文最稳的主线；
- kernel 优化会转移注意力。

### 10.5 不建议短期做 NLP/LM 扩展

原因：

- 会让论文从视觉架构分析扩散到另一个领域；
- 需要新的数据、模型、训练 recipe；
- 5 天内风险大于收益。

---

## 11. 最小可接受完成标准

如果时间非常紧，至少完成以下内容：

1. **RWKV-T role-aware selective replacement 3 seeds**；
2. **Vim-T α sensitivity 3 seeds**；
3. **DeiT-S/T RMSNorm baseline 3 seeds**；
4. **主文从 topology 改为 functional role narrative**；
5. **α sensitivity 从附录移到主文**；
6. **机制分析至少包含 RWKV key magnitude、Vim state norm、DyT saturation ratio**；
7. **主表包含 LN / RMSNorm / Full DyT / Role-aware DyT**；
8. **limitations 中主动承认 role taxonomy 是 empirical framework，不是 formal theory**。

完成这些后，论文质量会明显高于当前版本。

---

## 12. 理想完成标准

如果 4 张 GPU 运行顺利，建议额外完成：

1. ConvNeXt-T selective replacement 3 seeds；
2. Swin-T selective replacement 1–3 seeds；
3. Vim-T ±MLP 3 seeds；
4. PlainMamba-T +MLP 3 seeds；
5. ViT-Tiny 3 seeds；
6. DeiT-S/T resolution diagnostic 1 seed；
7. validation-selected checkpoint sanity check；
8. final epoch accuracy 对照。

这些会进一步提升 soundness 和 presentation。

---

## 13. 预期提升效果

吸纳该方案后，论文会从：

> DyT 在不同 topology 上表现不同。

提升为：

> LayerNorm 的功能角色决定 DyT 是否可替代。Full replacement 会失败，而 role-aware selective replacement 更可靠。

具体提升包括：

| 维度 | 当前版本 | 吸纳后版本 |
|---|---|---|
| 主线 | topology-dependent effectiveness | role-dependent compatibility |
| 贡献 | 多架构经验观察 | normalization replacement failure-mode diagnosis |
| 方法 | full DyT replacement | role-aware selective replacement |
| 反例处理 | topology 叙事的例外 | functional role 边界证据 |
| α tuning | 潜在漏洞 | 主结果之一：α-stability landscape |
| RWKV key_norm | 局部 ablation | architectural invariant 的核心证据 |
| 实践价值 | per-architecture validation | 判断哪些 norm 能换、哪些不能换 |
| NeurIPS 风险 | overclaim topology | claim 更稳、evidence 更匹配 |

---

## 14. 最终结论

这份提升方案的核心不是“多跑实验”，而是重新组织论文的科学问题：

> 从“DyT 在哪些 topology 上有效”转向“DyT 能替代 LayerNorm 的哪些功能角色”。

这个转变能显著提升论文质量，因为：

1. 它避免 topology 过强 claim 被反例打穿；
2. 它把 CaiT、Swin、ConvNeXt、RWKV 等负结果变成支持边界的证据；
3. 它让 RWKV key_norm ablation 成为核心贡献；
4. 它自然引出 role-aware selective replacement 这个正向方法；
5. 它把 α sensitivity 从漏洞变成稳定性 landscape；
6. 它给实践者明确建议：不要按模块名替换 LayerNorm，而要按功能角色替换。

最终建议提交版本的主张应当保持克制：

> Within the tested vision architectures and low-resolution settings, DyT is effective when replacing activation-stabilizing normalization sites, but becomes fragile or harmful for state-critical, invariant-enforcing, or hierarchical scale-controlling normalization sites. Therefore, DyT should be applied as a role-aware selective replacement rather than a blanket LayerNorm substitute.

这会比当前版本更稳、更深，也更像 NeurIPS Main Track 的 empirical diagnosis paper。
