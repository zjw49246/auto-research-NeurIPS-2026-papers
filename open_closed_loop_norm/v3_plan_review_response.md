# 对 Agent v3 方案评估回复的逐条分析与最终建议

> 本文件分析执行 agent 对 v3 方案的反馈，识别其合理之处和关键误判，给出最终执行建议。

---

## 一、总体判断

Agent 的反馈在 **实验精简** 方面做了合理判断（~35 runs, ~20 GPU-hours），但在 **最关键的战略问题——叙事重构** 上过于保守。这个保守可能会浪费此次提升的最大杠杆点。

**核心分歧：** Agent 认为"论文经过 7 轮 review 叙事已稳定，用加法策略更安全"。但问题是：内部 review 的叙事稳定 ≠ 外部竞争力最优。如果叙事本身有结构性缺陷（纯 diagnostic，无 prescriptive contribution），稳定只是把缺陷打磨得更光滑——NeurIPS reviewer 不会因为文字通顺就忽略"Where's the method?"这个根本问题。

---

## 二、采纳清单评价：基本认同

Agent 的实验选择高效且聚焦，具体评价：

| Agent 提议 | 评价 | 补充 |
|---|---|---|
| Vim-T RMSNorm 补 2 seeds | 正确，优先级高 | 确认 RMSNorm ordering 方向 |
| PlainMamba-T RMSNorm ×3 | 正确 | 第二个 closed-loop 数据点 |
| Swin-T selective ×3 | 正确且重要 | 最大 deficit (-17.18) 的恢复实验 |
| RWKV-T selective (only residual pre-norm) ×3 | 正确 | 比 excl.key_norm 更精准的 selective 策略 |
| ConvNeXt-T selective ×3 | 正确 | hierarchical role 验证 |
| Vim-T selective ×3 | 正确 | closed-loop selective |
| CaiT depth threshold (k=6,12,18) | 正确 | 因果方向最强验证之一 |
| CaiT/Swin RMSNorm ×1 seed | 合理 | 单 seed probe 足够 |
| All selective CIFAR-100-C eval | 必做 | Pareto/safe-deployment 定位的基础 |
| A1 per-layer α correlation | 正确且关键 | 最强非循环证据 |
| A2 activation distribution | 正确 | 直观可视化 |
| A4 early-α diagnostic | 正确 | 实践工具 |

**实验总量 ~35 runs ≈ 20 GPU-hours，4 GPU 并行 5-6 小时**——这个估算合理，留出大量时间给分析和写作。**同意。**

**唯一补充：** ConvNeXt-T RMSNorm (×1-3 seeds) 也建议加入（Agent 只列了 CaiT/Swin 的 RMSNorm probe）。ConvNeXt 是唯一的 CNN 架构，RMSNorm 数据有助于确认 hierarchical role 的特殊性。新增 1-3 runs，成本极低。

---

## 三、未采纳清单：逐条反驳与最终建议

### 3.1 "Narrative 全面重构为 role-aware prescriptive" —— 不做

**Agent 理由：** 论文经过 7 轮 review 叙事已稳定，大改 framing 会引入内部矛盾，用"加法"策略更安全。

**反驳：**

这是 Agent 反馈中 **最严重的误判**。原因如下：

1. **7 轮内部 review 的稳定性不等于 NeurIPS 竞争力。** 内部 review 可能优化了文字连贯性和数据一致性，但如果审稿人的第一个问题是 "Where's the method contribution?"，再稳定的叙事也无法回答。当前论文的 NeurIPS rejection risk #1 是 "Interesting observations but no prescriptive takeaway"——这不是加 1.5 页 results 能解决的。

2. **"加法策略"解决不了结构性问题。** 如果 Introduction 仍然 frame 论文为 "we test DyT across architectures and find it's topology-dependent"，那新增的 selective replacement results 只是 appendix-level 补充，不会改变审稿人对论文定位的判断。审稿人读 abstract + intro 的前 2 分钟就决定了论文的 "mental category"——diagnostic study vs. method paper。

3. **"引入内部矛盾"的风险可控。** v3 方案明确说的是 **精准手术**（改 abstract ~3 句话 + intro ~2 段 + 新增 1 个 results section），不是全面重写。现有 Related Work、Training Setup、大部分 Results、所有 Appendix 都不需要改。总改动量 ≤ 2 页。

**最终建议：必须做叙事调整，但范围严格控制。**

具体改动清单（总量 ~1.5 页新增 + ~0.5 页修改）：

| 位置 | 改动 | 改动量 |
|---|---|---|
| **Abstract** | 替换最后 2 句话：从 "normalization replacement is not a drop-in operation" → 加入 selective replacement 和 corruption robustness 的核心发现 | ~3 句话 |
| **Introduction 倒数第 2 段** | 新增 1 段：preview selective replacement finding + corruption robustness | ~6 行 |
| **Contribution list** | 从 3 条 → 4 条：新增 "role-aware selective replacement achieves clean recovery + corruption benefit" | 1 条 |
| **新增 Results section 5.X** | Selective replacement results + Pareto/safe-deployment 分析 | ~0.5-1 页 |
| **新增 Discussion paragraph** | α 分化作为 data-driven replacement guidelines | ~0.3 页 |

**不改的部分：** Related Work、Method 3.1-3.3、Training Setup、现有 Results 4.1-4.6、现有 Discussion 5.1-5.5、大部分 Appendix。

这不是 "全面重构"，而是在现有稳定叙事上 **增加一个 prescriptive layer**——从 "我们发现问题" 到 "我们发现问题并给出解决方案"。

### 3.2 "Algorithm 1 box" —— 不做

**Agent 理由：** 本质是 if-else lookup table，NeurIPS 审稿人对贡献膨胀敏感。改为 Discussion 中文字呈现。

**部分认同，但建议保留简化版。**

Agent 的担忧有道理——一个 5 行 if-else 确实不应该作为 "Algorithm 1" 大写呈现。但完全去掉会失去一个实践工具的锚点。

**折中方案：** 在 Discussion 或 Results 中用一个 **精简的 decision table**（不是 Algorithm box），例如：

```
Table X: Recommended DyT replacement strategy by normalization site type.
Site type                      | Action          | Rationale
Residual pre-attn/pre-FFN      | Replace with DyT | Activation stabilizer; α stays low
SSM state-adjacent              | Probe first      | α-sensitive; monitor early-α
Key/Query functional norm       | Preserve as LN   | Structural invariant
Downsample/stage transition     | Preserve as LN   | Cross-scale feature management
```

这比 Algorithm box 更谦虚，但比纯文字更有可引用性和实践价值。

### 3.3 "Role taxonomy 作为核心 Contribution" —— 不做

**Agent 理由：** 循环定义风险——role 的判断标准 = norm 在计算图中的位置 = topology。审稿人会问 "how do you determine role without running the experiment?"

**反驳：这个担忧基于误解。**

Role taxonomy 的分类标准是 **计算图中的结构位置**（pre-attn, pre-FFN, key_norm, downsample），这在运行任何 DyT 实验之前就可以确定。你不需要跑实验就知道 RWKV 的 key_norm 是一个 functional norm——看代码就知道。这与 topology-based prediction（open-loop → DyT works）不同，后者需要跑实验才知道 topology 与结果的关系。

**但 Agent 的替代方案也有价值。** "让 α 数据本身 reveal 分化" 是一个互补而非替代的证据。A1 分析（per-layer α correlation）如果显示不同位置的 norm 学到了不同的 α 分布，这是 data-driven 的验证，比 top-down taxonomy 更有说服力。

**最终建议：不把 role taxonomy 作为独立 Contribution 条目，但作为 organizing framework 在 Results 中使用。**

具体做法：
- 不在 Contribution list 中单独列 "role taxonomy"
- 在 Results 的 selective replacement section 中，用 role 作为 **组织工具** 解释为什么某些 sites 可替换、某些不可
- A1 的 α 分化数据作为 **empirical evidence** 支撑这个组织框架
- 避免使用 "taxonomy" 这个词（暗示 top-down 分类），改用 "normalization site characterization" 或 "site-specific replacement guidelines"

这样既避免了循环定义的攻击面，又保留了 role-based 组织的解释力。

### 3.4 "E5 DyT-v2 at 224×224" —— 不做

**Agent 理由：** (1) 训练时间被低估 (~8-12h/run 而非 ~1h)；(2) tanh bounded range 是数学硬约束；(3) 失败确认已知结论。

**完全同意。** v3 已经把 E5 降为 P1，Agent 进一步砍掉是合理的。α-deadlock 分析（Appendix K）本身就是强贡献，不需要 fix 来支撑。

### 3.5 "E4 DeiT-S/T α 补 seeds (10 runs)" —— 不做

**基本同意。** DeiT 已有充分的 3-seed 数据（DeiT-S: 64.9±0.6, DeiT-T: 60.8±0.6），α sensitivity 是诊断性分析，方向清晰。**可以砍掉。**

**唯一保留条件：** 如果 Day 2 有空闲 GPU 且其他实验已完成，可以补 DeiT-T α=0.25/1.0 各 1 seed 作为 bonus data。但不作为 must-have。

### 3.6 "E14 核心实验 5-seed 扩展" —— 不做

**同意。** 3 seeds 满足 NeurIPS 要求，5-seed 是 nice-to-have。

### 3.7 "新标题 / Contribution list 重写" —— 不做

**部分不同意。**

标题可以保留（风险低但收益也低，可以最后决定）。但 **Contribution list 必须调整**——不是重写，是从 3 条变 4 条，新增 selective replacement 相关的贡献。如果做了 selective replacement 实验但 contribution list 里没有提及，审稿人会认为这只是 appendix-level 补充。

### 3.8 "E12/E13" —— 不做

**同意。** 这些在 v3 中已经是 P2，优先级正确。

---

## 四、最终执行方案

### 4.1 实验清单（与 Agent 基本一致，微调）

| 实验 | Runs | GPU时间 | 优先级 |
|---|---|---|---|
| Vim-T RMSNorm 补 s123+s0 | 2 | ~80min | P0 |
| PlainMamba-T RMSNorm ×3 | 3 | ~2.5h | P0 |
| ConvNeXt-T RMSNorm ×1-3 | 1-3 | ~1.5h | P0（Agent 遗漏） |
| Swin-T selective ×3 | 3 | ~4h | P0 |
| RWKV-T selective (only residual pre-norm) ×3 | 3 | ~1.5h | P0 |
| ConvNeXt-T selective ×3 | 3 | ~4h | P0 |
| Vim-T selective ×3 | 3 | ~3.5h | P0 |
| CaiT depth threshold (k=6,12,18) | 3-4 | ~3h | P0 |
| CaiT RMSNorm ×1 | 1 | ~45min | P0 |
| Swin-T RMSNorm ×1 | 1 | ~80min | P0 |
| All selective CIFAR-100-C eval | 0 训练 | eval only | P0 |
| **总计** | **~37 runs** | **~22 GPU-hours** | |

### 4.2 分析清单（与 Agent 一致）

| 分析 | 时间 | 优先级 |
|---|---|---|
| A1: Per-layer α correlation | ~2h | P0 |
| A2: Activation distribution | ~2h | P0 |
| A4: Early-α diagnostic | ~2h | P1 |

### 4.3 写作清单（关键分歧点——必须包含叙事调整）

| 改动 | 改动量 | 优先级 |
|---|---|---|
| Abstract: 加入 selective replacement + corruption 核心发现 | ~3 句话替换 | **P0（必须）** |
| Introduction: 新增 1 段 preview selective + corruption finding | ~6 行 | **P0（必须）** |
| Contribution list: 从 3→4 条 | 1 条新增 | **P0（必须）** |
| 新增 Results section: Selective replacement + Pareto/safe-deployment | ~0.5-1 页 | **P0（必须）** |
| 新增 Results subsection: Corruption robustness summary (从 appendix 提炼) | ~0.3 页 | P0 |
| Discussion: α 分化 + replacement guidelines table | ~0.3 页 | P0 |
| Discussion: RMSNorm ordering | ~0.3 页 | P0 |
| α-deadlock mechanism 从 appendix 提要到 Discussion | ~0.3 页 | P1 |

**总新增/修改量：~1.5-2 页。主文从 ~8 页增加到 ≤9 页（NeurIPS 限制内）。**

### 4.4 不做清单（与 Agent 一致 + 微调）

| 不做 | 原因 |
|---|---|
| Method 中的独立 Role Taxonomy section | 改为 Results 中作为 organizing framework |
| Algorithm 1 box | 改为 Discussion 中的 decision table |
| E5 DyT-v2 224×224 | 成本高，极大概率失败 |
| E4 DeiT α补seeds | 边际 insight 低 |
| E14 5-seed 扩展 | 3 seeds 已足够 |
| E12/E13 跨数据集+分辨率 | 优先级低于 selective |
| 全面重写 | 精准手术即可 |

---

## 五、与 Agent 的关键分歧总结

| 问题 | Agent 立场 | 我的立场 | 最终建议 |
|---|---|---|---|
| **叙事调整** | 不做，加法策略 | 必须做，但严格控制范围 | **做：改 abstract 3 句 + intro 1 段 + contribution 1 条** |
| **Algorithm 1** | 不做 | 不做 box，改为 decision table | **折中：Discussion 中 decision table** |
| **Role taxonomy** | 不做为 Contribution | 不做为独立 Contribution，但作为 organizing framework | **折中：Results 中使用，不单独 claim** |
| **Contribution list** | 保持 3 条 | 必须 3→4 条 | **做：新增 1 条 selective replacement** |
| **ConvNeXt RMSNorm** | 未列入 | 应加入 | **做：1-3 runs** |

**最核心的分歧是叙事调整。** Agent 的"加法策略"只能把概率从 18-22% 提到 ~25-30%。加入叙事调整（仅改 abstract + intro + contribution list，~0.5 页改动量）可以再提 5-8 pp 到 30-38%。这是 ROI 最高的 0.5 页改动。

---

## 六、预期概率

| 执行方案 | 预估概率 |
|---|---|
| Agent 方案（纯加法，不改叙事） | 25-30% |
| 修正方案（Agent 实验 + 叙事微调） | 30-38% |
| 修正方案 + P1 实验全部完成 | 33-42% |
| 如 Pareto 效果强（selective corruption gain >2pp） | 上述 +3-5% |

**叙事微调的 5-8 pp 增量，是整个方案中 ROI 最高的单项改动。**
