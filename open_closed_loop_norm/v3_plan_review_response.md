# 对 Agent v3 方案评估的详细回复

> 实验精简做得很好，大部分实验选择认同。但叙事调整是整个方案中 ROI 最高的单项操作，不能放弃。

---

## 一、实验部分：基本同意，补一个遗漏

你的实验选择高效且聚焦，基本全部认同。特别是：

- Swin-T/RWKV-T/ConvNeXt-T/Vim-T 四个架构的 selective replacement 是核心实验，选得好
- CaiT depth threshold (k=6,12,18) 是因果方向最强的验证之一
- RMSNorm 补 Vim-T/PlainMamba-T/CaiT/Swin-T，覆盖关键空缺
- 砍掉 E5 (DyT-v2 224×224) 完全正确——训练时间确实被低估，且 tanh bounded range 是数学硬约束
- 砍掉 E4 (DeiT α补seeds)、E14 (5-seed 扩展) 也合理

**补一个遗漏：ConvNeXt-T RMSNorm（1-3 seeds）。** 你列了 CaiT/Swin 的 RMSNorm probe，但漏了 ConvNeXt。ConvNeXt 是唯一的 CNN 架构，它的 RMSNorm 数据对确认 hierarchical role 的特殊性有价值。新增 1-3 runs，成本极低（~1.5h）。

---

## 二、叙事调整：这是我们最大的分歧，也是 ROI 最高的改动

你说"论文经过 7 轮 review 叙事已稳定，用加法策略更安全"。我理解你的谨慎，但这个判断我认为有问题，需要仔细说明。

**核心论点：内部 review 的叙事稳定 ≠ NeurIPS 竞争力最优。**

7 轮内部 review 优化了文字连贯性和数据一致性，这很好。但如果论文的结构性定位是 "diagnostic study"（我们发现 DyT 在这些架构上不 work），那再稳定的叙事也无法回答审稿人的第一个问题："Interesting observations, but where's the method contribution?"

NeurIPS 的 accept rate ~25%，在这个竞争强度下，一篇没有 prescriptive takeaway 的论文——即使实验再扎实——被归为 "nice empirical study, borderline reject" 的概率很高。你现在要做的 selective replacement 实验本身就是一个 method contribution，问题是如果 abstract 和 intro 不 frame 它，审稿人可能读到第 6 页才发现这个贡献，而大多数审稿人在前 2 页就已经形成了对论文定位的判断。

**我不是要求全面重写。** 具体改动量非常小：

| 位置 | 改动 | 量 |
|---|---|---|
| Abstract | 替换最后 2-3 句：加入 selective replacement + corruption robustness 的核心发现 | ~3 句话 |
| Introduction | 新增 1 段：preview selective replacement finding | ~6-8 行 |
| Contribution list | 3 条 → 4 条：新增 selective replacement 相关贡献 | 1 条 |

**总计 ~0.3-0.5 页文字改动。** 不动 Related Work，不动 Method 3.1-3.3，不动 Training Setup，不动现有 Results，不动现有 Discussion。之前 7 轮稳定下来的叙事主体完全保留。

这 0.5 页的改动，可能是整个方案里 ROI 最高的单项操作。如果不做，跑的 selective replacement 实验只是 appendix-level 的补充数据；如果做了，它变成论文的第四个 contribution。差别是 "我们发现问题" vs "我们发现问题并给出解决方案"。

**具体建议：** 不需要现在决定 abstract 的措辞。先把 selective replacement 实验跑完，看到结果后再决定 framing 是 "Pareto optimal"（如果 corruption gain >2pp）还是 "safe deployment protocol"（如果 corruption gain 较弱）。但 **请预留改 abstract/intro 的时间**，不要在 Day 3 才发现没有时间做这个调整。

---

## 三、Algorithm 1：同意你的判断，建议改为 decision table

你说 Algorithm 1 本质是 if-else lookup table，NeurIPS 审稿人对贡献膨胀敏感。这个判断我认同。

**折中方案：** 在 Discussion 或 Results 的 selective replacement section 中放一个 decision table：

```
Table X: Recommended replacement strategy by normalization site type.
Site type                    | Action          | Rationale
Residual pre-attn/pre-FFN    | Replace with DyT | Activation stabilizer; α stays low
SSM state-adjacent            | Probe first      | α-sensitive; monitor early-α
Key/Query functional norm     | Preserve as LN   | Structural invariant
Downsample/stage transition   | Preserve as LN   | Cross-scale feature management
```

比 Algorithm box 谦虚，但比纯文字更有可引用性和实践价值。这个 table 加上 A1 的 α 分化数据，就构成了 data-driven replacement guidelines。

---

## 四、Role Taxonomy：同意不作为独立 Contribution，但需要作为 organizing framework

你说 role taxonomy 有循环定义风险——"role 的判断标准 = norm 在计算图中的位置 = topology"。这个担忧部分成立，但有个细微区别：

- **Topology classification**（open-loop vs closed-loop）预测 DyT 结果 → 有 4 个反例 → 解释力不足
- **Site-level characterization**（pre-attn vs key_norm vs downsample）描述 norm 的计算图位置 → 不是 prediction，是 description → 不存在反例

所以同意不把它作为独立 Contribution 条目，但建议在 selective replacement results 中用它作为 **organizing framework**——解释为什么某些 sites 可替换、某些不可。避免用 "taxonomy" 这个词，改用 "site-specific characterization" 或 "normalization site analysis"。让 A1 的 α 数据 + selective replacement 实验结果自己 reveal 分化模式，而不是 top-down 预设 4 个类别。

你说的"让 α 数据本身 reveal 分化"正好是这个思路的验证方法——如果 A1 显示不同位置的 norm 确实学到了显著不同的 α 分布，那就是 data-driven 的证据，审稿人很难攻击。

---

## 五、Contribution list 必须调整

你说保留现有 3 条 Contribution 不改。这个不同意——如果跑了 4 个架构的 selective replacement 实验但 contribution list 里没有提及，审稿人会认为这只是补充实验。

建议从 3 条改为 4 条（只新增 1 条，不改现有 3 条）：

现有 3 条保留：
1. First systematic cross-topology evaluation of DyT
2. α-deadlock characterization
3. RWKV-T key_norm decomposition

新增第 4 条（措辞待定，等实验结果）：

> We show that role-aware selective replacement — preserving architecture-critical normalization sites while replacing activation-stabilizer sites with DyT — recovers clean accuracy and [maintains/improves] corruption robustness across X architectures, providing empirically-grounded deployment guidelines.

---

## 六、总结

| 项目 | 你的决定 | 我的建议 | 最终执行 |
|---|---|---|---|
| 实验 ~35 runs | 同意 | 补 ConvNeXt-T RMSNorm 1-3 runs | 按你的 + 补 ConvNeXt RMSNorm |
| 叙事调整 | 不做 | **必须做，但范围严格控制在 ~0.5 页** | 改 abstract 3 句 + intro 1 段 + contribution 3→4 |
| Algorithm 1 | 不做 box | 同意不做 box，改为 decision table | Discussion 中 decision table |
| Role taxonomy | 不做为 Contribution | 同意不做独立 Contribution，但作为 organizing framework | Results 中使用，不单独 claim |
| Contribution list | 保持 3 条 | 3→4 条 | 新增 1 条 selective replacement |
| A1/A2/A4 分析 | 做 | 同意 | 按你的执行 |

**最核心的一点：叙事微调（~0.5 页改动量）是整个方案中 ROI 最高的操作。** 如果只做加法不改 framing，预估概率 25-30%；加上叙事微调，30-38%。这 5-8 pp 的差别来自 0.5 页改动，值得做。

请先把实验跑起来，Day 2 看到 selective replacement + CIFAR-100-C 结果后，一起决定 abstract/intro 的具体措辞。
