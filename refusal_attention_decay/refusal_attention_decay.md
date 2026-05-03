# NeurIPS 2026 投稿前论文优化方案（基于当前 PDF，v2 饱和执行版）

**论文题目**：*Structural Limits of Per-Head Safety Interventions in Large Language Models*
**目标会议**：NeurIPS 2026
**基于版本**：`refusal_attention_decay_2026-05-03.pdf`（40 页，含 Appendix A–AE + NeurIPS Checklist）
**更新日期**：2026-05-03

---

**硬件配置**：4 × RTX 5090 (32 GB)
**执行模式**：Agent Auto Research（全自动实验调度 + 结果分析 + 条件分支决策）
**计划时长**：4 天（~96 小时墙钟时间）
**计算预算**：4 GPU × 96 h = **384 GPU-hours**（论文原始实验总计 ~180 GPU-hours on RTX 6000 Pro）
**外部依赖**：GPT-4o API（攻击生成 orchestrator）

---

## 第一部分：当前论文内容与已完成实验全清单

### 1.1 核心问题与定位

论文研究 **per-head attention safety interventions 在多轮 jailbreak 场景下是否稳健**。核心发现是三类结构性约束：

1. **Model-specific encoding**：Mistral 上 ablation 有效（32.1% flip），Qwen 需 restoration 方向（17.8%），Llama 行为效应极弱（2%）
2. **Head selection instability**：K=15 的 catalytic step-up、alternative ranking 完全不重叠、selection 对 methodology 敏感
3. **Attack-strategy dependence**：crescendo 和 roleplay 激活完全不重叠的 head set（Jaccard=0.000），cross-attack 行为迁移近零

定位是 **negative result / mechanistic safety audit**。

### 1.2 已完成实验全清单（主文 + Appendix）

> **核心原则：在建议"补实验"之前，必须确认 appendix 中没有已完成的版本。**

#### 主文实验

| 实验 | 位置 | 模型 | n | 关键结果 |
|---|---|---|---|---|
| SC ratio/slope 三组差异 | §4.1 | Mistral/Qwen/Llama | 200 each | 76-81% heads 显著 (KW) |
| Confound control (truncation + window) | §3.3, §4.1 | All | — | 48-79% across all 9 parameter combos |
| S-vs-R head selection | §3.3 | Mistral/Qwen | — | S-vs-R 优于 (S+R)-vs-B |
| Mistral mean ablation K=20 | §4.2 | Mistral | 84 | 32.1% flips, 95% CI [22.4, 43.2] |
| Qwen restoration | §4.2 | Qwen | 73 | 17.8% comply→refuse flips |
| Llama causal + behavioral | §4.2 | Llama | 100 | d=1.76 logit, 2% behavioral |
| Dose-response K sweep | §4.3, Fig 1a | Mistral | 84 | K=15 catalytic step-up |
| Alternative ranking (KW η²) | §4.3 | Mistral | 50 | 0/20 overlap, 18% flip ≈ random |
| Cross-attack head overlap | §4.4 | Mistral/Qwen | — | Jaccard=0.000 (Mistral), 0.034 (Qwen) |
| Cross-attack behavioral transfer | §4.4 | Mistral | 11/50 | 9.1%/0% transfer, Fisher p=0.006/1.0 |
| Content-matched control | §4.5 | Mistral/Qwen | 50 each | 88%/16% significant |
| Noise direction control | §4.5, Fig 7 | Mistral | 84 | 32.1% ≈ 29.8% > 10.4% > 6.4% |
| Static vs temporal selection | §4.5 | Mistral | — | 0/20 overlap, slope stronger |
| (S+R)-vs-B negative control | §3.3, Fig 6 | Mistral | 84 | 1.2% ≈ random |
| Layer ablation | §4.5 | Mistral | 84 | Mid-layer 25%, early/late ~1-2% |
| Cross-model evidence table | Table 1 | All | — | 明确证据层级 |

#### Appendix 实验

| Appendix | 内容 | 模型 | 状态 |
|---|---|---|---|
| **A** | Confound analysis + truncation/window sensitivity | All | ✅ |
| **B** | Restoration results (logit + behavioral) | Mistral/Qwen | ✅ |
| **C** | Refusal/compliance token sets | — | ✅ |
| **D** | K-sensitivity, random control permutation (100 seeds), bootstrap stability, rank-group ablation | Mistral/Qwen | ✅ |
| **E** | Split validation (double-dipping control) | All | ✅ |
| **F** | Qwen2.5-7B exploratory replication | Qwen | ✅ |
| **G** | Llama-3.1-8B replication | Llama | ✅ |
| **H** | Gemma-2-9B-IT behavioral validation | Gemma | ✅ (null result) |
| **I** | KV-group level analysis (GQA) | All | ✅ |
| **J** | Label permutation null model | Mistral | ✅ |
| **K** | Content confound analysis (S-vs-R within attack) | All | ✅ |
| **L** | Ablation method comparison (mean vs zero) | Mistral | ✅ |
| **M** | Content-matched control (safety vs minimal prompt) | Mistral/Qwen | ✅ |
| **N** | Static vs temporal head selection | Mistral/Qwen/Llama | ✅ |
| **O** | Linear vs nonlinear SC slope models | Mistral/Qwen | ✅ |
| **P** | SC-slope jailbreak detection (AUROC) | All | ✅ |
| **Q** | Activation probing | Mistral | ✅ |
| **R** | Head selection contrast comparison (S-vs-R vs (S+R)-vs-B) | Mistral | ✅ |
| **S** | SC-differential head list (Mistral top-20) | Mistral | ✅ |
| **T** | Extended discussion | — | ✅ |
| **U** | Extended limitations | — | ✅ |
| **V** | Limitations (9 items) + 30-sample manual validation | All | ✅ |
| **W** | Noise control analysis (ordered comparison Table 24) | Mistral | ✅ |
| **X** | Roleplay attack generalization | Mistral/Qwen | ✅ |
| **Y** | Per-layer head distribution | All | ✅ |
| **Z** | Normalized SC ratio analysis | Mistral/Qwen | ✅ |
| **AA** | Post-hoc power analysis | All | ✅ |
| **AB** | Reproducibility details (hardware, software, hooks) | — | ✅ |
| **AC** | Category-wise robustness analysis | Mistral | ✅ |
| **AD** | Complete experiment summary (Table 28) | All | ✅ |
| **AE** | Practitioners' guide to per-head safety auditing | — | ✅ |

---

## 第二部分：旧方案问题状态跟踪

### 2.1 旧方案提出的问题——已在当前 PDF 中修复的

| 问题 | 旧方案建议 | 当前状态 |
|---|---|---|
| Qwen PPL 爆炸矛盾 | 修正或删除 | ✅ **已修复**：主文现在写 "negligible perplexity impact (+4.5-13.1%)"，Table 9 PPL ratios 1.02-1.08，无矛盾 |
| Noise control 叙述混淆 | 区分 logit-level 和 behavioral-level | ✅ **已改善**：§4.5 和 §5.1 现在一致使用 behavioral flip rate (~30% vs 10.4%)，不再混用 |
| 内部 baseline 散落 appendix | 提升到主文 | ✅ **大部分已整合**：static turn-1、(S+R)-vs-B、aggregate attention、layer-level 均在 §4.5 引用 |
| 证据层级不明确 | 加 evidence hierarchy table | ✅ **已添加**：Table 1 明确标注 Mistral primary / Qwen* exploratory / Llama† observational |
| Gemma 放太高 | 降级到 appendix | ✅ **已降级**：仅在脚注 2 提及，详见 Appendix H |
| 缺少完整实验汇总 | 加 summary table | ✅ **已添加**：Table 28 (Appendix AD) |
| 缺少 practitioner 指导 | 加审计协议 | ✅ **已添加**：Appendix AE 四步审计协议 |
| Split validation 仅在 appendix | 提到主文 | ✅ **已引用**：§4.2 (line 177) 和 §4.5 (line 243) 引用 split validation p 值 |
| 格式超页 | 压缩到 9 页 | ✅ **已压缩**：正文 9 页（含 Ethics + Broader Impact），References 从第 10 页开始 |

### 2.2 旧方案提出的问题——当前 PDF 中仍然存在的

| 严重度 | 问题 | 说明 | 旧方案建议 |
|---|---|---|---|
| **高** | Catalytic head 身份矛盾 | 主文 line 199 和 Conclusion line 339 称 K=15 catalytic head 是 L20H02；但 Table 23 中 L20H02 是 rank 10（K=10 即加入），rank 15 是 L20H14。Table 6 脚注 "†Random controls exclude L20H02 from the pool at K < 15" 进一步暗示 L20H02 早已在 safety set 中 | 做 single-head leave-one-out 确认 |
| **高** | Residual refusal direction baseline 缺失 | 论文 §5.1 line 260 承认 residual direction 可达 80-90% flip rate，但未做实验对比 | 在相同 conversations 上跑 Arditi et al. 方法 |
| **高** | 攻击类型仅 2 种 | 仅 crescendo + roleplay，cross-attack 结论基于 2×2 matrix | 增加 PAIR、TAP 等至少 2 种攻击 |
| **中高** | 无 scale variant | 仅 7-9B 模型，论文自己承认 "generalization to 70B+ ... untested" (§5.4 line 298) | 至少加一个 14B+ 模型 |
| **中高** | Llama 行为验证 underpowered | SUCCESS 仅 n=28 (ASR 14%)，power=0.49，2% flip 无法区分 absence 和 underpowered | 扩大攻击数量或换更强攻击 |
| **中** | 标题仍偏强 | "Structural Limits" 暗示理论性/普遍性结论 | 调整为 audit/negative result 风格 |
| **中** | Cross-attack matrix 仅 2×2 | 只有 crescendo↔roleplay，结论外推受限 | 扩展到 4×4+ |
| **中** | 缺少 single-head granularity | 仅有 rank-group ablation (Table 10-11)，无 single-head leave-one-out | 做 top-15 逐个移除 |
| **中** | 缺少 benign utility benchmark | 主文未报告 benign 条件下的 refusal rate/helpfulness | 补 utility 评估 |
| **中低** | 人工标注不足 | 仅 30 个 SUCCESS 样本，1 名标注者，无 Cohen's κ | 标注所有 flip 正例 (~40) + 第二标注者 |
| **中低** | Sample accounting 可更清晰 | 多处出现 n=84/96/94，脚注 1 解释了但仍易混淆 | 主文加统一 sample table |

### 2.3 总体评价更新

| 维度 | 旧方案评分 | 当前状态 | 说明 |
|---|---|---|---|
| Technical quality | 5/10 | **6/10** | PPL 和 noise 矛盾已修复，但 catalytic head 仍在 |
| Experimental solidity | 5/10 | **6/10** | Baselines 整合改善，但缺 residual direction |
| Clarity | 5/10 | **6.5/10** | Evidence hierarchy 清晰，Table 1 优秀 |
| Format readiness | 4/10 | **7/10** | 接近 9 页，appendix 组织良好 |
| Overall | 5/10 | **6/10** | 从 weak reject 提升到 borderline，但仍需关键实验 |

---

## 第三部分：4 天饱和执行方案

### 设计原则

1. **Day 1-2 解决"高"级问题**：catalytic head 确认、residual direction baseline、新攻击类型
2. **Day 3 解决"中高"和"中"级问题**：scale variant、Llama 扩展、benign utility
3. **Day 4 收尾 + 写作整合**
4. **每天结束有 Agent 分析检查点**
5. **注意 API 依赖**：攻击生成需要 GPT-4o，需并行安排避免 GPU 空闲等待 API

### 运行时间估算基准

基于论文 Appendix AB 的报告和 RTX 5090 性能（与 RTX 6000 Pro 对比，推理速度相当，VRAM 32GB 足够 7B-14B）：

| 操作 | 单次时间 | 备注 |
|---|---|---|
| 多轮对话生成 (crescendo, 200 convs) | ~8-10h | 含 GPT-4o API 交互 + 目标模型推理 |
| 多轮对话生成 (PAIR, 200 convs) | ~10-12h | PAIR 需更多迭代 |
| Attention 提取 (TransformerLens, 200 convs) | ~4-5h | 7B 模型全层 |
| SC slope 计算 + 统计检验 | ~1h | 纯 CPU/GPU 分析 |
| Causal patching (1 condition, n=84) | ~1.5h | 前向传播 + 干预 |
| Behavioral generation (1 condition, n=84) | ~2h | 贪心解码，max 256 tokens |
| K-sweep (9 K values, logit level) | ~5h | 不含 behavioral |
| K-sweep behavioral (9 K values) | ~18h | 每个 K 需完整 behavioral gen |
| Residual direction 估计 + ablation | ~3-4h | 已有 checkpoint，只需 forward pass |
| 14B 模型操作 | ~1.8× 7B | VRAM ~28GB fp16，fits on 32GB |

---

### Day 1：修正致命矛盾 + 最关键新 baseline（目标：解决"高"级问题）

**总计：~92 GPU-hours across 4 GPUs**

#### GPU-A：Catalytic Head 确认 — Single-Head Leave-One-Out（~23h）

**目的**：彻底解决 L20H02 vs L20H14 矛盾。这是论文内部一致性的核心问题，不修正会直接导致 reviewer 不信任。

```
Block 1: Top-15 逐个移除 — logit level (~8h)
  对 Mistral K=15 的 27 个 safety head set:
  每次移除 1 个 head（保留其余 14 个），测量 refusal logit shift
  15 条件 × n=84 × causal patching = 15 × ~0.5h = ~8h
  → 产出：哪个 head 移除后 logit shift 下降最多

Block 2: Top-15 逐个移除 — behavioral level (~12h)
  选 Block 1 中效应最大的 6 个 head，做完整 behavioral generation：
  6 条件 × n=84 × behavioral gen = 6 × ~2h = ~12h
  关键条件：
  - Top-15 remove L20H02: 如果 flip rate 从 32.1% 大幅下降 → L20H02 是 catalytic
  - Top-15 remove L20H14: 同上，rank 15 的候选
  - Top-15 remove rank 1 (L12H26): 作为对照
  → 产出：确认 catalytic head 真实身份

Block 3: Pairwise interaction 验证 (~3h)
  Top-14 + L20H02 alone vs Top-14 + L20H14 alone:
  2 条件 × behavioral gen = ~4h
  → 产出：区分 single-head contribution vs multi-head interaction
  如果两者都低 → catalytic 是 interaction effect，修改论文措辞
```

**产出**：
- 确认 catalytic head 真实身份，修正主文 line 199 和 Conclusion line 339
- 如果是 L20H14 → 全篇修改 head 标识
- 如果是 interaction effect → 改为 "catalytic head-set transition"
**依赖**：需要论文的原始 head list 和 checkpoint

#### GPU-B：Residual Refusal Direction Baseline — Mistral（~23h）

**目的**：补上论文最大的外部 baseline 缺口。论文自己引用 Arditi et al. 80-90% flip rate 但没有在同条件下对比。

```
Block 1: Refusal direction 估计 (~3h)
  用 Arditi et al. [2024] 方法：
  - 从 200 个 harmful/harmless prompt pairs 估计 residual stream refusal direction
  - 在多个 layer 上估计（选最强层）
  - 验证 direction 质量：PCA explained variance、cosine similarity stability

Block 2: Residual direction ablation — 相同 conversations (~4h)
  在论文使用的同一批 n=84 REFUSED conversations 上：
  - Residual direction ablation (hook residual stream)
  - 测量 refusal logit shift
  - n=84 × forward pass = ~2h
  + 对应 random direction control: ~2h

Block 3: Residual direction behavioral generation (~8h)
  - Greedy decoding behavioral flips: n=84 × ~2h
  - Stochastic decoding (T=0.7): n=84 × ~2h
  - 不同 ablation magnitude (0.5×, 1.0×, 2.0× direction norm): 3 × n=84 × ~1.5h ≈ 4h

Block 4: Cross-attack transfer (~4h)
  - 对 roleplay conversations 做 residual direction ablation
  - n=37 roleplay REFUSED × behavioral gen = ~2h
  - 与 per-head cross-attack failure (0%) 直接对比

Block 5: Utility cost 评估 (~4h)
  - 对 200 benign conversations 做 residual direction ablation
  - 测量 benign refusal rate, PPL, helpfulness
  - 与 per-head ablation 的 utility cost 对比
```

**产出**：
- Residual direction flip rate vs per-head 32.1% → 如果高 (80-90%)，证明 per-head 不是最优粒度
- Cross-attack transfer: 如果 residual direction 跨攻击稳健 → per-head 的 attack-dependence 是 head-level 特有问题
- Utility cost: 如果 residual direction 的 PPL/utility 差距大 → per-head 有 trade-off 优势
**依赖**：需要 Mistral checkpoint + harmful/harmless prompt pairs

#### GPU-C：PAIR Attack 生成 + Pipeline（~23h）

**目的**：增加第三种结构不同的攻击，将 cross-attack matrix 从 2×2 扩展到 3×3。

```
Block 1: PAIR 攻击生成 — Mistral (~10h)
  使用 PAIR [Chao et al., 2024] 方法：
  - GPT-4o 作为 attacker，Mistral 作为 target
  - 生成 200 PAIR conversations（迭代优化式攻击）
  - Llama-Guard-3-8B safety classification
  - 过滤 SUCCESS / REFUSED

Block 2: PAIR SC slope 提取 + head selection (~5h)
  - TransformerLens attention 提取
  - SC slope 计算 (truncate turns 1-5, w_r=5, w_s=50)
  - KW 检验 + head selection
  - 与 crescendo/roleplay top-20 对比 Jaccard overlap

Block 3: PAIR causal patching (~4h)
  - PAIR-selected heads 的 mean ablation
  - Logit shift + random control
  - 如果 PAIR heads 与 crescendo heads 不重叠 → 强化 attack-dependence 结论

Block 4: PAIR behavioral generation (~4h)
  - PAIR-selected heads ablation: behavioral flips
  - Crescendo-selected heads 在 PAIR conversations 上的 transfer test
  - PAIR-selected heads 在 crescendo conversations 上的 reverse transfer
```

**产出**：
- PAIR top-20 heads 与 crescendo/roleplay 的 Jaccard overlap
- 3×3 attack transfer matrix 的 PAIR 行和列
- 如果 3 种攻击的 heads 两两不重叠 → "attack-strategy dependence" 结论大幅加强
**依赖**：GPT-4o API

#### GPU-D：TAP Attack 生成 + Llama 数据扩展启动（~23h）

**目的**：(1) 增加第四种攻击；(2) 启动 Llama 行为验证的数据扩展。

```
Block 1: TAP 攻击生成 — Mistral (~10h)
  使用 TAP [Mehrotra et al., 2024] 方法：
  - 搜索树式多分支攻击
  - 200 TAP conversations
  - Safety classification

Block 2: TAP SC slope 提取 + head selection (~4h)
  - 同 PAIR pipeline
  - 与 crescendo/roleplay/PAIR top-20 对比

Block 3: Llama 数据扩展 (~9h)
  - 生成 600 additional crescendo conversations（总计 800）
  - 目标：获得 ≥80 SUCCESS conversations（当前 ASR 14% → 需 ~600 额外）
  - 如果 ASR 仍太低 → 尝试更强攻击变体（更长对话、更多轮次）
  - SC slope 提取开始
```

**产出**：
- TAP attack 数据 ready for Day 2 analysis
- Llama 扩展数据集开始生成
**依赖**：GPT-4o API

#### Day 1 Agent 检查点（~第 20 小时）

1. **Catalytic head 初步结果**（GPU-A Block 1 logit level 应已完成）：
   - 识别 logit shift 下降最多的 head
   - 如果是 L20H14 → 在 Block 2 中优先做该 head 的 behavioral 验证
   - 如果是 L20H02 → 需要理解为什么 rank 10 head 在 K=15 才表现为 catalytic

2. **Residual direction 初步结果**（GPU-B Block 1-2 应已完成）：
   - 检查 refusal direction 质量
   - 估计 flip rate 范围

3. **PAIR 攻击生成进度**：
   - 检查 ASR
   - 如果 PAIR ASR 太低 → 调整参数

---

### Day 2：完成 Cross-Attack Matrix + 开始 Scale Variant（目标：扩展证据广度）

**总计：~92 GPU-hours across 4 GPUs**

#### GPU-A：Cross-Attack Transfer Matrix 构建（~24h）

**目的**：从 2×2 扩展到 4×4 attack transfer matrix。

```
Block 1: TAP pipeline completion (~8h)
  - TAP causal patching
  - TAP behavioral generation
  - TAP random controls

Block 2: Cross-attack transfer — 系统性测试 (~16h)
  对 4 种攻击 (crescendo/roleplay/PAIR/TAP)：
  - 每种攻击选出的 top-15/20 heads
  - 在每种攻击的 REFUSED conversations 上做 ablation
  - 4 selection × 4 evaluation = 16 cells
  - 已有 cells: crescendo→crescendo (32.1%), crescendo→roleplay (9.1%), roleplay→crescendo (0%)
  - 需新跑: ~10 cells × behavioral gen (~2h each) = ~20h
  - 减去已有: ~16h

  产出表：
  | Selection ↓ \ Eval → | Crescendo | Roleplay | PAIR | TAP |
  |---|---|---|---|---|
  | Crescendo heads | 32.1% | 9.1% | TBD | TBD |
  | Roleplay heads | 0% | TBD | TBD | TBD |
  | PAIR heads | TBD | TBD | TBD | TBD |
  | TAP heads | TBD | TBD | TBD | TBD |
```

**产出**：
- 完整 4×4 attack transfer matrix → NeurIPS 级别的 cross-attack 证据
- 如果对角线强、非对角线弱 → 非常有力地支撑 attack-strategy dependence

#### GPU-B：Residual Direction — Qwen + Llama + Cross-model（~23h）

**目的**：将 residual direction baseline 从 Mistral 扩展到所有模型。

```
Block 1: Qwen residual direction (~10h)
  - 估计 Qwen 的 refusal direction (~2h)
  - Ablation on SUCCESS conversations: n=73 (~2h)
  - Restoration comparison: 与 per-head restoration (17.8%) 对比 (~2h)
  - Behavioral generation: ablation + restoration (~4h)

Block 2: Llama residual direction (~8h)
  - 估计 Llama 的 refusal direction (~2h)
  - Ablation on REFUSED conversations: n=100 (~2h)
  - Behavioral generation (~2h)
  - 与 per-head 的 2% flip 对比
  → 如果 residual direction 高 → 证明 Llama 的分布式编码确实是 head-level 特有

Block 3: Cross-model residual direction comparison (~5h)
  - 三模型的 refusal direction 余弦相似度
  - 跨模型迁移测试：用 Mistral direction 在 Qwen/Llama 上 ablate
  → 如果跨模型不迁移 → residual direction 也是 model-specific，但粒度更粗
```

**产出**：
- 三模型 × 两方法 (per-head vs residual direction) 的完整对比表
- Per-head vs residual direction 的 trade-off: flip rate, utility cost, cross-attack, cross-model

#### GPU-C：Scale Variant — Qwen2.5-14B（~23h）

**目的**：回应 "7B 模型结论能否推广到更大模型" 的 reviewer 质疑。选择 Qwen-14B 因为 (1) 同 family 对照；(2) 14B fits on 32GB in fp16；(3) Qwen 有 directional asymmetry 特征。

```
Block 1: Qwen-14B 对话生成 + attention 提取 (~14h)
  - 200 crescendo conversations (with GPT-4o): ~10h (14B 推理 ~1.8× 7B)
  - 200 benign conversations: ~3h
  - SC slope 提取: ~1h（更慢的 forward pass）

Block 2: Qwen-14B 观测 + 因果分析 (~9h)
  - KW 检验 + head selection: ~1h
  - Causal patching (K=20): ~3h
  - Behavioral generation (ablation + restoration): ~4h
  - Random controls: ~1h
  → 与 Qwen-7B 结果对比：directional asymmetry 是否保持？
```

**产出**：
- Qwen-14B vs Qwen-7B 的 SC dynamics 对比
- 如果 directional asymmetry 保持 → scale 不改变 encoding strategy
- 如果消失 → "asymmetry is a small-model artifact"

#### GPU-D：Llama 扩展完成 + Normalized SC Slope 验证（~22h）

```
Block 1: Llama 扩展数据 — 完成生成 + pipeline (~14h)
  - Day 1 GPU-D 的 600 额外对话的 attention 提取 + SC slope (~6h)
  - 基于扩展数据的 head selection (~1h)
  - Causal patching on expanded data (~3h)
  - 如果 SUCCESS ≥ 80:
    → Behavioral generation on expanded SUCCESS: ~4h
    → 与原始 2% flip rate 对比

Block 2: Normalized SC slope 因果验证 (~8h)
  论文 Appendix Z 发现 normalized slope top-20 与 raw slope top-20 几乎不重叠 (Jaccard ≈ 0.03)。
  但 normalized-selected heads 的因果效应未测试。
  - Normalized-slope-selected top-20 heads on Mistral
  - Mean ablation + behavioral generation: n=84 (~3h)
  - 与 raw-slope heads 的 32.1% 对比
  - Random controls: ~2h
  → 如果 normalized heads flip rate 更高 → head selection 方法有改进空间
  → 如果 flip rate 类似但 heads 不同 → 进一步支持 selection instability
```

**产出**：
- Llama 扩展后的行为验证（从 underpowered 变为 adequately powered）
- Normalized slope 的因果验证 → 回答 "normalization 是否改善 head selection quality"

#### Day 2 Agent 检查点（~第 44 小时）

1. **Cross-attack matrix 中间结果**：
   - 检查已完成的 cells
   - 如果对角线模式清晰 → Day 3 focus on depth 而非 breadth
   - 如果某些 attack pair 有意外高 transfer → 值得深入分析

2. **Residual direction 三模型结果**：
   - 汇总 per-head vs residual direction 对比
   - 决定论文中的 baseline table 设计

3. **Qwen-14B 中间结果**：
   - SC dynamics 是否与 7B 一致
   - Encoding strategy 是否保持

4. **Llama 扩展结果**：
   - 实际 SUCCESS 数量
   - 如果仍不够 → 考虑换攻击策略或只报告 observational

5. **决定 Day 3 资源分配**

---

### Day 3：深化分析 + 鲁棒性（目标：消除中级问题 + Novel 贡献）

**总计：~84 GPU-hours across 4 GPUs**

#### GPU-A：Direction Decomposition + Mechanism 深化（~22h）

**目的**：把 noise control 从 "directional vs random" 升级为 "parallel vs orthogonal decomposition"。

```
Block 1: Perturbation direction decomposition (~10h)
  估计 safety→benign direction vector d
  对 K=15 SC-differential heads:
  - Parallel-only perturbation (沿 d 方向的投影): n=84 behavioral (~2h)
  - Orthogonal-only perturbation (垂直于 d 的分量): n=84 behavioral (~2h)
  - Magnitude sweep: 0.25×, 0.5×, 1.0×, 2.0× direction norm
    各 n=84 logit level (~4h)
  - 最佳 magnitude 的 behavioral: (~2h)

Block 2: Per-epoch gradient analysis for catalytic transition (~6h)
  如果 Day 1 已确认 catalytic head 身份：
  - 提取 K=14 和 K=15 的 per-layer activation patterns
  - 分析 catalytic head 添加前后的 activation space 变化
  - 计算 interaction term: (K=15 effect) - (K=14 effect) - (single-head effect)
  → 量化 non-additive interaction contribution

Block 3: Boundary proximity × attack analysis (~6h)
  论文已发现 low-contribution conversations flip 更多 (39.3% vs 10.7%)
  扩展到新攻击类型：
  - PAIR/TAP conversations 的 boundary proximity 分布
  - 与 crescendo/roleplay 对比
  → 是否 attack-strategy dependence 部分由 boundary distribution 差异解释？
```

**产出**：
- Parallel vs orthogonal 分解 → 如果 parallel 有效而 orthogonal 无效，mechanism 更清晰
- Catalytic transition 的量化解释
- Boundary proximity 作为统一框架连接 attack-dependence 和 flip rate

#### GPU-B：Benign Utility Evaluation + Human Validation Support（~20h）

**目的**：补齐 utility cost 维度，避免 reviewer 认为 "flip 只是破坏模型"。

```
Block 1: Benign utility 系统评估 (~12h)
  对 Mistral 200 benign conversations：
  条件 1: No intervention (baseline): ~2h
  条件 2: K=15 SC-diff heads ablation: ~2h
  条件 3: K=20 SC-diff heads ablation: ~2h
  条件 4: Residual direction ablation: ~2h
  条件 5: Random K=15 heads ablation (3 seeds): ~3h

  每个条件测量：
  - Benign refusal rate (false positive)
  - PPL ratio (vs clean)
  - Answer completeness (token count ratio)
  - Safety judge label (over-refusal rate)

Block 2: PAIR/TAP 的 utility evaluation (~4h)
  对各攻击的 selected heads：
  - Benign conversations 上的 ablation
  - 与 crescendo heads 的 utility cost 对比
  → 不同攻击的 head set 可能有不同 utility trade-off

Block 3: Human validation 输出准备 (~4h)
  - 生成所有 behavioral flip 正例的完整对话文本
  - Mistral: 27 flips + Qwen: 13 flips + 新实验的 flips = ~60-80 个
  - 格式化为标注任务（redacted 有害内容，保留足够上下文判断 comply vs refuse）
  - 生成第二标注者指南
```

**产出**：
- 完整 utility cost table: intervention method × utility metric
- 人工标注材料 ready（标注本身可在 Day 4 由人完成，非 GPU 任务）

#### GPU-C：Qwen-14B Completion + DPO vs RLHF Comparison（~22h）

```
Block 1: Qwen-14B 补充实验 (~8h)
  - Roleplay attack on 14B: 100 conversations (~6h)
  - Cross-attack head overlap: 14B crescendo vs roleplay heads
  - Behavioral transfer test: ~2h
  → 产出：14B 上的 attack-strategy dependence

Block 2: Alignment procedure comparison (~14h)
  论文 Appendix W line 1045-1049 提出假设：Mistral DPO vs Qwen RLHF 可能导致不同 encoding
  如果能获取同 base 不同 alignment 的模型（例如 Mistral base vs instruct，或不同 RLHF variant）：
  - 200 crescendo on base/alternative variant: ~6h
  - SC slope + head selection: ~3h
  - Causal patching + behavioral: ~5h
  → 如果无法获取合适模型对 → 改为：
    Mistral 不同 system prompt format 对比 (2 formats × 100 convs = ~4h)
    + Qwen2.5-14B 的 full K-sweep (~10h)
```

**产出**：
- Qwen-14B cross-attack 结果
- Alignment/system prompt 格式的影响

#### GPU-D：统计整合 + 图表 + 写作准备（~20h）

```
Block 1: 完整统计检验套件 (~4h)
  对所有新增实验：
  - Fisher exact test for all behavioral comparisons
  - MWU for logit-level comparisons
  - BH-FDR correction across all tests
  - Effect sizes (Cohen's h for proportions, d for means)
  - 95% CIs (Clopper-Pearson for proportions)
  - Post-hoc power analysis for new experiments

Block 2: Cross-model × cross-attack 汇总分析 (~4h)
  - 4×4 attack transfer matrix visualization
  - 三模型 × 两方法 (per-head vs residual) comparison table
  - Scale comparison (7B vs 14B)
  - Normalized vs raw slope comparison

Block 3: 自动化图表生成 (~6h)
  - Figure: 4×4 attack transfer heatmap
  - Figure: Per-head vs residual direction comparison (per model)
  - Figure: Direction decomposition (parallel vs orthogonal)
  - Figure: Updated dose-response with catalytic head confirmed
  - Table: Comprehensive baseline comparison
  - Table: Benign utility costs
  - Table: Scale variant results

Block 4: 论文修改建议草稿 (~6h)
  - 修正 catalytic head 身份（主文 line 199, conclusion line 339, Figure 1 caption）
  - 新增 baseline table LaTeX 草稿
  - 更新 Abstract 和 Contribution
  - 新增 Appendix 草稿（attack transfer, residual direction, scale variant, utility）
  - 更新 Limitations（标记已修复项）
```

#### Day 3 Agent 检查点（~第 68 小时）

1. **所有核心实验应已完成**，检查 failed/anomalous runs
2. **4×4 transfer matrix 完整分析**：对角线 vs 非对角线对比
3. **Per-head vs residual direction**：final comparison across models
4. **Scale variant (14B) 结果**：是否改变结论
5. **Llama 扩展后**：behavioral 结果是否改变 (2% → ?)
6. **Direction decomposition**：parallel vs orthogonal 效应差异
7. **决定 Day 4 补跑优先级**

---

### Day 4：收尾 + 写作（目标：将实验转化为论文提升）

**总计：~56 GPU-hours 实验 + Agent 写作时间**

#### GPU-A/B：Failed 重跑 + 关键发现加固（~22h each, 共 ~44h）

```
基于 Day 3 检查点的补跑列表（优先级排序）：

Priority 1: 任何 failed/anomalous runs 重跑 (~8h)
Priority 2: Transfer matrix 中 anomalous cells 验证（additional seeds/sample splits）(~6h)
Priority 3: Residual direction 的 cross-attack transfer 完整测试 (~8h)
  - Residual direction on PAIR/TAP conversations
  - 与 per-head cross-attack failure 对比
Priority 4: Qwen-14B 补充 K-sweep 或 restoration (~6h)
Priority 5: Additional stochastic decoding for key conditions (~4h)
  - Residual direction under T=0.7
  - PAIR/TAP selected heads under T=0.7
Priority 6: Llama 扩展数据上的 residual direction (~4h)
Priority 7: 如果时间允许 → Qwen roleplay cross-attack behavioral (受 VRAM 限制未完成的) (~4h)
Priority 8: 探索性 — 同 base 不同 alignment 如果 Day 3 未完成 (~6h)
```

#### GPU-C：Sanity Checks + 鲁棒性验证（~10h）

```
Block 1: Benign mean 计算鲁棒性 (~3h)
  - 用 50% benign subsample 重新计算 mean activation
  - 与 full-data mean 的 ablation 效果对比
  - 复现论文 Appendix L 的 resample convergence

Block 2: Safety judge 一致性 (~4h)
  - 用 Llama-Guard 对所有新增 behavioral flips 做 safety classification
  - 与原始结果对比
  - 如果时间允许：用第二个 safety judge (如 GPT-4o) 对 30 个样本交叉验证

Block 3: 攻击生成种子鲁棒性 (~3h)
  - 用不同 GPT-4o 种子重新生成 50 个 crescendo conversations
  - 检查 SC slope distribution 是否稳定
  → 确认结果不依赖于特定攻击生成运行
```

#### GPU-D：最终分析 + 写作（~2h compute + Agent 写作）

```
Block 1: 最终版图表生成 (~2h)
  - 整合 Day 1-4 所有结果
  - 生成 publication-quality figures
  - 确保风格一致

Block 2: 完整论文修改建议 (Agent 写作)
  1. 修正 catalytic head identity（具体行号 + 新旧文本）
  2. 新增或更新的 main text sections:
     - Baseline comparison table (Table 2 或更新 Table 1)
     - 4×4 attack transfer matrix (new Figure or Table)
     - Sample accounting table
  3. 新增 Appendix:
     - Appendix AF: Residual Refusal Direction Baseline
     - Appendix AG: Extended Attack Transfer Matrix
     - Appendix AH: Scale Variant (Qwen-14B)
     - Appendix AI: Benign Utility Evaluation
     - Appendix AJ: Direction Decomposition
     - Appendix AK: Normalized SC Slope Causal Validation
  4. 更新 Abstract:
     - 加入 residual direction comparison
     - 加入 4-attack transfer matrix 结果
     - 如有 scale variant 结果，加入
  5. 更新 Contribution 表述
  6. 更新 Limitations:
     - 删除已修复项（catalytic head）
     - 更新 attack modality（从 2 种 → 4 种）
     - 更新 scale（加入 14B）
  7. 更新 Conclusion
  8. 标题建议（如果结果支持更 nuanced 的标题）
```

---

### 完整实验清单汇总

| 实验组 | 条件数 | GPU-hours | Day | 对中稿率影响 |
|---|---|---|---|---|
| **Single-head leave-one-out (catalytic)** | 23 | ~23h | 1 | 最高：修正内部矛盾 |
| **Residual direction baseline — Mistral** | 10+ | ~23h | 1 | 最高：缺失外部 baseline |
| **PAIR attack pipeline** (Mistral) | 6 | ~23h | 1 | 高：扩展 cross-attack |
| **TAP attack pipeline** (Mistral) | 5 | ~14h | 1-2 | 高：扩展 cross-attack |
| **4×4 attack transfer matrix** | ~10 cells | ~24h | 2 | 高：cross-attack 完整证据 |
| **Residual direction — Qwen + Llama** | 8 | ~23h | 2 | 高：cross-model baseline |
| **Qwen2.5-14B scale variant** | full pipeline | ~31h | 2-3 | 中高：scale concern |
| **Llama 数据扩展** (800→ conversations) | 1 pipeline | ~23h | 1-2 | 中高：underpowered 修复 |
| **Normalized SC slope causal validation** | 3 | ~8h | 2 | 中：selection 改进 |
| **Direction decomposition** (parallel/orthogonal) | 6 | ~10h | 3 | 中：mechanism 深化 |
| **Benign utility evaluation** | 5+ | ~12h | 3 | 中：utility cost |
| **Boundary proximity × attack analysis** | 4 | ~6h | 3 | 中：unified framework |
| **Alignment/prompt format comparison** | 4 | ~14h | 3 | 中：disentangle factors |
| **Human annotation preparation** | — | ~4h | 3 | 中低：validation |
| **补跑 + 关键发现加固** | ~15 | ~44h | 4 | 中高：结果可信度 |
| **Sanity checks** | 5 | ~10h | 4 | 中：reviewer 信任度 |
| **统计分析 + 图表 + 写作** | — | ~18h | 3-4 | — |
| **总计** | **~120** | **~324h** | | |

利用率：324 / 384 ≈ **84%**（剩余 ~60 GPU-hours 作为 buffer）。

> **注**：实际利用率受 GPT-4o API 响应速度影响。攻击生成步骤中 GPU 可能部分空闲等待 API 返回。建议在 API 等待期间在同一 GPU 上插入分析任务（SC slope 计算、统计检验等）以最大化利用率。

---

### 写法修改清单（与实验并行，Day 1–4 持续执行）

#### 不依赖新实验的写法修改（Day 1 即可开始）

| 修改 | 描述 | 对中稿率影响 |
|---|---|---|
| **修正 catalytic head** | Line 199, line 339, Figure 1 caption: 等 Day 1 实验结果确认后修正 | 最高 |
| **标题调整** | 考虑 "Auditing the Robustness of..." 或 "When Safety Heads Do Not Transfer..." | 中高 |
| **Sample accounting table** | 在 §4.1 后加统一样本表 (Mistral n=84/94/96, Qwen n=73, etc.) | 中 |
| **将关键 appendix 结果提升** | 100-seed permutation (App. D)、boundary proximity (App. W) 在主文引用 | 中 |
| **Claim 校准** | 确认所有 "structural" 措辞都有足够证据支撑 | 中 |

#### 实验完成后的写法更新（Day 3-4）

| 修改 | 描述 |
|---|---|
| **新增 Baseline Table** | Per-head vs residual direction vs random，三模型对比 |
| **新增 Transfer Matrix** | 4×4 attack transfer heatmap (Figure) + table |
| **新增 Appendix AF-AK** | 6 个新 appendix（见 Day 4 GPU-D） |
| **更新 Table 1** | 加入 Qwen-14B 行和 residual direction 列 |
| **更新 Abstract** | 加入 4-attack matrix、residual direction comparison |
| **更新 §5.3 Defense Implications** | 加入 residual direction 对比的 insight |
| **更新 §5.4 Limitations** | 标记已修复项，更新 attack/scale/sample size 表述 |
| **更新 Conclusion** | 加入新证据 |
| **更新 Appendix AE** | Audit protocol 根据新实验更新 |

---

### 条件分支决策树

```
Day 1 检查点:
├── Catalytic head leave-one-out 结果
│   ├── L20H14 移除导致最大 flip rate 下降 → 修正主文为 L20H14
│   ├── L20H02 移除导致最大下降 → 需要解释为什么 rank 10 在 K=15 才 catalytic
│   │   → 检查是否存在 interaction effect (Block 3)
│   └── 无单个 head 移除导致大幅下降 → 改为 "catalytic head-set transition"
│
├── Residual direction flip rate (Mistral)
│   ├── 80-90% (与文献一致) → 强参考点，per-head 明确弱于 residual
│   ├── 50-70% → 中等参考，per-head 有一定竞争力但仍低
│   └── <40% → 意外结果，需要检查 implementation
│
├── PAIR ASR
│   ├── >30% → 足够 SUCCESS 样本，继续 pipeline
│   └── <20% → 增加迭代次数或切换到其他自动攻击方法
│
└── Llama 扩展 ASR
    ├── ≥80 SUCCESS (ASR ~10%+) → Day 2 做完整行为验证
    └── <50 SUCCESS → 改为只报告 expanded observational + logit-level

Day 2 检查点:
├── 4×4 transfer matrix
│   ├── 对角线 >> 非对角线 → attack-strategy dependence 非常强
│   ├── 某些 off-diagonal cells 中等 (~10-20%) → nuanced story
│   └── 某些 attacks 有高 transfer → 有趣发现，需要分析这些 attack 的结构相似性
│
├── Residual direction cross-model
│   ├── 三模型 flip rate 都高 → per-head 粒度是核心问题
│   ├── 只有 Mistral 高 → Qwen/Llama 的分布式编码也抵抗 residual direction
│   └── Cross-model direction transfer 低 → residual direction 也是 model-specific
│
├── Qwen-14B
│   ├── Directional asymmetry 保持 → scale-independent finding
│   ├── 不保持 → scale-dependent, 需要在 limitations 中说明
│   └── ASR 低于 7B → 更强 safety alignment, 调整分析策略
│
└── Llama 扩展结果
    ├── Behavioral flip rate 显著提升 → distributed redundancy 假说需修正
    └── 仍 ~2% → 支持 distributed redundancy，现在有 adequate power 确认

Day 3 检查点:
├── Direction decomposition
│   ├── Parallel >> orthogonal → mechanism 清晰，"safety is encoded along a specific axis"
│   └── 差异不大 → direction 不是唯一因素
│
├── Utility evaluation
│   ├── Per-head utility cost << residual direction → per-head 有 precision advantage
│   └── 两者 utility cost 相当 → residual direction 严格优于 per-head
│
└── 所有实验完成度检查
    ├── >90% 完成 → Day 4 focus on writing + sanity checks
    └── <90% 完成 → Day 4 priority on completing core experiments
```

---

### 预估评分提升

| 阶段 | 预估评分 | 关键变化 |
|---|---|---|
| 当前 PDF | 6/10 | PPL/noise 修复后 borderline，但缺关键 baseline 和 cross-attack |
| Day 1 完成 | ~6.5/10 | Catalytic head 修正消除内部矛盾，residual direction + PAIR 开始 |
| Day 2 完成 | ~7/10 | 4×4 attack transfer matrix，residual direction 三模型，14B scale |
| Day 3 完成 | ~7-7.5/10 | Direction decomposition，utility evaluation，统计完整 |
| Day 4 完成 | ~7.5/10 | 写作整合，所有图表更新，sanity checks 完备 |

**诚实说明**：7.5 分对于 negative result / mechanistic audit paper 在 NeurIPS 已是较好水平。要冲 8+，需要 (1) 更大模型（70B+，但受 VRAM 限制）或 (2) 提出基于审计发现的 **新 defense 方法**（如 multi-head ensemble across attacks、distributed safety intervention），这超出当前 4 天计划范围。

**与旧方案对比**：旧方案是 1-2 周文本修订 + 少量实验的最小修订包。新方案将实验量从 ~20-30 GPU-hours 扩展到 ~324 GPU-hours，核心新增：residual direction baseline（~46h across 3 models）、4×4 attack transfer matrix（~61h including generation）、14B scale variant（~31h）、single-head catalytic（~23h）、direction decomposition（~10h）、utility evaluation（~16h）。

---

## 参考依据

- NeurIPS 2026 Main Track Handbook
- NeurIPS Paper Checklist Guidelines
- 论文 PDF：`refusal_attention_decay_2026-05-03.pdf`（40 页，Appendix A–AE）
- Arditi et al. [2024] "Refusal in language models is mediated by a single direction" (NeurIPS 2024)
- Chao et al. [2024] "Jailbreaking black box large language models in twenty queries" (NeurIPS 2024)
- Mehrotra et al. [2024] "Tree of attacks" (NeurIPS 2024)
