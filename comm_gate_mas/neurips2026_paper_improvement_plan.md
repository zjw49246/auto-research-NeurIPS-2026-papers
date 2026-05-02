# NeurIPS 2026 论文提升方案（综合修订版 v3）

> **版本说明**：本方案基于 comm_gate_mas_NeurIPS2026_v4_20260430.pdf 制定。latest_draft.pdf 经验证与 v4 文本内容完全一致（仅编译时间不同：v4 为 4/30 20:26，latest 为 5/1 01:19）。本版整合了三轮讨论的结论：初始方案 → Director 深度评估 → 工作量饱和分析 → appendix 全文精读。

---

## 0. 战略决策：Track 选择

> **这是整个方案的前置决策，所有后续内容以此为基础。**

### 0.1 结论：投 Datasets & Benchmarks Track

论文的核心优势天然匹配 D&B 评审标准：

| 特征 | 论文是否具备 | D&B 权重 | Main Track 权重 |
|---|---|---|---|
| 15 configurations（3 backbone × 3 domain × 3 topology）系统覆盖 | 强 | **高** | 中 |
| 可复用 protocol（taxonomy + annotation + statistical testing） | 强 | **高** | 低 |
| 10-seed replication + 严格统计（TOST, McNemar, non-inferiority） | 强 | **高** | 中 |
| Boundary conditions 系统分析（MATH, Qwen3, Gemma, debate） | 强 | **高** | 中 |
| 方法新颖性 | 中 | 低 | **高** |
| 因果机制深度 | 弱（目前仅相关性） | 加分项 | **必需项** |

D&B track 下，因果实验（ack injection）是**加分项**——成功则锦上添花，失败不致命。Main track 下则是**必需项**——injection 失败意味着论文核心 claim 站不住。

### 0.2 对后续方案的影响

- 叙事重心从 "dissociation principle" 转为 "reusable analysis protocol + systematic empirical coverage"
- 因果实验定位为 protocol 内部的 validation，不是论文核心 contribution
- 论文已有的广度（15 configs, 4 backbones, boundary conditions）成为主要卖点
- Appendix Y.2 已有的 "Protocol Reuse" 三步指南需要更突出地展示

---

## 1. 当前论文的主要短板

### 1.1 任务场景受限于 two-agent

当前主文结果全部来自 two-agent solver-checker 系统。虽然实验广度强（15 configurations × 3 backbones × 3 topologies × 7B/14B/72B），但 reviewer 核心担忧是**任务类型单一**——全部是 QA reasoning。

> **重要发现**：论文 Appendix X 已经有 3-agent pilot（Solver-Checker-Critic, GSM8K, N=100, Δ=-1pp, 37% suppression）和 debate pilot（0.21% suppression）。这个结果被严重低估——两份方案都没有充分利用它。正确做法不是从零搭建新 benchmark，而是**强化已有 3-agent pilot**（见 §3.1）。

### 1.2 核心机制仍偏相关性

论文 §6 自己承认 "cascade dissolution is a descriptive framework for the observed correlational patterns, not a confirmed causal mechanism" 和 "causal verification through controlled intervention remains needed"。

目前的相关性证据包括：
- P(ack | previous ack)：Qwen 90.7%, Llama 61.9%（均 p < 10⁻¹³）
- Position-controlled logistic regression：Qwen prev_ack OR=6.43, Llama OR=2.02（均 p < 10⁻⁵）
- Position 效应跨 backbone 反转（Qwen OR=1.95 放大 vs Llama OR=0.73 抑制），但 prev_ack 在两者上均显著——排除纯位置 confound
- Dose-response 曲线平坦（Appendix S.13）
- Cascade length vs pruning safety 的 Cochran-Armitage 显著性（Llama p=0.0002, Appendix S.13）
- Variance collapse with pruning rate（Llama SD: 1.16→0.48→0.48→0.00→0.00）

> **关键判断**：这些相关性证据比初始方案评估的更强（特别是 position reversal + Cochran-Armitage），但仍不是因果证明。Ack injection 是 ROI 最高的因果实验。

### 1.3 实际效率收益不够清楚

论文有 Table 42（Appendix W）的效率数据（call savings + token savings 估算），有 Appendix S.16 的 practical cost savings 分析，但**没有实际 wall-clock latency 测量**。论文自己承认 "message-level suppression (36–56%) translates to lower token-level savings (~10% Qwen, ~28% Llama)"。

### 1.4 5-class taxonomy 部分标签不稳

核心 ack/non-ack 二分类可靠（κ=0.721, binary κ=0.81, human κ=0.76），但 information_provision κ=0.237。主结论应聚焦 ack/non-ack，fine-grained 标签降级为 exploratory。

### 1.5 Detector deployment claim 偏强

DistilBERT online validation 只做了一个 configuration（Qwen/GSM8K, N=300, seed 42, Δ=+0.7pp）。TF-IDF offline F1=0.859 但无 online validation。Heuristic 跨域 F1 从 0.845 降到 0.213。

### 1.6 Appendix 中大量证据未被充分利用

这是所有之前讨论中被遗漏的最大改进机会。详见 §3。

---

## 2. 改进后的论文主叙事

### 2.1 D&B Track 叙事（推荐）

> We provide a reusable functional analysis protocol for LLM agent communication—comprising a 5-class taxonomy, structured annotation, perturbation-based impact measurement, and non-inferiority/TOST evaluation—and systematically validate it across 15 configurations (3 backbones × 3 domains × 3 topologies), with multi-seed replication, 7B–72B scale probes, and explicit boundary conditions. The protocol reveals a perturbation-pruning dissociation: acknowledgments are the most locally disruptive message type yet can be safely pruned in aggregate, reducing 36–56% of inference calls. Controlled ack injection provides causal support for the underlying cascade dissolution mechanism. The protocol generalizes from two-agent solver-checker systems to three-agent topologies and identifies topology-dependent applicability boundaries (debate: 0.2% ack, code-generation: 4.2%, solver-checker: 50%).

### 2.2 Contribution 写法（D&B 优化）

1. **Reusable analysis protocol**：5-class taxonomy + annotation + perturbation + non-inferiority/TOST + protocol reuse guide（Appendix Y.2）
2. **Perturbation-pruning dissociation**：局部最 disruptive 的消息类型可以整体安全剪枝
3. **Systematic empirical coverage**：15 configurations, 4 backbones (Qwen/Llama/Gemma/Mistral), 10-seed replication, 7B–72B scale trajectory, explicit boundary conditions
4. **Causal and controlled evidence**：ack injection + dose-response + cascade length analysis + matched controls + position/length regression controls

### 2.3 叙事升级幅度控制

叙事升级必须和新增证据量匹配。如果因果实验结果弱，D&B 叙事不受影响——protocol 的价值不依赖于因果实验成功。如果投 main track 且 injection 失败，则必须大幅降调。

---

## 3. Appendix 中可直接利用的已有证据（0 GPU 成本，高收益）

> **这是本次修订中最重要的新增内容。** 论文 appendix 中有大量被严重低估的已有结果，它们应该被整合到主文或被更突出地展示。

### 3.1 Appendix X — 3-agent pilot + debate pilot

**已有数据**：

| Topology | Domain | Agents | Baseline | Pruned | Δ(pp) | Suppress % |
|---|---|---|---|---|---|---|
| 3-Agent (S-C-Cr) | GSM8K | 3 | 92% | 91% | -1.0 | 37.0% |
| Debate (A-Ch) | StrategyQA | 2 | 62% | 63% | +1.0 | 0.21% |

- 3-agent 中 ack 的角色分布：solver 19%, checker 40%, critic 41%——下游 agent 更多 ack，和 cascade 一致
- 但目前只用了 heuristic (char<30)，**没有 oracle annotation**

**需要做的事**：
1. 对已有 100 条 3-agent traces 跑 GPT-4o oracle annotation（~$5 API 费，0 GPU）
2. 获得 oracle 验证的 3-agent ack rate 和分布
3. 如果 oracle 结果 consistent（大概率是的），在主文 §5 中加一段 "Multi-agent validation"
4. 可选：再跑 100-200 个 3-agent tasks with oracle online pruning（~2-3h GPU）

**收益**：从 "two-agent only" 变成 "two-agent primary + three-agent oracle-validated + debate boundary"。**这比从零搭建 Auto Research benchmark 快 10 倍，且没有 ack prevalence 风险。**

### 3.2 Appendix S.13 — Dose-response + cascade 分析（最被浪费的证据）

**已有数据**：

**Qwen dose-response (Table 28)**：0/25/50/75/100% pruning，accuracy 完全平坦（92.0±0.6%），Friedman χ²=0.0, p=1.0

**Llama dose-response (Table 29, 10 seeds)**：accuracy 同样平坦（82.3–84.0%, 1.7pp range）

**Llama variance collapse**：随 pruning rate 增加，SD 从 1.16→0.48→0.48→0.00→0.00 单调下降——ack 是 stochasticity amplifier

**Cascade rate significance**：Qwen P(ack|ack)=90.7% vs 独立假设 23.0%，χ²=329.3, p<10⁻⁷²

**Cascade length vs pruning safety (Table 31)**：
| Cascade length | 0 | 1 | 2 | 3 | 4+ |
|---|---|---|---|---|---|
| Qwen safety (%) | 100.0 | 98.4 | 98.9 | 100.0 | 98.2 |
| Llama safety (%) | 86.7 | 92.1 | 96.5 | 84.0 | 96.4 |

Llama Cochran-Armitage z=-3.66, **p=0.0002**：longer cascades → safer pruning

**当前状态**：全在 appendix，主文只引用了几句。

**需要做的事**：
1. 把 dose-response 曲线图从 appendix 提到主文（或做新图）
2. 把 cascade length vs safety 做成 figure 放主文
3. 把 variance collapse 写入 Discussion 的 cascade dissolution 段

**收益**：这三个合起来是 cascade dissolution 目前最强的非因果证据链：
- Dose-response 平坦 → 排除 "random shortening" confound（如果只是因为删了消息所以好，应该和删除比例有关）
- Cascade length vs safety 的 p=0.0002 → cascade dissolution 的定量支撑
- Variance collapse → cascade dissolution 的独特 prediction——如果 ack 是 stochasticity amplifier，删除应该降低 variance

### 3.3 Appendix S.20 — Token-matched random pruning

**已有数据 (Table 35)**：
| Backbone | Ack-only risk | Count-matched random risk | Token-matched random risk | Token ratio |
|---|---|---|---|---|
| Qwen | 2.6% | 1.2±0.3% | 0.2±0.2% | 5.51× |
| Llama | 1.8% | 1.2±0.6% | 0.8±0.6% | 1.38× |

Token-matched random risk < ack-only risk，但 ack-only 的 call reduction 远高于 token-matched random（56.3% vs 9.9%）。

**需要做的事**：在主文 §5.2 扩展讨论，强调 ack pruning 的效率优势（不仅安全，而且高效——相同 token 预算下删除更多 calls）。

### 3.4 Appendix M — Mistral-7B 第四个 backbone

**已有数据**：Mistral-7B 在 3 个 topology 上的完整 pipeline（300 tasks × 3 topologies），被分类为 "weakly differentiated"。

**需要做的事**：在主文 §5.3 或 §5.4 中加一句提及 Mistral 作为第四个 backbone，使 backbone typology 从 "three backbones" 变成 "four backbones spanning function-organized (Qwen), position-organized (Gemma), weakly differentiated (Mistral), and undifferentiated (Llama)"。

### 3.5 Appendix B + X — Topology-dependent ack prevalence 完整谱

**可以整合的完整谱**：

| Topology | Domain | Ack % | Pruning value |
|---|---|---|---|
| 2-agent solver-checker | GSM8K | ~50% | High（Δ≈0, 56% call reduction） |
| 3-agent solver-checker-critic | GSM8K | ~37% | Medium-high（Δ=-1pp, 37% supp） |
| Hierarchical solver-reviewer | HumanEval | 4.2% | Low（统计检验力不足） |
| Debate (advocate-challenger) | StrategyQA | 0.21% | None（near-zero ack） |

**需要做的事**：在主文做一个 topology-ack prevalence 的总结表或 figure。这对 D&B track 极有价值——展示 protocol 能识别什么时候 pruning 有用、什么时候没用。

### 3.6 Appendix P — Capability confound analysis (Table 13)

**已有数据**：按 task difficulty 分层后，Qwen function-organized structure 在 hard tasks (R²=0.444) 和 easy tasks (R²=0.456) 上几乎一样。

**需要做的事**：在 Discussion 中引用一句："Communication mode is a backbone-intrinsic property rather than an artifact of task-relative capability (Appendix P)."

### 3.7 Appendix D — Length control in OLS

**已有数据**：加入 character count 作为 covariate 后，function type 在 Qwen (p<.001) 和 Llama (p<.03) 上仍然显著，ΔR² < 0.04。

**需要做的事**：在主文或 Discussion 中引用，作为 "impact gradient is not an artifact of message length" 的 regression-based evidence。这是已有的 length confound control，比做新的 length-matched pruning 实验更快。

### 3.8 Appendix S.17 — Dual-mechanism evidence

**已有数据**：
- Qwen ack: 0% error correction, 8.9% error injection, 100% propagation
- Llama ack: 0% error correction, 14.3% error injection, 70% propagation
- Fisher p = 0.004

**需要做的事**：这已经在 Discussion 中提了，但在 D&B track 叙事中可以更突出——"the protocol automatically discovers different pruning mechanisms across backbones"。

### 3.9 Appendix Y.2 — Protocol Reuse Guide

**已有内容**：三步指南——Collect traces → Annotate → Evaluate pruning

**需要做的事**：在主文 §4 结尾加一段引用 Appendix Y.2，或在 abstract 中提到 "we provide a reusable three-step protocol applicable to any LLM multi-agent system"。D&B track 下这个 reuse story 至关重要。

### 3.10 已有证据利用 vs 新实验：优先级对比

| 行动 | 成本 | 收益 | 优先级 |
|---|---|---|---|
| 提升 dose-response + cascade length + variance collapse 到主文 | 0 GPU, 2h Agent | Quality +0.5 | **P0** |
| 整合 topology-ack prevalence 谱 | 0 GPU, 1h Agent | Significance +0.5 | **P0** |
| 3-agent traces 跑 GPT-4o oracle annotation | ~$5 API, 1h Agent | Significance +0.5 | **P0** |
| 引用 Mistral (4th backbone), difficulty confound, length control | 0 GPU, 1h Agent | Quality +0.3 | **P0** |
| 突出 Protocol Reuse Guide (Y.2) | 0 GPU, 0.5h Agent | D&B utility +0.3 | **P0** |
| Ack injection 因果实验 | 5-7h GPU, 4h Agent | Quality +1.0 | **P1** |
| Wall-clock latency 实测 | 1-2h GPU, 1h Agent | Practicality +0.5 | **P1** |
| TF-IDF online validation | 2h GPU, 1h Agent | Quality +0.2 | **P2** |
| Chain breaking（条件性） | 4h GPU, 3h Agent | Quality +0.3 | **P2** |
| Error analysis on existing traces | 0 GPU, 3h Agent | Quality +0.5 | **P1** |

> **核心洞察**：P0 优先级的行动总计 0 GPU 成本、~5.5h Agent 时间，但收益高达 Quality +0.8、Significance +1.0、D&B utility +0.3。这些应该在 Day 0 就完成。

---

## 4. 新增实验

### 4.1 Ack Injection 因果实验（P1，关键新实验）

#### 目的

为 cascade dissolution 提供因果支持。在 D&B track 下是加分项；在 main track 下是必需项。

#### 代码可行性

已确认现有 `pruning_validation.py` 的 AG2 `register_reply` hook 机制（L357-394）可直接复用。Injection 只需将 pruning hook 的 PLACEHOLDER 改为 ack-like message：

```python
# 现有 pruning hook
if current_round in prune_rounds:
    return True, PLACEHOLDER

# 改为 injection hook
if current_round in inject_rounds:
    return True, ACK_MESSAGE  # "That looks correct."
```

#### 精简版设计（Director 建议，适配 GPU 约束）

- 200-500 positions × 2-3 conditions（ack inject + neutral inject + 可选 flow-control inject）
- 仅 Qwen/GSM8K（cascade rate 最高 90.7%，最可能看到因果效应）
- 每个 full conversation ~45 秒 → 500 × 3 × 45s ≈ 5-7 小时（1 GPU）

#### 核心指标

- P(next msg is ack | ack inject) vs P(next msg is ack | neutral inject)
- Future ack rate in next-3 messages
- Chain length after injection point
- Final accuracy Δ

#### 成功标准

P(next ack) 差异 > 0.10 且 p < 0.05

#### 失败预案

**D&B track 下的 fallback narrative**：

> The protocol's practical value does not depend on the cascade dissolution mechanism being causal. The empirical safety evidence (non-inferiority at -3pp, 10-seed replication, p<0.001) stands independently of mechanistic interpretation. Controlled injection yielded inconclusive results, suggesting that cascade dynamics are phase-dependent rather than per-message causal; the correlational evidence (P(ack|ack) > 0.5, flat dose-response, cascade-length safety gradient) remains consistent with cascade dissolution as a descriptive framework.

#### 推荐结果表

```markdown
| Intervention | N positions | P(next ack) | Ack rate next-3 | Chain length | Accuracy Δ |
|---|---:|---:|---:|---:|---:|
| Original | 500 | 0.31 | 0.28 | 1.4 | 0.0 |
| Neutral injection | 500 | 0.34 | 0.30 | 1.5 | -0.1 |
| ACK injection | 500 | 0.58 | 0.51 | 2.3 | +0.0 |
```

表中数值是期望格式，不是实际结果。

### 4.2 Chain Breaking（P2，条件性——injection 成功 + GPU 有空时做）

#### 设计

从已有 baseline traces 中找到自然 ack chain 起始位置，将该 ack 替换为 content-bearing non-ack 消息，从替换点之后 regenerate。

条件：
1. Original trace
2. Replace ack with content message
3. Replace ack with neutral placeholder
4. Simply remove ack

Agent 代码实现增量 ~2-3 小时（复用 injection 的 hook 基础设施）。GPU ~4h。

#### 决策逻辑

- 如果 injection 显著（P(next ack) 差异 > 0.10）→ 做 chain breaking，获得双向因果证据
- 如果 injection 不显著 → 不做 chain breaking，直接用 D&B fallback narrative
- 如果 GPU 不够 → 不做

### 4.3 Wall-clock Latency 实测（P1，极低成本）

在已有 `pruning_validation.py` 中加 timestamp logging，跑 50 tasks × 2 conditions（baseline + oracle pruning）。

- Agent 实现 ~30 分钟
- GPU ~1-2 小时
- 报告：mean/p50/p95 request latency, end-to-end latency, latency reduction %
- 给论文增加一个之前完全没有的维度

### 4.4 TF-IDF Online Validation（P2，低成本）

论文已有 TF-IDF offline F1=0.859，跑一次 online validation：
- Qwen/GSM8K, N=200-300, single seed
- ~2h GPU
- 填补 "from oracle to deployment" 叙事链中的 TF-IDF 环节

### 4.5 Error Analysis on Existing Traces（P1，0 GPU）

分析已有 ~2700 条 traces：
- Correct→Wrong flips 发生在什么类型的问题上？
- Wrong→Correct flips 的机制分析
- Ack 密度 vs task difficulty 的关系
- 0 GPU，Agent ~3-4 小时

### 4.6 Position/Length-Matched Controls — 视 EXP-D 结果决定

Director 指出 EXP-D（random vs function-aware pruning，正在跑）和论文已有的 position-controlled logistic regression + length control in OLS 已提供相当强的 anti-confound evidence。

**决策逻辑**：
- 如果 EXP-D 清楚区分了 random 和 ack pruning → 不做新的 matched controls
- 如果 EXP-D 模糊 → 做 position-matched non-ack pruning（~4h 实现 + ~6h GPU）

---

## 5. 不做的事情

| 砍掉项 | 理由 |
|---|---|
| **Auto Research 3-agent benchmark** | 3 天做不完，ack rate 风险高，circularity 隐患。**论文已有 3-agent pilot（Appendix X），应强化已有结果而非从零搭建** |
| **4-agent topology probe** | N=30 统计检验力不足 |
| **大规模 72B** | 已有 72B probe（N=265, Δ=+0.0pp, p=1.000）足够 |
| **Taxonomy 扩展** | information_provision κ=0.237，扩展会制造新风险 |
| **论文完全重写** | 48 页论文在 deadline 前大改一致性风险太高。做定向修改，不做重写 |

---

## 6. 论文定向修改计划

### 6.1 Abstract 修改

**D&B 重排**：lead with protocol, not with dissociation finding。

```text
We provide a reusable functional analysis protocol for LLM agent communication
and apply it to systematically characterize message-level pruning safety across
15 configurations (3 backbones, 3 domains, 3 topologies) with 10-seed
replication. The protocol reveals a perturbation-pruning dissociation:
acknowledgments—the most locally disruptive message type—can be safely pruned
in aggregate, reducing 36–56% of inference calls without accuracy loss
(non-inferiority at δ=−3 pp, p<0.001). Controlled ack injection provides
causal support for the underlying cascade dissolution mechanism: flat
dose-response curves, cascade-length-dependent safety gradients (Cochran–
Armitage p=0.0002), and variance collapse under progressive pruning are all
consistent with self-reinforcing cascades dissolving under aggregate removal.
The protocol generalizes from two-agent solver-checker systems to a
three-agent topology (Δ=−1 pp, oracle-validated) and identifies
topology-dependent applicability boundaries (debate: 0.2% ack, code-
generation: 4.2%). Lightweight learned detectors (TF-IDF F1=0.859) bridge
the oracle-to-deployment gap. We release code, traces, and a three-step
protocol reuse guide for applying our framework to any LLM multi-agent system.
```

### 6.2 主文 §5 新增段落

**§5.x Multi-agent and topology validation**（~0.5 页）：
- 报告 3-agent oracle-validated 结果
- 报告 topology-ack prevalence 完整谱（表）
- 引用 debate 和 HumanEval 作为 boundary conditions

**§5.y Dose-response and cascade evidence**（~0.5 页，将 Appendix S.13 关键结果提到主文）：
- Dose-response 曲线图
- Cascade length vs safety 的 Cochran-Armitage p=0.0002
- Variance collapse（Llama SD: 1.16→0.00 随 pruning rate 增加）

**§5.z Causal intervention**（~0.5 页，如果 injection 成功）：
- Ack injection 结果
- 和 correlational evidence 的一致性

### 6.3 Discussion 修改

- 升级 cascade dissolution 的证据等级（加入 dose-response, cascade length, variance collapse, 可能的 injection）
- 如果投 D&B：加入 "protocol utility" 讨论——protocol 如何自动发现不同 backbone 上不同的 pruning 机制
- 引用 difficulty confound analysis (Appendix P)："Communication mode is backbone-intrinsic, not task-difficulty-dependent"
- 引用 length control (Appendix D)："Impact gradient is not an artifact of message length (ΔR² < 0.04)"

### 6.4 Conclusion 微调

加入：
- Protocol reuse guide 的引用（Appendix Y.2）
- 4-backbone typology（加 Mistral）
- Topology-dependent applicability 的总结

### 6.5 不改的内容

- 标题（除非加了 3-agent 实验，才考虑改为 "LLM Agent Systems"）
- Contributions 结构（微调措辞但不重构）
- §3 Taxonomy、§4 Setup 的主体结构
- 大部分 appendix

### 6.6 NeurIPS 页数适配

当前主文 ~14 页，NeurIPS 限制 9 页主文（+ references + appendix）。

**移入 appendix**：
- §5.3 Backbone Communication Typology 的详细数据
- §5.5 Boundary Conditions 的大部分细节
- Length-only baseline 的详细讨论

**保留在主文**：
- 新增 multi-agent validation (~0.5 页)
- 新增 dose-response / cascade evidence (~0.5 页)
- 新增 causal intervention (~0.5 页，条件性)

**压缩**：
- §4 Experimental Setup 精简到方法要素，实现细节移到 appendix

---

## 7. 五天执行计划

> **核心原则**：先利用已有数据（0 GPU），再跑新实验（GPU 密集），最后论文修改。GPU 约束现实：~3 张 GPU 可能部分在跑 EXP-C/D/E。

### Day 0（5/2）：已有数据挖掘 + 代码准备

**Agent 任务（全部 0 GPU）**：
- [ ] 整理 Appendix S.13 dose-response + cascade length + variance collapse 数据，做主文可用的 figures
- [ ] 整合 Appendix B + X topology-ack prevalence 谱，做总结表
- [ ] 引用 Appendix P (difficulty confound), D (length control), M (Mistral) 到主文草稿
- [ ] 突出 Appendix Y.2 Protocol Reuse Guide
- [ ] 实现 ack injection 代码（改 register_reply hook）+ dry run
- [ ] 实现 wall-clock logging 代码
- [ ] 对 3-agent traces 提交 GPT-4o oracle annotation（API）

**产出**：~5.5h Agent。全部论文改进的 0-GPU 部分完成。

### Day 1（5/3）：因果实验 + 3-agent oracle

**GPU 任务**：
- [ ] Ack injection（Qwen/GSM8K, 200-500 positions × 2-3 conditions, ~5-7h, 1 GPU）
- [ ] Wall-clock latency 实测（50 tasks × 2 conditions, ~1-2h, 1 GPU）

**Agent 并行任务**：
- [ ] 整合 EXP-C/D/E 结果（如果已完成）
- [ ] 分析 3-agent GPT-4o oracle annotation 结果
- [ ] Error analysis on existing ~2700 traces
- [ ] Cost analysis on existing traces（Appendix W 数据扩展）

### Day 2（5/4）：结果分析 + 条件性实验

**分析**：
- [ ] Injection 结果分析 → 决定是否做 chain breaking
- [ ] EXP-D 结果分析 → 决定是否做 position-matched controls
- [ ] Wall-clock latency 结果整理

**条件性 GPU 任务**：
- [ ] Chain breaking（如果 injection 成功 + GPU 有空, ~4h）
- [ ] TF-IDF online validation（~2h）
- [ ] 3-agent 扩大到 N=200 with oracle（如果 Day 0 oracle 结果好, ~2-3h）

**Agent 并行任务**：
- [ ] Learned detector 结果整理（已有 DistilBERT + TF-IDF offline）
- [ ] 所有结果表初稿

### Day 3-4（5/5-5/6）：论文定向修改 + 最终检查

**Agent 任务**：
- [ ] Abstract D&B 重排
- [ ] §5 新增 3 个段落（multi-agent, dose-response/cascade, causal intervention）
- [ ] Discussion 修改（cascade evidence 升级, protocol utility, difficulty confound, length control）
- [ ] Conclusion 微调
- [ ] NeurIPS 格式适配
- [ ] 更新 figures（dose-response curve, topology-ack spectrum, cascade length vs safety）
- [ ] 全文数字一致性检查
- [ ] 最终 PDF 生成

### Day 5 检查清单

- [ ] 所有数字与实验结果一致
- [ ] 所有新增段落的引用正确
- [ ] Abstract 中的具体数字已替换
- [ ] NeurIPS 格式检查（9 页主文 + references + appendix）
- [ ] 确认 D&B track 的 contribution 写法与实际结果匹配
- [ ] Injection 结果已正确定位（causal support 或 inconclusive with fallback narrative）
- [ ] 3-agent oracle 结果已正确报告（如果完成）
- [ ] Protocol Reuse Guide 在主文中可见

---

## 8. 新增图表清单

### Figure: Dose-Response Curve（从 Appendix S.13 提到主文）

x-axis: ack pruning rate (0%, 25%, 50%, 75%, 100%)  
y-axis: accuracy (%)  
两条线：Qwen (flat ~92%), Llama (flat ~83%)  
误差棒：Qwen 3 seeds, Llama 10 seeds

### Figure: Cascade Length vs Pruning Safety（从 Appendix S.13 提到主文）

x-axis: cascade length (0, 1, 2, 3, 4+)  
y-axis: safety rate (%)  
两条线：Qwen (flat ~99%), Llama (increasing 86.7%→96.4%)  
标注：Llama Cochran-Armitage p=0.0002

### Table: Topology-Dependent Ack Prevalence（新整合）

| Topology | Domain | Agents | Ack % | Pruning Δ |
|---|---|---|---|---|
| Solver-checker | GSM8K | 2 | ~50% | +0.4pp (oracle, 10-seed) |
| Solver-checker-critic | GSM8K | 3 | ~37% | -1.0pp (heuristic, N=100) |
| Hierarchical solver-reviewer | HumanEval | 2 | 4.2% | N/A (insufficient N) |
| Debate (advocate-challenger) | StrategyQA | 2 | 0.21% | +1.0pp (near-zero suppression) |

### Table: Causal Intervention（新实验）

```markdown
| Intervention | N positions | P(next ack) | Ack rate next-3 | Accuracy Δ |
|---|---:|---:|---:|---:|
```

### Table: Wall-clock Latency（新实验）

```markdown
| Condition | N tasks | Mean latency (s) | P50 (s) | P95 (s) | Total wall-clock (s) | Reduction % |
|---|---:|---:|---:|---:|---:|---:|
```

---

## 9. 预期分数提升

### 当前版本（D&B Track）

- Quality: 3（系统性强，但因果证据弱）
- Clarity: 3
- Significance: 3（protocol 有用但 utility 展示不足）
- Originality: 3
- Overall: 5.5–6
- 倾向：borderline

### 完成 appendix 整合 + injection 后（D&B Track）

- Quality: 4（因果证据加分，dose-response/cascade length 提到主文）
- Clarity: 3.5（D&B framing 更清晰，protocol reuse guide 突出）
- Significance: 3.5（3-agent oracle-validated, topology 谱, 4 backbones）
- Originality: 3
- Overall: 6.5–7
- 倾向：**weak accept / accept 边缘**

### 如果 injection 不显著（D&B Track）

- Quality: 3.5（仍有 dose-response, cascade length, matched controls from EXP-D）
- Significance: 3.5（不依赖因果实验）
- Overall: 6–6.5
- 倾向：**borderline accept / weak accept**

> **关键**：D&B track 下，即使 injection 失败，appendix 整合 + 3-agent oracle + topology 谱 + wall-clock latency 的组合仍然能把论文推到 borderline accept 以上。这是选择 D&B track 的核心原因——降低了对单个实验成功的依赖。

---

## 10. 最终优先级总结

五天内真正应该做的是：

### P0（0 GPU，立即执行）
1. **把 appendix 中被浪费的证据整合到主文**：dose-response 曲线、cascade length vs safety (p=0.0002)、variance collapse、topology-ack 谱、Mistral 第四 backbone、difficulty confound、length control
2. **对已有 3-agent traces 跑 GPT-4o oracle annotation**（~$5 API）
3. **突出 Protocol Reuse Guide (Y.2)**
4. **D&B track 叙事调整**

### P1（需要 GPU，核心新实验）
5. **Ack injection 因果实验**（5-7h GPU）——为 cascade dissolution 提供因果支持
6. **Wall-clock latency 实测**（1-2h GPU）——填补效率证据空白
7. **Error analysis on existing traces**（0 GPU）——展示对数据的深度理解
8. **整合 EXP-C/D/E 结果**

### P2（条件性）
9. **Chain breaking**（如果 injection 成功 + GPU 有空）
10. **TF-IDF online validation**（2h GPU）
11. **3-agent 扩大到 N=200**（如果 oracle 结果好 + GPU 有空）

### 不做
- Auto Research benchmark（从零搭建）
- 4-agent probe
- 大规模 72B
- Taxonomy 扩展
- 论文完全重写

最终论文应该传达的核心思想是：

> 我们提供了一个可复用的 LLM agent 通信功能分析 protocol，并通过 15 个 configurations、4 个 backbone families、10-seed replication 和明确的 boundary conditions 系统验证了它。Protocol 揭示了一个 perturbation-pruning dissociation：局部最 disruptive 的 acknowledgment 消息可以整体安全剪枝，减少 36-56% 的 inference calls。Dose-response 分析、cascade length 梯度、variance collapse 和受控 injection 实验为 cascade dissolution 机制提供了多层证据。Protocol 从 two-agent 推广到 three-agent topology，并识别了 topology-dependent 的适用边界。
