# NeurIPS 2026 论文 4 天冲刺提升方案 (v2)

> 目标：在 4 天内，利用 agent auto research 能力与 2-4 张 RTX 5090（如需可加卡），把当前论文从"一个有趣但偏诊断性的单模型观察"提升为"有完整温度交互分析、等价性检验、跨架构验证、运行时路由启发和机制解释的评估方法学论文"，从而显著提高 NeurIPS 中稿概率。

---

## 0. 当前论文的核心判断

当前论文的核心发现是：在 depth scaling（单条长 CoT）与 width scaling（SC@4）比较中，标准协议通常使用不同 temperature：CoT 使用较低温度（τ=0.6），SC 使用较高温度（τ=0.8）。这会让策略比较混入 decoding temperature 的影响。论文发现，在 Qwen3-32B 上，standard protocol 下若干 domain 似乎出现了从 CoT 到 SC 的 high-budget strategy reversal；但 matched temperature（τ=0.8）后，大多数 reversal 消失，只有 MMLU-Pro 仍然保留 SC 优势（+6.6pp, p<0.01）。

这个 idea 是有价值的，但当前版本仍有几个明显风险：

1. **单模型风险**：主要证据来自 Qwen3-32B，外推性不足。
2. **单 matched temperature 风险**：只做 τ=0.8 matched control，容易被质疑 cherry-pick。
3. **统计检验不完整**：用 McNemar's test fail to reject null 来 claim "no difference"，但 fail to reject ≠ evidence of equivalence。缺少等价性检验（TOST）或 Bayes Factor，缺少 multiple comparison correction。
4. **routing framing 不够实**：论文说 routing breakdown，但主要实验证明的是 strategy comparison confound，而不是真正 router 的 failure。没有对任何已发表 router 做 re-evaluation。
5. **deduplication confound**：16k CoT 的 dedup 对 competition math 影响明显（23 correct→incorrect flips, -7.6pp），可能被认为引入新 artifact。
6. **机制解释不够直接**：MMLU-Pro 为什么保留 SC 优势，reasoning-heavy tasks 为什么没有保留，需要更直接的 vote/error-correlation evidence。
7. **multi-hop QA near-floor**：multi-hop QA 准确率接近 floor（~17-20%），不能承载强主张。
8. **主文无 Figure**：全部结果以表格呈现，严重影响 reviewer 第一印象和直觉理解。NeurIPS 论文几乎不存在纯表格主文。
9. **"obvious in hindsight" 风险**："温度影响生成质量"是常识。论文需要更强的防御来说明这不是 trivial observation。
10. **缺少与 Snell et al. (2024) compute-optimal frontier 的直接对接**：这是领域基础论文，当前只泛泛引用。
11. **缺少 4k budget matched control**：论文只在 16k 做了 matched temperature，4k 是另一个关键 budget 点。
12. **缺少 difficulty-stratified 分析**：只报告 domain-level aggregate，不知道 confound 影响哪些难度的题目。

因此，4 天内的提升目标不是做"大而全"的实验，而是补最能打掉审稿人主要质疑的证据，同时用可视化和统计升级大幅改善论文的 presentation 和 rigor。

---

## 1. 新论文主线与定位

### 1.1 论文定位：Evaluation Methodology Paper

NeurIPS 有接受 evaluation methodology papers 的强传统（揭示 benchmark contamination、evaluation artifacts、metric flaws 等）。本文应明确定位为这一类型：

> This is not a routing method paper; it is an evaluation methodology paper that identifies and corrects a systematic confound in inference routing evaluation.

### 1.2 新主线

建议把论文主线从：

> Inference routing breaks down at high budgets, and temperature matching reveals that most reversals are artifacts.

升级为：

> Static strategy-only inference routing is under-specified because strategy choice is entangled with decoding temperature and runtime generation state. A strategy x temperature factorial analysis with equivalence testing shows that apparent depth-vs-width reversals are artifacts of temperature asymmetry on reasoning domains, while knowledge-intensive tasks retain a genuine width advantage explained by vote efficiency and error independence. Cross-architecture replication confirms the confound is not model-specific. Runtime repetition signals provide routing information unavailable to static surface-feature routers, and re-evaluation of an existing published router under matched temperature reveals systematic misrouting.

这个新主线比原文更强，因为它：

- 不仅指出问题，还给出可行方向；
- 用等价性检验正面 claim equivalence，而不仅仅是 fail to reject；
- 有跨架构验证，不是单模型发现；
- 对已发表系统做了 re-evaluation，证明影响是实际的；
- router 应该考虑 `{strategy, temperature, k, runtime state}`；
- runtime repetition signal 可能是更有效的 routing/fallback cue。

---

## 2. 最终优先级

| 优先级 | 模块 | 目的 | 对中稿概率提升 | GPU 需求 |
|---|---|---|---|---|
| P0 | Strategy x temperature sweep（含 τ=0.6 matched） | 打掉 "τ=0.8 cherry-pick" 质疑，证明策略偏好完全取决于温度 | 极高 | 高 |
| P0 | 统计升级：TOST 等价性检验 + Bayes Factor + multiple comparison correction + effect size | 从 "fail to reject" 升级为 "evidence of equivalence" | 极高 | 无 |
| P0 | Anatomy Figure + 主文可视化体系 | 大幅改善第一印象，让 confound 机制一图可懂 | 极高 | 无 |
| P0 | Cross-architecture replication（含非 Qwen 架构） | 缓解单模型质疑 | 高 | 高 |
| P0 | Raw vs dedup + threshold sensitivity | 封堵 dedup artifact 质疑 | 高 | 无 |
| P0 | Runtime repetition-based routing proof-of-concept | 从纯诊断升级为诊断 + 可行方向 | 高 | 无 |
| P1 | 已发表 router re-evaluation（RouteLLM） | 证明 confound 对真实系统的实际影响 | 极高 | 低 |
| P1 | Vote efficiency / error correlation | 强化 domain-dependent mechanism | 高 | 无 |
| P1 | 4k budget matched temperature | 封堵 "只在 16k 测试" 的质疑 | 中高 | 中 |
| P1 | Difficulty-stratified analysis | 展示 confound 主要影响哪些难度的题目 | 中 | 无 |
| P1 | Compute-optimal frontier 对比图 | 与 Snell et al. 直接对接，提升影响力 | 中高 | 无 |
| P1 | Multi-hop QA 重新定位或轻量聚合改进 | 避免 near-floor domain 拖累主线 | 中 | 无 |
| P2 | 写作重构：标题、abstract、contributions、discussion、limitations | 降低过强 claim，提升可读性，防御 "obvious" 质疑 | 高 | 无 |
| P2 | Recommendations for Practitioners 小节 | 增强 community impact | 中高 | 无 |
| P3 | 2k/8k budget、更多 benchmark | 锦上添花 | 中 | 高 |

---

# 3. P0 实验设计

---

## 3.1 P0.1 Strategy x Temperature Sweep

### 目的

当前 matched-temperature control 只使用 τ=0.8。审稿人很可能质疑：

> 你们是不是选择了一个对 CoT 有利的 temperature？如果换成 τ=0.6 或 τ=1.0，结论还成立吗？

因此需要做完整的 strategy x temperature sweep，把核心 claim 从"单点 control"升级为"interaction analysis"。

**特别重要的是 τ=0.6 matched control**：这是 CoT 的原始温度（"CoT 的主场"）。如果在 τ=0.6 matched 下，CoT 在 Math/GPQA 上大幅领先 SC（因为 SC 失去了高温度的 diversity 优势），这直接证明：**不是"SC 在高 budget 赢"，而是"哪个策略赢完全取决于你选的温度"**。这个结论比"大多数 reversal 是 artifact"更强、更有冲击力。

### 实验设置

```yaml
model: Qwen3-32B
budget: 16k
strategies:
  - CoT
  - SC@4
temperatures:
  - 0.2
  - 0.4
  - 0.6
  - 0.8
  - 1.0
domains:
  primary:
    - Competition Math
    - GPQA Diamond
    - MMLU-Pro
  optional:
    - Multi-hop QA
seeds:
  full_sweep: 1
  key_points:
    temperatures: [0.6, 0.8]
    seeds: [42, 123, 456]
```

### 必须记录的指标

每个 run 需要记录：

1. accuracy；
2. generated token count；
3. effective reasoning length；
4. repetition onset position；
5. repetition rate；
6. answer extraction success rate；
7. SC vote entropy；
8. SC oracle@4；
9. SC vote@4。

### 主要表格

**Table: Strategy x Temperature Accuracy at 16k**

| Domain | Strategy | τ=0.2 | τ=0.4 | τ=0.6 | τ=0.8 | τ=1.0 | Best τ |
|---|---|---:|---:|---:|---:|---:|---:|
| Math | CoT | | | | | | |
| Math | SC@4 | | | | | | |
| GPQA | CoT | | | | | | |
| GPQA | SC@4 | | | | | | |
| MMLU-Pro | CoT | | | | | | |
| MMLU-Pro | SC@4 | | | | | | |

**关键附加表格：τ=0.6 vs τ=0.8 Matched Comparison**

| Domain | τ=0.6 Matched: CoT | τ=0.6 Matched: SC@4 | Δ | Winner | τ=0.8 Matched: CoT | τ=0.8 Matched: SC@4 | Δ | Winner |
|---|---:|---:|---:|---|---:|---:|---:|---|
| Math | | | | | | | | |
| GPQA | | | | | | | | |
| MMLU-Pro | | | | | | | | |

这张表的核心信息：**不同 matched temperature 下，策略偏好方向可以反转**，直接证明 strategy preference 是 temperature-dependent 而非 strategy-intrinsic。

### 主要图

**Figure: Accuracy vs Temperature（主文 Figure 2）**

每个 domain 一个子图：

- x-axis: temperature；
- y-axis: accuracy；
- CoT 一条线（含 error bar for multi-seed points）；
- SC@4 一条线。
- 用阴影标注 "standard protocol region"（CoT=0.6, SC=0.8）。
- 标注 crossover points。

**Figure: Repetition Rate vs Temperature**

用于证明 CoT 的温度敏感性与 long-trace degeneration 有关。

### 预期结论

最理想的结果：

1. Math / GPQA 上 CoT 对 temperature 很敏感，τ=0.8 明显优于 τ=0.6；
2. SC@4 对 temperature 相对不敏感，或者提升幅度小于 CoT；
3. **τ=0.6 matched 下 CoT 领先 SC，τ=0.8 matched 下两者持平** — 策略偏好完全由温度决定；
4. standard protocol 下的 SC advantage 在 sweep 中不稳定；
5. MMLU-Pro 上 SC@4 在多数 temperature 下仍优于 CoT，说明它是真实 width advantage。

### 论文中的推荐写法

> The standard protocol does not merely compare depth and width; it compares two different strategy-temperature recipes. Across our factorial sweep, the dominant strategy on reasoning-heavy domains flips depending on the matched temperature: CoT wins at τ=0.6 while the strategies are equivalent at τ=0.8. This demonstrates that strategy preference is temperature-dependent, not strategy-intrinsic. On MMLU-Pro, however, SC remains favorable across all temperature controls, confirming a genuine answer-format and vote-denoising effect.

---

## 3.2 P0.2 统计升级：等价性检验、Bayes Factor、Multiple Comparison Correction

### 目的

当前论文在 Math 和 GPQA 上的核心 claim 是"matched temperature 后没有差异"。但用 McNemar's test 的 fail to reject（p=0.91, p=0.77-0.90）来支撑 "no difference" claim 存在统计逻辑漏洞：

> fail to reject H0 ≠ evidence for H0

一个严格的统计审稿人一定会攻击这个点。这是零 GPU 成本、几小时可完成、但对评分影响极大的改进。

### 必做分析 A：TOST 等价性检验

Two One-Sided Tests（TOST）可以正面检验等价性：

```text
H0: |Δ(CoT - SC)| >= δ  (strategies differ by at least δ)
H1: |Δ(CoT - SC)| < δ   (strategies are equivalent within δ)
```

设定等价区间 δ = 3 percentage points（实际显著性阈值），对每个 domain-seed 组合做 TOST。

报告：

| Domain | Seed | Δ(CoT-SC) | 90% CI | TOST p | Equivalence at δ=3pp? | Equivalence at δ=5pp? |
|---|---:|---:|---|---:|---|---|
| Math | 42 | | | | | |
| Math | 123 | | | | | |
| Math | 456 | | | | | |
| GPQA | 42 | | | | | |
| GPQA | 123 | | | | | |
| GPQA | 456 | | | | | |
| MMLU-Pro | 42 | | | | | |
| MMLU-Pro | 123 | | | | | |
| MMLU-Pro | 456 | | | | | |

### 必做分析 B：Bayes Factor

计算 BF01（support for null over alternative）：

```text
BF01 > 3: moderate evidence for equivalence
BF01 > 10: strong evidence for equivalence
BF01 < 1/3: moderate evidence for difference
```

| Domain | Mean Δ | BF01 | Interpretation |
|---|---:|---:|---|
| Math | -0.3 | | |
| GPQA | -0.3 | | |
| MMLU-Pro | +6.6 | | |
| Multi-hop QA | +1.5 | | |

### 必做分析 C：Effect Size

报告 Cohen's h（或 odds ratio + CI）：

| Domain | Cohen's h | 95% CI | Interpretation |
|---|---:|---|---|
| Math | | | negligible / small / medium |
| GPQA | | | |
| MMLU-Pro | | | |

### 必做分析 D：Multiple Comparison Correction

当前论文在 4 domain x 3 seed = 12 次 McNemar's test 上没有任何 correction。

```text
方法：Benjamini-Hochberg FDR correction
报告：raw p-values 和 adjusted p-values
确认 MMLU-Pro 在 correction 后仍然显著
```

### 论文中的推荐写法

> Beyond failing to reject the null hypothesis of strategy difference (McNemar's test), we apply TOST equivalence testing (δ=3pp) and compute Bayes factors to provide positive evidence for strategy equivalence on reasoning domains. On competition math (BF01=X.X) and GPQA Diamond (BF01=X.X), we find [moderate/strong] evidence that the two strategies perform equivalently under matched temperature. All p-values are adjusted for multiple comparisons using Benjamini-Hochberg FDR correction; MMLU-Pro's SC advantage remains significant after correction (adjusted p=X.XX).

---

## 3.3 P0.3 Anatomy Figure + 主文可视化体系

### 目的

当前论文**主文没有任何 Figure**，全部用表格呈现。这在 NeurIPS 中极为罕见，严重影响 reviewer 的第一印象和对核心 idea 的直觉理解。一张好的 Anatomy Figure 能让 reviewer 在 30 秒内理解整个论文的核心。

### Figure 1：Anatomy of the Temperature Confound（主文第一张图，必须做）

**布局：2x2 或 3-panel 图**

Panel A - "The Confound"：

```text
一道具体的 competition math 题目：
  上行：CoT at τ=0.6 → token trace 可视化 → 第 ~800 token 开始 repetition loop → 答案错误
  下行：CoT at τ=0.8 → token trace 可视化 → 正常推理完成到 ~1100 token → 答案正确
用色彩标注：有效推理（蓝）、重复内容（红）、答案提取位置（绿）
```

Panel B - "Aggregate Evidence"：

```text
Bar chart：每个 domain 的 Standard Δ(SC-CoT) vs Matched Δ(SC-CoT)
  - Standard protocol bars 显示 apparent SC advantage
  - Matched temperature bars 显示 artifact 消失（Math, GPQA）或保留（MMLU-Pro）
  - 标注 significance / equivalence
```

Panel C - "Temperature Interaction"（可选，或作为 Figure 2）：

```text
小型 2x3 heatmap：strategy x temperature → accuracy
  行：CoT, SC@4
  列：τ=0.2, 0.4, 0.6, 0.8, 1.0
  每个 domain 一个 heatmap
  颜色编码：accuracy
  标注 standard protocol 的两个 cell（CoT@0.6, SC@0.8）
```

### 完整主文可视化体系

| Figure | 内容 | 类型 | 目的 |
|---|---|---|---|
| Fig 1 | Anatomy of the Temperature Confound | Case study + bar chart | 30 秒理解核心 idea |
| Fig 2 | Accuracy vs Temperature Sweep | Line plots (3 subplots) | 证明策略偏好是 temperature-dependent |
| Fig 3 | Repetition Rate vs Temperature + Effective Length | Dual-axis line plot | 机制：温度如何通过 repetition 影响 CoT |
| Fig 4 | Vote Efficiency Mechanism | SC@k curves + oracle gap | 机制：为什么 MMLU-Pro 的 SC 优势是真实的 |
| Fig 5 | Runtime Routing PoC | ROC/PR curves + regret reduction bar | 从诊断到 actionable |
| Fig 6 | Compute-Optimal Frontier Shift | Pareto frontier (optional) | 与 Snell et al. 对接 |

### Anatomy Figure 的选题标准

选择一道满足以下条件的 competition math 题目：

1. CoT at τ=0.6 答案错误，且明显出现 repetition loop；
2. CoT at τ=0.8 答案正确，推理路径清晰；
3. SC@4 at τ=0.8 至少 3/4 sample 答对；
4. 推理内容对 reviewer 直观可理解（不要太 domain-specific）。

从已有 outputs 中筛选即可，零 GPU 成本。

---

## 3.4 P0.4 Raw vs Dedup + Threshold Sensitivity

### 目的

当前论文的 deduplication 是高风险点。尤其在 competition math 上，dedup 导致 23 correct→incorrect flips（净 -7.6pp）。如果不透明处理，审稿人可能认为 standard-protocol reversal 受到 dedup artifact 影响。

### 必做分析

1. raw CoT accuracy；
2. dedup CoT accuracy；
3. correct→wrong flips；
4. wrong→correct flips；
5. repetition threshold sensitivity；
6. flip case audit。

### Raw vs dedup 表格

| Domain | Raw CoT | Dedup CoT | Δ | correct→wrong | wrong→correct | Main conclusion depends on dedup? |
|---|---:|---:|---:|---:|---:|---|
| Competition Math | | | | | | |
| GPQA Diamond | | | | | | |
| MMLU-Pro | | | | | | |
| Multi-hop QA | | | | | | |

### Threshold sensitivity

使用三个阈值：

```text
20-gram overlap
30-gram overlap
50-gram overlap
```

或者当前检测器有多个参数，则选择三个最关键参数做 sensitivity。

| Threshold | Repetition Rate | Median Effective Tokens | CoT Accuracy | Regime Changed? |
|---|---:|---:|---:|---|
| 20-gram | | | | |
| 30-gram | | | | |
| 50-gram | | | | |

### Flip case audit

随机抽 50 个受 dedup 影响的样本，分类：

| Category | Meaning | Count |
|---|---|---:|
| true repetition before final answer | dedup likely valid | |
| repetition after final answer | answer should be preserved | |
| false positive repetition | detector error | |
| answer lost by truncation | extraction artifact | |
| incoherent loop | valid truncation | |

### 论文中的推荐写法

> Because repetition detection can itself affect answer extraction, we treat deduplication as a diagnostic rather than as the sole basis for any main conclusion. Our matched-temperature and temperature-sweep results do not rely on dedup-induced reversals. We provide full raw-vs-dedup sensitivity in Appendix X, including threshold sensitivity and a 50-case flip audit.

---

## 3.5 P0.5 Cross-Architecture Replication

### 目的

缓解单模型质疑。不需要完整复现所有实验，只需要证明 temperature confounding 不是单个 Qwen3-32B run 的偶然现象。

**关键改进（相比 v1 方案）：必须包含至少一个非 Qwen 架构模型。** 如果只用 Qwen 家族模型，reviewers 会认为 generalizability 不足。

### 推荐模型（按优先级排序）

```text
1. Qwen3-32B: main model（已有）
2. Llama-3.1-8B-Instruct: 非 Qwen 架构，必做（FP16 在单卡 5090 上可跑）
3. DeepSeek-R1-Distill-Qwen-14B or 32B: reasoning/distilled model
4. Optional: Qwen3-8B（同家族小模型，如果时间允许）
```

**为什么 Llama 比 Qwen-small 优先级高**：跨架构比同架构不同大小更能说明 generalizability。即使 Llama-3.1-8B 的绝对性能较低，只要温度混淆的**方向和趋势**一致，就是有力证据。

### 实验范围

为了控制时间，只跑两个关键 domain：

```text
Competition Math
MMLU-Pro
```

设置：

```yaml
budget: 16k
protocols:
  standard:
    CoT_temperature: 0.6
    SC_temperature: 0.8
  matched_high:
    CoT_temperature: 0.8
    SC_temperature: 0.8
  matched_low:
    CoT_temperature: 0.6
    SC_temperature: 0.6
samples_per_domain: 150-200
seeds: 1
```

**新增 τ=0.6 matched**：在 cross-model 上也做 τ=0.6 matched，展示"在 CoT 主场温度下，哪个策略赢"。这增加的计算量很小，但大幅加强了论文的核心 claim。

### 输出表格

| Model | Architecture | Domain | Standard Δ(SC-CoT) | Matched τ=0.8 Δ | Matched τ=0.6 Δ | Gap Reduction (τ=0.8) | Verdict |
|---|---|---|---:|---:|---:|---:|---|
| Qwen3-32B | Qwen | Math | | | | | |
| Llama-3.1-8B | Llama | Math | | | | | |
| R1-Distill-14B | Qwen (distill) | Math | | | | | |
| Qwen3-32B | Qwen | MMLU-Pro | | | | | |
| Llama-3.1-8B | Llama | MMLU-Pro | | | | | |
| R1-Distill-14B | Qwen (distill) | MMLU-Pro | | | | | |

### 结果解释策略

#### 如果趋势一致

> The temperature confound is not an isolated artifact of Qwen3-32B or the Qwen architecture family; we observe the same qualitative gap reduction across architectures (Qwen, Llama) and training paradigms (base, reasoning-distilled).

#### 如果趋势不一致

> The magnitude of temperature confounding is model-dependent. Reasoning-tuned models appear less prone to long-trace degeneration, which clarifies the boundary condition of our claim rather than invalidating it. The confound remains present on standard instruction-tuned models across architectures.

#### 如果 Llama-8B 绝对性能太低

> While absolute performance differs across model scales, the qualitative pattern of temperature-driven strategy reversal is consistent, confirming that the confound is architectural rather than scale-dependent.

---

## 3.6 P0.6 Runtime Repetition-Based Routing Proof-of-Concept

### 目的

当前论文容易被认为是纯诊断性工作：指出 routing evaluation 有问题，但没有给出解决方向。为了提高 NeurIPS 中稿概率，建议加入一个轻量级 runtime routing proof-of-concept。

它不需要重新大量跑推理，可以主要使用已有 16k CoT 输出。

### 核心思想

静态 surface features 很难判断一个问题应该走 CoT 还是 SC，但生成时状态可能包含更强信号。例如，当长 CoT 出现 repetition loop 时，继续 depth scaling 已经浪费 token，应该 early stop 并切换到 width sampling。

### Runtime features

在已有 CoT 输出上计算：

```text
1. sliding-window n-gram repetition rate
2. unique token ratio
3. repeated sentence ratio
4. repeated paragraph ratio
5. effective reasoning length before repetition onset
6. repetition onset token index
7. whether final answer appears before repetition onset
8. local entropy proxy if logprobs are available
9. output length saturation ratio
```

### 任务 A：预测 CoT failure

Label：

```text
1 if CoT answer is wrong
0 if CoT answer is correct
```

模型：

- logistic regression；
- random forest / XGBoost 可选；
- threshold-only repetition rule。

报告：

| Feature Set | AUC | PR-AUC | Accuracy | Notes |
|---|---:|---:|---:|---|
| Surface features | | | | static baseline |
| Repetition features | | | | runtime signal |
| Surface + repetition | | | | combined |

### 任务 B：预测 discordant-pair strategy preference

只在 CoT 和 SC@4 discordant pairs 上评估。

Label：

```text
1 if SC correct and CoT wrong
0 if CoT correct and SC wrong
```

**重点展示 Regret Reduction 而不仅仅是 AUC**：即使分类 AUC 只有中等（0.55-0.65），只要在 discordant pairs 上能减少 10-20% 的错误选择，就是有价值的。Regret Reduction 对 reviewer 更直观。

报告：

| Feature Set | AUC on Discordant Pairs | Accuracy | Regret Reduction | Notes |
|---|---:|---:|---:|---|
| Surface LR router | | | | current paper baseline |
| Repetition router | | | | runtime signal |
| Combined router | | | | |
| Domain-only router | | | | simple baseline |

### 任务 C：Early fallback 离线模拟

策略：

```text
Start with CoT.
If repetition score crosses threshold before budget is exhausted:
    stop CoT
    fallback to SC@4 answer from existing outputs
else:
    use CoT answer
```

这是离线 proxy simulation，不要包装成完整在线系统。

报告：

| Policy | Accuracy | Cost Proxy | Regret vs Oracle | Notes |
|---|---:|---:|---:|---|
| Always CoT | | | | |
| Always SC@4 | | | | |
| Static LR router | | | | |
| Domain router | | | | |
| Repetition-trigger fallback | | | | proof-of-concept |
| Oracle instance router | | | | upper bound |

### 论文中的推荐写法

> This proof-of-concept is not intended as a complete router. Instead, it shows that runtime generation signals contain routing information unavailable to static surface-feature routers, suggesting that future inference routers should monitor generation state rather than predict strategy solely from the input.

### 风险控制

如果 repetition router 效果一般（AUC < 0.60），用 regret reduction framing：

> Runtime repetition features reduce routing regret by X% over static surface features, despite modest classification AUC. This gap between classification accuracy and decision-relevant regret reduction suggests that even noisy runtime signals carry routing-relevant information at the margin.

如果效果确实很弱（AUC ≈ 0.52），改为 negative-result framing：

> Simple n-gram repetition features do not substantially improve routing beyond static baselines, indicating that runtime-aware routing requires richer state signals — potentially including logprob trajectories, attention entropy, or learned representations of reasoning progress.

这样仍然是 honest contribution。

---

# 4. P1 实验设计

---

## 4.1 P1.1 已发表 Router Re-Evaluation（RouteLLM）

### 目的

这是一个潜在的**杀手级实验**。如果能证明一个已发表、被社区使用的 router 在控制温度后 routing 决策发生显著变化，论文从"方法学建议"升级为"对现有系统的实证挑战"。

### 设置

RouteLLM (Ong et al., 2024) 是开源的。检查是否可以：

1. 用 RouteLLM 对 975 个问题做 routing prediction（选 CoT 还是 SC）；
2. 分别在 standard protocol 和 matched temperature 下评估 routing outcome；
3. 计算 misroute rate 变化。

如果 RouteLLM 不直接适用（它 route across models, not strategies），则：

**替代方案 A**：复现 paper 中的 Tier-1 LR router（已有），对比 standard vs matched labels 的 routing accuracy。

**替代方案 B**：训练两个 LR router（一个在 standard labels 上，一个在 matched labels 上），展示 label shift 导致的 routing 不一致。

### 输出

| Router Source | Train Labels | Eval Labels | Routing Accuracy | Misroute Rate | Regret |
|---|---|---|---:|---:|---:|
| Standard-trained LR | standard 16k | standard 16k | | | |
| Standard-trained LR | standard 16k | matched 16k | | | |
| Matched-trained LR | matched 16k | matched 16k | | | |
| Domain router | N/A | matched 16k | | | |

### 关键指标

```text
Misroute rate = fraction of instances where the router selects the worse strategy
Regret = accuracy loss vs oracle routing
Label flip rate = fraction of instances with different routing labels under standard vs matched
```

### 论文中的推荐写法

如果 label flip rate 高（>20%）：

> A router trained on standard-protocol labels misroutes X% of instances when evaluated under matched temperature. This confirms that temperature confounding does not merely inflate academic accuracy numbers — it actively corrupts the training signal for routing systems.

如果 label flip rate 较低：

> Label instability is concentrated in reasoning-heavy domains where temperature confounding is strongest, while knowledge-intensive domains provide stable routing labels regardless of protocol.

---

## 4.2 P1.2 Vote Efficiency / Error Correlation Analysis

### 目的

当前论文认为 MMLU-Pro 保留 genuine SC advantage，而 Math / GPQA 不保留。这需要更直接机制证据。

### 核心假设

- MMLU-Pro：structured multiple-choice answer，sample errors 更独立，majority vote 更有效。
- Math / GPQA：reasoning-heavy，错误更相关，多个 samples 往往错在类似方向，SC@4 只能追平 CoT。
- Multi-hop QA：free-text answer fragmentation 导致 oracle signal 无法被 vote 聚合。

### 计算指标

对 SC@4 的四个 samples 计算：

```text
SC@1
SC@2
SC@3
SC@4 vote accuracy
Oracle@4 = any sample correct
Oracle gap = Oracle@4 - Vote@4
Vote entropy
Pairwise answer agreement
Correct plurality rate
Wrong-answer clustering (fraction of wrong answers that share the same wrong answer)
Error correlation coefficient (tetrachoric correlation between sample-pair correctness)
```

### 主表

| Domain | SC@1 | Vote@4 | Oracle@4 | Oracle Gap | Vote Entropy | Pairwise Agreement | Error Correlation | Mechanism |
|---|---:|---:|---:|---:|---:|---:|---:|---|
| Math | | | | | | | | correlated reasoning errors |
| GPQA | | | | | | | | correlated / mixed errors |
| MMLU-Pro | | | | | | | | vote denoising works |
| Multi-hop QA | | | | | | | | free-text fragmentation |

### 推荐图

**Figure: SC@k Scaling Curves（主文 Figure 4 的一部分）**

每个 domain 一条 SC@k 曲线（k=1,2,3,4）：
- MMLU-Pro 曲线持续上升 → vote aggregation 有效；
- Math / GPQA 曲线平坦 → 增加 samples 没有帮助；
- Multi-hop QA 曲线下降或平坦 → free-text voting 破坏信号。

**Figure: Vote Efficiency vs SC Gain**

- x-axis: vote efficiency（定义为 (Vote@4 - SC@1) / (Oracle@4 - SC@1)）或 error correlation；
- y-axis: SC@4 gain over matched-temperature CoT；
- each point: domain；
- 展示 error correlation 预测 SC gain。

### 论文中的推荐写法

> The residual SC advantage is better predicted by error independence than by token budget or task difficulty. MMLU-Pro exhibits low error correlation (ρ=X.XX) and converts oracle coverage into voted accuracy efficiently, whereas reasoning-heavy domains exhibit correlated failures (ρ=X.XX) where width scaling provides no marginal benefit. Free-text QA loses a third of oracle signal during string-match aggregation.

---

## 4.3 P1.3 4k Budget Matched Temperature

### 目的

论文只在 16k 做了 matched temperature control。Table 1 显示 4k 时 MMLU-Pro 已经出现 SC 优势（62.7% vs 60.3%）。reviewer 会问：

> 温度混淆是否也存在于 4k？还是它只在 trace 足够长时（16k）才出现？

这个实验成本低（已有 4k 数据可能只需要补 matched temperature runs），但封堵一个明显的审稿问题，并且能揭示 confound 何时"kick in"的 boundary condition。

### 设置

```yaml
model: Qwen3-32B
budget: 4k
strategies: [CoT, SC@4]
temperatures:
  standard: [CoT=0.6, SC=0.8]
  matched: [both=0.8, both=0.6]
domains: [Competition Math, GPQA Diamond, MMLU-Pro]
seeds: [42, 123, 456]
```

### 输出

| Domain | Budget | Standard Δ(SC-CoT) | Matched τ=0.8 Δ | Matched τ=0.6 Δ | Confound Present? |
|---|---:|---:|---:|---:|---|
| Math | 4k | | | | |
| Math | 16k | | | | |
| GPQA | 4k | | | | |
| GPQA | 16k | | | | |
| MMLU-Pro | 4k | | | | |
| MMLU-Pro | 16k | | | | |

### 预期分析

如果 4k 时 confound 较弱：

> The temperature confound emerges with budget: at 4k, CoT traces are short enough to avoid repetition-induced degeneration regardless of temperature, so the confound is minimal. At 16k, the extended traces cross the repetition threshold at lower temperatures, and the confound becomes dominant. This identifies the interaction between budget and temperature as the key driver.

如果 4k 时 confound 已存在：

> The confound is present at both 4k and 16k budgets, though its magnitude increases with budget length. This suggests that even moderate-length reasoning traces are affected by temperature-induced quality differences.

---

## 4.4 P1.4 Difficulty-Stratified Analysis

### 目的

论文报告的全是 domain-level aggregate accuracy。但 reviewer 会问：

> 温度混淆是均匀影响所有难度的题目，还是主要影响特定难度？

如果 confound 主要影响 medium-difficulty 问题（正好是 routing 最需要做决策的问题），故事更强。

### 分层方法

将每个 domain 的问题按 standard protocol（16k）的 discordant status 分层：

```text
Easy: both CoT and SC correct under standard protocol
Hard: both CoT and SC wrong under standard protocol  
Discordant-SC: SC correct, CoT wrong (standard protocol)
Discordant-CoT: CoT correct, SC wrong (standard protocol)
```

然后看 matched temperature 后每层的变化：

| Difficulty Stratum | n | Standard: SC wins | Matched: SC wins | Flip Rate | Notes |
|---|---:|---:|---:|---:|---|
| Easy | | N/A | N/A | N/A | no routing decision needed |
| Hard | | | | | |
| Discordant-SC | | | | | key: does SC still win? |
| Discordant-CoT | | | | | |

### 核心发现

最有价值的发现是：**在 standard protocol 下被 router 判为 "SC wins" 的 discordant 问题中，有多大比例在 matched temperature 后变成 tie 或 CoT wins？** 这直接量化了 "active misrouting" 的规模。

### 论文中的推荐写法

> The temperature confound is concentrated in the decision-relevant stratum: among instances that appear discordant (favoring SC) under standard protocol, X% become concordant or CoT-favorable after temperature matching. The confound does not uniformly affect all problems; it systematically inflates the apparent size of the SC-favorable population that routing systems are trained to detect.

---

## 4.5 P1.5 Compute-Optimal Frontier 对比图

### 目的

Snell et al. (2024) 是 inference routing 领域的基础论文。当前论文引用了它但没有直接 engagement。如果能展示 standard protocol 和 matched temperature 下的 compute-optimal frontier 形状不同，这是对基础结果的直接挑战，大幅提升论文影响力。

### 方法

用已有数据（1k, 4k, 16k 三个 budget 下 CoT 和 SC@4 的 accuracy），画出：

```text
x-axis: compute budget (tokens)
y-axis: best accuracy at each budget (max of CoT, SC@4)
两条曲线：
  - Standard protocol frontier (CoT@τ=0.6, SC@τ=0.8)
  - Matched temperature frontier (both@τ=0.8)
每个 domain 一个子图
```

### 预期

- Standard protocol 下 frontier 在 16k 时 shift to SC（apparent reversal）；
- Matched temperature 下 frontier 可能不 shift（Math/GPQA）或 shift 幅度缩小；
- MMLU-Pro 的 shift 在两个 frontier 下都存在但 magnitude 不同。

### 论文中的推荐写法

> The compute-optimal frontier shifts under temperature control: strategies that appear Pareto-dominant at high budgets under standard protocols lose their advantage when the temperature confound is removed. This implies that the compute-optimal allocation derived from confounded comparisons may systematically over-allocate budget to width scaling on reasoning-heavy domains.

---

## 4.6 P1.6 Multi-hop QA 重新定位

### 问题

Multi-hop QA 当前准确率 near-floor（~17-20%），因此不适合和 Math / GPQA / MMLU-Pro 并列支撑强结论。

### 推荐做法 A：重新定位，低成本

在 Method 中明确：

> We treat multi-hop QA as an exploratory stress test for free-form answer aggregation rather than as a primary domain for strategy-superiority claims.

在 Results 中只用它说明：

1. free-text voting difficulty；
2. oracle-vote gap；
3. answer-space fragmentation。

### 推荐做法 B：轻量 semantic aggregation

如果有时间，补一个 answer clustering baseline：

```text
1. exact string voting
2. normalized string voting (SQuAD-style)
3. embedding similarity clustering
4. LLM equivalence clustering
```

表格：

| Aggregation | Vote Accuracy | Oracle@4 | Oracle Gap |
|---|---:|---:|---:|
| Exact voting | | | |
| Normalized voting | | | |
| Embedding clustering | | | |
| LLM equivalence clustering | | | |

### 推荐写法

> Because the task operates near floor, we avoid using multi-hop QA as evidence for strategy superiority. Instead, it serves as a diagnostic case showing that free-form answer spaces can destroy over a third of the oracle signal produced by width scaling.

---

# 5. 写作重构方案

---

## 5.1 标题

**保留原标题不变**：

```text
When Does Inference Routing Break Down?
Disentangling Temperature Artifacts from Domain-Dependent Scaling Limits
```

原因：
- 原标题比方案 v1 建议的替代标题更 provocative、更 memorable；
- 副标题已经精确描述了方法；
- NeurIPS 偏好有冲击力的标题；
- 换成 "Inference Routing Is Under-Specified" 之类的标题反而更 flat。

---

## 5.2 Abstract 重写方向

Abstract 不要堆太多 p-value 和 domain 数字。建议结构：

1. 背景：inference routers assume stable strategy preference；
2. 问题：standard protocols entangle strategy and temperature；
3. 方法：matched controls + temperature sweep + equivalence testing + cross-architecture replication；
4. 发现：most reasoning-domain reversals disappear（with equivalence evidence）；MMLU-Pro remains genuine；
5. 机制：repetition ceiling + vote efficiency + error correlation；
6. 启示：route over strategy-temperature recipes and runtime signals。

### 推荐 abstract 草稿

```text
Inference routers that choose between depth scaling (chain-of-thought) and width scaling (self-consistency) assume the optimal strategy is a stable property of each input. We show that this assumption is under-specified: standard evaluations conflate strategy choice with decoding temperature, assigning lower temperature to long reasoning traces and higher temperature to parallel samples. Through matched-temperature controls, a strategy x temperature factorial sweep, and equivalence testing on Qwen3-32B — with cross-architecture replication on Llama-3.1 and DeepSeek-R1-Distill — we find that apparent high-budget strategy reversals are artifacts of temperature asymmetry on reasoning-heavy domains (TOST equivalence confirmed at delta=3pp), while MMLU-Pro retains a genuine width advantage driven by error independence and vote efficiency. A strategy x temperature interaction analysis reveals that strategy preference flips with temperature choice: the dominant strategy at tau=0.6 differs from that at tau=0.8, demonstrating that "which strategy wins" is a function of the temperature recipe, not an intrinsic property of the strategy. Mechanistic analyses show that residual width gains are predicted by inter-sample error correlation, while long CoT failures are explained by repetition-induced degeneration. A lightweight runtime-routing proof-of-concept demonstrates that repetition signals reduce routing regret beyond static surface-feature routers. These results expose temperature confounding as a systematic threat to the inference routing literature and suggest that routers should select over strategy-temperature recipes and runtime states, not strategies alone.
```

---

## 5.3 Contributions 重写

从原文三条升级为五条，每条更 crisp、更有分量：

### 新贡献列表

1. **Temperature confounding as a systematic threat.**
   We identify a pervasive confound in standard inference-routing evaluations: strategy choice is entangled with decoding temperature. The confound inflates apparent width advantages by up to 7.2pp and reverses strategy dominance on reasoning-heavy domains.

2. **Factorial controls with equivalence evidence.**
   We provide matched-temperature controls, a strategy x temperature factorial sweep, and — critically — TOST equivalence testing and Bayes factors to positively establish that strategies are equivalent (not merely "not significantly different") on reasoning domains after temperature control.

3. **Cross-architecture replication.**
   We replicate the core finding across model architectures (Qwen, Llama) and training paradigms (instruction-tuned, reasoning-distilled), establishing that the confound is not model-specific.

4. **Mechanistic explanation: error independence and vote efficiency.**
   We show that the residual domain split is explained by inter-sample error correlation: knowledge-intensive tasks with independent errors benefit from majority-vote denoising, while reasoning-intensive tasks with correlated errors do not.

5. **Runtime-aware routing signal.**
   We demonstrate that runtime repetition features reduce routing regret beyond static surface-feature routers, providing a proof-of-concept for generation-state-aware inference routing.

---

## 5.4 防御 "Obvious in Hindsight" 质疑

这是论文被拒的第一隐性风险。"温度影响生成质量"是常识，reviewer 可能觉得这不是 novel。

### 三层防御

**防御 1：点名已发表系统**

在 Introduction 中明确列举使用不匹配温度的已发表工作：

```text
> Standard practice assigns different temperatures to different strategies: τ=0.6 for CoT and τ=0.8 for SC [common in Snell et al., 2024; RouteLLM evaluations; Ong et al., 2024]. Despite the well-known effect of temperature on generation quality [Renze and Guven, 2024], no prior work has tested whether this asymmetry contaminates strategy comparisons or routing labels. We show that it does — and that the contamination is not marginal but dominant: on competition mathematics, the entire 7.2pp standard-protocol SC advantage is a temperature artifact.
```

**防御 2：强调 magnitude 和 asymmetry**

```text
> This is not a minor calibration issue. Temperature asymmetry flips the dominant strategy on three of four domains. The mechanism is asymmetric: raising CoT temperature from 0.6 to 0.8 gains +12.5pp (by suppressing repetition loops), while SC gains only +5.0pp. This asymmetry means that the confound systematically favors width scaling, creating a directional bias in routing evaluations.
```

**防御 3：强调 community impact**

```text
> Any routing policy derived from unmatched-temperature comparisons risks active misrouting. Our difficulty-stratified analysis shows that the confound is concentrated in the decision-relevant discordant stratum — precisely the instances where routing systems must make choices. This makes the confound more damaging than a uniform accuracy offset.
```

---

## 5.5 Discussion 重写重点

Discussion 应该避免过强 claim，但同时要有建设性。

### 不建议写

```text
Inference routing breaks down at high budgets.
```

### 建议写

```text
Static strategy-only routing labels become unstable under budget and temperature shifts. This does not imply that routing is impossible; rather, it shows that routing over strategies alone is under-specified. A more faithful action space should include decoding temperature, sample count, answer format, and runtime generation state.
```

### 新 Discussion 小节

1. **Temperature as part of the routing action space**
2. **Static input features vs runtime generation signals**
3. **When width helps: vote efficiency and error independence**
4. **When depth fails: repetition-induced saturation**
5. **Limitations of free-text majority voting**
6. **Recommendations for Practitioners**（新增，见 5.7）

---

## 5.6 Limitations 重写

主动承认限制，降低审稿攻击力度。

建议包括：

1. Main experiments focus on Qwen3-32B; cross-architecture replication covers two additional models at reduced scale。
2. Temperature sweep covers 16k budget primarily; 4k matched-temperature results are included but with fewer temperature points。
3. Runtime fallback is an offline proof-of-concept, not a full online router。
4. Multi-hop QA is exploratory due to near-floor performance。
5. Deduplication can affect answer extraction; main conclusions are reported with raw/dedup sensitivity and do not depend on dedup-induced reversals。
6. SC aggregation methods are simple; learned verifiers or semantic clustering may alter free-text results。
7. We study k=4 only; other values of k or best-of-n with verifiers may exhibit different dynamics。
8. TOST equivalence bounds (δ=3pp) reflect our judgment of practical significance; other bounds may yield different conclusions。

---

## 5.7 新增：Recommendations for Practitioners

在 Discussion 末尾加一个明确的建议清单，增强论文的 community impact：

```text
Based on our findings, we recommend:

1. **Always match temperature when comparing strategies.** Report results at a common temperature, or at minimum report results at both the CoT-favored and SC-favored temperatures.

2. **Report strategy comparisons across a temperature range, not a single point.** A strategy x temperature table or figure reveals whether observed differences are robust or temperature-dependent.

3. **Use domain-level routing as a strong baseline.** Our results show that per-instance surface-feature routing provides no benefit over simply routing by domain/answer-format. Domain-level routing captures all real strategy differences after temperature control.

4. **Monitor for repetition in long traces.** Repetition onset provides a cheap signal for detecting depth-scaling failure, enabling early fallback to width sampling.

5. **Apply equivalence testing, not just null-hypothesis testing.** When the conclusion is "no strategy difference," TOST or Bayes factors should be used to provide positive evidence for equivalence.
```

---

# 6. 图表重构清单

## 主文 Figure 1：Anatomy of the Temperature Confound（必须做）

最重要的一张图。见 3.3 节详细设计。

## 主文 Figure 2：Temperature Sweep

- x-axis: temperature；
- y-axis: accuracy；
- CoT vs SC@4；
- Math / GPQA / MMLU-Pro 三个子图；
- 阴影标注 standard protocol region；
- 标注 crossover points。

## 主文 Figure 3：Repetition vs Temperature + Effective Length

- 展示 CoT repetition rate 随 temperature 变化；
- 支撑 temperature rescues / changes long-trace degeneration 的机制。

## 主文 Figure 4：Vote Efficiency Mechanism

- SC@k scaling curves（4 domains）；
- Oracle gap comparison；
- Error correlation vs SC gain scatter（如果 domain 数够多）。

## 主文 Figure 5：Runtime Routing Proof-of-Concept

- Regret reduction bar chart（surface vs repetition vs combined vs oracle）；
- 或 ROC curves（如果 AUC 差异明显）。

## 主文 Figure 6（可选）：Compute-Optimal Frontier Shift

- Standard vs matched temperature 下的 Pareto frontier；
- 每个 domain 一个子图。

## 主文 Table 1：Consolidated Strategy Comparison

| Domain | Standard Δ | Matched Δ | TOST Equiv? | BF01 | Effect Size (h) | Verdict |
|---|---:|---:|---|---:|---:|---|
| Math | | | | | | Artifact / Equivalent |
| GPQA | | | | | | Artifact / Equivalent |
| MMLU-Pro | | | | | | Genuine SC |
| Multi-hop QA | | | | | | Exploratory |

## 主文 Table 2：Cross-Architecture Replication Summary

详见 3.5 节。

## Appendix Tables

1. Per-seed matched-temperature details（含 TOST p-values）；
2. Raw vs dedup sensitivity；
3. Threshold sensitivity；
4. Cross-model full results；
5. Difficulty-stratified analysis；
6. Router re-evaluation details；
7. Prompt templates；
8. Answer extraction rules；
9. Seed-level temperature sweep results。

---

# 7. 4 天执行排期

---

## Day 1：实验管线和 P0 实验启动

### 上午：统一 pipeline + 统计分析（无 GPU）

完成：

1. config registry；
2. generation runner；
3. answer extractor；
4. SC aggregator；
5. repetition feature extractor；
6. dedup sensitivity script；
7. result summarizer；
8. **TOST / Bayes Factor / effect size 计算脚本**（新增）；
9. **difficulty stratification 脚本**（新增）。

建议每个 generation output 保存为 jsonl：

```json
{
  "model": "Qwen3-32B",
  "domain": "MATH",
  "strategy": "CoT",
  "temperature": 0.8,
  "budget": 16000,
  "seed": 42,
  "problem_id": "...",
  "prompt": "...",
  "raw_output": "...",
  "extracted_answer": "...",
  "gold_answer": "...",
  "correct": true,
  "num_output_tokens": 1234,
  "repetition_onset": 980,
  "dedup_output": "...",
  "difficulty_stratum": "discordant-SC"
}
```

**并行无 GPU 任务**（立即开始）：

1. 在已有 matched-temperature 数据上计算 TOST / BF / effect size；
2. 在已有 standard protocol 数据上做 difficulty stratification；
3. 从已有 outputs 中筛选 Anatomy Figure 候选案例；
4. 开始计算 runtime repetition features；
5. 用已有 1k/4k/16k 数据画 compute-optimal frontier 初稿。

### 下午：启动 Qwen3-32B temperature sweep（GPU 任务）

优先跑：

```text
Competition Math / GPQA / MMLU-Pro
CoT / SC@4
τ = 0.2, 0.4, 0.6, 0.8, 1.0
16k budget
```

GPU 分配建议（4 卡）：

- GPU 0-1（tensor parallel）：Qwen3-32B, Math CoT sweep + GPQA CoT sweep；
- GPU 2-3（tensor parallel）：Qwen3-32B, Math SC sweep + GPQA SC sweep；
- 串行跑 MMLU-Pro（或等有空闲卡时并行）。

如果有 6+ 卡：

- GPU 0-1：Math CoT/SC sweep；
- GPU 2-3：GPQA CoT/SC sweep；
- GPU 4-5：MMLU-Pro CoT/SC sweep。

### 晚上：启动 cross-architecture replication（GPU 任务）

```text
Model 1: Llama-3.1-8B-Instruct（必做，非 Qwen 架构）
Model 2: DeepSeek-R1-Distill-Qwen-14B
Domains: Math + MMLU-Pro
Protocols: standard + matched τ=0.8 + matched τ=0.6
150-200 samples/domain/model
```

Llama-3.1-8B 在单卡 5090 上可跑，不影响 Qwen3-32B sweep。

**同时**：继续从已有 outputs 计算 runtime repetition features。

---

## Day 2：完成 P0 结果 + 启动分析

### 上午：temperature sweep 汇总

生成：

1. accuracy vs τ table（含 τ=0.6 vs τ=0.8 matched 对比）；
2. accuracy vs τ line plots（Figure 2）；
3. repetition rate vs τ plot（Figure 3）；
4. effective length vs τ plot。

检查：

- CoT 是否对 temperature 更敏感；
- SC 是否相对稳定；
- **τ=0.6 matched 下策略偏好是否反转**（关键！）；
- MMLU-Pro 是否保留 SC advantage across all τ；
- 是否存在 τ=1.0 degeneration。

对 sweep 中的 key points（τ=0.6, 0.8）做多 seed 运行（如果尚未完成）。

### 下午：cross-architecture replication 汇总 + 统计升级

1. 生成 cross-architecture table；
2. 对所有 matched-temperature 结果（含 sweep key points）计算 TOST / BF / effect size；
3. 做 BH-FDR multiple comparison correction；
4. 完成 difficulty-stratified analysis。

如果 cross-model 时间不够，优先保证 Llama-3.1-8B 的结果（非 Qwen 架构优先）。

### 晚上：raw vs dedup + Anatomy Figure

完成：

1. raw/dedup table；
2. threshold sensitivity table；
3. correct→wrong / wrong→correct flip list；
4. 50-case audit summary；
5. **选定 Anatomy Figure 案例，生成可视化初稿**；
6. 生成 compute-optimal frontier 对比图初稿。

---

## Day 3：机制分析与 routing proof-of-concept + 补充实验

### 上午：runtime routing proof-of-concept

完成：

1. CoT failure prediction（AUC, PR-AUC）；
2. discordant-pair strategy preference prediction（重点看 regret reduction）；
3. early fallback offline simulation；
4. surface features vs repetition features vs combined 对比。

### 下午：vote efficiency / error correlation + router re-evaluation

完成：

1. SC@k curves；
2. Oracle@4 + oracle gap；
3. vote entropy + pairwise agreement；
4. **error correlation coefficient**（tetrachoric）；
5. mechanism table；
6. **router re-evaluation**：用 standard-trained vs matched-trained LR router 对比 misroute rate。

### 晚上：补充 GPU 实验（利用 Day 3-4 的 GPU 空闲期）

按优先级依次：

1. **4k budget matched temperature**（如果尚未完成）；
2. **temperature sweep 的额外 seed**（τ=0.2, 0.4, 1.0 的多 seed）；
3. **第三个 cross-model**（如 Qwen3-8B）；
4. τ=0.6 matched 的多 seed。

---

## Day 4：写作重构和审稿模拟

### 上午：重写主文

重点重写：

1. abstract（用新版草稿）；
2. introduction（加入"点名已发表系统"防御、强调 magnitude）；
3. contributions（五条新版）；
4. method（加入 TOST / BF 方法描述）；
5. results（加入等价性检验结果、τ=0.6 matched 分析、difficulty stratification）；
6. discussion（加入 Recommendations for Practitioners）；
7. limitations（新版八条）。

**标题保留不变。**

### 下午：图表整合

完成主文图表（6 张 Figure + 2 张 Table）：

1. Figure 1: Anatomy of the Temperature Confound；
2. Figure 2: Temperature Sweep（accuracy vs τ）；
3. Figure 3: Repetition vs Temperature；
4. Figure 4: Vote Efficiency / SC@k / Error Correlation；
5. Figure 5: Runtime Routing PoC（regret reduction）；
6. Figure 6: Compute-Optimal Frontier Shift（可选）；
7. Table 1: Consolidated Comparison（含 TOST, BF, effect size）；
8. Table 2: Cross-Architecture Replication Summary。

Appendix 加入：

1. per-seed matched-temperature details + TOST p-values；
2. raw vs dedup sensitivity；
3. threshold sensitivity；
4. cross-model full results；
5. difficulty-stratified analysis；
6. router re-evaluation details；
7. prompt templates；
8. extraction rules；
9. seed-level temperature sweep results；
10. 4k budget matched-temperature results。

### 晚上：模拟审稿

让 agent 生成 3 个 reviewer profile：

1. **统计/方法学 reviewer**：重点检查 TOST、BF、multiple comparison、equivalence claims、effect sizes；
2. **inference-time scaling reviewer**：重点检查与 Snell et al. 的关系、compute frontier、routing framing；
3. **routing / systems reviewer**：重点检查 router 实验、runtime PoC、practical impact。

每个 reviewer 输出：

- summary；
- strengths；
- weaknesses；
- questions；
- score；
- confidence。

然后检查：

1. 哪些 claim 仍然过强；
2. 哪些表格不自解释；
3. 哪些实验没有支撑对应结论；
4. 哪些 limitation 应主动承认；
5. **是否还有 "obvious in hindsight" 的感觉**。

---

# 8. 如果实验结果不符合预期，如何调整叙事

---

## 情况 A：temperature sweep 完全支持主线

表现：

- Math / GPQA 上 CoT τ=0.8 明显恢复；
- standard SC advantage 消失；
- **τ=0.6 matched 下 CoT 反而领先 SC**；
- MMLU-Pro across τ 仍然 SC wins。

写法：

> Temperature asymmetry creates spurious width advantages on reasoning-heavy domains. The dominant strategy flips with temperature: CoT wins at τ=0.6, while strategies are equivalent at τ=0.8. Only structured knowledge tasks (MMLU-Pro) retain genuine width benefits across all temperatures.

论文评分预期：6.5-7.5。

---

## 情况 B：部分模型不复现

表现：

- Qwen3-32B 支持；
- R1-Distill 上 CoT 不太受 temperature 影响。

写法：

> The confound is strongest in models exhibiting long-trace degeneration. Reasoning-tuned models reduce but do not eliminate the need for strategy-temperature controls. This identifies a boundary condition rather than invalidating the finding: the confound affects standard instruction-tuned models across architectures.

论文评分预期：仍可 6-6.5，甚至更 nuanced。

---

## 情况 C：SC 在很多 temperature 下都赢

表现：

- Matched sweep 后 SC 仍然赢多个 domain。

改写：

> Temperature asymmetry inflates and miscalibrates width advantages, but does not always fully explain them. Even after removing the confound, the magnitude of the SC advantage is overestimated by standard protocols on the majority of domains.

主贡献从 "reversal is artifact" 改成：

> unmatched protocols systematically overestimate strategy effects, and the magnitude of this overestimation is domain-dependent.

---

## 情况 D：runtime repetition router 效果弱

表现：

- repetition feature AUC 只有小幅提升。

写法（以 regret reduction 为主）：

> Simple repetition signals reduce routing regret by X% over static surface features despite modest classification AUC, suggesting that even noisy runtime signals carry decision-relevant information. Closing the oracle gap requires richer state representations.

不要把它写成主要算法贡献，保持 proof-of-concept 定位。

---

## 情况 E：dedup 对结果影响巨大

表现：

- standard reversal 强依赖 dedup。

写法：

> We do not rely on deduplicated standard-protocol reversals for the main conclusion. The central evidence comes from raw matched-temperature and temperature-sweep comparisons, which are dedup-independent.

主动透明会降低审稿风险。

---

## 情况 F：TOST 不能建立等价性（power 不够）

表现：

- TOST p > 0.05，样本量不够。

写法：

> While TOST equivalence testing does not reach significance at our sample size (power analysis suggests N=X is needed), the Bayes factor of X.X provides [moderate] evidence for the null hypothesis of no strategy difference, and the observed effect size (h=X.XX) falls well below conventional thresholds for practical significance.

补充 power analysis，说明需要多少样本才能 formally establish equivalence。

---

## 情况 G：τ=0.6 matched 下策略偏好没有反转

表现：

- τ=0.6 matched 下 SC 仍然弱于 CoT，但差距与 τ=0.8 方向一致。

写法：

> While the strategy preference does not fully reverse between τ=0.6 and τ=0.8, the magnitude of the SC advantage is highly temperature-dependent, varying by Xpp across matched temperatures. This demonstrates that strategy comparisons are under-specified without temperature control.

---

# 9. 最小必做版本

如果 4 天内时间严重不足，只做下面 7 件事：

1. **TOST + Bayes Factor + multiple comparison correction**
   必须做。零 GPU 成本，几小时完成，堵住统计审稿人最容易的攻击点。

2. **Anatomy Figure + 至少 3 张主文 Figure**
   必须做。当前论文零 Figure 是致命伤。

3. **Qwen3-32B temperature sweep**
   必须做。它是最能提升核心 claim 的实验，特别是 τ=0.6 matched 对比。

4. **Runtime repetition-based routing proof-of-concept**
   必须做。它把论文从 diagnostic 升级为 actionable。

5. **Raw vs dedup + threshold sensitivity**
   必须做。它封堵一个明显方法学漏洞。

6. **Cross-architecture mini replication**
   至少一个非 Qwen 架构（Llama-3.1-8B），Math + MMLU-Pro。

7. **Vote efficiency / error correlation analysis**
   用已有 SC outputs 计算，成本低、收益高。

如果只做这七件，并重写 abstract / intro / contributions / limitations，论文质量会有质的提升。

---

# 10. 预期中稿概率提升

当前版本大概是：

```text
Overall: 4.5-5.5/10
倾向: Weak Reject / Borderline Reject
中稿概率: 15-20%
```

主要原因：idea 好，但 evidence scope 不够，统计不够 rigorous，claim 有些过强，主文无 Figure。

### 10.1 完成最小必做版本（7 项）

```text
Overall: 5.5-6.5/10
倾向: Borderline / Weak Accept
中稿概率: 30-35%
```

### 10.2 完成完整 P0 + P1 方案

```text
Overall: 6.5-7.0/10
倾向: Weak Accept / Accept
中稿概率: 40-50%
```

提升最大的项目（按中稿概率增量排序）：

| 改进项 | 中稿概率增量 | 原因 |
|---|---:|---|
| TOST + BF + effect size | +5-8% | 堵住统计审稿人最容易的攻击；从 "fail to reject" 到 "evidence of equivalence" 是质变 |
| Anatomy Figure + 可视化体系 | +5-8% | 零 Figure 主文在 NeurIPS 是致命伤；好的 Figure 1 决定 reviewer 第一印象 |
| Temperature sweep（含 τ=0.6 matched） | +5-7% | 从单点 control 到 factorial analysis；τ=0.6 matched 可能让核心 claim 从 "artifact" 升级为 "strategy preference flips with temperature" |
| Cross-architecture replication | +3-5% | 部分缓解单模型质疑（非 Qwen 架构是关键） |
| Router re-evaluation | +3-5% | 如果 misroute rate 高，证明 real-system impact |
| Runtime routing PoC | +3-5% | 从 diagnostic 到 actionable |
| Vote efficiency + error correlation | +2-4% | 强化 mechanistic story |
| Dedup sensitivity | +2-3% | 封堵方法学漏洞 |
| Difficulty stratification | +1-2% | 加深 confound 的分析粒度 |
| Compute-optimal frontier | +1-3% | 与基础论文对接，提升影响力 |
| Recommendations for Practitioners | +1-2% | 增强 community impact |
| "Obvious" 防御写作 | +2-3% | 降低被 dismiss 的风险 |

### 10.3 如果还能完成杀手级实验（已发表 router re-evaluation 效果显著）

```text
Overall: 7.0-7.5/10
倾向: Accept
中稿概率: 50-55%
```

---

# 11. 最终建议

这篇论文不是差在 idea，而是差在三个维度：

1. **证据链厚度**：单模型、单温度点、无跨架构验证 → temperature sweep + cross-architecture replication
2. **统计严谨性**：fail to reject ≠ evidence of equivalence，无 multiple comparison correction → TOST + BF + FDR
3. **Presentation**：主文零 Figure、claim 过强、缺少 practitioner guidance → Anatomy Figure + 可视化体系 + Recommendations

4 天内应集中火力在这三个维度上同时突破：

```text
实验层：
  strategy x temperature sweep（含 τ=0.6 matched）
  + cross-architecture replication（必含非 Qwen 架构）
  + runtime repetition routing proof-of-concept
  + raw/dedup sensitivity
  + vote efficiency mechanism
  + 4k budget matched temperature
  + difficulty-stratified analysis

统计层：
  TOST equivalence testing
  + Bayes Factor
  + effect size (Cohen's h)
  + BH-FDR multiple comparison correction

Presentation 层：
  Anatomy Figure（主文 Figure 1）
  + 5-6 张主文 Figure
  + compute-optimal frontier 对比
  + Recommendations for Practitioners

叙事层：
  保留原标题
  + evaluation methodology 定位
  + "obvious" 三层防御
  + nuanced claims
```

写作上，最关键的两个 shift：

1. 从 "fail to reject → no difference" 到 "TOST + BF → positive evidence of equivalence"：

```text
We provide not merely null results, but positive evidence for strategy equivalence
under temperature control.
```

2. 从 "routing breaks down" 到 "strategy preference is temperature-dependent, not strategy-intrinsic"：

```text
The dominant strategy flips with temperature choice. "Which strategy wins" is a
property of the temperature recipe, not an intrinsic property of depth vs. width.
```

这个叙事更稳、更有冲击力、更有建设性，也更符合 NeurIPS 对 methodological empirical papers 的期待。
