# 附录数据审计：可直接利用的实验结果

**基于论文版本：** 2026-05-02（43 页，正文 9 页 + 附录 A–Z.1）  
**目的：** 识别附录中已有但未被主文充分利用的高价值数据，为剩余实验和写作改动提供零成本支撑。

---

## 一、对剩余实验的直接支撑

### 1.1 Divergence position 分析 → Table 51 已基本完成

**数据来源：** Appendix W, Table 51（n=1,466 questions, θ=0.03）

| Trace position | Overall NM | Numerical | Special | Reas. KW | Other |
|---|---:|---:|---:|---:|---:|
| Q1 (0–25%) | 17.8% | 39.6% | 21.2% | 4.6% | 10.6% |
| Q2 (25–50%) | 16.4% | 37.0% | 19.5% | 2.3% | 9.1% |
| Q3 (50–75%) | 15.3% | 34.9% | 18.4% | 1.8% | 8.4% |
| Q4 (75–100%) | 18.7% | 38.9% | 23.8% | 1.4% | 10.3% |

**关键发现：U 形分布。**
- Q1（exploration phase）和 Q4（answer production）NM 最高
- Q3（convergent reasoning）NM 最低
- Q4 spike 来自 answer segment：`</think>` 后 NM=32.7% vs reasoning body 15.1%（Table 45）
- Reasoning KW 随 trace 深入显著稳定化：4.6% → 1.4%（3.2× decay）

**对论文的意义：**
- 计划中的 divergence position 分析（P0-sub）大部分数据已存在，只需补充 "first divergence token position" 的 CDF 即可
- U 形分布直接支撑两种 constructive 策略：(1) warm-up（Q1 保护），(2) answer-segment protection（Q4 保护）
- 可以在新增的 Section 5.3（segment-level exit）中直接引用 Table 51 作为 position-based motivation
- **工时估算从 ~1h 降至 ~0.5h**

---

### 1.2 Segment-level exit → Table 41 + Table 33 提供关键 segment accuracy 差异

#### Table 41（Appendix S）：per-segment exit accuracy

**数据来源：** Appendix S, Table 41（R1-Qwen3-8B, 500 GSM8K questions, boxed-answer subset n=200）

| Layer | Overall | Reasoning | Answer | Boxed | Answer Correct |
|---|---:|---:|---:|---:|---:|
| L20 | 3.7% | 3.8% | 3.4% | 0.5% | 0% |
| L24 | 17.4% | 17.4% | 18.0% | 16.0% | 0% |
| L28 | 32.4% | 32.3% | 33.5% | 27.4% | 0% |
| L32 | 61.6% | 61.4% | 65.1% | 78.3% | 1.6% |
| L35 (final) | 100.0% | 100.0% | 100.0% | 100.0% | 84.3% |

**关键发现：**
- Boxed tokens 在 L32 收敛最快（78.3%），reasoning tokens 最慢（61.4%）——segment 类型间有显著的 exit-readiness 差异
- 即使只评估 boxed answer，L32 仍只有 1.6% answer correctness——compound error 在 answer segment 内也是毁灭性的
- L28 → L32 是所有 segment 的 phase transition 区间（accuracy 约翻倍）

**对论文的意义：**
- **直接支持 segment-level exit 的核心设计**：不同 segment 可以使用不同 exit layer
- 建议 segment-level exit 实验中设计 "reasoning 段用 L_low + answer/boxed 段用 L_high" 的策略
- 主文 Section 5.3 可以引用 Table 41 展示 segment-level accuracy 差异作为 motivation

#### Table 33（Appendix P）：V2a gate 自动学习的 per-type 行为

| Token Role | Exit Rate | Token Acc |
|---|---:|---:|
| Numerical | 68.50% | 99.31% |
| Reasoning KW | 37.88% | 100.00% |
| Other | 48.18% | 99.30% |
| Special | 55.70% | 99.55% |

**对论文的意义：**
- Gate 已经自动学习了 differentiated exit strategy——numerical tokens 最积极（68.5% exit），reasoning KW 最保守（37.9%）
- 所有类型在 γ=0.99 下 token accuracy 均 >99.3%——差异在 exit 频率而非 accuracy
- **这证明 segment-level exit 在概念上合理**：gate 已经在 per-token 粒度做了类似的事，segment-level 只是把这个逻辑提升到更粗的粒度
- 可以用 gate 的 per-type exit rate 作为 segment-level exit 的 informed baseline

---

### 1.3 Native thinking CALM+gate → Table 45 提供丰富的预期参照

**数据来源：** Appendix U, Table 45（Qwen3-8B architecture, n=499）

| Metric | Thinking | R1-Qwen | Qwen3 (non-reas.) |
|---|---:|---:|---:|
| Overall NM | 9.1% | 20.8% | 11.6% |
| Numerical NM | 16.0% | 42.9% | 15.3% |
| Answer NM | 10.0% | 32.7% | — |
| Exit accuracy L28 | 34.7% | 32.4% | 47.7% |
| Exit accuracy L32 | 62.6% | 61.6% | 73.3% |
| **Oracle savings (95%, 9-layer grid)** | **20.0%** | **19.2%** | — |
| Proxy ρ (cos-sim vs exit) | 0.109 | — | — |
| Ground-truth accuracy | 95.6% | 84.3% | — |
| Short traces exit acc | 72.5% | 57.9% | — |
| Long traces exit acc | 29.6% | — | — |

**三个被忽略的重要发现：**

**发现 1：Native thinking oracle savings = 20.0%，与 R1-Qwen（19.2%）几乎相同。**
- 意义：native thinking 的 representation 在 oracle 层面与 distilled reasoning 有相当的 early-exit potential
- 这是 CALM+gate 实验的关键预期参照：如果 gate 在 native thinking 上能恢复到接近 oracle 的 20% savings，就完全消除 distillation confound
- 建议在 Section 4.4 扩展时明确引用这个数字

**发现 2：Answer segment NM——thinking 10.0% vs R1-Qwen 32.7%（3.3× 差异）。**
- 意义：distillation 显著放大了 answer segment 的 oscillation，但 native thinking 的 answer segment 几乎不受影响
- 这进一步证明 oscillation 是 distillation-associated，且 distillation 的影响在 answer production 阶段最为显著
- 建议在 Section 4.4 中补充讨论这个数据点

**发现 3：短 trace vs 长 trace exit accuracy 方向反转。**
- Native thinking：短 trace 72.5% > 长 trace 29.6%（expected direction）
- R1-Qwen：57.9% short，方向 reversed（Table 45 标注 "reversed"）
- 意义：distillation 可能引入了 trace-length dependent 的 representation 特性，导致长 trace 反而 exit accuracy 更高
- 这是一个值得在 Discussion 中探讨的 counterintuitive finding

---

## 二、应在主文中引用的附录数据（零成本写作改动）

### 2.1 Compound error 量化支撑

#### Table 52（Appendix W）：Trace-length stratification

| Quintile | Token range | Non-mono (%) | CosSim L34 | Q_Acc (θ=0.60) |
|---|---|---:|---:|---:|
| Q1 (shortest) | 274–967 | 11.4±4.6 | 0.903 | 47.0% |
| Q2 | 967–1,695 | 13.7±5.7 | 0.889 | 26.0% |
| Q3 | 1,695–2,839 | 12.7±6.7 | 0.871 | 14.0% |
| Q4 | 2,839–4,549 | 11.1±5.9 | 0.859 | 2.0% |
| Q5 (longest) | 4,549–16,384 | 10.6±5.0 | 0.845 | 0.0% |

**NM rate 在所有 quintile 上稳定（~11–14%），但 Q_Acc 从 47% 单调下降到 0%。**

**对论文的意义：**
- 这是 compound error p^T 模型的**完美实证验证**：per-token quality 稳定，trace-level quality purely decays with length
- 47% → 0% 的 decay 比 Table 50 的理论预测更直观
- **建议在主文 Section 5.1 引用关键数据点**：例如 "Non-monotonicity rates remain stable at 11–14% across all trace-length quintiles (Table 52), yet question accuracy decays monotonically from 47% (shortest quintile, median 620 tokens) to 0% (longest quintile, median 10,467 tokens), confirming that compound error—not per-token degradation—drives the collapse."

#### Table 50（Appendix W）：Compound error 理论 vs 实测

| Per-token acc (p) | T=500 | T=1,000 | T=3,000 |
|---|---:|---:|---:|
| 0.999 | 60.6% | 36.8% | 5.0% |
| 0.9995 | 77.9% | 60.6% | 22.3% |
| Required p for Q_Acc ≥ 50% | 99.861% | 99.931% | **99.977%** |

**对论文的意义：**
- 主文 Section 5.1 已引用，但最后一行（Required p）是最有冲击力的数据：要在 T=3,000 的 trace 上达到 50% Q_Acc，需要 per-token accuracy 达到 99.977%——远超任何当前方法
- 建议在 Abstract 或 Introduction 中引用这个数字增强 motivation

#### Figure 8（Appendix W）：Compound error 三面板可视化

三个 panel 展示：(a) cosine similarity 随 trace length 略降，(b) token-level accuracy 稳定在 ~99.5%，(c) Q_Acc 急剧下降。

**对论文的意义：**
- 这是论文中最直观的 compound error 可视化，比 Table 50 更易理解
- **强烈建议考虑前移到主文**（如果空间允许），或至少在主文增加明确引用
- 如果空间不够，可以只前移 panel (c)

---

### 2.2 Gate 性能上限与 deployment 可行性

#### Table 32（Appendix P）：V2d logistic regression——91% Q_Acc

| Variant | Strategy | Q_Acc | Savings | Early Exit |
|---|---|---:|---:|---:|
| V2a | Learned MLP gate (γ=0.99) | 63.0% | 6.28% | — |
| V2b | Layer gating (per-type threshold) | 13% | 41.06% | 75% |
| V2c | Per-type MLP gate | 64% | 18.14% | 52% |
| **V2d** | **Logistic regression (L2)** | **91%** | **3.75%** | **33%** |

**对论文的意义：**
- V2d 达到 91% Q_Acc（n=100 test set），远高于 V2a 的 63.0%
- 这证明：(1) compound error ceiling ≈91%，不是 100%——即使最保守的策略也有 ~9% irreducible error；(2) V2a 与 V2d 的 gap（63% vs 91%）说明 gate 还有改进空间
- V2d 的 savings 只有 3.75%——这是极端保守策略的代价
- **建议在主文 Section 5.2 讨论 V2a-V2d tradeoff**，展示 accuracy-savings frontier 上的不同操作点
- 但需注意 V2d 使用 single 400/100 split（n=100），不是 nested CV，所以数字可能偏乐观

#### Table 44（Appendix T.1）：Computational overhead

| Component | Cost |
|---|---|
| Affine transform (per layer) | ~7% of one layer |
| Gate MLP evaluation (per layer) | <0.1% of one layer |
| Avg. layers evaluated before exit | ~5 of 9 candidates |
| Total probe overhead | ~1% of full inference |
| Gross token savings (V2a) | 6.28% |
| **Net token savings** | **~5.3%** |
| **ROI (net savings / overhead)** | **~5:1** |

**对论文的意义：**
- Gate 系统总 overhead 只有 full inference 的 ~1%——这对 deployment 论证至关重要
- Net savings ~5.3% 虽然 modest，但 ROI 是 5:1
- **建议在主文 Section 5.2 或 Discussion 中引用**："The gate's computational overhead is approximately 1% of full inference (Appendix T.1), yielding a net savings ROI of approximately 5:1."

---

### 2.3 Cross-benchmark 泛化

#### Table 58（Appendix Z.1）：V2a MATH-500 → GSM8K transfer

| γ | GSM8K Q_Acc | GSM8K Savings | MATH-500 Q_Acc | MATH-500 Savings |
|---|---:|---:|---:|---:|
| 0.99 | 86.4±1.6% | 2.3±0.2% | 63.0±5.2% | 6.28% |
| 0.95 | 40.5±3.4% | 7.7±0.4% | — | — |
| 0.90 | 15.1±4.1% | 12.4±0.6% | — | — |

**GSM8K Q_Acc = 86.4%，远高于 MATH-500 的 63.0%。**

**对论文的意义：**
- GSM8K traces 更短（median ~85 tokens vs ~3,000），所以 compound error 积累更少
- 86.4% vs 63.0% 的差异**是 compound error 作为 binding constraint 的最强间接证据**
- Savings 从 6.28% 降至 2.3%——threshold calibration 是 benchmark-specific 的
- **建议在主文 Section 5.1 或 5.2 引用**："When the same MATH-500-trained gate is evaluated on GSM8K—where median trace length drops from ~3,000 to ~85 tokens—Q_Acc rises from 63.0% to 86.4% (Appendix Z), consistent with reduced compound-error accumulation on shorter traces."

---

### 2.4 Scale analysis 中的关键细节

#### Table 46（Appendix V）：1.5B reversal 详细数据

| Scale | H1 reasoning NM | H1 non-reas. NM | H1 direction | H2 reasoning exit acc | H2 non-reas. exit acc |
|---|---:|---:|---|---:|---:|
| 1.5B | 14.1% | 29.7% | **reversed** | 81.3% | 79.2% |
| 8B | 20.8% | 11.6% | ✓ | 32.4% | 47.7% |
| 14B | 25.5% | 11.6% | ✓ | 96.7% | 91.2% |

**对论文的意义：**
- 1.5B 的 reversal 非常清晰：non-reasoning 的 NM（29.7%）远高于 reasoning（14.1%），方向完全反转
- 但 H2 exit accuracy 差异很小（81.3% vs 79.2%）——说明 1.5B 的 reversal 主要在 oscillation，不在 representation immaturity
- **建议在主文 Limitations/Discussion 中更精确地描述**：reversal 主要影响 H1（oscillation），H2（exit accuracy）差异在 1.5B 上缩小到 marginal

#### Phi-4 penultimate layer effect（Appendix V, p.31）

86.4% 的 Phi-4 tokens 在 L38→L39 出现 cosine-similarity dip。这是 architecture-specific 的 representation shift，不出现在 Qwen/Llama 中。Running-max NM 包含 L38→L39 时达到 70.0%（vs trimmed 定义的 11.6%）。

**对论文的意义：**
- 解释了为什么 Phi-4 使用 trimmed NM 定义（排除 final column）
- 展示了 architecture-specific 的 convergence dynamics
- 如果空间允许，可以在 Discussion 中提及，说明不同架构有不同的 convergence challenge

---

### 2.5 Tuned lens 深度分析

#### Table 42（Appendix T）：Raw vs tuned exit accuracy

| Layer | R1-Qwen Raw (500q) | R1-Qwen Tuned (500q) | Qwen3 Non-reas Raw (500q) | Qwen3 Non-reas Tuned (500q) |
|---|---:|---:|---:|---:|
| L20 | 2.4% | 50.6% | — | — |
| L24 | 11.7% | 65.7% | 14.2% | 72.3% |
| L28 | 27.3% | 72.1% | — | — |
| L32 | 53.5% | 87.0% | — | — |

**对论文的意义：**
- Tuned lens 在所有层上都大幅提升 exit accuracy（+35–54pp）
- 关键：**reasoning-non-reasoning gap 在 tuned lens 下仍然持续**（L24–L32: 4–6pp），说明 representation immaturity 是 genuine property，不是 logit lens 的 projection artifact
- 已在主文 Section 4.2 引用，但可以在 Section 5.2 再次强调

#### Table 43（Appendix T）：Phi-4 14B tuned lens

| Layer | Raw | Tuned | Δ (pp) |
|---|---:|---:|---:|
| L5 | 6.9 | 75.3 | +68.4 |
| L25 | 26.2 | 83.8 | +57.7 |
| L35 | 65.3 | 91.0 | +25.7 |
| **L37** | — | **91.8** | — |
| L38 | 77.8 | 89.6 | +11.9 |

**对论文的意义：**
- Phi-4 的 peak tuned accuracy 在 L37（91.8%），不在 final layer L38（89.6%）——这是 non-monotonic convergence 在 tuned lens 层面的体现
- 与 Phi-4 的 penultimate layer representation shift 一致
- 可以在 Discussion 中用来说明：即使最好的 projection 也无法完全消除 architecture-specific 的 convergence challenges

---

## 三、综合建议：主文应新增的引用

以下是按优先级排序的零成本写作改动建议：

### P0 级别（强烈建议引用，直接增强核心论证）

| 数据 | 来源 | 建议引用位置 | 一句话说明为什么重要 |
|---|---|---|---|
| Trace-length Q_Acc decay: 47% → 0%，NM stable | Table 52 | Section 5.1 | 最直接的 compound error 实证——per-token 质量稳定但 trace-level 纯粹因长度崩溃 |
| GSM8K Q_Acc = 86.4% (vs MATH-500 63.0%) | Table 58 | Section 5.1 或 5.2 | 短 trace 上 compound error 显著降低，证明 p^T 是 binding constraint |
| Required p ≥ 99.977% for 50% Q_Acc at T=3,000 | Table 50 | Abstract 或 Introduction | 量化了 per-token accuracy bar，极具冲击力 |
| Positional NM U-shape: Q1=17.8%, Q3=15.3%, Q4=18.7% | Table 51 | Section 5.3（新增） | 为 segment-level exit 提供 position-based motivation |
| Native thinking oracle savings = 20.0% | Table 45 | Section 4.4 | Native thinking 有与 R1-Qwen 相当的 early-exit potential |
| Answer NM: thinking 10.0% vs R1-Qwen 32.7% | Table 45 | Section 4.4 | Distillation 对 answer segment oscillation 的影响是 3.3× |

### P1 级别（建议引用，增强辅助论证）

| 数据 | 来源 | 建议引用位置 | 一句话说明为什么重要 |
|---|---|---|---|
| V2d 91% Q_Acc at 3.75% savings | Table 32 | Section 5.2 | 展示 gate accuracy ceiling 和 accuracy-savings tradeoff |
| Gate overhead ~1%, ROI ~5:1 | Table 44 | Section 5.2 或 Discussion | Deployment feasibility 的具体数字 |
| Per-segment exit accuracy 差异: boxed 78.3% vs reasoning 61.4% at L32 | Table 41 | Section 5.3（新增） | 支持 segment-level exit 使用 segment-specific exit layers |
| Gate per-type exit rate: numerical 68.5% vs reas. KW 37.9% | Table 33 | Section 5.3（新增） | Gate 已自动学习 differentiated strategy，segment-level 是自然延伸 |
| 1.5B reversal 细节: H1 reversed, H2 近似 | Table 46 | Limitations | 更精确地描述 reversal 的 failure mode decomposition |
| Figure 8 compound error 可视化 | Figure 8 | Section 5.1 | 最直观的 compound error 图，值得前移 |

---

## 四、对剩余实验设计的具体建议

### 4.1 Segment-level exit 实验设计优化

基于 Table 41 和 Table 51 的数据，建议优化 segment-level exit 的设计：

```
优化策略 1: Position-aware segment exit
  - Q1 (0-25%): 使用 full depth 或保守 threshold（exploration phase，NM 高）
  - Q2-Q3 (25-75%): 使用标准 early exit（reasoning body，NM 最低）
  - Q4 (75-100%): answer segment 使用 full depth（NM spike + answer correctness 关键）

优化策略 2: Segment-type-aware exit
  - Reasoning segment: 可以更积极 early exit（per-type accuracy 相近）
  - Answer segment: 使用保守 threshold（answer correctness 极度敏感）
  - Boxed segment: 使用最保守策略（boxed accuracy 最重要）
```

### 4.2 Native thinking CALM+gate 实验的预期参照

基于 Table 45 的数据，可以设定以下预期：

| 指标 | 预期范围 | 依据 |
|---|---|---|
| CALM Q_Acc（cos, τ=0.99） | 0–5% | R1-Qwen 是 0%；native thinking exit accuracy 接近 |
| Tuned CALM Q_Acc（τ=0.99） | 20–35% | Oracle savings 20.0% ≈ R1-Qwen 19.2%；exit accuracy 34.7% ≈ 32.4% |
| V2a gate Q_Acc（γ=0.99） | 50–65% | 如果 gate 泛化到 native thinking，应接近 R1-Qwen 的 63.0% |
| V2a gate savings | 4–7% | Oracle savings 20.0% 给出上限；R1-Qwen 的 6.28% 是参照 |

如果实际结果落在这些范围内 → 完全消除 distillation confound。如果显著偏离 → 提供 distilled vs native 的有趣对比。

---

## 五、附录覆盖度总结

| 附录 | 内容 | 主文引用状态 | 建议 |
|---|---|---|---|
| A | Taxonomy robustness（keyword perturbation, boundary, clustering, stability） | 充分引用 | 无需改动 |
| B | Bootstrap CI | 充分引用 | 无需改动 |
| C | ANOVA details | 充分引用 | 无需改动 |
| D | Cross-correlation H1/H2 | 部分引用 | 可选引用 |
| E | Chain length analysis | 部分引用 | 建议引用 Table 52 的 Q_Acc decay |
| F | Per-type exit accuracy tables | 充分引用 | 无需改动 |
| G | Oracle savings details | 部分引用 | 无需改动 |
| H | Cross-model probe comparison | 部分引用 | 无需改动 |
| I | OOD truncation analysis | 充分引用 | 无需改动 |
| J | Simpson's paradox | 充分引用 | 无需改动 |
| K | Broader impact | 完整 | 无需改动 |
| L | Compute resources | 完整 | 无需改动 |
| M | Licenses | 完整 | 无需改动 |
| N | Sample sizes | 完整 | 无需改动 |
| O | Exit accuracy details | 部分引用 | 无需改动 |
| P | Exit strategy details（V2a-V2d, feature importance, gate architecture） | 部分引用 | **建议引用 V2d 91% 和 Table 44 overhead** |
| Q | OOD generalization（HumanEval） | 充分引用 | 无需改动 |
| R | Cross-difficulty validation（AIME, MATH-500 tuned lens） | 充分引用 | 无需改动 |
| S | Answer segment analysis | 部分引用 | **建议在 Section 5.3 引用 Table 41** |
| T | Tuned lens comparison + Phi-4 | 部分引用 | 可选引用 Table 43 |
| U | Native thinking full results | 部分引用 | **建议引用 oracle savings + answer NM** |
| V | Scale analysis（1.5B, 14B, 32B） | 充分引用 | 可选引用 Table 46 reversal 细节 |
| W | Trace-length + temporal analysis | 部分引用 | **建议引用 Table 51 + Table 52** |
| X | End-to-end exit details | 充分引用 | 无需改动 |
| Y | Logistic regression taxonomy validation | 充分引用 | 无需改动 |
| Z | GSM8K cross-benchmark | 部分引用 | **建议在主文引用 Q_Acc=86.4%** |
