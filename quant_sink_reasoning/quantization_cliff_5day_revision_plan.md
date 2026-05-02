# Quantization Cliff 论文 5 天饱和提升方案

> 目标：在 5 天内，利用 agent auto research 的快速写作/coding 能力，以及 2–4 张 RTX 5090 的持续实验资源，把论文从“现象清楚但 claim 偏强、quantizer confound 和 RE confound 容易被攻击”的状态，提升为一篇更完整的 **diagnostic + mechanism + mitigation** 型 NeurIPS 投稿。

---

## 0. 总体判断

当前论文已经有很强的核心现象：W4A16 基本安全，而 W3A16 GPTQ 在多个 7B–14B 模型上暴露 reasoning cliff，包括 collapse / cliff / mild 三档退化。但当前版本的主要风险不是“实验不够多”，而是：

1. **主 claim 偏强**：容易被理解成 “W3A16 普遍导致 reasoning cliff”。
2. **quantizer confound 明显**：主结果是 W4A16 AWQ vs W3A16 GPTQ，而 DeepSeek 上 HQQ W3 明显优于 GPTQ。
3. **RE 指标可能被质疑**：GSM8K 是 generation，MMLU 是 log-likelihood，可能混入 evaluation-format confound。
4. **appendix 强结果没有充分前移**：within-MMLU、error taxonomy、group-size mitigation、degenerate examples、perplexity blind spot 等已经有，但正文没有充分利用。
5. **机制线需要克制**：length correlation、mid-layer sensitivity、perturbation replay 都是 converging evidence，但还不是完整 causal proof。

因此 5 天的核心策略是：

> **先把 appendix 里已有的高价值结果前移并重组，再用 GPU 时间补齐 reviewer 最可能攻击的 4 个缺口：quantizer dependence、collapse artifact、mechanistic baseline、rescue/mitigation。**

最终论文应从：

> “W3 GPTQ makes reasoning worse.”

升级为：

> “The W4-to-W3 boundary is a reasoning-stability boundary. GPTQ exposes severe failures, but method, granularity, and layer protection can modulate or rescue them. Aggregate metrics miss the instability; reasoning-safe deployment needs dedicated diagnostics.”

---

## 1. 最终主 claim

建议全文主 claim 改为：

> **W3A16 is a high-risk precision regime for LLM reasoning. GPTQ exposes severe cliffs in several model families, but the realized severity is strongly modulated by quantization algorithm, grouping granularity, and architecture. Standard aggregate metrics such as perplexity and MMLU can miss reasoning-trajectory instability; reasoning-safe quantization requires RE, flip rate, degeneration audit, and long-chain evaluation.**

对应中文理解：

> W3A16 不是一定不可用，而是 reasoning 的高风险区域。GPTQ 会在若干模型家族上暴露严重 cliff，但实际严重程度受到 quantizer、group size 和模型家族共同调节。perplexity 或平均 MMLU 不足以判断 reasoning 是否安全，必须看 RE、flip rate、退化模式和长链推理评估。

---

## 2. 五天工作的总优先级

### P0：不需要新 GPU，但必须立刻做

这些任务 ROI 最高，主要是写作和 appendix 前移。

| 任务 | 目的 | 是否已有结果 |
|---|---|---|
| 把 within-MMLU analysis 前移正文 | 防 RE 被质疑为 evaluation-format artifact | 已有 |
| 把 DeepSeek error taxonomy 前移正文 | 证明 W3 损坏 reasoning trajectory，不只是算错 | 已有 |
| 把 degenerate examples 总结成 failure-mode table | 解释 collapse 到 0% 到底是什么坏法 | 已有 |
| 把 group-size mitigation 前移正文 | 说明 cliff 可被 granularity 缓解 | 已有 |
| 把 perplexity-vs-reasoning 前移 Discussion | 强化部署意义 | 已有 |
| 精简 sink/profile 主文占比 | 避免主线发散 | 已有 profile-Fisher 不显著 |
| 全文 claim audit | 防 reviewer 抓过强结论 | 写作任务 |

### P1：必须新跑，高 ROI

| 实验 | 目的 |
|---|---|
| Qwen2.5-14B RTN W3A16 | 检验 cliff 模型是否非 GPTQ 下仍 cliff |
| Phi-3 RTN W3A16 或 HQQ W3A16 | 第二个 cliff 模型 cross-method，使 quantizer dependence 更稳 |
| Llama-3.1-8B / Qwen3-8B collapse sanity check：RTN 或 GPTQ gs64 | 检查 0% collapse 是否 GPTQ-specific / group-size artifact |
| DeepSeek matched-protocol cross-method | 解决 0-shot vs 8-shot protocol mismatch |
| Qwen2.5-14B mixed-precision rescue | 把 layer sensitivity / group-size mitigation 变成 intervention 证据 |
| Perturbation replay shuffled baselines | 补强 “error structure matters” 机制 claim |

### P2：有余力继续跑

| 实验 | 目的 |
|---|---|
| Calibration sensitivity：WikiText2 vs C4 vs math calibration | 排除 calibration artifact |
| AIME / GPQA-mini / GSM-hard subset | 提供更难 reasoning stress test |
| RTN/HQQ on Gemma/Qwen2.5-7B matched protocol | 完整 mild control |
| Mixed precision on DeepSeek | 第二个 rescue 模型 |
| Answer-extraction robustness audit | 排除 extraction failure |

---

## 3. Day 0 / Day 1 上午：统一实验 harness

5 天饱和工作最怕每个 agent 跑出不同 protocol，导致最后不可比。因此第一步必须统一 harness。

### 3.1 默认实验配置

| 项目 | 默认设置 |
|---|---|
| Benchmark | GSM8K N=500 |
| Decoding | greedy |
| max_new_tokens | 4096；DeepSeek long-CoT 可额外 8192 |
| Prompt | 与主表一致，0-shot native chat template |
| Metrics | accuracy, flip rate, truncation rate, mean length, no-answer rate, repetition rate |
| 保存内容 | full output, extracted answer, FP16 correctness, quantized correctness |
| 每次实验产物 | JSONL + CSV summary + LaTeX table row |

### 3.2 每个实验必须自动输出的指标

1. Accuracy
2. Δ vs matched FP16
3. Flip rate
4. Truncation rate
5. Answer extraction success rate
6. No numeric answer rate
7. Repeated 4-gram ratio
8. Unique token ratio
9. Mean / median generation length
10. 20 个错误样例

### 3.3 为什么必须先统一 harness

后续所有实验都要服务于同一个 reviewer 防御链：

> 不是 prompt artifact，不是 answer extraction artifact，不是 token limit artifact，不是 single quantizer artifact。

统一 harness 后，所有新增表格都可以直接进入论文正文或 appendix。

---

## 4. Day 1：Appendix 前移 + Qwen2.5-14B RTN 启动

### 4.1 写作任务 A：重写 Abstract / Introduction / Contribution

#### Abstract 需要收紧

原先类似：

> reducing to W3A16 triggers a quantization cliff

建议改成：

> Weight-only W4A16 generally preserves reasoning accuracy, but moving to W3A16 enters a high-risk regime where GPTQ can expose sharp reasoning cliffs. Across seven 7B–14B models, GPTQ W3A16 produces collapse, cliff, and mild regimes. The severity is not determined by bit-width alone: cross-method and group-size analyses show strong dependence on quantization algorithm and granularity.

#### Contribution 建议改成 5 点

1. **W4-to-W3 reasoning cliff characterization**：系统刻画 GPTQ W3A16 下 collapse / cliff / mild 三档。
2. **Reasoning-specific diagnostics**：RE + within-MMLU + HumanEval 共同支持 reasoning-specific degradation。
3. **Failure-mode taxonomy**：解释 collapse 不是单一现象，而是多种 reasoning trajectory instability。
4. **Method / group-size / rescue analysis**：说明 3-bit 是 high-risk regime，严重程度受 quantizer 和 granularity 调节。
5. **Perturbation replay + layer sensitivity**：提供 converging mechanistic evidence，但不宣称完整 causal proof。

---

### 4.2 写作任务 B：within-MMLU 前移正文

放在 RE 主结果之后，新增小节：

#### Ruling out evaluation-format confounding

核心写法：

> RE compares GSM8K generation against MMLU log-likelihood, which could confound capability with evaluation format. We therefore repeat the analysis within MMLU itself by partitioning subjects into reasoning-intensive and knowledge-intensive groups. Because both groups use the same log-likelihood scoring protocol, this directly tests whether reasoning-heavy knowledge tasks are more quantization-sensitive.

建议正文加入一个小表：

| Model | GSM8K/MMLU RE | within-MMLU RE | Interpretation |
|---|---:|---:|---|
| Qwen2.5-14B | 4.43× | 2.48× | reasoning-specific confirmed |
| Phi-3 | 2.57× | 2.76× | confirmed |
| Qwen2.5-7B | 1.26× | 1.71× | stronger within MMLU |
| Gemma-2 | 0.65× | 1.43× | global inversion resolved |
| DeepSeek | 2.51× | 0.99× | mixed |
| Llama | 5.47× | 0.96× | collapse/generation-driven |

这一表格非常关键，尤其是 Gemma-2：原始 RE < 1，但 within-MMLU RE > 1，说明 “Gemma 的全局 MMLU 掉分更多” 并不否定 reasoning-heavy subject 更脆弱。

---

### 4.3 写作任务 C：Failure-mode taxonomy 前移

新增 section：

#### Collapse Is Not Monolithic

主文建议放两个表。

##### Table A：Quantized collapse pathways

| Model | W3 symptom | Truncation | Interpretation |
|---|---|---:|---|
| Qwen3-8B | punctuation / exclamation loop | high / 100% if observed | generation-level collapse |
| Llama-3.1-8B | cross-lingual token loop | high | token-subspace instability |
| DeepSeek-R1-7B | self-doubt cascade: “wait, no...” | high | self-correction instability |
| Gemma-2-9B | normal-length but wrong reasoning | low | coherent reasoning degradation |

##### Table B：DeepSeek error taxonomy

将 appendix error taxonomy 压缩成主文表：

| Error type | FP16 | W3A16 | Change |
|---|---:|---:|---:|
| Repetitive loop | low | very high | ↑ |
| Hit token limit | low | high | ↑ |
| Answer corrupted by loop | low | high | ↑ |
| Self-contradiction | low | higher | ↑ |
| Arithmetic error | higher | lower | ↓ |

重点解释：

> W3A16 does not primarily increase local arithmetic mistakes. It destabilizes the reasoning trajectory before normal arithmetic failure occurs.

---

### 4.4 新实验任务 1：Qwen2.5-14B RTN W3A16

这是 Day 1 必须启动的最高优先级新实验。

#### 配置

| 项目 | 设置 |
|---|---|
| Model | Qwen2.5-14B |
| Quantization | RTN W3A16, group size 128 |
| Benchmark | GSM8K N=500 |
| Protocol | 0-shot, same as main table |
| Metrics | accuracy, flip rate, truncation, repetition, mean length |

#### 为什么是 Qwen2.5-14B

它是 cliff 模型，不是 collapse 模型，也不是 mild 模型。它最适合回答：

> cliff 是否仅仅是 GPTQ artifact？

#### 结果解释模板

| 结果 | 论文怎么写 |
|---|---|
| RTN 仍掉 15–25 pp | cliff is not GPTQ-exclusive; severity method-modulated |
| RTN 只掉 5–10 pp | Qwen2.5-14B cliff is largely GPTQ-induced; W3 is vulnerability regime |
| RTN 比 GPTQ 更差 | naive rounding also destabilizes reasoning; GPTQ not uniquely harmful |
| RTN 接近 FP16 | quantizer choice can rescue cliff; strengthen method-dependence story |

无论哪种结果，都比没有结果好。

---

## 5. Day 2：扩展 cross-method + collapse sanity check

Day 2 的目标是把 “method dependence” 从 DeepSeek 特例，变成跨模型证据。

### 5.1 新实验任务 2：Phi-3 RTN / HQQ W3A16

#### 优先顺序

1. Phi-3 RTN W3A16
2. 如果 HQQ 支持，再跑 Phi-3 HQQ W3A16

#### 为什么跑 Phi-3

Phi-3 是另一个 cliff-regime 模型，且和 Qwen2.5-14B 属于不同 family。它能帮助回答：

> method dependence 是否跨架构稳定？

#### 最小结果表

| Model | Method | GSM8K | Δ | Flip | Trunc |
|---|---|---:|---:|---:|---:|
| Qwen2.5-14B | GPTQ | existing | existing | existing | existing |
| Qwen2.5-14B | RTN | new | new | new | new |
| Phi-3 | GPTQ | existing | existing | existing | existing |
| Phi-3 | RTN | new | new | new | new |
| DeepSeek | GPTQ/RTN/HQQ | existing | existing | existing | existing |
| Gemma | GPTQ/RTN | existing | existing | existing | existing |

这张表能成为新论文里的核心表之一。

---

### 5.2 新实验任务 3：collapse sanity check

目标不是完整展开 collapse cross-method，而是做最小 artifact check。

#### 优先模型

1. Qwen3-8B
2. Llama-3.1-8B

#### 优先方法

1. GPTQ gs=64
2. RTN W3A16
3. HQQ，如果方便

#### Benchmark

GSM8K N=200 先跑 sanity check；如果结果重要，再扩到 N=500。

#### 为什么 N=200 可以接受

collapse 模型当前是 0%。如果 N=200 下 RTN/gs64 能从 0% 恢复到 30%+，信号已经非常强；如果仍然 0–2%，也说明 collapse 很鲁棒。后续再决定是否扩到 N=500。

#### 结果解释

| 结果 | 意义 |
|---|---|
| gs64 救回 | collapse 与 group granularity 相关 |
| RTN 救回 | GPTQ-specific collapse |
| RTN 仍 collapse | 3-bit precision 本身对该模型高风险 |
| HQQ 救回 | advanced quantizer can avoid collapse |
| Llama/Qwen3 表现不同 | collapse mechanisms are architecture-specific |

这个实验非常适合和 failure-mode taxonomy 结合。

---

### 5.3 写作任务：cross-method 变成主线

新增或重写 section：

#### Method and Granularity Modulate the Cliff

这节的核心不是削弱论文，而是让结论更可信：

> The W3 boundary is not a universal law of bit-width alone. It is a vulnerability regime whose realized severity depends on the quantizer and grouping granularity.

建议结构：

1. DeepSeek：GPTQ / RTN / HQQ 跨度巨大。
2. Gemma：GPTQ / RTN 都 mild。
3. Qwen2.5-14B / Phi-3：新增 RTN 结果。
4. Llama/Qwen3：collapse sanity check。
5. Group-size mitigation：gs64 能救部分 severe 模型。

这样 reviewer 很难再说作者忽略 quantizer confound。

---

## 6. Day 3：mixed-precision rescue + matched protocol

Day 3 的目标是从 diagnosis 进入 mitigation。

### 6.1 新实验任务 4：Qwen2.5-14B mixed-precision rescue

这是高 ROI 实验。它可以把 layer sensitivity 从观察变成干预。

#### 为什么选 Qwen2.5-14B

- 是 cliff 模型；
- 不像 DeepSeek 那样 long-CoT/truncation 特别复杂；
- 不像 collapse 模型那样可能有 generation collapse confound；
- 已有 group-size mitigation；
- 结果容易解释。

#### 配置

假设模型有 N 层，做 4 个版本：

| 配置 | 含义 |
|---|---|
| all W3 | baseline |
| front W4, rest W3 | 保护前 1/3 层 |
| middle W4, rest W3 | 保护中间 1/3 层 |
| back W4, rest W3 | 保护后 1/3 层 |
| all W4 | upper bound，已有或可用 |

如果实现成本高，可以用 fake quant / layer-wise load mixed checkpoint；不需要追求生产级推理效率，只要 accuracy diagnostic。

#### Benchmark

GSM8K N=500；必要时 N=300 先出结果。

#### 结果价值

| 结果 | 解释 |
|---|---|
| middle W4 recovery 最大 | 和 layer replay 的 mid-layer vulnerability 形成 observation → intervention 闭环 |
| front/back 也有恢复 | rescue 不是单点层，而是 distributed precision budget |
| mixed precision 不明显恢复 | group-size mitigation 可能比 layer-level bit allocation 更关键，作为 negative result 放 appendix |

#### 主文怎么写

新增小节：

#### Targeted Precision Rescue

核心表：

| Config | Bits protected | GSM8K | Recovery | Flip | Trunc |
|---|---|---:|---:|---:|---:|

这会让论文从 “W3 有风险” 变成 “我们知道一些高 ROI mitigation axis”。

---

### 6.2 新实验任务 5：DeepSeek matched-protocol cross-method

这个实验用于解决 protocol mismatch。

#### 当前问题

DeepSeek cross-method 可能使用 8-shot，而主结果是 0-shot；这会给 reviewer 留攻击空间。

#### 目标

用同一协议跑：

| Model | Method |
|---|---|
| DeepSeek-R1-7B | FP16 |
| DeepSeek-R1-7B | GPTQ W3 |
| DeepSeek-R1-7B | RTN W3 |
| DeepSeek-R1-7B | HQQ W3 |

Protocol：0-shot, GSM8K N=500, 4096 tokens。

#### 如果时间不够

只补 RTN/HQQ 在主协议下的 accuracy + truncation，不必重跑 FP16/GPTQ，如果已有 matched baseline 可复用。

#### 论文价值

它会让 cross-method section 更干净：

> Under a matched 0-shot protocol, method choice still changes W3 reasoning degradation by X pp.

如果 matched result 和原 8-shot 一致，就很好；如果不一致，也说明 few-shot protocol interacts with quantization，这同样有价值。

---

## 7. Day 4：perturbation shuffled baselines + calibration sensitivity

Day 4 的目标是机制补强和 artifact 排除。

### 7.1 新实验任务 6：perturbation replay shuffled baselines

当前 structured vs Gaussian 已经不错，但 reviewer 可能说 Gaussian 太 naive。饱和版应该补这个。

#### 选模型

优先两个：

1. Gemma-2-9B：mild 模型，稳定、低 truncation，机制结果干净；
2. Phi-3：cliff 模型，structured replay 已能复现 cliff。

DeepSeek 可选，因为 long-CoT/truncation 太复杂，variance 大。

#### Baselines

| Perturbation | 保留什么 | 破坏什么 |
|---|---|---|
| structured ΔW | 真实量化误差结构 | 无 |
| block-shuffled ΔW | 局部 group 分布 | 全局 alignment |
| row-shuffled ΔW | row/channel-level 向量 | output channel assignment |
| column-shuffled ΔW | input-channel 分布 | input alignment |
| element-shuffled ΔW | 值分布 | 空间结构 |
| sign-randomized ΔW | magnitude | sign/direction |
| Gaussian | per-layer std | 全部结构 |

#### Benchmark

GSM8K N=300；如果结果清楚，不一定扩 N=500。

#### 最理想结果

Damage 随结构破坏程度单调增加：

```text
structured < block-shuffled < row/column < element/sign < Gaussian
```

#### 如果结果不单调

也有价值。可以写：

> Structure matters, but the relevant structure is not captured by simple row/column alignment alone; Gaussian remains a destructive upper baseline.

这比完全没有 shuffled baseline 更稳。

#### 主文位置

Mechanistic section 里，structured-vs-Gaussian 后面加：

> To test whether this is merely because Gaussian noise is distributionally unnatural, we progressively destroy the structure of ΔW while preserving different marginal statistics.

这能显著增强机制实验质量。

---

### 7.2 新实验任务 7：calibration sensitivity

这个实验是 artifact check，不一定进正文主表，但能放 appendix 或一句话 main text。

#### 模型

优先：

1. Qwen2.5-14B
2. DeepSeek
3. Gemma，可选

#### Calibration sets

| Calibration | 意义 |
|---|---|
| WikiText-2 | 当前默认 |
| C4 subset | general text control |
| GSM8K train / math prompts | reasoning-domain calibration |
| Pile subset | 更接近原文设置或 general corpus |

#### Benchmark

GSM8K N=300 即可。

#### 可能结论

| 结果 | 解释 |
|---|---|
| math calibration 救回 | calibration distribution is mitigation axis |
| C4/WikiText 差不多 | cliff robust to calibration corpus |
| calibration 影响很大 | current GPTQ result should be qualified |
| calibration 影响小 | stronger claim |

#### 论文价值

它能防 reviewer 问：

> 你是不是用了不合适的 calibration data，导致 GPTQ W3 异常差？

---

## 8. Day 5：整合、重写、制表、claim audit

Day 5 不再开新实验，除非前四天失败。主要做论文整合。

### 8.1 新主文结构

建议最终结构改成：

#### 1 Introduction

- W4 common deployment；
- W3 high-risk boundary；
- claim 收紧：GPTQ exposes, severity method-dependent；
- contributions。

#### 2 Related Work

增强：

- quantization hurts reasoning；
- signal degradation vs computation collapse；
- QAT / recovery；
- 本文独特点：RE + flip + perturbation replay + failure taxonomy。

#### 3 Methodology

- 模型；
- quantization；
- evaluation；
- RE；
- flip；
- failure taxonomy；
- perturbation replay；
- protocol caveats 前置。

#### 4 W4-to-W3 Cliff

主表 + 主图。

#### 5 Reasoning-Specific Degradation

- RE；
- within-MMLU；
- HumanEval auxiliary；
- ARC optional。

#### 6 Reasoning-Trajectory Instability

- flip rate；
- MATH difficulty；
- error taxonomy；
- collapse pathways。

#### 7 Method, Granularity, and Rescue

- cross-method；
- Qwen2.5-14B RTN；
- Phi RTN/HQQ；
- collapse sanity；
- group-size mitigation；
- mixed precision rescue。

#### 8 Perturbation Mechanism

- structured vs Gaussian；
- shuffled baselines；
- dose response；
- layer localization；
- 机制写成 converging evidence。

#### 9 Discussion

- W3 is vulnerability regime；
- perplexity is insufficient；
- deployment checklist；
- limitations。

---

### 8.2 必须重画 / 新增的表和图

#### Main Figure 1：原 W4-to-W3 cliff

保留。

#### Main Table 1：主 GSM8K table

保留，但 caption 写清楚 W3 是 GPTQ main setting。

#### Main Table 2：RE + within-MMLU RE

新增/前移。

#### Main Table 3：failure taxonomy

新增/前移。

#### Main Figure 2：MATH difficulty scaling

建议图形化。

#### Main Table 4：cross-method + group-size

核心新表。

#### Main Table 5：mixed-precision rescue

如果实验完成，放正文；否则 appendix。

#### Main Figure 3：perturbation structure baseline

如果完成，放正文；否则保留原 dose-response。

#### Main Figure 4：perplexity blind spot

放 discussion 或 appendix，但正文必须引用。

---

### 8.3 Claim audit 清单

全文逐个替换：

| 原表述 | 改成 |
|---|---|
| W3A16 triggers a cliff | GPTQ W3A16 can expose a cliff |
| W3 is dangerous | W3 is a high-risk regime for reasoning workloads |
| architecture-determined | model-family-dependent and method-modulated |
| token length drives vulnerability | generation length is the most consistent observable predictor |
| error structure determines damage | error structure modulates damage beyond magnitude |
| W4 is safe | W4 appears safe under evaluated settings |
| mechanism proves | evidence converges on |

---

## 9. GPU 饱和调度建议

### 9.1 如果有 4 张 RTX 5090

#### GPU 0：Qwen2.5-14B lane

- Day 1：Qwen2.5-14B RTN W3；
- Day 2：Qwen2.5-14B HQQ 或 calibration；
- Day 3：Qwen2.5-14B mixed precision；
- Day 4：calibration / rerun failed configs。

#### GPU 1：Phi-3 lane

- Day 1–2：Phi-3 RTN/HQQ；
- Day 3：Phi mixed precision optional；
- Day 4：Phi perturbation shuffled。

#### GPU 2：collapse lane

- Day 1：Qwen3 GPTQ gs64 / RTN N=200；
- Day 2：Qwen3 HQQ or N=500 expansion；
- Day 3：Llama GPTQ gs64 / RTN N=200；
- Day 4：Llama expansion or failure taxonomy outputs。

#### GPU 3：mechanism lane

- Day 1：prepare perturbation replay scripts；
- Day 2：Gemma shuffled baselines；
- Day 3：Phi shuffled baselines；
- Day 4：DeepSeek matched protocol or calibration；
- Day 5：rerun / ablation backup。

### 9.2 如果只有 2 张 RTX 5090

- GPU 0：Qwen2.5-14B + Phi cross-method；
- GPU 1：collapse sanity + perturbation shuffled；
- mixed precision 和 calibration 视前两天结果决定。

---

## 10. 实验失败时的 fallback

### 10.1 如果 RTN/HQQ 框架不稳定

立即切换到：

- GPTQ group-size sweep：gs=128, 64, 32；
- 或者只做 RTN fake quant；
- 目的仍然是 method/granularity dependence。

### 10.2 如果 mixed precision 工程复杂

改成 layer-wise perturbation + selective dequantization simulation：

- 不做真正 checkpoint；
- 用 forward hook 恢复部分 layer 的 FP16 weights；
- 看 GSM8K recovery；
- 这也能作为 diagnostic rescue。

### 10.3 如果 shuffled replay 太慢

只做两个 baseline：

1. structured ΔW；
2. element-shuffled ΔW；
3. Gaussian。

三者已经足以说明 “value distribution alone is not enough”。

### 10.4 如果 collapse sanity 结果混乱

不要硬写主文，只放 appendix：

> Collapse models are highly method-sensitive; full diagnosis left to future work.

---

## 11. 最低 / 标准 / 冲刺三档交付

### 11.1 最低交付：保证 5 天内一定完成

1. Appendix 前移：within-MMLU、error taxonomy、group-size、failure examples；
2. Abstract/Intro/Conclusion claim 收紧；
3. Qwen2.5-14B RTN W3；
4. Protocol caveat 前置；
5. Cross-method 主线化。

这已经能显著提升论文。

### 11.2 标准交付：推荐目标

1. 最低交付全部完成；
2. Phi-3 RTN/HQQ；
3. Qwen3/Llama collapse sanity check；
4. DeepSeek matched-protocol cross-method；
5. MATH difficulty 图形化；
6. Perplexity blind spot 放 discussion。

这个版本会比较稳。

### 11.3 冲刺交付：4 卡饱和理想状态

1. 标准交付全部完成；
2. Qwen2.5-14B mixed-precision rescue；
3. Gemma/Phi shuffled perturbation replay；
4. calibration sensitivity；
5. deployment checklist；
6. appendix reproducibility 完整化。

这个版本可以明显把论文从 weak accept 边缘推向更强的 accept 叙事。

---

## 12. 最终优先级排序

### 必做，不需要新实验

1. **Within-MMLU analysis 前移**
2. **DeepSeek error taxonomy 前移**
3. **Degenerate collapse examples 前移**
4. **Group-size mitigation 前移**
5. **MATH difficulty stratification 图形化**
6. **Perplexity blind spot 进入 discussion**
7. **全文 claim audit**
8. **Protocol mismatch 在正文和 caption 前置说明**
9. **Sink/profile 降级，保留为 supporting evidence**

### 必做，需要新实验

10. **Qwen2.5-14B RTN W3A16**
11. **Phi-3 RTN/HQQ W3A16**
12. **Qwen3/Llama collapse sanity check**
13. **DeepSeek matched-protocol cross-method**

### 高 ROI 可选

14. **Qwen2.5-14B mixed-precision rescue**
15. **Gemma/Phi perturbation shuffled baselines**
16. **Calibration sensitivity**
17. **Deployment checklist 量化化**

---

## 13. Reviewer concern → evidence 对照表

| Reviewer 质疑 | 对应补强 |
|---|---|
| W3 cliff 是不是 GPTQ artifact？ | Qwen2.5-14B RTN、Phi RTN/HQQ、DeepSeek matched cross-method |
| W4 AWQ vs W3 GPTQ 是不是 method confound？ | GPTQ W4 control + cross-method 主线化 + claim 收紧 |
| RE 是不是 GSM8K generation vs MMLU log-likelihood 的格式差异？ | within-MMLU subject analysis 前移正文 |
| Collapse 到 0% 是不是 answer extraction / prompt bug？ | collapse sanity check + failure-mode examples + no-answer/repetition/truncation metrics |
| W3 只是算错更多吗？ | DeepSeek error taxonomy：loop/truncation/self-contradiction 增加，arithmetic error 下降 |
| 机制是不是过度解释？ | 改成 converging evidence：flip length、perturbation replay、layer sensitivity |
| Gaussian baseline 是否太 naive？ | shuffled perturbation baselines |
| 有没有 mitigation？ | group-size mitigation 前移 + mixed-precision rescue |
| Perplexity 是否足够判断量化模型？ | perplexity-vs-reasoning blind spot 进入 Discussion |
| Protocol mismatch 是否影响结论？ | matched-protocol DeepSeek + caption 前置说明 |

---

## 14. 建议最终 Abstract 草稿

```text
Weight-only 4-bit quantization (W4A16) generally preserves LLM accuracy, but we show that moving to 3-bit precision enters a high-risk regime for multi-step reasoning. Across seven 7B–14B open-weight models from five families, GPTQ W3A16 exposes a three-tier degradation spectrum, ranging from mild loss to severe cliffs and complete generation collapse. The damage is not captured by aggregate metrics alone: Reasoning Excess, within-MMLU subject analysis, and per-sample flip rates show that reasoning-heavy tasks degrade disproportionately relative to knowledge-oriented controls. Collapse is not monolithic: models fail through punctuation loops, cross-lingual token collapse, self-doubt cascades, or coherent-but-wrong reasoning. Cross-method and group-size experiments show that W3 severity is strongly modulated by quantization algorithm and granularity, making W3A16 a vulnerability regime rather than an inevitable failure regime. Finally, perturbation replay and layer-level analysis provide converging evidence that error structure and intermediate-layer sensitivity shape reasoning degradation. These findings suggest that reasoning-safe quantized deployment requires diagnostics beyond perplexity, including RE, flip rate, degeneration audit, and long-chain evaluation.
```

---

## 15. 建议最终 Contributions 草稿

```text
Our contributions are:

1. We systematically characterize the W4-to-W3 reasoning cliff across seven 7B–14B models, revealing collapse, cliff, and mild regimes under GPTQ W3A16.

2. We introduce and validate reasoning-specific diagnostics, including Reasoning Excess, within-MMLU subject analysis, HumanEval RE, and per-sample flip rate, showing that reasoning-heavy tasks degrade disproportionately.

3. We provide a failure-mode taxonomy of quantized reasoning collapse, distinguishing punctuation loops, cross-lingual token collapse, self-doubt cascades, truncation-driven failure, and coherent-but-wrong reasoning.

4. We show that W3A16 is a method- and granularity-modulated vulnerability regime: RTN/HQQ comparisons, GPTQ group-size mitigation, and targeted precision rescue substantially alter cliff severity.

5. We provide converging mechanistic evidence via structured perturbation replay, shuffled-error controls, dose-response analysis, and layer-level sensitivity, showing that error structure and intermediate-layer vulnerability modulate reasoning damage beyond magnitude alone.
```

---

## 16. 最终判断

如果 5 天内只做最小修改，论文会从 borderline / weak accept 边缘变得更稳。  
如果按照本饱和方案执行，尤其完成：

1. Qwen2.5-14B RTN；
2. Phi RTN/HQQ；
3. collapse sanity check；
4. mixed-precision rescue；
5. shuffled perturbation replay；
6. appendix 前移 + claim audit；

那么论文会从单纯的 “W3 GPTQ 退化 benchmark paper” 变成一篇更完整的：

> **reasoning stability boundary + diagnostic framework + failure taxonomy + mitigation analysis**

这会显著增强 NeurIPS reviewer 对 novelty、soundness、significance 和 reproducibility 的信心。

