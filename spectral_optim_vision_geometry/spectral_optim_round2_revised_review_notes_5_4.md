# Spectral Optimization Paper: Review Notes

## Part 1. 论文主体内容

### 1.1 核心结论

这篇文章的核心主张是：

> 在作者设定的 controlled、from-scratch、short-to-medium schedule 视觉训练范式下，显式利用权重矩阵谱结构的优化器家族（Muon / SOAP / Shampoo）比 AdamW 和 SGD 具有更强的跨架构鲁棒性，而且这种优势会随着架构类型而显著变化。

作者想说明的主要是三件事：

1. 优化器表现不是纯超参数问题，而是和架构设计深度耦合。
2. 谱优化器的优势不是 Muon 一个方法的偶然，而是 spectral family 的共同性质。
3. 这种优势和层级谱结构有关，不同架构中甚至呈现出相反的 layer-wise 相关性，因此背后可能存在不同机制。

### 1.2 主要证据

论文的证据链大致如下：

1. 在 3 个数据集、3 个架构的 20-epoch 对比里，Muon 全面优于 AdamW，差距约为 +5.9 到 +24.3 个点。
2. 在 CIFAR-100 上再加入 SOAP 和 Shampoo 后，三种 spectral methods 都明显优于 AdamW，且 family 内部差距远小于它们相对 AdamW 的优势。
3. 20 epoch 扩展到 50 epoch 后，ResNet-50 和 ConvNeXt-T 上的优势会缩小但不会消失，而 DeiT-S 上几乎不缩小，作者据此区分“收敛速度优势”和“解质量优势”。
4. SGD 在 layer-normalized 架构上几乎失效，而在 ResNet-50 上还算可用，作者据此强调 normalization / architecture 对 optimizer viability 的决定性作用。
5. 通过 layer-wise weight swap，作者声称 Muon 在 ResNet-50 上几乎所有层都是 beneficial，在 DeiT-S 上则更 selective。
6. 通过 spectral response ratio 与 per-layer benefit 的 Spearman correlation，作者给出 CNN 为正、transformer 为负的 sign reversal，并据此提出“不同架构上谱优化器通过不同机制起作用”的解释。

### 1.3 有价值的地方

这篇文章真正有价值的点，不只是“某个新优化器在某些表上更强”，而是它抓住了一个更值得讨论的问题：

> optimizer selection 在现代视觉模型里，可能首先是一个 architecture-coupled structural decision，而不是一个靠统一 recipe 就能抹平的次要训练细节。

如果这个故事成立，它的意义会比单纯报新 SOTA 或报一个 optimizer win-rate 更大，因为它把“优化器表现为何跨 backbone 不稳定”这件事从经验现象往结构性解释推进了一步。

### 1.4 文章的 story telling 和主线

这篇文章当前的主线是比较清楚的：

1. 先用 SGD 在 ResNet-50 和 ConvNeXt-T 上的巨大反差，建立 optimizer-architecture coupling 的问题意识。
2. 再用 Muon vs AdamW 的 3x3 结果说明，谱优化器具有更稳定的跨架构转移性。
3. 然后把 SOAP / Shampoo 拉进来，试图把结论从“Muon 有效”升级到“spectral family 有效”。
4. 接着用 20 vs 50 epoch 对比说明，这种优势有时是更快到达高点，有时是到达了更好的解。
5. 最后用 weight swap 和谱相关分析，把全文从 benchmark comparison 往 mechanism paper 推一步。

前半段是强的，尤其是：

- 主结果矩阵整齐；
- gap 幅度大；
- spectral family comparison 至少部分支持“不是 Muon 偶然”；
- 50-epoch 补充实验帮助回应“只是收敛快一点吗”这个自然质疑。

但后半段要明显更谨慎：

- “主要由 normalization 决定”这个解释目前还没有被干净隔离出来；
- sign reversal 很有趣，但离“机制成立”还有距离；
- 从有限受控实验直接推到“layer norm 架构默认应使用 spectral optimizer”还偏强。

所以更稳的写法应该是：

> 在本文的受控实验设定下，spectral optimizers 展现出一致而显著的跨架构优势，并且这种优势与架构相关的层级谱特征有关；但具体的因果机制和最关键的架构决定因素，还需要更干净的消融与更强的 baseline 才能完全坐实。

## Part 2. 当前版本的优点和问题

### 2.1 优点

#### 1. 主结果很醒目，而且不是单次偶然

Muon 相对 AdamW 的差距在 9 个配置上全为正，而且不少配置是双位数大幅领先。这种结果在 optimizer paper 里是有吸引力的，不是那种只能靠统计显著性硬撑的小增益。

#### 2. 文章不是只停留在换个 optimizer 分数更高

作者尝试回答三个更深的问题：这种现象是否跨架构稳定、是否只是收敛速度、以及优势主要来自哪些层。这比很多只报最终 accuracy 的优化器论文更有研究味道。

#### 3. spectral family 的视角独特

把 SOAP 和 Shampoo 纳入比较非常重要，因为这让文章有机会把 claim 从“Muon 有效”提升到“利用谱结构这一类方法有效”。

#### 4. 控制实验矩阵

三种架构、三种数据集、统一 batch size / cosine schedule / from-scratch 训练，使得 paper 读起来是一个设计过的 controlled study，而不是杂乱地拼 benchmark。

#### 5. 50-epoch 扩展实验是必要且有价值的

这部分至少让作者没有把所有 gap 都偷换成“就是谱方法更好”，而是承认有些 gap 更像早期收敛收益，有些才像 persistent solution quality gap。

### 2.2 主要问题

#### 1. 最大的问题是 baseline protocol 过于特殊，导致 external validity 很弱

全文最危险的地方不是结果不显著，而是实验 recipe 太“非主流”，从而让 reviewer 很容易怀疑大 gap 主要来自 baseline 被压低了。

主要风险包括：

- 全部模型 from scratch；
- 数据集较小，且 CIFAR-100 被 resize 到 224x224；
- 没有 warmup；
- 没有 Mixup / CutMix / RandAugment / label smoothing；
- batch size 固定为 128；
- 单卡训练。

对 ResNet-50 来说，这种 recipe 可能还算能跑；但对 DeiT-S、ConvNeXt-T 这类现代架构，AdamW 往往高度依赖更完整的训练 recipe。于是当前论文里的巨大 gap，可能混合了两部分：

1. spectral optimizer 的真实优势；
2. baseline 在一个对其并不友好的 recipe 下被系统性低估。

这会直接冲击全文最重要的解释力度。更稳的结论应该是：

> 在作者当前的受控训练协议下，spectral methods 比 AdamW 更 robust。

而不是更强地写成：

> spectral optimization 本身就是现代视觉架构的默认最优选择。

#### 2. 学习率调优协议不够对称，也不够强

作者强调做了独立 learning rate sweep，但仔细看设定，问题不少：

- sweep 只做 5 epoch；
- 27 个配置里有 7 个在扫到的范围内仍是 monotonic；
- 其中 2 个甚至直接用了 literature defaults；
- Shampoo 还做了额外 refined sweep，且这个 refined sweep 明显改写了其排名。

这说明 tuning budget 本身就可能深刻影响结论。既然文章的核心论点是“不是 hyperparameter failure”，那现在的 protocol 反而留下了一个 reviewer 最容易追问的口子：

> 你证明的到底是“AdamW/SGD 真不行”，还是“在你这套简化 tuning protocol 下它们没被调到位”？

尤其当作者自己已经展示出 Shampoo 因为更细 sweep 可以直接提升 4.4 个点时，这个问题就更严重了。

#### 3. “family-level advantage” 的 claim 目前写得偏满

SOAP 和 Shampoo 只在 CIFAR-100 上被系统纳入主比较，而 Tiny-ImageNet 与 ImageNet-100 实际上只有 Muon 和 AdamW 的主结果。因此当前证据能稳妥支撑的是：

> 在 CIFAR-100 上，spectral family 的优势不是 Muon 独有。

但还不够支撑更宽的说法：

> spectral optimization 作为一个 family，在整个 3x3 任务矩阵里都显示出统一的家族优势。

换句话说，family-level claim 的证据范围小于正文里给人的 impression。

#### 4. “normalization 是 primary mediator” 这个解释有明显混杂

作者多次把现象解释为 batch norm vs layer norm 的差异，但当前三个 backbone 同时改变了很多因素：

- normalization 类型；
- convolution vs attention；
- standard conv vs depthwise separable conv；
- token mixing 方式；
- layer composition 和 parameterization。

因此现在的设计更像是“不同 architecture families 的整体差异”，而不是“normalization causal effect”的识别实验。当前最多能说：

> 这些现象和 normalization change 一致相关。

但还不能强到说：

> normalization 是主要中介因素。

要想撑起这个 claim，至少需要一个更干净的 controlled ablation，例如在同一 backbone 上只改 normalization，或者在同一 family 内找更可比的变体。

#### 5. 机制部分目前证据还偏薄

sign reversal 很抓人，但它现在更像一个 interesting clue，而不是一个被充分验证的 mechanism。

更关键的是，后半段会给人一种比较明显的“先有现象、再事后组织机制解释”的阅读观感。前半段先建立了一个相对清楚、也相对扎实的主故事：spectral methods 跨架构更稳；但到了后半段，文章更像是在看到若干 layer-wise 异质性结果之后，再努力把这些结果整理成一个统一机制框架。这样的写法不是不能做，但 reviewer 很容易问：

> 这些分析究竟是在验证一个先验机制假说，还是在对已有结果做 post-hoc interpretation？

这种观感之所以会出现，主要是因为：

- 机制指标是作者自定义的 spectral response ratio；
- 证据主体依然是相关性分析，而不是干预性证据；
- CNN 和 transformer 上出现 sign reversal 后，文章更像是把“不一致的结果”升级成“不同机制”的新发现；
- ConvNeXt-T 是主结果里最亮眼的架构之一，但机制分析没有和 ResNet-50 / DeiT-S 一样扎实；
- ResNet-50 的 weight-swap 分析还有 BatchNorm running stats 未重算这个 caveat。

几个具体问题：

- correlation 不是因果；
- 只在 ResNet-50 和 DeiT-S 上给了最强的机制叙事，ConvNeXt-T 没有同等强度的 layer-wise analysis；
- spectral response ratio 是作者自定义指标，为什么它比其他谱指标更能承载机制解释，目前更多是合理化而不是证明；
- weight swap 对 ResNet-50 还存在不重算 BatchNorm running stats 的 caveat，这恰好又影响 CNN 侧的重要结果。

因此现在更稳的定位应该是：

> layer-wise analyses provide supporting mechanistic evidence

而不是：

> they reveal fundamentally different optimization mechanisms

后者对于当前证据强度来说偏满。

#### 6. 统计与评估协议还有几个会让 reviewer 犹豫的点

包括但不限于：

- 不同实验混用 3 seed、4 seed 和 single seed；
- 一些 50-epoch 结果尤其是 IN100 部分是 single-seed estimate；
- 用 best validation accuracy 作为核心指标，会弱化对真实训练稳定性和最终收敛行为的刻画；
- Welch t-test 建立在非常小的 n 上，虽然方向上没问题，但会让人对精确 p 值的说服力打折。

这些点单独看都不致命，但叠加在一起，会让“统计上很稳”“机制上很稳”的印象被削弱。

#### 7. practical recommendation 过强

文中在 discussion 里已经接近提出：

> 任何使用 layer normalization 的架构，都应默认采用 spectral optimizers。

这在当前证据下明显太强。因为：

- 只有两个 layer-norm architecture；
- 数据集都不大；
- 训练设定是 from-scratch + light recipe；
- 还没有和更现代、更强的非谱 optimizer baseline 系统比较。

更安全的实践结论应该是：

> 对 layer-normalized 视觉架构，spectral optimizers 是非常值得优先尝试的强 baseline，尤其在较短训练预算和简化 recipe 下。

#### 8. 全文最容易被抓住的总问题是：control 做得很整齐，但 realism 不够

这篇文章的 controlled study 气质很强，这是优点；但同时，它也会逼 reviewer 问一个更尖锐的问题：

> 你测到的是“真实世界会发生的 optimizer ranking”，还是“在一个极度简化的实验世界里会发生的 ranking”？

如果这个问题不正面处理，paper 会很容易落到“interesting but too protocol-specific”。

## Part 3. 建议改法

### 3.1 先改 claim，而不是先加机制词汇

我觉得这篇文章当前最需要先做的，不是把“why”讲得更满，而是把“what exactly is shown”讲得更准。

abstract / introduction / conclusion 里更稳的表述应该是：

> Under a controlled from-scratch training protocol on three representative vision backbones, spectral optimizers consistently outperform AdamW and SGD, with gains that are architecture-dependent and especially pronounced on layer-normalized architectures.

而不是更强的：

> spectral optimization is the default robust choice for modern vision architectures.

前者稳，后者会让 reviewer 立刻反问外部有效性。

机制部分也一样。当前最稳的写法不是：

> 我们通过 sign reversal 揭示了 CNN 和 transformer 中 fundamentally different optimization mechanisms。

而更应该是：

> 我们观察到 layer-wise spectral statistics 与 optimizer benefit 之间存在架构相关的异质性，这为不同架构上可能存在不同优化机制提供了支持性证据。

这样会更符合当前证据强度，也能减少“把意外结果直接上升为机制发现”的观感。

### 3.2 主线建议改成“controlled empirical study of coupling”，而不是“normalization mechanism paper”

当前这篇稿子最有说服力的，其实不是它已经证明了 normalization/谱机制，而是它证明了：

1. optimizer ranking 对 architecture 很敏感；
2. spectral family 在作者的受控设定下表现出异常稳定的强势；
3. 这种强势不是单一 backbone 的偶发现象；
4. layer-wise spectrum 提供了一个有希望但仍待加强的解释窗口。

如果把主线稍微往这个方向收，整篇 paper 会稳很多，也更符合标题里的 “controlled empirical study”。

### 3.3 最需要补强的三件事

#### 1. 给 AdamW / SGD 一个更难被质疑的强 baseline

这是最关键的。

最理想的做法是至少补一组更现实的训练 recipe，对主要结论做 stress test，例如：

- warmup；
- stronger augmentation；
- label smoothing；
- 对 DeiT/ConvNeXt 更贴近常见 recipe 的训练细节；
- 或者至少更长、更对称的 tuning budget。

不一定要全矩阵重跑，但至少要选最关键的几个配置验证：

- ConvNeXt-T on CIFAR-100 / ImageNet-100；
- DeiT-S on CIFAR-100。

只要证明在更强 baseline 下 spectral gap 仍然明显，这篇 paper 的可信度会立刻提高很多。

#### 2. 补一个更干净的 architecture factor ablation

如果作者想继续强调 normalization 是 primary mediator，就需要更干净的证据。

可行方向包括：

- 同一 backbone 改 BN/LN；
- 同一 family 内比较 normalization 变体；
- 或者至少把“normalization is primary mediator”降级为“one plausible driver”。

如果做不到干净消融，我建议直接降 claim，不要硬讲。

#### 3. 把 mechanism 部分降级为 supporting evidence，或者补强到更完整

现在 weight-swap + spectral ratio 是有趣的，但不够成为全文最硬的 mechanistic pillar。

如果作者不准备大补实验，我反而建议明确接受这样一个定位：

> 后半段的价值主要在于提出可检验的 mechanistic hypotheses，而不是已经完成机制证明。

这个定位其实并不吃亏，反而会让整篇文章显得更诚实，也更符合当前证据形态。

两条路二选一：

1. 继续补强：
   - ConvNeXt-T 也给出完整 layer-wise analysis；
   - Recompute BatchNorm stats after swap；
   - 对比多个谱指标而不只一个自定义指标；
   - 检查相关性是否被 layer type / parameter count / matrix shape 混杂。
2. 或者主动降级：
   - 把这些分析写成 hypothesis-supporting evidence；
   - 不再把 sign reversal 说成“揭示了 fundamentally different mechanisms”。

### 3.4 建议补的实验

#### 1. 更强 baseline recipe 下的关键复验

这是优先级最高的补实验。

最小可行做法：

- 选 2 到 3 个最关键配置；
- 用更标准的 AdamW / SGD 训练 recipe 重跑；
- 保持 Muon / SOAP / Shampoo 也在同一 recipe 下比较；
- 报最终 gap 是否仍然存在，以及缩小了多少。

这组实验不一定要把全文推翻重做，但会极大减少“baseline 不公平”的质疑。

#### 2. 把 SOAP 或 Shampoo 扩展到至少一个非 CIFAR 数据集

这是为了真正支撑 family-level claim。

哪怕不扩到完整 3x3，至少也应该在 Tiny-ImageNet 或 ImageNet-100 上补一个 architecture pair。否则 reviewer 很容易说：

> 你证明的是 Muon 普遍有效，而不是 spectral family 普遍有效。

#### 3. 更对称的 tuning budget

如果作者继续强调“不是超参数问题”，那就应该在 tuning protocol 上更强势。

比如：

- 不只 5-epoch sweep；
- 对 monotonic case 继续外扩搜索范围；
- 不再混用 formal sweep 和 literature default；
- 对所有 optimizer 采用尽量一致的 tuning budget。

这组实验的价值不在于再涨多少点，而在于堵住最自然的 reviewer 质疑。

#### 4. 更干净的 normalization / architecture ablation

如果资源有限，哪怕只补一个 controlled case，也很值。

因为现在整篇 paper 最容易被指出的是：

> 你观察到的是 architecture bundle 的差异，而不是某一个清晰可识别因素的因果作用。

#### 5. 机制分析补严谨性

至少建议做这几件事：

- ResNet swap 后重算 BatchNorm running stats；
- ConvNeXt-T 补完整 layer-wise 分析；
- 加一个对照谱指标；
- 检查 per-layer correlation 是否在 layer type 分组后仍成立。

做完这些，机制部分就不会像现在这样“很有趣，但偏脆”。

## Part 4. 如果只能做最小修改

如果时间不够大补实验，我建议最低限度先做：

1. 重写 abstract / introduction / conclusion，把 claim 收紧到“controlled from-scratch protocol”。
2. 明确承认当前 recipe 的局限，不要把结论直接外推到现代标准大规模视觉训练。
3. 把 “normalization is the primary mediator” 降到相关性解释，不要写成已识别的主因。
4. 把 “for any architecture with layer normalization, spectral optimizers should be the default” 改成更保守的 practical takeaway。
5. 把 mechanism 相关表述从“reveals”改成“suggests”或“is consistent with”。
6. 在正文而不是只在 limitations 里更主动地讲清楚 tuning asymmetry、single-seed result 和 best-checkpoint metric 的边界。

简单说，这篇文章即使暂时不补很多实验，也至少要做到：

> 让 strongest evidence 和 strongest claim 之间重新对齐。

## Part 5. 总体判断

这篇论文是有潜力的，它最大的优点在于现象确实强，实验矩阵也有设计感，paper 想回答的问题也值得关心：optimizer 是否和 architecture 存在结构性耦合，以及谱方法是否因此更稳。

但当前版本把结论推得比证据更远了一些。最大短板是 baseline recipe 的外部有效性、learning-rate tuning 的不对称性、以及把 architecture bundle 直接解释成 normalization mechanism 的跨步过大。机制部分有启发性，但还不够硬；practical recommendation 也明显写满了。

> 现象强、问题重要、但当前版本仍需要更强 baseline 和更克制 claim 的 empirical study。

如果作者能补上至少一组更现实的 baseline 对照，并把核心 claim 收紧，这篇稿子的说服力会明显上一个台阶。否则它很容易被评价为：

> interesting controlled study, but too protocol-specific and somewhat over-claimed.
