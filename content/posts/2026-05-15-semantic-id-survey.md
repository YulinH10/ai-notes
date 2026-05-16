---
title: "从 Item ID 到 Semantic ID：推荐系统中召回与精排技术路线的分化、收敛与统一"
date: 2026-05-15T15:00:00+08:00
draft: false
tags: ["Semantic ID", "推荐系统", "生成式推荐", "长序列建模", "召回", "精排", "综述"]
categories: ["AI学习"]
summary: "系统梳理推荐系统从传统 Item ID 向 Semantic ID 演进的技术脉络，涵盖长序列建模、生成式召回、语义精排及召回-精排统一框架，串联 SIM、MIMN、TIGER、GLASS、UxSID、DIG 等代表性工作。"
---

## 摘要

近年来，推荐系统面临两个日益尖锐的结构性矛盾：**用户行为序列越来越长**，**物料空间越来越大**。传统以原子化 Item ID 为核心建模单元的范式，在效率、泛化和可扩展性上均遭遇瓶颈。**Semantic ID (SID)** ——通过量化编码将物料映射为层次化离散语义 token 的技术——正在从生成式召回中的一个局部技巧，扩散为同时影响召回、精排和全链路架构的核心中间表示。

本文系统梳理这一技术演进的脉络。首先回顾长序列用户行为建模的两条经典路线（search-based 与 compression-based）及其局限；然后介绍 Semantic ID 的基础原理与构建方法；接着分别从召回侧（以 TIGER、GLASS 为代表的生成式推荐）和精排侧（以 UxSID 为代表的语义化长序列建模）两个方向，分析 SID 如何重塑各自的问题形态；最后，以 DIG 为切入点，讨论召回与精排围绕统一语义边界收敛的可能性，并展望开放问题与未来方向。

---

## 1. 引言

推荐系统的核心任务，可以抽象为一个信息匹配问题：给定用户的历史行为和上下文，从海量物料中找到最相关的内容并排序呈现。长期以来，这个过程被拆分为两个阶段：

- **召回（Retrieval）**：从全量物料中快速筛选出候选集（通常数百到数千个），强调覆盖率和效率。
- **精排（Ranking）**：在候选集上做更精细的排序，强调准确率和区分度。

两者各自演化出独立的技术体系。召回侧以双塔模型 + ANN 检索为主流范式，精排侧以 DIN/DIEN 一类注意力模型为基础不断演进。然而，随着系统规模的持续增长，两个基本矛盾变得越来越突出：

**矛盾一：用户行为越来越长。** 在短视频、电商、信息流等场景中，用户累积的行为历史已经从早期的几十条扩展到上千甚至上万条。理论上，更长的历史意味着更丰富的兴趣信号；但实际上，直接将超长序列送入模型会带来巨大的计算开销和严重的噪声干扰。

**矛盾二：物料空间越来越大。** 当物料规模从百万扩展到千万甚至更大，基于原始 Item ID 的表示和检索方式变得日益笨重。每个 item 一个独立的 embedding，既缺乏结构化的语义信息，也难以有效泛化到冷启动场景。

正是在这一背景下，两个技术方向在过去几年迅速升温：**超长序列建模（Ultra-Long Sequential Modeling, ULSM）** 和 **Semantic ID**。前者关注如何高效处理超长用户历史，后者关注如何将 item 从孤立的 ID 转化为更适合建模、检索和生成的语义单元。

本文的核心论点是：这两个方向看似独立，实则正在走向深度融合。Semantic ID 不仅在改变物料的表示方式，更在重塑长序列的组织方式、召回的生成方式和精排的决策方式。推荐系统正在从「围绕 Item 做匹配」走向「围绕 Semantic Interface 做组织」。

---

## 2. 长序列用户行为建模：从 DIN 到 SIM

在讨论 Semantic ID 之前，有必要先回顾长序列建模的技术演进，因为它构成了 SID 进入精排系统的直接动因。

### 2.1 基础：注意力机制与 DIN/DIEN

**DIN (Deep Interest Network)**[^zhou2018din] 是深度推荐模型引入注意力机制的标志性工作。它的核心思想是：用户的兴趣是多样化的（multi-interest），面对不同的候选 item，应该激活用户历史中不同的行为。DIN 通过 target attention 机制，让候选 item 作为 query 对用户行为序列做加权聚合，取代了此前简单的 sum/mean pooling。

**DIEN (Deep Interest Evolution Network)**[^zhou2019dien] 进一步引入 GRU 建模兴趣的时序演化，提出 AUGRU（Attention-based Update Gate Recurrent Unit）捕捉与目标相关的兴趣演化路径。

这些方法在短序列（几十到一两百条行为）上效果显著，但当序列长度扩展到数千条以上时，target attention 的 O(L·d) 计算开销和 GRU 的顺序依赖都会成为瓶颈。

### 2.2 Compression-Based 路线：把历史压进记忆

面对超长序列，第一类思路是**先将用户历史压缩为固定大小的记忆表示**，在线阶段仅使用这些压缩后的状态。

**MIMN (Multi-channel User Interest Memory Network)**[^pi2019mimn]（KDD 2019）是第一个在工业规模部署的超长序列方案。它的关键创新是将用户兴趣建模从在线推理中解耦出来：通过一个异步的 UIC（User Interest Center）服务器，基于 NTM（Neural Turing Machine）架构，每当用户产生新行为时增量更新一个多通道记忆矩阵。这样，在线推理阶段面对的始终是一组大小固定的兴趣状态，而非原始的长序列。MIMN 在阿里巴巴部署，支持长度达 ~1000 的行为序列。

**HPMN (Hierarchical Periodic Memory Network)**[^ren2019hpmn]（SIGIR 2019）进一步引入多尺度层次记忆结构。基于用户兴趣具有不同时间周期性的观察（日级、周级、季节级），HPMN 在不同层级维护不同时间分辨率的记忆，高层捕捉长期周期模式，低层捕捉近期细粒度行为。

**Compression-based 方法的共同局限：** 这些方法本质上是 **target-unaware** 的——记忆的更新和压缩过程不依赖于候选 item。它们擅长回答「这个用户整体喜欢什么」，却不擅长回答「面对当前这个候选物料，历史中的哪一部分最相关」。此外，一旦信息被压缩进固定数量的 memory slot，稀有但重要的兴趣信号容易被稀释。

### 2.3 Search-Based 路线：先检索再建模

第二类思路更具突破性：**不试图压缩整条历史，而是根据候选 item 从超长历史中检索出最相关的子集，再对子集做精细建模。**

**SIM (Search-based User Interest Modeling)**[^pi2020sim]（CIKM 2020）是这条路线的奠基工作。SIM 提出了两阶段架构：
- **GSU（General Search Unit）**：从用户完整历史中检索 top-K 最相关行为。提供两种检索方式——基于品类匹配的硬检索（hard search）和基于向量内积的软检索（soft search）。
- **ESU（Exact Search Unit）**：对检索出的子序列施加 target attention，提取精确的兴趣表示。

SIM 在阿里巴巴部署，处理长度达 54,000+ 的用户行为序列，是首次在工业级别验证超长序列建模收益的工作。它确立了一个重要原则：**target-aware 的检索是超长序列建模的关键**，直接解决了 MIMN 等方法 target-unaware 的根本缺陷。

**ETA (Efficient Target Attention)**[^qin2021eta] 针对 SIM 检索阶段的计算瓶颈做了优化，使用 SimHash（局部敏感哈希）将 item embedding 转换为二进制哈希码，通过 Hamming 距离近似最近邻检索，将检索复杂度从 O(L·d) 降低到近似 O(L)。

**SDIM (Sampling-based Deep Interest Modeling)**[^cao2022sdim]（CIKM 2022）提出了另一种路径：通过多轮 SimHash 碰撞采样来识别与候选 item 相似的历史行为，避免了显式构建检索索引的工程开销，所有操作都是张量化的、对 GPU 友好的。

**TWIN (TWo-stage Interest Network)**[^twin2023]（KDD 2023）在快手部署，进一步将序列长度推到 100,000+，改进了 GSU 与 ESU 之间的一致性问题，引入 target-aware 的检索阶段注意力机制。

### 2.4 两条路线的权衡与未解问题

| 维度 | Compression-Based | Search-Based |
|------|-------------------|--------------|
| 在线复杂度 | 稳定（固定大小记忆） | 与候选数相关（每个候选需检索一次） |
| Target-Aware | 否（记忆预计算） | 是（按候选检索） |
| 信息保留 | 有损压缩，稀有兴趣容易丢失 | 较好，但受 top-K 截断影响 |
| 工程复杂度 | 需维护异步更新服务 | 需维护在线检索索引 |

两条路线各有优势，但共同面临一个更深层的问题：**它们都在 Item 粒度上操作**。无论是把 item 行为压进 memory slot，还是从 item 历史中检索相关行为，基本建模单元始终是原始的、原子化的 item。当 item 空间继续膨胀、用户历史继续延长时，这种 item-level 的操作方式会遇到越来越明显的天花板。

这正是 Semantic ID 进入长序列建模的切入点。

---

## 3. Semantic ID：基础原理与构建方法

### 3.1 什么是 Semantic ID

Semantic ID 的核心思想是：**将每个 item 从一个孤立的原子 ID（如整数编号），映射为一个多层离散 token 序列，使得语义相近的 item 在编码空间中共享部分前缀结构。**

形式化地，给定 item $i$ 的内容特征向量 $\mathbf{x}_i$，Semantic ID 是一个映射：

$$f: \mathbf{x}_i \rightarrow (c_1, c_2, \ldots, c_L)$$

其中 $c_l \in \{1, 2, \ldots, K_l\}$ 是第 $l$ 层 codebook 中的离散编码，$L$ 为编码层数，$K_l$ 为第 $l$ 层 codebook 的大小。

这种编码具有**层次化语义结构**：第一层 token $c_1$ 捕捉最粗粒度的语义（如大类），后续 token 逐层细化，形成从粗到细的语义树。语义相近的 item 会共享更多前缀 token。

### 3.2 RQ-VAE：主流构建方法

**RQ-VAE (Residual-Quantized Variational Autoencoder)**[^lee2022rqvae] 是目前 Semantic ID 构建的事实标准方法，最初由 Lee 等人在 CVPR 2022 为图像生成提出，后被 TIGER 引入推荐系统。

RQ-VAE 的工作流程：
1. **编码**：Encoder 将 item 特征 $\mathbf{x}_i$ 映射为连续向量 $\mathbf{z}_i$。
2. **逐层残差量化**：
   - 第 1 层：在 codebook $\mathcal{C}_1$ 中找到最近的 codeword $\mathbf{e}_{c_1}$，记录残差 $\mathbf{r}_1 = \mathbf{z}_i - \mathbf{e}_{c_1}$。
   - 第 $l$ 层：在 codebook $\mathcal{C}_l$ 中量化残差 $\mathbf{r}_{l-1}$，得到 $c_l$ 和新残差 $\mathbf{r}_l = \mathbf{r}_{l-1} - \mathbf{e}_{c_l}$。
3. **解码**：Decoder 从量化表示 $\sum_l \mathbf{e}_{c_l}$ 重构原始特征。
4. **训练目标**：重构损失 + commitment loss + VQ loss。

RQ-VAE 的关键优势在于它自然产生层次化编码：前面的 codebook 捕捉主要语义信息，后面的 codebook 编码残余细节，这完美匹配了 Semantic ID 所需要的粗到细语义结构。

### 3.3 其他 Semantic ID 构建方法

除了 RQ-VAE，社区也探索了其他构建方式：

- **层次化聚类**：DSI[^tay2022dsi] 最初使用层次化 k-means 聚类生成文档 ID。优点是简单直接，缺点是聚类质量不稳定、难以端到端优化。
- **VQ-Rec**[^hou2023vqrec]：使用标准向量量化（非残差）构建跨域可迁移的 item 表示。
- **LC-Rec**[^zheng2024lcrec]：在 RQ-VAE 基础上引入协同过滤信号，使得 Semantic ID 不仅编码内容语义，还捕捉协同模式。
- **LETTER**[^li2024letter]：提出端到端可学习的 tokenizer，取代固定的 RQ-VAE，使 tokenization 过程能随推荐任务联合优化。
- **文本化 Semantic ID**：IDGenRec 等工作探索生成人类可读的文本 token 序列作为 Semantic ID，尝试桥接 LLM 能力与推荐任务。

### 3.4 Semantic ID 与传统 Item ID 的本质区别

| 维度 | 传统 Item ID | Semantic ID |
|------|-------------|-------------|
| 表示形式 | 单一原子整数 | 多层离散 token 序列 |
| 语义结构 | 无（ID 之间无天然关系） | 有（共享前缀 = 语义相近） |
| 泛化能力 | 无（新 item 需从头学 embedding） | 有（相似 item 共享编码子结构） |
| 可扩展性 | 线性增长（每个 item 独立 embedding） | 组合增长（codebook 大小远小于 item 总数） |
| 适用场景 | 匹配、检索 | 匹配、检索、**生成** |

最关键的区别是：Semantic ID 让 item 空间具备了**可生成性**。传统 Item ID 是一个无结构的标签空间，模型只能在其中做分类或检索；而 Semantic ID 将 item 映射到了一个有层次的 token 空间，模型可以像生成语言一样逐 token 生成 item 表示。这一性质直接催生了生成式推荐范式。

---

## 4. Semantic ID 在召回侧：生成式推荐的兴起

### 4.1 从检索到生成：范式转移

传统召回的核心操作是**检索**：将用户和 item 映射到同一向量空间，通过 ANN（近似最近邻）搜索找到候选集。双塔模型（DSSM 及其变体）是这一范式的代表。

双塔模型的核心限制在于：用户和 item 的表示在最后做内积之前是完全独立编码的，缺乏 early interaction，表达能力有上界。此外，当 item 空间达到千万级别时，即便是 ANN 检索也面临索引维护、更新延迟和精度损失等问题。

**生成式推荐（Generative Recommendation）** 提出了一种根本不同的思路：**不再从外部索引中检索 item，而是让模型直接生成 item 的标识符。** 如果 item 的标识符是 Semantic ID，那么原本在百万/千万 item 上的一步分类问题，就被改写为在较小 codebook 上的多步自回归生成问题。

### 4.2 DSI：生成式检索的开山之作

**DSI (Differentiable Search Index)**[^tay2022dsi]（NeurIPS 2022）是生成式检索的开创性工作。Tay 等人提出将整个文档语料库编码进 Transformer 模型的参数中，模型直接将查询映射为文档标识符。DSI 探索了多种文档 ID 方案，包括无结构的原子 ID、简单的字符串 ID，以及基于层次聚类的语义化结构 ID。

DSI 确立了「model-as-index」范式：**一个单一的 Transformer 同时充当索引和检索模型**，消除了传统 retrieve-then-rank 管线中对外部索引的依赖。虽然 DSI 最初针对的是文档检索而非推荐，但它为后续工作铺平了道路。

### 4.3 TIGER：Semantic ID 推荐的里程碑

**TIGER (Transformer Index for GEnerative Recommenders)**[^rajput2023tiger]（NeurIPS 2023）是 Google 将生成式检索范式正式引入推荐系统的里程碑工作。TIGER 的核心贡献包括：

1. **RQ-VAE 构建 Semantic ID**：首次系统性地使用 RQ-VAE 为推荐系统中的 item 学习层次化离散语义编码。
2. **Seq2Seq 自回归生成**：将推荐建模为序列到序列的生成任务——输入是用户的行为序列（以 Semantic ID 表示），输出是下一个 item 的 Semantic ID，逐 token 自回归生成。
3. **语义相似性验证**：TIGER 验证了 Semantic ID 确实捕捉了有意义的 item 相似结构——内容相似的 item 获得相似的 ID 前缀。

TIGER 在多个推荐基准上优于传统检索方法，证明了生成式推荐的可行性。但 TIGER 的原始设计主要针对中等长度的用户序列，对超长历史和超大规模物料的扩展性仍有不足。

### 4.4 GLASS：让 SID 层次结构参与生成决策

**GLASS**[^glass2024] 在 TIGER 的基础上做了两个关键推进：将生成式召回扩展到超大规模物料和超长用户行为序列，以及更重要的——让 Semantic ID 的层次结构真正参与到生成过程的决策分工中。

GLASS 抓住的核心问题是：**在多层 SID token 的自回归生成中，不同层 token 的职责并不对等。** 第一个 token 承担最粗粒度的语义路由，它一旦走偏，后续 token 再怎么生成也只是在错误的分支里越走越深。然而，此前多数生成式推荐方法将所有 token 的生成交给同一个统一模型，默认它们职责相似。

GLASS 提出 **SID-Tier 机制**，将长期兴趣建模拆分为两种不同机制：

- **首 token 生成**：构造面向首层语义空间的全局兴趣表示。具体做法是为首层 codebook 中每个 codeword 计算全局语义原型（prototype），然后测量用户长期行为与各原型的相似度分布，形成一张「用户在首层语义 codebook 上的兴趣热力图」。这张热力图帮助模型在第一步做出稳定的粗粒度路由。
- **后续 token 生成**：用已生成的首层 SID 作为检索键，从用户长期历史中做 **Semantic Hard Search**——仅保留首层 SID 一致的历史行为作为证据。Decoder 通过 gate 机制动态融合短期行为编码和长期语义一致的历史片段，决定后续细粒度 token 的生成。

GLASS 还引入了 **Sparsity Augmentation** 技术，通过查找首层 SID 的语义邻居来扩展检索范围，缓解长尾 SID 下历史行为过于稀疏的问题。

GLASS 的意义在于：**SID 的层次结构不再只是编码结果，而开始重构召回模型本身的决策逻辑。** 生成式推荐从「把 SID 用起来」进入了「认真思考 SID 的层次结构该如何参与决策」的阶段。

### 4.5 其他生成式召回相关工作

- **P5**[^geng2022p5]：将推荐统一为自然语言处理任务，使用 T5 架构处理多种推荐任务。虽然使用的是文本化 item 表示而非 RQ-VAE 型 Semantic ID，但确立了「Recommendation as Language Processing」的思想基础。
- **HSTU (Hierarchical Sequential Transduction Unit)**[^zhai2024hstu]（ICML 2024）：Meta 的万亿参数生成式推荐架构。核心贡献是证明了推荐系统可以像 LLM 一样遵循 scaling laws——更多参数和更多数据带来持续提升。HSTU 设计了专门面向推荐序列的高效 Transformer 变体，比 FlashAttention-2 快 15.2 倍。在 Meta 多个产品线部署，关键指标提升高达 12.4%。这项工作为整个生成式推荐方向提供了工业级别的 scaling evidence。
- **约束解码（Constrained Decoding）**：多项 2024 年的工作研究如何确保生成的 Semantic ID 对应到有效 item，使用前缀树（Trie）和约束 beam search 等技术。

---

## 5. Semantic ID 在精排侧：从长序列建模切入

如果说 Semantic ID 在召回侧的应用路径是「从编码到生成」，那么它在精排侧的进入方式则完全不同——**从长序列管理切入**。

### 5.1 精排为什么需要 Semantic ID

精排阶段的任务是在召回给出的候选集中做更精细的排序判断。随着用户行为序列越来越长，精排模型希望利用更多历史信息来提升判断准确率。然而，如第 2 节所述，直接在 item 粒度上处理超长序列面临 search-based 和 compression-based 两种路线的各自局限。

关键的思路转换是：**也许问题不在于模型还不够强，而在于我们组织用户历史的方式还不对。** 如果 item 本身可以被映射到更抽象、更有结构的语义空间中，用户历史也就可以被重新组织。

### 5.2 UxSID：将用户历史从事件序列改造为语义兴趣库

**UxSID**[^uxsid2024] 是精排侧引入 Semantic ID 的代表性工作。它的核心思想是：**将用户历史从一个平铺的 item 行为序列，改造为一个可按语义寻址的兴趣库。**

具体做法：
1. **SID 映射**：通过 RQ-VAE 等量化方法，将用户历史中每个 item 映射为其 Semantic ID。
2. **语义重组**：按 SID 将用户历史行为路由到不同的语义桶（semantic bucket）中。
3. **桶内聚合**：在每个语义桶内聚合局部兴趣表示，构建 **semantic-specific user representation**。
4. **按需访问**：在线阶段，根据目标候选 item 的 SID，直接读取用户在对应语义方向上的兴趣表示进行交互。

这样一来，问题从「去原始行为序列里找最相关的几个 item」（item-to-item 的在线检索），变成了「这个候选 item 属于哪个语义簇，用户在这个语义簇上表现出怎样的兴趣」（item-to-semantics-to-user 的两跳访问）。大量原本需要在线完成的细粒度筛选，被前移到了离线的语义组织阶段。

UxSID 的模型架构包含 **IAIC（Interest Anchored Interaction Component）**模块：
- 以 Semantic ID 作为 Interest Anchor
- 通过 Cross-Attention 机制让候选 item 与对应语义桶内的用户行为交互
- 使用 Gate 机制融合全局兴趣表示（Global Embedding）和局部兴趣表示（Local Embedding）
- 最终结合 UID、SID 等特征通过 MLP 层进行多任务监督学习

### 5.3 UxSID vs MIMN：兴趣单元定义的根本差异

UxSID 与 MIMN 的对比极具启发性，因为两者都在解决「超长序列怎么用」的问题，但它们对「兴趣单元」的定义完全不同：

| 维度 | MIMN | UxSID |
|------|------|-------|
| 兴趣单元性质 | Latent interest state（模型内部学出的抽象状态） | Semantic-specific representation（与 item 侧 SID 对齐的语义表示） |
| 可寻址性 | 否（记忆槽位无显式语义地址） | 是（按 SID 直接寻址） |
| Target-Aware | 否（记忆预计算） | 是（按目标 item 的 SID 访问） |
| 核心问题 | 超长序列怎么存下来、怎么压进去 | 超长序列应该按什么语义结构被组织起来 |

MIMN 本质上是在做「用户内部兴趣压缩」，UxSID 则更像是在做「用户历史的语义重组与按需访问」。后者的关键进步在于，它将长序列建模接到了 Semantic ID 这条更大的技术路线上——SID 不再只是 item 编码工具，更成为了用户历史管理工具。

### 5.4 工程价值与局限

UxSID 的工程价值不只在于指标提升，更在于重新定义了超长序列的管理方式。用户历史长到一定程度后，它不再适合作为一张平铺的流水账被直接输入模型，而更像一个需要被组织、索引和按需访问的**兴趣数据库**。

但 UxSID 也有明确的边界条件：
- **SID 聚类质量依赖**：如果同一 SID 桶内的 item 语义不稳定，semantic-specific representation 也会变模糊。
- **离线组织开销**：虽然降低了在线 item-level matching 成本，但引入了新的离线语义组织和维护开销。
- **新物料更新**：新 item 不断涌入后，语义编码需要定期更新，如何实现增量式 SID 更新是工程挑战。

---

## 6. 从分化到收敛：召回与精排走向统一

### 6.1 表面分化，底层收敛

单独来看，GLASS 是在优化生成式召回，UxSID 是在解决长序列精排。但将它们放在同一坐标系里，会发现一个深层的共识：**原始 Item ID 已经越来越不适合作为推荐系统的基本建模单元。**

- 在 GLASS 中，SID 让 item 召回从一步到位的直接决策，变成按语义层级逐层展开的生成过程。
- 在 UxSID 中，SID 让用户历史从一串平铺事件，变成可按语义寻址的兴趣结构。

两者都在做同一件事：**在 item 空间和用户空间之间插入一个语义中间层，围绕这个中间层重新组织计算。**

### 6.2 DIG：判别即生成

**DIG (Discrimination Is Generation)**[^dig2024] 将这种收敛趋势推到了逻辑终点。它的核心洞察是：

> 精排在 item 空间做 discrimination，生成式召回在 token 空间做 generation。如果 Semantic ID 是连接这两个空间的映射，那么 discrimination 和 generation 本质上是在不同表示空间里求解同一个 argmax 问题。

基于这个观察，DIG 做了一个关键的设计决策：**不再将 tokenizer 视为推荐任务的前置预处理，而是将其直接嵌入 discriminative ranker 联合训练。** 用点击监督和 BCE 目标去塑造 codebook 的边界，使得学出的离散编码不再只是为重构服务的压缩码，而是被推荐任务本身校准过的**判别性语义单元**。

DIG 的训练过程和推理过程体现了这种统一性：
- **训练阶段**：包含 Rank Loss 和 Recall Loss 的多任务学习。Rank Loss 在传统的 User/Context/U2I/Item 特征 embedding 上做判别；Recall Loss 在 SID tokenized 的 token embedding 上做生成。两路通过 Mixer Layer 融合。
- **推理（召回）阶段**：直接使用 token embedding 路径进行生成式检索，同时融合传统特征信息。

这种设计的深远意义在于：一旦 SID 的学习目标面向 retrieval 和 ranking 的共同判别边界，召回和精排就不再只是「各自用了 SID」，而是开始**共享同一套语义地基**。

### 6.3 统一框架的愿景

DIG 指向的方向可以用一句话概括：**与其让召回和精排各自在自己的链路里引入 Semantic ID，不如先学出一套真正贴合推荐任务边界的 Semantic ID，再让召回和精排围绕它展开。**

如果这个方向成立，推荐系统的架构可能从当前的：

```
[用户行为] → [召回模型(双塔/生成)] → [候选集] → [精排模型(DIN/SIM)] → [排序结果]
```

演化为：

```
[用户行为] → [SID 语义组织层] → [统一的语义检索+排序模型] → [排序结果]
```

在新范式中，Semantic ID 不再只是某一阶段使用的表示技巧，而是贯穿全链路的语义基础设施。

---

## 7. 技术演进总览

下表从时间线和技术定位两个维度，梳理本文涉及的关键工作：

| 论文 | 年份 | 会议 | 阶段 | 方法类别 | 核心贡献 |
|------|------|------|------|----------|----------|
| DIN | 2018 | KDD | 精排 | 注意力机制 | 引入 target attention，开启兴趣建模 |
| DIEN | 2019 | AAAI | 精排 | 兴趣演化 | GRU + AUGRU 建模兴趣时序演化 |
| MIMN | 2019 | KDD | 精排 | Compression-based | NTM 多通道记忆，首个工业级长序列方案 |
| HPMN | 2019 | SIGIR | 精排 | Compression-based | 多尺度层次周期性记忆 |
| SIM | 2020 | CIKM | 精排 | Search-based | 两阶段检索+注意力，处理 54K+ 序列 |
| ETA | 2021 | - | 精排 | Search-based | LSH 加速检索，O(L) 复杂度 |
| DSI | 2022 | NeurIPS | 检索 | 生成式检索 | Model-as-index，生成式检索开山 |
| RQ-VAE | 2022 | CVPR | 编码 | 量化 | 残差量化 VAE，SID 构建基础 |
| SDIM | 2022 | CIKM | 精排 | Hashing-based | 多轮哈希碰撞采样，GPU 友好 |
| TIGER | 2023 | NeurIPS | 召回 | 生成式推荐 | RQ-VAE SID + Seq2Seq 生成，推荐中 SID 的里程碑 |
| TWIN | 2023 | KDD | 精排 | Search-based | 100K+ 序列，改进 GSU-ESU 一致性 |
| HSTU | 2024 | ICML | 全链路 | 生成式推荐 | 万亿参数，推荐 Scaling Laws，Meta 部署 |
| GLASS | 2024 | - | 召回 | 生成式推荐 | SID-Tier，层次化语义决策分工 |
| UxSID | 2024 | - | 精排 | SID + 长序列 | SID 语义重组用户历史，semantic-specific 表示 |
| DIG | 2024 | - | 召回+精排 | 统一框架 | 判别即生成，SID 联合训练统一召回精排 |

---

## 8. 开放问题与未来方向

尽管 Semantic ID 的技术路线已经展现出强大的潜力，但从研究原型到全面落地仍有诸多挑战值得关注。

### 8.1 Semantic ID 的质量与稳定性

Semantic ID 的所有下游应用——无论是 GLASS 的层次化生成还是 UxSID 的语义重组——都高度依赖于 SID 本身的编码质量。如果同一语义桶内的 item 语义不一致，整个体系的收益就会大打折扣。

此外，当 RQ-VAE 随新物料重新训练时，同一 item 的 SID 可能发生变化（codebook 漂移问题）。如何保证 SID 的**时序稳定性**，使得历史编码不会因为 codebook 更新而失效，是一个重要的工程问题。

### 8.2 冷启动与增量更新

传统 Item ID 的冷启动问题已经很困难，Semantic ID 引入了新的复杂度：新 item 需要被实时编码为 SID 并纳入检索/生成体系。如何实现**增量式 SID 更新**——在不重训整个 RQ-VAE 的前提下为新 item 分配合理的 Semantic ID——仍是开放问题。

### 8.3 多模态 Semantic ID

当前多数 SID 构建方法仅基于 item 的文本特征或协同特征。在短视频、电商等场景中，item 往往包含文本、图像、视频、音频等多模态信息。如何构建**多模态 Semantic ID**，使得不同模态的语义信息都能被有效编码进离散 token 结构中，是一个值得深入探索的方向。UxSID 的 SIDs Generator 已经初步引入 MLLM Encoder 处理多模态 item 属性（文本、图像、世界知识等），但这个方向仍处于早期。

### 8.4 从「两阶段」到「端到端」

当前推荐系统中 Semantic ID 的构建与使用仍然是相对分离的：先训练 RQ-VAE 得到 SID，再用 SID 训练下游模型。DIG 开始尝试端到端联合训练，但更彻底的端到端方案——例如让推荐任务的梯度直接回传到 codebook 的构建中——仍有很大的探索空间。

### 8.5 Semantic ID 与 LLM 的融合

大语言模型在理解用户意图和 item 内容方面展现出强大能力。如何让 LLM 与 Semantic ID 体系深度融合——例如用 LLM 生成的语义向量作为 RQ-VAE 的输入，或者让 LLM 直接在 SID token 空间上做推理——是 2025 年以来一个快速升温的方向。

### 8.6 部署与系统约束

从工程落地角度看，生成式推荐的在线延迟（自回归生成天然比一次前向传播更慢）、Semantic ID 的存储与更新维护、以及与现有推荐系统基础设施的兼容性，都是真实存在的约束。HSTU 在 Meta 的成功部署证明了生成式推荐可以在工业规模下工作，但其背后的工程投入不可忽视。

---

## 9. 结语

回顾过去几年推荐系统的技术演进，一条清晰的主线正在浮现：

**从 Item ID 到 Semantic ID，推荐系统正在经历从「原子化匹配」到「语义化组织」的范式转换。**

这一转换不是发生在某一个环节上，而是同时发生在召回、精排和全链路架构中。在召回侧，Semantic ID 让物料检索从向量匹配走向层次化语义生成（TIGER → GLASS）；在精排侧，Semantic ID 让长序列建模从 item-level 操作走向 semantic-level 组织（SIM → UxSID）；在全链路层面，Semantic ID 正在成为连接 retrieval 和 ranking 的统一语义接口（DIG）。

如果要用一句话概括这篇综述的核心观察：

> **推荐系统正在从「围绕 item 做匹配」走向「围绕 semantic interface 做组织」。从长序列到 Semantic ID，这条路也许才刚刚开始。**

---

## 参考文献

[^zhou2018din]: Zhou, G., et al. "Deep Interest Network for Click-Through Rate Prediction." KDD 2018.

[^zhou2019dien]: Zhou, G., et al. "Deep Interest Evolution Network for Click-Through Rate Prediction." AAAI 2019.

[^pi2019mimn]: Pi, Q., et al. "Practice on Long Sequential User Behavior Modeling for Click-Through Rate Prediction." KDD 2019.

[^ren2019hpmn]: Ren, K., et al. "Lifelong Sequential Modeling with Personalized Memorization for User Response Prediction." SIGIR 2019.

[^pi2020sim]: Pi, Q., et al. "Search-based User Interest Modeling with Lifelong Sequential Behavior Data for Click-Through Rate Prediction." CIKM 2020.

[^qin2021eta]: Qin, J., et al. "End-to-End User Behavior Retrieval in Click-Through Rate Prediction Model." arXiv 2021.

[^cao2022sdim]: Cao, Y., et al. "Sampling Is All You Need on Modeling Long-Term User Behaviors for CTR Prediction." CIKM 2022.

[^twin2023]: "TWIN: TWo-stage Interest Network for Lifelong User Behavior Modeling in CTR Prediction." KDD 2023.

[^tay2022dsi]: Tay, Y., et al. "Transformer Memory as a Differentiable Search Index." NeurIPS 2022.

[^lee2022rqvae]: Lee, D., et al. "Autoregressive Image Generation using Residual Quantization." CVPR 2022.

[^rajput2023tiger]: Rajput, S., et al. "Recommender Systems with Generative Retrieval." NeurIPS 2023.

[^geng2022p5]: Geng, S., et al. "Recommendation as Language Processing (RLP): A Unified Pretrain, Personalized Prompt & Predict Paradigm." RecSys 2022.

[^zhai2024hstu]: Zhai, J., et al. "Actions Speak Louder than Words: Trillion-Parameter Sequential Transducers for Generative Recommendations." ICML 2024.

[^hou2023vqrec]: Hou, Y., et al. "Learning Vector-Quantized Item Representation for Transferable Sequential Recommendations." WWW 2023.

[^zheng2024lcrec]: Zheng, Z., et al. "LC-Rec: Learnable Cross-modal Codebook for Sequential Recommendation." 2024.

[^li2024letter]: Li, Y., et al. "LETTER: Learnable Tokenizer for Sequential Recommendation." 2024.

[^glass2024]: "GLASS: Generative Large-Scale Semantic Search." 2024.

[^uxsid2024]: "UxSID: Ultra-Long Sequence Modeling with Semantic ID for Ranking." 2024.

[^dig2024]: "DIG: Discrimination Is Generation." 2024.
