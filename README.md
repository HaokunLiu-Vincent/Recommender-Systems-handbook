# Recommender Systems — 模型、算法、术语与评估指标手册

> 面向学习与工程实践的 RecSys 速查手册：**先说明是什么、为什么需要，再用公式与例子解释怎样工作。**  
> 适用场景：电商推荐；项目实例采用 **H&M Personalized Fashion Recommendations**（预测未来 7 天购买的 `article_id`，主要指标 MAP@12）。  
> 文档类型：知识笔记 / Reference，不代表这些模型均已在项目中训练或上线。最近整理：2026-10-30。

## 目录（Outline）

- [1. 推荐系统的整体流程](#1-推荐系统的整体流程)
- [2. 名词与缩写速查表](#2-名词与缩写速查表)
- [3. 召回 Retrieval：模型与原理](#3-召回-retrieval模型与原理)
  - [3.1 Popularity / Trending](#31-popularity--trending)
  - [3.2 CF：UserCF / ItemCF](#32-cfusercf--itemcf)
  - [3.3 MF / Implicit ALS / WALS](#33-mf--implicit-als--wals)
  - [3.4 Two-Tower](#34-two-tower)
  - [3.5 LightGCN](#35-lightgcn)
  - [3.6 CBF：Text / Image Embedding](#36-cbftext--image-embedding)
- [4. 相似度、向量和 FAISS](#4-相似度向量和-faiss)
- [5. 多路召回融合 Fusion](#5-多路召回融合-fusion)
- [6. 排序 Ranking：LightGBM / FM / DeepFM](#6-排序-rankinglightgbm--fm--deepfm)
- [7. Re-ranking、冷启动与兜底](#7-re-ranking冷启动与兜底)
- [8. Metrics：Precision / Recall / AP / MAP / NDCG](#8-metricsprecision--recall--ap--map--ndcg)
- [9. 训练与离线评估：如何避免数据泄漏](#9-训练与离线评估如何避免数据泄漏)
- [10. 数据工程与部署术语：Parquet / DuckDB / Feast / FAISS](#10-数据工程与部署术语parquet--duckdb--feast--faiss)
- [11. H&M 项目中的具体定位](#11-hm-项目中的具体定位)
- [12. 最小实验矩阵与 Checklist](#12-最小实验矩阵与-checklist)
- [13. 常见易混概念与问答](#13-常见易混概念与问答)
- [14. 参考链接](#14-参考链接)

---

## 1. 推荐系统的整体流程

**笼统解释**：推荐系统通常不是让一个模型在全站所有商品中直接输出最终 12 件。它常被拆成“生成候选 → 给候选打分排序 → 调整最终列表”。

```text
历史购买 / 用户信息 / 商品属性
             │
       特征工程 + 模型训练
             │
多路 Retrieval / Recall（高覆盖、低成本）
             │ 每路 Top-K
       Fusion（Union + Dedup）
             │ 合并候选、保留来源和各路分数
      Pre-ranking（如 LightGBM）
             │
   Fine-ranking（可选 DeepFM）
             │
 Re-ranking（约束、多样性、兜底）
             │
         推荐 Top 12
             │
    未来 7 天真实购买 → MAP@12
```

- **Retrieval / Candidate Generation / Recall（召回）**：尽可能找回可能相关的商品；每路依赖不同信号。
- **Fusion（融合）**：去重、汇总候选、保留每路命中标记/排名/分数；候选太多才需进一步截断。
- **Ranking（排序）**：对已召回候选进行更复杂的相关性预测；**无法找回召回阶段遗漏的商品**。
- **Re-ranking（重排）**：处理约束、重复、商品多样性等最终展示问题。

**Retrieval 不等于一定算相似度**：向量召回通常计算相似度或内积；Popularity 根据历史热度而非 user-item 向量相似度召回。

## 2. 名词与缩写速查表

| 缩写 / 名词 | 英文全称 | 中文 / 核心意义 | 分类 |
|---|---|---|---|
| RecSys | Recommender System | 推荐系统 | 系统 |
| Retrieval / Recall | Candidate Retrieval / Candidate Generation | 候选召回 | 流程阶段 |
| CF | Collaborative Filtering | 协同过滤：通过用户–物品交互推断偏好 | 方法家族 |
| UserCF | User-based Collaborative Filtering | 找相似用户，再推荐其喜欢的商品 | 算法 |
| ItemCF / ItemKNN | Item-based Collaborative Filtering / Item k-Nearest Neighbors | 找与历史购买商品相似的商品 | 算法 |
| CBF | Content-Based Filtering | 基于文本、图片、属性等内容推荐 | 方法家族 |
| MF | Matrix Factorization | 矩阵分解，学习用户和商品潜在向量 | 方法家族 |
| ALS | Alternating Least Squares | 交替最小二乘 | 优化方法 |
| Implicit ALS | Implicit-Feedback Alternating Least Squares | 面向隐式反馈的加权矩阵分解 | 模型 / 训练方法 |
| WALS | Weighted Alternating Least Squares | 加权交替最小二乘 | 优化方法 |
| Two-Tower | Two-Tower Model / Dual Encoder | 用户塔和商品塔分别编码，再匹配 | 召回架构 |
| GCN | Graph Convolutional Network | 图卷积网络 | 模型家族 |
| LightGCN | Light Graph Convolutional Network | 简化的图协同过滤模型 | 召回模型 |
| SBERT | Sentence-BERT / Sentence Transformers | 句子级语义向量模型 | 内容编码模型 |
| CLIP | Contrastive Language–Image Pre-training | 图文对比预训练 | 多模态编码模型 |
| ANN | Approximate Nearest Neighbor | 近似最近邻搜索 | 检索算法家族 |
| FAISS | Facebook AI Similarity Search | 向量索引与相似性检索库 | 软件库 |
| IVF | Inverted File Index | 聚类中心 + 倒排向量列表 | ANN 索引 |
| HNSW | Hierarchical Navigable Small World | 分层近邻图索引 | ANN 索引 |
| PQ | Product Quantization | 乘积量化 / 向量压缩 | 索引压缩 |
| RRF | Reciprocal Rank Fusion | 倒数排名融合 | 融合方法 |
| GBDT | Gradient Boosted Decision Trees | 梯度提升决策树 | 模型家族 |
| LightGBM | Light Gradient Boosting Machine | 高效梯度提升树框架 | 模型框架 |
| FM | Factorization Machine | 因子分解机，建模二阶特征交互 | 模型 |
| DNN | Deep Neural Network | 深度神经网络 | 模型家族 |
| DeepFM | Deep Factorization Machine | FM + DNN 联合建模特征交互 | 排序模型 |
| CTR | Click-Through Rate | 点击率 | 业务指标 / 标签 |
| CVR | Conversion Rate | 转化率 | 业务指标 / 标签 |
| P@K | Precision at K | 前 K 个预测中命中的比例 | 评估指标 |
| R@K | Recall at K | 真实相关商品中被前 K 个找回的比例 | 评估指标 |
| AP@K | Average Precision at K | 单用户按命中位置累积的平均精确率 | 评估指标 |
| MAP@K | Mean Average Precision at K | 跨用户 AP@K 平均 | 评估指标 |
| DCG@K | Discounted Cumulative Gain at K | 位置折扣后的相关性总增益 | 评估指标 |
| NDCG@K | Normalized Discounted Cumulative Gain at K | DCG 除以理想 DCG | 评估指标 |
| PIT | Point-in-Time | 按事件发生时可用的信息做历史特征查询 | 数据正确性 |
| OOV | Out of Vocabulary | 未见过的 ID / token | 冷启动 / 特征 |
| Embedding | Vector Representation | 模型学习的稠密向量表示 | 数据表示 |
| Negative Sampling | Negative Sampling | 从未观测商品中采样训练负例 | 训练策略 |
| Baseline | Baseline Model | 基准方案，用于对比新方法 | 实验角色 |
| Fallback | Fallback Strategy | 失败或数据不足时的兜底推荐 | 线上机制 |

> **分类层级不要混淆**：CF/CBF 是方法家族；ALS 是优化方法；LightGCN/DeepFM 是具体模型；LightGBM 是软件框架；FAISS 是检索库；Recall/Ranking 是流程阶段。

## 3. 召回 Retrieval：模型与原理

### 3.1 Popularity / Trending

**是什么**：用历史最受欢迎商品产生候选。**为什么要用**：实现简单、稳定，是有效的 Baseline 与 Fallback。

例如，在截止日之前的 7 天统计每件商品的购买次数：

\[
score(i) = \sum_{(u,i,t)\in\text{history}} \mathbf{1}[t\ge T-7\text{d}]
\]

也可加入时间衰减：近期交易权重更大。**注意**：它不需要计算用户–商品相似度；Top 热门商品不能使用预测窗口里的未来交易数据。

### 3.2 CF：UserCF / ItemCF

**是什么**：协同过滤通过“谁买过什么”学习关联；无需商品文本或图片。

- **UserCF**：找与目标用户行为相似的用户，再推荐他们购买过的商品。
- **ItemCF**：离线计算商品–商品协同相似度；推理时查询用户过去购买商品的相似邻居。

例子：买过 A 和 B 的许多用户也买了 C，那么 C 是 A/B 的潜在协同近邻，即使 C 的文字描述与 A/B 不相似。

一种简单共现分数（示意，并非唯一 ItemCF 定义）：

\[
s(i,j)=\frac{|U_i\cap U_j|}{\sqrt{|U_i|\,|U_j|}}
\]

这里 \(U_i\) 是购买过商品 \(i\) 的用户集合。分母抑制极度热门商品带来的虚假关联。真实实现还需处理高活跃用户、时间、重复购买和流行度偏差。

**工程实践**：不要保存全部 \(N\times N\) 相似度；为每个商品保存 Top-L 邻居表即可。

### 3.3 MF / Implicit ALS / WALS

**是什么**：把稀疏用户–商品交互矩阵分解为用户向量和商品向量，再用内积估算匹配度。

\[
\hat r_{ui}=p_u^\top q_i
\]

**本质机制**：令用户向量固定，优化商品向量；再固定商品向量，优化用户向量，交替直到收敛（ALS）。

隐式反馈里，购买表示“观察到偏好信号”，未购买**不等于明确不喜欢**。经典隐式 ALS 用偏好 \(p_{ui}\in\{0,1\}\) 与置信度 \(c_{ui}\) 构造加权目标：

\[
\min_{x,y}\sum_{u,i}c_{ui}(p_{ui}-x_u^\top y_i)^2
+\lambda\left(\sum_u\|x_u\|^2+\sum_i\|y_i\|^2\right)
\]

常见 \(c_{ui}=1+\alpha r_{ui}\)（\(r_{ui}\) 可为行为次数）。**WALS** 强调加权目标，**Implicit ALS** 是此类算法的常见实现；两者有关联但不是任何语境下都完全同义。

### 3.4 Two-Tower

**是什么**：分别使用 User Tower 和 Item Tower 把用户与商品编码到可匹配的向量空间。

\[
u=f_\theta(\text{user/history}),\qquad v_i=g_\phi(\text{item features})
\]
\[
score(u,i)=u^\top v_i
\]

**为什么适合召回**：所有商品向量可以离线预先算好；推理时只需生成当前用户向量，然后搜索相匹配的 Top-K 商品。

**例子**：用户曾买过运动外套、跑鞋；User Tower 输出 128 维 \(u\)。Item Tower 已经给 10 万件衣物生成 128 维 \(v_i\)。检索 \(u^\top v_i\) 最大的候选即可。

**训练要点**：正样本来自购买记录；负样本可能来自 batch 内其他商品或额外采样。必须意识到 in-batch negatives、热门商品采样偏差和 false negatives 的问题。双塔支持高效独立编码，但复杂用户–商品交叉特征通常留给后续 Ranking。

### 3.5 LightGCN

**是什么**：将用户与商品构成二部图，用相邻节点 Embedding 传播学习协同信息。

\[
e_u^{(l+1)}=\sum_{i\in N(u)}\frac{e_i^{(l)}}{\sqrt{|N(u)|\,|N(i)|}}
\]

层数不同，覆盖一阶和多跳协同信号；最终结合多层表示，再用用户与商品 Embedding 内积评分。与一般 GCN 相比，LightGCN 去掉特征线性变换与非线性激活，聚焦图上邻居聚合。

**什么时候值得用**：购买二部图连接度足够、相较 ALS/ItemCF 有可测增益；其训练、采样与图传播成本不可忽略。

### 3.6 CBF：Text / Image Embedding

**是什么**：即使用户–商品交互较少，也能用内容属性找到语义/视觉近邻。

- **Text**：把商品名、品类、描述编码成文本向量（例如 SBERT）。
- **Image**：把商品图片编码成视觉向量（例如 CLIP Image Encoder）。
- **User profile**：可以对用户历史购买商品向量求加权平均，得到用户偏好向量，再搜相似商品。

\[
q_u=\frac{\sum_{i\in H_u} w_i v_i}{\sum_{i\in H_u} w_i},\qquad score(u,j)=\cos(q_u,v_j)
\]

**局限**：视觉相似不等于购买偏好相同；新商品有内容不一定有足够交互验证。图片文件存在也不等于图片成功解码或已生成有效 Embedding。

## 4. 相似度、向量和 FAISS

### 4.1 Dot Product vs Cosine

\[
\text{Dot}(u,v)=\sum_{d=1}^D u_dv_d
\]
\[
\cos(u,v)=\frac{u^\top v}{\|u\|\,\|v\|}
\]

Dot Product 同时受夹角与向量长度影响；Cosine 只看方向。**只有都做了 L2 归一化时，Dot Product 才等价于 Cosine**。Two-Tower 的训练目标与检索索引度量必须保持一致。

### 4.2 为什么需要 FAISS？

**直观原理**：精确搜 100 万个 128 维向量，需要对全部向量打分；FAISS 是提供多种检索实现的库，并非一种会自动加速的文件格式。

- **IndexFlatIP**：精确内积检索，依然扫描全部向量；可作为准确结果 Ground Truth。
- **IVFFlat**：离线将商品向量聚成 `nlist` 组；推理先找最匹配的 `nprobe` 个中心，只扫描这些组里的商品。大幅减少点积数，但**可能错过真正 Top-K**。
- **HNSW**：预先构建分层邻居图，查询时从入口节点沿近邻边逐步走向更匹配的商品；不遍历所有节点，也可能漏掉精确最近邻。
- **PQ**：压缩向量以节省内存，可能再牺牲一些精度。

**示例（仅数学示意）**：100 万商品、1,000 个均匀簇、`nprobe=10`，大约要比较 1,000 个簇中心 + 10,000 个商品，而不是 100 万个商品。实际簇不均衡，真实延迟还取决于内存访问、线程、硬件和索引实现。

```python
import faiss
import numpy as np

# item_vectors: shape (N, 128)，float32；两塔 item 向量需使用一致的度量
item_vectors = np.ascontiguousarray(item_vectors, dtype=np.float32)
index = faiss.IndexFlatIP(128)  # 精确 baseline，并不会跳过向量
index.add(item_vectors)

user_vector = np.ascontiguousarray(user_vector.reshape(1, 128), dtype=np.float32)
scores, row_indices = index.search(user_vector, 100)
# row_indices 是索引位置；务必保存独立的 row_index -> article_id 映射
```

**Benchmark**：以 FlatIP 的 Top-100 为真值，计算 `ANN Recall@100 = |ANN Top100 ∩ Exact Top100| / 100`，并比较 P50/P95 latency、内存和最终 MAP@12。**ANN Recall@100 与针对真实购买标签的 RecSys Recall@100 不是同一指标。**

### 4.3 存储规模直觉

100 万商品 × 128 维 × FP32（4 bytes）≈ 488 MiB 的**原始向量数据**。索引结构、ID 映射、复制、模型及运行时内存另外计算。将 `.faiss` 文件保存在 S3 后，在线服务通常在启动时下载并加载到内存，而不是每个请求都重新下载。

## 5. 多路召回融合 Fusion

**为什么要多路**：ItemCF 捕捉共购关系、ALS 捕捉矩阵潜因子、Two-Tower 学非线性匹配、内容召回捕捉语义/视觉、Popularity 为冷启动兜底。它们互补，不保证越多越好。

**最低成本的 Fusion**：

1. 每路产出 `(user_id, article_id, source, rank, raw_score)`。
2. `Union + Dedup`，同一 `article_id` 只保留一个候选。
3. 保留 `was_recalled_by_itemcf`、`itemcf_rank`、`als_score`、`source_count` 等特征。
4. 如果要控制候选总数，可用 RRF 排序截断，但检查各路独有相关商品是否被误删。

**RRF（Reciprocal Rank Fusion）**：

\[
RRF(i)=\sum_{c:i\in L_c}\frac{w_c}{k+rank_c(i)}
\]

通常先用 `w_c=1` 和固定 `k`（例如 60）作为不调权重的 Baseline。RRF 按**名次**而非原始分数融合，因为不同路的 ALS 内积、Cosine、ItemCF 相似度不在同一尺度上。

**为何 LightGBM 不替代 Fusion**：LightGBM 只能重新排列已进入候选池的商品；若 Fusion 过早丢掉真阳性，排序模型无法补救。

## 6. 排序 Ranking：LightGBM / FM / DeepFM

### 6.1 Ranking 的工作单元

一条训练样本通常是 `(user_id, article_id, cutoff_time, features, label)`。特征可来自用户统计、商品类别、近期热度、历史购买、各路 Recall 的排名/分数、价格、内容向量派生分数。标签可定义为用户在未来 7 天**是否购买该 article**。

### 6.2 LightGBM

**是什么**：Light Gradient Boosting Machine，属于高效 GBDT 框架，不是一种协同过滤算法。

**本质机制**：每一轮训练一棵树拟合当前模型损失函数的梯度方向，多个弱树逐步形成非线性预测函数。树能自动学习阈值和特征交互（例如“最近 7 天买鞋超过 3 次且这件鞋的 Two-Tower 排名很高”）。

- 分类目标：可以学习购买与否的分数/概率，但离线任务不自动等同 CTR 或 CVR。
- Learning-to-Rank：也可用 LambdaRank 等目标；需按照用户（或 user-cutoff query）正确组织分组和标签。
- 特征缺失：召回通道未命中时，建议同时保留 missing indicator，不能机械地将“未召回”当作“0 相似度”。

### 6.3 FM 与 DeepFM

**FM（Factorization Machine）**：使用低维因子学习二阶交互：

\[
\hat y=w_0+\sum_i w_ix_i+\sum_{i<j}\langle v_i,v_j\rangle x_ix_j
\]

例如在稀疏特征下，模型可学习“某年龄段 × 某类商品”的交互，不需要为每个组合独立存一个参数。

**DeepFM（Deep Factorization Machine）**：FM 分支负责低阶特征交互，DNN 分支学习更复杂的高阶非线性组合，共享部分输入 Embeddings，联合训练。

**工程取舍**：DeepFM 未必一定优于 LightGBM；如果 MAP@12 没有稳定增加但推理延迟显著升高，可以保留 LightGBM 为最终方案。

## 7. Re-ranking、冷启动与兜底

- **Re-ranking**：对已排序候选应用品类多样性、商品规则、重复款式约束、是否允许再次推荐已购买商品等规则。**是否过滤历史购买需要业务和实验验证**，服装商品可能重复购买。
- **Cold-start User**：新用户无交互，协同模型无法产生可信个性化向量；可用热门商品、用户画像或上下文内容。
- **Cold-start Item**：新商品缺购买历史；内容召回更有机会覆盖。
- **Fallback**：服务故障、索引不可用或候选太少时，用 Popularity/Trending 补足，而非返回空列表。
- **Baseline ≠ Fallback**：Baseline 用于实验比较；Fallback 用于线上可靠性。同一热门推荐算法可以兼任两种角色。

## 8. Metrics：Precision / Recall / AP / MAP / NDCG

### 8.1 统一符号

对一个用户，令：

- \(G\)：未来 7 天真实购买的**不同**商品集合，\(|G|\) 是实际购买的不同商品数量。
- \(L_K\)：模型输出的前 K 个**不同**商品。
- \(h=|G\cap L_K|\)：命中数。
- \(rel(k)=1\) 表示第 k 位商品属于 \(G\)，否则为 0。

**一句话**：Precision 的分母是**推荐数**；Recall 的分母是**实际相关商品数**。这是最容易混淆的地方。

### 8.2 Precision@K 与 Recall@K

\[
Precision@K=\frac{h}{K},\qquad Recall@K=\frac{h}{|G|}
\]

（上述 Precision 约定输出恰好 K 件不同商品；若不足 K 件，要提前固定评估约定。）

**例子**：未来实际购买 `[A,B,C,D,E]`，推荐 Top-12 只有 `A,B,C` 命中：

\[
P@12=3/12=0.25,\quad R@12=3/5=0.60
\]

**如果只实际买了 2 件**，且都出现在推荐的 Top-12，`Recall@12 = 2/2 = 1`，但 `Precision@12 = 2/12 ≈ 0.167`。二者可以相差很大。

### 8.3 AP@K 与 MAP@K

**AP（Average Precision）** 是单个用户的排序得分：只有命中的位置才贡献该位置的 `Precision@k`，更靠前命中通常得分越高。

采用常见的截断版本：

\[
AP@K(u)=\frac{\sum_{k=1}^{K}Precision@k(u)\cdot rel_u(k)}{\min(K,|G_u|)}
\]

**MAP（Mean Average Precision）** 再跨参与评分的用户平均：

\[
MAP@K=\frac{1}{|U_{eval}|}\sum_{u\in U_{eval}}AP@K(u)
\]

**例子**：实际买了 `A,B,C`；模型在第 1、3、6 位命中它们：

\[
AP@12=\frac{1/1+2/3+3/6}{3}\approx 0.7222
\]

**如果只买 1 件**，且命中在第 5 位，AP@12 = 1/5 = 0.2；如果 3 件都排在前三，AP@12 = 1。**分母不是固定 12**。

> **评估实现约定很重要**：AP@K 在不同库/比赛中可能有不同归一化与去重细节。官方比赛提交与评分规则应优先于通用教程。对于未来 7 天没有购买记录的用户，必须明确其是否参与离线指标均值；不要擅自把无标签用户的 AP 设为 0 并与官方结果比较。

### 8.4 DCG@K 与 NDCG@K

**DCG** 按位置折扣相关性；**NDCG** 用最佳可能的 DCG 做归一化：

\[
DCG@K=\sum_{k=1}^K\frac{2^{rel(k)}-1}{\log_2(k+1)},\quad
NDCG@K=\frac{DCG@K}{IDCG@K}
\]

对于购买 / 未购买二元标签，\(rel(k)\in\{0,1\}\) 时，每个命中位置贡献 \(1/\log_2(k+1)\)。

**MAP vs NDCG**：两者都关注排名；MAP 看命中点的前缀精确率，NDCG 看位置折扣并可自然支持多级相关性。它们的数值**不能直接横向比较大小**，尤其 K 不同时。

### 8.5 示例对照：相同 Recall，不同 MAP

真实购买 `A,B,C`，比较前 5 名：

| 模型 | 推荐前 5 | Recall@5 | AP@5 | NDCG@5（二元） |
|---|---|---:|---:|---:|
| M1 | `A,B,X,Y,C` | 1.000 | 0.867 | ≈ 0.947 |
| M2 | `X,A,B,C,Y` | 1.000 | 0.639 | ≈ 0.733 |

三个商品都找回了，所以 Recall 一样；M1 排得更靠前，所以 AP/NDCG 更高。

### 8.6 Offline / Online / 系统指标

| 层次 | 常见指标 | 回答的问题 |
|---|---|---|
| 向量索引 | ANN Recall@K、P50/P95 latency、RAM | 近似检索漏了多少“精确近邻”？ |
| 候选召回 | Candidate Recall@K、coverage、unique relevant contribution | 真实购买商品有多少进入候选池？ |
| 最终排序 | **MAP@12**、NDCG@12、Recall@12 | 候选是否排对、前 12 质量如何？ |
| 可靠性 | Timeout rate、fallback rate、index load success | 服务是否稳定可用？ |
| 真实线上业务（有真实日志时） | CTR、CVR、Revenue、A/B uplift | 推荐是否改善了真实用户行为？ |

**重要**：H&M 历史购买数据不包含完整曝光与点击日志，不能从离线购买数据直接声称真实 CTR/CVR 或 A/B 提升。

## 9. 训练与离线评估：如何避免数据泄漏

**任务定义**：在截止时间 \(T\)，只使用 \(T\) 之前可获得的信息，预测 `(T, T+7天]` 用户购买的商品。

```text
──────── 训练历史 ────────│截止时间 T│──── 未来 7 天标签 ────
      只能在左边构造特征                  仅用于评估
```

核心约束：

1. **Chronological Split（按时间切分）**，而非对交易行随机切分。
2. ItemCF 共现、Popularity 趋势、ALS/Two-Tower/LightGCN 的交互训练数据，都必须来自截止日期之前。
3. 特征表最好携带 `event_timestamp`；采用 Point-in-Time Join，不能把未来统计值拼进过去样本。
4. Ranker 训练候选应由历史可用的召回模型产生，不得根据真实未来标签“反选”候选。
5. 没购买过的商品是 **unobserved**，不代表真正不喜欢。负采样会影响训练分布。
6. 统计“交易行数”“用户–商品去重边数”“未来 7 天不同商品标签数”时，grain 要一致；同一天同一商品重复购买不应被不加说明地当作脏数据删除。
7. 没有对应历史销售上架快照时，不要宣称准确复原当日可售商品；需记录静态商品元数据假设。

## 10. 数据工程与部署术语：Parquet / DuckDB / Feast / FAISS

**先用一句话区分**：

| 工具 / 技术 | 是什么 | 负责什么 | 不负责什么 |
|---|---|---|---|
| **Parquet** | 列式数据文件格式 | 存放结构化特征、统计表等 | 不自行执行查询 |
| **DuckDB** | 嵌入式 SQL 分析引擎 | 高效查询 CSV/Parquet、Join、Feature Engineering | 不是特征管理平台 |
| **Feast** | Feature Store 框架 | Feature 定义、历史 PIT 查询、离线到在线特征 materialization / 统一读取接口 | 不代替 ETL，也不自动负责 ANN 检索 |
| **FAISS** | 向量检索库 | 建立/加载索引，按 Dot/Cosine 等查 Top-K | 不管理全部业务特征 |
| **S3** | 对象存储 | 持久化 CSV/Parquet、模型、Embeddings、`.faiss` 文件 | 不直接替代服务内存的低延迟索引 |
| **DynamoDB / Redis** | 在线 KV / 数据存储 | 根据用户/商品 ID 快速读取最新特征 | 不是向量召回算法本身 |
| **DVC** | 数据/模型文件版本管理工具 | 跟踪大文件版本与 Git 提交的对应关系 | 不等于 Feature Store |
| **MLflow** | 实验跟踪与模型管理工具 | 记录参数、指标、模型产物 | 不代替特征计算 |
| **FastAPI** | Python Web API 框架 | 提供 `/recommend` 等接口 | 不负责模型训练 |

**典型文件流**：

```text
S3 / CSV 原始购买数据
   └── DuckDB / Python 特征工程
       ├── user_features.parquet / item_features.parquet
       ├── 模型训练 → model artifacts
       └── 商品 Embeddings → FAISS index → items.faiss

FastAPI 服务启动：加载模型 / ID mapping / FAISS index
FastAPI 请求到来：user_id → 用户特征/向量 → FAISS Top-K → Fusion → Rank → Top-12
```

**Feast 什么时候需要？** 若主要是一次性离线训练、Parquet 文件、静态特征 + 本地/Colab 训练，完全可以先不用 Feast；当多个模型共享特征、线上特征频繁更新、需要一致的历史/在线读取时，它才更有价值。

**VS Code + Colab 的注意事项**：Notebook 文件在本地打开 ≠ Python 在本地执行。远程 Colab Kernel 默认不能直接 import 仅存在于本地 VS Code 工作区的模块；需把代码同步到远程环境（如 Git clone、项目安装）。

## 11. H&M 项目中的具体定位

> **下表是已拟定的路线 / 候选方案，不是所有模型已实现。** 当前数据集是 H&M 购买日志，**不是**旧版 Amazon Reviews'23 评论数据。现有 Phase 0 EDA 已完成基础表格统计；时间切分、模型训练与 Benchmark 尚需后续阶段验证。

| 状态 | Lane / 模型 | 主要输入信号 | 推荐系统阶段 | 主要检验 |
|---|---|---|---|---|
| 核心 | Popularity / Trending | 历史购买数、近期变化 | Recall / Fallback | 热门 Baseline，冷启动覆盖 |
| 核心 | ItemCF | User–Item 共购关系 | Recall | 额外命中与共现计算成本 |
| 核心 | Implicit ALS / WALS | 隐式反馈交互矩阵 | Recall | 与 ItemCF/Popularity 的增益 |
| 核心 | Two-Tower | 用户历史 + 商品特征 | Recall | MAP/Recall 增量、向量检索效率 |
| 条件性 | LightGCN | 二部图多跳协同信号 | Recall | 质量增量 vs 图训练成本 |
| 条件性 | Text Embedding | 商品名、类目、描述 | Recall | 新商品覆盖与语义增量 |
| 条件性 | Image Embedding | 商品图片 | Recall | 图片质量、覆盖、实际贡献 |
| 排序 Baseline | LightGBM | 用户/商品/召回来源特征 | Pre-ranking / Ranking | MAP@12、性能 |
| 可选 Challenger | DeepFM | 稀疏与稠密交叉特征 | Fine-ranking | 增量是否值得延迟 |
| 最终规则 | Re-ranking | 重复、品类、可用性约束 | 输出 Top-12 | 最终 MAP@12、覆盖率 |

### H&M 数据规模（Phase 0 已审计）

- 31,788,324 条购买记录；1,371,980 个客户；105,542 个 `article_id`。
- 27,306,439 个不同的 customer–article 交互对。
- 商品图片存在性：105,100 个唯一有效 JPG 文件名能映射到商品，约 99.58% 的商品有对应文件名；**这不等于全部图片已能解码，也不等于已有图像 Embeddings**。
- `article_id` 是保留前导零的字符串；不要误与 `product_code` 合并。

来源：本项目 `phase0-data download & EDA report.md` 与 `proposal.md`（2026-10-29）。

## 12. 最小实验矩阵与 Checklist

| 对比 | 固定条件 | 检验目的 |
|---|---|---|
| Popularity vs ItemCF vs ALS vs Two-Tower | 同一时间切分、合格商品、用户群、Top-K | 各 Recall Lane 的真正增益 |
| Best single lane vs Union / RRF | 同一候选预算 | 融合是否增加 Candidate Recall |
| FAISS FlatIP vs IVF vs HNSW | 同一向量、度量、Top-K、硬件 | ANN Recall vs P95 vs 内存 |
| RRF vs LightGBM | 同一候选池 | Ranker 是否优于简单排序 |
| LightGBM vs LightGBM + DeepFM | 同一用户、候选与标签 | Fine-ranking 是否有净收益 |
| 有/无 Re-ranking | 相同模型分数与合法候选 | 多样性/规则是否伤害 MAP |

**复现实验前的 Checklist**：

- [ ] 冻结 train/validation/test 的时间边界和评估用户集合。
- [ ] 明确未来 7 天 label 是不同 `article_id` 集合，而非购买数量。
- [ ] 固定处理“未来没有购买”的用户的评估约定，并对齐官方规则。
- [ ] 固定候选 Item eligibility、已购商品政策及 cold/warm cohorts。
- [ ] 只用 cutoff 之前信息构建召回与特征；检查泄漏。
- [ ] 记录 Candidate Recall@K、MAP@12、NDCG@12、P95 latency、内存。
- [ ] 保存模型、索引、ID mapping 和对应数据版本；重载验证结果一致。
- [ ] 对增加的每路模型做消融，证明确实产生独有相关候选或排序增益。

## 13. 常见易混概念与问答

<details>
<summary><strong>Q1. Recall@20 是命中商品数 / 20 吗？</strong></summary>

**不是**。这是推荐恰好 20 件商品时的 `Precision@20`。`Recall@20 = 命中数 / 用户真实相关的不同商品数`。
</details>

<details>
<summary><strong>Q2. 用户未来只购买了 3 件，MAP@12 怎么计算？</strong></summary>

仍推荐至多 12 件，单用户 AP@12 用命中位置的 Precision 累加，常见截断定义的分母为 `min(12, 3)=3`，不是 12。比赛细节以官方评分规则为准。
</details>

<details>
<summary><strong>Q3. MAP@12 和 NDCG@12 都考虑排序，为什么两个指标不同？</strong></summary>

MAP 按**命中时的前缀 Precision**奖励排序；NDCG 按**位置折扣**奖励相关性（并支持多等级相关性）。不能简单比较数值大小。
</details>

<details>
<summary><strong>Q4. 每个 Recall Lane 必须手动设置权重吗？</strong></summary>

不需要。先试 `Union + Dedup`；如需压缩候选，用等权 RRF Baseline。后续让 LightGBM 学习各路 rank/score/missing/source_count 等特征，而不是先手调 7 个权重。
</details>

<details>
<summary><strong>Q5. FAISS 文件为什么查询快？</strong></summary>

关键不在文件扩展名，而在**索引算法**。FlatIP 没有避免逐一计算；IVF 提前分簇，只查部分簇；HNSW 建图后局部搜索。后二者是近似检索，要做准确率与速度的 Benchmark。
</details>

<details>
<summary><strong>Q6. 已有 Parquet，是否必须上 Feast？</strong></summary>

不是。Parquet 存数据，DuckDB 处理数据，Feast 管理离线/在线特征定义和读取。对离线 MVP，Feast 可选而非必需。
</details>

<details>
<summary><strong>Q7. Popularity 为什么既是 Baseline 又是 Fallback？</strong></summary>

Baseline 用来**比较实验效果**；Fallback 用来**保证系统在缺少用户信息或模型失败时仍有结果**。一个简单热门榜可以承担两种角色。
</details>

## 14. 参考链接

- [H&M Personalized Fashion Recommendations — Kaggle](https://www.kaggle.com/competitions/h-and-m-personalized-fashion-recommendations)
- [FAISS 官方文档与索引说明](https://github.com/facebookresearch/faiss/wiki/Faiss-indexes)
- [LightGBM 官方文档](https://lightgbm.readthedocs.io/)
- [Feast 官方文档](https://docs.feast.dev/)
- [LightGCN 论文](https://arxiv.org/abs/2002.02126)
- [DeepFM 论文](https://arxiv.org/abs/1703.04247)
- [Two-Tower 检索模型工程介绍（TensorFlow Recommenders）](https://www.tensorflow.org/recommenders/examples/basic_retrieval)
- [Implicit Feedback ALS 论文](https://doi.org/10.1109/ICDM.2008.22)

---

**阅读方式**：GitHub 将 README 自动渲染并支持标题跳转；VS Code 可使用 `Markdown: Open Preview to the Side`（`Cmd+K` 后按 `V`）。macOS **Preview（预览）App** 不原生支持 Markdown 渲染；要用带目录的阅读体验，建议 VS Code、GitHub 或将 Markdown 转成 PDF/HTML。
