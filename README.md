# 手写 RAG 文档问答

不依赖 LangChain，用约 200 行 Python 从零实现「切块 → 向量化 → 召回 → 重排 → 生成」全链路。

输入一篇文档，就能针对它提问；文档里没有的内容，系统会明确拒答，而不是编一个答案。

> 想一步步跟着走：看 [`tutorial.ipynb`](./tutorial.ipynb)。想直接跑起来：看 [`pipeline.py`](./pipeline.py)。

```
问题：安子的爸爸是谁
召回   0.595  和安子一样工作的还有她的父亲…因为姓马，大家喜欢叫他老马
       0.555  安子每天都来桥边卖玻璃弹珠…
重排   +3.81  和安子一样工作的还有她的父亲…因为姓马，大家喜欢叫他老马
       -0.81  安子每天都来桥边卖玻璃弹珠…
答案   安子的爸爸是马先生，大家都喜欢叫他老马。
```

## 为什么不用 LangChain

不是为了炫技。教程里 LangChain 版通常是这样：

```python
vectorstore = Chroma.from_documents(chunks, embeddings)
docs = vectorstore.similarity_search(query, k=3)
```

三行跑完，但有三件事你看不见：

- 检索出来的分数是多少、能不能筛
- 不相关的结果怎么处理（默认是照单全收喂给模型）
- 相似度到底是哪种距离，`0.75` 这个数字意味着什么

这个项目把每一段摊开写，代价是多写 150 行，收益是每个环节都能改、每个数字都能解释。

## 五个环节

| 环节 | 实现 | 关键点 |
| --- | --- | --- |
| 切块 | `split_into_chunks()` | 按空行切；短段落可合并到 400 字左右，避免块太碎 |
| 向量化 | `sentence-transformers` + `bge-small-zh-v1.5` | 512 维，95MB，CPU 可跑；文档块不加前缀，查询加检索指令 |
| 建库 | ChromaDB `PersistentClient` | 显式设 `hnsw:space="cosine"`，距离直接是 `1 - cos` |
| 召回 | 向量检索 Top-10 | 多召回几乎零成本，为重排留余地 |
| 重排 | `CrossEncoder` | 分数 > 0 判为相关，< 0 判为不相关 |
| 生成 | OpenAI 兼容接口 | 只喂相关块；没有相关块就拒答，不调模型 |

## 两个设计决定

**1. 用 CrossEncoder 分数的正负号做相关性判据**

一般做法是「检索 Top-3 → 拼进 prompt」，检索不到也让模型硬答。这里把召回扩到 10 条，用 CrossEncoder 重新打分，只把**分数为正**的块喂给模型；全为负则直接返回「文档中没有相关信息」。

好处是阈值不用调。这个模型按「相关/不相关」二分类训练，正负号本身就是判据，换文档、换领域都不用重新试。

实测中相关与不相关的差距非常明显：

| 问题 | 最相关块 CE 分数 | 次高 | 处理 |
| --- | --- | --- | --- |
| 那个女人带来的玫瑰是什么样子的 | +2.95 | -3.54 | 喂 1 块，答对 |
| 安子今年几岁 | -3.60 | — | 不喂，拒答 |
| 移动AI梦的 3 个秘密道具 | -4.77 | — | 不喂，拒答 |

「安子几岁」是个典型例子：全文都在写安子，向量相似度不低，但没有一处提到年龄。纯向量检索会把无关块送进 prompt，模型只能瞎猜。

**2. 显式指定 cosine 距离空间**

Chroma 默认 `hnsw:space="l2"`，返回的是**平方**欧氏距离。向量归一化后 `cos = 1 - d/2`，但如果直接当 `1 - d` 用，算出来的相似度会整体偏低（0.667 变成 0.6）。

这类错误不会改变排序，检索结果依然是对的——所以极难发现，只有显示的分数是错的。建库时写死 `cosine` 就不存在这个换算问题。

## 快速开始

```bash
# 1. 装依赖（需要 uv，没有的话见 https://docs.astral.sh/uv/）
uv sync

# 2. 配置
cp .env.example .env
# 编辑 .env，填入 LLM_API_KEY

# 3. 换语料：把你的文档放进 doc.md
#    格式要求：段落之间空一行

# 4. 运行
uv run python pipeline.py "你的问题"
```

国内网络下载模型较慢时，在 `.env` 里打开 `HF_ENDPOINT="https://hf-mirror.com"`。首次运行会下载两个模型：向量模型约 95MB，重排模型约 450MB。

不想用 uv 的话：

```bash
pip install -r requirements.txt
python pipeline.py "你的问题"
```

### 在 notebook 里用

```python
from pipeline import ask, split_into_chunks, build_index

build_index(split_into_chunks("doc.md"))
result = ask("安子的爸爸是谁")

print(result["answer"])
for score, chunk in result["reranked"]:
    print(f"{score:+.2f}  {chunk[:40]}")
```

`ask()` 返回 `answer`（答案）、`recalled`（召回结果与分数）、`reranked`（重排结果）、`used`（实际喂给模型的块）、`time`（各阶段耗时）。

## 文件说明

```
pipeline.py         全链路，可直接运行
tutorial.ipynb      跟教程手敲的原始记录，逐格对应教程步骤
doc.md              示例语料（一篇约 3000 字的短篇，一行一段）
pyproject.toml      依赖声明
uv.lock             依赖版本锁定
.env.example        配置模板
```

## 从教程版到工程版

`tutorial.ipynb` 是跟着教程一步步敲下来的过程，`pipeline.py` 是在它基础上改出来的可运行版本。四处关键改动：

| 教程版 | 问题 | 工程版 |
| --- | --- | --- |
| `EphemeralClient()` | 数据只在内存，进程退出即丢 | `PersistentClient(path="./chroma_db")` 落盘 |
| `[embed_chunk(c) for c in chunks]` | 逐条编码，36 块编码 36 次 | `encode(chunks, ...)` 一次批量，快一个数量级 |
| 重排只取排序结果 | 分数没打出来，正负号的信息浪费了 | 打印分数，用 `score >= 0` 做相关性门槛 |
| 检索到什么就喂什么 | 无关问题也会硬答 | 全为负则不调模型，直接返回拒答 |

notebook 里保留了教程当时的写法（`retreive` 的拼写、`print(f"[{i} {chunks}\n]")` 打印整个 `chunks`），没有回头美化——它是过程记录，不是成品代码。

## 实测性能

CPU（无 GPU）、36 个文本块：

| 阶段 | 耗时 |
| --- | --- |
| 召回（向量检索 Top-10） | 约 20 ms |
| 重排（CrossEncoder，10 条） | 约 240 ms |
| 生成（glm-4.5-air，关思考） | 约 900 ms |

模型加载是一次性的：向量模型数秒，重排模型约 7 秒。所以两个模型都在模块级加载，不要放进函数里——否则每次调用都会重新加载。

## 已知限制

- **单文件语料**。`doc.md` 一个文件，没做多文档、增量索引、按来源过滤。
- **无分页信息**。切块后只保留文本，回答里无法标注「来自第几页」。处理 PDF 时需要自己保留页码。
- **中文专用**。`bge-small-zh` 和 mmarco 重排模型都偏中文，英文场景建议换 `bge-small-en`。
- **无评测集**。目前靠人工看检索结果判断好坏，没有 hit@k 之类的量化指标。
- **重排模型有长度上限**。`mmarco-mMiniLMv2` 的 `max_seq_length` 是 512 token，输入是「问题 + 文档块」拼接，块超过 400 字时后半段会被截断——切块别贪长。

## 参考

- [BAAI/bge-small-zh-v1.5](https://huggingface.co/BAAI/bge-small-zh-v1.5)
- [cross-encoder/mmarco-mMiniLMv2-L12-H384-v1](https://huggingface.co/cross-encoder/mmarco-mMiniLMv2-L12-H384-v1)
- [ChromaDB 距离函数说明](https://docs.trychroma.com/docs/collections/configure)

## License

MIT
