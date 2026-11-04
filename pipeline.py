"""
RAG 全链路，一个文件跑完：
  切块 -> 向量化 -> 建库 -> 召回 -> 重排 -> 生成

用法（命令行）：
    uv run python pipeline.py "安子的爸爸是谁"

用法（notebook 里）：
    from pipeline import ask
    result = ask("安子的爸爸是谁")
    print(result["answer"])
"""

import os
import time
from pathlib import Path

from dotenv import load_dotenv
from openai import OpenAI
import chromadb
from sentence_transformers import SentenceTransformer, CrossEncoder

# ---------------------------------------------------------------- 配置

load_dotenv()

EMBED_MODEL = "BAAI/bge-small-zh-v1.5"
RERANK_MODEL = "cross-encoder/mmarco-mMiniLMv2-L12-H384-v1"

# bge 系列做检索时，query 要加这句指令；建库时对文档块不加
QUERY_PREFIX = "为这个句子生成表示以用于检索相关文章："

DB_DIR = "./chroma_db"
COLLECTION_NAME = "doc"

# 模型加载一次，别放进函数里（CrossEncoder 加载要 7 秒）
embedding_model = SentenceTransformer(EMBED_MODEL)
cross_encoder = CrossEncoder(RERANK_MODEL)

llm_client = OpenAI(
    api_key=os.environ["LLM_API_KEY"],
    base_url=os.environ.get("LLM_BASE_URL", "https://open.bigmodel.cn/api/paas/v4"),
)
LLM_MODEL = os.environ.get("LLM_MODEL", "glm-4.5-air")

# 智谱 4.5 之后的模型默认带思考过程，回答前会先想很久；
# RAG 有明确上下文，关掉思考更快也更简洁。
# 注意：glm-5.3-flash 这类「始终思考」的模型不接受 disabled，会报 400。
THINKING = os.environ.get("LLM_THINKING", "").strip().lower()


# ---------------------------------------------------------------- 1. 切块

def split_into_chunks(path: str, target: int | None = None) -> list[str]:
    """读文本文件切块。

    target=None  按空行切，一段一块（doc.md 这种一行一段的文件最合适）
    target=400   把短段落合并到约 400 字，块更完整，适合长文
    """
    text = Path(path).read_text(encoding="utf-8")
    text = text.replace("\r\n", "\n").replace("\r", "\n")
    paras = [p.strip() for p in text.split("\n\n") if p.strip()]

    if target is None:
        return paras

    chunks, buf = [], ""
    for p in paras:
        if buf and len(buf) + len(p) > target:
            chunks.append(buf)
            buf = p
        else:
            buf = (buf + "\n" + p) if buf else p
    if buf:
        chunks.append(buf)
    return chunks


# ---------------------------------------------------------------- 2. 建库

def build_index(chunks: list[str], rebuild: bool = True) -> chromadb.Collection:
    """把块编码后写进 Chroma（磁盘持久化）。"""
    client = chromadb.PersistentClient(path=DB_DIR)

    if rebuild:
        try:
            client.delete_collection(COLLECTION_NAME)
        except Exception:
            pass

    collection = client.get_or_create_collection(
        name=COLLECTION_NAME,
        metadata={"hnsw:space": "cosine"},   # 距离 = 1 - cos，分数直接可读
    )
    if collection.count() == 0:
        vectors = embedding_model.encode(chunks, normalize_embeddings=True)
        collection.add(
            documents=chunks,
            embeddings=vectors.tolist(),
            ids=[str(i) for i in range(len(chunks))],
        )
    return collection


def get_collection() -> chromadb.Collection:
    return chromadb.PersistentClient(path=DB_DIR).get_collection(COLLECTION_NAME)


# ---------------------------------------------------------------- 3. 召回

def retrieve(query: str, top_k: int = 10) -> list[tuple[float, str]]:
    """向量召回，返回 [(余弦相似度, 文本块), ...]"""
    collection = get_collection()
    qv = embedding_model.encode(QUERY_PREFIX + query, normalize_embeddings=True)
    res = collection.query(query_embeddings=[qv.tolist()], n_results=top_k)

    return [
        (1 - d, doc)                       # cosine 空间：distance = 1 - cos
        for d, doc in zip(res["distances"][0], res["documents"][0])
    ]


# ---------------------------------------------------------------- 4. 重排

def rerank(query: str, candidates: list[str], top_k: int = 3) -> list[tuple[float, str]]:
    """CrossEncoder 精排。分数 > 0 判为相关，< 0 判为不相关。"""
    if not candidates:
        return []
    scores = cross_encoder.predict([(query, c) for c in candidates])
    ranked = sorted(zip(candidates, scores), key=lambda p: -p[1])
    return [(float(s), c) for c, s in ranked[:top_k]]


# ---------------------------------------------------------------- 5. 生成

PROMPT_TEMPLATE = """你是一位知识助手，请根据用户的问题和下列片段生成准确的回答。

用户问题：{query}

相关片段：
{context}

请基于上述内容作答，不要编造信息。如果片段中找不到答案，直接回答「文档中没有相关信息」。"""


def generate(query: str, chunks: list[str]) -> str:
    prompt = PROMPT_TEMPLATE.format(query=query, context="\n\n".join(chunks))

    extra = {}
    if THINKING in ("enabled", "disabled"):
        extra["extra_body"] = {"thinking": {"type": THINKING}}

    response = llm_client.chat.completions.create(
        model=LLM_MODEL,
        messages=[{"role": "user", "content": prompt}],
        temperature=0.1,
        **extra,
    )
    return response.choices[0].message.content.strip()


# ---------------------------------------------------------------- 串联

def ask(query: str, recall_k: int = 10, top_k: int = 3, min_ce: float = 0.0) -> dict:
    """完整链路。返回答案、引用块、各阶段耗时。"""
    t0 = time.time()
    recalled = retrieve(query, top_k=recall_k)
    t1 = time.time()

    reranked = rerank(query, [doc for _, doc in recalled], top_k=top_k)
    t2 = time.time()

    # 只有 CrossEncoder 判为相关的块才喂给模型
    useful = [doc for score, doc in reranked if score >= min_ce]

    if not useful:
        answer = "文档中没有相关信息。"
        t3 = time.time()
    else:
        answer = generate(query, useful)
        t3 = time.time()

    return {
        "answer": answer,
        "recalled": recalled,
        "reranked": reranked,
        "used": useful,
        "time": {"recall": t1 - t0, "rerank": t2 - t1, "generate": t3 - t2},
    }


def main():
    import sys

    query = sys.argv[1] if len(sys.argv) > 1 else "安子的爸爸是谁"

    chunks = split_into_chunks("doc.md")
    build_index(chunks)
    print(f"建库完成：{len(chunks)} 块\n")

    result = ask(query)
    print(f"问题：{query}\n")
    print("召回（余弦）：")
    for score, doc in result["recalled"][:5]:
        print(f"  {score:.3f}  {doc[:34]}")
    print("\n重排（CrossEncoder）：")
    for score, doc in result["reranked"]:
        print(f"  {score:+.2f}  {doc[:34]}")
    print(f"\n答案：{result['answer']}")
    t = result["time"]
    print(f"\n耗时：召回 {t['recall']*1000:.0f}ms | 重排 {t['rerank']*1000:.0f}ms | 生成 {t['generate']*1000:.0f}ms")


if __name__ == "__main__":
    main()
