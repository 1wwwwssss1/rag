# 对应：
# Query Understand → 1st Retrieval → Rerank

from config import Config
from vector_store import get_vectorstore
from llm import call_llm
import re

def rewrite_query(query: str):
    """Query Understand"""
    try:
        prompt = f"Rewrite this query more clearly for a search engine. Only return the rewritten query, no other text:\n{query}"
        new_query = call_llm(prompt, model=Config.FAST_MODEL)
        
        # 确保返回的是干净字符串
        if not new_query or not isinstance(new_query, str):
            return query
        
        cleaned_query = new_query.strip().strip('"').strip("'").strip()
        
        # 如果重写后为空，回退到原查询
        if not cleaned_query:
            return query
            
        return cleaned_query
    except Exception as e:
        print(f"Query rewrite failed: {e}. Using original query.")
        return query


def retrieve(query):
    """
    支持单条字符串或列表批量查询向量数据库。
    自动清洗 query，并调用 retriever.retrieve()，兼容 Qdrant / PGVector。
    """
    # 处理列表输入
    if isinstance(query, list):
        results = []
        for q in query:
            if not isinstance(q, str):
                try:
                    q = str(q)
                except Exception:
                    print(f"无法将 {q} 转换为字符串，跳过")
                    continue
            results.extend(retrieve(q))
        return results

    # 单条输入
    if query is None:
        print("Query 为 None")
        return []

    if not isinstance(query, str):
        try:
            query = str(query)
        except Exception:
            print(f"Query 无法转换为字符串: {query}")
            return []

    query = query.strip()
    if not query:
        print("Query 为空或仅包含空格")
        return []

    # 清洗 query，只保留中文、英文、数字和常用标点
    cleaned_query = re.sub(r"[^\w\s\u4e00-\u9fff.,!?]", "", query).strip()
    if not cleaned_query:
        print("Query 清洗后为空")
        return []

    print(f"Searching for: '{cleaned_query}'")

    try:
        vectordb = get_vectorstore()
        retriever = vectordb.as_retriever(search_kwargs={"k": Config.RAG_TOP_K})

        # LangChain 当前检索器统一入口是 invoke()，部分旧版本仍保留 get_relevant_documents()
        if hasattr(retriever, "invoke"):
            results = retriever.invoke(cleaned_query)
        elif hasattr(retriever, "get_relevant_documents"):
            results = retriever.get_relevant_documents(cleaned_query)
        else:
            print("Retriever 对象没有可用的检索方法，请确认 VectorStore 类型")
            results = []

        # 确保返回列表
        if not isinstance(results, list):
            results = list(results)

        return results

    except Exception as e:
        print(f"Retrieval error: {e}")
        return []


def rerank(query: str, docs):
    """2nd Retrieval（用LLM简单模拟rerank）"""
    if not docs:
        return []
        
    scored = []

    for doc in docs:
        # 确保 doc.page_content 是字符串且非空
        content = doc.page_content if doc.page_content else ""
        if not content:
            scored.append((0.0, doc))
            continue

        prompt = f"""
Rate relevance (0-1) of the document to the query. Return only a number.
Query: {query}
Doc: {content[:200]}...
"""
        try:
            score_str = call_llm(prompt, model=Config.FAST_MODEL)
            # 提取第一个浮点数
            match = re.search(r'(\d+\.?\d*)', score_str)
            score = float(match.group(1)) if match else 0.0
            # 限制范围 0-1
            score = max(0.0, min(1.0, score))
        except Exception as e:
            print(f"Rerank error for doc: {e}")
            score = 0.0

        scored.append((score, doc))

    scored.sort(key=lambda x: x[0], reverse=True)
    return [d for _, d in scored[:Config.RERANK_TOP_K]]
