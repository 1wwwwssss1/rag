import os
from dotenv import load_dotenv

load_dotenv()

class Config:
    # ========== LLM 配置 ==========
    API_KEY = os.getenv("DASHSCOPE_API_KEY")
    API_BASE = os.getenv("OPENAI_API_BASE")

    # 主模型（用于最终生成答案 Reader）
    LLM_MODEL = os.getenv("LLM_MODEL")

    # 快速模型（用于 Query Understand）
    FAST_MODEL = os.getenv("LLM_FAST_MODEL")

    # ========== RAG 配置（核心） ==========
    
    # 向量数据库类型（当前用 Chroma，本地轻量）
    VECTOR_STORE_TYPE = os.getenv("VECTOR_STORE_TYPE", "chroma")

    # embedding 模型（用于 chunk → 向量）
    EMBEDDING_MODEL = os.getenv("EMBEDDING_MODEL")

    # rerank 模型（第二阶段精排）
    RERANK_MODEL = os.getenv("RERANK_MODEL")

    # 第一阶段召回数量（越大召回越全，但噪声越多）
    RAG_TOP_K = int(os.getenv("RAG_TOP_K", 10))

    # rerank后保留数量（最终喂给LLM）
    RERANK_TOP_K = int(os.getenv("RERANK_TOP_K", 3))

    # 工具调用超时（这里预留）
    TOOL_TIMEOUT = int(os.getenv("TOOL_TIMEOUT", 30))

    # 工具重试次数
    TOOL_RETRY_COUNT = int(os.getenv("TOOL_RETRY_COUNT", 3))