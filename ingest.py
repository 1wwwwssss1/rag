# 对应：
# Source Data → Parser → Splitter → Embedding → Database

from langchain_community.document_loaders import TextLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from vector_store import get_vectorstore
import os

def ingest():
    # 1️⃣ Source Data + Parser
    # 尝试多种编码读取文件
    file_path = "data.txt"
    loader = None
    
    # 先尝试 utf-8
    try:
        loader = TextLoader(file_path, encoding='utf-8')
        docs = loader.load()
    except UnicodeDecodeError:
        try:
            # 如果 utf-8 失败，尝试 gbk (中文 Windows 常见)
            loader = TextLoader(file_path, encoding='gbk')
            docs = loader.load()
        except Exception as e:
            print(f"无法读取文件: {e}")
            return

    if not docs:
        print("文档为空，请检查 data.txt 内容")
        return

    print(f"加载了 {len(docs)} 个文档片段")

    # 2️⃣ Splitter（切块）
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=500,
        chunk_overlap=100
    )
    chunks = splitter.split_documents(docs)
    
    # 过滤空内容
    chunks = [c for c in chunks if c.page_content.strip()]

    if not chunks:
        print("切分后无有效内容")
        return

    # 3️⃣ 存入向量数据库（自动Embedding）
    vectordb = get_vectorstore()
    vectordb.add_documents(chunks)

    print(f"已入库 {len(chunks)} 个 chunks")

if __name__ == "__main__":
    ingest()
