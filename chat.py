from retriever import rewrite_query, retrieve, rerank
from llm import call_llm
from config import Config
def chat():
    print("🤖 RAG Chat 已启动（输入 exit 退出）")

    history = []

    while True:
        query = input("\n你：").strip()
        if query.lower() == "exit":
            break

        if not query:
            print("⚠️ 请输入更完整的问题或上下文。")
            continue

        # 1️⃣ Query Understand
        new_query = rewrite_query(query)
        if not new_query:
            new_query = query

        # 2️⃣ Retrieval
        try:
            docs = retrieve(new_query)
        except Exception as e:
            print(f"❌ 检索出错: {e}")
            docs = []

        # 3️⃣ Rerank
        try:
            top_docs = rerank(new_query, docs)
        except Exception as e:
            print(f"❌ Rerank 出错: {e}")
            top_docs = docs[:Config.RAG_TOP_K] if docs else []

        # 4️⃣ Reader（最终生成）
        context = "\n\n".join([d.page_content for d in top_docs])

        prompt = f"""
你是一个知识助手，只能基于提供内容回答。

上下文：
{context}

问题：
{query}
"""
        try:
            answer = call_llm(prompt)
            print(f"\n🤖：{answer}")
        except Exception as e:
            print(f"❌ 回答生成出错: {e}")

if __name__ == "__main__":
    chat()