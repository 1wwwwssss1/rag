import streamlit as st
import requests
import json
from datetime import datetime

st.set_page_config(
    page_title="RAG Chat Assistant",
    page_icon="🤖",
    layout="wide",
    initial_sidebar_state="expanded"
)

st.markdown("""
<style>
.stApp {
    background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
    min-height: 100vh;
}
.chat-container {
    background-color: rgba(255, 255, 255, 0.95);
    border-radius: 15px;
    padding: 20px;
    margin-bottom: 20px;
    box-shadow: 0 8px 32px rgba(0, 0, 0, 0.1);
}
.user-message {
    background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
    color: white;
    padding: 12px 18px;
    border-radius: 18px 18px 4px 18px;
    margin: 8px 0;
    max-width: 70%;
}
.bot-message {
    background: linear-gradient(135deg, #f5f7fa 0%, #c3cfe2 100%);
    color: #333;
    padding: 12px 18px;
    border-radius: 18px 18px 18px 4px;
    margin: 8px 0;
    max-width: 70%;
}
.header-title {
    text-align: center;
    color: white;
    font-size: 2.5em;
    font-weight: bold;
    text-shadow: 2px 2px 4px rgba(0,0,0,0.3);
    margin-bottom: 30px;
}
.sidebar {
    background-color: rgba(255, 255, 255, 0.1);
    backdrop-filter: blur(10px);
}
.stTextInput > div > div > input {
    border-radius: 25px;
    padding: 12px 20px;
    font-size: 16px;
}
.stButton>button {
    border-radius: 25px;
    background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
    color: white;
    font-weight: bold;
    padding: 10px 30px;
    border: none;
    transition: all 0.3s ease;
}
.stButton>button:hover {
    transform: translateY(-2px);
    box-shadow: 0 5px 15px rgba(102, 126, 234, 0.4);
}
</style>
""", unsafe_allow_html=True)

def init_session_state():
    if "messages" not in st.session_state:
        st.session_state.messages = []

def generate_response(query):
    try:
        response = requests.post(
            "http://localhost:8501/api/chat",
            json={"query": query},
            timeout=30
        )
        return response.json().get("answer", "未获得回答")
    except Exception as e:
        return f"错误：{str(e)}"

def main():
    init_session_state()
    
    st.markdown('<h1 class="header-title">🤖 RAG Chat Assistant</h1>', unsafe_allow_html=True)
    
    col1, col2, col3 = st.columns([1, 2, 1])
    with col2:
        chat_box = st.container()
    
    with chat_box:
        for message in st.session_state.messages:
            with st.chat_message(message["role"]):
                st.markdown(message["content"])
        
        if prompt := st.chat_input("输入你的问题..."):
            st.session_state.messages.append({"role": "user", "content": prompt})
            with st.chat_message("user"):
                st.markdown(prompt)
            
            with st.chat_message("assistant"):
                with st.spinner("🤔 思考中..."):
                    answer = generate_response(prompt)
                st.markdown(answer)
            
            st.session_state.messages.append({
                "role": "assistant", 
                "content": answer
            })
    
    with st.sidebar:
        st.title("⚙️ 设置")
        api_url = st.text_input("API URL", "http://localhost:8501")
        clear_btn = st.button("🗑️ 清除对话历史")
        
        if clear_btn:
            st.session_state.messages = []
            st.rerun()
        
        st.markdown("---")
        st.markdown("**功能特点**:")
        st.markdown("""
        • **智能检索**:基于 RAG 的知识问答
        • **美观界面**:现代化渐变设计
        • **流畅体验**:实时消息展示
        • **上下文记忆**:支持多轮对话
        """)
        
        st.markdown("---")
        st.metric("对话次数", len(st.session_state.messages))

if __name__ == "__main__":
    main()
