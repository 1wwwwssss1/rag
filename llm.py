from openai import OpenAI
from config import Config

client = OpenAI(
    api_key=Config.API_KEY,
    base_url=Config.API_BASE
)

def call_llm(prompt: str, model=None):
    """统一 LLM 调用"""
    model = model or Config.LLM_MODEL

    resp = client.chat.completions.create(
        model=model,
        messages=[{"role": "user", "content": prompt}],
        temperature=0.3
    )
    return resp.choices[0].message.content