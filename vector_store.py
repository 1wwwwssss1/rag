# 对应：Database
from pathlib import Path
from tempfile import gettempdir

from langchain_chroma import Chroma
from langchain_openai import OpenAIEmbeddings
from config import Config


def _is_writable_directory(path: Path) -> bool:
    try:
        path.mkdir(parents=True, exist_ok=True)
        probe = path / ".write_test"
        probe.write_text("ok", encoding="utf-8")
        probe.unlink()
        return True
    except OSError:
        return False


def _get_persist_directory() -> str:
    base_dir = Path(__file__).resolve().parent
    configured_dir = getattr(Config, "CHROMA_PERSIST_DIRECTORY", None)
    preferred_dir = Path(configured_dir) if configured_dir else base_dir / ".chroma_db"

    if not preferred_dir.is_absolute():
        preferred_dir = (base_dir / preferred_dir).resolve()

    if _is_writable_directory(preferred_dir):
        return str(preferred_dir)

    fallback_dir = Path(gettempdir()) / "rag_demo_chroma_db"
    if _is_writable_directory(fallback_dir):
        print(f"Chroma 目录不可写，已回退到临时目录: {fallback_dir}")
        return str(fallback_dir)

    raise PermissionError(
        f"无法写入 Chroma 目录: {preferred_dir}，且临时目录也不可写。"
    )

def get_embedding():
    # 使用 DashScope embedding
    return OpenAIEmbeddings(
        model=Config.EMBEDDING_MODEL,
        openai_api_key=Config.API_KEY,
        openai_api_base=Config.API_BASE,
        check_embedding_ctx_length=False,
        model_kwargs={"encoding_format": "float"}
    )

def get_vectorstore():
    return Chroma(
        collection_name="rag_demo",
        embedding_function=get_embedding(),
        persist_directory=_get_persist_directory()
    )
