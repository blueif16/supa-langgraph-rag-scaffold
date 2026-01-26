import os
from dotenv import load_dotenv

load_dotenv()

class Config:
    """统一配置管理"""
    # Supabase
    SUPABASE_URL = os.getenv("SUPABASE_URL")
    SUPABASE_KEY = os.getenv("SUPABASE_KEY")
    DATABASE_URL = os.getenv("DATABASE_URL")

    # OpenAI
    OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
    EMBEDDING_MODEL = os.getenv("EMBEDDING_MODEL", "text-embedding-3-small")
    CHAT_MODEL = os.getenv("CHAT_MODEL", "gpt-4o-mini")

    # RAG 参数
    MATCH_COUNT = int(os.getenv("MATCH_COUNT", "5"))
    RRF_K = int(os.getenv("RRF_K", "60"))
    GRAPH_DEPTH = int(os.getenv("GRAPH_DEPTH", "2"))
    MAX_RETRY = int(os.getenv("MAX_RETRY", "2"))

    # 服务配置
    HOST = os.getenv("HOST", "0.0.0.0")
    PORT = int(os.getenv("PORT", "8000"))

config = Config()
