from typing_extensions import TypedDict
from langchain_core.messages import BaseMessage

class AgentState(TypedDict):
    """Agent 状态定义"""
    question: str
    context: str
    messages: list[BaseMessage]
    retry_count: int
    grade: str
    user_id: str
