# src/agents/state.py

from typing import TypedDict, List, Annotated, Sequence
from langchain_core.messages import BaseMessage
import operator

class AgentState(TypedDict):
    messages: Annotated[Sequence[BaseMessage], operator.add]
    question: str
    plan: List[str]
    current_step: int
    past_steps: List[str]
    response: str
    next_agent: str
    niveau_support: int