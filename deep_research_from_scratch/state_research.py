
"""
研究智能体的状态定义和Pydantic模式

此模块定义了研究智能体工作流程中使用的状态对象和结构化模式，
包括研究者状态管理和输出模式。
"""

import operator
from typing_extensions import TypedDict, Annotated, List, Sequence
from pydantic import BaseModel, Field
from langchain_core.messages import BaseMessage
from langgraph.graph.message import add_messages

# ===== 状态定义 =====

class ResearcherState(TypedDict):
    """
    包含消息历史和研究元数据的研究智能体状态。

    此状态跟踪研究者的对话、用于限制工具调用的迭代计数、
    正在调查的研究主题、压缩的发现结果和用于详细分析的原始研究笔记。
    """
    researcher_messages: Annotated[Sequence[BaseMessage], add_messages]
    tool_call_iterations: int
    research_topic: str
    compressed_research: str
    raw_notes: Annotated[List[str], operator.add]

class ResearcherOutputState(TypedDict):
    """
    包含最终研究结果的研究智能体输出状态。

    这表示研究过程的最终输出，包含压缩的研究发现
    和研究过程中的所有原始笔记。
    """
    compressed_research: str
    raw_notes: Annotated[List[str], operator.add]
    researcher_messages: Annotated[Sequence[BaseMessage], add_messages]

# ===== 结构化输出模式 =====

class ClarifyWithUser(BaseModel):
    """范围确定阶段用户澄清决策的模式。"""
    need_clarification: bool = Field(
        description="是否需要向用户询问澄清问题。",
    )
    question: str = Field(
        description="向用户询问以澄清报告范围的问题",
    )
    verification: str = Field(
        description="确认消息，表示在用户提供必要信息后我们将开始研究。",
    )

class ResearchQuestion(BaseModel):
    """研究简报生成的模式。"""
    research_brief: str = Field(
        description="将用于指导研究的研究问题。",
    )

class Summary(BaseModel):
    """网页内容摘要的模式。"""
    summary: str = Field(description="网页内容的简洁摘要")
    key_excerpts: str = Field(description="内容中的重要引用和摘录")