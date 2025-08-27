# state_scope.py
"""研究范围界定的状态定义和Pydantic架构。
这定义了研究代理范围界定工作流程使用的状态对象和结构化架构，
包括研究人员状态管理和输出架构。
"""

import operator
from typing_extensions import Optional, Annotated, List, Sequence

from langchain_core.messages import BaseMessage
from langgraph.graph import MessagesState
from langgraph.graph.message import add_messages
from pydantic import BaseModel, Field


# ===== 状态定义 =====

class AgentInputState(MessagesState):
    """完整代理的输入状态 - 仅包含来自用户输入的消息。"""
    pass
class AgentState(MessagesState):
    """
    完整多代理研究系统的主状态。

    使用额外的字段扩展MessagesState以进行研究协调。
    注意：某些字段在不同状态类之间重复，以便在子图和主工作流程之间
    进行适当的状态管理。
    """
    # 从用户对话历史生成的研究简报
    research_brief: Optional[str]
    # 与监督代理交换的协调消息
    supervisor_messages: Annotated[Sequence[BaseMessage], add_messages]
    # 在研究阶段收集的原始未处理研究笔记
    raw_notes: Annotated[list[str], operator.add] = []
    # 为报告生成准备的已处理和结构化笔记
    notes: Annotated[list[str], operator.add] = []
    # 最终格式化的研究报告
    final_report: str


# ===== 结构化输出架构 =====
class ClarifyWithUser(BaseModel):
    """用户澄清决策和问题的架构。"""
    need_clarification: bool = Field(
        description="是否需要向用户询问澄清问题。",
    )
    question: str = Field(
        description="询问用户澄清报告范围的问题",
    )
    verification: str = Field(
        description="用户提供必要信息后我们将开始研究的验证消息。",
    )


class ResearchQuestion(BaseModel):
    """结构化研究简报生成的架构。"""
    research_brief: str = Field(
        description="将用于指导研究的研究问题。",
    )