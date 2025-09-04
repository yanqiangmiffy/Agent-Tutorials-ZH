"""
多智能体研究监督器的状态定义

该模块定义了用于多智能体研究监督器工作流的状态对象和工具，
包括协调状态和研究工具。
"""

import operator
from typing_extensions import Annotated, TypedDict, Sequence

from langchain_core.messages import BaseMessage
from langchain_core.tools import tool
from langgraph.graph.message import add_messages
from pydantic import BaseModel, Field

class SupervisorState(TypedDict):
    """
    多智能体研究监督器的状态。

    管理监督器和研究智能体之间的协调，跟踪
    研究进度并累积来自多个子智能体的发现。
    """

    # 与监督器交换的消息，用于协调和决策
    supervisor_messages: Annotated[Sequence[BaseMessage], add_messages]
    # 详细的研究简报，指导整体研究方向
    research_brief: str
    # 处理和结构化的笔记，准备用于最终报告生成
    notes: Annotated[list[str], operator.add] = []
    # 计数器，跟踪已执行的研究迭代次数
    research_iterations: int = 0
    # 从子智能体研究中收集的原始未处理研究笔记
    raw_notes: Annotated[list[str], operator.add] = []

@tool
class ConductResearch(BaseModel):
    """用于将研究任务委托给专门的子智能体的工具。"""
    research_topic: str = Field(
        description="要研究的主题。应该是单一主题，并且应该详细描述（至少一段话）。",
    )

@tool
class ResearchComplete(BaseModel):
    """用于指示研究过程已完成的工具。"""
    pass