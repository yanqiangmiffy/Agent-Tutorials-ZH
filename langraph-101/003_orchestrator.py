from typing import Annotated, List
import operator
import getpass
import os

from dotenv import load_dotenv
from langchain_deepseek import ChatDeepSeek
from typing import TypedDict
from langgraph.graph import StateGraph, START, END
from pydantic import BaseModel,Field
from typing_extensions import Literal
from langchain_core.messages import HumanMessage, SystemMessage
load_dotenv()
def _set_env(var: str):
    if not os.environ.get(var):
        os.environ[var] = getpass.getpass(f"{var}: ")


_set_env("DEEPSEEK_API_KEY")

llm = ChatDeepSeek(model="deepseek-chat")
# 用于规划的结构化输出 Schema
class Section(BaseModel):
    name: str = Field(
        description="报告此部分的名称。",
    )
    description: str = Field(
        description="本部分将涵盖的主要主题和概念的简要概述。",
    )

class Sections(BaseModel):
    sections: List[Section] = Field(
        description="报告的各个部分。",)

# 使用结构化输出的 Schema 增强 LLM
planner = llm.with_structured_output(Sections)
print(planner)


from langgraph.types import Send

# 图状态
class State(TypedDict):
    topic: str# 报告主题
    sections: list[Section]  # 报告部分列表
    completed_sections: Annotated[
        list, operator.add
    ]  # 所有工作者并行写入此键
    final_report: str  # 最终报告

# 工作者状态
class WorkerState(TypedDict):
    section: Section
    completed_sections: Annotated[list, operator.add]

# 节点
def orchestrator(state: State):
    """生成报告计划的编排者"""

    # 生成查询
    report_sections = planner.invoke(
        [
            SystemMessage(content="生成报告的计划。"),
            HumanMessage(content=f"这是报告主题: {state['topic']}"),
        ]
    )

    return {"sections": report_sections.sections}

def llm_call(state: WorkerState):
    """工作者编写报告的一个部分"""

    # 生成部分
    section = llm.invoke(
        [
            SystemMessage(
                content="按照提供的名称和描述编写报告的一个部分。不要包含每个部分的前导语。使用 Markdown格式。"
            ),
            HumanMessage(
                content=f"这是部分名称: {state['section'].name} 和描述: {state['section'].description}"
            ),
        ]
    )
    # 将更新后的部分写入已完成部分
    # 这里需要是一个列表，因为 Annotated[list, operator.add] 要求
    return {"completed_sections": [section.content]}

def synthesizer(state: State):
    """从各部分合成完整报告"""

    # 已完成部分列表
    completed_sections = state["completed_sections"]

    # 将已完成部分格式化为字符串，用作最终部分的上下文
    completed_report_sections = "\n\n---\n\n".join(completed_sections)

    return {"final_report": completed_report_sections}

# 条件边函数，用于创建每个编写报告部分的 llm_call 工作者
def assign_workers(state: State):
    """为计划中的每个部分分配一个工作者"""

    # 通过 Send() API 并行启动部门编写
    return [Send("llm_call",{"section": s}) for s in state["sections"]]

# 构建工作流
orchestrator_worker_builder = StateGraph(State)

# 添加节点
orchestrator_worker_builder.add_node("orchestrator", orchestrator)
orchestrator_worker_builder.add_node("llm_call", llm_call)
orchestrator_worker_builder.add_node("synthesizer", synthesizer)

# 添加边以连接节点
orchestrator_worker_builder.add_edge(START, "orchestrator")
orchestrator_worker_builder.add_conditional_edges(
    "orchestrator", assign_workers, ["llm_call"]
)
orchestrator_worker_builder.add_edge("llm_call", "synthesizer")
orchestrator_worker_builder.add_edge("synthesizer", END)

# 编译工作流
orchestrator_worker = orchestrator_worker_builder.compile()

# 显示工作流

png_data=orchestrator_worker.get_graph().draw_mermaid_png()
with open("orchestrator_worker.png", "wb") as f:
    f.write(png_data)
# 调用
state = orchestrator_worker.invoke({"topic": "创建一份关于 大模型 scaling laws的综述报告"})
print(state["final_report"])
from IPython.display import Markdown
print(Markdown(state["final_report"]))