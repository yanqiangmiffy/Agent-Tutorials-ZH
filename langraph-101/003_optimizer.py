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
# 图状态
class State(TypedDict):
    joke: str
    topic: str
    feedback: str
    funny_or_not: str

# 用于评估的结构化输出 Schema
class Feedback(BaseModel):
    grade: Literal["funny", "not funny"] = Field(
        description="判断这个笑话是否好笑。",
    )
    feedback: str = Field(
        description="如果笑话不好笑，请提供改进建议。",
    )

# 使用结构化输出的 Schema 增强 LLM
evaluator = llm.with_structured_output(Feedback)

# 节点
def llm_call_generator(state: State):
    """LLM 生成一个笑话"""

    if state.get("feedback"):
        msg = llm.invoke(
            f"写一个关于{state['topic']}的笑话，但要考虑以下反馈: {state['feedback']}"
        )
    else:
        msg = llm.invoke(f"写一个关于{state['topic']}的笑话")
    return {"joke":msg.content}

def llm_call_evaluator(state: State):
    """LLM 评估这个笑话"""

    grade = evaluator.invoke(f"给这个笑话打分: {state['joke']}")
    return {"funny_or_not": grade.grade, "feedback": grade.feedback}

# 条件边函数，根据评估者的反馈，路由回笑话生成器或结束
def route_joke(state: State):
    """根据评估者的反馈，路由回笑话生成器或结束"""

    if state["funny_or_not"] == "funny":
        return "Accepted"
    elif state["funny_or_not"] == "not funny":
        return "Rejected + Feedback"

# 构建工作流
optimizer_builder = StateGraph(State)

# 添加节点
optimizer_builder.add_node("llm_call_generator", llm_call_generator)
optimizer_builder.add_node("llm_call_evaluator", llm_call_evaluator)

# 添加边以连接节点
optimizer_builder.add_edge(START, "llm_call_generator")
optimizer_builder.add_edge("llm_call_generator", "llm_call_evaluator")
optimizer_builder.add_conditional_edges(
    "llm_call_evaluator",
    route_joke,
    {
        # route_joke 返回的名称 : 要访问的下一个节点的名称
        "Accepted": END,
        "Rejected + Feedback": "llm_call_generator",
    },
)

# 编译工作流
optimizer_workflow = optimizer_builder.compile()

# 显示工作流
png_data=optimizer_workflow.get_graph().draw_mermaid_png()
with open("optimizer_workflow.png", "wb") as f:
    f.write(png_data)
# 调用
state = optimizer_workflow.invoke({"topic": "猫"})
print(state["joke"])