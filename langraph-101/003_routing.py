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
# 用于路由逻辑的结构化输出 Schema
class Route(BaseModel):
    step: Literal["poem", "story", "joke"] = Field(
        None, description="路由过程中的下一步"
    )

# 使用结构化输出的 Schema 增强 LLM
router = llm.with_structured_output(Route)# 状态
class State(TypedDict):
    input: str
    decision: str
    output: str

# 节点
def llm_call_1(state: State):
    """写一个故事"""

    result = llm.invoke(state["input"])
    return {"output": result.content}

def llm_call_2(state: State):
    """写一个笑话"""

    result= llm.invoke(state["input"])
    return {"output": result.content}

def llm_call_3(state: State):
    """写一首诗歌"""

    result = llm.invoke(state["input"])
    return {"output": result.content}

def llm_call_router(state: State):
    """将输入路由到适当的节点"""

    # 运行带有结构化输出的增强型 LLM，作为路由逻辑
    decision = router.invoke(
        [
            SystemMessage(
                content="根据用户的请求，将输入路由到故事、笑话或诗歌。"
            ),HumanMessage(content=state["input"]),
        ]
    )

    return {"decision": decision.step}

# 条件边函数，用于路由到相应的节点
def route_decision(state: State):
    # 返回要访问的下一个节点名称
    if state["decision"] == "story":
        return "llm_call_1"
    elif state["decision"] == "joke":
        return "llm_call_2"
    elif state["decision"] == "poem":
        return "llm_call_3"

# 构建工作流
router_builder = StateGraph(State)

# 添加节点
router_builder.add_node("llm_call_1", llm_call_1)
router_builder.add_node("llm_call_2", llm_call_2)
router_builder.add_node("llm_call_3", llm_call_3)
router_builder.add_node("llm_call_router", llm_call_router)

# 添加边以连接节点
router_builder.add_edge(START, "llm_call_router")
router_builder.add_conditional_edges(
    "llm_call_router",
    route_decision,
    {# route_decision 返回的名称 : 要访问的下一个节点的名称
        "llm_call_1": "llm_call_1",
        "llm_call_2": "llm_call_2",
        "llm_call_3": "llm_call_3",
    },
)
router_builder.add_edge("llm_call_1", END)
router_builder.add_edge("llm_call_2", END)
router_builder.add_edge("llm_call_3", END)

# 编译工作流
router_workflow = router_builder.compile()

#显示工作流
png_data=router_workflow.get_graph().draw_mermaid_png()
with open("routing_graph.png", "wb") as f:
    f.write(png_data)
# 调用
state = router_workflow.invoke({"input": "给我写一个关于猫的笑话"})
print(state["output"])