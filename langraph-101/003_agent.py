import getpass
import os

from dotenv import load_dotenv
from langchain_core.messages import HumanMessage, SystemMessage
from langchain_core.messages import ToolMessage
from langchain_core.tools import tool
from langchain_deepseek import ChatDeepSeek
from langgraph.graph import MessagesState
from langgraph.graph import StateGraph, START, END
from typing_extensions import Literal

load_dotenv()


def _set_env(var: str):
    if not os.environ.get(var):
        os.environ[var] = getpass.getpass(f"{var}: ")


_set_env("DEEPSEEK_API_KEY")

llm = ChatDeepSeek(model="deepseek-chat")


# 定义工具
@tool
def multiply(a: int, b: int) -> int:
    """计算 a 乘以 b 的结果。

    Args:
        a: 第一个整数
        b: 第二个整数"""
    return a * b


@tool
def add(a: int, b: int) -> int:
    """计算 a 加上 b 的结果。

    Args:
        a: 第一个整数b: 第二个整数
    """
    return a + b


@tool
def divide(a: int, b: int) -> float:
    """计算 a 除以 b 的结果。

    Args:
        a: 第一个整数
        b: 第二个整数
    """
    return a / b


# 使用工具增强 LLM
tools = [add, multiply, divide]
tools_by_name = {tool.name: tool for tool in tools}
llm_with_tools = llm.bind_tools(tools)


# 节点
def llm_call(state: MessagesState):
    """LLM 决定是否调用工具"""

    return {
        "messages": [
            llm_with_tools.invoke(
                [
                    SystemMessage(
                        content="你是一个乐于助人的助手，负责对一组输入执行算术运算。"
                    )
                ] + state["messages"]
            )
        ]
    }


def tool_node(state: dict):
    """执行工具调用"""

    result = []
    for tool_call in state["messages"][-1].tool_calls:
        tool = tools_by_name[tool_call["name"]]
        observation = tool.invoke(tool_call["args"])
        result.append(ToolMessage(content=observation, tool_call_id=tool_call["id"]))
    return {"messages": result}


# 条件边函数，根据 LLM 是否进行了工具调用，路由到工具节点或结束
def should_continue(state: MessagesState) -> Literal["Action", END]:
    """根据 LLM 是否进行了工具调用，决定是否继续循环或停止"""

    messages = state["messages"]
    last_message = messages[-1]
    # 如果 LLM 进行了工具调用，则执行一个动作
    if last_message.tool_calls:
        return "Action"
    # 否则，我们停止（回复用户）
    return END


# 构建工作流
agent_builder = StateGraph(MessagesState)

# 添加节点
agent_builder.add_node("llm_call", llm_call)
agent_builder.add_node("environment", tool_node)  # 添加边以连接节点
agent_builder.add_edge(START, "llm_call")
agent_builder.add_conditional_edges(
    "llm_call",
    should_continue, {
        # should_continue
        # 返回的名称 : 要访问的下一个节点的名称
        "Action": "environment",
        END: END,
    },
)
agent_builder.add_edge("environment", "llm_call")

# 编译智能体
agent = agent_builder.compile()

# 显示智能体
png_data = agent.get_graph().draw_mermaid_png()
with open("agent.png", "wb") as f:
    f.write(png_data)
# 调用
messages = [HumanMessage(content="计算 3 加 4。")]
messages = agent.invoke({"messages": messages})
for m in messages["messages"]:
    m.pretty_print()
