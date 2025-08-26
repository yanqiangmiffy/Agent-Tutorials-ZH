import random
from typing import Literal
from typing import TypedDict

from langgraph.graph import StateGraph, END, START


# 定义状态 State
class State(TypedDict):
    graph_state: str


# 定义节点 None

def node_1(state: State):
    print("我在运行节点1的功能")
    return {"graph_state": state["graph_state"] + ",我的心情是:"}


def node_2(state: State):
    print("我在运行节点2的功能")
    return {"graph_state": state["graph_state"] + "开心的！"}


def node_3(state: State):
    print("我在运行节点3的功能")
    return {"graph_state": state["graph_state"] + "伤心的！"}


def decide_mood(state: State) -> Literal["node_2", "node_3"]:
    # 用户的输入决定了下一个访问节点
    user_input = state["graph_state"]

    if random.random() < 0.5:
        return "node_2"
    return "node_3"


# 构建图：基于节点+边
builder = StateGraph(State)
# 添加节点
builder.add_node("node_1", node_1)
builder.add_node("node_2", node_2)
builder.add_node("node_3", node_3)

# 添加边
builder.add_edge(START, "node_1")
builder.add_conditional_edges("node_1", decide_mood)
builder.add_edge("node_2", END)
builder.add_edge("node_3", END)

# 编译图
graph = builder.compile()
# 显示工作流
png_data = graph.get_graph().draw_mermaid_png()
with open("graph.png", "wb") as f:
    f.write(png_data)

response = graph.invoke({"graph_state": "你好，我是小明"})
print(response)
