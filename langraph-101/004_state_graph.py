from typing import TypedDict
from langgraph.graph import StateGraph, END

class GraphState(TypedDict):
    input: str
    output: str
    step_count: int


# 创建StateGraph，指定状态类型
workflow = StateGraph(GraphState)


def node_1(state: GraphState) -> GraphState:
    """第一个节点的处理逻辑"""
    return {
        "input": state["input"],
        "output": f"node_1 处理了: {state['input']}",
        "step_count": state.get("step_count", 0) + 1
    }

def node_2(state: GraphState) -> GraphState:
    """第二个节点的处理逻辑"""
    print("第二个节点的处理逻辑")
    return {
        "input": state["input"],
        "output": state["output"] + " node_2-> 进一步处理",
        "step_count": state["step_count"] + 1
    }


workflow.add_node("step1", node_1)
workflow.add_node("step2", node_2)
# 设置入口点
workflow.set_entry_point("step1")
# 添加边
# 添加条件边
def should_continue(state: GraphState) -> str:
    """决定下一步走向的条件函数"""
    print("continue")
    if state["step_count"] < 3:
        return "continue"
    else:
        return "end"
workflow.add_conditional_edges(
    "step1",
    should_continue,
    {
        "continue": "step2",
        "end": END
    }
)
workflow.add_edge("step2", END)


# 编译图
state_graph = workflow.compile()

# 运行图
initial_state = {
    "input": "Hello World,LangGraph",
    "output": "",
    "step_count": 0
}

result = state_graph.invoke(initial_state)
print(result)
png_data=state_graph.get_graph().draw_mermaid_png()
with open("state_graph.png", "wb") as f:
    f.write(png_data)


