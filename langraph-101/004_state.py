# from typing import TypedDict
# from langgraph.graph import StateGraph, START, END
#
# class SimpleState(TypedDict):
#     input: str
#     output: str
#     step_count: int
#
# def process_node(state: SimpleState) -> SimpleState:
#     return {
#         "input": state["input"],
#         "output": f"处理结果: {state['input']}",
#         "step_count": state.get("step_count", 0) + 1
#     }
#
# # 创建简单的状态图
# workflow = StateGraph(SimpleState)
# workflow.add_node("processor", process_node)
# workflow.add_edge(START, "processor")
# workflow.add_edge("processor", END)
#
# app = workflow.compile()
# result = app.invoke({"input": "Hello", "output": "", "step_count": 0})
#
# print(result)
from typing import TypedDict
from langgraph.graph import StateGraph, START, END


class InputState(TypedDict):
    """用户输入接口"""
    user_input: str


class OutputState(TypedDict):
    """最终输出接口"""
    graph_output: str
    step_count:int

class OverallState(TypedDict):
    """内部完整状态"""
    user_input: str
    intermediate_result: str
    processed_data: str
    graph_output: str
    step_count: int


class PrivateState(TypedDict):
    """节点间私有通信"""
    private_data: str
    internal_flag: bool


def input_node(state: InputState) -> OverallState:
    """输入处理节点"""
    return {
        "user_input": state["user_input"],
        "intermediate_result": f"开始处理: {state['user_input']}",
        "step_count": 1
    }


def processing_node(state: OverallState) -> PrivateState:
    """数据处理节点"""
    processed = state["intermediate_result"].upper()
    return {
        "private_data": f"私有处理: {processed}",
        "internal_flag": len(state["user_input"]) > 5
    }


def intermediate_node(state: PrivateState) -> OverallState:
    """中间处理节点"""
    additional_info = " (长文本)" if state["internal_flag"] else " (短文本)"
    return {
        "processed_data": state["private_data"] + additional_info
    }


def output_node(state: OverallState) -> OutputState:
    """输出生成节点"""
    final_result = f"最终结果: {state['processed_data']} | 步骤: {state.get('step_count', 0)}"
    return {
        "graph_output": final_result
    }


# 创建多模式状态图
builder = StateGraph(
    OverallState,
    input=InputState,
    output=OutputState
)

builder.add_node("input_processor", input_node)
builder.add_node("data_processor", processing_node)
builder.add_node("intermediate_processor", intermediate_node)
builder.add_node("output_processor", output_node)

builder.add_edge(START, "input_processor")
builder.add_edge("input_processor", "data_processor")
builder.add_edge("data_processor", "intermediate_processor")
builder.add_edge("intermediate_processor", "output_processor")
builder.add_edge("output_processor", END)

graph = builder.compile()

# 测试输入
input_data = {"user_input": "Hello LangGraph","extra_input":"额外输入"}

print("输入:", input_data)

 # 执行图
result = graph.invoke(input_data)

print("输出:", result)


png_data=graph.get_graph().draw_mermaid_png()
with open("state.png", "wb") as f:
    f.write(png_data)