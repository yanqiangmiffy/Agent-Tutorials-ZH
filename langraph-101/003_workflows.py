import getpass
import os

from dotenv import load_dotenv
from langchain_deepseek import ChatDeepSeek
from typing import TypedDict
from langgraph.graph import StateGraph, START, END
load_dotenv()


def _set_env(var: str):
    if not os.environ.get(var):
        os.environ[var] = getpass.getpass(f"{var}: ")


_set_env("DEEPSEEK_API_KEY")

llm = ChatDeepSeek(model="deepseek-chat")

# from pydantic import BaseModel, Field
#
# class SearchQuery(BaseModel):
#     search_query: str = Field(None, description="为网络搜索优化过的查询。")
#     justification: str = Field(
#         None, description="说明此查询与用户请求的相关性。"
#     )
#
# # 使用结构化输出的 schema 增强 LLM
# structured_llm = llm.with_structured_output(SearchQuery)
#
# # 调用增强后的 LLM
# output = structured_llm.invoke("钙化分数（Calcium CT score）与高胆固醇之间有什么关系？")# 定义一个工具
# print(output)
#
# def multiply(a: int, b: int) -> int:
#     return a * b
#
# # 使用工具增强 LLM
# llm_with_tools = llm.bind_tools([multiply])
#
# # 调用 LLM，输入触发工具调用
# msg = llm_with_tools.invoke("2 乘以 3 是多少？")
#
# # 获取工具调用
#
# print(msg.tool_calls)


# from typing_extensions import TypedDict
# from langgraph.graph import StateGraph, START, END
# from IPython.display import Image, display
#
#
# # 图状态
# class State(TypedDict):
#     topic: str
#     joke: str
#     improved_joke: str
#     final_joke: str
#
#
# # 节点
# def generate_joke(state: State):
#     """第一次 LLM 调用，生成初始笑话"""
#
#     msg = llm.invoke(f"写一个关于{state['topic']}的短笑话")
#     return {"joke": msg.content}
#
#
# def check_punchline(state: State):
#     """门控函数，检查笑话是否有包袱"""
#
#     # 简单检查 - 笑话是否包含"?"或"!"
#     if "?" in state["joke"] or "!" in state["joke"]:
#         return "Pass"
#     return "Fail"
#
#
# def improve_joke(state: State):
#     """第二次 LLM 调用，改进笑话"""
#
#     msg = llm.invoke(f"通过添加双关语让这个笑话更有趣: {state['joke']}")
#     return {"improved_joke": msg.content}
#
#
# def polish_joke(state: State):
#     """第三次 LLM 调用，进行最终润色"""
#
#     msg = llm.invoke(f"给这个笑话添加一个出人意料的转折: {state['improved_joke']}")
#     return {"final_joke": msg.content}
#
#
# # 构建工作流
# workflow = StateGraph(State)
#
# # 添加节点
# workflow.add_node("generate_joke", generate_joke)
# workflow.add_node("improve_joke", improve_joke)
# workflow.add_node("polish_joke", polish_joke)
#
# # 添加边以连接节点
# workflow.add_edge(START, "generate_joke")
# workflow.add_conditional_edges(
#     "generate_joke", check_punchline, {"Fail": "improve_joke", "Pass": END}
# )
# workflow.add_edge("improve_joke", "polish_joke")
# workflow.add_edge("polish_joke", END)
#
# # 编译
# chain = workflow.compile()
#
# # 显示工作流
# display(Image(chain.get_graph().draw_mermaid_png()))
# png_data=chain.get_graph().draw_mermaid_png()
# # 保存PNG格式
# with open("workflow_graph.png", "wb") as f:
#     f.write(png_data)
#
#
#
# # 调用
# state = chain.invoke({"topic": "猫"})
# print("初始笑话:")
# print(state["joke"])
# print("\n--- --- ---\n")
# if "improved_joke" in state:
#     print("改进后的笑话:")
#     print(state["improved_joke"])
#     print("\n--- --- ---\n")
#     print("最终笑话:")
#     print(state["final_joke"])
# else:
#     print("笑话未能通过质量门检查 - 未检测到包袱！")



# from langgraph.func import entrypoint, task
#
# # 任务
# @task
# def generate_joke(topic: str):
#     """第一次 LLM 调用，生成初始笑话"""
#     msg = llm.invoke(f"写一个关于{topic}的短笑话")
#     return msg.content
#
# def check_punchline(joke: str):
#     """门控函数，检查笑话是否有包袱"""
#     # 简单检查 - 笑话是否包含"?"或"!"
#     if "?" in joke or "!" in joke:
#         return "Fail"
#     return "Pass"
#
# @task
# def improve_joke(joke: str):
#     """第二次 LLM 调用，改进笑话"""
#     msg = llm.invoke(f"通过添加双关语让这个笑话更有趣: {joke}")
#     return msg.content
#
# @task
# def polish_joke(joke: str):
#     """第三次 LLM 调用，进行最终润色"""
#     msg = llm.invoke(f"给这个笑话添加一个出人意料的转折: {joke}")
#     return msg.content
#
# @entrypoint()
# def prompt_chaining_workflow(topic: str):
#     original_joke = generate_joke(topic).result()
#     if check_punchline(original_joke) == "Pass":
#         return original_joke
#
#     improved_joke = improve_joke(original_joke).result()
#     return polish_joke(improved_joke).result()
#
# # 调用
# for step in prompt_chaining_workflow.stream("猫", stream_mode="updates"):
#     print(step)
#     print("\n")
#


# 图状态
class State(TypedDict):
    topic: str
    joke: str
    story: str
    poem: str
    combined_output: str# 节点
def call_llm_1(state: State):
    """第一次 LLM 调用，生成初始笑话"""

    msg = llm.invoke(f"写一个关于{state['topic']}的笑话")
    return {"joke": msg.content}

def call_llm_2(state: State):
    """第二次 LLM 调用，生成故事"""

    msg = llm.invoke(f"写一个关于{state['topic']}的故事")
    return {"story": msg.content}

def call_llm_3(state: State):
    """第三次 LLM 调用，生成诗歌"""

    msg= llm.invoke(f"写一首关于{state['topic']}的诗歌")
    return {"poem": msg.content}

def aggregator(state: State):
    """将笑话、故事和诗歌合并为单个输出"""

    combined = f"这里有一个关于{state['topic']}的故事、笑话和诗歌!\n\n"
    combined += f"故事:\n{state['story']}\n\n"
    combined += f"笑话:\n{state['joke']}\n\n"
    combined += f"诗歌:\n{state['poem']}"
    return {"combined_output": combined}# 构建工作流
parallel_builder = StateGraph(State)

# 添加节点
parallel_builder.add_node("call_llm_1", call_llm_1)
parallel_builder.add_node("call_llm_2", call_llm_2)
parallel_builder.add_node("call_llm_3", call_llm_3)
parallel_builder.add_node("aggregator", aggregator)

# 添加边以连接节点
parallel_builder.add_edge(START, "call_llm_1")
parallel_builder.add_edge(START, "call_llm_2")
parallel_builder.add_edge(START, "call_llm_3")
parallel_builder.add_edge("call_llm_1", "aggregator")
parallel_builder.add_edge("call_llm_2", "aggregator")
parallel_builder.add_edge("call_llm_3", "aggregator")
parallel_builder.add_edge("aggregator", END)
parallel_workflow = parallel_builder.compile()

# 显示工作流
png_data=parallel_workflow.get_graph().draw_mermaid_png()
with open("parallel_workflow.png", "wb") as f:
    f.write(png_data)

# 调用
state = parallel_workflow.invoke({"topic":"猫"})
print(state["combined_output"])