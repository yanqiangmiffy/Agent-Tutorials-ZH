import json
from pydantic import BaseModel, Field
from langchain_core.messages import HumanMessage, AIMessage
# from langchain_openai import ChatOpenAI
from langchain_deepseek import ChatDeepSeek
from langgraph.graph import StateGraph, MessagesState, END
from dotenv import load_dotenv
load_dotenv()

# 定义结构化输出
class QueryResult(BaseModel):
    topic: str = Field(description="主题")
    summary: str = Field(description="摘要")
    keywords: list[str] = Field(description="关键词")

# 初始化模型
model = ChatDeepSeek(model="deepseek-chat")
structured_llm = model.with_structured_output(QueryResult).with_config(tags=["query_writer"])

# 定义节点
def query_node(state: MessagesState):
    result = structured_llm.invoke(state["messages"])
    print(result)
    # 将QueryResult转换为AIMessage
    response_content = f"主题: {result.topic}\n摘要: {result.summary}\n关键词: {', '.join(result.keywords)}"
    return {"messages": [AIMessage(content=response_content)]}

# 构建图
workflow = StateGraph(MessagesState)
workflow.add_node("query", query_node)
workflow.set_entry_point("query")
workflow.add_edge("query", END)
graph = workflow.compile()

# 使用示例
graph_input = {"messages": [HumanMessage(content="分析一下人工智能的发展趋势")]}

# 同步流式获取结果函数
def stream_results():
    for mode, chunk in graph.stream(graph_input, stream_mode=["messages"]):
        if mode == "messages":
            print(f"Mode: {mode}")
            print(f"Chunk: {chunk}")
            # msg, metadata = chunk
            # print(f"Message: {msg.content}")
            # print(f"Metadata: {metadata}")
            # if "tool_calls" in msg.additional_kwargs:
            #     content = msg.additional_kwargs["tool_calls"][0]["function"]["arguments"]
            #     print("content",content)
# 运行函数
def run_stream():
    stream_results()

# 执行
run_stream()