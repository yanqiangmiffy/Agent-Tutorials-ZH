"""用户澄清和研究简报生成。

此模块实现研究工作流程的范围界定阶段，我们：
1. 评估用户的请求是否需要澄清
2. 从对话中生成详细的研究简报

工作流程使用结构化输出来做出关于是否存在足够上下文
以继续研究的确定性决策。
"""
# 初始化模型
import getpass
import os

from dotenv import load_dotenv
from langchain_deepseek import ChatDeepSeek
from typing import TypedDict
from langgraph.graph import StateGraph, START, END
load_dotenv()

from datetime import datetime
from typing_extensions import Literal

from langchain.chat_models import init_chat_model
from langchain_core.messages import HumanMessage, AIMessage, get_buffer_string
from langgraph.graph import StateGraph, START, END
from langgraph.types import Command
from langgraph.checkpoint.memory import InMemorySaver

from deep_research_from_scratch.prompts_zh import clarify_with_user_instructions, \
    transform_messages_into_research_topic_prompt
from deep_research_from_scratch.state_scope import AgentState, ClarifyWithUser, ResearchQuestion, AgentInputState


# ===== 实用函数 =====

def get_today_str() -> str:
    """获取人类可读格式的当前日期。"""
    print(datetime,datetime.now().strftime("%a %b %d, %Y"))
    return datetime.now().strftime("%a %b %d, %Y")


# ===== 配置 =====



def _set_env(var: str):
    if not os.environ.get(var):
        os.environ[var] = getpass.getpass(f"{var}: ")


_set_env("DEEPSEEK_API_KEY")

model = ChatDeepSeek(model="deepseek-chat")


# ===== 工作流程节点 =====

def clarify_with_user(state: AgentState) -> Command[Literal["write_research_brief", "__end__"]]:
    """
    确定用户的请求是否包含足够的信息以继续研究。

    使用结构化输出做出确定性决策并避免幻觉。
    路由到研究简报生成或以澄清问题结束。
    """
    # 设置结构化输出模型
    structured_output_model = model.with_structured_output(ClarifyWithUser)

    # 使用澄清指令调用模型
    response = structured_output_model.invoke([
        HumanMessage(content=clarify_with_user_instructions.format(
            messages=get_buffer_string(messages=state["messages"]),
            date=get_today_str()
        ))
    ])

    # 基于澄清需求进行路由
    if response.need_clarification:
        return Command(
            goto=END,
            update={"messages": [AIMessage(content=response.question)]}
        )
    else:
        return Command(
            goto="write_research_brief",
            update={"messages": [AIMessage(content=response.verification)]}
        )


def write_research_brief(state: AgentState):
    """
    将对话历史转换为全面的研究简报。

    使用结构化输出确保简报遵循所需格式
    并包含有效研究的所有必要细节。
    """
    # 设置结构化输出模型
    structured_output_model = model.with_structured_output(ResearchQuestion)

    # 从对话历史生成研究简报
    response = structured_output_model.invoke([
        HumanMessage(content=transform_messages_into_research_topic_prompt.format(
            messages=get_buffer_string(state.get("messages", [])),
            date=get_today_str()
        ))
    ])

    # 使用生成的研究简报更新状态并将其传递给监督者
    return {
        "research_brief": response.research_brief,
        "supervisor_messages": [HumanMessage(content=f"{response.research_brief}。")]
    }


# ===== 图构建 =====

# 构建范围界定工作流程
deep_researcher_builder = StateGraph(AgentState, input_schema=AgentInputState)

# 添加工作流程节点
deep_researcher_builder.add_node("clarify_with_user", clarify_with_user)
deep_researcher_builder.add_node("write_research_brief", write_research_brief)

# 添加工作流程边
deep_researcher_builder.add_edge(START, "clarify_with_user")
deep_researcher_builder.add_edge("write_research_brief", END)

# 编译工作流程
scope_research = deep_researcher_builder.compile()
png_data=scope_research.get_graph(xray=True).draw_mermaid_png()
with open("scope_research.png", "wb") as f:
    f.write(png_data)

checkpointer = InMemorySaver()

# # 运行工作流程
# from deep_research_from_scratch.utils import format_messages
# from langchain_core.messages import HumanMessage
# thread = {"configurable": {"thread_id": "1"}}
# result = scope_research.invoke({"messages": [HumanMessage(content="我想调研北京市最好的咖啡店。")]}, config=thread)
# format_messages(result['messages'])
#
#
# result = scope_research.invoke({"messages": [HumanMessage(content="让我们通过咖啡的质量来评估北京最好的咖啡店，质量的标准有价格、环境、服务、位置，这些信息你来不确定，不用询问我,评估10家就行，其他因素不用考虑了")]}, config=thread)
# format_messages(result['messages'])
#
# from rich.markdown import Markdown
# # print(Markdown(result["research_brief"]))
# print(result["research_brief"])