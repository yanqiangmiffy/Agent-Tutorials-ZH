"""研究智能体实现。

此模块实现了一个研究智能体，可以执行迭代式网络搜索
和综合分析来回答复杂的研究问题。
"""
import getpass
import os

from dotenv import load_dotenv
from langchain_deepseek import ChatDeepSeek
load_dotenv()
from deep_research_from_scratch.utils import format_messages
from typing_extensions import Literal

from langgraph.graph import StateGraph, START, END
from langchain_core.messages import SystemMessage, HumanMessage, ToolMessage, filter_messages

from deep_research_from_scratch.state_research import ResearcherState, ResearcherOutputState
from deep_research_from_scratch.utils import tavily_search, get_today_str, think_tool
from deep_research_from_scratch.prompts_zh import research_agent_prompt, compress_research_system_prompt, \
    compress_research_human_message

# ===== 配置 =====

# 设置工具和模型绑定
tools = [tavily_search, think_tool]
tools_by_name = {tool.name: tool for tool in tools}


def _set_env(var: str):
    if not os.environ.get(var):
        os.environ[var] = getpass.getpass(f"{var}: ")


_set_env("DEEPSEEK_API_KEY")

model = ChatDeepSeek(model="deepseek-chat")

# 初始化模型
model = ChatDeepSeek(model="deepseek-chat")
model_with_tools = model.bind_tools(tools)
summarization_model = ChatDeepSeek(model="deepseek-chat")
compress_model = ChatDeepSeek(model="deepseek-chat")


# ===== 智能体节点 =====

def llm_call(state: ResearcherState):
    """分析当前状态并决定下一步行动。

    模型分析当前对话状态并决定是否：
    1. 调用搜索工具收集更多信息
    2. 基于收集的信息提供最终答案

    返回包含模型响应的更新状态。
    """
    return {
        "researcher_messages": [
            model_with_tools.invoke(
                [SystemMessage(content=research_agent_prompt)] + state["researcher_messages"]
            )
        ]
    }


def tool_node(state: ResearcherState):
    """执行来自上一个LLM响应的所有工具调用。

    执行来自上一个LLM响应的所有工具调用。
    返回包含工具执行结果的更新状态。
    """
    tool_calls = state["researcher_messages"][-1].tool_calls

    # 执行所有工具调用
    observations = []
    for tool_call in tool_calls:
        tool = tools_by_name[tool_call["name"]]
        observations.append(tool.invoke(tool_call["args"]))

    # 创建工具消息输出
    tool_outputs = [
        ToolMessage(
            content=observation,
            name=tool_call["name"],
            tool_call_id=tool_call["id"]
        ) for observation, tool_call in zip(observations, tool_calls)
    ]

    return {"researcher_messages": tool_outputs}


def compress_research(state: ResearcherState) -> dict:
    """将founded发现压缩成简洁摘要。

    获取所有研究消息和工具输出，创建
    适合监督者决策的压缩摘要。
    """

    system_message = compress_research_system_prompt.format(date=get_today_str())
    messages = [SystemMessage(content=system_message)] + state.get("researcher_messages", []) + [
        HumanMessage(content=compress_research_human_message)]
    print("compress_research",messages)
    response = compress_model.invoke(messages)

    # 从工具和AI消息中提取原始笔记
    raw_notes = [
        str(m.content) for m in filter_messages(
            state["researcher_messages"],
            include_types=["tool", "ai"]
        )
    ]

    return {
        "compressed_research": str(response.content),
        "raw_notes": ["\n".join(raw_notes)]
    }


# ===== 路由逻辑 =====

def should_continue(state: ResearcherState) -> Literal["tool_node", "compress_research"]:
    """确定是否继续研究或提供最终答案。

    根据LLM是否进行了工具调用来确定智能体应该
    继续研究循环还是提供最终答案。

    返回:
        "tool_node": 继续工具执行
        "compress_research": 停止并压缩研究
    """
    messages = state["researcher_messages"]
    last_message = messages[-1]

    # 如果LLM进行了工具调用，继续工具执行
    if last_message.tool_calls:
        return "tool_node"
    # 否则，我们有了最终答案
    return "compress_research"


# ===== 图构建 =====

# 构建智能体工作流
agent_builder = StateGraph(ResearcherState, output_schema=ResearcherOutputState)

# 向图中添加节点
agent_builder.add_node("llm_call", llm_call)
agent_builder.add_node("tool_node", tool_node)
agent_builder.add_node("compress_research", compress_research)

# 添加边来连接节点
agent_builder.add_edge(START, "llm_call")
agent_builder.add_conditional_edges(
    "llm_call",
    should_continue,
    {
        "tool_node": "tool_node",  # 继续研究循环
        "compress_research": "compress_research",  # 提供最终答案
    },
)
agent_builder.add_edge("tool_node", "llm_call")  # 循环回去进行更多研究
agent_builder.add_edge("compress_research", END)

# 编译智能体
researcher_agent = agent_builder.compile()
png_data = researcher_agent.get_graph(xray=True).draw_mermaid_png()
with open("researcher_agent.png", "wb") as f:
    f.write(png_data)
# # 示例研究简报
# research_brief = """我需要研究北京最好的10家咖啡店，基于咖啡质量的四个核心标准：价格、环境、服务、位置。具体来说：
#
# 1. 价格维度：分析每家咖啡店的咖啡价格区间，包括不同饮品（如美式、拿铁、手冲等）的价格水平，但用户未指定具体的预算约束，因此考虑所有价格范围
#
# 2. 环境维度：评估咖啡店的装修风格、座位舒适度、空间布局、噪音水平、整体氛围等环境因素
#
# 3. 服务维度：考察员工专业程度、服务态度、出餐速度、个性化服务体验等服务质量指标
#
# 4. 位置维度：分析咖啡店的地理位置便利性，包括交通可达性、周边环境、是否靠近商业区或景点等
#
# 我需要收集这10家咖啡店的详细信息，包括但不限于：
# - 每家店的具体地址和联系方式
# - 营业时间
# - 价格菜单和饮品选择
# - 环境照片或描述
# - 顾客评价和评分
# - 特色咖啡和服务
#
# 优先考虑来自官方咖啡店网站、大众点评、美团等本地生活平台，以及咖啡爱好者社区的真实评价和信息。研究应该基于2025年的最新数据，确保信息的时效性和准确性。"""
#
# result = researcher_agent.invoke({"researcher_messages": [HumanMessage(content=f"{research_brief}.")]})
# print(result)
# format_messages(result['researcher_messages'])
