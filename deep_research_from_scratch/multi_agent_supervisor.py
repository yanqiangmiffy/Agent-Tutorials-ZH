"""用于协调多个专业智能体研究的多智能体监督器。

该模块实现了一种监督模式，其中：
1. 监督智能体协调研究活动并分配任务
2. 多个研究智能体独立处理特定子主题
3. 结果被聚合并压缩用于最终报告

监督器使用并行研究执行来提高效率，同时
为每个研究主题维护隔离的上下文窗口。
"""

import asyncio
import getpass
import os
from langchain.chat_models import init_chat_model
from langchain_deepseek import ChatDeepSeek
from langchain_core.messages import (
    HumanMessage,
    BaseMessage,
    SystemMessage,
    ToolMessage,
    filter_messages
)
from langgraph.graph import StateGraph, START, END
from langgraph.types import Command
from typing_extensions import Literal

from deep_research_from_scratch.prompts_zh import lead_researcher_prompt
from deep_research_from_scratch.research_agent import researcher_agent
from deep_research_from_scratch.state_multi_agent_supervisor import (
    SupervisorState,
    ConductResearch,
    ResearchComplete
)
from deep_research_from_scratch.utils import format_messages
from deep_research_from_scratch.utils import get_today_str, think_tool


def get_notes_from_tool_calls(messages: list[BaseMessage]) -> list[str]:
    """从监督器消息历史中的ToolMessage对象提取研究笔记。

    此函数检索子智能体作为ToolMessage内容返回的压缩founded发现。
    当监督器通过ConductResearch工具调用将研究委托给子智能体时，
    每个子智能体将其压缩发现作为ToolMessage的内容返回。
    此函数提取所有此类ToolMessage内容以编译最终研究笔记。

    参数：
        messages: 来自监督器对话历史的消息列表

    返回：
        从ToolMessage对象中提取的研究笔记字符串列表
    """
    return [tool_msg.content for tool_msg in filter_messages(messages, include_types="tool")]


# ===== 配置 =====
def _set_env(var: str):
    if not os.environ.get(var):
        os.environ[var] = getpass.getpass(f"{var}: ")


_set_env("DEEPSEEK_API_KEY")


supervisor_tools = [ConductResearch, ResearchComplete, think_tool]
supervisor_model = ChatDeepSeek(model="deepseek-chat")
supervisor_model_with_tools = supervisor_model.bind_tools(supervisor_tools)

# 系统常量
# 单个研究智能体的最大工具调用迭代次数
# 这可以防止无限循环并控制每个主题的研究深度
max_researcher_iterations = 6  # 对think_tool + ConductResearch的调用
# 监督器可以启动的最大并发研究智能体数量
# 这会传递给lead_researcher_prompt以限制并行研究任务
max_concurrent_researchers = 3


# ===== 监督器节点 =====

async def supervisor(state: SupervisorState) -> Command[Literal["supervisor_tools"]]:
    """协调研究活动。

    分析研究简报和当前进度以决定：
    - 需要调查哪些研究主题
    - 是否进行并行研究
    - 何时完成研究

    参数：
        state: 包含消息和研究进度的当前监督器状态

    返回：
        带有更新状态的命令，继续到supervisor_tools节点
    """
    supervisor_messages = state.get("supervisor_messages", [])

    # 准备带有当前日期和约束的系统消息
    system_message = lead_researcher_prompt.format(
        date=get_today_str(),
        max_concurrent_research_units=max_concurrent_researchers,
        max_researcher_iterations=max_researcher_iterations
    )
    messages = [SystemMessage(content=system_message)] + supervisor_messages

    # 对下一步研究步骤做出决定
    response = await supervisor_model_with_tools.ainvoke(messages)

    return Command(
        goto="supervisor_tools",
        update={
            "supervisor_messages": [response],
            "research_iterations": state.get("research_iterations", 0) + 1
        }
    )


async def supervisor_tools(state: SupervisorState) -> Command[Literal["supervisor", "__end__"]]:
    """执行监督器决策 - 进行研究或结束流程。

    处理：
    - 执行think_tool调用进行战略反思
    - 为不同主题启动并行研究智能体
    - 聚合研究结果
    - 确定何时完成研究

    参数：
        state: 包含消息和迭代计数的当前监督器状态

    返回：
        继续监督、结束流程或处理错误的命令
    """
    supervisor_messages = state.get("supervisor_messages", [])
    research_iterations = state.get("research_iterations", 0)
    most_recent_message = supervisor_messages[-1]

    # 初始化单一返回模式的变量
    tool_messages = []
    all_raw_notes = []
    next_step = "supervisor"  # 默认下一步
    should_end = False

    # 首先检查退出条件
    exceeded_iterations = research_iterations >= max_researcher_iterations
    no_tool_calls = not most_recent_message.tool_calls
    research_complete = any(
        tool_call["name"] == "ResearchComplete"
        for tool_call in most_recent_message.tool_calls
    )

    if exceeded_iterations or no_tool_calls or research_complete:
        should_end = True
        next_step = END

    else:
        # 在决定下一步之前执行所有工具调用
        try:
            # 将think_tool调用与ConductResearch调用分开
            think_tool_calls = [
                tool_call for tool_call in most_recent_message.tool_calls
                if tool_call["name"] == "think_tool"
            ]

            conduct_research_calls = [
                tool_call for tool_call in most_recent_message.tool_calls
                if tool_call["name"] == "ConductResearch"
            ]

            # 处理think_tool调用（同步）
            for tool_call in think_tool_calls:
                observation = think_tool.invoke(tool_call["args"])
                tool_messages.append(
                    ToolMessage(
                        content=observation,
                        name=tool_call["name"],
                        tool_call_id=tool_call["id"]
                    )
                )

            # 处理ConductResearch调用（异步）
            if conduct_research_calls:
                # 启动并行研究智能体
                coros = [
                    researcher_agent.ainvoke({
                        "researcher_messages": [
                            HumanMessage(content=tool_call["args"]["research_topic"])
                        ],
                        "research_topic": tool_call["args"]["research_topic"]
                    })
                    for tool_call in conduct_research_calls
                ]

                # 等待所有研究完成
                tool_results = await asyncio.gather(*coros)

                # 将研究结果格式化为工具消息
                # 每个子智能体在result["compressed_research"]中返回压缩的研究发现
                # 我们将这些压缩研究作为ToolMessage的内容写入，这允许
                # 监督器稍后通过get_notes_from_tool_calls()检索这些发现
                research_tool_messages = [
                    ToolMessage(
                        content=result.get("compressed_research", "Error synthesizing research report"),
                        name=tool_call["name"],
                        tool_call_id=tool_call["id"]
                    ) for result, tool_call in zip(tool_results, conduct_research_calls)
                ]

                tool_messages.extend(research_tool_messages)

                # 聚合所有研究的原始笔记
                all_raw_notes = [
                    "\n".join(result.get("raw_notes", []))
                    for result in tool_results
                ]

        except Exception as e:
            print(f"Error in supervisor tools: {e}")
            should_end = True
            next_step = END

    # 带有适当状态更新的单一返回点
    if should_end:
        return Command(
            goto=next_step,
            update={
                "notes": get_notes_from_tool_calls(supervisor_messages),
                "research_brief": state.get("research_brief", "")
            }
        )
    else:
        return Command(
            goto=next_step,
            update={
                "supervisor_messages": tool_messages,
                "raw_notes": all_raw_notes
            }
        )


# ===== 图构建 =====

# 构建监督器图
supervisor_builder = StateGraph(SupervisorState)
supervisor_builder.add_node("supervisor", supervisor)
supervisor_builder.add_node("supervisor_tools", supervisor_tools)
supervisor_builder.add_edge(START, "supervisor")
supervisor_agent = supervisor_builder.compile()

png_data = supervisor_agent.get_graph(xray=True).draw_mermaid_png()
with open("supervisor_agent.png", "wb") as f:
    f.write(png_data)

# 运行多智能体监督器智能体

research_brief = """我需要研究北京最好的10家咖啡店，基于咖啡质量的四个核心标准：价格、环境、服务、位置。具体来说：

1. 价格维度：分析每家咖啡店的咖啡价格区间，包括不同饮品（如美式、拿铁、手冲等）的价格水平，但用户未指定具体的预算约束，因此考虑所有价格范围

2. 环境维度：评估咖啡店的装修风格、座位舒适度、空间布局、噪音水平、整体氛围等环境因素

3. 服务维度：考察员工专业程度、服务态度、出餐速度、个性化服务体验等服务质量指标

4. 位置维度：分析咖啡店的地理位置便利性，包括交通可达性、周边环境、是否靠近商业区或景点等

我需要收集这10家咖啡店的详细信息，包括但不限于：
- 每家店的具体地址和联系方式
- 营业时间
- 价格菜单和饮品选择
- 环境照片或描述
- 顾客评价和评分
- 特色咖啡和服务

优先考虑来自官方咖啡店网站、大众点评、美团等本地生活平台，以及咖啡爱好者社区的真实评价和信息。研究应该基于2025年的最新数据，确保信息的时效性和准确性。"""


async def main():

    result = await supervisor_agent.ainvoke({"supervisor_messages": [HumanMessage(content=f"{research_brief}.")]})
    print(result)
    format_messages(result['supervisor_messages'])


# # 运行异步主函数
# if __name__ == "__main__":
#     asyncio.run(main())
