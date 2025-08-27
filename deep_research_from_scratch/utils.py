"""研究工具和实用程序。

此模块为研究智能体提供搜索和内容处理实用程序，
包括网络搜索功能和内容摘要工具。
"""
from rich.console import Console
from rich.panel import Panel
from rich.text import Text
import json
from pathlib import Path
from datetime import datetime
from typing_extensions import Annotated, List, Literal

from langchain.chat_models import init_chat_model
from langchain_core.messages import HumanMessage
from langchain_core.runnables import RunnableConfig
from langchain_core.tools import tool, InjectedToolArg
from tavily import TavilyClient

from deep_research_from_scratch.state_research import Summary
from deep_research_from_scratch.prompts_zh import summarize_webpage_prompt

# ===== 实用工具函数 =====

def get_today_str() -> str:
    """获取当前日期的人类可读格式。"""
    return datetime.now().strftime("%a %b %-d, %Y")

def get_current_dir() -> Path:
    """获取模块的当前目录。

    此函数与Jupyter笔记本和常规Python脚本兼容。

    Returns:
        表示当前目录的Path对象
    """
    try:
        return Path(__file__).resolve().parent
    except NameError:  # __file__ 未定义
        return Path.cwd()

# ===== 配置 =====

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

model = ChatDeepSeek(model="deepseek-chat")

tavily_client = TavilyClient()

# ===== 搜索函数 =====

def tavily_search_multiple(
    search_queries: List[str],
    max_results: int = 3,
    topic: Literal["general", "news", "finance"] = "general",
    include_raw_content: bool = True,
) -> List[dict]:
    """使用Tavily API对多个查询执行搜索。

    Args:
        search_queries: 要执行的搜索查询列表
        max_results: 每个查询的最大结果数
        topic: 搜索结果的主题过滤器
        include_raw_content: 是否包含原始网页内容

    Returns:
        搜索结果字典列表
    """

    # 顺序执行搜索。注意：你可以使用AsyncTavilyClient来并行化这一步。
    search_docs = []
    for query in search_queries:
        result = tavily_client.search(
            query,
            max_results=max_results,
            include_raw_content=include_raw_content,
            topic=topic
        )
        search_docs.append(result)

    return search_docs

def summarize_webpage_content(webpage_content: str) -> str:
    """使用配置的摘要模型总结网页内容。

    Args:
        webpage_content: 要总结的原始网页内容

    Returns:
        带有关键摘录的格式化摘要
    """
    try:
        # 设置用于摘要的结构化输出模型
        structured_model = summarization_model.with_structured_output(Summary)

        # 生成摘要
        summary = structured_model.invoke([
            HumanMessage(content=summarize_webpage_prompt.format(
                webpage_content=webpage_content,
                date=get_today_str()
            ))
        ])

        # 格式化摘要，结构清晰
        formatted_summary = (
            f"<summary>\n{summary.summary}\n</summary>\n\n"
            f"<key_excerpts>\n{summary.key_excerpts}\n</key_excerpts>"
        )

        return formatted_summary

    except Exception as e:
        print(f"网页摘要失败: {str(e)}")
        return webpage_content[:1000] + "..." if len(webpage_content) > 1000 else webpage_content

def deduplicate_search_results(search_results: List[dict]) -> dict:
    """通过URL去重搜索结果，避免处理重复内容。

    Args:
        search_results: 搜索结果字典列表

    Returns:
        将URL映射到唯一结果的字典
    """
    unique_results = {}

    for response in search_results:
        for result in response['results']:
            url = result['url']
            if url not in unique_results:
                unique_results[url] = result

    return unique_results

def process_search_results(unique_results: dict) -> dict:
    """通过在可用时总结内容来处理搜索结果。

    Args:
        unique_results: 唯一搜索结果字典

    Returns:
        带有摘要的处理结果字典
    """
    summarized_results = {}

    for url, result in unique_results.items():
        # 如果没有原始内容用于摘要，则使用现有内容
        if not result.get("raw_content"):
            content = result['content']
        else:
            # 总结原始内容以便更好地处理
            content = summarize_webpage_content(result['raw_content'])

        summarized_results[url] = {
            'title': result['title'],
            'content': content
        }

    return summarized_results

def format_search_output(summarized_results: dict) -> str:
    """将搜索结果格式化为结构良好的字符串输出。

    Args:
        summarized_results: 处理过的搜索结果字典

    Returns:
        格式化的搜索结果字符串，具有清晰的来源分离
    """
    if not summarized_results:
        return "未找到有效的搜索结果。请尝试不同的搜索查询或使用不同的搜索API。"

    formatted_output = "搜索结果: \n\n"

    for i, (url, result) in enumerate(summarized_results.items(), 1):
        formatted_output += f"\n\n--- 来源 {i}: {result['title']} ---\n"
        formatted_output += f"URL: {url}\n\n"
        formatted_output += f"摘要:\n{result['content']}\n\n"
        formatted_output += "-" * 80 + "\n"

    return formatted_output

# ===== 研究工具 =====

@tool(parse_docstring=True)
def tavily_search(
    query: str,
    max_results: Annotated[int, InjectedToolArg] = 3,
    topic: Annotated[Literal["general", "news", "finance"], InjectedToolArg] = "general",
) -> str:
    """从Tavily搜索API获取结果并进行内容摘要。

    Args:
        query: 要执行的单个搜索查询
        max_results: 返回的最大结果数
        topic: 按主题过滤结果（'general'、'news'、'finance'）

    Returns:
        带有摘要的格式化搜索结果字符串
    """
    # 对单个查询执行搜索
    search_results = tavily_search_multiple(
        [query],  # 将单个查询转换为列表以供内部函数使用
        max_results=max_results,
        topic=topic,
        include_raw_content=True,
    )

    # 通过URL去重结果，避免处理重复内容
    unique_results = deduplicate_search_results(search_results)

    # 使用摘要处理结果
    summarized_results = process_search_results(unique_results)

    # 格式化输出以供使用
    return format_search_output(summarized_results)

@tool(parse_docstring=True)
def think_tool(reflection: str) -> str:
    """用于研究进展和决策制定的战略反思工具。

    在每次搜索后使用此工具来系统地分析结果并规划下一步。
    这在研究工作流程中创建了一个深思熟虑的暂停，以便进行高质量的决策制定。

    何时使用：
    - 收到搜索结果后：我找到了什么关键信息？
    - 决定下一步之前：我是否有足够的信息来全面回答？
    - 评估研究差距时：我仍然缺少什么具体信息？
    - 结束研究之前：我现在能提供完整的答案吗？

    反思应该涉及：
    1. 当前发现的分析 - 我收集了什么具体信息？
    2. 差距评估 - 仍然缺少什么关键信息？
    3. 质量评估 - 我是否有足够的证据/例子来提供好的答案？
    4. 战略决策 - 我应该继续搜索还是提供我的答案？

    Args:
        reflection: 你对研究进展、发现、差距和下一步的详细反思

    Returns:
        确认反思已记录用于决策制定
    """
    return f"反思已记录: {reflection}"


console = Console()


def format_message_content(message):
    """将消息内容转换为可显示的字符串"""
    parts = []
    tool_calls_processed = False

    # 处理主要内容
    if isinstance(message.content, str):
        parts.append(message.content)
    elif isinstance(message.content, list):
        # 处理复杂内容，如工具调用（Anthropic格式）
        for item in message.content:
            if item.get('type') == 'text':
                parts.append(item['text'])
            elif item.get('type') == 'tool_use':
                parts.append(f"\n🔧 工具调用: {item['name']}")
                parts.append(f"   参数: {json.dumps(item['input'], indent=2)}")
                parts.append(f"   ID: {item.get('id', 'N/A')}")
                tool_calls_processed = True
    else:
        parts.append(str(message.content))

    # 处理附加到消息的工具调用（OpenAI格式）- 仅在尚未处理时
    if not tool_calls_processed and hasattr(message, 'tool_calls') and message.tool_calls:
        for tool_call in message.tool_calls:
            parts.append(f"\n🔧 工具调用: {tool_call['name']}")
            parts.append(f"   参数: {json.dumps(tool_call['args'], indent=2)}")
            parts.append(f"   ID: {tool_call['id']}")

    return "\n".join(parts)


def format_messages(messages):
    """使用Rich格式化格式化和显示消息列表"""
    for m in messages:
        msg_type = m.__class__.__name__.replace('Message', '')
        content = format_message_content(m)

        if msg_type == 'Human':
            console.print(Panel(content, title="🧑 人类", border_style="blue"))
        elif msg_type == 'Ai':
            console.print(Panel(content, title="🤖 助手", border_style="green"))
        elif msg_type == 'Tool':
            console.print(Panel(content, title="🔧 工具输出", border_style="yellow"))
        else:
            console.print(Panel(content, title=f"📝 {msg_type}", border_style="white"))


def format_message(messages):
    """format_messages的别名，用于向后兼容"""
    return format_messages(messages)


def show_prompt(prompt_text: str, title: str = "提示", border_style: str = "blue"):
    """
    使用丰富的格式和XML标签高亮显示提示。

    Args:
        prompt_text: 要显示的提示字符串
        title: 面板标题（默认："提示"）
        border_style: 边框颜色样式（默认："blue"）
    """
    # 创建提示的格式化显示
    formatted_text = Text(prompt_text)
    formatted_text.highlight_regex(r'<[^>]+>', style="bold blue")  # 高亮XML标签
    formatted_text.highlight_regex(r'##[^#\n]+', style="bold magenta")  # 高亮标题
    formatted_text.highlight_regex(r'###[^#\n]+', style="bold cyan")  # 高亮子标题

    # 在面板中显示以获得更好的呈现效果
    console.print(Panel(
        formatted_text,
        title=f"[bold green]{title}[/bold green]",
        border_style=border_style,
        padding=(1, 2)
    ))