import asyncio
import time
from typing import Annotated, List, TypedDict, Dict
import operator
from dotenv import load_dotenv
from langchain_deepseek import ChatDeepSeek
from langgraph.graph import StateGraph, START, END
from langgraph.types import Send
from pydantic import BaseModel, Field
from langchain_core.messages import HumanMessage, SystemMessage

load_dotenv()

# 初始化 LLM 模型
llm = ChatDeepSeek(model="deepseek-chat")


# 详细大纲结构定义
class SubSection(BaseModel):
    title: str = Field(description="子章节标题")
    key_points: List[str] = Field(description="该子章节的关键要点")
    content_focus: str = Field(description="内容重点和角度")
    word_count: int = Field(description="预期字数")


class DetailedSection(BaseModel):
    title: str = Field(description="章节标题")
    description: str = Field(description="章节总体描述")
    agent_type: str = Field(description="负责的智能体类型")
    subsections: List[SubSection] = Field(description="子章节列表")
    key_concepts: List[str] = Field(description="本章节涉及的关键概念")
    connections: List[str] = Field(description="与其他章节的关联点")


class ComprehensiveOutline(BaseModel):
    report_title: str = Field(description="报告标题")
    executive_summary: str = Field(description="执行摘要")
    sections: List[DetailedSection] = Field(description="详细章节列表")
    key_themes: List[str] = Field(description="贯穿全文的主要主题")
    terminology: Dict[str, str] = Field(description="统一术语定义")
    writing_style: str = Field(description="写作风格指导")


# 使用结构化输出的大纲规划器
outline_planner = llm.with_structured_output(ComprehensiveOutline)


# 改进的状态定义
class ImprovedReportState(TypedDict):
    topic: str  # 报告主题
    comprehensive_outline: ComprehensiveOutline  # 详细大纲
    completed_sections: Annotated[List[str], operator.add]  # 已完成的章节内容
    final_report: str  # 最终合并的报告
    shared_context: Dict[str, str]  # 共享上下文信息
    metadata: dict  # 元数据信息


# 工作者状态定义
class ImprovedWorkerState(TypedDict):
    section: DetailedSection
    topic: str
    comprehensive_outline: ComprehensiveOutline
    shared_context: Dict[str, str]
    completed_sections: Annotated[List[str], operator.add]


# 1. 大纲规划智能体 - 生成详细的统一大纲
def outline_planning_agent(state: ImprovedReportState) -> ImprovedReportState:
    """大纲规划智能体：生成详细的统一大纲和写作指导"""
    print("📋 大纲规划智能体开始工作...")

    outline_prompt = f"""
    你是一个专业的报告大纲规划专家。请为主题「{state['topic']}」制定一个详细、连贯的报告大纲。

    要求：
    1. 报告应包含4个主要章节，逻辑递进，相互关联
    2. 每个章节需要详细的子章节规划，包含具体要点
    3. 明确各章节的关键概念和相互关联
    4. 统一术语定义，确保全文一致性
    5. 为每个章节指定最适合的智能体类型：
       - research_agent: 负责背景研究、现状分析
       - technical_agent: 负责技术原理、方法论
       - analysis_agent: 负责案例分析、数据评估
       - strategy_agent: 负责前景展望、政策建议

    请确保：
    - 章节间有明确的逻辑关系和过渡
    - 避免内容重复，明确各章节的独特角度
    - 提供统一的写作风格指导
    - 定义关键术语，确保全文一致性
    """

    comprehensive_outline = outline_planner.invoke([
        SystemMessage(content=outline_prompt),
        HumanMessage(content=f"报告主题：{state['topic']}")
    ])

    print(f"📋 详细大纲规划完成：")
    print(f"  报告标题：{comprehensive_outline.report_title}")
    print(f"  主要主题：{', '.join(comprehensive_outline.key_themes)}")
    print(f"  章节数量：{len(comprehensive_outline.sections)}")

    for i, section in enumerate(comprehensive_outline.sections, 1):
        print(f"  {i}. {section.title} ({section.agent_type})")
        print(f"     子章节：{len(section.subsections)}个")
        print(f"     关键概念：{', '.join(section.key_concepts[:3])}...")

    # 创建共享上下文
    shared_context = {
        "key_themes": ", ".join(comprehensive_outline.key_themes),
        "writing_style": comprehensive_outline.writing_style,
        "terminology": str(comprehensive_outline.terminology)
    }

    return {
        "comprehensive_outline": comprehensive_outline,
        "shared_context": shared_context,
        "metadata": {
            "outline_planning_time": time.time(),
            "total_sections": len(comprehensive_outline.sections)
        }
    }


# 2. 改进的研究智能体
def improved_research_agent(state: ImprovedWorkerState) -> ImprovedWorkerState:
    """改进的研究智能体：基于详细大纲进行背景研究"""
    section = state["section"]
    topic = state["topic"]
    outline = state["comprehensive_outline"]
    shared_context = state["shared_context"]

    print(f"🔍 研究智能体正在撰写：{section.title}")

    research_prompt = f"""
    你是一个专业的研究智能体，请严格按照以下大纲和指导撰写报告章节。

    【报告整体信息】
    - 报告主题：{topic}
    - 报告标题：{outline.report_title}
    - 主要主题：{shared_context['key_themes']}
    - 写作风格：{shared_context['writing_style']}

    【本章节详细要求】
    - 章节标题：{section.title}
    - 章节描述：{section.description}
    - 关键概念：{', '.join(section.key_concepts)}
    - 与其他章节关联：{', '.join(section.connections)}

    【子章节结构】
    {chr(10).join([f"- {sub.title}: {sub.content_focus} (约{sub.word_count}字)" for sub in section.subsections])}

    【具体要求】
    1. 严格按照子章节结构组织内容
    2. 重点关注背景研究和现状分析
    3. 使用统一的术语定义：{shared_context['terminology']}
    4. 确保与其他章节的关联点得到体现
    5. 使用Markdown格式，包含适当的标题层级
    6. 总字数控制在800-1200字

    请开始撰写，确保内容与整体大纲高度一致：
    """

    response = llm.invoke([
        SystemMessage(content=research_prompt),
        HumanMessage(content=f"开始撰写章节：{section.title}")
    ])

    return {"completed_sections": [f"## {section.title}\n\n{response.content}"]}


# 3. 改进的技术智能体
def improved_technical_agent(state: ImprovedWorkerState) -> ImprovedWorkerState:
    """改进的技术智能体：基于详细大纲阐述技术原理"""
    section = state["section"]
    topic = state["topic"]
    outline = state["comprehensive_outline"]
    shared_context = state["shared_context"]

    print(f"⚙️ 技术智能体正在撰写：{section.title}")

    technical_prompt = f"""
    你是一个专业的技术智能体，请严格按照以下大纲和指导撰写报告章节。

    【报告整体信息】
    - 报告主题：{topic}
    - 报告标题：{outline.report_title}
    - 主要主题：{shared_context['key_themes']}
    - 写作风格：{shared_context['writing_style']}

    【本章节详细要求】
    - 章节标题：{section.title}
    - 章节描述：{section.description}
    - 关键概念：{', '.join(section.key_concepts)}
    - 与其他章节关联：{', '.join(section.connections)}

    【子章节结构】
    {chr(10).join([f"- {sub.title}: {sub.content_focus} (约{sub.word_count}字)" for sub in section.subsections])}

    【具体要求】
    1. 严格按照子章节结构组织内容
    2. 重点阐述技术原理和方法论
    3. 使用统一的术语定义：{shared_context['terminology']}
    4. 与前面章节的背景研究形成呼应
    5. 为后续章节的案例分析提供技术基础
    6. 使用Markdown格式，包含代码示例或技术图表
    7. 总字数控制在800-1200字

    请开始撰写，确保技术内容与整体报告逻辑一致：
    """

    response = llm.invoke([
        SystemMessage(content=technical_prompt),
        HumanMessage(content=f"开始撰写章节：{section.title}")
    ])

    return {"completed_sections": [f"## {section.title}\n\n{response.content}"]}


# 4. 改进的分析智能体
def improved_analysis_agent(state: ImprovedWorkerState) -> ImprovedWorkerState:
    """改进的分析智能体：基于前面章节进行案例分析"""
    section = state["section"]
    topic = state["topic"]
    outline = state["comprehensive_outline"]
    shared_context = state["shared_context"]

    print(f"📊 分析智能体正在撰写：{section.title}")

    analysis_prompt = f"""
    你是一个专业的分析智能体，请严格按照以下大纲和指导撰写报告章节。

    【报告整体信息】
    - 报告主题：{topic}
    - 报告标题：{outline.report_title}
    - 主要主题：{shared_context['key_themes']}
    - 写作风格：{shared_context['writing_style']}

    【本章节详细要求】
    - 章节标题：{section.title}
    - 章节描述：{section.description}
    - 关键概念：{', '.join(section.key_concepts)}
    - 与其他章节关联：{', '.join(section.connections)}

    【子章节结构】
    {chr(10).join([f"- {sub.title}: {sub.content_focus} (约{sub.word_count}字)" for sub in section.subsections])}

    【具体要求】
    1. 严格按照子章节结构组织内容
    2. 基于前面章节的背景和技术基础进行案例分析
    3. 使用统一的术语定义：{shared_context['terminology']}
    4. 提供具体数据和实证分析
    5. 为最后的前景展望提供现实依据
    6. 使用Markdown格式，包含表格和数据展示
    7. 总字数控制在800-1200字

    请开始撰写，确保分析内容承上启下：
    """

    response = llm.invoke([
        SystemMessage(content=analysis_prompt),
        HumanMessage(content=f"开始撰写章节：{section.title}")
    ])

    return {"completed_sections": [f"## {section.title}\n\n{response.content}"]}


# 5. 改进的策略智能体
def improved_strategy_agent(state: ImprovedWorkerState) -> ImprovedWorkerState:
    """改进的策略智能体：基于全文内容提供前景展望"""
    section = state["section"]
    topic = state["topic"]
    outline = state["comprehensive_outline"]
    shared_context = state["shared_context"]

    print(f"🎯 策略智能体正在撰写：{section.title}")

    strategy_prompt = f"""
    你是一个专业的策略智能体，请严格按照以下大纲和指导撰写报告的总结章节。

    【报告整体信息】
    - 报告主题：{topic}
    - 报告标题：{outline.report_title}
    - 主要主题：{shared_context['key_themes']}
    - 写作风格：{shared_context['writing_style']}

    【本章节详细要求】
    - 章节标题：{section.title}
    - 章节描述：{section.description}
    - 关键概念：{', '.join(section.key_concepts)}
    - 与其他章节关联：{', '.join(section.connections)}

    【子章节结构】
    {chr(10).join([f"- {sub.title}: {sub.content_focus} (约{sub.word_count}字)" for sub in section.subsections])}

    【具体要求】
    1. 严格按照子章节结构组织内容
    2. 总结前面章节的核心发现和观点
    3. 基于背景、技术、案例分析提出前景展望
    4. 使用统一的术语定义：{shared_context['terminology']}
    5. 提供具体可行的政策建议
    6. 使用Markdown格式，条理清晰
    7. 总字数控制在600-1000字

    请开始撰写，确保内容与前面章节形成完整闭环：
    """

    response = llm.invoke([
        SystemMessage(content=strategy_prompt),
        HumanMessage(content=f"开始撰写章节：{section.title}")
    ])

    return {"completed_sections": [f"## {section.title}\n\n{response.content}"]}


# 6. 改进的编辑智能体
def improved_editor_agent(state: ImprovedReportState) -> ImprovedReportState:
    """改进的编辑智能体：基于大纲整合并优化报告"""
    print("✏️ 编辑智能体开始整合报告...")

    completed_sections = state["completed_sections"]
    outline = state["comprehensive_outline"]
    topic = state["topic"]

    # 创建报告头部
    report_header = f"""# {outline.report_title}

## 执行摘要

{outline.executive_summary}

> **报告信息**
> - 生成时间：{time.strftime('%Y-%m-%d %H:%M:%S')}
> - 主要主题：{', '.join(outline.key_themes)}
> - 章节数量：{len(completed_sections)}
> - 生成方式：多智能体协作（基于统一大纲）

---

"""

    # 整合所有章节
    full_content = "\n\n---\n\n".join(completed_sections)

    # 添加术语表和结尾
    terminology_section = "\n\n---\n\n## 术语表\n\n"
    for term, definition in outline.terminology.items():
        terminology_section += f"**{term}**: {definition}\n\n"

    report_footer = "\n\n---\n\n## 报告说明\n\n本报告由LangGraph多智能体系统协作生成，基于统一详细大纲，确保各章节内容连贯一致。包含大纲规划、研究分析、技术阐述、案例评估、策略建议等多个专业智能体的协作贡献。"

    final_report = report_header + full_content + terminology_section + report_footer

    print(f"📄 报告整合完成，总字数约：{len(final_report)}字")
    print(f"📋 基于统一大纲生成，确保内容连贯性")

    return {"final_report": final_report}


# 智能体路由函数
def route_to_improved_agent(state: ImprovedReportState):
    """根据章节类型路由到对应的改进智能体"""
    sends = []
    outline = state["comprehensive_outline"]
    shared_context = state["shared_context"]

    for section in outline.sections:
        agent_type = section.agent_type

        # 根据智能体类型路由到对应的节点
        worker_state = {
            "section": section,
            "topic": state["topic"],
            "comprehensive_outline": outline,
            "shared_context": shared_context
        }

        if agent_type == "research_agent":
            sends.append(Send("improved_research_agent", worker_state))
        elif agent_type == "technical_agent":
            sends.append(Send("improved_technical_agent", worker_state))
        elif agent_type == "analysis_agent":
            sends.append(Send("improved_analysis_agent", worker_state))
        elif agent_type == "strategy_agent":
            sends.append(Send("improved_strategy_agent", worker_state))
        else:
            # 默认使用研究智能体
            sends.append(Send("improved_research_agent", worker_state))

    return sends


# 构建改进的多智能体工作流
print("🏗️ 构建改进的多智能体报告写作系统...")

improved_workflow_builder = StateGraph(ImprovedReportState)

# 添加所有智能体节点
improved_workflow_builder.add_node("outline_planning_agent", outline_planning_agent)
improved_workflow_builder.add_node("improved_research_agent", improved_research_agent)
improved_workflow_builder.add_node("improved_technical_agent", improved_technical_agent)
improved_workflow_builder.add_node("improved_analysis_agent", improved_analysis_agent)
improved_workflow_builder.add_node("improved_strategy_agent", improved_strategy_agent)
improved_workflow_builder.add_node("improved_editor_agent", improved_editor_agent)

# 设置工作流连接
improved_workflow_builder.add_edge(START, "outline_planning_agent")
improved_workflow_builder.add_conditional_edges(
    "outline_planning_agent",
    route_to_improved_agent,
    ["improved_research_agent", "improved_technical_agent", "improved_analysis_agent", "improved_strategy_agent"]
)

# 所有写作智能体完成后都流向编辑智能体
improved_workflow_builder.add_edge("improved_research_agent", "improved_editor_agent")
improved_workflow_builder.add_edge("improved_technical_agent", "improved_editor_agent")
improved_workflow_builder.add_edge("improved_analysis_agent", "improved_editor_agent")
improved_workflow_builder.add_edge("improved_strategy_agent", "improved_editor_agent")
improved_workflow_builder.add_edge("improved_editor_agent", END)

# 编译改进的工作流
improved_multiagent_system = improved_workflow_builder.compile()
# 生成工作流图
try:
    png_data = improved_multiagent_system.get_graph().draw_mermaid_png()
    with open("improved_multiagent_system.png", "wb") as f:
        f.write(png_data)
    print("📊 工作流图已保存为 improved_multiagent_system.png")
except Exception as e:
    print(f"⚠️ 无法生成工作流图：{e}")


# 改进的同步执行函数
def run_improved_multiagent_report(topic: str):
    """运行改进的多智能体报告生成系统"""
    print("=" * 80)
    print(f"🚀 启动改进的多智能体报告写作系统")
    print(f"📋 报告主题：{topic}")
    print(f"✨ 新特性：统一大纲指导 + 内容协调")
    print("=" * 80)

    start_time = time.time()

    # 初始状态
    initial_state = {
        "topic": topic,
        "comprehensive_outline": None,
        "completed_sections": [],
        "final_report": "",
        "shared_context": {},
        "metadata": {}
    }

    try:
        # 执行改进的工作流
        result = improved_multiagent_system.invoke(initial_state)

        end_time = time.time()
        duration = end_time - start_time

        print("\n" + "=" * 80)
        print("✅ 改进的多智能体报告生成完成！")
        print(f"⏱️ 总耗时：{duration:.2f} 秒")
        print(f"📄 报告长度：{len(result['final_report'])} 字符")
        print("=" * 80)

        # 保存报告到文件
        filename = f"improved_multiagent_report_{int(time.time())}.md"
        with open(filename, "w", encoding="utf-8") as f:
            f.write(result["final_report"])

        print(f"💾 改进报告已保存到文件：{filename}")
        print("\n🎯 改进效果：")
        print("• 📋 统一详细大纲指导各章节写作")
        print("• 🔗 章节间逻辑关联更加紧密")
        print("• 📚 统一术语定义确保一致性")
        print("• 🎨 统一写作风格提升可读性")
        print("• 🧩 内容互补避免重复")

        return result

    except Exception as e:
        print(f"❌ 报告生成失败：{e}")
        return None


# 运行改进的演示
if __name__ == "__main__":
    topic = "人工智能在医疗健康领域的应用与发展前景"
    result = run_improved_multiagent_report(topic)