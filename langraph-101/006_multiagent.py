import asyncio
import time
from typing import Annotated, List, TypedDict
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

# 报告章节结构定义
class ReportSection(BaseModel):
    title: str = Field(description="章节标题")
    description: str = Field(description="章节内容描述和要求")
    agent_type: str = Field(description="负责此章节的智能体类型")

class ReportPlan(BaseModel):
    sections: List[ReportSection] = Field(description="报告的各个章节")

# 使用结构化输出的规划器
planner = llm.with_structured_output(ReportPlan)

# 主状态定义
class ReportState(TypedDict):
    topic: str  # 报告主题
    sections: List[ReportSection]  # 报告章节列表
    completed_sections: Annotated[List[str], operator.add]  # 已完成的章节内容
    final_report: str  # 最终合并的报告
    metadata: dict  # 元数据信息

# 工作者状态定义
class WorkerState(TypedDict):
    section: ReportSection
    topic: str
    completed_sections: Annotated[List[str], operator.add]

# 1. 规划智能体 - 负责制定报告结构
def planning_agent(state: ReportState) -> ReportState:
    """规划智能体：分析主题并制定报告结构"""
    print("🎯 规划智能体开始工作...")
    
    planning_prompt = f"""
    请为主题「{state['topic']}」制定一个详细的报告结构计划。
    
    要求：
    1. 报告应包含4个主要章节
    2. 每个章节应有明确的标题和详细描述
    3. 为每个章节指定最适合的智能体类型：
       - research_agent: 负责研究背景、现状分析等
       - technical_agent: 负责技术原理、方法论等
       - analysis_agent: 负责数据分析、案例研究等
       - summary_agent: 负责总结、展望、建议等
    
    请确保章节之间逻辑连贯，内容互补。
    """
    
    report_plan = planner.invoke([
        SystemMessage(content=planning_prompt),
        HumanMessage(content=f"报告主题：{state['topic']}")
    ])
    
    print(f"📋 规划完成，共{len(report_plan.sections)}个章节：")
    for i, section in enumerate(report_plan.sections, 1):
        print(f"  {i}. {section.title} ({section.agent_type})")
    
    return {
        "sections": report_plan.sections,
        "metadata": {"planning_time": time.time(), "total_sections": len(report_plan.sections)}
    }

# 2. 研究智能体 - 负责背景研究和现状分析
def research_agent(state: WorkerState) -> WorkerState:
    """研究智能体：专门负责背景研究和现状分析"""
    section = state["section"]
    topic = state["topic"]
    
    print(f"🔍 研究智能体正在撰写：{section.title}")
    
    research_prompt = f"""
    你是一个专业的研究智能体，擅长背景调研和现状分析。
    
    任务：为报告「{topic}」撰写章节「{section.title}」
    
    章节要求：{section.description}
    
    写作要求：
    1. 深入研究相关背景和历史发展
    2. 分析当前现状和主要趋势
    3. 引用相关数据和事实
    4. 使用专业术语，保持学术严谨性
    5. 使用Markdown格式，包含适当的标题层级
    6. 字数控制在800-1200字
    
    请开始撰写：
    """
    
    response = llm.invoke([
        SystemMessage(content=research_prompt),
        HumanMessage(content=f"开始撰写章节：{section.title}")
    ])
    
    return {"completed_sections": [f"## {section.title}\n\n{response.content}"]}

# 3. 技术智能体 - 负责技术原理和方法论
def technical_agent(state: WorkerState) -> WorkerState:
    """技术智能体：专门负责技术原理和方法论"""
    section = state["section"]
    topic = state["topic"]
    
    print(f"⚙️ 技术智能体正在撰写：{section.title}")
    
    technical_prompt = f"""
    你是一个专业的技术智能体，擅长技术原理分析和方法论阐述。
    
    任务：为报告「{topic}」撰写章节「{section.title}」
    
    章节要求：{section.description}
    
    写作要求：
    1. 详细阐述相关技术原理和核心概念
    2. 介绍主要方法论和实现途径
    3. 分析技术优势和局限性
    4. 提供具体的技术示例或案例
    5. 使用Markdown格式，包含代码块或图表说明
    6. 字数控制在800-1200字
    
    请开始撰写：
    """
    
    response = llm.invoke([
        SystemMessage(content=technical_prompt),
        HumanMessage(content=f"开始撰写章节：{section.title}")
    ])
    
    return {"completed_sections": [f"## {section.title}\n\n{response.content}"]}

# 4. 分析智能体 - 负责数据分析和案例研究
def analysis_agent(state: WorkerState) -> WorkerState:
    """分析智能体：专门负责数据分析和案例研究"""
    section = state["section"]
    topic = state["topic"]
    
    print(f"📊 分析智能体正在撰写：{section.title}")
    
    analysis_prompt = f"""
    你是一个专业的分析智能体，擅长数据分析和案例研究。
    
    任务：为报告「{topic}」撰写章节「{section.title}」
    
    章节要求：{section.description}
    
    写作要求：
    1. 进行深入的数据分析和统计研究
    2. 提供具体的案例研究和实例分析
    3. 使用图表、数据来支撑观点
    4. 对比分析不同方案或方法的效果
    5. 使用Markdown格式，包含表格和数据展示
    6. 字数控制在800-1200字
    
    请开始撰写：
    """
    
    response = llm.invoke([
        SystemMessage(content=analysis_prompt),
        HumanMessage(content=f"开始撰写章节：{section.title}")
    ])
    
    return {"completed_sections": [f"## {section.title}\n\n{response.content}"]}

# 5. 总结智能体 - 负责总结和展望
def summary_agent(state: WorkerState) -> WorkerState:
    """总结智能体：专门负责总结、展望和建议"""
    section = state["section"]
    topic = state["topic"]
    
    print(f"📝 总结智能体正在撰写：{section.title}")
    
    summary_prompt = f"""
    你是一个专业的总结智能体，擅长归纳总结和前瞻分析。
    
    任务：为报告「{topic}」撰写章节「{section.title}」
    
    章节要求：{section.description}
    
    写作要求：
    1. 总结前面章节的核心观点和发现
    2. 提出未来发展趋势和展望
    3. 给出具体的建议和行动方案
    4. 分析潜在的挑战和机遇
    5. 使用Markdown格式，条理清晰
    6. 字数控制在600-1000字
    
    请开始撰写：
    """
    
    response = llm.invoke([
        SystemMessage(content=summary_prompt),
        HumanMessage(content=f"开始撰写章节：{section.title}")
    ])
    
    return {"completed_sections": [f"## {section.title}\n\n{response.content}"]}

# 6. 编辑智能体 - 负责最终整合和编辑
def editor_agent(state: ReportState) -> ReportState:
    """编辑智能体：整合所有章节并进行最终编辑"""
    print("✏️ 编辑智能体开始整合报告...")
    
    completed_sections = state["completed_sections"]
    topic = state["topic"]
    
    # 创建报告头部
    report_header = f"""# {topic}

> 本报告由多智能体协作完成
> 生成时间：{time.strftime('%Y-%m-%d %H:%M:%S')}
> 总章节数：{len(completed_sections)}

---

"""
    
    # 整合所有章节
    full_content = "\n\n---\n\n".join(completed_sections)
    
    # 添加结尾
    report_footer = "\n\n---\n\n## 报告说明\n\n本报告由LangGraph多智能体系统协作生成，包含规划、研究、技术、分析、总结等多个专业智能体的贡献。"
    
    final_report = report_header + full_content + report_footer
    
    print(f"📄 报告整合完成，总字数约：{len(final_report)}字")
    
    return {"final_report": final_report}

# 智能体路由函数
def route_to_agent(state: ReportState):
    """根据章节类型路由到对应的智能体"""
    sends = []
    for section in state["sections"]:
        agent_type = section.agent_type
        
        # 根据智能体类型路由到对应的节点
        if agent_type == "research_agent":
            sends.append(Send("research_agent", {"section": section, "topic": state["topic"]}))
        elif agent_type == "technical_agent":
            sends.append(Send("technical_agent", {"section": section, "topic": state["topic"]}))
        elif agent_type == "analysis_agent":
            sends.append(Send("analysis_agent", {"section": section, "topic": state["topic"]}))
        elif agent_type == "summary_agent":
            sends.append(Send("summary_agent", {"section": section, "topic": state["topic"]}))
        else:
            # 默认使用研究智能体
            sends.append(Send("research_agent", {"section": section, "topic": state["topic"]}))
    
    return sends

# 构建多智能体工作流
print("🏗️ 构建多智能体报告写作系统...")

workflow_builder = StateGraph(ReportState)

# 添加所有智能体节点
workflow_builder.add_node("planning_agent", planning_agent)
workflow_builder.add_node("research_agent", research_agent)
workflow_builder.add_node("technical_agent", technical_agent)
workflow_builder.add_node("analysis_agent", analysis_agent)
workflow_builder.add_node("summary_agent", summary_agent)
workflow_builder.add_node("editor_agent", editor_agent)

# 设置工作流连接
workflow_builder.add_edge(START, "planning_agent")
workflow_builder.add_conditional_edges(
    "planning_agent", 
    route_to_agent, 
    ["research_agent", "technical_agent", "analysis_agent", "summary_agent"]
)

# 所有写作智能体完成后都流向编辑智能体
workflow_builder.add_edge("research_agent", "editor_agent")
workflow_builder.add_edge("technical_agent", "editor_agent")
workflow_builder.add_edge("analysis_agent", "editor_agent")
workflow_builder.add_edge("summary_agent", "editor_agent")
workflow_builder.add_edge("editor_agent", END)

# 编译工作流
multiagent_report_system = workflow_builder.compile()

# 生成工作流图
try:
    png_data = multiagent_report_system.get_graph().draw_mermaid_png()
    with open("multiagent_report_workflow.png", "wb") as f:
        f.write(png_data)
    print("📊 工作流图已保存为 multiagent_report_workflow.png")
except Exception as e:
    print(f"⚠️ 无法生成工作流图：{e}")

# 异步执行函数
async def run_multiagent_report(topic: str):
    """异步运行多智能体报告生成"""
    print("="*80)
    print(f"🚀 启动多智能体报告写作系统")
    print(f"📋 报告主题：{topic}")
    print("="*80)
    
    start_time = time.time()
    
    # 初始状态
    initial_state = {
        "topic": topic,
        "sections": [],
        "completed_sections": [],
        "final_report": "",
        "metadata": {}
    }
    
    try:
        # 执行工作流
        result = await multiagent_report_system.ainvoke(initial_state)
        
        end_time = time.time()
        duration = end_time - start_time
        
        print("\n" + "="*80)
        print("✅ 多智能体报告生成完成！")
        print(f"⏱️ 总耗时：{duration:.2f} 秒")
        print(f"📄 报告长度：{len(result['final_report'])} 字符")
        print("="*80)
        
        # 保存报告到文件
        with open(f"multiagent_report_{int(time.time())}.md", "w", encoding="utf-8") as f:
            f.write(result["final_report"])
        
        print(f"💾 报告已保存到文件")
        
        return result
        
    except Exception as e:
        print(f"❌ 报告生成失败：{e}")
        return None

# 同步执行函数
def run_multiagent_report_sync(topic: str):
    """同步运行多智能体报告生成"""
    print("="*80)
    print(f"🚀 启动多智能体报告写作系统（同步模式）")
    print(f"📋 报告主题：{topic}")
    print("="*80)
    
    start_time = time.time()
    
    # 初始状态
    initial_state = {
        "topic": topic,
        "sections": [],
        "completed_sections": [],
        "final_report": "",
        "metadata": {}
    }
    
    try:
        # 执行工作流
        result = multiagent_report_system.invoke(initial_state)
        
        end_time = time.time()
        duration = end_time - start_time
        
        print("\n" + "="*80)
        print("✅ 多智能体报告生成完成！")
        print(f"⏱️ 总耗时：{duration:.2f} 秒")
        print(f"📄 报告长度：{len(result['final_report'])} 字符")
        print("="*80)
        
        # 保存报告到文件
        filename = f"multiagent_report_{int(time.time())}.md"
        with open(filename, "w", encoding="utf-8") as f:
            f.write(result["final_report"])
        
        print(f"💾 报告已保存到文件：{filename}")
        print("\n📖 报告预览：")
        print("-"*50)
        print(result["final_report"][:500] + "..." if len(result["final_report"]) > 500 else result["final_report"])
        
        return result
        
    except Exception as e:
        print(f"❌ 报告生成失败：{e}")
        return None

# 主函数
async def main():
    """主函数：演示多智能体报告写作"""
    
    # 测试主题列表
    test_topics = [
        "人工智能在医疗健康领域的应用与发展前景",
        "区块链技术在供应链管理中的创新应用",
        "5G技术对智慧城市建设的推动作用",
        "大数据分析在金融风控中的实践与挑战"
    ]
    
    # 选择一个主题进行演示
    selected_topic = test_topics[0]
    
    print("🎯 可选报告主题：")
    for i, topic in enumerate(test_topics, 1):
        print(f"  {i}. {topic}")
    print(f"\n📝 当前选择：{selected_topic}")
    print()
    
    # 运行多智能体报告生成
    result = await run_multiagent_report(selected_topic)
    
    if result:
        print("\n🎉 多智能体协作报告写作演示完成！")
        print("\n💡 系统特点：")
        print("• 🎯 规划智能体：制定报告结构和章节分工")
        print("• 🔍 研究智能体：负责背景研究和现状分析")
        print("• ⚙️ 技术智能体：阐述技术原理和方法论")
        print("• 📊 分析智能体：进行数据分析和案例研究")
        print("• 📝 总结智能体：提供总结展望和建议")
        print("• ✏️ 编辑智能体：整合内容并最终编辑")
        print("\n🚀 优势：专业分工、并行处理、质量保证")

# 运行演示
if __name__ == "__main__":
    # 异步运行（推荐）
    # asyncio.run(main())
    
    # 同步运行（简单演示）
    topic = "人工智能在医疗健康领域的应用与发展前景"
    result = run_multiagent_report_sync(topic)