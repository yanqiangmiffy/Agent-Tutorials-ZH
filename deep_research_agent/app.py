import streamlit as st
import json
import os
import time
from datetime import datetime
import pandas as pd

from agents import ReportStructureAgent, FirstSearchAgent, FirstSummaryAgent, ReflectionAgent, ReflectionSummaryAgent, \
    ReportFormattingAgent
from state import State
from utils import tavily_search, update_state_with_search_results
from language_utils import detect_language

from dotenv import load_dotenv

load_dotenv()

# 设置页面配置
st.set_page_config(
    page_title="深度研究代理",
    page_icon="🔍",
    layout="wide",
    initial_sidebar_state="expanded"
)

# 设置默认查询和参数
DEFAULT_QUERY = "请帮我调研一下RAG有哪些实用技巧以及经验总结"
NUM_REFLECTIONS = 2
NUM_RESULTS_PER_SEARCH = 3
CAP_SEARCH_LENGTH = 20000

# 初始化会话状态
if 'state' not in st.session_state:
    st.session_state.state = State()
if 'logs' not in st.session_state:
    st.session_state.logs = []
if 'current_step' not in st.session_state:
    st.session_state.current_step = None
if 'final_report' not in st.session_state:
    st.session_state.final_report = None
if 'is_running' not in st.session_state:
    st.session_state.is_running = False

# 添加日志函数
def add_log(message, level="INFO"):
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    st.session_state.logs.append({"timestamp": timestamp, "level": level, "message": message})
    
# 侧边栏
st.sidebar.title("深度研究代理")
st.sidebar.markdown("这个应用使用多个AI代理来进行深度研究，生成详细的研究报告。")

# 查询输入
query = st.sidebar.text_area("输入研究主题", value=DEFAULT_QUERY, height=100)

# 参数设置
with st.sidebar.expander("高级设置", expanded=False):
    num_reflections = st.slider("反思次数", min_value=1, max_value=5, value=NUM_REFLECTIONS)
    num_results = st.slider("每次搜索结果数", min_value=1, max_value=10, value=NUM_RESULTS_PER_SEARCH)
    cap_length = st.slider("搜索结果最大长度", min_value=5000, max_value=50000, value=CAP_SEARCH_LENGTH, step=5000)

# 运行按钮
run_button = st.sidebar.button("开始研究", type="primary", disabled=st.session_state.is_running)

# 重置按钮
if st.sidebar.button("重置", type="secondary"):
    st.session_state.state = State()
    st.session_state.logs = []
    st.session_state.current_step = None
    st.session_state.final_report = None
    st.session_state.is_running = False
    st.experimental_rerun()

# 主界面
st.title("深度研究代理")

# 创建两列布局
col1, col2 = st.columns([3, 2])

# 左侧列：研究进度和结果
with col1:
    # 进度指示器
    if st.session_state.current_step:
        st.subheader("当前进度")
        st.info(st.session_state.current_step)
    
    # 最终报告
    if st.session_state.final_report:
        st.subheader("研究报告")
        st.markdown(st.session_state.final_report)
        
        # 下载按钮
        report_filename = f"report_{datetime.now().strftime('%Y-%m-%d_%H-%M-%S')}.md"
        st.download_button(
            label="下载报告",
            data=st.session_state.final_report,
            file_name=report_filename,
            mime="text/markdown"
        )

# 右侧列：日志和中间结果
with col2:
    st.subheader("研究日志")
    
    # 显示日志
    if st.session_state.logs:
        log_df = pd.DataFrame(st.session_state.logs)
        st.dataframe(log_df, use_container_width=True, height=300)
    
    # 段落信息
    if len(st.session_state.state.paragraphs) > 0:
        st.subheader("段落信息")
        
        for i, paragraph in enumerate(st.session_state.state.paragraphs):
            with st.expander(f"段落 {i+1}: {paragraph.title}"):
                st.markdown(f"**计划内容**: {paragraph.content}")
                
                if paragraph.research.latest_summary:
                    st.markdown(f"**当前总结**: {paragraph.research.latest_summary}")
                
                if paragraph.research.search_history:
                    st.markdown("**搜索历史**:")
                    for j, search in enumerate(paragraph.research.search_history):
                        st.markdown(f"- 搜索 {j+1}: [{search.url}]({search.url})")

# 主要研究流程
def run_research(topic, num_reflections, num_results_per_search, cap_search_length):
    try:
        st.session_state.is_running = True
        
        # 检测输入语言
        st.session_state.current_step = "检测输入语言..."
        lang = detect_language(topic)
        add_log(f"检测到的语言: {lang}")
        
        # 创建报告结构
        st.session_state.current_step = "创建报告结构..."
        add_log("开始创建报告结构")
        report_structure_agent = ReportStructureAgent(topic)
        _ = report_structure_agent.mutate_state(st.session_state.state)
        add_log(f"报告结构创建完成，共 {len(st.session_state.state.paragraphs)} 个段落")
        
        # 初始化代理
        first_search_agent = FirstSearchAgent()
        first_summary_agent = FirstSummaryAgent()
        reflection_agent = ReflectionAgent()
        reflection_summary_agent = ReflectionSummaryAgent()
        report_formatting_agent = ReportFormattingAgent()
        
        # 处理每个段落
        for j in range(len(st.session_state.state.paragraphs)):
            st.session_state.current_step = f"处理段落 {j+1}/{len(st.session_state.state.paragraphs)}: {st.session_state.state.paragraphs[j].title}"
            add_log(f"开始处理段落 {j+1}: {st.session_state.state.paragraphs[j].title}", "INFO")
            
            # 第一次搜索
            add_log(f"段落 {j+1}: 执行初始搜索", "INFO")
            message = json.dumps({
                "title": st.session_state.state.paragraphs[j].title,
                "content": st.session_state.state.paragraphs[j].content
            })
            
            output = first_search_agent.run(message)
            add_log(f"段落 {j+1}: 搜索查询 - {output['search_query']}", "INFO")
            
            search_results = tavily_search(output["search_query"], max_results=num_results_per_search)
            add_log(f"段落 {j+1}: 获取到 {len(search_results['results'])} 条搜索结果", "INFO")
            
            _ = update_state_with_search_results(search_results, j, st.session_state.state)
            
            # 第一次总结
            add_log(f"段落 {j+1}: 生成初始总结", "INFO")
            message = {
                "title": st.session_state.state.paragraphs[j].title,
                "content": st.session_state.state.paragraphs[j].content,
                "search_query": search_results["query"],
                "search_results": [result["raw_content"][0:cap_search_length] for result in search_results["results"] if
                                  result["raw_content"]]
            }
            
            _ = first_summary_agent.mutate_state(message=json.dumps(message), idx_paragraph=j, state=st.session_state.state)
            add_log(f"段落 {j+1}: 初始总结完成", "INFO")
            
            # 反思循环
            for i in range(num_reflections):
                st.session_state.current_step = f"段落 {j+1}: 执行反思 {i+1}/{num_reflections}"
                add_log(f"段落 {j+1}: 执行反思 {i+1}/{num_reflections}", "INFO")
                
                message = {
                    "paragraph_latest_state": st.session_state.state.paragraphs[j].research.latest_summary,
                    "title": st.session_state.state.paragraphs[j].title,
                    "content": st.session_state.state.paragraphs[j].content
                }
                
                output = reflection_agent.run(message=json.dumps(message))
                add_log(f"段落 {j+1}, 反思 {i+1}: 搜索查询 - {output['search_query']}", "INFO")
                
                search_results = tavily_search(output["search_query"], max_results=num_results_per_search)
                add_log(f"段落 {j+1}, 反思 {i+1}: 获取到 {len(search_results['results'])} 条搜索结果", "INFO")
                
                _ = update_state_with_search_results(search_results, j, st.session_state.state)
                
                message = {
                    "title": st.session_state.state.paragraphs[j].title,
                    "content": st.session_state.state.paragraphs[j].content,
                    "search_query": search_results["query"],
                    "search_results": [result["raw_content"][0:cap_search_length] for result in search_results["results"] if
                                      result["raw_content"]],
                    "paragraph_latest_state": st.session_state.state.paragraphs[j].research.latest_summary
                }
                
                _ = reflection_summary_agent.mutate_state(message=json.dumps(message), idx_paragraph=j, state=st.session_state.state)
                add_log(f"段落 {j+1}, 反思 {i+1}: 反思总结完成", "INFO")
                
                # 更新界面
                st.experimental_rerun()
                time.sleep(0.5)
        
        # 生成最终报告
        st.session_state.current_step = "生成最终报告..."
        add_log("开始生成最终报告", "INFO")
        
        report_data = [
            {"title": paragraph.title, "paragraph_latest_state": paragraph.research.latest_summary} 
            for paragraph in st.session_state.state.paragraphs
        ]
        
        final_report = report_formatting_agent.run(json.dumps(report_data))
        st.session_state.final_report = final_report
        add_log("最终报告生成完成", "SUCCESS")
        
        # 保存报告到文件
        os.makedirs("reports", exist_ok=True)
        report_filename = f"reports/report_{datetime.now().strftime('%Y-%m-%d_%H-%M-%S')}.md"
        with open(report_filename, "w") as f:
            f.write(final_report)
        add_log(f"报告已保存到 {report_filename}", "SUCCESS")
        
        st.session_state.current_step = "研究完成！"
        
    except Exception as e:
        add_log(f"发生错误: {str(e)}", "ERROR")
        st.error(f"发生错误: {str(e)}")
    finally:
        st.session_state.is_running = False

# 当点击运行按钮时执行研究
if run_button:
    st.session_state.state = State()  # 重置状态
    st.session_state.logs = []  # 重置日志
    st.session_state.final_report = None  # 重置报告
    run_research(query, num_reflections, num_results, cap_length)