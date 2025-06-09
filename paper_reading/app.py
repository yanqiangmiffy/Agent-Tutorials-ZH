import streamlit as st
import os
import asyncio
from dotenv import load_dotenv
from arxiv_search_download import ArxivSearcher
from pdf_parser import PDFParser
from paper_reading import analyze_paper_streaming, combine_analysis
import openai
from pathlib import Path
import tempfile
from datetime import datetime

# 加载环境变量
load_dotenv()

class OpenAIModelManager:
    """OpenAI模型管理器"""
    
    def __init__(self, api_key: str, base_url: str = None, model: str = "deepseek-chat"):
        self.client = openai.AsyncOpenAI(
            api_key=api_key,
            base_url=base_url
        )
        self.model = model
    
    async def generate_content(self, messages: list, temperature: float = 0.7, max_tokens: int = 2000):
        """异步生成内容"""
        try:
            response = await self.client.chat.completions.create(
                model=self.model,
                messages=messages,
                temperature=temperature,
                max_tokens=max_tokens
            )
            return response.choices[0].message.content
        except Exception as e:
            return f"生成内容时出错: {str(e)}"

def init_session_state():
    """初始化session state"""
    if 'search_results' not in st.session_state:
        st.session_state.search_results = []
    if 'selected_paper' not in st.session_state:
        st.session_state.selected_paper = None
    if 'analysis_results' not in st.session_state:
        st.session_state.analysis_results = {}
    if 'current_step' not in st.session_state:
        st.session_state.current_step = 'search'

def main():
    st.set_page_config(
        page_title="论文解读助手",
        page_icon="📚",
        layout="wide"
    )
    
    st.title("📚 论文解读助手")
    st.markdown("自动化论文搜索、下载、解析和解读流程")
    
    # 初始化session state
    init_session_state()
    
    # 侧边栏配置
    with st.sidebar:
        st.header("⚙️ 配置")
        
        # API配置
        st.subheader("API 配置")
        openai_api_key = st.text_input(
            "DeepSeek API Key",
            value=os.getenv("DEEPSEEK_API_KEY", ""),
            type="password",
            help="用于论文解读的OpenAI API密钥"
        )
        
        openai_base_url = st.text_input(
            "DeepSeek Base URL (可选)",
            value=os.getenv("DEEPSEEK_BASE_URL", ""),
            help="自定义OpenAI API基础URL"
        )
        
        openai_model = st.selectbox(
            "选择模型",
            ["deepseek-chat"],
            index=0
        )
        
        doc2x_api_key = st.text_input(
            "Doc2X API Key",
            value=os.getenv("DOC2X_API_KEY", ""),
            type="password",
            help="用于PDF解析的Doc2X API密钥"
        )
        
        # 搜索配置
        st.subheader("搜索配置")
        max_results = st.slider("最大搜索结果数", 1, 50, 10)
        days_back = st.selectbox(
            "搜索时间范围",
            [None, 7, 14, 30],
            format_func=lambda x: "不限制" if x is None else f"最近{x}天"
        )
    
    # 检查必要的API密钥
    if not openai_api_key:
        st.error("请在侧边栏配置OpenAI API Key")
        return
    
    if not doc2x_api_key:
        st.error("请在侧边栏配置Doc2X API Key")
        return
    
    # 初始化工具
    try:
        searcher = ArxivSearcher()
        parser = PDFParser(api_key=doc2x_api_key)
        model_manager = OpenAIModelManager(
            api_key=openai_api_key,
            base_url=openai_base_url if openai_base_url else None,
            model=openai_model
        )
    except Exception as e:
        st.error(f"初始化工具时出错: {e}")
        return
    
    # 主界面
    tab1, tab2, tab3, tab4 = st.tabs(["🔍 搜索论文", "📥 下载解析", "🤖 智能解读", "📄 查看结果"])
    
    with tab1:
        st.header("搜索Arxiv论文")
        
        col1, col2 = st.columns([3, 1])
        with col1:
            query = st.text_input(
                "搜索关键词",
                placeholder="例如: Retrieval Augmented Generation",
                help="输入论文相关的关键词进行搜索"
            )
        
        with col2:
            search_button = st.button("🔍 搜索", type="primary")
        
        if search_button and query:
            with st.spinner("正在搜索论文..."):
                try:
                    results = searcher.search_papers(
                        query=query,
                        max_results=max_results,
                        days_back=days_back
                    )
                    st.session_state.search_results = results
                    st.success(f"找到 {len(results)} 篇相关论文")
                except Exception as e:
                    st.error(f"搜索失败: {e}")
        
        # 显示搜索结果
        if st.session_state.search_results:
            st.subheader("搜索结果")
            
            for i, paper in enumerate(st.session_state.search_results):
                with st.expander(f"📄 {paper.title[:100]}...", expanded=False):
                    col1, col2 = st.columns([3, 1])
                    
                    with col1:
                        st.write(f"**作者:** {', '.join([author.name for author in paper.authors[:3]])}")
                        st.write(f"**发布日期:** {paper.published.strftime('%Y-%m-%d')}")
                        st.write(f"**摘要:** {paper.summary[:300]}...")
                        st.write(f"**论文ID:** {paper.entry_id.split('/')[-1]}")
                    
                    with col2:
                        if st.button(f"选择此论文", key=f"select_{i}"):
                            st.session_state.selected_paper = paper
                            st.session_state.current_step = 'download'
                            st.success("论文已选择！")
                            st.rerun()
    
    with tab2:
        st.header("下载和解析论文")
        
        if st.session_state.selected_paper:
            paper = st.session_state.selected_paper
            st.info(f"已选择论文: {paper.title}")
            
            col1, col2 = st.columns(2)
            
            with col1:
                if st.button("📥 下载PDF", type="primary"):
                    with st.spinner("正在下载PDF..."):
                        try:
                            paper_id = paper.entry_id.split('/')[-1]
                            download_dir = "./downloads"
                            Path(download_dir).mkdir(exist_ok=True)
                            
                            pdf_path = searcher.download_paper(
                                paper_id=paper_id,
                                download_dir=download_dir,
                                filename=f"{paper_id}.pdf"
                            )
                            
                            st.session_state.pdf_path = pdf_path
                            st.success(f"PDF已下载: {pdf_path}")
                            
                        except Exception as e:
                            st.error(f"下载失败: {e}")
            
            with col2:
                if hasattr(st.session_state, 'pdf_path') and st.button("🔄 解析PDF"):
                    with st.spinner("正在解析PDF为Markdown..."):
                        try:
                            success, failed, flag, extract_dir = parser.parse_pdf_to_markdown_with_auto_extract(
                                pdf_path=st.session_state.pdf_path,
                                output_path="./parsed_output",
                                output_format="md",
                                auto_extract=True,
                                keep_zip=False
                            )
                            
                            if not flag:
                                # 查找生成的markdown文件
                                if extract_dir:
                                    md_files = list(Path(extract_dir).glob("*.md"))
                                    if md_files:
                                        st.session_state.markdown_path = str(md_files[0])
                                        with open(md_files[0], 'r', encoding='utf-8') as f:
                                            st.session_state.markdown_content = f.read()
                                        st.success("PDF解析完成！")
                                    else:
                                        st.error("未找到生成的Markdown文件")
                                else:
                                    st.error("解析失败，未生成输出目录")
                            else:
                                st.error(f"解析失败: {failed}")
                                
                        except Exception as e:
                            st.error(f"解析失败: {e}")
        else:
            st.warning("请先在搜索页面选择一篇论文")
    
    with tab3:
        st.header("智能论文解读")
        
        if hasattr(st.session_state, 'markdown_content'):
            st.info("论文内容已准备就绪，可以开始解读")
            
            if st.button("🤖 开始智能解读", type="primary"):
                # 创建进度显示区域
                progress_bar = st.progress(0)
                status_text = st.empty()
                results_container = st.container()
                
                def progress_callback(section, result, current, total):
                    progress = current / total
                    progress_bar.progress(progress)
                    status_text.text(f"正在生成: {section} ({current}/{total})")
                    
                    # 实时显示结果
                    with results_container:
                        st.subheader(section)
                        st.write(result)
                        st.divider()
                
                try:
                    # 开始分析
                    analysis = analyze_paper_streaming(
                        st.session_state.markdown_content,
                        model_manager,
                        progress_callback
                    )
                    
                    st.session_state.analysis_results = analysis
                    progress_bar.progress(1.0)
                    status_text.text("解读完成！")
                    st.success("论文解读已完成！")
                    
                except Exception as e:
                    st.error(f"解读失败: {e}")
        else:
            st.warning("请先完成PDF下载和解析")
    
    with tab4:
        st.header("查看解读结果")
        
        if st.session_state.analysis_results:
            # 生成完整报告
            full_report = combine_analysis(st.session_state.analysis_results)
            
            # 显示报告
            st.markdown(full_report)
            
            # 下载按钮
            col1, col2 = st.columns(2)
            
            with col1:
                st.download_button(
                    label="📄 下载Markdown报告",
                    data=full_report,
                    file_name=f"paper_analysis_{datetime.now().strftime('%Y%m%d_%H%M%S')}.md",
                    mime="text/markdown"
                )
            
            with col2:
                if st.button("🔄 重新解读"):
                    st.session_state.analysis_results = {}
                    st.rerun()
        else:
            st.warning("暂无解读结果，请先完成论文解读")

if __name__ == "__main__":
    main()