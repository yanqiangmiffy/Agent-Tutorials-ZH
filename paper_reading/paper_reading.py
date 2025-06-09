import os
import json
import asyncio
import pandas as pd
import streamlit as st

def read_markdown_file(file_path):
    """读取markdown文件内容"""
    try:
        with open(file_path, 'r', encoding='utf-8') as file:
            return file.read()
    except Exception as e:
        return f"读取文件时出错: {e}"


def scan_papers_directory(papers_dir):
    """扫描指定目录下的所有论文解析文件"""
    papers = []
    try:
        for item in os.listdir(papers_dir):
            item_path = os.path.join(papers_dir, item)
            if os.path.isdir(item_path):
                md_file = os.path.join(item_path, f"{item}.md")
                if os.path.exists(md_file):
                    papers.append({
                        "id": item,
                        "path": md_file,
                        "dir_path": item_path,
                        "source_type": "file"
                    })
    except Exception as e:
        print(f"扫描目录时出错: {e}")

    return papers


def load_papers_from_dataframe(df_path):
    """从parquet文件加载论文数据"""
    try:
        papers_df = pd.read_parquet(df_path)
        papers = []
        
        for idx, row in papers_df.iterrows():
            # 确保必要的列存在
            if 'entry_id' in papers_df.columns and 'title' in papers_df.columns and 'content' in papers_df.columns:
                papers.append({
                    "id": row['entry_id'],
                    "title": row['title'],
                    "content": row['content'],
                    "source_type": "dataframe",
                    "df_index": idx,
                    # 可选添加其他有用的元数据
                    "metadata": {
                        "authors": row.get('authors', ''),
                        "published": row.get('published', ''),
                        "primary_category": row.get('primary_category', ''),
                        "summary": row.get('summary', '')
                    }
                })
            else:
                print("DataFrame缺少必要的列：entry_id, title, content")
                break
                
        return papers
    except Exception as e:
        print(f"加载DataFrame时出错: {e}")
        return []


def get_paper_metadata(paper_dir):
    """尝试获取论文的元数据"""
    try:
        content_list_path = os.path.join(paper_dir, "content_list.json")
        if os.path.exists(content_list_path):
            with open(content_list_path, 'r', encoding='utf-8') as f:
                data = json.load(f)
                if isinstance(data, list) and len(data) > 0:
                    # 假设文章标题在content_list.json的第一个元素中
                    title = data[0].get("title", "未知标题")
                    return {"title": title}
    except Exception as e:
        pass

    return {"title": "未知标题"}


async def generate_analysis_section_async(markdown_content, prompt, model_manager):
    """使用异步方式通过OpenAI模型生成论文分析的各个部分"""
    try:
        # 准备消息内容
        messages = [
            {"role": "system", "content": "你是一个专业的论文分析助手，擅长解读学术论文并提取关键信息，请用中文生成。"},
            {"role": "user", "content": f"{prompt}\n\n论文内容：{markdown_content[:50000]}"}  # 限制长度以适应API限制
        ]

        return await model_manager.generate_content(messages)
    except Exception as e:
        return f"生成分析时出错: {str(e)}"


def generate_analysis_section(markdown_content, prompt, model_manager):
    """同步包装器，用于调用异步生成函数"""
    loop = asyncio.new_event_loop()
    asyncio.set_event_loop(loop)
    result = loop.run_until_complete(generate_analysis_section_async(markdown_content, prompt, model_manager))
    loop.close()
    return result


def analyze_paper_streaming(markdown_content, model_manager, progress_callback=None):
    """分析论文内容，生成8个部分的分析，支持流式返回结果"""
    prompts = {
        "研究动机与问题定义": """
    请分析论文的研究动机和所要解决的核心问题。具体需要：
    1. 概括论文研究的背景和领域，使用简洁明了的语言介绍研究方向
    2. 明确指出论文试图解决的关键问题或挑战
    3. 解释为什么这个问题重要，以及解决此问题的潜在价值和影响
    4. 分析作者对问题的独特视角或切入点
    5. 总结研究动机与大的技术或社会趋势的关联

    回答应当以"这篇论文研究的是..."或类似表述开头，确保与后续章节的分析有良好的连贯性。保持语言通俗易懂，即使是非专业人士也能理解研究的价值和意义。
            """,
        "相关工作综述与现状评估": """
    请分析论文中的相关工作综述部分，评估该研究领域的现状。具体需要：
    1. 总结论文中提到的主要相关工作和研究方向(可用表格整理)
    2. 分析当前领域研究的主流方法和技术路线(可用表格整理)
    3. 指出现有方法的局限性和不足，尤其是论文作者特别强调的问题
    4. 解释论文的工作如何定位在这些相关研究之中
    5. 评估论文对前人工作的借鉴和改进点

    回答应以"在探讨本文的创新点之前，我们需要了解该领域的研究现状..."或类似表述开头，确保与前后章节自然衔接。保持客观分析的语气，突出领域发展的脉络。
            """,
        "研究创新与思路来源": """
    请分析论文的创新点和核心思路来源。具体需要：
    1. 明确指出论文的主要创新点（技术、方法、视角等方面）
    2. 分析这些创新的灵感或思路来源
    3. 解释为什么这些创新对解决前述问题是有效的
    4. 评估创新点的原创性和科学价值
    5. 探讨论文思路与其他领域或方法的跨界融合（如果有）

    回答应以"基于前文对研究背景和现状的分析，本论文的创新主要体现在..."或类似表述开头，确保与前述内容自然衔接。使用生动的语言突出作者的独特贡献，让读者理解这一研究的"闪光点"。
            """,
        "提出的解决方案与方法细节": """
    请详细分析论文提出的解决方案和方法细节。具体需要：
    1. 清晰描述论文提出的整体框架或方法论
    2. 分解该方法的关键组成部分和技术要点
    3. 解释各组成部分如何协同工作以解决目标问题
    4. 分析方法中的创新设计及其功能意义
    5. 阐述方法的理论基础和核心机制

    回答应以"为了解决上述问题，论文提出了..."或类似表述开头，注重与前面章节的衔接。使用图文结合的方式(可以引用论文中的关键图表)，清晰地解释复杂概念。用浅显易懂的语言解释技术细节，同时保持专业准确性，作为全文的核心部分可以适当展开。
            """,
        "实验设计与验证方法": """
    请分析论文的实验设计和验证方法。具体需要：
    1. 概述论文的实验设置，包括数据集、基准测试和评估指标(可用表格整理)
    2. 分析实验的设计思路和合理性
    3. 总结实验的对比方法和消融实验设计
    4. 评估实验是否充分验证了方法的有效性
    5. 指出实验中的创新点或特别之处

    回答应以"为验证所提方法的有效性，论文设计了一系列实验..."或类似表述开头，与方法部分自然衔接。使用客观准确的语言描述实验过程，避免主观评价。重点突出实验设计如何针对性地验证方法的优势。
            """,
        "研究结果与关键结论": """
    请分析论文的研究结果和关键结论。具体需要：
    1. 总结论文的主要实验结果和量化指标(可用表格整理)
    2. 分析结果对比，突出提出方法的优势
    3. 解释这些结果如何支持论文的论点
    4. 总结作者从结果中得出的关键结论
    5. 评估结论的可靠性和普适性

    回答应以"通过上述实验设计，论文得到了以下关键结果..."或类似表述开头，与前文实验部分紧密衔接。使用数据支持论点，同时解释数据背后的意义。平衡呈现方法的优势和可能的局限性，保持分析的客观性。内容篇幅约350-450字。
            """,
        "未来工作展望与开放问题": """
    请分析论文中提出的未来工作方向和尚未解决的开放问题。具体需要：
    1. 总结论文明确提出的后续研究方向
    2. 分析论文方法的局限性和可能的改进空间
    3. 探讨该研究可能的扩展应用领域
    4. 结合领域发展趋势，评估哪些方向最有研究价值
    5. 提出你认为有价值但论文未提及的研究方向

    回答应以"尽管取得了上述成果，本研究仍有进一步探索和改进的空间..."或类似表述开头，自然承接结果部分。使用前瞻性的语言，既要客观反映论文提出的未来方向，也可加入自己的见解和思考。
            """,
        "核心算法伪代码（Python风格）": """
    请根据论文中描述的核心算法或方法，生成Python风格的伪代码。具体需要：
    1. 将论文中最核心的算法或方法流程转换为清晰的伪代码
    2. 使用Python语法和常见库（如NumPy、PyTorch等）的函数命名风格
    3. 适当添加注释解释关键步骤和参数
    4. 确保伪代码逻辑完整，能够反映算法的本质
    5. 简化复杂实现细节，保留算法的框架和核心思想

    回答应以"最后，我们将论文的核心算法用Python风格的伪代码表示如下..."或类似表述开头，作为全文的收尾部分。伪代码应简洁易懂，突出算法的关键步骤和创新点，帮助读者把握方法的实现思路。添加必要的前置说明和后续解释，使代码部分与文章其他部分融为一体。
            """
    }
    
    analysis = {}
    for i, (section, prompt) in enumerate(prompts.items()):
        with st.spinner(f"正在生成 {section} 部分..."):
            result = generate_analysis_section(markdown_content, prompt, model_manager)
            analysis[section] = result
            
            # 调用回调函数，实时更新结果
            if progress_callback:
                progress_callback(section, result, i + 1, len(prompts))
    
    return analysis


def combine_analysis(analysis):
    """将所有分析部分组合成一个完整的Markdown文档"""
    combined = "# 论文解析报告\n\n"

    for section, content in analysis.items():
        combined += f"## {section}\n\n{content}\n\n"

    return combined





