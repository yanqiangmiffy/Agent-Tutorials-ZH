import arxiv
import os
from typing import List, Optional, Generator
from pathlib import Path
from datetime import datetime, timedelta

class ArxivSearcher:
    """Arxiv论文搜索和下载工具类"""
    
    def __init__(self):
        """初始化Arxiv客户端"""
        self.client = arxiv.Client()
    
    def search_papers(self, 
                     query: str, 
                     max_results: int = 10, 
                     sort_by: arxiv.SortCriterion = arxiv.SortCriterion.Relevance,
                     days_back: Optional[int] = None) -> List[arxiv.Result]:
        """搜索论文
        
        Args:
            query: 搜索查询词
            max_results: 最大结果数量
            sort_by: 排序方式
            days_back: 搜索最近几天的论文（7或14天），None表示不限制日期
            
        Returns:
            论文结果列表
        """
        # 构建查询字符串
        search_query = query
        
        # 如果指定了日期范围，添加日期过滤
        if days_back:
            end_date = datetime.now()
            start_date = end_date - timedelta(days=days_back)
            
            # 格式化日期为YYYYMMDD格式
            start_date_str = start_date.strftime('%Y%m%d')
            end_date_str = end_date.strftime('%Y%m%d')
            
            # 添加日期范围到查询中
            search_query = f"{query} AND submittedDate:[{start_date_str} TO {end_date_str}]"
        
        search = arxiv.Search(
            query=search_query,
            max_results=max_results,
            sort_by=sort_by
        )
        
        results = list(self.client.results(search))
        return results
    
    def search_papers_by_relevance_and_date(self,
                                           query: str,
                                           max_results: int = 10,
                                           days_back: int = 7) -> List[arxiv.Result]:
        """按相关性搜索最近指定天数的论文
        
        Args:
            query: 搜索查询词
            max_results: 最大结果数量
            days_back: 搜索最近几天的论文（推荐7或14天）
            
        Returns:
            按相关性排序的论文结果列表
        """
        return self.search_papers(
            query=query,
            max_results=max_results,
            sort_by=arxiv.SortCriterion.Relevance,
            days_back=days_back
        )
    
    def search_by_id(self, paper_ids: List[str]) -> List[arxiv.Result]:
        """根据论文ID搜索
        
        Args:
            paper_ids: 论文ID列表
            
        Returns:
            论文结果列表
        """
        search = arxiv.Search(id_list=paper_ids)
        results = list(self.client.results(search))
        return results
    
    def download_paper(self, 
                      paper_id: str, 
                      download_dir: str = "./downloads", 
                      filename: Optional[str] = None) -> str:
        """下载指定论文的PDF
        
        Args:
            paper_id: 论文ID
            download_dir: 下载目录
            filename: 自定义文件名
            
        Returns:
            下载文件的完整路径
        """
        # 确保下载目录存在
        Path(download_dir).mkdir(parents=True, exist_ok=True)
        
        # 搜索论文
        papers = self.search_by_id([paper_id])
        if not papers:
            raise ValueError(f"未找到ID为 {paper_id} 的论文")
        
        paper = papers[0]
        
        # 下载PDF
        if filename:
            filepath = paper.download_pdf(dirpath=download_dir, filename=filename)
        else:
            filepath = paper.download_pdf(dirpath=download_dir)
        
        return filepath
    
    def print_paper_info(self, papers: List[arxiv.Result]) -> None:
        """打印论文信息
        
        Args:
            papers: 论文结果列表
        """
        for i, paper in enumerate(papers, 1):
            print(f"\n{i}. 标题: {paper.title}")
            print(f"   作者: {', '.join([author.name for author in paper.authors])}")
            print(f"   发布日期: {paper.published.strftime('%Y-%m-%d')}")
            print(f"   摘要: {paper.summary[:200]}...")
            print(f"   PDF链接: {paper.pdf_url}")
            print(f"   论文ID: {paper.entry_id.split('/')[-1]}")
    
    def search_and_display_with_date_filter(self, 
                                           query: str, 
                                           max_results: int = 10,
                                           days_back: Optional[int] = None) -> List[arxiv.Result]:
        """搜索并显示论文信息（支持日期过滤）
        
        Args:
            query: 搜索查询词
            max_results: 最大结果数量
            days_back: 搜索最近几天的论文，None表示不限制日期
            
        Returns:
            论文结果列表
        """
        print(f"正在搜索: {query}")
        print(f"最大结果数: {max_results}")
        if days_back:
            print(f"日期范围: 最近{days_back}天")
        else:
            print("日期范围: 不限制")
        print("-" * 80)
        
        papers = self.search_papers(query, max_results, days_back=days_back)
        self.print_paper_info(papers)
        
        return papers


# 使用示例
if __name__ == "__main__":
    # 创建搜索器实例
    searcher = ArxivSearcher()
    
    # 示例1: 搜索最近7天的"Retrieval Augmented Generation"相关论文（按相关性排序）
    print("=" * 80)
    print("示例1: 搜索最近30天的 'Retrieval Augmented Generation' 相关论文（按相关性排序）")
    print("=" * 80)
    
    rag_papers_30days = searcher.search_papers_by_relevance_and_date(
        query="Retrieval Augmented Generation RAG",
        max_results=30,
        days_back=30
    )
    searcher.print_paper_info(rag_papers_30days)
    
    # 示例2: 搜索最近14天的"Large Language Model"相关论文
    print("\n" + "=" * 80)
    print("示例2: 搜索最近14天的 'Large Language Model' 相关论文")
    print("=" * 80)

    llm_papers_14days = searcher.search_and_display_with_date_filter(
        query="Large Language Model LLM",
        max_results=5,
        days_back=14
    )
    
    # 示例3: 传统搜索（不限制日期）
    print("\n" + "=" * 80)
    print("示例3: 传统搜索（不限制日期）")
    print("=" * 80)

    all_papers = searcher.search_and_display_with_date_filter(
        query="Transformer Architecture",
        max_results=3
    )

    # 示例4: 下载论文
    if rag_papers_30days:
        print("\n" + "=" * 80)
        print("示例4: 下载第一篇论文")
        print("=" * 80)

        first_paper = rag_papers_30days[-1]
        paper_id = first_paper.entry_id.split('/')[-1]

        try:
            downloaded_path = searcher.download_paper(
                paper_id=paper_id,
                download_dir="./downloads",
                filename=f"recent_rag_paper_{paper_id}.pdf"
            )
            print(f"论文已下载到: {downloaded_path}")
        except Exception as e:
            print(f"下载失败: {e}")

