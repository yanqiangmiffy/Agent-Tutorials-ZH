from pdfdeal import Doc2X
from pathlib import Path
from typing import Union, List, Tuple, Optional
import os
import zipfile
import shutil

class PDFParser:
    """PDF解析器类，用于将PDF文件转换为Markdown内容"""
    
    def __init__(self, api_key: str, debug: bool = True, thread: int = 5, full_speed: bool = True):
        """
        初始化PDF解析器
        
        Args:
            api_key: Doc2X API密钥
            debug: 是否开启调试模式
            thread: 线程数
            full_speed: 是否开启全速模式
        """
        self.client = Doc2X(
            apikey=api_key,
            debug=debug,
            thread=thread,
            full_speed=full_speed
        )
    
    def _extract_zip_file(self, zip_path: str, extract_to: str = None) -> str:
        """
        解压ZIP文件
        
        Args:
            zip_path: ZIP文件路径
            extract_to: 解压目标目录，如果为None则解压到ZIP文件同目录
            
        Returns:
            解压后的目录路径
        """
        if not os.path.exists(zip_path):
            raise FileNotFoundError(f"ZIP文件不存在: {zip_path}")
        
        # 如果没有指定解压目录，则使用ZIP文件同目录
        if extract_to is None:
            extract_to = os.path.dirname(zip_path)
        
        # 创建解压目录
        Path(extract_to).mkdir(parents=True, exist_ok=True)
        
        # 解压文件
        with zipfile.ZipFile(zip_path, 'r') as zip_ref:
            zip_ref.extractall(extract_to)
        
        print(f"ZIP文件已解压到: {extract_to}")
        return extract_to
    
    def parse_pdf_to_markdown_with_auto_extract(self, 
                                               pdf_path: str,
                                               output_path: str = "./Output",
                                               output_format: str = "md",
                                               ocr: bool = True,
                                               convert: bool = False,
                                               auto_extract: bool = True,
                                               keep_zip: bool = False) -> Tuple[Union[str, List[str]], List[dict], bool, str]:
        """
        将PDF文件解析为Markdown内容并自动解压（如果生成了ZIP文件）
        
        Args:
            pdf_path: PDF文件路径
            output_path: 输出目录路径
            output_format: 输出格式，支持 'md', 'md_dollar', 'text', 'texts', 'detailed'
            ocr: 是否使用OCR
            convert: 是否将 [ 和 [[ 转换为 $ 和 $$
            auto_extract: 是否自动解压ZIP文件
            keep_zip: 是否保留原ZIP文件
            
        Returns:
            成功转换的内容或文件路径、失败信息、是否有错误、解压目录路径的元组
        """
        # 检查PDF文件是否存在
        if not os.path.exists(pdf_path):
            raise FileNotFoundError(f"PDF文件不存在: {pdf_path}")
        
        # 确保输出目录存在
        Path(output_path).mkdir(parents=True, exist_ok=True)
        
        # 调用Doc2X进行转换
        success, failed, flag = self.client.pdf2file(
            pdf_file=pdf_path,
            output_path=output_path,
            output_format=output_format,
            ocr=ocr,
            convert=convert,
        )
        
        extract_dir = None
        
        # 如果转换成功且需要自动解压
        if not flag and auto_extract:
            # 检查是否生成了ZIP文件
            if isinstance(success, str) and success.endswith('.zip'):
                try:
                    # 解压ZIP文件
                    extract_dir = self._extract_zip_file(success)
                    
                    # 如果不保留ZIP文件，则删除它
                    if not keep_zip:
                        os.remove(success)
                        print(f"已删除ZIP文件: {success}")
                    
                    print(f"解压完成，文件位于: {extract_dir}")
                    
                except Exception as e:
                    print(f"解压ZIP文件时出错: {e}")
            elif isinstance(success, list):
                # 处理多个文件的情况
                for file_path in success:
                    if isinstance(file_path, str) and file_path.endswith('.zip'):
                        try:
                            extract_dir = self._extract_zip_file(file_path)
                            
                            if not keep_zip:
                                os.remove(file_path)
                                print(f"已删除ZIP文件: {file_path}")
                                
                        except Exception as e:
                            print(f"解压ZIP文件 {file_path} 时出错: {e}")
        
        return success, failed, flag, extract_dir
    
    def parse_existing_zip(self, zip_path: str, extract_to: str = None, keep_zip: bool = False) -> str:
        """
        解析已存在的ZIP文件
        
        Args:
            zip_path: ZIP文件路径
            extract_to: 解压目标目录
            keep_zip: 是否保留原ZIP文件
            
        Returns:
            解压后的目录路径
        """
        extract_dir = self._extract_zip_file(zip_path, extract_to)
        
        if not keep_zip:
            os.remove(zip_path)
            print(f"已删除ZIP文件: {zip_path}")
        
        return extract_dir
    
    def parse_pdf_to_markdown(self, 
                             pdf_path: str,
                             output_path: str = "./Output",
                             output_format: str = "md",
                             ocr: bool = True,
                             convert: bool = False,
                             ) -> Tuple[Union[str, List[str]], List[dict], bool]:
        """
        将PDF文件解析为Markdown内容
        
        Args:
            pdf_path: PDF文件路径
            output_path: 输出目录路径
            output_format: 输出格式，支持 'md', 'md_dollar', 'text', 'texts', 'detailed'
            ocr: 是否使用OCR
            convert: 是否将 [ 和 [[ 转换为 $ 和 $$

        Returns:
            成功转换的内容或文件路径、失败信息、是否有错误的元组
        """
        # 检查PDF文件是否存在
        if not os.path.exists(pdf_path):
            raise FileNotFoundError(f"PDF文件不存在: {pdf_path}")
        
        # 确保输出目录存在
        Path(output_path).mkdir(parents=True, exist_ok=True)
        
        # 调用Doc2X进行转换
        success, failed, flag = self.client.pdf2file(
            pdf_file=pdf_path,
            output_path=output_path,
            output_format=output_format,
            ocr=ocr,
            convert=convert,
        )
        
        return success, failed, flag
    
    def parse_pdf_to_text(self, pdf_path: str) -> str:
        """
        将PDF文件解析为纯文本字符串
        
        Args:
            pdf_path: PDF文件路径
            
        Returns:
            解析后的文本内容
        """
        success, failed, flag = self.parse_pdf_to_markdown(
            pdf_path=pdf_path,
            output_format="text"
        )
        
        if flag:  # 有错误
            raise Exception(f"PDF解析失败: {failed}")
        
        return success
    
    def parse_pdf_to_pages(self, pdf_path: str) -> List[str]:
        """
        将PDF文件按页解析为文本列表
        
        Args:
            pdf_path: PDF文件路径
            
        Returns:
            按页分割的文本列表
        """
        success, failed, flag = self.parse_pdf_to_markdown(
            pdf_path=pdf_path,
            output_format="texts"
        )
        
        if flag:  # 有错误
            raise Exception(f"PDF解析失败: {failed}")
        
        return success
    
    def parse_pdf_to_markdown_file(self, 
                                  pdf_path: str,
                                  output_path: str = "./Output",
                                  custom_filename: Optional[str] = None) -> str:
        """
        将PDF文件转换为Markdown文件并保存
        
        Args:
            pdf_path: PDF文件路径
            output_path: 输出目录路径
            custom_filename: 自定义输出文件名
            
        Returns:
            生成的Markdown文件路径
        """
        output_names = None
        if custom_filename:
            output_names = [custom_filename]
        
        success, failed, flag = self.client.pdf2file(
            pdf_file=pdf_path,
            output_names=output_names,
            output_path=output_path,
            output_format="md",
            ocr=True
        )
        
        if flag:  # 有错误
            raise Exception(f"PDF转换失败: {failed}")
        
        return success[0] if isinstance(success, list) else success
    
    def batch_parse_pdfs(self, 
                        pdf_paths: List[str],
                        output_path: str = "./Output",
                        output_format: str = "md") -> Tuple[List[str], List[dict], bool]:
        """
        批量解析多个PDF文件
        
        Args:
            pdf_paths: PDF文件路径列表
            output_path: 输出目录路径
            output_format: 输出格式
            
        Returns:
            成功转换的文件路径列表、失败信息列表、是否有错误
        """
        # 检查所有PDF文件是否存在
        for pdf_path in pdf_paths:
            if not os.path.exists(pdf_path):
                raise FileNotFoundError(f"PDF文件不存在: {pdf_path}")
        
        # 确保输出目录存在
        Path(output_path).mkdir(parents=True, exist_ok=True)
        
        # 批量转换
        success, failed, flag = self.client.pdf2file(
            pdf_file=pdf_paths,
            output_path=output_path,
            output_format=output_format,
            ocr=True
        )
        
        return success, failed, flag
    
    def get_markdown_content(self, pdf_path: str) -> str:
        """
        直接获取PDF的Markdown内容（不保存文件）
        
        Args:
            pdf_path: PDF文件路径
            
        Returns:
            Markdown格式的文本内容
        """
        success, failed, flag = self.parse_pdf_to_markdown(
            pdf_path=pdf_path,
            output_format="text",
            convert=True  # 转换数学公式格式
        )
        
        if flag:  # 有错误
            raise Exception(f"PDF解析失败: {failed}")
        
        return success


# 使用示例
if __name__ == "__main__":
    # 初始化解析器（需要替换为您的API密钥）
    parser = PDFParser(api_key="sk-8vnrrnhtttc6xtk1qout8cqti65g3ocz")
    # 示例2: 解析PDF并自动解压
    pdf_path = "downloads/recent_rag_paper_2505.22571v3.pdf"
    
    if os.path.exists(pdf_path):
        try:
            print("\n正在解析PDF并自动解压...")
            success, failed, flag, extract_dir = parser.parse_pdf_to_markdown_with_auto_extract(
                pdf_path=pdf_path,
                output_path="./auto_extract_output",
                output_format="md",
                auto_extract=True,
                keep_zip=False  # 不保留ZIP文件
            )
            
            if not flag:
                print(f"PDF解析成功！")
                if extract_dir:
                    print(f"内容已自动解压到: {extract_dir}")
                else:
                    print(f"生成的文件: {success}")
            else:
                print(f"PDF解析失败: {failed}")
                
        except Exception as e:
            print(f"解析PDF时出错: {e}")