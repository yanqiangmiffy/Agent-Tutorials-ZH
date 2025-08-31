from pydantic import BaseModel, Field
from langchain_deepseek import ChatDeepSeek
from typing import Optional
from dotenv import load_dotenv
load_dotenv()
# 定义Pydantic模型
class ProductInfo(BaseModel):
    """产品基本信息"""

    name: str = Field(description="产品名称")
    price: float = Field(description="产品价格，单位为元")
    category: str = Field(description="产品所属类别")
    rating: Optional[int] = Field(
        default=None, description="产品评分，1-10分"
    )


# 初始化模型
llm = ChatDeepSeek(model="deepseek-chat")
structured_llm = llm.with_structured_output(ProductInfo)

# 调用示例
result = structured_llm.invoke("帮我分析一下iPhone 15这款手机")
print(result)
# 输出: ProductInfo(name='iPhone 15', price=5999.0, category='智能手机', rating=9)


from typing_extensions import Annotated, TypedDict

class NewsSummary(TypedDict):
    """新闻摘要信息"""
    title: Annotated[str, ..., "新闻标题"]
    summary: Annotated[str, ..., "新闻摘要内容"]
    keywords: Annotated[list[str], ..., "新闻关键词"]

structured_llm = llm.with_structured_output(NewsSummary)

print("流式输出示例:")
for chunk in structured_llm.stream("总结今天的科技新闻"):
    print(chunk)
