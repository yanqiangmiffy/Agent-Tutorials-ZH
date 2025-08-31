from dotenv import load_dotenv
import json
from fastapi import FastAPI

from typing import AsyncIterable
from fastapi.responses import StreamingResponse

from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.pydantic_v1 import BaseModel, Field
from langchain_deepseek import ChatDeepSeek
from dotenv import load_dotenv

load_dotenv()
app = FastAPI()

# 中文提示词模板
prompt_template = ChatPromptTemplate.from_messages([
    ('human', "请为{person_count}个中国历史人物生成详细传记，包括姓名、出生地和生平介绍")
])

llm_model = ChatDeepSeek(model="deepseek-chat")


class ChineseBiography(BaseModel):
    name: str = Field(description='人物的姓名')
    courtesy_name: str = Field(description='人物的字或号', default="")
    birth_place: str = Field(description='人物的出生地')
    dynasty: str = Field(description='人物所处的朝代')
    biography: str = Field(description='人物的详细生平介绍')
    achievements: str = Field(description='人物的主要成就和贡献')


class BiographyList(BaseModel):
    biographies: list[ChineseBiography] = Field(description='历史人物传记的列表')


# 配置结构化输出
structured_model = llm_model.with_structured_output(BiographyList)


async def generate_biographies_stream(person_count: int) -> AsyncIterable[str]:
    """
    生成中国历史人物传记并流式返回JSON结果

    Args:
        person_count: 要生成传记的人物数量

    Yields:
        str: 流式返回的JSON格式传记数据
    """
    # 格式化提示词
    formatted_prompt = prompt_template.format_prompt(person_count=person_count)

    # 调用模型并流式返回JSON结果
    async for chunk in structured_model.astream(formatted_prompt.to_messages()):
        result = json.dumps(chunk.dict(), ensure_ascii=False, indent=2)
        print(result)
        yield result


# 更新后的 FastAPI 端点
@app.post("/generate_biographies/")
async def stream_biographies(person_count: int):
    """
    流式生成中国历史人物传记

    Args:
        person_count: 要生成传记的人物数量

    Returns:
        流式响应，包含生成的人物传记数据
    """
    return StreamingResponse(
        generate_biographies_stream(person_count),
        media_type="text/event-stream",
        headers={
            "Content-Type": "text/event-stream; charset=utf-8",
            "Cache-Control": "no-cache",
            "Connection": "keep-alive"
        }
    )


@app.get("/")
async def root():
    return {
        "message": "中文人物传记生成API",
        "usage": "POST请求到 /generate_biographies/ 端点，传入person_count参数",
        "example": "curl -X POST 'http://localhost:8000/generate_biographies/' -H 'Content-Type: application/json' -d '{\"person_count\": 3}'"
    }


if __name__ == '__main__':
    import uvicorn

    uvicorn.run(app, host="0.0.0.0", port=8001)