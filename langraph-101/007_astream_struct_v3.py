from dotenv import load_dotenv
import json
from fastapi import FastAPI
from typing import AsyncIterable, TypedDict
from fastapi.responses import StreamingResponse

from langchain_core.prompts import ChatPromptTemplate
from langchain_core.pydantic_v1 import BaseModel, Field
from langchain_deepseek import ChatDeepSeek
from langgraph.graph import StateGraph, END
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


# LangGraph状态定义
class BiographyState(TypedDict):
    person_count: int
    formatted_prompt: str
    result: BiographyList


# 配置结构化输出
structured_model = llm_model.with_structured_output(BiographyList)


# LangGraph节点函数
def format_prompt_node(state: BiographyState) -> BiographyState:
    """格式化提示词节点"""
    formatted_prompt = prompt_template.format_prompt(person_count=state["person_count"])
    return {
        **state,
        "formatted_prompt": formatted_prompt.to_string()
    }


async def generate_biography_node(state: BiographyState) -> BiographyState:
    """生成传记节点"""
    formatted_prompt = prompt_template.format_prompt(person_count=state["person_count"])
    result = await structured_model.ainvoke(formatted_prompt.to_messages())
    return {
        **state,
        "result": result
    }


# 创建LangGraph工作流
def create_biography_graph():
    """创建传记生成的状态图"""
    workflow = StateGraph(BiographyState)

    # 添加节点
    workflow.add_node("format_prompt", format_prompt_node)
    workflow.add_node("generate_biography", generate_biography_node)

    # 设置边
    workflow.set_entry_point("format_prompt")
    workflow.add_edge("format_prompt", "generate_biography")
    workflow.add_edge("generate_biography", END)

    return workflow.compile()


# 创建图实例
biography_graph = create_biography_graph()


async def generate_biographies_stream(person_count: int) -> AsyncIterable[str]:
    """
    使用LangGraph生成中国历史人物传记并流式返回JSON结果

    Args:
        person_count: 要生成传记的人物数量

    Yields:
        str: 流式返回的JSON格式传记数据
    """
    # 初始化状态
    initial_state = {
        "person_count": person_count,
        "formatted_prompt": "",
        "result": None
    }

    # 使用LangGraph流式执行
    # async for mode,event in biography_graph.astream(initial_state,stream_mode=["messages","values","updates"]):
    async for mode,event in biography_graph.astream(initial_state,stream_mode=["messages","values","updates"]):
        # 当到达最终结果时，流式返回
        if "generate_biography" in event and event["generate_biography"]["result"]:
            result = event["generate_biography"]["result"]
            json_result = json.dumps(result.dict(), ensure_ascii=False, indent=2)
            print(f"mode:{mode}===>", event)
            print(json_result)
            yield json_result


# FastAPI 端点
@app.post("/generate_biographies/")
async def stream_biographies(person_count: int):
    """
    使用LangGraph流式生成中国历史人物传记

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
        "message": "基于LangGraph的中文人物传记生成API",
        "usage": "POST请求到 /generate_biographies/ 端点，传入person_count参数",
        "example": "curl -X POST 'http://localhost:8001/generate_biographies/' -H 'Content-Type: application/json' -d '{\"person_count\": 3}'",
        "technology": "使用LangGraph状态图进行工作流管理"
    }


if __name__ == '__main__':
    import uvicorn

    uvicorn.run(app, host="0.0.0.0", port=8001)