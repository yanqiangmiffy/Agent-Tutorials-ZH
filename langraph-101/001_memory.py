from langgraph.prebuilt import create_react_agent
from langchain_deepseek import ChatDeepSeek
from dotenv import load_dotenv
from langgraph.checkpoint.memory import InMemorySaver
from pydantic import BaseModel
from typing import List, Optional
load_dotenv()
model=ChatDeepSeek(
    model="deepseek-chat",
    temperature=0,
    max_retries=2,
)
# 定义一个简单的工具函数
def get_weather(city: str) -> str:
    """获取天气信息"""
    weather_data = {
        "北京": "晴天，25°C",
        "上海": "多云，28°C",
        "深圳": "小雨，26°C"
    }
    return weather_data.get(city, f"{city}天气暂无数据")

# 创建内存检查点
checkpointer = InMemorySaver()

# 创建带记忆的代理
agent = create_react_agent(
    model=model,
    tools=[get_weather],
    checkpointer=checkpointer
)

# 配置会话ID
config = {"configurable": {"thread_id": "chat-001"}}

# 第一轮对话
print("=== 第一轮对话 ===")
response1 = agent.invoke(
    {"messages": [{"role": "user", "content": "你好，我叫张三，请查询北京天气"}]},
    config
)
print("用户: 你好，我叫张三，请查询北京天气")
print("助手:", response1['messages'][-1].content)

# 第二轮对话 - 测试记忆功能
print("\n=== 第二轮对话 ===")
response2 = agent.invoke(
    {"messages": [{"role": "user", "content": "我叫什么名字？"}]},
    config
)
print("用户: 我叫什么名字？")
print("助手:", response2['messages'][-1].content)

# 第三轮对话 - 继续使用记忆
print("\n=== 第三轮对话 ===")
response3 = agent.invoke(
    {"messages": [{"role": "user", "content": "上海天气怎么样？"}]},
    config
)
print("用户: 上海天气怎么样？")
print("助手:", response3['messages'][-1].content)