from langgraph.prebuilt import create_react_agent
from langchain_deepseek import ChatDeepSeek
from dotenv import load_dotenv

load_dotenv(verbose=True)

model=ChatDeepSeek(
    model="deepseek-chat",
    temperature=0,
    max_tokens=None,
    timeout=None,
    max_retries=2,
)

# response=model.invoke("你好,请帮我生成一首儿歌")
# print(response)
def get_weather(city: str) -> str:
    """Get weather for a given city."""
    return f" {city}的天气是晴朗的!"

agent = create_react_agent(
    model=model,
    tools=[get_weather],
    prompt="你是一个有用的助手"
)

# Run the agent
response=agent.invoke(
    {"messages": [{"role": "user", "content": "请问北京的天气怎么样"}]}
)

print(response.keys())
print(response["messages"])