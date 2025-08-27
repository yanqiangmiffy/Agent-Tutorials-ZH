from langgraph.prebuilt import create_react_agent
from langchain_deepseek import ChatDeepSeek
from dotenv import load_dotenv
from langchain_core.messages import HumanMessage

load_dotenv(verbose=True)

deepseek_chat=ChatDeepSeek(
    model="deepseek-chat",
    temperature=0,
    max_tokens=None,
    timeout=None,
    max_retries=2,
)


# Create a message
msg = HumanMessage(content="Hello world", name="Lance")
# Message list
messages = [msg]
print(messages)
# Invoke the model with a list of messages
response=deepseek_chat.invoke(messages)
print(response)
print(response.content)

response=deepseek_chat.stream(messages)

for res in response:
    print(res)