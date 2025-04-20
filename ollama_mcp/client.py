import asyncio
import threading
import queue

from pathlib import Path
from typing import Any, Optional, Union
from pydantic import BaseModel, Field, create_model
from mcp import ClientSession, StdioServerParameters
from mcp.client.stdio import stdio_client
from ollama import chat,Client


client = Client(
  # host='http://192.168.1.5:11434',
  host='http://localhost:11434',
  headers={'x-some-header': 'some-value'}
)

class OllamaMCP:

    def __init__(self, server_params: StdioServerParameters):
        self.server_params = server_params
        self.request_queue = queue.Queue()
        self.response_queue = queue.Queue()
        self.initialized = threading.Event()
        self.tools: list[Any] = []
        self.thread = threading.Thread(target=self._run_background, daemon=True)
        self.thread.start()

    def _run_background(self):
        asyncio.run(self._async_run())

    async def _async_run(self):
        try:
            async with stdio_client(self.server_params) as (read, write):
                async with ClientSession(read, write) as session:
                    await session.initialize()
                    self.session = session
                    tools_result = await session.list_tools()
                    self.tools = tools_result.tools
                    self.initialized.set()

                    while True:
                        try:
                            tool_name, arguments = self.request_queue.get(block=False)
                        except queue.Empty:
                            await asyncio.sleep(0.01)
                            continue

                        if tool_name is None:
                            break
                        try:
                            result = await session.call_tool(tool_name, arguments)
                            self.response_queue.put(result)
                        except Exception as e:
                            self.response_queue.put(f"错误: {str(e)}")
        except Exception as e:
            print("MCP会话初始化错误:", str(e))
            self.initialized.set()  # 即使初始化失败也解除等待线程的阻塞
            self.response_queue.put(f"MCP初始化错误: {str(e)}")

    def call_tool(self, tool_name: str, arguments: dict[str, Any]) -> Any:
        """
        发布工具调用请求并等待结果
        """
        if not self.initialized.wait(timeout=30):
            raise TimeoutError("MCP会话未能及时初始化。")
        self.request_queue.put((tool_name, arguments))
        result = self.response_queue.get()
        return result

    def shutdown(self):
        """
        干净地关闭持久会话
        """
        self.request_queue.put((None, None))
        self.thread.join()
        print("持久MCP会话已关闭。")


    @staticmethod
    def convert_json_type_to_python_type(json_type: str):
        """简单地将JSON类型映射到Python（Pydantic）类型。"""
        if json_type == "integer":
            return (int, ...)
        if json_type == "number":
            return (float, ...)
        if json_type == "string":
            return (str, ...)
        if json_type == "boolean":
            return (bool, ...)
        return (str, ...)

    def create_response_model(self):
        """
        基于获取的工具创建动态Pydantic响应模型
        """
        dynamic_classes = {}
        for tool in self.tools:
            print("tool",tool)
            class_name = tool.name.capitalize()
            properties: dict[str, Any] = {}
            for prop_name, prop_info in tool.inputSchema.get("properties", {}).items():
                json_type = prop_info.get("type", "string")
                properties[prop_name] = self.convert_json_type_to_python_type(json_type)

            model = create_model(
                class_name,
                __base__=BaseModel,
                __doc__=tool.description,
                **properties,
            )
            dynamic_classes[class_name] = model

        if dynamic_classes:
            all_tools_type = Union[tuple(dynamic_classes.values())]
            Response = create_model(
                "Response",
                __base__=BaseModel,
                __doc__="LLm响应类",
                response=(str, Field(..., description= "向用户确认函数将被调用。")),
                tool=(all_tools_type, Field(
                    ...,
                    description="用于运行和获取魔法输出的工具"
                )),
            )
        else:
            Response = create_model(
                "Response",
                __base__=BaseModel,
                __doc__="LLm响应类",
                response=(str, ...),
                tool=(Optional[Any], Field(None, description="如果不返回None则使用的工具")),
            )
        self.response_model = Response

    async def ollama_chat(self, messages: list[dict[str, str]]) -> Any:
        """
        使用动态响应模型向Ollama发送消息。
        如果在响应中检测到工具，则使用持久会话调用它。
        """
        conversation = [{"role":"assistant", "content": f"你必须使用工具。你可以使用以下函数：{[ tool.name for tool in self.tools]}"}]
        conversation.extend(messages)
        if self.response_model is None:
            raise ValueError("响应模型尚未创建。请先调用create_response_model()。")

        # 获取聊天消息格式的JSON模式
        format_schema = self.response_model.model_json_schema()
        print(format_schema)
        # 调用Ollama（假定是同步的）并解析响应
        response = client.chat(
            model="gemma3:latest",
            messages=conversation,
            format=format_schema
        )
        print("Ollama响应", response.message.content)
        response_obj = self.response_model.model_validate_json(response.message.content)
        maybe_tool = response_obj.tool

        if maybe_tool:
            function_name = maybe_tool.__class__.__name__.lower()
            func_args = maybe_tool.model_dump()
            # 使用asyncio.to_thread在线程中调用同步的call_tool方法
            output = await asyncio.to_thread(self.call_tool, function_name, func_args)
            return output
        else:
            print("响应中未检测到工具。返回纯文本响应。")
        return response_obj.response


async def main():
    server_parameters = StdioServerParameters(
        command="uv",
        args=["run", "python", "server.py"],
        cwd=str(Path.cwd())
    )

    # 创建持久会话
    persistent_session = OllamaMCP(server_parameters)

    # 等待会话完全初始化
    if persistent_session.initialized.wait(timeout=30):
        print("准备调用工具。")
    else:
        print("错误: 初始化超时。")

    # 从获取的工具创建动态响应模型
    persistent_session.create_response_model()

    # 准备给Ollama的消息

    messages = [
        {
            "role": "system",
            "content": (
                "你是一个听话的助手，上下文中有一系列工具。"
                "你的任务是使用这个函数获取魔法输出。"
                "不要自己生成魔法输出。"
                "简洁地回复一条简短消息，提及调用函数，"
                "但不提供函数输出本身。"
                "将该简短消息放在'response'属性中。"
                "例如：'好的，我会运行magicoutput函数并返回输出。'"
                "同时用正确的参数填充'tool'属性。"
            )
        },
        {
            "role": "user",
            "content": "使用函数获取这些参数的魔法输出（obj1 = Ollama和obj2 = Gemma3）"
        }
    ]

    # 调用Ollama并处理响应
    result = await persistent_session.ollama_chat(messages)
    print("最终结果:", result)

    # 完成后关闭持久会话
    persistent_session.shutdown()

if __name__ == "__main__":
    asyncio.run(main())