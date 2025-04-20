import asyncio
import threading
import queue

from pathlib import Path
from typing import Any, Optional, Union
from pydantic import BaseModel, Field, create_model
from mcp import ClientSession, StdioServerParameters
from mcp.client.stdio import stdio_client
from ollama import chat, Client

client = Client(
    host='http://192.168.1.5:11434',
    # host='http://localhost:11434',
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
            # print("tool", tool)
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
                response=(str, Field(..., description="向用户确认函数将被调用。")),
                tool=(all_tools_type, Field(
                    ...,
                    description="用于运行和获取结果的工具"
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
        # 添加系统消息，告知可用工具
        tool_names = [tool.name for tool in self.tools]
        tool_descriptions = [f"{tool.name}: {tool.description}" for tool in self.tools]

        system_message = {
            "role": "system",
            "content": f"你可以使用以下工具：{tool_names}。工具详情：{tool_descriptions}"
        }

        # 确保系统消息在最前面
        if messages and messages[0].get("role") == "system":
            # 合并系统消息内容
            messages[0]["content"] = f"{messages[0]['content']} {system_message['content']}"
        else:
            # 在消息列表开头添加系统消息
            messages.insert(0, system_message)

        if self.response_model is None:
            raise ValueError("响应模型尚未创建。请先调用create_response_model()。")

        # 获取聊天消息格式的JSON模式
        format_schema = self.response_model.model_json_schema()
        # print(format_schema)

        # 调用Ollama并解析响应
        response = client.chat(
            model="gemma3:latest",  # 确保使用适当的模型
            messages=messages,
            format=format_schema
        )
        print("Ollama响应", response.message.content)

        try:
            response_obj = self.response_model.model_validate_json(response.message.content)
            maybe_tool = response_obj.tool

            if maybe_tool:
                function_name = maybe_tool.__class__.__name__.lower()
                func_args = maybe_tool.model_dump()
                # 使用asyncio.to_thread在线程中调用同步的call_tool方法
                output = await asyncio.to_thread(self.call_tool, function_name, func_args)
                print(output.content[0].text)

                weather_messages = [
                    {
                        "role": "system",
                        "content": (
                            "你是一个有用的天气助手，更够解释天气。"
                        )
                    },
                    {
                        "role": "user",
                        "content":f"下面获取的天气情况{output.content[0].text}\n"
                                  f"帮我解释下天气,请用中文解释，输出文本"

                    }
                ]
                weather_response = client.chat(
                    model="gemma3:latest",  # 确保使用适当的模型
                    messages=weather_messages,
                )
                print(type(weather_response.message.content),weather_response.message.content)

                return f"{response_obj.response}\n\n工具结果:\n{output}"
            else:
                print("响应中未检测到工具。返回纯文本响应。")
                return response_obj.response
        except Exception as e:
            print(f"解析响应时出错: {e}")
            return f"解析响应时出错: {e}\n原始响应: {response.message.content}"


async def main():
    server_parameters = StdioServerParameters(
        command="python",  # 修改为适当的命令
        args=["server.py"],  # 修改为天气服务脚本的名称
        cwd=str(Path.cwd())
    )

    # 创建持久会话
    persistent_session = OllamaMCP(server_parameters)

    # 等待会话完全初始化
    if persistent_session.initialized.wait(timeout=30):
        print("准备调用工具。")
        for tool in persistent_session.tools:
            print(f"可用工具: {tool.name} - {tool.description}")
    else:
        print("错误: 初始化超时。")
        return

    # 从获取的工具创建动态响应模型
    persistent_session.create_response_model()

    # 测试天气查询
    weather_messages = [
        {
            "role": "system",
            "content": (
                "你是一个有用的助手，能够查询全球各地的天气。"
                "当用户询问特定城市的天气情况时，你应该使用get_weather工具获取实时天气数据。"
                "你的回复应该简洁明了，首先确认你将查询天气，然后在tool属性中提供正确的参数。"
            )
        },
        {
            "role": "user",
            "content": "我想知道北京的天气情况。"
        }
    ]

    # 调用Ollama并处理响应
    result = await persistent_session.ollama_chat(weather_messages)
    print("\n最终结果:\n", result)

    # # 再测试另一个城市
    # second_query = [
    #     {
    #         "role": "user",
    #         "content": "除了北京天气，我还想知道上海现在的天气怎么样？"
    #     }
    # ]
    #
    # # 继续对话，添加新的用户消息
    # weather_messages.extend(second_query)
    # result = await persistent_session.ollama_chat(weather_messages)
    # print("\n第二次查询结果:\n", result)

    # 完成后关闭持久会话
    persistent_session.shutdown()


if __name__ == "__main__":
    asyncio.run(main())