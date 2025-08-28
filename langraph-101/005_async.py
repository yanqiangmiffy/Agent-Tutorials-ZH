import asyncio
import time
from langchain.chat_models import init_chat_model
from langchain_deepseek import ChatDeepSeek
from langgraph.graph import MessagesState, StateGraph
from dotenv import load_dotenv
load_dotenv()
# 初始化 LLM 模型
llm = ChatDeepSeek(model="deepseek-chat")


# 同步版本的节点
def sync_node(state: MessagesState):
    """同步版本：会阻塞等待"""
    new_message = llm.invoke(state["messages"])
    return {"messages": [new_message]}


# 异步版本的节点
async def async_node(state: MessagesState):
    """异步版本：可以并发执行"""
    new_message = await llm.ainvoke(state["messages"])
    return {"messages": [new_message]}


# 创建同步图
sync_builder = StateGraph(MessagesState).add_node("node", sync_node).set_entry_point("node")
sync_graph = sync_builder.compile()

# 创建异步图
async_builder = StateGraph(MessagesState).add_node("node", async_node).set_entry_point("node")
async_graph = async_builder.compile()

# 准备测试消息
# messages = [
#     {"role": "user", "content": "Hello, how are you?"},
#     {"role": "user", "content": "What's the weather like?"},
#     {"role": "user", "content": "Tell me a joke"},
#     {"role": "user", "content": "Explain quantum physics"},
#     {"role": "user", "content": "What's 2+2?"}
# ]
# 准备测试消息
messages = [
    {"role": "user", "content": "你好，请介绍一下自己"},
    {"role": "user", "content": "请解释一下什么是人工智能"},
    {"role": "user", "content": "给我讲个笑话吧"},
    {"role": "user", "content": "请推荐几本好书"},
    {"role": "user", "content": "今天天气怎么样？"}
]


def run_sync_sequential():
    """测试同步顺序执行"""
    print("🔄 同步顺序执行测试...")
    start_time = time.time()

    results = []
    for i, msg in enumerate(messages):
        print(f"  处理消息 {i + 1}/5...")
        result = sync_graph.invoke({"messages": [msg]})
        results.append(result)

    end_time = time.time()
    duration = end_time - start_time
    print(f"✅ 同步执行完成，总耗时: {duration:.2f} 秒")
    return results, duration





async def run_async_sequential():
    """测试异步顺序执行（对比参考）"""
    print("⏳ 异步顺序执行测试...")
    start_time = time.time()

    results = []
    for i, msg in enumerate(messages):
        print(f"  处理消息 {i + 1}/5...")
        result = await async_graph.ainvoke({"messages": [msg]})
        results.append(result)

    end_time = time.time()
    duration = end_time - start_time
    print(f"✅ 异步顺序执行完成，总耗时: {duration:.2f} 秒")
    return results, duration

async def run_async_concurrent():
    """测试异步并发执行"""
    print("🚀 异步并发执行测试...")
    start_time = time.time()

    # 创建所有任务
    tasks = []
    for i, msg in enumerate(messages):
        print(f"  启动任务 {i + 1}/5...")
        task = async_graph.ainvoke({"messages": [msg]})
        tasks.append(task)

    # 并发执行所有任务
    print("  🔥 所有任务并发运行中...")
    results = await asyncio.gather(*tasks)

    end_time = time.time()
    duration = end_time - start_time
    print(f"✅ 异步执行完成，总耗时: {duration:.2f} 秒")
    return results, duration

async def main():
    """主函数：运行所有测试"""
    print("=" * 60)
    print("🧪 LangGraph 异步 vs 同步性能测试")
    print("=" * 60)
    print(f"📝 测试场景：处理 {len(messages)} 个 LLM 请求")
    print()

    # 1. 同步顺序执行
    sync_results, sync_time = run_sync_sequential()
    print()

    # 2. 异步顺序执行（对比参考）
    async_seq_results, async_seq_time = await run_async_sequential()
    print()

    # 3. 异步并发执行
    async_con_results, async_con_time = await run_async_concurrent()
    print()

    # 性能对比分析
    print("=" * 60)
    print("📊 性能对比分析")
    print("=" * 60)
    print(f"同步顺序执行:     {sync_time:.2f} 秒")
    print(f"异步顺序执行:     {async_seq_time:.2f} 秒")
    print(f"异步并发执行:     {async_con_time:.2f} 秒")
    print()

    # 计算性能提升
    if async_con_time > 0:
        speedup_vs_sync = sync_time / async_con_time
        speedup_vs_async_seq = async_seq_time / async_con_time

        print("🎯 性能提升:")
        print(f"异步并发 vs 同步顺序: {speedup_vs_sync:.1f}x 倍速提升")
        print(f"异步并发 vs 异步顺序: {speedup_vs_async_seq:.1f}x 倍速提升")
        print()

        print("💡 关键发现:")
        print("• 异步并发执行可以显著减少总耗时")
        print("• 当有多个独立的 LLM 调用时，并发执行效果最明显")
        print("• 异步顺序执行与同步执行时间相近（都是逐个等待）")
        print("• 实际加速比取决于网络延迟和 LLM 响应时间")


# 运行测试
if __name__ == "__main__":
    # 如果在 Jupyter 中运行，使用：
    # await main()

    # 如果在命令行中运行，使用：
    asyncio.run(main())