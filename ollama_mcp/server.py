from fastmcp import FastMCP
# 创建MCP服务器
mcp = FastMCP("TestServer")
# 我的工具:
@mcp.tool()
def magicoutput(obj1: str, obj2: str) -> int:
    """使用此函数获取魔法输出"""
    print(f"输入参数：obj1:{obj1}，obj2:{obj2}")
    return f"输入参数：obj1:{obj1}，obj2:{obj2},魔法输出：Hello MCP，MCP Hello"
if __name__ == "__main__":
    mcp.run()