"""
LangGraph 结构化输出示例
使用 Pydantic 模型生成符合特定格式的响应
"""

from langgraph.prebuilt import create_react_agent
from langchain_deepseek import ChatDeepSeek
from dotenv import load_dotenv
from pydantic import BaseModel
from typing import List, Optional
load_dotenv()
model=ChatDeepSeek(
    model="deepseek-chat",
    temperature=0,
    max_retries=2,
)
# 定义工具函数
def get_weather(city: str) -> str:
    """获取城市天气信息"""
    weather_data = {
        "北京": "晴天，气温25°C，空气质量良好",
        "上海": "多云转阴，气温28°C，湿度80%",
        "深圳": "小雨，气温26°C，风力3级",
        "广州": "阴天，气温30°C，空气质量中等"
    }
    return weather_data.get(city, f"{city}暂无天气数据")

def search_restaurant(city: str, cuisine: str = "") -> str:
    """搜索餐厅信息"""
    restaurants = {
        "北京": "老北京炸酱面馆、全聚德烤鸭店、海底捞火锅",
        "上海": "小笼包店、本帮菜馆、外滩西餐厅",
        "深圳": "潮汕牛肉火锅、茶餐厅、海鲜大排档"
    }
    return restaurants.get(city, f"{city}暂无餐厅推荐")

# 示例1: 天气查询结构化响应
class WeatherResponse(BaseModel):
    """天气查询响应格式"""
    city: str                    # 城市名称
    conditions: str              # 天气状况
    temperature: Optional[str]   # 温度信息
    additional_info: Optional[str]  # 附加信息

# 创建天气查询代理
weather_agent = create_react_agent(
    model=model,
    tools=[get_weather],
    response_format=WeatherResponse
)

print("=== 示例1: 天气查询结构化输出 ===")
weather_response = weather_agent.invoke(
    {"messages": [{"role": "user", "content": "查询北京的天气情况"}]}
)

print("用户: 查询北京的天气情况")
print("结构化响应:")
print(f"  城市: {weather_response['structured_response'].city}")
print(f"  天气状况: {weather_response['structured_response'].conditions}")
print(f"  温度: {weather_response['structured_response'].temperature}")
print(f"  附加信息: {weather_response['structured_response'].additional_info}")


# 示例2: 餐厅推荐结构化响应
class RestaurantRecommendation(BaseModel):
    """餐厅推荐响应格式"""
    city: str                           # 城市
    restaurants: List[str]              # 餐厅列表
    cuisine_type: Optional[str]         # 菜系类型
    recommendation_reason: str          # 推荐理由

# 创建餐厅推荐代理
restaurant_agent = create_react_agent(
    model=model,
    tools=[search_restaurant],
    response_format=RestaurantRecommendation
)

print("\n=== 示例2: 餐厅推荐结构化输出 ===")
restaurant_response = restaurant_agent.invoke(
    {"messages": [{"role": "user", "content": "推荐上海的好餐厅"}]}
)

print("用户: 推荐上海的好餐厅")
print("结构化响应:")
print(f"  城市: {restaurant_response['structured_response'].city}")
print(f"  餐厅列表: {restaurant_response['structured_response'].restaurants}")
print(f"  菜系类型: {restaurant_response['structured_response'].cuisine_type}")
print(f"  推荐理由: {restaurant_response['structured_response'].recommendation_reason}")



# 示例3: 综合查询结构化响应
class TravelInfo(BaseModel):
    """旅行信息响应格式"""
    destination: str                    # 目的地
    weather_info: str                   # 天气信息
    restaurant_suggestions: List[str]   # 餐厅建议
    travel_tips: List[str]              # 旅行建议
    overall_rating: str                 # 总体评价

# 创建旅行助手代理
travel_agent = create_react_agent(
    model=model,
    tools=[get_weather, search_restaurant],
    response_format=TravelInfo
)

print("\n=== 示例3: 旅行信息结构化输出 ===")
travel_response = travel_agent.invoke(
    {"messages": [{"role": "user", "content": "我想去深圳旅游，给我一些建议"}]}
)

print("用户: 我想去深圳旅游，给我一些建议")
print("结构化响应:")
print(f"  目的地: {travel_response['structured_response'].destination}")
print(f"  天气信息: {travel_response['structured_response'].weather_info}")
print(f"  餐厅建议: {travel_response['structured_response'].restaurant_suggestions}")
print(f"  旅行建议: {travel_response['structured_response'].travel_tips}")
print(f"  总体评价: {travel_response['structured_response'].overall_rating}")
print("\n=== 结构化输出的优势 ===")
print("1. 数据格式统一，便于后续处理")
print("2. 类型安全，避免数据格式错误")
print("3. 易于集成到其他系统中")
print("4. 提供清晰的数据结构定义")


# ========with_structured_output=======
from pydantic import BaseModel, Field

class PersonInfo(BaseModel):
    name: str = Field(description="人的姓名")
    age: int = Field(description="年龄")
    skills: list[str] = Field(description="技能列表")

# 创建结构化输出模型
structured_llm = model.with_structured_output(PersonInfo)

response = structured_llm.invoke("分析这个人：张三，28岁，会Python和机器学习")
print(response)