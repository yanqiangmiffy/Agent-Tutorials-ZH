from json_repair import repair_json, loads
import json

# 示例1: 修复单引号替代双引号的问题
bad_json1 = "{'name': 'John', 'age': 30, 'city': 'New York'}"
fixed_json1 = repair_json(bad_json1)
print("修复单引号:")
print(f"修复前: {bad_json1}")
print(f"修复后: {fixed_json1}")
print()

# 示例2: 修复缺少引号的键
bad_json2 = "{name: 'John', age: 30, city: 'New York'}"
fixed_json2 = repair_json(bad_json2)
print("修复缺少引号的键:")
print(f"修复前: {bad_json2}")
print(f"修复后: {fixed_json2}")
print()

# 示例3: 修复逗号问题
bad_json3 = '{"name": "John", "age": 30, "city": "New York",}'  # 结尾多余的逗号
fixed_json3 = repair_json(bad_json3)
print("修复多余的逗号:")
print(f"修复前: {bad_json3}")
print(f"修复后: {fixed_json3}")
print()

# 示例4: 修复缺少大括号的问题
bad_json4 = '"name": "John", "age": 30, "city": "New York"'
fixed_json4 = repair_json(bad_json4)
print("修复缺少括号:")
print(f"修复前: {bad_json4}")
print(f"修复后: {fixed_json4}")
print()

# 示例5: 修复非标准的布尔值或空值
bad_json5 = '{"name": "John", "active": True, "data": None}'
fixed_json5 = repair_json(bad_json5)
print("修复非标准的布尔值或空值:")
print(f"修复前: {bad_json5}")
print(f"修复后: {fixed_json5}")
print()

# 示例6: 修复嵌套结构中的错误
bad_json6 = '{"user": {"name": "John", "contacts": {"email": "john@example.com", phone: "123-456-7890"}}}'
fixed_json6 = repair_json(bad_json6)
print("修复嵌套结构中的错误:")
print(f"修复前: {bad_json6}")
print(f"修复后: {fixed_json6}")
print()

# 示例7: 修复数组中的错误
bad_json7 = '{"items": [1, 2, 3,, 4, 5]}'  # 数组中有多余的逗号
fixed_json7 = repair_json(bad_json7)
print("修复数组中的错误:")
print(f"修复前: {bad_json7}")
print(f"修复后: {fixed_json7}")
print()

# 示例8: 修复不匹配的括号
bad_json8 = '{"name": "John", "items": [1, 2, 3}'  # 方括号没有闭合
fixed_json8 = repair_json(bad_json8)
print("修复不匹配的括号:")
print(f"修复前: {bad_json8}")
print(f"修复后: {fixed_json8}")
print()

# 示例9: 修复中文等非ASCII字符的问题
bad_json9 = "{'name': '张三', 'city': '北京'}"
fixed_json9 = repair_json(bad_json9, ensure_ascii=False)
print("修复包含中文的JSON并保留中文字符:")
print(f"修复前: {bad_json9}")
print(f"修复后: {fixed_json9}")
print()

# 示例10: 直接获取Python对象而不是JSON字符串
bad_json10 = "{'name': 'John', 'age': 30, 'skills': ['Python', 'JavaScript']}"
fixed_obj10 = loads(bad_json10)  # 等同于 repair_json(bad_json10, return_objects=True)
print("直接获取Python对象:")
print(f"修复前: {bad_json10}")
print(f"修复后(Python对象): {fixed_obj10}")
print(f"对象类型: {type(fixed_obj10)}")
print()

# 示例11: 处理严重破损的JSON
severely_broken_json = "{这不是有效的JSON，name: 'John', age: missing_value}"
try:
    fixed_severely_broken = repair_json(severely_broken_json,ensure_ascii=False)
    print("修复严重破损的JSON:")
    print(f"修复前: {severely_broken_json}")
    print(f"修复后: {fixed_severely_broken}")
except Exception as e:
    print(f"修复失败: {e}")
print()

# 示例12: 处理包含注释的JSON (JSON标准不支持注释)
json_with_comments = """
{
  "name": "John", // 这是用户名
  "age": 30, /* 这是年龄 */
  "city": "New York"
}
"""
fixed_json_comments = repair_json(json_with_comments)
print("修复包含注释的JSON:")
print(f"修复前: {json_with_comments}")
print(f"修复后: {fixed_json_comments}")