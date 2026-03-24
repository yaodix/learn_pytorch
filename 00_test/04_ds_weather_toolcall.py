import json
import requests
from openai import OpenAI
import os
from dotenv import load_dotenv

load_dotenv()

client = OpenAI(
    api_key=os.environ.get('DEEPSEEK_API_KEY'),
    base_url="https://api.deepseek.com",
)

# ── 真实天气查询函数（使用 wttr.in，免费、无需 API Key）──────────────────────
def get_weather(location: str) -> str:
    """调用 wttr.in 获取真实天气数据"""
    try:
        url = f"https://wttr.in/{requests.utils.quote(location)}?format=j1"
        resp = requests.get(url, timeout=10)
        resp.raise_for_status()
        data = resp.json()

        current = data["current_condition"][0]
        temp_c   = current["temp_C"]
        feels_c  = current["FeelsLikeC"]
        humidity = current["humidity"]
        desc     = current["weatherDesc"][0]["value"]
        wind_kmh = current["windspeedKmph"]

        return (
            f"天气：{desc}，气温：{temp_c}°C（体感 {feels_c}°C），"
            f"湿度：{humidity}%，风速：{wind_kmh} km/h"
        )
    except Exception as e:
        return f"获取天气失败：{e}"


# ── Tool 声明 ────────────────────────────────────────────────────────────────
tools = [
    {
        "type": "function",
        "function": {
            "name": "get_weather",
            "description": "Get real-time weather for a given location.",
            "parameters": {
                "type": "object",
                "properties": {
                    "location": {
                        "type": "string",
                        "description": "City name, e.g. 'Hangzhou' or 'New York'",
                    }
                },
                "required": ["location"],
            },
        },
    }
]

# 工具名 → 函数 映射表（便于扩展多工具）
tool_registry = {
    "get_weather": get_weather,
}


def send_messages(messages):
    response = client.chat.completions.create(
        model="deepseek-chat",
        messages=messages,
        tools=tools,
    )
    return response.choices[0].message


# ── 主流程 ───────────────────────────────────────────────────────────────────
user_query = "长春今天天气怎么样？"
messages = [{"role": "user", "content": user_query}]
print(f"User>\t {user_query}\n")

# ── 第一轮：模型决定调用哪个工具 ─────────────────────────────────────────────
message = send_messages(messages)
messages.append(message)

# 支持模型在一次回复中请求多个工具调用
for tool_call in message.tool_calls:  # 遍历所有工具调用请求
    func_name = tool_call.function.name
    func_args = json.loads(tool_call.function.arguments)

    print(f"[Tool Call] {func_name}({func_args})")
    result = tool_registry[func_name](**func_args)
    print(f"[Tool Result] {result}\n")

    messages.append({
        "role": "tool",
        "tool_call_id": tool_call.id,
        "content": result,
    })

# ── 第二轮：模型基于真实天气数据生成最终回复 ──────────────────────────────────
final = send_messages(messages)
print(f"Model>\t {final.content}")
