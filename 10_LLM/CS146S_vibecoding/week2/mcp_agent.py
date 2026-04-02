import asyncio
import json
import os
from pathlib import Path
from typing import Any, Dict, List

from dotenv import load_dotenv
from fastmcp.client import Client
from openai import OpenAI

load_dotenv()

MODEL_NAME = os.environ.get("DEEPSEEK_MODEL", "deepseek-chat")
MAX_TOOL_STEPS = 10
DEBUG_LOGS_ENABLED = os.environ.get("MCP_AGENT_DEBUG", "1") != "0"
MAX_DEBUG_TEXT_LENGTH = 1000

SYSTEM_PROMPT = """
You are a coding assistant whose goal is to help solve coding tasks.
Use the available tools when they are needed.
Prefer inspecting files and directories before editing them.
When you have enough information, answer clearly and concisely.
""".strip()

YOU_COLOR = "\u001b[94m"
ASSISTANT_COLOR = "\u001b[93m"
DEBUG_COLOR = "\u001b[90m"
RESET_COLOR = "\u001b[0m"


def create_llm_client() -> OpenAI:
    api_key = os.environ.get("DEEPSEEK_API_KEY")
    if not api_key:
        raise RuntimeError("DEEPSEEK_API_KEY is not set")
    return OpenAI(api_key=api_key, base_url="https://api.deepseek.com")


def get_server_path() -> Path:
    return Path(__file__).with_name("simple_mcp.py")


def debug_log(message: str) -> None:
    if DEBUG_LOGS_ENABLED:
        print(f"{DEBUG_COLOR}[debug]{RESET_COLOR} {message}")


def preview_text(value: str, max_length: int = MAX_DEBUG_TEXT_LENGTH) -> str:
    compact = value.replace("\n", "\\n")
    # if len(compact) <= max_length:
        # return compact
    # return compact[:max_length] + "..."
    return compact


def mcp_tools_to_openai_tools(mcp_tools: List[Any]) -> List[Dict[str, Any]]:
    openai_tools = []
    for tool in mcp_tools:
        openai_tools.append(
            {
                "type": "function",
                "function": {
                    "name": tool.name,
                    "description": tool.description or "",
                    "parameters": tool.inputSchema,
                },
            }
        )
    return openai_tools


def serialize_tool_result(result: Any) -> str:
    if getattr(result, "structured_content", None) is not None:
        return json.dumps(result.structured_content, ensure_ascii=False)
    if getattr(result, "data", None) is not None:
        return json.dumps(result.data, ensure_ascii=False)

    content_items = []
    for item in getattr(result, "content", []) or []:
        text = getattr(item, "text", None)
        if text is not None:
            content_items.append(text)

    if content_items:
        return "\n".join(content_items)
    return json.dumps({"result": str(result)}, ensure_ascii=False)


def build_assistant_message(message: Any) -> Dict[str, Any]:
    assistant_message: Dict[str, Any] = {
        "role": "assistant",
        "content": message.content or "",
    }
    if message.tool_calls:
        assistant_message["tool_calls"] = [
            {
                "id": tool_call.id,
                "type": tool_call.type,
                "function": {
                    "name": tool_call.function.name,
                    "arguments": tool_call.function.arguments,
                },
            }
            for tool_call in message.tool_calls
        ]
    return assistant_message


async def run_single_turn(
    llm_client: OpenAI,
    mcp_client: Client,
    conversation: List[Dict[str, Any]],
    openai_tools: List[Dict[str, Any]],
) -> str:
    for step in range(MAX_TOOL_STEPS):
        debug_log(f"LLM step {step + 1}/{MAX_TOOL_STEPS}")
        print(f"(conversation so far: {conversation})")
        response = llm_client.chat.completions.create(
            model=MODEL_NAME,
            messages=conversation,
            tools=openai_tools,
            tool_choice="auto",
            max_completion_tokens=2000,
        )
        message = response.choices[0].message
        print(f"{ASSISTANT_COLOR}LLM RAW response:{RESET_COLOR} {message}")
        conversation.append(build_assistant_message(message))

        if not message.tool_calls:
            debug_log(f"Final answer: {preview_text(message.content or '')}")
            return message.content or ""

        for tool_call in message.tool_calls:
            debug_log(
                "Tool selected: "
                f"{tool_call.function.name} args={preview_text(tool_call.function.arguments or '{}')}"
            )
            try:
                arguments = json.loads(tool_call.function.arguments or "{}")
            except json.JSONDecodeError as exc:
                tool_output = json.dumps({"error": f"Invalid tool arguments: {exc}"})
                debug_log(f"Tool argument parsing failed: {tool_output}")
            else:
                try:
                    result = await mcp_client.call_tool(tool_call.function.name, arguments)
                    tool_output = serialize_tool_result(result)
                    debug_log(f"Tool result: {preview_text(tool_output)}")
                except Exception as exc:
                    tool_output = json.dumps({"error": str(exc)}, ensure_ascii=False)
                    debug_log(f"Tool execution failed: {tool_output}")

            conversation.append(
                {
                    "role": "tool",
                    "tool_call_id": tool_call.id,
                    "name": tool_call.function.name,
                    "content": tool_output,
                }
            )

    return "Stopped after reaching the maximum number of tool steps."


async def run_coding_agent_loop() -> None:
    llm_client = create_llm_client()
    server_path = get_server_path()

    async with Client(server_path) as mcp_client:  # server_path告诉MCP客户端连接到哪个MCP服务器，这里我们直接把simple_mcp.py作为服务器
        mcp_tools = await mcp_client.list_tools()
        openai_tools = mcp_tools_to_openai_tools(mcp_tools)

        print(f"Loaded {len(mcp_tools)} MCP tools from {server_path}")
        debug_log("Available tools: " + ", ".join(tool["function"]["name"] for tool in openai_tools))
        debug_log("Set MCP_AGENT_DEBUG=0 to disable debug logs")

        conversation: List[Dict[str, Any]] = [
            {
                "role": "system",
                "content": SYSTEM_PROMPT,
            }
        ]

        while True:
            try:
                user_input = input(f"{YOU_COLOR}You:{RESET_COLOR} ").strip()
            except (KeyboardInterrupt, EOFError):
                print()
                break

            if not user_input:
                continue

            debug_log(f"User input: {preview_text(user_input)}")
            conversation.append({"role": "user", "content": user_input})
            answer = await run_single_turn(llm_client, mcp_client, conversation, openai_tools)
            print(f"{ASSISTANT_COLOR}Assistant:{RESET_COLOR} {answer}")


if __name__ == "__main__":
    asyncio.run(run_coding_agent_loop())