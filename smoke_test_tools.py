"""Smoke test: confirm Qwen returns a tool_calls block for a trivial dummy tool.

Run: uv run smoke_test_tools.py
"""

import json

from qwen_client import call

DUMMY_TOOL = {
    "type": "function",
    "function": {
        "name": "get_weather",
        "description": "Get the current weather for a city.",
        "parameters": {
            "type": "object",
            "properties": {
                "city": {"type": "string", "description": "City name, e.g. 'Beijing'"},
            },
            "required": ["city"],
        },
    },
}


def main():
    messages = [{"role": "user", "content": "What's the weather like in Hangzhou right now?"}]
    response = call(model="qwen-turbo", messages=messages, tools=[DUMMY_TOOL], tool_choice="auto")

    choice = response.choices[0]
    tool_calls = choice.message.tool_calls

    if not tool_calls:
        print("FAIL: no tool_calls returned")
        print("message content:", choice.message.content)
        raise SystemExit(1)

    print("PASS: model returned tool_calls")
    for tc in tool_calls:
        print(f"  id={tc.id} name={tc.function.name} arguments={tc.function.arguments}")
        args = json.loads(tc.function.arguments)
        assert "city" in args, "expected 'city' argument in tool call"

    print("\nfull tool_calls block:")
    print(tool_calls)


if __name__ == "__main__":
    main()
