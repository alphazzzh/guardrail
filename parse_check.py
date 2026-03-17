import json

def parse(result):
    if "choices" in result and len(result["choices"]) > 0:
        msg = result["choices"][0].get("message", {})
        content = msg.get("content", "")
        # Try to extract reasoning if it's separated
        reasoning = msg.get("reasoning_content", "")
        if reasoning:
            # Reconstruct the <think> format so downstream parsing works
            return f"<think>\n{reasoning}\n</think>\n{content}"
        return content
    return result.get("content", "")

res = {
    "choices": [
        {
            "message": {
                "role": "assistant",
                "content": "Final answer",
                "reasoning_content": "This is the thinking process"
            }
        }
    ]
}

print(parse(res))
