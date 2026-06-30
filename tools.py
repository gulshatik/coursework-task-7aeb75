import random
import re

def unreliable_tool(task: str) -> str:
    """
    Simulates a tool that sometimes fails.
    With ~30% probability it raises ValueError.
    Otherwise, if the task is a simple arithmetic expression like "2+2",
    returns the evaluated result as a string. For any other input,
    returns a placeholder string.
    """
    # 30% chance to fail
    if random.random() < 0.3:
        raise ValueError("Tool encountered an unexpected error.")

    # Simple regex to detect expressions like "a+b"
    match = re.fullmatch(r"\s*(\d+)\s*\+\s*(\d+)\s*", task)
    if match:
        a, b = int(match.group(1)), int(match.group(2))
        return str(a + b)

    # Default placeholder
    return "result_placeholder"
