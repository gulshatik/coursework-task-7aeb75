#!/usr/bin/env python3
"""
Self‑correcting LangGraph agent demo.

The agent executes a task via an unreliable tool, verifies the result with an LLM,
and retries until success or max_attempts is reached.
"""

import os
import random
from typing import TypedDict

# ────────────────────── Environment & LLM setup ──────────────────────
from dotenv import load_dotenv
load_dotenv()

from langchain_openai import ChatOpenAI

llm = ChatOpenAI(
    base_url="https://llm.brojs.ru/v1",
    api_key=os.getenv("BROJS_PAT_TOKEN"),
    model="openai/gpt-oss-20b",
    temperature=0.1,
)

# ────────────────────── State definition ──────────────────────
class AgentState(TypedDict):
    task: str
    result: str
    attempts: int
    status: str  # pending | success | failed | max_attempts
    error: str | None
    max_attempts: int

# ────────────────────── Unreliable tool ──────────────────────
def unreliable_tool(task: str) -> str:
    """
    Simulates a flaky tool. With ~30% probability it raises ValueError.
    Otherwise returns the result of evaluating simple arithmetic expressions.
    """
    if random.random() < 0.3:
        raise ValueError("Tool failure (simulated).")
    # Very naive evaluator: only supports "число + число" or "число - число"
    try:
        left, op, right = task.split()
        left_val = float(left)
        right_val = float(right)
        if op == "+":
            return str(left_val + right_val)
        elif op == "-":
            return str(left_val - right_val)
        else:
            raise ValueError(f"Unsupported operator: {op}")
    except Exception as exc:
        raise ValueError(f"Invalid task format: {task}") from exc

# ────────────────────── Node implementations ──────────────────────
def execute_task(state: AgentState) -> AgentState:
    """
    Calls the unreliable tool and stores its output.
    On exception, records error and sets status to 'failed'.
    """
    try:
        result = unreliable_tool(state["task"])
        state.update(
            result=result,
            status="pending",
            error=None,
        )
    except Exception as exc:
        state.update(
            result="",
            status="failed",
            error=str(exc),
        )
    return state

def verify_result(state: AgentState) -> AgentState:
    """
    Uses the LLM to judge whether the tool output is correct.
    The prompt forces the model to answer only 'success' or 'failed'.
    """
    if state["status"] != "pending":
        # Skip verification if previous step failed
        return state

    prompt = (
        f"Task: {state['task']}\n"
        f"Result from tool: {state['result']}\n\n"
        "Is the result correct? Answer only 'success' or 'failed'."
    )
    response = llm.invoke([{"role": "user", "content": prompt}])
    verdict = response.content.strip().lower()
    if verdict == "success":
        state["status"] = "success"
    else:
        state["status"] = "failed"
    return state

def handle_error(state: AgentState) -> AgentState:
    """
    Increments attempts and prepares for retry.
    If max_attempts reached, sets status to 'max_attempts'.
    """
    state["attempts"] += 1
    if state["attempts"] >= state["max_attempts"]:
        state["status"] = "max_attempts"
    else:
        # Reset error and keep status pending for retry
        state["error"] = None
        state["status"] = "pending"
    return state

# ────────────────────── Graph construction ──────────────────────
from langgraph.graph import StateGraph, START, END

builder = StateGraph(AgentState)

# Add nodes
builder.add_node("execute_task", execute_task)
builder.add_node("verify_result", verify_result)
builder.add_node("handle_error", handle_error)

# Define edges
builder.add_edge(START, "execute_task")
builder.add_edge("execute_task", "verify_result")

def transition(state: AgentState):
    """
    Conditional edge after verification.
    """
    if state["status"] == "success":
        return END
    if state["status"] == "failed" and state["attempts"] < state["max_attempts"]:
        return "handle_error"
    if state["status"] in ("failed", "max_attempts"):
        return END

builder.add_conditional_edges("verify_result", transition)
builder.add_edge("handle_error", "execute_task")

graph = builder.compile()

# ────────────────────── Demo runner ──────────────────────
def run_demo(task: str, max_attempts: int = 5):
    """
    Runs the graph until completion and prints progress.
    """
    # Initial state
    state: AgentState = {
        "task": task,
        "result": "",
        "attempts": 0,
        "status": "pending",
        "error": None,
        "max_attempts": max_attempts,
    }

    print(f"Задача: {task}")
    while True:
        state = graph.invoke(state)
        attempt_num = state["attempts"] + (1 if state["status"] == "pending" else 0)
        if state["error"]:
            print(
                f"Попытка {attempt_num}: Error → verify: failed\n"
                f"  Ошибка: {state['error']}"
            )
        elif state["status"] == "success":
            print(f"Попытка {attempt_num}: результат {state['result']} -> verify: success")
            break
        elif state["status"] == "max_attempts":
            print(
                f"Попытка {attempt_num}: достигнут лимит попыток ({max_attempts})."
            )
            break
        else:
            # status pending after retry preparation
            continue

    final_status = state["status"]
    attempts_used = state["attempts"] + (1 if final_status == "success" else 0)
    print(f"\nИтог: {final_status} за {attempts_used} попытки{'и' if attempts_used>1 else ''}")

if __name__ == "__main__":
    # Demo task: compute 2+2
    run_demo("2 + 2")
