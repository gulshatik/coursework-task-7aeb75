from typing import TypedDict, Dict, Any
import os

# Import the LLM and tools
from langchain_openai import ChatOpenAI
from langgraph.graph import StateGraph, START, END
from langgraph.checkpoint.memory import InMemorySaver
from langgraph.prebuilt import create_react_agent

# Import the unreliable tool
from tools import unreliable_tool

# Define the state structure
class AgentState(TypedDict):
    task: str
    result: str
    attempts: int
    status: str  # pending | success | failed | max_attempts
    error: str | None
    max_attempts: int


def build_agent_graph() -> StateGraph[AgentState]:
    """
    Build a LangGraph that executes a task, verifies the result,
    and retries on failure up to `max_attempts`.
    """

    # LLM configuration (use environment variables)
    llm = ChatOpenAI(
        base_url=os.getenv("BROJS_BASE_URL", "https://llm.brojs.ru/v1"),
        api_key=os.getenv("BROJS_PAT_TOKEN"),
        model="openai/gpt-oss-20b",
        temperature=0.1,
    )

    # Define the tool list
    tools = [unreliable_tool]

    # Create a simple react agent that will be used in the graph
    react_agent = create_react_agent(llm, tools)

    # Helper functions for each node

    def execute_task(state: AgentState) -> AgentState:
        """Run the unreliable tool and update state."""
        try:
            result = unreliable_tool(state["task"])
            state.update(
                {
                    "result": result,
                    "error": None,
                    "status": "pending",
                }
            )
        except Exception as e:
            state.update(
                {
                    "result": "",
                    "error": str(e),
                    "status": "failed",
                }
            )
        return state

    def verify_result(state: AgentState) -> AgentState:
        """Ask the LLM to judge if the result is correct."""
        prompt = (
            f"Task: {state['task']}\n"
            f"Result: {state['result']}\n\n"
            "Answer with only one word: 'success' or 'failed'."
        )
        response = llm.invoke(prompt).content.strip().lower()
        if response == "success":
            state["status"] = "success"
        else:
            state["status"] = "failed"
        return state

    def handle_error(state: AgentState) -> AgentState:
        """Increment attempts and prepare for retry."""
        state["attempts"] += 1
        # If we hit max_attempts, mark as such
        if state["attempts"] >= state["max_attempts"]:
            state["status"] = "max_attempts"
        else:
            state["status"] = "pending"
        return state

    # Build the graph
    graph = StateGraph(AgentState)

    # Add nodes
    graph.add_node("execute_task", execute_task)
    graph.add_node("verify_result", verify_result)
    graph.add_node("handle_error", handle_error)

    # Define edges
    graph.set_entry_point("execute_task")
    graph.add_edge("execute_task", "verify_result")

    # Conditional transitions from verify_result
    def check_success(state: AgentState) -> str:
        return "END" if state["status"] == "success" else (
            "handle_error"
            if state["attempts"] < state["max_attempts"]
            else "END"
        )

    graph.add_conditional_edges(
        "verify_result",
        check_success,
        {
            "END": END,
            "handle_error": "handle_error",
        },
    )

    # From handle_error back to execute_task if not maxed
    def retry_condition(state: AgentState) -> str:
        return "execute_task" if state["status"] == "pending" else "END"

    graph.add_conditional_edges(
        "handle_error",
        retry_condition,
        {
            "execute_task": "execute_task",
            "END": END,
        },
    )

    # Compile the graph
    return graph.compile()
