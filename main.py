import os
from agent import build_agent_graph, AgentState
from dotenv import load_dotenv

load_dotenv()

def run_demo() -> None:
    # Инициализируем начальное состояние
    initial_state: AgentState = {
        "task": "Вычисли 2+2",
        "result": "",
        "attempts": 0,
        "status": "pending",
        "error": None,
        "max_attempts": 5,
    }

    graph = build_agent_graph()
    final_state = graph.invoke(initial_state)

    print("\n=== Итог ===")
    if final_state["status"] == "success":
        print(f"Задача выполнена успешно за {final_state['attempts'] + 1} попыток.")
    else:
        print(f"Не удалось выполнить задачу после {final_state['attempts'] + 1} попыток. Статус: {final_state['status']}")

if __name__ == "__main__":
    run_demo()
