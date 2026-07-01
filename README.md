# Экзамен: Самокорректирующийся агент

## Описание задания

В этом практическом задании вы реализуете **LangGraph**‑агента, который после выполнения задачи автоматически проверяет свой результат (LLM‑as‑judge) и при неудаче повторяет работу до тех пор, пока не получит статус `success` или не исчерпает предельное число попыток (`max_attempts`).  
В отличие от HITL‑варианта, подтверждение человеком заменяется автоматической самопроверкой.

---

## Стек технологий

- Python 3.10+
- `langgraph`
- `langchain-openai` **или** `langchain-ollama`

```bash
pip install langgraph langchain-openai langchain-ollama
```

> ⚠️ Если вы используете OpenAI, убедитесь, что переменная окружения `OPENAI_API_KEY` установлена.  
> Для Ollama достаточно запустить локальный сервер (`docker run -p 11434:11434 ollama/ollama`) и указать в коде модель `llama3`.

---

## Структура проекта

```
.
├── agent_state.py   # TypedDict для состояния графа
├── nodes.py         # Реализация узлов execute_task, verify_result, handle_error
├── tools.py         # unreliable_tool (30% вероятность ошибки)
├── graph.py         # Построение StateGraph с циклом retry
├── main.py          # CLI‑демо: ввод задачи и запуск графа
└── README.md        # Текущий файл
```

---

## 1. Состояние графа

```python
# agent_state.py
from typing import TypedDict, Literal, Optional

class AgentState(TypedDict):
    task: str                     # исходная задача
    result: str                   # результат выполнения задачи
    attempts: int                 # количество попыток
    status: Literal["pending", "success", "failed", "max_attempts"]
    error: Optional[str]          # сообщение об ошибке, если есть
    max_attempts: int             # лимит попыток (по умолчанию 3)
```

---

## 2. Узлы

| Узел | Описание |
|------|----------|
| `execute_task` | Вызывает инструмент `unreliable_tool`, сохраняет результат в `state.result`. При исключении заполняет `state.error` и ставит статус `failed`. |
| `verify_result` | Отправляет LLM запрос: «Is the result correct? Answer only “success” or “failed”. Result: …». На основе ответа меняет `state.status`. |
| `handle_error` | Увеличивает счётчик `attempts`, очищает `error`, возвращает статус в `pending` для повторного запуска. |

---

## 3. Построение графа

```python
# graph.py
from langgraph.graph import StateGraph
from agent_state import AgentState
from nodes import execute_task, verify_result, handle_error

def build_graph() -> StateGraph:
    graph = StateGraph(AgentState)

    # Добавляем узлы
    graph.add_node("execute_task", execute_task)
    graph.add_node("verify_result", verify_result)
    graph.add_node("handle_error", handle_error)

    # Путь: start → execute_task → verify_result
    graph.set_entry_point("execute_task")
    graph.add_edge("execute_task", "verify_result")

    # Условные переходы после проверки результата
    graph.add_conditional_edges(
        "verify_result",
        lambda state: (
            "success" if state["status"] == "success"
            else ("max_attempts" if state["attempts"] >= state["max_attempts"]
                  else "handle_error")
        ),
        {
            "success": None,          # конец графа
            "max_attempts": None,     # конец графа
            "handle_error": "execute_task",
        },
    )

    return graph.compile()
```

> ⚡ `StateGraph` автоматически обрабатывает переходы и сохраняет состояние в памяти.

---

## 4. Тестовый инструмент

```python
# tools.py
import random
from langchain.tools import BaseTool

class UnreliableTool(BaseTool):
    name = "unreliable_tool"
    description = (
        "Выполняет арифметическое выражение, но с вероятностью 30% бросает ValueError."
    )

    def _run(self, expression: str) -> str:
        if random.random() < 0.3:
            raise ValueError("Инструмент случайно отказал")
        try:
            # Безопасный eval для простых выражений
            result = eval(expression, {"__builtins__": {}})
            return str(result)
        except Exception as e:
            raise ValueError(f"Невозможно вычислить: {e}")

unreliable_tool = UnreliableTool()
```

---

## 5. Узлы (полный код)

```python
# nodes.py
from typing import Dict, Any
from langchain_openai import ChatOpenAI
from agent_state import AgentState
from tools import unreliable_tool

llm = ChatOpenAI(temperature=0)  # стабильные ответы

def execute_task(state: AgentState) -> AgentState:
    try:
        result = unreliable_tool.invoke({"expression": state["task"]})
        state.update(
            result=result,
            status="pending",
            error=None
        )
    except Exception as e:
        state.update(
            result="",
            status="failed",
            error=str(e)
        )
    return state

def verify_result(state: AgentState) -> AgentState:
    prompt = (
        f"Is the result correct? Answer only 'success' or 'failed'.\n"
        f"Task: {state['task']}\nResult: {state['result']}"
    )
    response = llm.invoke(prompt).content.strip().lower()
    if "success" in response:
        state["status"] = "success"
    else:
        state["status"] = "failed"
    return state

def handle_error(state: AgentState) -> AgentState:
    state["attempts"] += 1
    state["error"] = None
    state["status"] = "pending"
    return state
```

---

## 6. CLI‑демо

```python
# main.py
import sys
from graph import build_graph
from agent_state import AgentState

def main():
    if len(sys.argv) > 1:
        task = " ".join(sys.argv[1:])
    else:
        task = input("Введите задачу (пример: 2+2): ")

    initial_state: AgentState = {
        "task": task,
        "result": "",
        "attempts": 0,
        "status": "pending",
        "error": None,
        "max_attempts": 3,
    }

    graph = build_graph()
    state = initial_state

    while True:
        state = graph.invoke(state)
        print(f"\nПопытка {state['attempts'] + 1}:")
        if state["status"] == "failed" and state["error"]:
            print(f"  Ошибка: {state['error']}")
        elif state["status"] == "success":
            print(f"  Результат: {state['result']}")
            break
        elif state["status"] == "max_attempts":
            print("  Достигнут лимит попыток. Завершено.")
            break

    print("\nИтог:")
    print(f"  Статус: {state['status']}")
    print(f"  Попытки: {state['attempts'] + 1}")

if __name__ == "__main__":
    main()
```

> Запуск:
> ```bash
> python main.py "2+2"
> ```

---

## 7. Как проверить

1. Установите зависимости (`pip install -r requirements.txt`).
2. Настройте переменную `OPENAI_API_KEY` (или используйте Ollama).
3. Выполните:
   ```bash
   python main.py "2+2"
   ```
4. Вы увидите вывод с номером попытки, ошибками и финальным статусом.

---

## 8. Критерии зачёта

| Компонент | Проверяется |
|-----------|-------------|
| **AgentState** | Полный TypedDict |
| **Граф** | `StateGraph` с правильными переходами |
| **Самопроверка** | `verify_result` использует LLM и возвращает только `success`/`failed` |
| **Retry** | Цикл до `max_attempts`, корректные обновления состояния |
| **Тестовый инструмент** | 30% вероятность ошибки |
| **CLI‑демо** | Принимает задачу, выводит попытки и финальный статус |

Если все пункты выполнены – студент получает зачёт.

---

## 9. Дополнительные ресурсы

- [LangGraph Docs](https://langchain-ai.github.io/langgraph/)
- [LLM-as-Judge](https://github.com/hwchase17/langchain/tree/main/examples/llm_as_judge)
- [Checkpointing в LangGraph](https://langchain-ai.github.io/langgraph/concepts/checkpointing/)

---

## 10. Автор

*Разработано как часть экзаменационного задания по теме «Самокорректирующийся агент».*

---
