import sys
from RagFramework.agent import ask, get_agent


def handle_query(user_query: str):
    user_query = user_query.strip()
    if not user_query:
        return

    response = ask(user_query)
    print("\n--- Assistant ---")
    print(response)


def run_chat():
    print("[Initializing agent...]")
    get_agent()

    print("\nVIPER Mission Planner — LLM Assistant (DSPy ReAct)")
    print("Ask about the trajectory, drill sites, or maps. Examples:")
    print("  - 'Run the simulation'")
    print("  - 'Show the plot'")
    print("  - 'Add a POI at (30, 45)'")
    print("  - 'Use candidates 2, 5, 8'")
    print("  - 'Worst case scenario'")
    print("  - 'What are the axes of the plot?'")
    print("  - 'How does the planner pick drill sites?'")
    print("Type 'exit' to quit.\n")

    while True:
        try:
            user_query = input("\nYou: ")
        except (EOFError, KeyboardInterrupt):
            break
        if user_query.strip().lower() in {"exit", "quit"}:
            break
        handle_query(user_query)


if __name__ == '__main__':
    if len(sys.argv) > 1 and sys.argv[1] == '--query':
        query = ' '.join(sys.argv[2:])
        if query:
            handle_query(query)
        else:
            print("Usage: python -m RagFramework.chatbot --query \"your question\"")
    else:
        run_chat()
