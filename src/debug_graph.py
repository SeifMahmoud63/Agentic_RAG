import sys
import os

# Add src to path
sys.path.append(os.path.join(os.getcwd(), "src"))

from agent.graph import graph
from langchain_core.messages import HumanMessage

def debug_graph():
    query = "What is the email of Youssab Kamal? Check my local documents please."
    print(f"--- Testing Graph with: '{query}' ---\n")
    
    config = {"configurable": {"thread_id": "test_thread"}}
    input_state = {"messages": [HumanMessage(content=query)]}
    
    try:
        # Run graph
        final_state = graph.invoke(input_state, config=config)
        
        print("\n--- Message Chain ---")
        for i, msg in enumerate(final_state["messages"]):
            type_name = type(msg).__name__
            content_preview = str(msg.content)[:100]
            print(f"[{i}] {type_name}: {content_preview}...")
            if hasattr(msg, "tool_calls") and msg.tool_calls:
                print(f"    Tool Calls: {msg.tool_calls}")

        print("\n--- Final Answer ---")
        print(final_state["messages"][-1].content)
        
    except Exception as e:
        print(f"Graph execution failed: {e}")

if __name__ == "__main__":
    debug_graph()
