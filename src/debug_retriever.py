import sys
import os

# Add src to path
sys.path.append(os.path.join(os.getcwd(), "src"))

from retriever.RetrieveChunks import advanced_retrieve
from agent.tools import Search_Local_Documents
from helpers.config import get_settings

def debug():
    query = "What are the core features?" 
    print(f"--- Debugging retrieval for: '{query}' ---")
    
    try:
        results = advanced_retrieve(query=query)
        print(f"Direct Retrieval Count: {len(results)}")
        if results:
            print(f"Sample Content: {results[0].page_content[:100]}...")
        else:
            print("Direct Retrieval returned NONE.")
    except Exception as e:
        print(f"Direct Retrieval CRASHED: {e}")

    try:
        tool_output = Search_Local_Documents.invoke({"query": query})
        print(f"\nTool Output Snapshot (first 200 chars):")
        print(tool_output[:200])
    except Exception as e:
        print(f"Tool Invoke CRASHED: {e}")

if __name__ == "__main__":
    debug()
