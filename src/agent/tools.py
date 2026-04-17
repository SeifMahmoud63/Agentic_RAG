from langchain_core.tools import tool
from langchain_community.retrievers.tavily_search_api import TavilySearchAPIRetriever
from retriever.RetrieveChunks import advanced_retrieve 
from helpers import config

tavily_retriever = TavilySearchAPIRetriever(k=config.get_settings().TOP_K_TAVILY)

@tool
def Search_Local_Documents(query: str) -> str:
    """Use this tool to search for specific facts, personal details, contact information, project data, or technical knowledge contained in the user's uploaded documents (PDFs, PPTXs, TXTs)."""
    print(f"--- [TOOL CALL] Search_Local_Documents called with query: '{query}' ---")
    results = advanced_retrieve(query=query)
    print(f"--- [TOOL CALL] Search_Local_Documents found {len(results)} results ---")
    if not results:
        return "No relevant information found in local documents."
    
    return "\n\n".join([f"Source: {doc.metadata.get('source', 'Unknown')}\n{doc.page_content}" for doc in results])

@tool
def Tavily_Tool(query: str) -> str:
    """Useful for general knowledge, current events, or information not found in local documents."""
    docs = tavily_retriever.invoke(query)
    if not docs:
        return "No results found on the internet."
    
    return "\n\n".join([f"Source: {doc.metadata.get('url', 'Internet')}\n{doc.page_content}" for doc in docs])

def get_tools():
    """Returns a list of tools available for the agent."""
    return [Search_Local_Documents, Tavily_Tool]
