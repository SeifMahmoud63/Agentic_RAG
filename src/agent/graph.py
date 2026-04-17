from typing import Annotated, TypedDict, Union
from langgraph.graph import StateGraph, START, END, MessagesState
from langgraph.prebuilt import ToolNode, tools_condition
from langgraph.checkpoint.memory import MemorySaver
from langchain_core.messages import SystemMessage, HumanMessage
import os

from llm.llm import get_llm
from .tools import get_tools

def create_agent_graph():
    tools = get_tools()
    llm = get_llm()
    llm_with_tools = llm.bind_tools(tools)
    
    system_prompt_path = os.path.join(os.path.dirname(__file__), "SystemPrompt.txt")
    with open(system_prompt_path, "r", encoding="utf-8") as f:
        system_prompt = f.read()

    def call_model(state: MessagesState):
        messages = state["messages"]
        
        if not any(isinstance(m, SystemMessage) for m in messages):
            current_messages = [SystemMessage(content=system_prompt)] + messages
        else:
            current_messages = messages

        response = llm_with_tools.invoke(current_messages)
        

        if hasattr(response, "tool_calls") and response.tool_calls:
            valid_calls = []
            for tc in response.tool_calls:
                if tc.get("name"):
                    valid_calls.append(tc)
            response.tool_calls = valid_calls

        print(f"--- [AGENT THINKING] ---\n{response.content}\n-----------------------")
        
        return {"messages": [response]}

    tool_node = ToolNode(tools)

    workflow = StateGraph(MessagesState)
    
    workflow.add_node("agent", call_model)
    workflow.add_node("tools", tool_node)
    
    workflow.add_edge(START, "agent")
    workflow.add_conditional_edges("agent", tools_condition)
    workflow.add_edge("tools", "agent")

    memory = MemorySaver()

    return workflow.compile(checkpointer=memory)


graph = create_agent_graph()
