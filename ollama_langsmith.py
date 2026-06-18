from langchain_ollama import ChatOllama  # Change this import
from langgraph.graph import MessageGraph, START, END
from langgraph.graph.state import StateGraph
from langgraph.prebuilt import ToolNode
from typing_extensions import TypedDict
from langchain_core.messages import BaseMessage
from typing import Annotated
from langchain_core.runnables import RunnableConfig
from langchain_core.tools import tool
from langgraph.graph.message import add_messages
from dotenv import load_dotenv
import os

load_dotenv()

#Langsmith API Key
os.environ["LANGSMITH_API_KEY"] = os.getenv("LANGSMITH_API_KEY")


class State(TypedDict):
    messages: Annotated[list[BaseMessage], add_messages]
    


#Initializing LLM
model = ChatOllama(  # Changed from OllamaLLM to ChatOllama
    model="qwen2.5:latest",
    temperature=0.0
)


def make_default_graph():
    """Make a simple LLM agent"""
    graph_workflow = StateGraph(State)
    
    def call_model(state: State):
        return {"messages": [model.invoke(state["messages"])]}
    
    graph_workflow.add_node("agent", call_model)
    graph_workflow.add_edge(START, "agent")
    graph_workflow.add_edge("agent", END)
    
    agent = graph_workflow.compile()
    return agent

agent = make_default_graph()



