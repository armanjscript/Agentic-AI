from langgraph.graph import StateGraph, START, END
from typing_extensions import TypedDict
from pydantic import BaseModel
from langchain_ollama import ChatOllama
from IPython.display import Image, display

class State(BaseModel):
    a: str

def node(state: State):
    return {"a": "Hello welcome to Agentic AI"}


#Build the state graph
builder = StateGraph(State)
builder.add_node("node", node)
builder.add_edge(START, "node")
builder.add_edge("node", END)

graph = builder.compile()


message = graph.invoke({"a": "Hello"})
print(message)


#Multiple Nodes-Run time validation
class OverallState(BaseModel):
    a: str
    
def bad_node(state: OverallState):
    #Exception raising
    # return {
    #     "a": 123
    # } 
    
    # return OverallState(a=123)
    return OverallState(a="Hello to Agentic AI")
    
def ok_node(state: OverallState):
    # return {
    #     "a": "goodbye"
    # }
    return OverallState(a="goodbye")

builder = StateGraph(OverallState)
builder.add_node("bad_node", bad_node)
builder.add_node("ok_node", ok_node)

builder.add_edge(START, "bad_node")
builder.add_edge("bad_node", "ok_node")
builder.add_edge("ok_node", END)

graph = builder.compile()

try:
    message = graph.invoke({"a": "Arman"})
    print(message)
except Exception as e:
    print("An exception was raised because bad_node `a` to an integer.")
    print(e)

#####################################

#Prompt Chaining

##Initializing LLM
llm = ChatOllama(  # Changed from OllamaLLM to ChatOllama
    model="qwen2.5:latest",
    temperature=0.7
)


##Graph State
class State(TypedDict):
    topic: str
    tagline: str
    improved_tagline: str
    final_tagline: str
    


## Nodes
def generate_tagline(state: State):
    """First LLM call to generate initial tagline"""
    msg = llm.invoke(f"Write a catchy tagline for the product: {state['topic']}")
    return {"tagline": msg.content}


def check_tagline_quality(state: State):
    """Gate function to check if the tagline has strong impact"""

    # Simple heuristic - check for presence of action word or emotion
    if any(word in state["tagline"].lower() for word in ["buy", "get", "love", "feel", "now"]):
        return "Pass"
    return "Fail"


def improve_tagline(state: State):
    """Second LLM call to improve tagline with creativity"""

    msg = llm.invoke(f"Make this tagline more creative and appealing: {state['tagline']}")
    return {"improved_tagline": msg.content}


def polish_tagline(state: State):
    """Third LLM call for final polish"""

    msg = llm.invoke(f"Add a memorable twist to this tagline: {state['improved_tagline']}")
    return {"final_tagline": msg.content}


## Build workflow
workflow = StateGraph(State)

## Add nodes
workflow.add_node("generate_tagline", generate_tagline)
workflow.add_node("improve_tagline", improve_tagline)
workflow.add_node("polish_tagline", polish_tagline)

## Add edges to connect nodes
workflow.add_edge(START, "generate_tagline")
workflow.add_conditional_edges("generate_tagline",check_tagline_quality,{"Fail":"improve_tagline","Pass":END})
workflow.add_edge("improve_tagline", "polish_tagline")
workflow.add_edge("polish_tagline", END)

## Compile
chain = workflow.compile()

## Show workflow
# display(Image(chain.get_graph().draw_mermaid_png()))


##invoke
state=chain.invoke({"topic":"Samsung Smart phone"})

print(state)




