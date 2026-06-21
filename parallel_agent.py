from langgraph.graph import StateGraph, START, END
from typing_extensions import TypedDict
from langchain_ollama import ChatOllama
from IPython.display import Image, display


# Graph state
class State(TypedDict):
    topic: str
    tagline: str
    usecases: str
    marketingidea: str
    combined_output: str


#Initializing LLM
llm = ChatOllama(  # Changed from OllamaLLM to ChatOllama
    model="qwen2.5:latest",
    temperature=0.7
)



# Nodes
def call_llm_1(state: State):
    """First LLM call to generate initial tag_line"""

    msg = llm.invoke(f"Write a tag_line about {state['topic']}")
    return {"tagline": msg.content}


def call_llm_2(state: State):
    """Second LLM call to generate use case"""

    msg = llm.invoke(f"Write a use cases about {state['topic']}")
    return {"usecases": msg.content}


def call_llm_3(state: State):
    """Third LLM call to generate marketing idea"""

    msg = llm.invoke(f"Write a marketing idea about {state['topic']}")
    return {"marketingidea": msg.content}


def aggregator(state: State):
    """Combine the tagline,use cases and marketing idea into a single output"""

    combined = f"Here's a tagline , use cases,  and marketing about {state['topic']}!\n\n"
    combined += f"tagline:\n{state['tagline']}\n\n"
    combined += f"usecases:\n{state['usecases']}\n\n"
    combined += f"marketingidea:\n{state['marketingidea']}"
    return {"combined_output": combined}


# Build workflow
parallel_builder = StateGraph(State)

# Add nodes
parallel_builder.add_node("call_llm_1", call_llm_1)
parallel_builder.add_node("call_llm_2", call_llm_2)
parallel_builder.add_node("call_llm_3", call_llm_3)
parallel_builder.add_node("aggregator", aggregator)

# Add edges to connect nodes
parallel_builder.add_edge(START, "call_llm_1")
parallel_builder.add_edge(START, "call_llm_2")
parallel_builder.add_edge(START, "call_llm_3")
parallel_builder.add_edge("call_llm_1", "aggregator")
parallel_builder.add_edge("call_llm_2", "aggregator")
parallel_builder.add_edge("call_llm_3", "aggregator")
parallel_builder.add_edge("aggregator", END)
parallel_workflow = parallel_builder.compile()

# Show workflow
# display(Image(parallel_workflow.get_graph().draw_mermaid_png()))

# Invoke
state = parallel_workflow.invoke({"topic": "samsung AI smart phone"})
print(state["combined_output"])