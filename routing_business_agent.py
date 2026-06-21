from langgraph.graph import StateGraph, START, END
from typing_extensions import TypedDict, Literal
from langchain_ollama import ChatOllama
from pydantic import BaseModel, Field
from langchain_core.messages import HumanMessage, SystemMessage
from IPython.display import Image, display



class Route(BaseModel):
    step: Literal["tag_line", "use_cases", "marketing_idea"] = Field(
        default="use_cases", 
        description="The next step in the routing process"
    )


#Initializing LLM
llm = ChatOllama(
    model="qwen2.5:latest",
    temperature=0.7
)



#Augment the LLM with schema for structured output
router = llm.with_structured_output(Route, method='json_mode')


#State
class State(TypedDict):
    input: str
    decision: str
    output: str
    

#Nodes
def llm_call_1(state: State):
    """Write a tag_line based on the user input"""
    result = llm.invoke(state["input"])
    return {"output": result.content}


def llm_call_2(state: State):
    """Write the use_cases"""
    result = llm.invoke(state["input"])
    return {"output": result.content}



def llm_call_3(state: State):
    """Write the marketing idea"""
    result = llm.invoke(state["input"])
    return {"output": result.content}


def llm_call_router(state: State):
    """Route the input to the appropriate node"""
    decision = router.invoke(
        [SystemMessage(
            content="Route the input to one of these categories: tag_line, use_cases, or marketing_idea. Respond with a JSON object containing the 'step' field set to one of these exact values."
        ),
         HumanMessage(content=state["input"])
        ]
    )
    # Ensure decision.step is not None, default to "use_cases" if needed
    if decision.step is None:
        return {"decision": "use_cases"}  # Default fallback
    return {"decision": decision.step}


#Conditional edge function to route to the appropriate node
def route_decision(state: State):
    if state["decision"] == "tag_line":
        return "llm_call_1"
    elif state["decision"] == "use_cases":
        return "llm_call_2"
    elif state["decision"] == "marketing_idea":
        return "llm_call_3"

#Build workflow
router_builder = StateGraph(State)

#Add nodes
router_builder.add_node("llm_call_1", llm_call_1)
router_builder.add_node("llm_call_2", llm_call_2)
router_builder.add_node("llm_call_3", llm_call_3)
router_builder.add_node("llm_call_router", llm_call_router)


#Add edges
router_builder.add_edge(START, "llm_call_router")
router_builder.add_conditional_edges(
    "llm_call_router",
    route_decision,
    {
        "llm_call_1": "llm_call_1",
        "llm_call_2": "llm_call_2",
        "llm_call_3": "llm_call_3",
    }
)
router_builder.add_edge("llm_call_1", END)
router_builder.add_edge("llm_call_2", END)
router_builder.add_edge("llm_call_3", END)

#Compile workflow
router_workflow = router_builder.compile()

#Invoke
state = router_workflow.invoke({"input": "Create use cases for Apple Iphone with AI"})
print(state["output"])

print("======================================================>")

state = router_workflow.invoke({"input": "Create tag lines for Apple Iphone with AI"})
print(state["output"])