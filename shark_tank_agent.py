from langgraph.graph import StateGraph, START, END
from typing_extensions import TypedDict, Literal
from langchain_ollama import ChatOllama
from pydantic import BaseModel, Field
from langchain_core.messages import HumanMessage, SystemMessage
from IPython.display import Image, display


#Initializing LLM
llm = ChatOllama(
    model="qwen2.5:latest",
    temperature=0.7
)


#Graph State
class State(TypedDict):
    pitch: str
    product: str
    judge_feedback: str
    pitch_good_or_not: str
    
    
#Schema for structured output to use in evaluation
class Feedback(BaseModel):
    grade: Literal["good pitch", "not good pitch"] = Field(description="Decide topic pitch is good for investment by shark tank judge")
    feedback: str = Field(
        description="If the pitch is not investable in current AI Market, provide feedback on how to make product AI marker fit"
    )
    

#Augment the LLM with schema for structured output
evaluator = llm.with_structured_output(Feedback, method="json_mode")


#Nodes

def llm_call_pitch_generator(state: State):
    """LLM generates a pitch based on the topic"""
    if state.get("judge_feedback"):
        msg = llm.invoke(
            f"Write a investment pitch for {state['product']} but take into account the judge feedback: {state['judge_feedback']}"
        )
    else:
        msg = llm.invoke(
            f"Write a investment pitch about {state['product']}"
        )
    return {"pitch": msg.content}


def llm_shark_tank_judge_pitch_evaluator(state: State):
    """LLM evaluates the pitch as a shark tank judge"""
    # Create a more explicit prompt that enforces the required format
    grade = evaluator.invoke(
        [SystemMessage(
            content="You are a Shark Tank judge evaluating a pitch. You must respond with a valid JSON object containing exactly two fields: 'grade' and 'feedback'. "
                    "The 'grade' field must be a string that is exactly either 'good pitch' or 'not good pitch'. "
                    "The 'feedback' field must be a single string containing your detailed evaluation and suggestions for improvement. "
                    "Do not include any other fields or nested objects in your response."
        ),
         HumanMessage(
            content=f"Evaluate this pitch for the product: {state['product']}.\n\nPitch content: {state['pitch']}"
        )]
    )
    return {"pitch_good_or_not": grade.grade, "judge_feedback": grade.feedback}


#Conditional edge function to route back to joke generator or end base upon feedback from the evaluator
def route_pitch(state: State):
    """Route back to pitch generator or end base upon the feedback from shark tank judge evaluator"""
    if state.get("pitch_good_or_not") == "good pitch":
        return "Accepted"
    elif state.get("pitch_good_or_not") == "not good pitch":
        return "Rejected +Feedback"
    

#Build workflow
shark_tank_builder=StateGraph(State)

#Add the nodes
shark_tank_builder.add_node("llm_call_pitch_generator",llm_call_pitch_generator)
shark_tank_builder.add_node("llm_shark_tank_judge_pitch_evaluator",llm_shark_tank_judge_pitch_evaluator)

#Add edges to connect nodes 
shark_tank_builder.add_edge(START,"llm_call_pitch_generator")
shark_tank_builder.add_edge("llm_call_pitch_generator","llm_shark_tank_judge_pitch_evaluator")

shark_tank_builder.add_conditional_edges(
    "llm_shark_tank_judge_pitch_evaluator",
    route_pitch,{
        # Name returned by route_joke:NAME OF NEXT NODE TO VISIT
        "Accepted":END,
        "Rejected +Feedback":"llm_call_pitch_generator"
    },
)

#Compile the workflow
shark_tank_workflow=shark_tank_builder.compile()

#show the workflow
# display(Image(shark_tank_workflow.get_graph().draw_mermaid_png()))


#Invoke
state=shark_tank_workflow.invoke({"product":"AI Image generator"})

print(state["judge_feedback"])