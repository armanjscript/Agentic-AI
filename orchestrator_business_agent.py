from langgraph.graph import StateGraph, START, END
from typing_extensions import TypedDict
from pydantic import BaseModel, Field
from langchain_ollama import ChatOllama
from IPython.display import Image, display
from typing import Annotated, List
from langgraph.types import Send
from langchain_core.messages import HumanMessage, SystemMessage
import operator


#Initializing LLM
llm = ChatOllama(
    model="qwen2.5:latest",
    temperature=0.7
)


#Schema for structured output to use in planning
class Section(BaseModel):
    name: str = Field(description="Name for this section of the report")
    description: str = Field(description="Brief overview of the main topics and concepts to be converted in this section.")


class Sections(BaseModel):
    sections: List[Section]=Field(description="Sections of the report.")


#Augment the LLM with schema for structured output
planner = llm.with_structured_output(Sections, method="json_mode")


#Graph State
class State(TypedDict):
    topic: str #Report topic
    sections: list[Section] #List of report sections
    completed_sections: Annotated[
        list, operator.add
    ] #All workers write to this key in parallel
    final_report: str #Final report
    

#Worker state
class WorkerState(TypedDict):
    section: Section
    completed_sections: Annotated[list, operator.add]
    
    
#Nodes
def orchestrator(state: State):
    """Orchestrator that generates a plan for the report"""
    # Generate the plan using the LLM
    report_sections = planner.invoke([
        SystemMessage(content="Generate a plan for the report. The output must be a JSON object containing a 'sections' key, which maps to a list of objects, each with a 'name' and 'description'."),
        HumanMessage(content=f"Here is the report topic: {state['topic']}")
    ])

    # The 'planner' returns an object of type 'Sections' which expects a list.
    # If the model returns a dictionary, we need to convert it.
    # Check if the returned object has the 'sections' attribute and if it's a list.
    if hasattr(report_sections, 'sections') and isinstance(report_sections.sections, list):
        final_sections = report_sections.sections
    else:
        # If the output is a dictionary (or some other format), try to parse it.
        # Based on the error, the model returns a dict with a 'sections' key that is a dict.
        # We need to convert that dict of sections into a list of Section objects.
        if isinstance(report_sections, dict) and 'sections' in report_sections:
            sections_dict = report_sections['sections']
            # Convert the dictionary values into a list of Section objects
            # Assuming each value in the dict has 'name' and 'description' keys
            final_sections = []
            for section_name, section_content in sections_dict.items():
                # If section_content is a dict, try to extract name and description
                if isinstance(section_content, dict):
                    name = section_name
                    description = section_content.get('description', '')
                    final_sections.append(Section(name=name, description=description))
                else:
                    # Fallback if content is not as expected
                    final_sections.append(Section(name=section_name, description=str(section_content)))
        else:
            # Ultimate fallback: use the original object if it's already a list
            final_sections = report_sections if isinstance(report_sections, list) else []

    print("Report sections", final_sections)
    return {"sections": final_sections}

def llm_call(state: WorkerState):
    """Worker writes a section of the report"""
    
    #Generate Sections
    section = llm.invoke([
       SystemMessage(
                content="Write a report section following the provided name and description. Include no preamble for each section. Use markdown formatting."
        ),
        HumanMessage(
                content=f"Here is the section name: {state['section'].name} and description: {state['section'].description}"
        ),
    ])
    
    # Write the updated section to completed sections
    return {"completed_sections": [section.content]}


def synthesizer(state: State):
    """Synthesize full report from sections"""

    # List of completed sections
    completed_sections = state["completed_sections"]

    # Format completed section to str to use as context for final sections
    completed_report_sections = "\n\n---\n\n".join(completed_sections)

    return {"final_report": completed_report_sections}


# Conditional edge function to create llm_call workers that each write a section of the report
def assign_workers(state: State):
    """Assign a worker to each section in the plan"""

    # Kick off section writing in parallel via Send() API
    return [Send("llm_call", {"section": s}) for s in state["sections"]]


#Build workflow
orchestrator_worker_builder=StateGraph(State)

#Add the nodes
orchestrator_worker_builder.add_node("orchestrator",orchestrator)
orchestrator_worker_builder.add_node("llm_call",llm_call)
orchestrator_worker_builder.add_node("synthesizer",synthesizer)

#Add edges to connect nodes
orchestrator_worker_builder.add_edge(START,"orchestrator")
orchestrator_worker_builder.add_conditional_edges(
    "orchestrator",assign_workers,["llm_call"]
)
orchestrator_worker_builder.add_edge("llm_call","synthesizer")
orchestrator_worker_builder.add_edge("synthesizer",END)
#compilte the workflow
orchestrator_worker=orchestrator_worker_builder.compile()

#show the workflow
# display(Image(orchestrator_worker.get_graph().draw_mermaid_png()))


#Invoke 
state=orchestrator_worker.invoke({"topic":"Create a report on How to increase sales for my Agentic AI Rag customer care  for my company which hels lawyers to solve cases"})
print(state["final_report"])
