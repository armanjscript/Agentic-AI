from langgraph.graph import StateGraph, START, END
from typing_extensions import TypedDict
from typing import List
from langchain_ollama import ChatOllama
from pydantic import BaseModel, Field
from langchain_core.messages import HumanMessage, SystemMessage
from IPython.display import Image, display
from langgraph.checkpoint.memory import MemorySaver

#Initializing LLM
llm = ChatOllama(
    model="qwen2.5:latest",
    temperature=0.7
)


class MedicalAnalyst(BaseModel):
    affiliation: str = Field(
        description="Primary qualification of the medical analyst"
    )
    name: str = Field(
        description="Name of the Medical Analyst"
    )
    role: str = Field(
        description="Role of the medical analyst in the context of the research topic"
    )
    description: str = Field(
        description="Description of the medical analyst research focus, concerns and research objectives to achieve"
    )

    @property
    def persona(self) -> str:
        return f"Name: {self.name} \n Role: {self.role} \n Affiliation: {self.affiliation} \n Description: {self.description}\n"


class Perspectives(BaseModel):
    medicalanalysts: List[MedicalAnalyst] = Field(
        description="Comprehensive list of medical analyst with their roles and affiliations"
    )
    

class GenerateMedicalAnalystState(TypedDict):
    researchTopic: str #Research topic
    max_medical_analysts: int #Number of analysts
    human_analyst_feedback: str #Human feedback
    medicalanalysts: List[MedicalAnalyst] #Medical analysts asking questions
    


# Update the instructions to be more explicit about the expected JSON structure
medical_analyst_instructions = """You are tasked with creating a set of Medical AI analyst personas. Follow these instructions carefully:

1. First, review the medical research topic:
{researchTopic}
        
2. Examine any medical editorial feedback or medical research that has been optionally provided to guide creation of the medical analysts that will work to generate medicines for the research or you can use your own knowledge for this task: 
        
{human_analyst_feedback}
    
3. Determine the most interesting research parts based upon documents and / or feedback above.
                    
4. Pick the top {max_medical_analysts} research parts.

5. Assign one analyst to each research part.

6. **CRITICAL: Your output MUST be a JSON object with a single key called "medicalanalysts" that maps to a LIST of analyst objects. Each analyst object must have exactly these fields: "name", "role", "affiliation", and "description". Do not use keys like "Analyst 1", "Analyst 2", etc.**

Example format:
{{
  "medicalanalysts": [
    {{
      "name": "Dr. Jane Smith",
      "role": "Vaccine Research Specialist",
      "affiliation": "CDC",
      "description": "Focuses on vaccine efficacy and distribution strategies"
    }}
  ]
}}"""


def create_medical_analysts(state: GenerateMedicalAnalystState):
    """Create medical analysts"""

    researchTopic = state["researchTopic"]
    max_medical_analysts = state["max_medical_analysts"]
    human_analyst_feedback = state.get('human_analyst_feedback', '')

    # Enforce structured output
    structured_llm = llm.with_structured_output(Perspectives, method='json_mode')

    # System message
    system_message = medical_analyst_instructions.format(
        researchTopic=researchTopic,
        human_analyst_feedback=human_analyst_feedback,
        max_medical_analysts=max_medical_analysts
    )
    
    # Generate analysts with explicit instruction about the output format
    analysts = structured_llm.invoke([
        SystemMessage(content=system_message),
        HumanMessage(content="Generate the set of medical analysts. Remember to output a JSON object with a 'medicalanalysts' key containing a list of analyst objects.")
    ])

    # Write the list of analysts to state
    return {"medicalanalysts": analysts.medicalanalysts}


def human_feedback(state:GenerateMedicalAnalystState):
    """No operation node that should be interrupted"""
    pass


def should_continue(state:GenerateMedicalAnalystState):
    """Return the next node to execute"""

    #check if human feedback

    human_analyst_feedback=state.get("human_analyst_feedback",None)
    if( human_analyst_feedback):
        return "create_medical_analysts"
    return END


#Add nodes and edges 

medical_analyst_builder=StateGraph(GenerateMedicalAnalystState)

medical_analyst_builder.add_node("create_medical_analysts",create_medical_analysts)
medical_analyst_builder.add_node("human_feedback",human_feedback)
medical_analyst_builder.add_edge(START,"create_medical_analysts")
medical_analyst_builder.add_edge("create_medical_analysts","human_feedback")
medical_analyst_builder.add_conditional_edges("human_feedback",should_continue,["create_medical_analysts",END])
#compile

memory=MemorySaver()
graph=medical_analyst_builder.compile(interrupt_before=["human_feedback"],checkpointer=memory)

#View
# display(Image(graph.get_graph().draw_mermaid_png()))


# Input
max_medical_analysts=4
topic="Vaccines for chicken pox diseaes"
thread={"configurable":{"thread_id":"1"}}



# Run the graph until the first interruption
for event in graph.stream({"researchTopic":topic,"max_medical_analysts":max_medical_analysts},thread,stream_mode="values"):
     analysts=event.get("medicalanalysts",'')
     if analysts:
          for analyst in analysts:
               print(f"Name :{analyst.name}")
               print(f"Affiliation:{analyst.affiliation}" )
               print(f"Role:{analyst.role}")
               print(f"Description:{analyst.description}")
               print("-"*50)

state=graph.get_state(thread)
print(state.next)

# We now update the state as if we are the human_feedback node
graph.update_state(thread, {"human_analyst_feedback": 
                            "Add in someone from a research for children chicken pox"}, as_node="human_feedback")


# Continue the graph execution
for event in graph.stream(None, thread, stream_mode="values"):
    # Review
    analysts = event.get('medicalanalysts', '')
    if analysts:
        for analyst in analysts:
            print(f"Name: {analyst.name}")
            print(f"Affiliation: {analyst.affiliation}")
            print(f"Role: {analyst.role}")
            print(f"Description: {analyst.description}")
            print("-" * 50) 

further_feedack = None
graph.update_state(thread, {"human_analyst_feedback": 
                            further_feedack}, as_node="human_feedback")


# Continue the graph execution to end
for event in graph.stream(None, thread, stream_mode="updates"):
    print("--Node--")
    node_name = next(iter(event.keys()))
    print(node_name)
    

final_state = graph.get_state(thread)
analysts = final_state.values.get('medicalanalysts')


print(analysts)


print(final_state.next)


print(analysts)