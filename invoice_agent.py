from langchain_ollama import ChatOllama  # Change this import
from langgraph.graph import MessagesState, START, StateGraph
from langgraph.prebuilt import tools_condition, ToolNode
from IPython.display import Image, display
from typing_extensions import TypedDict
from langchain_core.messages import AnyMessage, HumanMessage, SystemMessage
from typing import Annotated
from langgraph.graph.message import add_messages
from langgraph.checkpoint.memory import MemorySaver

def add_state_tax(base_price: float, tax_rate: float) -> float:
    """Add state tax to the base price
    Args:
        base_price (float): The price before state tax.
        tax_rate (float): Tax rate in percentage.

    Returns:
        float: The applied state tax on the base price.
    """
    return base_price + (base_price * tax_rate / 100)


def add_country_tax(base_price: float, tax_rate: float) -> float:
    """Add country tax to the base price
    Args:
        base_price (float): The price before country tax.
        tax_rate (float): Country tax rate in percentage.

    Returns:
        float: The applied country tax on the base price.
    """
    return base_price + (base_price * tax_rate / 100)


def apply_discount(taxedPrice: float, discount_percent: float) -> float:
    """Apply discount to the taxed price

    Args:
        taxedPrice (float): The taxedPrice before discount
        discount_percent (float): Discount percentage to apply

    Returns:
        float: The taxedPrice with discount
    """
    
    return taxedPrice - (taxedPrice * discount_percent / 100)


#Registering tools
tools = [add_state_tax, add_country_tax, apply_discount]


#Initializing LLM
llm = ChatOllama(  # Changed from OllamaLLM to ChatOllama
    model="qwen2.5:latest",
    temperature=0.7
)

#Binding tools to the LLM
llm_with_tools = llm.bind_tools(tools)


#Human or AI Messages
class MessagesState(TypedDict):
    messages: Annotated[list[AnyMessage], add_messages]
    
    
#Update system message to fit the business use case
sys_msg = SystemMessage(content="You are an invoice assistant that helps calculating final prices after state tax, country tax and discount.")


#Assistant node that invokes the LLM with tools
def assistant(state: MessagesState):
    return {
        "messages": [llm_with_tools.invoke([sys_msg] + state["messages"])]
    }
    

#Building agent workflow - langgraph
builder = StateGraph(MessagesState)

##Define the node
builder.add_node("assistant", assistant)
builder.add_node("tools", ToolNode(tools))

##Define the edges
builder.add_edge(START, "assistant")
builder.add_conditional_edges(
    "assistant",
    tools_condition
)

builder.add_edge("tools", "assistant")
react_graph=builder.compile()


#Displaying the react agent workflow diagram
# display(Image(react_graph.get_graph().draw_mermaid_png()))


#Example business message from a user
# messages = [
#     HumanMessage(content="The product is $100. Add 5% state, 5% country tax and apply a 5% discount. what is the final price?")
# ] 

# #Calling the langgraph react agent
# messages = react_graph.invoke({"messages": messages})
# # print(messages)


# #Showing the messages
# for m in messages["messages"]:
#     m.pretty_print()
    
# #Adding more messages (without the memory)
# messages = [HumanMessage(content="Apply 5% more discount")]
# messages = react_graph.invoke({"messages": messages})
# for m in messages["messages"]:
#     m.pretty_print()
    
    
#Adding memory for agent (Agent with Memory)
memory = MemorySaver()
react_graph = builder.compile(checkpointer=memory)

config = {"configurable": {"thread_id": "1"}}

messages = [
    HumanMessage(content="The product is $100. Add 5% state, 5% country tax and apply a 5% discount. what is the final price?")
] 

messages = react_graph.invoke({"messages": messages}, config)
# print(f"The size of messages: {len(messages['messages'])}")


##Showing the messages (with memory)
for m in messages["messages"]:
    m.pretty_print()
    
##More messages with memory
messages = [HumanMessage(content="Apply 5% more discount")]
messages = react_graph.invoke({"messages": messages}, config)
# print(f"The size of messages: {len(messages['messages'])}")
for m in messages["messages"]:
    m.pretty_print()