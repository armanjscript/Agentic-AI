from langchain_ollama import ChatOllama
from langchain_community.document_loaders import WebBaseLoader
from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import OllamaEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.tools import create_retriever_tool
from typing import Annotated, Sequence, Literal
from typing_extensions import TypedDict
from langchain_core.messages import BaseMessage, HumanMessage, SystemMessage
from langchain_core.output_parsers import StrOutputParser
from langgraph.graph.message import add_messages
from pydantic import BaseModel, Field
from langchain_core.prompts import PromptTemplate
from langgraph.graph import END, StateGraph, START
from langgraph.prebuilt import ToolNode
from langgraph.prebuilt import tools_condition
from langchain_classic import hub




#Initializing the Embedding model
embedding = OllamaEmbeddings(
    model="nomic-embed-text:latest"
)


urls=[
    "https://www.microsoft.com/investor/reports/ar24/index.html/",
    "https://www.microsoft.com/en-us/investor/earnings/fy-2024-q4/segment-revenues/",
    "https://www.investopedia.com/how-microsoft-makes-money-4798809/"
]

docs=[WebBaseLoader(url).load() for url in urls]
# print(docs)
print("+" * 70 + ">>")


doc_list = [item for sublist in docs for item in sublist]
text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=100)
doc_splits = text_splitter.split_documents(doc_list)


#Add all these text to vectordb
vectorstore = FAISS.from_documents(
    documents=doc_splits,
    embedding=embedding
)

retriever = vectorstore.as_retriever()
# print(retriever.invoke("Microsoft revenue 2024"))

# Update the retriever tool creation with a more specific description
retriever_tool = create_retriever_tool(
    retriever,
    "retriever_vector_db_blog",
    "Search and retrieve information specifically about Microsoft's revenue, earnings, financial performance, and how the company makes money. Use this tool ONLY for questions about Microsoft's financial data, revenue streams, or business segments."
)

# print(retriever_tool)

#Initializing LLM
llm = ChatOllama(
    model="qwen2.5:latest",
    temperature=0.7
)

# Add system prompt to guide the agent when to use tools
system_prompt = """You are a helpful assistant with access to a Microsoft financial data retriever tool.
Use the 'retriever_vector_db_blog' tool ONLY when the user asks about:
- Microsoft's revenue
- Microsoft's earnings
- Microsoft's financial performance
- How Microsoft makes money
- Microsoft's business segments
- Microsoft's profit

For general questions not about Microsoft's finances, just answer directly without using the tool."""

# Bind tools with system prompt
tools = [retriever_tool]
llm_with_tools = llm.bind_tools(tools)

#Agent State
class AgentState(TypedDict):
    # The add_messages function defines how an update should be processed
    # Default is to replace. add_messages says "append"
    messages: Annotated[Sequence[BaseMessage], add_messages]
    

def agent(state):
    """
    Invokes the agent model to generate a response based on the current state.
    """
    print("---CALL AGENT---")
    messages = state["messages"]
    
    # Add system message to guide tool usage
    system_msg = SystemMessage(content=system_prompt)
    # Prepend system message to the conversation
    full_messages = [system_msg] + messages
    
    response = llm_with_tools.invoke(full_messages)
    return {"messages": [response]}



### Edges
def grade_documents(state) -> Literal["generate", "rewrite"]:
    """
    Determines whether the retrieved documents are relevant to the question.

    Args:
        state (messages): The current state

    Returns:
        str: A decision for whether the documents are relevant or not
    """

    print("---CHECK RELEVANCE---")

    # Data model
    class grade(BaseModel):
        """Binary score for relevance check."""

        binary_score: str = Field(description="Relevance score 'yes' or 'no'")

    

    # LLM with tool and validation
    llm_with_tool = llm.with_structured_output(grade, method='json_mode')

    # Prompt
    prompt = PromptTemplate(
        template="""You are a grader assessing relevance of a retrieved document to a user question. \n
        Here is the retrieved document: \n\n {context} \n\n
        Here is the user question: {question} \n
        If the document contains keyword(s) or semantic meaning related to the user question, grade it as relevant. \n
        Give a binary score 'yes' or 'no' score to indicate whether the document is relevant to the question.""",
        input_variables=["context", "question"],
    )

    # Chain
    chain = prompt | llm_with_tool

    messages = state["messages"]
    last_message = messages[-1]

    question = messages[0].content
    docs = last_message.content

    scored_result = chain.invoke({"question": question, "context": docs})

    score = scored_result.binary_score

    if score == "yes":
        print("---DECISION: DOCS RELEVANT---")
        return "generate"

    else:
        print("---DECISION: DOCS NOT RELEVANT---")
        print(score)
        return "rewrite"
    

def generate(state):
    """
    Generate answer

    Args:
        state (messages): The current state

    Returns:
         dict: The updated message
    """
    print("---GENERATE---")
    messages = state["messages"]
    question = messages[0].content
    last_message = messages[-1]

    docs = last_message.content

    # Prompt
    prompt = hub.pull("rlm/rag-prompt")

    # Post-processing
    def format_docs(docs):
        return "\n\n".join(doc.page_content for doc in docs)

    # Chain
    rag_chain = prompt | llm | StrOutputParser()

    # Run
    response = rag_chain.invoke({"context": docs, "question": question})
    return {"messages": [response]}


def rewrite(state):
    """
    Transform the query to produce a better question.

    Args:
        state (messages): The current state

    Returns:
        dict: The updated state with re-phrased question
    """

    print("---TRANSFORM QUERY---")
    messages = state["messages"]
    question = messages[0].content

    msg = [
        HumanMessage(
            content=f""" \n
    Look at the input and try to reason about the underlying semantic intent / meaning. \n
    Here is the initial question:
    \n ------- \n
    {question}
    \n ------- \n
    Formulate an improved question: """,
        )
    ]

    response = llm.invoke(msg)
    return {"messages": [response]}


# Define a new graph
workflow = StateGraph(AgentState)

# Define the nodes we will cycle between
workflow.add_node("agent", agent)  # agent
retrieve = ToolNode(tools)
workflow.add_node("retrieve", retrieve)  # retrieval
workflow.add_node("rewrite", rewrite)  # Re-writing the question
workflow.add_node(
    "generate", generate
)  # Generating a response after we know the documents are relevant
# Call agent node to decide to retrieve or not
workflow.add_edge(START, "agent")

# Decide whether to retrieve
workflow.add_conditional_edges(
    "agent",
    # Assess agent decision
    tools_condition,
    {
        # Translate the condition outputs to nodes in our graph
        "tools": "retrieve",
        END: END,
    },
)

# Edges taken after the `action` node is called.
workflow.add_conditional_edges(
    "retrieve",
    # Assess agent decision
    grade_documents,
)
workflow.add_edge("generate", END)
workflow.add_edge("rewrite", "agent")

# Compile
graph = workflow.compile()


print(">>>" * 50 + "---")

# Use HumanMessage objects for proper tool calling
result1 = graph.invoke({
    "messages": [HumanMessage(content="What is happening on Planet Mars")]
})
print(result1)

print(">>>" * 50 + "---")

result2 = graph.invoke({
    "messages": [HumanMessage(content="What are the revenue distribution of Microsoft company")]
})
print(result2)

print(">>>" * 50 + "---")

inputs = {
    "messages": [HumanMessage(content="How Microsoft is earning money?")],
}
for output in graph.stream(inputs):
    print(output)

print(">>>" * 50 + "---")

inputs = {
    "messages": [HumanMessage(content="How is Microsoft making profit?")],
}
for output in graph.stream(inputs):
    print(output)
