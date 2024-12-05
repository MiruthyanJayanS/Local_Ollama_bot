# import ollama
# import streamlit as st

# note = 'notes.md'

# with open(note, 'r') as file:
#     content = file.read()

# myprompt = f"this is note is about simple cokking recipe where u need to pick one dish at random\n{content}"

# response = ollama.generate(model="llama3", prompt=myprompt)

# actualresponse = response['response']


# st.write("Here is the response text:")
# st.text_area("PDF Text", actualresponse, height=400)
from pydantic import BaseModel
from langchain_core.messages import SystemMessage
from langgraph.checkpoint.memory import MemorySaver
from langgraph.graph import END, MessagesState, StateGraph
from langgraph.prebuilt import ToolNode, tools_condition
from langchain_community.llms.ollama import Ollama
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.document_loaders import PyMuPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain.chains import RetrievalQA
import torch

# Loading the language model
llm = Ollama(model="llama3", temperature=0)

# Function to load the embedding model
def load_embedding_model(model_path, normalize_embedding=True):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    return HuggingFaceEmbeddings(
        model_name=model_path,
        model_kwargs={'device': device},
        encode_kwargs={'normalize_embedding': normalize_embedding}
    )

# Function to split documents into chunks
def split_docs(documents, chunk_size=1000, chunk_overlap=20):
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap
    )
    chunks = text_splitter.split_documents(documents=documents)
    return chunks

# Function to create embeddings
def create_embeddings(documents, embed):
    vectorstore = FAISS.from_documents(documents, embed)
    return vectorstore

# Function to load QA chain
def load_qa_chain(retriever, llm, prompt):
    return RetrievalQA.from_chain_type(
        llm=llm,
        retriever=retriever,
        chain_type="stuff",
        return_source_documents=True,
        chain_type_kwargs={'prompt': prompt}
    )

# Define the retrieve function
def retrieve(query, retriever):
    results = retriever.retrieve(query)
    return results

# Step 1: Generate an AIMessage that may include a tool-call to be sent.
def query_or_respond(state: MessagesState):
    llm_with_tools = llm.bind_tools([retrieve])
    response = llm_with_tools.invoke(state["messages"])
    return {"messages": [response]}

# Step 2: Execute the retrieval.
tools = ToolNode([retrieve])

# Step 3: Generate a response using the retrieved content.
def generate(state: MessagesState):
    recent_tool_messages = []
    for message in reversed(state["messages"]):
        if message.type == "tool":
            recent_tool_messages.append(message)
        else:
            break
    tool_messages = recent_tool_messages[::-1]

    docs_content = "\n\n".join(doc.content for doc in tool_messages)
    system_message_content = (
        "You are an assistant for question-answering tasks. "
        "Use the following pieces of retrieved context to answer "
        "the question. If you don't know the answer, say that you "
        "don't know. Use three sentences maximum and keep the "
        "answer concise."
        "\n\n"
        f"{docs_content}"
    )
    conversation_messages = [
        message
        for message in state["messages"]
        if message.type in ("human", "system")
        or (message.type == "ai" and not message.tool_calls)
    ]
    prompt = [SystemMessage(system_message_content)] + conversation_messages

    response = llm.invoke(prompt)
    return {"messages": [response]}

# Build graph
graph_builder = StateGraph(MessagesState)
graph_builder.add_node(query_or_respond)
graph_builder.add_node(tools)
graph_builder.add_node(generate)
graph_builder.set_entry_point("query_or_respond")
graph_builder.add_conditional_edges(
    "query_or_respond",
    tools_condition,
    {END: END, "tools": "tools"},
)
graph_builder.add_edge("tools", "generate")
graph_builder.add_edge("generate", END)

memory = MemorySaver()
graph = graph_builder.compile(checkpointer=memory)

# Example usage
if __name__ == "__main__":
    # Load documents and create embeddings
    loader = PyMuPDFLoader(r"E:\Project\llama3_chat\data\ML Project Description.pdf")
    documents = loader.load()
    embed = load_embedding_model(model_path="all-MiniLM-L6-v2")
    chunks = split_docs(documents)
    vectorstore = create_embeddings(chunks, embed)
    retriever = vectorstore.as_retriever()

    # Create QA chain
    prompt_template = "Your prompt template here"
    qa_chain = load_qa_chain(retriever, llm, prompt_template)

    # Initialize state
    state = MessagesState()
    state["messages"] = [{"type": "human", "content": "Your initial query here"}]

    # Run the graph
    result = graph.invoke(state)
    print(result)