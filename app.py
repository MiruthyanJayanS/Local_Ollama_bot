import tempfile
from embeddings_utils import load_embedding_model, create_embeddings
from pdf_utils import load_pdf_data, split_docs
from qa_chain_utils import load_qa_chain, get_response
from langchain_community.llms import Ollama
from langchain_core.prompts import PromptTemplate
import streamlit as st

# Prompt Template
template = """
### System:
You are an respectful and honest assistant. You have to answer the user's \
questions using only the context provided to you. If you don't know the answer, \
just say you don't know. Don't try to make up an answer.

### Context:
{context}

### User:
{question}

### Response:
"""
st.title("Conversational Chatbot")
st.sidebar.title("Upload PDF")
uploaded_file = st.sidebar.file_uploader("Choose a PDF file", type="pdf")

if uploaded_file is not None:
    with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as temp_file:
        temp_file.write(uploaded_file.read())
        temp_file_path = temp_file.name

    # Load and process PDF
    docs = load_pdf_data(temp_file_path)
    documents = split_docs(documents=docs)

    # Load models and vectorstore
    llm = Ollama(model="llama3", temperature=0)
    embed = load_embedding_model(model_path="all-MiniLM-L6-v2")
    vectorstore = create_embeddings(documents, embed)
    retriever = vectorstore.as_retriever()
    prompt = PromptTemplate.from_template(template)
    chain = load_qa_chain(retriever, llm, prompt)
    def get_response(query, chain):
        return chain.invoke({"query": query})
# Main app
def main():
    if "messages" not in st.session_state:
        st.session_state.messages = []

    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])

    if input_text := st.chat_input("Enter your query here"):
        st.session_state.messages.append({"role": "user", "content": input_text})
        with st.chat_message("user"):
            st.markdown(input_text)

        response = get_response(query=input_text, chain=chain)
        with st.chat_message("assistant"):
            st.markdown(response['result'])
        st.session_state.messages.append({"role": "assistant", "content": response['result']})

if __name__ == '__main__':
    main()
