from langchain.chains import RetrievalQA
import textwrap

# Creating the chain for Question Answering
def load_qa_chain(retriever, llm, prompt):
    return RetrievalQA.from_chain_type(
        llm=llm,
        retriever=retriever,
        chain_type="stuff",
        return_source_documents=True,
        chain_type_kwargs={'prompt': prompt}
    )

# Prettifying the response
def get_response(query, chain):
    response = chain.invoke({'query': query})
    wrapped_text = textwrap.fill(response['result'], width=100)
    print(wrapped_text)
    return wrapped_text
