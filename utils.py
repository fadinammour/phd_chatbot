from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser
import json

def load_json(path):
    with open(path, 'r') as f:
        data = json.load(f)
    return data

def format_docs(docs):
    """Convert loaded documents into strings by concatenating their content
    and ignoring metadata."""
    return "\n\n".join(doc.page_content for doc in docs)

def get_qa_chain(rag_prompt, model, retriever):
    chain = (
        {"context": retriever | format_docs, "question": RunnablePassthrough()}
        | rag_prompt
        | model
        | StrOutputParser()
    )
    return chain
