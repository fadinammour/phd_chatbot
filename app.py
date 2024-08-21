import gradio as gr
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_chroma import Chroma
from langchain_ollama import (
    OllamaEmbeddings,
    ChatOllama
)
from langchain_core.prompts import ChatPromptTemplate
from langchain_community.document_loaders import PyMuPDFLoader
from utils import (
    get_qa_chain,
    load_json
)

config = load_json('config.json')

loader = PyMuPDFLoader(config['path']['corpus'])
data = loader.load()

text_splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=0)
all_splits = text_splitter.split_documents(data)

local_embeddings = OllamaEmbeddings(model="nomic-embed-text")

vectorstore = Chroma.from_documents(documents=all_splits, embedding=local_embeddings)

model = ChatOllama(
    model=config['llm']['model'],
)

RAG_TEMPLATE = load_json(config['path']['prompt'])

rag_prompt = ChatPromptTemplate.from_template(RAG_TEMPLATE)

retriever = vectorstore.as_retriever()

qa_chain = get_qa_chain(rag_prompt, model, retriever)

def get_answer(message, history):
    response = qa_chain.invoke(message)
    return response

with gr.Blocks() as page:
    gr.ChatInterface(fn=get_answer, examples=["C'est quoi un problème inverse?"], title="Fadi's PhD Q&A Bot")
    
page.launch()