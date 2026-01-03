import os
import glob

from dotenv import load_dotenv
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_chroma import Chroma
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.document_loaders import DirectoryLoader, TextLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.messages import SystemMessage, HumanMessage
import gradio as gr
load_dotenv()


def fetch_documents():
    folders = glob.glob("knowledge-base/*")

    documents=[]

    for folder in folders:
        doc_type = os.path.basename(folder)
        loader = DirectoryLoader(folder, glob="**/*.md", loader_cls=TextLoader, loader_kwargs={'encoding' : 'utf-8'})
        folder_docs = loader.load()
        for doc in folder_docs:
            doc.metadata["doc_type"] = doc_type
            documents.append(doc)
    return documents



def create_chunks(documents):
    text_splitter = RecursiveCharacterTextSplitter(chunk_size = 800, chunk_overlap = 250)
    chunks= text_splitter.split_documents(documents) 
    return chunks



def create_embeddings(chunks):
    embeddings = HuggingFaceEmbeddings(model_name = "all-MiniLM-L6-v2")

    db_name = "vector_db"

    if os.path.exists(db_name):
        Chroma(persist_directory=db_name, embedding_function=embeddings).delete_collection()
    
    vectorstore = Chroma.from_documents(documents=chunks, embedding=embeddings, persist_directory=db_name)
    return vectorstore

if __name__ == "__main__":
    documents = fetch_documents()
    chunks = create_chunks(documents)
    embeddings = create_embeddings(chunks)
    print("Ingestion complete")