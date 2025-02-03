import os
import streamlit as st
from dotenv import load_dotenv
from langchain_qdrant import QdrantVectorStore
from qdrant_client import QdrantClient
from qdrant_client.http.models import Distance, VectorParams
from langchain_ollama import OllamaEmbeddings
from langchain_core.prompts import ChatPromptTemplate
from langchain_ollama.llms import OllamaLLM
from langchain_community.document_loaders import PDFPlumberLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter

load_dotenv()
QDRANT_URL = os.getenv("URL")
QDRANT_API_KEY = os.getenv("API_KEY")

pdfs_directory = "./pdfs/"
template = """
    You are an assistant for question-answering tasks. Use the following pieces of retrieved context to answer the question.
    If you don't know the answer, just say that you don't know.
    Keep the answer clear and concise.
    Question: {question}
    Context: {context}
    Answer:
"""

embeddings = OllamaEmbeddings(model="deepseek-r1:8b")
model = OllamaLLM(model="deepseek-r1:8b")

client = QdrantClient(":memory:")

client.create_collection(
    collection_name="pdfs",
    vectors_config=VectorParams(size=4096, distance=Distance.COSINE),
)

vector_store = QdrantVectorStore(
    client=client,
    collection_name="pdfs",
    embedding=embeddings,
)

def upload_pdf(file):
    with open(pdfs_directory + file.name, "wb") as f:
        f.write(file.getbuffer())
        
def load_pdf(file_path):
    loader = PDFPlumberLoader(file_path)
    documents = loader.load()
    
    return documents

def split_text(documents):
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=200,
        add_start_index=True
    )
    
    return text_splitter.split_documents(documents)
    
def index_docs(documents):
    vector_store.add_documents(documents)

def retrieve_docs(query):
    results = vector_store.similarity_search(
        query, k=5 
    )
    
    return results

def answer_question(question, documents):
    context = "\n\n".join([doc.page_content for doc in documents])
    prompt = ChatPromptTemplate.from_template(template)
    chain = prompt | model
    
    return chain.invoke({"question": question, "context": context})

st.title('Chat with your PDF 📑')

uploaded_file = st.file_uploader(
    "Upload a PDF", 
    type=["pdf"],
    accept_multiple_files=False
)

if uploaded_file:
    upload_pdf(uploaded_file)
    documents = load_pdf(pdfs_directory + uploaded_file.name)
    chunks = split_text(documents)
    index_docs(chunks)
    
    question = st.chat_input()
    
    if question:
        st.chat_message("user").write(question)
        related_docs = retrieve_docs(question)
        answer = answer_question(question, chunks)
        st.chat_message("assistant").write(answer)
        


    