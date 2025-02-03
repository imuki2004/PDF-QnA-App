import logging
from typing import List

import streamlit as st
from langchain_qdrant import QdrantVectorStore
from qdrant_client import QdrantClient
from qdrant_client.http.models import Distance, VectorParams
from langchain_ollama import OllamaEmbeddings
from langchain_core.prompts import ChatPromptTemplate
from langchain_ollama.llms import OllamaLLM
from langchain_community.document_loaders import PDFPlumberLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter

PDFS_DIRECTORY = "./pdfs/"
CHAT_TEMPLATE = """
You are an assistant for question-answering tasks. Use the following pieces of retrieved context to answer the question.
If you don't know the answer, just say that you don't know.
Keep the answer short and concise.
Question: {question}
Context: {context}
Answer:
"""
CHUNK_SIZE = 1000
CHUNK_OVERLAP = 200

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

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


def save_uploaded_pdf(uploaded_file) -> str:
    """
    Save the uploaded PDF to the PDFs directory.
    
    Parameters:
        uploaded_file: The file object uploaded via Streamlit.
    
    Returns:
        The file path of the saved PDF.
    """
    file_path = os.path.join(PDFS_DIRECTORY, uploaded_file.name)
    try:
        with open(file_path, "wb") as f:
            f.write(uploaded_file.getbuffer())
        logger.info(f"Saved file to {file_path}")
    except Exception as e:
        logger.error(f"Failed to save file: {e}")
        raise
    return file_path


def load_pdf_documents(file_path: str) -> List:
    """
    Load documents from a PDF file.
    
    Parameters:
        file_path (str): Path to the PDF file.
    
    Returns:
        List of documents loaded from the PDF.
    """
    loader = PDFPlumberLoader(file_path)
    documents = loader.load()
    logger.info(f"Loaded {len(documents)} document(s) from {file_path}")
    return documents


def split_documents(documents: List) -> List:
    """
    Split documents into smaller chunks.
    
    Parameters:
        documents (List): List of documents.
    
    Returns:
        List of document chunks.
    """
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=CHUNK_SIZE,
        chunk_overlap=CHUNK_OVERLAP,
        add_start_index=True
    )
    chunks = text_splitter.split_documents(documents)
    logger.info(f"Split documents into {len(chunks)} chunks")
    return chunks


def index_documents(documents: List) -> None:
    """
    Add document chunks to the vector store.
    
    Parameters:
        documents (List): List of document chunks.
    """
    vector_store.add_documents(documents)
    logger.info("Indexed documents in vector store")


def retrieve_relevant_docs(query: str, k: int = 2) -> List:
    """
    Retrieve documents relevant to the query.
    
    Parameters:
        query (str): The search query.
        k (int): Number of similar documents to retrieve.
    
    Returns:
        List of relevant document chunks.
    """
    results = vector_store.similarity_search(query, k=k)
    logger.info(f"Retrieved {len(results)} relevant document(s) for query")
    return results


def generate_answer(question: str, documents: List) -> str:
    """
    Generate an answer based on the question and document context.
    
    Parameters:
        question (str): The user's question.
        documents (List): List of document chunks for context.
    
    Returns:
        Generated answer as a string.
    """
    context = "\n\n".join(doc.page_content for doc in documents)
    prompt = ChatPromptTemplate.from_template(CHAT_TEMPLATE)
    chain = prompt | model
    answer = chain.invoke({"question": question, "context": context})
    logger.info("Generated answer for the question")
    return answer


def main():
    """Main function to run the Streamlit app."""
    st.title('Chat with your PDF 📑')
    
    uploaded_file = st.file_uploader("Upload a PDF", type=["pdf"], accept_multiple_files=False)
    
    if uploaded_file:
        try:
            pdf_path = save_uploaded_pdf(uploaded_file)
            documents = load_pdf_documents(pdf_path)
            chunks = split_documents(documents)
            index_documents(chunks)
        except Exception as e:
            st.error(f"Error processing PDF: {e}")
            return

        question = st.chat_input("Ask a question about your PDF:")
        
        if question:
            st.chat_message("user").write(question)
            relevant_docs = retrieve_relevant_docs(question)
            answer = generate_answer(question, relevant_docs)
            st.chat_message("assistant").write(answer)


if __name__ == "__main__":
    main()
