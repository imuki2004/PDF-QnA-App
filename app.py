import streamlit as st
from rag import (
    save_uploaded_pdf,
    load_pdf_documents,
    split_documents,
    index_documents,
    retrieve_relevant_docs,
    generate_answer,
)


def initialize_session_state():
    """
    Initialize session state variables if they don't exist.
    """
    if "pdf_path" not in st.session_state:
        st.session_state.pdf_path = None
    if "doc_chunks" not in st.session_state:
        st.session_state.doc_chunks = None
    if "conversation" not in st.session_state:
        st.session_state.conversation = []


def process_pdf(uploaded_file):
    """
    Process the uploaded PDF by saving, loading, splitting, and indexing it.
    """
    pdf_path = save_uploaded_pdf(uploaded_file)
    documents = load_pdf_documents(pdf_path)
    chunks = split_documents(documents)
    index_documents(chunks)
    st.session_state.pdf_path = pdf_path
    st.session_state.doc_chunks = chunks
    st.success("PDF processed and indexed successfully.")


def display_chat_history():
    """
    Display the conversation history stored in session state.
    """
    for msg in st.session_state.conversation:
        role = msg.get("role", "assistant")
        content = msg.get("content", "")
        st.chat_message(role).write(content)


def main():
    """Main function to run the Streamlit app."""
    st.set_page_config(page_title="Chat with your PDF", layout="wide")
    initialize_session_state()

    with st.sidebar:
        st.header("Instructions")
        st.markdown(
            """
            1. **Upload a PDF** using the uploader below.
            2. **Ask questions** related to your PDF content.
            3. The app will retrieve relevant sections of the PDF and generate an answer.
            """
        )

    st.title("Chat with your PDF 📑")

    # PDF Upload Section
    uploaded_file = st.file_uploader("Upload a PDF", type=["pdf"], key="pdf_uploader")
    if uploaded_file:
        if st.session_state.pdf_path is None or uploaded_file.name not in st.session_state.pdf_path:
            with st.spinner("Processing your PDF..."):
                try:
                    process_pdf(uploaded_file)
                except Exception as e:
                    st.error(f"Error processing PDF: {e}")
                    return

    if st.session_state.doc_chunks is not None:
        question = st.chat_input("Ask a question about your PDF:")
        if question:
            st.session_state.conversation.append({"role": "user", "content": question})
            st.chat_message("user").write(question)

            with st.spinner("Generating answer..."):
                try:
                    relevant_docs = retrieve_relevant_docs(question)
                    answer = generate_answer(question, relevant_docs)
                except Exception as e:
                    st.error(f"Error generating answer: {e}")
                    return

            st.session_state.conversation.append({"role": "assistant", "content": answer})
            st.chat_message("assistant").write(answer)

    if st.button("Clear Chat"):
        st.session_state.conversation = []
        st.rerun()


if __name__ == "__main__":
    main()
