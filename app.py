import streamlit as st
from dotenv import load_dotenv

def main():
    load_dotenv()
    st.set_page_config(page_title="Mutli-Chat PDF", page_icon=":books:")

    st.header("Chat with multiple PDFs :books:")
    st.text_input("Ask a question about your documents: ")

    with st.sidebar:
        st.subheader("Your PDFs")
        pdf_docs = st.file_uploader("Upload your PDFs here", type="pdf", accept_multiple_files=True)
        if st.button("Upload"):
            st.spinner("Uploading PDFs...")
            #1. Get the PDF
            #2. Get the text chunks
            #3. Create vector store

if __name__ == "__main__":
    main()