import streamlit as st
from dotenv import load_dotenv
from PyPDF2 import PdfReader
from langchain.text_splitter import CharacterTextSplitter
from langchain.embeddings import OpenAIEmbeddings, HuggingFaceInstructEmbeddings
from langchain.vectorstores import FAISS
from langchain.memory import ConversationBufferMemory
from langchain.chains import ConversationalRetrievalChain
from langchain.chat_models import ChatOpenAI
from html_templates import css, bot_template, user_template

def get_pdf_text(pdf_docs):
    text = ""
    for pdf in pdf_docs:
        pdf_reader = PdfReader(pdf)
        for page in pdf_reader.pages:
            text += page.extract_text()
    return text

def get_text_chunks(text):
    text_splitter = CharacterTextSplitter(
        separator="\n",
        chunk_size=1000, 
        chunk_overlap=200,
        length_function=len
    )
    chunks = text_splitter.split_text(text)
    return chunks

# Since OpenAIEmbeddings cost money we can use HuggingFaceInstructEmbeddings to get embeddings in our computer
def get_embeddings(text_chunks):
    embeddings = OpenAIEmbeddings()
    # embeddings = HuggingFaceInstructEmbeddings(model_name="hkunlp/instructor-xl")
    vector_store = FAISS.from_texts(texts=text_chunks, embeddings=embeddings)
    return vector_store

def get_conversation_chain(vector_store):
    llm = ChatOpenAI()
    memory = ConversationBufferMemory(memory_key="chat_history", return_messages=True)
    conversation_chain = ConversationalRetrievalChain.from_llm(
        llm=llm,
        retriever=vector_store.as_retriever(),
        memory=memory
    )


def main():
    load_dotenv()
    st.set_page_config(page_title="Mutli-Chat PDF", page_icon=":books:")
    st.write(css, unsafe_allow_html=True)

    # Persists the conversation state variable across re-renders
    if "conversation" not in st.session_state:
        st.session_state.conversation = None

    st.header("Chat with multiple PDFs :books:")
    user_question = st.text_input("Ask a question about your documents: ")
    

    st.write(user_template.replace("{{MSG}}", "Hello App!"), unsafe_allow_html=True)
    st.write(bot_template.replace("{{MSG}}", "Hello Human!"), unsafe_allow_html=True)

    with st.sidebar:
        st.subheader("Your PDFs")
        pdf_docs = st.file_uploader("Upload your PDFs here", type="pdf", accept_multiple_files=True)
        if st.button("Upload"):
            with st.spinner("Uploading PDFs..."):
                #1. Extract the text from the PDF
                raw_text = get_pdf_text(pdf_docs)
                
                #2. Get the text chunks from the PDF
                text_chunks = get_text_chunks(raw_text)
                
                #3. Create vector store or embeddings
                vector_store = get_embeddings(text_chunks)

                #4. Create conversation chain
                st.session_state.conversation = get_conversation_chain(vector_store) # Making session state persistent because streamlit reinitializes the environment

if __name__ == "__main__":
    main()