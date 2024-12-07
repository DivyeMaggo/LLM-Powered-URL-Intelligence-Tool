import os
import streamlit as st
import pickle
import time
from langchain import OpenAI
from langchain.chains import RetrievalQAWithSourcesChain
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.document_loaders import UnstructuredURLLoader
from langchain.embeddings import OpenAIEmbeddings
# from langchain.vectorstores import FAISS
from langchain_community.vectorstores import FAISS
import dill
import warnings
warnings.filterwarnings("ignore")


from dotenv import load_dotenv
load_dotenv()  # take environment variables from .env (especially openai api key)

if "vector_store" not in st.session_state:
    st.session_state["vector_store"] = None


st.title("News Research Tool 📈")
st.sidebar.title("News Article URLs")

urls = []
for i in range(3):
    url = st.sidebar.text_input(f"URL {i+1}")
    urls.append(url)

process_url_clicked = st.sidebar.button("Process URLs")
main_placeholder = st.empty()
llm = OpenAI(temperature=0.9, max_tokens=500)

if process_url_clicked:
    urls = [url for url in urls if url.strip()]  # Remove empty URLs
    if not urls:
        st.error("Please provide at least one valid URL.")
    else:
        try:
            loader = UnstructuredURLLoader(urls=urls)
            st.write("Data Loading...Started...✅✅✅")
            data = loader.load()
            st.write(f"Data Loaded. Found {len(data)} documents.")
            
            text_splitter = RecursiveCharacterTextSplitter(
                separators=['\n\n', '\n', '.', ','],
                chunk_size=1000
            )
            st.write("Text Splitter...Started...✅✅✅")
            docs = text_splitter.split_documents(data)

            embeddings = OpenAIEmbeddings()
            st.session_state["vector_store"] = FAISS.from_documents(docs, embeddings)
            st.write("Embedding Vector Started Building...✅✅✅")
            st.write("Enter your question in the above text box and wait for then answer!")

        except Exception as e:
            st.error(f"Error during processing: {e}")

query = main_placeholder.text_input("Questions.. :")
if query:
    if st.session_state["vector_store"] is None:
        st.error("No embeddings have been created yet. Please process URLs first.")
    else:
        chain = RetrievalQAWithSourcesChain.from_llm(
            llm=llm, 
            retriever=st.session_state["vector_store"].as_retriever()
        )
        result = chain({"question": query}, return_only_outputs=True)

        st.header("Answer")
        st.write(result.get("answer", "No answer available."))

        sources = result.get("sources", "")
        if sources:
            st.subheader("Sources:")
            for source in sources.split("\n"):
                st.write(source)