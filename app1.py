import streamlit as st
import time
from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain_chroma import Chroma
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.chains import create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_core.prompts import ChatPromptTemplate
import os
from dotenv import load_dotenv

load_dotenv()

# Ensure the GOOGLE_API_KEY is set in your .env file
if "GOOGLE_API_KEY" not in os.environ:
    st.error("Please set the GOOGLE_API_KEY in your .env file.")
    st.stop()

st.title("RAG Application built on Gemini Model")

try:
    loader = PyPDFLoader("yolov9_paper.pdf")
    data = loader.load()

    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000)
    docs = text_splitter.split_documents(data)

    embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001")
    vectorstore = Chroma.from_documents(documents=docs, embedding=embeddings)

    retriever = vectorstore.as_retriever(search_type="similarity", search_kwargs={"k": 10})

    llm = ChatGoogleGenerativeAI(model="gemini-1.5-pro", temperature=0, max_tokens=None, timeout=None)

    query = st.chat_input("Say something: ")
    prompt = query

    system_prompt = (
        "You are an assistant for question-answering tasks. "
        "Use the following pieces of retrieved context to answer "
        "the question. If you don't know the answer, say that you "
        "don't know. Use three sentences maximum and keep the "
        "answer concise."
        "\n\n"
        "{context}"
    )

    prompt = ChatPromptTemplate.from_messages(
        [
            ("system", system_prompt),
            ("human", "{input}"),
        ]
    )

    if query:
        question_answer_chain = create_stuff_documents_chain(llm, prompt)
        rag_chain = create_retrieval_chain(retriever, question_answer_chain)

        response = rag_chain.invoke({"input": query})
        st.write(response["answer"])

except ModuleNotFoundError as e:
    if "langchain_community" in str(e):
        st.error(
            "Error: The 'langchain-community' package is not installed.\n"
            "Please open your terminal or Anaconda Prompt and run: `pip install langchain-community`"
        )
    else:
        st.error(f"An unexpected module import error occurred: {e}")
    st.stop()
except FileNotFoundError:
    st.error("Error: The file 'yolov9_paper.pdf' was not found. Please ensure it is in the correct directory.")
    st.stop()
except Exception as e:
    st.error(f"An unexpected error occurred: {e}")
    st.stop()
