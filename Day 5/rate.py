import streamlit as st
from langchain_community.tools.ddg_search import DuckDuckGoSearchRun
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.prompts import ChatPromptTemplate
from langchain.schema.output_parser import StrOutputParser
from langchain.schema.runnable import RunnableMap, RunnablePassthrough
from langchain.text_splitter import CharacterTextSplitter
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.vectorstores import FAISS
from langchain.docstore.document import Document
import os

# --- HARDCODED GOOGLE API KEY (For testing only, use .env for production) ---
os.environ["GOOGLE_API_KEY"] = "AIzaSyC0_ttd2-0uIAJiPzTK73aPbIsSqVMDD-I"

# --- Streamlit Page Config ---
st.set_page_config(page_title="Gold Rate AI Agent", layout="centered")
st.title("AI Agent: Find Current Gold Rate")

# --- User Input ---
query = st.text_input("Ask your question", value="Current gold rate in India")

# --- Run Agent Button ---
if st.button("Run Agent") and query:

    # Step 1: Web search using DuckDuckGo
    search_tool = DuckDuckGoSearchRun()
    raw_result = search_tool.run(query)
    st.subheader("Web Search Result")
    st.write(raw_result)

    # Step 2: Document Wrapping and Splitting
    docs = [Document(page_content=raw_result)]
    splitter = CharacterTextSplitter(chunk_size=300, chunk_overlap=50)
    split_docs = splitter.split_documents(docs)

    # Step 3: Create Vectorstore with Embeddings
    embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
    vectorstore = FAISS.from_documents(split_docs, embeddings)
    retriever = vectorstore.as_retriever()

    # Step 4: Prompt Template
    prompt = ChatPromptTemplate.from_template("""
    Answer the question based on the context below. Be concise and accurate.

    Context:
    {context}

    Question:
    {question}
    """)

    # Step 5: Gemini LLM Setup
    llm = ChatGoogleGenerativeAI(
        model="models/gemini-1.5-flash-latest",
        temperature=0.3,
        google_api_key=os.environ["GOOGLE_API_KEY"]
    )

    # Step 6: Build RAG Chain
    rag_chain = (
        RunnableMap({
            "context": lambda q: retriever.get_relevant_documents(q),
            "question": RunnablePassthrough()
        })
        | prompt
        | llm
        | StrOutputParser()
    )

    # Step 7: Invoke and Display
    with st.spinner("?? Thinking..."):
        answer = rag_chain.invoke(query)
        st.subheader("?? Final Answer")
        st.write(answer)
