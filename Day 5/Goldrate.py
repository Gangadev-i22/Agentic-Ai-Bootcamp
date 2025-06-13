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

# --- HARDCODED API KEY (for test only) ---
GOOGLE_API_KEY = "AIzaSyC0_ttd2-0uIAJiPzTK73aPbIsSqVMDD-I"

# --- Streamlit UI ---
st.set_page_config(page_title="Gold Rate AI Agent", layout="centered")
st.title(" AI Agent: Find Current Gold Rate")

query = st.text_input("?? Ask your question", value="Current gold rate in India")

if st.button("Run Agent") and query:

    # Step 1: Web search via DuckDuckGo
    search_tool = DuckDuckGoSearchRun()
    raw_result = search_tool.run(query)
    st.subheader("?? Web Search Result")
    st.write(raw_result)

    # Step 2: Wrap result and split
    docs = [Document(page_content=raw_result)]
    splitter = CharacterTextSplitter(chunk_size=300, chunk_overlap=50)
    split_docs = splitter.split_documents(docs)

    # Step 3: Embed and store with FAISS
    embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
    vectorstore = FAISS.from_documents(split_docs, embeddings)
    retriever = vectorstore.as_retriever()

    # Step 4: Custom prompt
    prompt = ChatPromptTemplate.from_template("""
Answer the question based on the context below. Be concise and accurate.

Context:
{context}

Question:
{question}
""")

    # Step 5: Gemini 1.5 Flash model
    llm = ChatGoogleGenerativeAI(
        model="models/gemini-1.5-flash-latest",
        google_api_key=GOOGLE_API_KEY,
        temperature=0.3
    )

    # Step 6: RAG chain composition
    rag_chain = (
        RunnableMap({
            "context": lambda q: retriever.get_relevant_documents(q),
            "question": RunnablePassthrough()
        })
        | prompt
        | llm
        | StrOutputParser()
    )

    # Step 7: Run and show result
    with st.spinner("?? Thinking..."):
        answer = rag_chain.invoke(query)
        st.subheader("?? Final Answer")
        st.write(answer)
