import streamlit as st
from langchain_community.document_loaders import PyPDFLoader
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.prompts import ChatPromptTemplate
from langchain.memory import ConversationBufferMemory
from langchain.chains import ConversationalRetrievalChain
from langchain.schema import HumanMessage
import tempfile

# --- Hardcoded Google API Key ---
GOOGLE_API_KEY = "AIzaSyC0_ttd2-0uIAJiPzTK73aPbIsSqVMDD-I"

# --- Streamlit UI ---
st.set_page_config(page_title="Gemini PDF Chat (with Memory)", layout="wide")
st.title("Ask Questions About Your PDF (with Conversational Memory)")

# Session state for memory
if "memory" not in st.session_state:
    st.session_state.memory = ConversationBufferMemory(memory_key="chat_history", return_messages=True)

# Upload PDF
uploaded_file = st.file_uploader("Upload a PDF file", type=["pdf"])

# Input question
query = st.text_input("Ask a question based on your PDF:")

if uploaded_file:
    with tempfile.NamedTemporaryFile(delete=False) as tmp_file:
        tmp_file.write(uploaded_file.read())
        tmp_path = tmp_file.name

    try:
        # Load and extract PDF content
        loader = PyPDFLoader(tmp_path)
        pages = loader.load()
        full_text = "\n\n".join([page.page_content for page in pages])

        # Initialize Gemini LLM
        llm = ChatGoogleGenerativeAI(
            model="gemini-2.0-flash",
            google_api_key=GOOGLE_API_KEY,
            temperature=0.5
        )

        # Define prompt with context and memory
        prompt = ChatPromptTemplate.from_template(
            """
            You are a helpful assistant. Use the following document content and chat history to answer the user's question.
            
            Document:
            {context}

            Chat History:
            {chat_history}

            User Question:
            {question}
            """
        )

        # Format the prompt with memory
        messages = prompt.format_messages(
            context=full_text,
            chat_history=st.session_state.memory.load_memory_variables({})["chat_history"],
            question=query
        )

        # Get response from model
        response = llm(messages)

        # Update memory
        st.session_state.memory.chat_memory.add_user_message(query)
        st.session_state.memory.chat_memory.add_ai_message(response.content)

        # Display response
        st.success("Answer:")
        st.write(response.content)

    except Exception as e:
        st.error(f"? Error: {str(e)}")

elif query:
    st.info("?? Please upload a PDF file.")
