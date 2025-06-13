import streamlit as st
from langchain_community.document_loaders import PyPDFLoader
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.prompts import ChatPromptTemplate
import tempfile

# --- Hardcoded Google API Key ---
GOOGLE_API_KEY = "AIzaSyC0_ttd2-0uIAJiPzTK73aPbIsSqVMDD-I"

# --- Streamlit UI ---
st.set_page_config(page_title="Simple Gemini Q&A", layout="wide")
st.title("Ask Questions About Your PDF (No Embeddings)")

# Upload file
uploaded_file = st.file_uploader("Upload a PDF file", type=["pdf"])

# Input box
query = st.text_input("Ask a question based on your PDF:")

if uploaded_file and query:
    with tempfile.NamedTemporaryFile(delete=False) as tmp_file:
        tmp_file.write(uploaded_file.read())
        tmp_path = tmp_file.name

    try:
        # Load and extract content
        loader = PyPDFLoader(tmp_path)
        pages = loader.load()
        full_text = "\n\n".join([page.page_content for page in pages])

        # Setup Gemini model
        llm = ChatGoogleGenerativeAI(
            model="gemini-2.0-flash",
            google_api_key=GOOGLE_API_KEY,
            temperature=0.5
        )

        # Prompt: inject document content and question
        prompt = ChatPromptTemplate.from_template(
            "You are an assistant. Given the following document:\n\n{context}\n\nAnswer this question:\n{question}"
        )
        formatted_prompt = prompt.format_messages(context=full_text, question=query)

        # Run model
        response = llm(formatted_prompt)
        st.success("Answer:")
        st.write(response.content)

    except Exception as e:
        st.error(f"? Error: {str(e)}")

elif uploaded_file:
    st.info("?? Please enter a question.")
elif query:
    st.info("?? Please upload a PDF file.")