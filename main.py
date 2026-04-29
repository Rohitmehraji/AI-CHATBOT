import streamlit as st
import os
import tempfile
from src.chain import rag_chain
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.output_parsers import StrOutputParser
from src.prompt import system_prompt
from langchain_core.prompts import ChatPromptTemplate
from src.vector_store import vector_store_retriver
from src.utils import file_loader, split_doc, embedding

# ── API Key: Streamlit Secrets se lo (deploy pe) ya .env se (local pe) ──────
GOOGLE_API_KEY = st.secrets.get("GOOGLE_API_KEY", os.getenv("GOOGLE_API_KEY", ""))
if not GOOGLE_API_KEY:
    st.error("⚠️ GOOGLE_API_KEY not found. Add it in Streamlit Secrets or .env file.")
    st.stop()
os.environ["GOOGLE_API_KEY"] = GOOGLE_API_KEY

# ── Page Config ───────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="RAG Chatbot – Rohit Kumar",
    page_icon="🤖",
    layout="centered"
)

st.title("🤖 RAG-Based AI Chatbot")
st.caption("Upload any PDF and chat with it using Google Gemini + LangChain + FAISS")

# ── Session State ─────────────────────────────────────────────────────────────
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []
if "chain" not in st.session_state:
    st.session_state.chain = None
if "uploaded_file_name" not in st.session_state:
    st.session_state.uploaded_file_name = None

# ── Sidebar: PDF Upload ───────────────────────────────────────────────────────
with st.sidebar:
    st.header("📄 Upload Document")
    uploaded_file = st.file_uploader("Upload a PDF", type=["pdf"])

    if uploaded_file:
        if uploaded_file.name != st.session_state.uploaded_file_name:
            with st.spinner("Processing PDF... please wait ⏳"):
                try:
                    # Save to temp file (file_loader expects a path)
                    with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp:
                        tmp.write(uploaded_file.read())
                        tmp_path = tmp.name

                    pages          = file_loader(tmp_path)
                    splitter       = split_doc()
                    docs           = splitter.split_documents(pages)
                    embeddings_model = embedding(model_name="sentence-transformers/all-MiniLM-L6-v2")
                    llm            = ChatGoogleGenerativeAI(model="gemini-2.5-flash")
                    parser         = StrOutputParser()
                    prompt         = ChatPromptTemplate.from_template(system_prompt)
                    retriever      = vector_store_retriver(docs, embeddings_model)

                    st.session_state.chain             = rag_chain(llm, parser, prompt, retriever)
                    st.session_state.uploaded_file_name = uploaded_file.name
                    st.session_state.chat_history      = []  # reset chat on new doc

                    os.unlink(tmp_path)  # cleanup temp file
                    st.success(f"✅ '{uploaded_file.name}' loaded!")
                except Exception as e:
                    st.error(f"❌ Error processing PDF: {e}")
    else:
        st.info("👆 Upload a PDF to start chatting")

    if st.button("🗑️ Clear Chat"):
        st.session_state.chat_history = []
        st.rerun()

# ── Chat UI ───────────────────────────────────────────────────────────────────
# Show history
for msg in st.session_state.chat_history:
    with st.chat_message(msg["role"]):
        st.write(msg["content"])

# Initial greeting if no history
if not st.session_state.chat_history:
    with st.chat_message("ai"):
        st.write("Hello 👋 Upload a PDF from the sidebar and ask me anything about it!")

# Input
user_input = st.chat_input(
    "Ask something about the document..." if st.session_state.chain else "Upload a PDF first..."
)

if user_input:
    if not st.session_state.chain:
        st.warning("⚠️ Please upload a PDF first from the sidebar.")
    else:
        # Show user message
        with st.chat_message("human"):
            st.write(user_input)
        st.session_state.chat_history.append({"role": "human", "content": user_input})

        # Get AI response
        with st.chat_message("ai"):
            with st.spinner("Thinking... 🧠"):
                try:
                    response = st.session_state.chain.invoke(user_input)
                    st.write(response)
                    st.session_state.chat_history.append({"role": "ai", "content": response})
                except Exception as e:
                    err = f"❌ Error: {e}"
                    st.error(err)
                    st.session_state.chat_history.append({"role": "ai", "content": err})