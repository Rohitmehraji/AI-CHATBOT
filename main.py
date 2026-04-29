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

# ── API Key ───────────────────────────────────────────────────────────────────
GOOGLE_API_KEY = st.secrets.get("GOOGLE_API_KEY", os.getenv("GOOGLE_API_KEY", ""))
if not GOOGLE_API_KEY:
    st.error("⚠️ GOOGLE_API_KEY not found. Add it in Streamlit Secrets.")
    st.stop()
os.environ["GOOGLE_API_KEY"] = GOOGLE_API_KEY

# ── Page Config ───────────────────────────────────────────────────────────────
st.set_page_config(page_title="RAG Chatbot – Rohit Kumar", page_icon="🤖", layout="centered")
st.title("🤖 RAG-Based AI Chatbot")
st.caption("Upload any PDF and chat with it using Google Gemini + LangChain + FAISS")

# ── Session State ─────────────────────────────────────────────────────────────
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []
if "chain" not in st.session_state:
    st.session_state.chain = None
if "uploaded_file_name" not in st.session_state:
    st.session_state.uploaded_file_name = None

# ── Sidebar ───────────────────────────────────────────────────────────────────
with st.sidebar:
    st.header("📄 Upload Document")
    uploaded_file = st.file_uploader("Upload a PDF", type=["pdf"])

    if uploaded_file:
        if uploaded_file.name != st.session_state.uploaded_file_name:
            with st.spinner("Processing PDF... ⏳"):
                try:
                    # FIX 1: getvalue() use karo, read() nahi
                    tmp_file = tempfile.NamedTemporaryFile(delete=False, suffix=".pdf")
                    tmp_file.write(uploaded_file.getvalue())
                    tmp_file.flush()
                    tmp_file.close()

                    pages = file_loader(tmp_file.name)

                    # FIX 2: empty pages check
                    if not pages:
                        st.error("❌ PDF empty hai ya scanned image hai — text-based PDF chahiye")
                        os.unlink(tmp_file.name)
                        st.stop()

                    splitter = split_doc()
                    docs     = splitter.split_documents(pages)

                    # FIX 3: empty chunks filter karo
                    docs = [d for d in docs if d.page_content.strip()]

                    if not docs:
                        st.error("❌ Koi valid content nahi mila PDF mein")
                        os.unlink(tmp_file.name)
                        st.stop()

                    embeddings_model = embedding(model_name="sentence-transformers/all-MiniLM-L6-v2")
                    llm              = ChatGoogleGenerativeAI(model="gemini-2.0-flash")
                    parser           = StrOutputParser()
                    prompt           = ChatPromptTemplate.from_template(system_prompt)
                    retriever        = vector_store_retriver(docs, embeddings_model)

                    st.session_state.chain              = rag_chain(llm, parser, prompt, retriever)
                    st.session_state.uploaded_file_name = uploaded_file.name
                    st.session_state.chat_history       = []
                    os.unlink(tmp_file.name)

                    st.success(f"✅ '{uploaded_file.name}' ready! ({len(docs)} chunks)")

                except Exception as e:
                    st.error(f"❌ Error: {e}")
    else:
        st.info("👆 Upload a PDF to start chatting")

    if st.button("🗑️ Clear Chat"):
        st.session_state.chat_history = []
        st.rerun()

# ── Chat ──────────────────────────────────────────────────────────────────────
for msg in st.session_state.chat_history:
    with st.chat_message(msg["role"]):
        st.write(msg["content"])

if not st.session_state.chat_history:
    with st.chat_message("ai"):
        st.write("Hello 👋 Upload a PDF from the sidebar and ask me anything about it!")

user_input = st.chat_input(
    "Ask something about the document..." if st.session_state.chain else "Upload a PDF first..."
)

if user_input:
    if not st.session_state.chain:
        st.warning("⚠️ Pehle PDF upload karo sidebar se.")
    else:
        with st.chat_message("human"):
            st.write(user_input)
        st.session_state.chat_history.append({"role": "human", "content": user_input})

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
