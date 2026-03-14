"""
app.py — UltraShip Doc Intelligence
"""

import streamlit as st
from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter
from rag import ask_question
from extract import extract_structured_data, extract_from_text
import tempfile, os

# ── Page setup ────────────────────────────────────────────────────────────────
st.set_page_config(page_title="UltraShip Doc Intelligence", page_icon="🚚", layout="wide")
st.title("🚚 UltraShip Doc Intelligence")
st.caption("Upload a logistics document (PDF, DOCX, or TXT) and ask questions about it.")

# ── Load embedding model once ─────────────────────────────────────────────────
@st.cache_resource
def load_embeddings():
    return HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")

embeddings = load_embeddings()

# ── Session state init ────────────────────────────────────────────────────────
for key, default in [("vectorstore", None), ("doc_text", ""), ("history", []), ("file_id", None)]:
    if key not in st.session_state:
        st.session_state[key] = default


# ── Text extraction ───────────────────────────────────────────────────────────
def extract_text(uploaded_file) -> str:
    name = uploaded_file.name.lower()

    if name.endswith(".txt"):
        return uploaded_file.read().decode("utf-8", errors="ignore")

    elif name.endswith(".pdf"):
        try:
            import pdfplumber
        except ImportError:
            st.error("Run: pip install pdfplumber")
            return ""
        with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp:
            tmp.write(uploaded_file.read())
            tmp_path = tmp.name
        text = ""
        with pdfplumber.open(tmp_path) as pdf:
            for page in pdf.pages:
                text += (page.extract_text() or "") + "\n"
        os.unlink(tmp_path)
        return text

    elif name.endswith(".docx"):
        try:
            import docx
        except ImportError:
            st.error("Run: pip install python-docx")
            return ""
        with tempfile.NamedTemporaryFile(delete=False, suffix=".docx") as tmp:
            tmp.write(uploaded_file.read())
            tmp_path = tmp.name
        doc = docx.Document(tmp_path)
        os.unlink(tmp_path)
        return "\n".join([p.text for p in doc.paragraphs])

    st.error("Unsupported file type.")
    return ""


# ── Sidebar: Upload ───────────────────────────────────────────────────────────
with st.sidebar:
    st.header("📄 Upload Document")
    uploaded_file = st.file_uploader("Supported: PDF, DOCX, TXT", type=["txt", "pdf", "docx"])

    if uploaded_file:
        file_id = uploaded_file.name + str(uploaded_file.size)
        if st.session_state.file_id != file_id:
            with st.spinner("Processing document..."):
                raw_text = extract_text(uploaded_file)
                if raw_text.strip():
                    splitter = RecursiveCharacterTextSplitter(
                        chunk_size=500,
                        chunk_overlap=50,
                        separators=["\n\n", "\n", ".", " "]
                    )
                    chunks = splitter.create_documents([raw_text])
                    st.session_state.vectorstore = FAISS.from_documents(chunks, embeddings)
                    st.session_state.doc_text = raw_text
                    st.session_state.history = []
                    st.session_state.file_id = file_id
                else:
                    st.error("Could not extract text from this file.")

        if st.session_state.vectorstore:
            st.success(f"✅ Ready — {uploaded_file.name}")


# ── Tabs ──────────────────────────────────────────────────────────────────────
tab1, tab2 = st.tabs(["💬 Ask Questions", "🗂️ Extract Structured Data"])

# ── Tab 1: Q&A ────────────────────────────────────────────────────────────────
with tab1:
    if not st.session_state.vectorstore:
        st.info("👈 Upload a document to start asking questions.")
    else:
        for item in st.session_state.history:
            with st.chat_message("user"):
                st.write(item["question"])
            with st.chat_message("assistant"):
                st.write(item["answer"])
                conf = item["confidence"]
                color = "green" if conf >= 0.65 else "orange" if conf >= 0.45 else "red"
                st.markdown(f"**Confidence:** :{color}[{conf}]")
                if item.get("guardrail"):
                    st.warning(f"⚠️ Blocked — question not related to document.")
                elif item.get("source"):
                    with st.expander("📎 Source chunk"):
                        st.text(item["source"])

        question = st.chat_input("Ask something about your document...")
        if question:
            with st.spinner("Searching..."):
                result = ask_question(
                    question,
                    st.session_state.vectorstore,
                    all_doc_text=st.session_state.doc_text
                )
            st.session_state.history.append({"question": question, **result})
            st.rerun()

# ── Tab 2: Extraction ─────────────────────────────────────────────────────────
with tab2:
    if not st.session_state.vectorstore:
        st.info("👈 Upload a document first.")
    else:
        st.markdown("Automatically extract key shipment fields from the document.")
        if st.button("🔍 Run Extraction", type="primary"):
            with st.spinner("Extracting..."):
                data = extract_from_text(st.session_state.doc_text) if st.session_state.doc_text else extract_structured_data(st.session_state.vectorstore)

            st.subheader("Extracted Shipment Data")
            st.json(data)

            if isinstance(data, dict) and "error" not in data:
                found   = {k: v for k, v in data.items() if v is not None}
                missing = [k for k, v in data.items() if v is None]
                col1, col2 = st.columns(2)
                with col1:
                    st.success(f"✅ {len(found)} fields found")
                    for k, v in found.items():
                        st.write(f"**{k}:** {v}")
                with col2:
                    st.warning(f"❓ {len(missing)} fields not in document")
                    for k in missing:
                        st.write(f"• {k}: null")
