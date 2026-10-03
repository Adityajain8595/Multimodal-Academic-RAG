import base64
import json
import os
import shutil
import time
from pathlib import Path
from dotenv import load_dotenv
import streamlit as st
import chromadb
from langchain_chroma import Chroma
from langchain_cohere import CohereEmbeddings

# Import modular backend components
from backend.pipeline import run_pipeline
from backend.rag import ask_paper, get_retriever
from backend.exceptions import RAGError

# Load environment configuration
load_dotenv()

# Page setup and layout
st.set_page_config(
    page_title="Multimodal Academic RAG",
    page_icon="🔬",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Apply background styling
def get_background_css(bg_file: str = "assets/background.jpg") -> str:
    if not Path(bg_file).exists():
        return ""
    data = base64.b64encode(Path(bg_file).read_bytes()).decode()
    return f"""
    <style>
    .stApp {{
        background: linear-gradient(rgba(10, 15, 29, 0.88), rgba(10, 15, 29, 0.94)),
                    url("data:image/jpeg;base64,{data}") no-repeat center center fixed;
        background-size: cover;
    }}
    </style>
    """

st.markdown(get_background_css(), unsafe_allow_html=True)

# Initialize application session state
if "messages" not in st.session_state:
    st.session_state.messages = []
if "retriever" not in st.session_state:
    st.session_state.retriever = None
if "current_paper" not in st.session_state:
    st.session_state.current_paper = "No paper indexed yet"
if "selected_source" not in st.session_state:
    st.session_state.selected_source = None
if "selected_name" not in st.session_state:
    st.session_state.selected_name = None

# Restore existing Chroma database
def load_existing_store(chroma_dir: str = "./chroma_db"):
    db_path = Path(chroma_dir)
    if db_path.exists() and st.session_state.retriever is None:
        try:
            embed_model = CohereEmbeddings(model="embed-english-v3.0", client=None, async_client=None)
            client = chromadb.PersistentClient(path=chroma_dir)
            vstore = Chroma(
                client=client,
                collection_name="multimodal_rag_clean",
                embedding_function=embed_model
            )
            st.session_state.retriever = get_retriever(vstore, k=8)
            meta_path = db_path / "paper_meta.json"
            if meta_path.exists():
                meta = json.loads(meta_path.read_text(encoding="utf-8"))
                st.session_state.current_paper = meta.get("paper_name", "Indexed Paper")
        except Exception:
            st.session_state.retriever = None

load_existing_store()

# Sidebar paper management controls
with st.sidebar:
    st.title("🔬 Paper Control")
    st.markdown("Multimodal RAG equipped for deep paper layout, figures, and tables comprehension.")
    st.markdown("---")
    
    st.subheader("📄 Upload Paper")
    uploaded_pdf = st.file_uploader("Upload PDF Paper", type=["pdf"])

    if uploaded_pdf is not None:
        upload_folder = Path("uploaded_docs")
        upload_folder.mkdir(parents=True, exist_ok=True)
        if st.session_state.selected_name != uploaded_pdf.name:
            cache_folder = upload_folder / "extracted_figures"
            if cache_folder.exists():
                for f in cache_folder.glob("*"):
                    if f.is_file():
                        try:
                            f.unlink(missing_ok=True)
                        except Exception:
                            pass

        dest_file = upload_folder / "document.pdf"
        try:
            dest_file.write_bytes(uploaded_pdf.getbuffer())
        except PermissionError:
            dest_file = upload_folder / f"doc_{int(time.time())}.pdf"
            dest_file.write_bytes(uploaded_pdf.getbuffer())
        st.session_state.selected_source = str(dest_file)
        st.session_state.selected_name = uploaded_pdf.name

    st.markdown("##### Or Select Sample Paper")
    col_a, col_b = st.columns(2)
    if col_a.button("Diffusibility", use_container_width=True):
        upload_folder = Path("uploaded_docs")
        upload_folder.mkdir(parents=True, exist_ok=True)
        src_path = Path("assets/sample_papers/Diffusibility_High_Dim_Latents.pdf")
        dest_file = upload_folder / "document.pdf"
        if src_path.exists():
            shutil.copy2(src_path, dest_file)
            st.session_state.selected_source = str(dest_file)
            st.session_state.selected_name = "On the Diffusibility of High-Dimensional Latents"

    if col_b.button("Attention", use_container_width=True):
        upload_folder = Path("uploaded_docs")
        upload_folder.mkdir(parents=True, exist_ok=True)
        src_path = Path("assets/sample_papers/Attention_Is_All_You_Need.pdf")
        dest_file = upload_folder / "document.pdf"
        if src_path.exists():
            shutil.copy2(src_path, dest_file)
            st.session_state.selected_source = str(dest_file)
            st.session_state.selected_name = "Attention Is All You Need"

    target_src = st.session_state.selected_source
    target_title = st.session_state.selected_name

    if target_src and target_title:
        st.info(f"Target: **{target_title}**")
        if st.button("🚀 Process & Index Paper", type="primary", use_container_width=True):
            with st.status(f"Ingesting '{target_title}'...", expanded=True) as status_box:
                progress_bar = st.progress(0.0)

                def on_progress(msg: str, frac: float):
                    status_box.write(msg)
                    progress_bar.progress(frac)

                try:
                    result = run_pipeline(
                        pdf_source=target_src,
                        upload_dir="uploaded_docs",
                        chroma_dir="./chroma_db",
                        paper_name=target_title,
                        status_cb=on_progress
                    )
                    st.session_state.retriever = result["retriever"]
                    st.session_state.current_paper = target_title
                    st.session_state.messages = []
                    status_box.update(label="Ready for Q&A!", state="complete", expanded=False)
                    st.success(f"Indexed {len(result['figs'])} figures and {len(result['tabs'])} tables!")
                    st.rerun()
                except RAGError as err:
                    status_box.update(label="Ingestion failed", state="error")
                    st.error(str(err))

    st.markdown("---")
    st.markdown(f"**Active Paper:** `{st.session_state.current_paper}`")
    if st.button("🧹 Clear Chat History", use_container_width=True):
        st.session_state.messages = []
        st.rerun()

# Main conversational chat interface
st.title("Multimodal Academic Research Assistant")
st.caption(f"Currently querying: **{st.session_state.current_paper}**")

# Render persisted conversation history
for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])
        if msg.get("images"):
            valid_paths = [p for p in msg["images"] if Path(p).exists()]
            if valid_paths:
                st.markdown("##### 🖼️ Grounded Visual Sources:")
                num_cols = min(3, len(valid_paths))
                cols = st.columns(num_cols)
                for idx, img_path in enumerate(valid_paths):
                    cols[idx % num_cols].image(img_path, caption=Path(img_path).name, use_container_width=True)

        if msg.get("docs"):
            top_passages = msg["docs"][:3]
            with st.expander("🔍 Top Retrieved Passages & Metadata"):
                if msg.get("search_query") and msg.get("search_query") != msg.get("user_query"):
                    st.caption(f"**Search Query:** `{msg['search_query']}`")
                passages_md = []
                for idx, (doc_type, doc_page, doc_snippet) in enumerate(top_passages, 1):
                    clean_text = " ".join(doc_snippet.strip().split())
                    if len(clean_text) > 180:
                        clean_text = clean_text[:180] + "..."
                    passages_md.append(f"{idx}. **[{doc_type.upper()} · p.{doc_page}]** {clean_text}")
                st.markdown("\n".join(passages_md))

# Handle new user query
if prompt_text := st.chat_input("Ask a question about figures, tables, math, or methodology..."):
    with st.chat_message("user"):
        st.markdown(prompt_text)
    st.session_state.messages.append({"role": "user", "content": prompt_text})

    with st.chat_message("assistant"):
        if st.session_state.retriever is None:
            warning_msg = "Please process or select a research paper from the sidebar first."
            st.warning(warning_msg)
            st.session_state.messages.append({"role": "assistant", "content": warning_msg})
        else:
            with st.spinner("Analyzing paper context and visuals..."):
                try:
                    ans, docs, img_paths, search_query = ask_paper(
                        st.session_state.retriever,
                        prompt_text,
                        history=st.session_state.messages[:-1],
                        max_history=5
                    )
                    valid_grounded_imgs = [p for p in img_paths if Path(p).exists()]
                    formatted_docs = [
                        (d.metadata.get("type", "unknown"), d.metadata.get("page", "?"), d.page_content)
                        for d in docs
                    ]

                    st.session_state.messages.append({
                        "role": "assistant",
                        "content": ans,
                        "images": valid_grounded_imgs,
                        "docs": formatted_docs,
                        "search_query": search_query,
                        "user_query": prompt_text
                    })
                    st.rerun()
                except RAGError as err:
                    st.error(f"Error during retrieval: {err}")
