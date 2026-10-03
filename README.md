# Multimodal Academic Research Assistant (Multimodal RAG)

A production-grade Multimodal Retrieval-Augmented Generation (RAG) assistant designed specifically for complex academic research papers. It combines vision-language models for layout parsing, dense vector embeddings, deterministic metadata pinning, cross-encoder reranking, and grounded conversational generation with mathematical LaTeX rendering.

---

## 🌟 Why Multimodal Academic RAG?

Standard text-only RAG pipelines frequently fail on academic research papers because:
- **Embedded Visuals:** Critical findings reside in architectural diagrams, line plots, and qualitative case studies.
- **Complex Tables:** Multi-column benchmarks, ablations, and hyperparameter tables lose structural alignment when flattened into plain text.
- **Mathematical Equations:** Subscripts, superscripts, intervals, and symbols are often mangled by standard PDF text extractors into unreadable tokens.
- **Retrieval Bias:** Dense embeddings often match general semantic topics rather than explicit entities (e.g., retrieving generic prose about "training" instead of the exact "Figure 2" plot).

This application solves these challenges through a specialized multimodal extraction and hybrid retrieval pipeline.

---

## 🏗️ Architecture & Retrieval Pipeline

```
                                  [ Research Paper PDF ]
                                             │
                       ┌─────────────────────┴─────────────────────┐
                       ▼                                           ▼
             [ VLM Element Extractor ]                   [ Margin & Prose Extractor ]
           • PyMuPDF page rendering                     • Automatic header/footer margins
           • Pass 0: Caption inventory scan             • Exclusion of figure/table boxes
           • Pass 1: Parallel Gemini detection          • Hyphen stitching across lines
           • Pass 2: Targeted missed recovery           • Continuous clean prose text
                       │                                           │
         ┌─────────────┴─────────────┐                             │
         ▼                           ▼                             │
  [ Table Parser ]           [ Figure Analyzer ]                   │
• Gemini markdown tables   • Architecture tracing                  │
• Key-value column rows    • Plot axes & numerical trends          │
         │                           │                             │
         └─────────────┬─────────────┘                             │
                       ▼                                           ▼
            [ Cohere Embeddings ]                       [ Cohere Embeddings ]
            (embed-english-v3.0)                        (embed-english-v3.0)
                       │                                           │
                       └─────────────────────┬─────────────────────┘
                                             ▼
                                  [ Persistent ChromaDB ]
                                             │
 ┌───────────────────────────────────────────┴───────────────────────────────────────────┐
 │                                   QUERY PROCESSING                                   │
 └───────────────────────────────────────────┬───────────────────────────────────────────┘
                                             ▼
                             [ Conversational Query Rewriting ]
                             (Resolves multi-turn history)
                                             │
                       ┌─────────────────────┴─────────────────────┐
                       ▼                                           ▼
          [ Deterministic Pinning ]                   [ Diverse Vector Search ]
        • Regex scan for Figure/Table IDs            • ChromaDB MMR Search
        • Direct metadata database lookup            • fetch_k=40, k=8, lambda=0.7
        • Bypasses similarity competition            • Balanced prose/table/figure pool
                       │                                           │
                       │                                           ▼
                       │                                [ Cohere Cross-Encoder ]
                       │                                • rerank-english-v3.0
                       │                                • Deep query-document attention
                       │                                • Top-7 reranked candidates
                       │                                           │
                       └─────────────────────┬─────────────────────┘
                                             ▼
                                 [ Context Fusion & LLM ]
                               • Pinned Docs + Reranked Semantic Docs
                               • Cohere Command-R+ (temp=0, freq_penalty=0.2)
                               • Native LaTeX formatting ($\hat{s}$, $\hat{e}$, $h_t$)
                                             │
                                             ▼
                              [ Grounded Visual Filtering ]
                              • Matches element IDs in query & answer
                              • Prevents visual bleeding / phantom images
                                             │
                                             ▼
                             [ Streamlit Interactive Frontend ]
```

---

## 📁 Repository Structure

```
Multimodal-RAG/
├── assets/
│   ├── background.jpg             # App UI background styling
│   └── sample_papers/             # Built-in sample papers for quick testing
├── backend/
│   ├── __init__.py                # Package exports
│   ├── exceptions.py              # Domain-specific exceptions & stage wrapper
│   ├── extractor.py               # Dual-pass VLM extraction & clean prose parser
│   ├── pipeline.py                # End-to-end ingestion and indexing coordinator
│   ├── rag.py                     # Hybrid retriever, Cohere reranker & QA chain
│   └── vision.py                  # Gemini table transcription & figure analysis
├── uploaded_docs/
│   └── .gitkeep                   # Local cache directory for extractions
├── app.py                         # Streamlit interactive application
├── requirements.txt               # Production dependencies
├── .env.example                   # Environment configuration template
├── .gitignore                     # Git exclusion rules (secrets, caches, PDFs)
└── README.md                      # Documentation
```

---

## 🚀 Quickstart & Local Setup

### 1. Prerequisites
- Python 3.10+ (tested on Python 3.11 and 3.13)
- API Keys:
  - **Google Gemini API Key** (for multimodal layout detection & transcription): [Google AI Studio](https://aistudio.google.com/)
  - **Cohere API Key** (for embeddings, cross-encoder reranking & generation): [Cohere Dashboard](https://dashboard.cohere.com/)

### 2. Installation

```bash
# Clone the repository
git clone https://github.com/your-username/Multimodal-RAG.git
cd Multimodal-RAG

# Create and activate virtual environment
python -m venv .venv

# On Linux/macOS:
source .venv/bin/activate
# On Windows (PowerShell):
.venv\Scripts\Activate.ps1

# Install dependencies
pip install -r requirements.txt
```

### 3. Configure Credentials

Create a `.env` file from the provided template:

```bash
cp .env.example .env
```

Edit `.env` and add your API keys:

```ini
GOOGLE_API_KEY=your_google_gemini_api_key
COHERE_API_KEY=your_cohere_api_key
```

### 4. Run the Application

```bash
streamlit run app.py
```

The web interface will launch automatically at `http://localhost:8501`.

---

## 🌐 Deployment on Streamlit Community Cloud

This project is configured out-of-the-box for seamless deployment on [Streamlit Community Cloud](https://streamlit.io/cloud):

1. **Push your repository to GitHub** (ensure `.env` and `chroma_db/` are ignored by `.gitignore`).
2. Create a new app on Streamlit Cloud and link your repository.
3. In **App Settings** $\to$ **Secrets**, configure your environment variables:

```toml
GOOGLE_API_KEY = "your_google_gemini_api_key"
COHERE_API_KEY = "your_cohere_api_key"
```

4. Deploy! The application automatically checks `st.secrets` on startup.

---

## 🔬 Core Features & Guardrails

- **Zero-Loss Visual Grounding:** Uses a Caption Inventory Pass (Pass 0) directly from PDF text blocks, followed by parallel VLM detection (Pass 1) and targeted single-element recovery (Pass 2) to ensure no figures or tables are skipped.
- **Strict Grounded Visual Filtering:** Artifact images are only displayed if the element is explicitly cited in the answer or query, eliminating visual clutter and false positives.
- **Native LaTeX Mathematics:** Formats all formulas, intervals, and variables (e.g., $\hat{s}$, $\hat{e}$, $h_t$, $b'$, $R@0.5$, $\text{mIoU}$) using valid KaTeX syntax without hardcoded sanitizers.
- **Hallucination Prevention:** The model is constrained to retrieved context; if an element does not exist in the paper, it refuses gracefully rather than fabricating information.
- **Cross-Encoder Precision:** Re-scores semantic candidates with `rerank-english-v3.0` before passing context to the generator.

---

## 📄 License

This project is open-source and available under the [MIT License](LICENSE).
