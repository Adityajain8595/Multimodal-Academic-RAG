# Multimodal Academic Research Assistant (Multimodal RAG)

A production-grade, layout-aware Retrieval-Augmented Generation (RAG) system engineered for deep parsing, indexing, and reasoning over multimodal academic documents—including dense mathematical derivations, multi-panel architectural schematics, benchmark tables, and experimental plots.

---

## 1. Overview & Architectural Rationale

Academic literature presents significant hurdles for standard text-only RAG pipelines:
- **Visual Information Density:** Core contributions, model architectures, and experimental proofs frequently reside inside figures, diagrams, and plots rather than body paragraphs.
- **Tabular Structural Degradation:** Standard text extraction flattens multi-column tables, benchmarks, and ablation studies into disorganized token sequences, destroying row-column relationships.
- **Mathematical Expression Corruption:** Formulae, subscripts, superscripts, and Greek letters are routinely mangled by naive OCR or PDF scrapers into disjointed characters.
- **Semantic Dilution in Dense Retrieval:** Embedding-only similarity often matches high-level topical prose while overlooking the exact figure or table containing the empirical answer.

This system resolves these failure modes through an end-to-end multimodal pipeline that decouples visual elements from textual prose, enriches visual artifacts with analytical descriptions and structured transcriptions, and retrieves information via a hybrid multi-stage engine.

---

## 2. System Architecture

```
                                      [ Research Paper PDF ]
                                                 │
                         ┌───────────────────────┴───────────────────────┐
                         ▼                                               ▼
               [ Visual Layout Track ]                         [ Textual Prose Track ]
            • High-DPI Page Rendering                       • Dynamic Header/Footer Margin Bounds
            • Pass 0: Regex Caption Inventory               • Bounding Box Occlusion Masking
            • Pass 1: Parallel VLM Bounding Box Detection   • Cross-Line Dehyphenation & Stitching
            • Pass 2: Targeted Missed Element Recovery      • Clean Paragraph Prose Extraction
                         │                                               │
           ┌─────────────┴─────────────┐                                 │
           ▼                           ▼                                 │
    [ Table Parser ]          [ Figure Analyzer ]                        │
  • Full visual cell parse   • Architecture tracing                      │
  • Structured Markdown rows • Plots, axes, & metrics                    │
           │                           │                                 │
           └─────────────┬─────────────┘                                 │
                         ▼                                               ▼
              [ Cohere Embeddings ]                           [ Cohere Embeddings ]
              (embed-english-v3.0)                            (embed-english-v3.0)
                         │                                               │
                         └───────────────────────┬───────────────────────┘
                                                 ▼
                                      [ Persistent ChromaDB ]
                                                 │
  ┌──────────────────────────────────────────────┴──────────────────────────────────────────────┐
  │                                      RETRIEVAL PIPELINE                                     │
  └──────────────────────────────────────────────┬──────────────────────────────────────────────┘
                                                 ▼
                                [ Multi-Turn Query Rewriter ]
                                (Resolves conversational context)
                                                 │
                         ┌───────────────────────┴───────────────────────┐
                         ▼                                               ▼
            [ Deterministic Pinning ]                       [ Semantic MMR Retrieval ]
          • Entity scanner (Figure/Table IDs)             • Diverse Chroma vector search
          • Direct metadata database lookup               • fetch_k=40, k=8, lambda=0.7
          • Zero semantic dilution                        • Balanced prose, table, & figure pool
                         │                                               │
                         │                                               ▼
                         │                                    [ Cross-Encoder Reranker ]
                         │                                    • Cohere rerank-english-v3.0
                         │                                    • Deep query-document attention
                         │                                    • Top candidate re-scoring
                         │                                               │
                         └───────────────────────┬───────────────────────┘
                                                 ▼
                                     [ Unified Context Fusion ]
                                  (Pinned Elements + Reranked Docs)
                                                 │
                                                 ▼
                                  [ Grounded Generator (LLM) ]
                                • Cohere Command Models (temp=0)
                                • Mathematical notation in standard LaTeX ($...$, $$...$$)
                                • Strict citation of pages & element identifiers
                                                 │
                                                 ▼
                                  [ Grounded Visual Filtering ]
                                • Validates cited elements in answer
                                • Suppresses phantom or unrelated imagery
                                                 │
                                                 ▼
                                 [ Streamlit Interactive Interface ]
```

---

## 3. Deep Pipeline Walkthrough

### 3.1 Dual-Path Document Ingestion

1. **Pass 0 — Caption Inventory Scan:**
   - Pre-scans the PDF text blocks using PyMuPDF to catalog every declared `Figure` and `Table` along with its exact identifier and full caption text.
   - Establishes a ground-truth registry of visual assets present in the document.

2. **Pass 1 — Parallel Visual Language Model (VLM) Detection:**
   - Renders candidate pages to high-resolution bitmaps and submits them to Google Gemini with a structured JSON schema.
   - Detects tightly bounded 2D boxes `[ymin, xmin, ymax, xmax]` normalized to a `[0, 1000]` coordinate space.

3. **Pass 2 — Targeted Missed Element Recovery:**
   - Reconciles the detected boxes against the Pass 0 inventory.
   - If any figure or table was overlooked during the global pass, an isolated, targeted query focuses specifically on recovering that element's bounding coordinates.

4. **Spatial Occlusion Masking & Clean Prose Extraction:**
   - Computes dynamic page margins to strip running headers, page numbers, conference footers, and arXiv banners.
   - Maps the 2D bounding boxes of all extracted figures and tables onto the PyMuPDF document coordinates and excludes overlapping text blocks.
   - Dehyphenates wrapped words across line breaks and stitches continuous paragraphs, producing clean, uncorrupted prose.

5. **Visual Enrichment (Tables & Figures):**
   - **Tables:** Processed by a visual parser that transcribes columns, headers, units, and values into clean, structured Markdown records.
   - **Figures:** Processed by an analytical vision prompt that documents system flow, arrows, inputs, outputs, plot axes, trends, and qualitative comparisons.

### 3.2 Vector Indexing & Lifecycle Management

- **Embeddings:** Documents, parsed tables, and figure visual summaries are vectorized using Cohere `embed-english-v3.0` (1024-dimensional semantic space).
- **Metadata Association:** Every chunk is enriched with its structural origin (`type`, `id`, `page`, `caption`, and local image path).
- **Clean State Guarantees:** Prior to indexing a new document, the system purges any existing ChromaDB directory and unlinks legacy asset folders, preventing cross-document contamination and orphaned segment accumulation.

### 3.3 Hybrid 3-Stage Retrieval Engine

1. **Stage 1: Deterministic Element Pinning**
   - User queries (and rewritten queries) are scanned for explicit element markers (e.g., `Figure 2`, `Table 1`, sub-panel letter notations).
   - Matching items are pulled directly from the vector store via exact metadata filters. This guarantees that direct questions about specific artifacts bypass semantic competition.

2. **Stage 2: Diverse Dense Retrieval (MMR)**
   - Unpinned semantic context is retrieved using Maximal Marginal Relevance (MMR) (`k=8`, `fetch_k=40`, `lambda_mult=0.7`).
   - Ensures a balanced pool across continuous text, table entries, and visual figure analyses without redundant clustering.

3. **Stage 3: Cross-Encoder Neural Reranking**
   - Retrieved semantic candidates are scored against the query using Cohere `rerank-english-v3.0`.
   - Re-orders candidates based on deep cross-attention, eliminating false positives and ensuring high information density in the top-ranked context.

### 3.4 Grounded Generation & Guardrails

- **Zero-Hallucination Prompting:** The language model operates with zero temperature and a frequency penalty, strictly bound to the supplied context.
- **Mathematical Formatting:** Mathematical notation, parameters, intervals, and formulas are natively formatted in standard LaTeX delimiters (`$` for inline math, `$$` for display equations), allowing Streamlit's KaTeX engine to render them without post-processing sanitizers.
- **Grounded Visual Verification:** Extracted images are only rendered in the user interface if their identifier is explicitly cited in the answer or requested in the query, preventing visual clutter.

---

## 4. Repository Structure

```
Multimodal-RAG/
├── assets/
│   ├── background.jpg             # Interface background aesthetic
│   └── sample_papers/             # Curated sample research papers
├── backend/
│   ├── __init__.py                # Package exports
│   ├── exceptions.py              # Domain exception handling & stage wrappers
│   ├── extractor.py               # Dual-pass layout detection & clean prose extraction
│   ├── pipeline.py                # End-to-end ingestion and indexing coordinator
│   ├── rag.py                     # Hybrid retrieval, reranking, and generation chain
│   └── vision.py                  # Gemini table transcription & figure analysis
├── uploaded_docs/
│   └── .gitkeep                   # Scratch folder for active PDF & visual crops
├── app.py                         # Streamlit interactive application
├── requirements.txt               # Pinned project dependencies
├── .env.example                   # Environment configuration template
├── .gitignore                     # Git ignore rules (secrets, caches, vector DBs)
└── README.md                      # System architecture documentation
```

---

## 5. Getting Started

### 5.1 Prerequisites
- Python 3.10 or higher.
- API Keys:
  - **Google Gemini API Key:** For multimodal layout detection, table transcription, and figure analysis ([Google AI Studio](https://aistudio.google.com/)).
  - **Cohere API Key:** For dense embeddings, cross-encoder reranking, and grounded response generation ([Cohere Dashboard](https://dashboard.cohere.com/)).

### 5.2 Installation

```bash
# Clone the repository
git clone https://github.com/your-username/Multimodal-RAG.git
cd Multimodal-RAG

# Create a virtual environment
python -m venv .venv

# Activate the virtual environment
# On Linux/macOS:
source .venv/bin/activate
# On Windows (PowerShell):
.venv\Scripts\Activate.ps1

# Install dependencies
pip install -r requirements.txt
```

### 5.3 Configuration

Create a `.env` file in the project root:

```bash
cp .env.example .env
```

Populate the keys in `.env`:

```ini
GOOGLE_API_KEY=your_google_gemini_api_key
COHERE_API_KEY=your_cohere_api_key
```

### 5.4 Running the Application

```bash
streamlit run app.py
```

Access the interface in your browser at `http://localhost:8501`.

---

## 6. Deployment on Streamlit Community Cloud

1. Push your repository to GitHub. Ensure that `.env` and `chroma_db/` remain excluded via `.gitignore`.
2. Link your repository in [Streamlit Community Cloud](https://share.streamlit.io/).
3. Navigate to **App Settings** $\to$ **Secrets** and configure your API keys:

```toml
GOOGLE_API_KEY = "your_google_gemini_api_key"
COHERE_API_KEY = "your_cohere_api_key"
```

4. Launch the app. The pipeline automatically reads credentials from the environment.

---

## 7. License

This project is licensed under the [MIT License](LICENSE).
