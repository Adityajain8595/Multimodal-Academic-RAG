import json
import os
from pathlib import Path
import re
from typing import Any, Dict, List, Optional, Tuple

import chromadb
import cohere
from dotenv import load_dotenv
from langchain_chroma import Chroma
from langchain_cohere import ChatCohere, CohereEmbeddings
from langchain_core.documents import Document
from langchain_core.messages import AIMessage, HumanMessage
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_text_splitters import RecursiveCharacterTextSplitter

from .exceptions import handle_error

load_dotenv()

COLLECTION_NAME = "multimodal_rag_clean"
EMBED_MODEL_NAME = "embed-english-v3.0"
COHERE_CHAT_MODEL = "command-a-plus-05-2026"
COHERE_RERANK_MODEL = "rerank-english-v3.0"


def get_embedding_model() -> CohereEmbeddings:
    return CohereEmbeddings(model=EMBED_MODEL_NAME, client=None, async_client=None)


def load_vector_store(chroma_dir: str = "./chroma_db") -> Chroma:
    client = chromadb.PersistentClient(path=chroma_dir)
    return Chroma(
        client=client,
        collection_name=COLLECTION_NAME,
        embedding_function=get_embedding_model()
    )


@handle_error("Vector store indexing")
def build_index(
    page_docs: List[Dict],
    tabs: List[Dict],
    figs: List[Dict],
    chroma_dir: str = "./chroma_db",
    paper_name: Optional[str] = None
) -> Chroma:
    chroma_path = Path(chroma_dir)
    chroma_path.mkdir(parents=True, exist_ok=True)

    client = chromadb.PersistentClient(path=chroma_dir)
    try:
        client.delete_collection(COLLECTION_NAME)
    except Exception:
        pass

    splitter = RecursiveCharacterTextSplitter(chunk_size=700, chunk_overlap=80)
    prose_docs = [
        Document(page_content=doc["text"], metadata={"type": "prose", "page": doc["page"]})
        for doc in page_docs
    ]
    text_splits = splitter.split_documents(prose_docs)

    tab_splits = [
        Document(
            page_content=tab["parsed_text"],
            metadata={
                "type": "table",
                "id": tab["id"],
                "page": tab["page"],
                "caption": tab["caption"],
                "path": tab["path"]
            }
        )
        for tab in tabs
    ]

    fig_splits = [
        Document(
            page_content=f"Figure {fig['id']} on page {fig['page']}.\nCaption: {fig['caption']}\nVisual Analysis: {fig['summary']}",
            metadata={
                "type": "figure",
                "id": fig["id"],
                "page": fig["page"],
                "path": fig["path"]
            }
        )
        for fig in figs
    ]

    all_splits = text_splits + tab_splits + fig_splits
    vstore = Chroma(
        client=client,
        collection_name=COLLECTION_NAME,
        embedding_function=get_embedding_model()
    )
    if all_splits:
        vstore.add_documents(all_splits)

    if paper_name:
        meta_path = chroma_path / "paper_meta.json"
        meta_path.write_text(json.dumps({
            "paper_name": paper_name,
            "figs_count": len(figs),
            "tabs_count": len(tabs),
            "pages_count": len(page_docs)
        }, indent=2), encoding="utf-8")

    return vstore


def get_retriever(vstore: Chroma, k: int = 8):
    return vstore.as_retriever(
        search_type="mmr",
        search_kwargs={"k": k, "fetch_k": 40, "lambda_mult": 0.7}
    )


def cohere_rerank(query: str, docs: List[Document], top_n: int = 7) -> List[Document]:
    if not docs:
        return docs
    client = cohere.Client(os.getenv("COHERE_API_KEY"))
    response = client.rerank(
        model=COHERE_RERANK_MODEL,
        query=query,
        documents=[doc.page_content for doc in docs],
        top_n=min(top_n, len(docs)),
        return_documents=False
    )
    return [docs[res.index] for res in response.results]

def sanitize_answer(text: str) -> str:
    if not text:
        return text

    # 1. Normalize unicode spaces, zero-width characters, and non-breaking hyphens
    text = text.replace('\u202f', ' ').replace('\u00a0', ' ').replace('\u200b', '')
    text = text.replace('\u2011', '-').replace('\u2010', '-')

    # 2. Convert raw HTML breaks into clean Markdown line breaks
    text = re.sub(r'<br\s*/?>\s*([•\-\*])', r'\n\1', text, flags=re.I)
    text = re.sub(r'<br\s*/?>', r'\n', text, flags=re.I)

    # 3. Clean up double or unicode bullet markers (- •, * •, •) into standard Markdown (- )
    text = re.sub(r'(?m)^(\s*)[-*]\s*[•·]\s*', r'\1- ', text)
    text = re.sub(r'(?m)^(\s*)[•·]\s*', r'\1- ', text)

    # 4. Wrap naked LaTeX commands not already enclosed in $...$ or $$...$$
    def wrap_latex(segment: str) -> str:
        pattern = re.compile(r'(\\(?:[a-zA-Z]+)(?:\{[^{}]*\}|\[[^\[\]]*\]|\([^\(\)]*\)|[^\s\$\(\)]+)+)')
        matches = list(pattern.finditer(segment))
        if not matches:
            return segment
        out = []
        last_idx = 0
        for m in matches:
            start, end = m.span()
            out.append(segment[last_idx:start])
            raw_match = m.group(0)
            math_expr = raw_match.rstrip('.,;:')
            trail = raw_match[len(math_expr):]
            out.append(f"${math_expr}${trail}")
            last_idx = end
        out.append(segment[last_idx:])
        return "".join(out)

    tokens = text.split('$')
    for i in range(0, len(tokens), 2):
        tokens[i] = wrap_latex(tokens[i])

    return '$'.join(tokens)


def create_chain():
    llm = ChatCohere(
        model=COHERE_CHAT_MODEL,
        temperature=0,
        frequency_penalty=0.2
    )
    prompt = ChatPromptTemplate.from_messages([
        ("system",
         "You are an expert AI research assistant for academic papers.\n"
         "Answer ONLY from the retrieved context provided below.\n"
         "Be precise, structured, and factual. Cite page numbers, table IDs, and figure IDs directly.\n\n"
         "CRITICAL FORMATTING RULES (STRICT COMPLIANCE REQUIRED):\n"
         "1. MATHEMATICAL NOTATION & DELIMITERS:\n"
         "   - EVERY single mathematical formula, operation, function, equation, parameter, variable, or Greek letter MUST be explicitly enclosed in single dollar signs ($...$) for inline math or double dollar signs ($$...$$) for standalone block equations.\n"
         "   - Example: write $\\text{{Attention}}(Q, K, V) = \\text{{softmax}}\\left(\\frac{{QK^\\top}}{{\\sqrt{{d_k}}}}\\right)V$, NEVER bare \\text{{Attention}}...\n"
         "   - NEVER output bare/naked LaTeX commands (such as \\text{{...}}, \\frac{{...}}, \\sqrt{{...}}, \\top, \\sum, \\alpha, \\beta, \\cdot) outside of dollar sign delimiters.\n"
         "   - Always generate valid, clean LaTeX directly.\n"
         "2. NO RAW HTML:\n"
         "   - NEVER generate HTML tags such as <br>, <br/>, <span>, <div>, <p>, or <b> anywhere in your response.\n"
         "   - Always use standard Markdown line breaks and separate paragraphs with blank lines.\n"
         "3. CLEAN LISTS & BULLET POINTS:\n"
         "   - Format all bullet points using standard Markdown dashes ('- ').\n"
         "   - NEVER combine hyphens and bullet symbols (NEVER write '- •' or '* •').\n"
         "   - NEVER use the unicode bullet character '•'. Each bullet point must be on its own line starting with '- ' followed by a space.\n"
         "4. STANDARD ASCII TYPOGRAPHY:\n"
         "   - Use only standard ASCII spaces (space ' ') and standard ASCII hyphens ('-').\n"
         "   - NEVER output narrow non-breaking spaces (U+202F), non-breaking spaces (U+00A0), or non-breaking hyphens (U+2011).\n"
         "   - Always write standard text: 'Table 4', 'Figure 1', 'RNN-based', 'English-to-German'.\n"
         "5. TABLES & COMPARISONS:\n"
         "   - When presenting data from tables or comparing metrics, format them as clear Markdown tables.\n"
         "6. STRICT GROUNDING:\n"
         "   - Never claim a figure or table does not exist if it is in the context.\n"
         "   - If genuinely absent from the retrieved context, say: 'The retrieved context does not include Figure/Table X — try re-indexing or rephrasing your query.' Do not hallucinate external facts."),
        MessagesPlaceholder(variable_name="history"),
        ("human", "Retrieved Paper Context:\n\n<docs>{documents}</docs>\n\nUser Question: <question>{question}</question>")
    ])
    return prompt | llm | StrOutputParser()


def rewrite_query(query: str, history: List[Any]) -> str:
    if not history:
        return query
    llm = ChatCohere(model=COHERE_CHAT_MODEL, temperature=0)
    prompt = ChatPromptTemplate.from_messages([
        ("system", "You are a query reformulator. Given the chat history, reformulate the follow-up question into a concise standalone search query for document retrieval. Do NOT answer the question. Output ONLY the standalone search query without preamble or quotes."),
        MessagesPlaceholder(variable_name="history"),
        ("human", "Rewrite this follow-up question into a standalone search query: {question}\nStandalone search query:")
    ])
    chain = prompt | llm | StrOutputParser()
    res = chain.invoke({"history": history, "question": query})
    return res.strip() or query


@handle_error("RAG query execution")
def ask_paper(
    retriever: Any,
    query: str,
    history: Optional[List[Dict[str, str]]] = None,
    max_history: int = 5
) -> Tuple[str, List[Document], List[str], str]:
    msg_history = []
    if history:
        for item in history[-max_history:]:
            if item.get("role") == "user":
                msg_history.append(HumanMessage(content=item["content"]))
            elif item.get("role") == "assistant":
                msg_history.append(AIMessage(content=item["content"]))

    search_query = rewrite_query(query, msg_history) if msg_history else query

    pinned_docs: List[Document] = []
    collection = retriever.vectorstore._collection
    for match in re.finditer(r'\b(figure|fig\.?|table|tab\.?)\s*([0-9a-zA-Z]+)\b', query, re.I):
        kind = "figure" if match.group(1).lower().startswith("fig") else "table"
        raw_id = match.group(2)
        base_match = re.match(r'^(\d+)', raw_id)
        int_id = int(base_match.group(1)) if base_match else (int(raw_id) if raw_id.isdigit() else None)
        try:
            result = collection.get(where={"type": {"$eq": kind}}, include=["documents", "metadatas"])
            for doc_text, meta in zip(result.get("documents", []), result.get("metadatas", [])):
                stored_id = meta.get("id")
                if stored_id == int_id or str(stored_id).lower() == raw_id.lower() or (base_match and str(stored_id) == base_match.group(1)):
                    pinned_docs.append(Document(page_content=doc_text, metadata=meta or {}))
                    break
        except Exception:
            pass

    semantic_docs = retriever.invoke(search_query)
    pinned_texts = {doc.page_content for doc in pinned_docs}
    unpinned = [doc for doc in semantic_docs if doc.page_content not in pinned_texts]
    reranked = cohere_rerank(query, unpinned, top_n=7)
    merged_docs = list(pinned_docs) + reranked

    chain = create_chain()
    context = "\n\n---\n\n".join(doc.page_content for doc in merged_docs)
    raw_ans = chain.invoke({
        "documents": context,
        "question": query,
        "history": msg_history
    }).strip()
    ans = sanitize_answer(raw_ans)

    grounded_imgs: List[str] = []
    lower_ans = ans.lower()
    lower_query = query.lower()

    for doc in merged_docs:
        path = doc.metadata.get("path")
        elem_type = doc.metadata.get("type")
        elem_id = str(doc.metadata.get("id", ""))
        if not path or not elem_type or not elem_id or not Path(path).exists():
            continue

        prefix = "fig" if elem_type == "figure" else "tab"
        patterns = [
            rf'\b{elem_type}\s*{re.escape(elem_id)}(?:[a-zA-Z]|\([a-zA-Z]\))?\b',
            rf'\b{prefix}\.?\s*{re.escape(elem_id)}(?:[a-zA-Z]|\([a-zA-Z]\))?\b'
        ]
        is_refused = f"does not include {elem_type} {elem_id}" in lower_ans or f"does not include {prefix} {elem_id}" in lower_ans
        is_cited = any(re.search(p, lower_query) for p in patterns) or any(re.search(p, lower_ans) for p in patterns)

        if is_cited and not is_refused and path not in grounded_imgs:
            grounded_imgs.append(path)

    return ans, merged_docs, grounded_imgs, search_query

