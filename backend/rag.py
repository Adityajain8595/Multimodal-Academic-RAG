import json
import os
import re
import shutil
from pathlib import Path
from typing import List, Dict, Tuple, Optional, Any
from dotenv import load_dotenv
import chromadb
import cohere
from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_chroma import Chroma
from langchain_cohere import ChatCohere, CohereEmbeddings
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.output_parsers import StrOutputParser
from langchain_core.messages import HumanMessage, AIMessage
from .exceptions import handle_error

load_dotenv()

# Build multimodal Chroma vector store
@handle_error("Vector store indexing")
def build_index(
    page_docs: List[Dict],
    tabs: List[Dict],
    figs: List[Dict],
    chroma_dir: str = "./chroma_db",
    paper_name: Optional[str] = None
) -> Chroma:
    chroma_path = Path(chroma_dir)
    if chroma_path.exists():
        shutil.rmtree(chroma_path, ignore_errors=True)
    chroma_path.mkdir(parents=True, exist_ok=True)

    client = chromadb.PersistentClient(path=chroma_dir)

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
    embed_model = CohereEmbeddings(model="embed-english-v3.0", client=None, async_client=None)

    vstore = Chroma(
        client=client,
        collection_name="multimodal_rag_clean",
        embedding_function=embed_model
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

# Initialize diverse MMR retriever
def get_retriever(vstore: Chroma, k: int = 8):
    return vstore.as_retriever(
        search_type="mmr",
        search_kwargs={"k": k, "fetch_k": 40, "lambda_mult": 0.7}
    )

# Rerank candidates with cross encoder
def cohere_rerank(query: str, docs: List[Document], top_n: int = 7) -> List[Document]:
    if not docs:
        return docs
    client = cohere.Client(os.getenv("COHERE_API_KEY"))
    response = client.rerank(
        model="rerank-english-v3.0",
        query=query,
        documents=[doc.page_content for doc in docs],
        top_n=min(top_n, len(docs)),
        return_documents=False
    )
    return [docs[res.index] for res in response.results]

# Setup conversational question answering chain
def create_chain():
    llm = ChatCohere(
        model="command-a-plus-05-2026",
        temperature=0,
        frequency_penalty=0.2
    )
    prompt = ChatPromptTemplate.from_messages([
        ("system",
         "You are an expert AI research assistant for academic papers.\n"
         "Answer ONLY from the retrieved context provided below.\n"
         "Be precise, structured, and factual. Cite page numbers, table IDs, and figure IDs directly.\n\n"
         "CRITICAL FORMATTING GUIDELINES:\n"
         "1. MATHEMATICAL NOTATION: Express all mathematical variables, parameters, symbols, intervals, "
         "tuples, and metric notations using valid LaTeX enclosed in single dollar signs for inline math "
         "(e.g., $\\hat{{s}}$, $\\hat{{e}}$, $\\hat{{y}} = (\\hat{{s}}, \\hat{{e}})$, $h_t$, $h_{{t+1}}$, $b$, $b'$, "
         "$C_0T_0$, $C_1T_1$, $R@0.5$, $\\text{{mIoU}}$, $\\text{{Macro }} S$) or double dollar signs for equations. "
         "NEVER write broken ASCII symbols, disjointed carats like 's ^', or loose spaces like 'h t'. "
         "Always generate well-formed LaTeX directly.\n"
         "2. TABLES & COMPARISONS: When presenting data from tables or comparing metrics, format them "
         "as clear Markdown tables. Do not generate repetitive loops or restatements.\n"
         "3. STRICT GROUNDING: Never claim a figure or table does not exist if it is in the context. "
         "If genuinely absent from the retrieved context, say: 'The retrieved context does not include "
         "Figure/Table X — try re-indexing or rephrasing your query.' Do not hallucinate external facts."),
        MessagesPlaceholder(variable_name="history"),
        ("human", "Retrieved Paper Context:\n\n<docs>{documents}</docs>\n\nUser Question: <question>{question}</question>")
    ])
    return prompt | llm | StrOutputParser()

# Reformulate follow up query
def rewrite_query(query: str, history: List[Any]) -> str:
    if not history:
        return query
    llm = ChatCohere(model="command-a-plus-05-2026", temperature=0)
    prompt = ChatPromptTemplate.from_messages([
        ("system", "You are a query reformulator. Given the chat history, reformulate the follow-up question into a concise standalone search query for document retrieval. Do NOT answer the question. Output ONLY the standalone search query without preamble or quotes."),
        MessagesPlaceholder(variable_name="history"),
        ("human", "Rewrite this follow-up question into a standalone search query: {question}\nStandalone search query:")
    ])
    chain = prompt | llm | StrOutputParser()
    res = chain.invoke({"history": history, "question": query})
    return res.strip() or query

# Execute hybrid multimodal paper query
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

    # Deterministic metadata lookup for elements
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

    # Retrieve and rerank semantic candidates
    semantic_docs = retriever.invoke(search_query)
    pinned_texts = {doc.page_content for doc in pinned_docs}
    unpinned = [doc for doc in semantic_docs if doc.page_content not in pinned_texts]
    reranked = cohere_rerank(query, unpinned, top_n=7)
    merged_docs = list(pinned_docs) + reranked

    # Generate answer with language model
    chain = create_chain()
    context = "\n\n---\n\n".join(doc.page_content for doc in merged_docs)
    ans = chain.invoke({
        "documents": context,
        "question": query,
        "history": msg_history
    }).strip()

    # Filter strictly grounded visual sources
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
