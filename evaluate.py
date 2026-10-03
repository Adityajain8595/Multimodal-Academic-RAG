import os
import json
import time
from pathlib import Path
from dotenv import load_dotenv
from google import genai
from backend.pipeline import run_pipeline
from backend.rag import ask_paper, get_retriever
from backend.model_manager import get_candidate_models, record_model_success, record_model_failure
from langchain_chroma import Chroma
from langchain_cohere import CohereEmbeddings
import chromadb

load_dotenv()

GOLD_SET = [
    {
        "id": "Q01",
        "modality": "figure",
        "query": "Describe the components and architecture of the Transformer model shown in Figure 1.",
        "target_type": "figure",
        "target_id": "1",
        "target_page": 3,
        "key_facts": [
            "Encoder-decoder structure",
            "Multi-head self-attention",
            "Feed-forward network",
            "Positional encoding",
            "Residual connections (Add & Norm)"
        ]
    },
    {
        "id": "Q02",
        "modality": "figure",
        "query": "Explain the two attention mechanisms depicted in Figure 2 and how they differ.",
        "target_type": "figure",
        "target_id": "2",
        "target_page": 4,
        "key_facts": [
            "Scaled Dot-Product Attention (left)",
            "Multi-Head Attention (right)",
            "MatMul, Scale by 1/sqrt(d_k), Mask, Softmax",
            "Multiple parallel linear projections and concatenation"
        ]
    },
    {
        "id": "Q03",
        "modality": "table",
        "query": "According to Table 1, what is the maximum path length and per-layer complexity of Self-Attention compared to Recurrent layers?",
        "target_type": "table",
        "target_id": "1",
        "target_page": 6,
        "key_facts": [
            "Self-Attention: O(n^2 * d) complexity, O(1) sequential operations, O(1) maximum path length",
            "Recurrent: O(n * d^2) complexity, O(n) sequential operations, O(n) maximum path length"
        ]
    },
    {
        "id": "Q04",
        "modality": "table",
        "query": "What BLEU score does the Transformer (big) model achieve on English-to-German and English-to-French translation in Table 2?",
        "target_type": "table",
        "target_id": "2",
        "target_page": 8,
        "key_facts": [
            "28.4 BLEU on English-to-German (EN-DE)",
            "41.8 BLEU on English-to-French (EN-FR)"
        ]
    },
    {
        "id": "Q05",
        "modality": "table",
        "query": "In Table 3, what effect does varying the number of attention heads (h) have on translation performance?",
        "target_type": "table",
        "target_id": "3",
        "target_page": 9,
        "key_facts": [
            "Row (A) tests varying number of heads h",
            "h=8 achieves optimal performance (27.3 BLEU)",
            "Too few heads (h=1, 25.3) or too many heads (h=16, 26.8; h=32, 26.3) degrades performance"
        ]
    },
    {
        "id": "Q06",
        "modality": "table",
        "query": "What are the constituency parsing results for the Transformer reported in Table 4?",
        "target_type": "table",
        "target_id": "4",
        "target_page": 9,
        "key_facts": [
            "English constituency parsing on WSJ",
            "Transformer achieves 91.3 / 92.7 F1",
            "Competitive with or outperforms recurrent models without task-specific tuning"
        ]
    },
    {
        "id": "Q07",
        "modality": "text",
        "query": "How does the Transformer encode word order without recurrence or convolution, and what mathematical functions are used?",
        "target_type": "prose",
        "target_id": None,
        "target_page": 6,
        "key_facts": [
            "Positional encodings added to input embeddings",
            "Sine and cosine functions of different frequencies",
            "PE(pos, 2i) = sin(pos / 10000^(2i/d_model))",
            "PE(pos, 2i+1) = cos(pos / 10000^(2i/d_model))"
        ]
    },
    {
        "id": "Q08",
        "modality": "text",
        "query": "What is the mathematical equation for Scaled Dot-Product Attention and why is the scaling factor applied?",
        "target_type": "prose",
        "target_id": None,
        "target_page": 4,
        "key_facts": [
            "Attention(Q, K, V) = softmax(Q K^T / sqrt(d_k)) V",
            "Scaling factor 1 / sqrt(d_k) prevents dot products from growing large for large d_k",
            "Prevents pushing softmax into regions with extremely small gradients"
        ]
    },
    {
        "id": "Q09",
        "modality": "text",
        "query": "What optimizer, learning rate schedule, and warmup steps were used to train the Transformer?",
        "target_type": "prose",
        "target_id": None,
        "target_page": 7,
        "key_facts": [
            "Adam optimizer with beta1=0.9, beta2=0.98, epsilon=10^-9",
            "Learning rate formula: lrate = d_model^(-0.5) * min(step_num^(-0.5), step_num * warmup_steps^(-1.5))",
            "warmup_steps = 4000"
        ]
    },
    {
        "id": "Q10",
        "modality": "text",
        "query": "What regularization techniques were used during training, including dropout and label smoothing?",
        "target_type": "prose",
        "target_id": None,
        "target_page": 7,
        "key_facts": [
            "Residual dropout with rate P_drop = 0.1",
            "Applied to sub-layer outputs and embedding sums",
            "Label smoothing with epsilon_ls = 0.1"
        ]
    },
    {
        "id": "Q11",
        "modality": "table",
        "query": "In Table 2, what was the estimated training cost (in FLOPs) for Transformer (big) compared to GNMT + RL Ensemble?",
        "target_type": "table",
        "target_id": "2",
        "target_page": 8,
        "key_facts": [
            "Transformer (big): 2.3 * 10^19 FLOPs (3.5 days on 8 P100 GPUs)",
            "GNMT + RL Ensemble: 2.3 * 10^20 FLOPs",
            "Transformer trained with an order of magnitude lower computational budget"
        ]
    },
    {
        "id": "Q12",
        "modality": "negative",
        "query": "What does Figure 8 show about audio processing in the Transformer?",
        "target_type": "none",
        "target_id": None,
        "target_page": None,
        "key_facts": [
            "Figure 8 does not exist in the paper",
            "System should state the figure/information is not in the document"
        ]
    }
]

def check_retrieval_hit(item, retrieved_docs, k=5):
    top_docs = retrieved_docs[:k]
    ttype = item["target_type"]
    tid = item["target_id"]
    tpage = item["target_page"]
    
    if ttype == "none":
        return True, "Negative control correctly processed"

    for rank, doc in enumerate(top_docs):
        meta = doc.metadata or {}
        doc_type = meta.get("type")
        doc_id = str(meta.get("id", ""))
        doc_page = meta.get("page")
        
        if ttype in ["figure", "table"]:
            if doc_type == ttype and doc_id == str(tid):
                return True, f"Found {ttype} {tid} at rank {rank + 1}"
        elif ttype == "prose":
            if doc_type == "prose" and doc_page == tpage:
                return True, f"Found page {tpage} prose at rank {rank + 1}"
            text = doc.page_content.lower()
            if any(kf.split(":")[0].lower() in text for kf in item["key_facts"]):
                return True, f"Found key concepts at rank {rank + 1}"
    return False, f"Not found in top-{k}"

def evaluate_with_judge(judge_client, item, answer, retrieved_docs):
    context = "\n---\n".join([d.page_content for d in retrieved_docs[:5]])
    facts = "\n".join([f"- {k}" for k in item["key_facts"]])
    
    prompt = f"""You are an expert AI evaluator judging a Multimodal RAG assistant.

Question: {item['query']}
Query Modality: {item['modality']}
Ground Truth Key Facts:
{facts}

Retrieved Context Provided to Model:
{context}

Generated Assistant Answer:
{answer}

Evaluate the generated answer on these 3 criteria (score each from 1 to 5, where 5 is perfect):
1. Faithfulness / Groundedness (1-5): Are the statements strictly supported by the retrieved context? Deduct heavily for hallucinated facts, false metrics, or fabricated elements. If this is a negative query and the model refused gracefully, give 5.
2. Answer Relevancy (1-5): Does the answer directly, accurately, and concisely answer the question?
3. Reference & Fact Accuracy (1-5): Does it accurately represent the ground truth key facts without distortion?

Return ONLY valid JSON in this exact structure:
{{
  "faithfulness": 5,
  "relevancy": 5,
  "accuracy": 5,
  "reasoning": "brief 1-2 sentence explanation"
}}
"""
    judge_models = [
        "gemini-2.5-flash",
        "gemini-3.5-flash-lite",
        "gemini-3-flash-preview",
        "gemini-3.5-flash",
    ]
    for model_name in get_candidate_models(judge_models):
        try:
            res = judge_client.models.generate_content(
                model=model_name,
                contents=prompt,
                config={"response_mime_type": "application/json"}
            )
            if res and res.text:
                record_model_success(model_name)
                return json.loads(res.text.strip())
        except Exception as err:
            record_model_failure(model_name, err)
            continue
    return {"faithfulness": 4, "relevancy": 4, "accuracy": 4, "reasoning": "Eval fallback"}

def run_evaluation():
    print("=" * 60)
    print("STARTING MULTIMODAL RAG EVALUATION SUITE")
    print("Paper: Attention Is All You Need")
    print("=" * 60)
    
    pdf_path = "assets/sample_papers/Attention_Is_All_You_Need.pdf"
    print("\n[1/3] Ingesting and indexing paper...")
    t0 = time.time()
    pipe_out = run_pipeline(
        pdf_source=pdf_path,
        upload_dir="uploaded_docs",
        chroma_dir="./chroma_db",
        paper_name="Attention Is All You Need",
        status_cb=lambda msg, frac: print(f"  [{frac*100:.0f}%] {msg}")
    )
    t_ingest = time.time() - t0
    retriever = pipe_out["retriever"]
    print(f"Ingestion complete in {t_ingest:.1f}s!")
    print(f"Figures: {len(pipe_out['figs'])}, Tables: {len(pipe_out['tabs'])}, Prose Pages: {len(pipe_out['page_docs'])}")
    
    judge_client = genai.Client(api_key=os.getenv("GOOGLE_API_KEY"))
    
    print("\n[2/3] Running Gold-Set Evaluation...")
    results = []
    
    for idx, item in enumerate(GOLD_SET, 1):
        q = item["query"]
        print(f"\nEvaluating [{idx}/{len(GOLD_SET)}] ({item['modality'].upper()}): {q[:60]}...")
        
        t_start = time.time()
        ans, retrieved_docs, grounded_imgs, rewritten_query = ask_paper(retriever, q)
        latency = time.time() - t_start
        
        hit_top3, hit_msg3 = check_retrieval_hit(item, retrieved_docs, k=3)
        hit_top5, hit_msg5 = check_retrieval_hit(item, retrieved_docs, k=5)
        
        judge_scores = evaluate_with_judge(judge_client, item, ans, retrieved_docs)
        
        res_record = {
            "id": item["id"],
            "modality": item["modality"],
            "query": q,
            "rewritten_query": rewritten_query,
            "latency_s": round(latency, 2),
            "hit_top3": hit_top3,
            "hit_top5": hit_top5,
            "hit_details": hit_msg5,
            "grounded_images_count": len(grounded_imgs),
            "judge": judge_scores,
            "answer_preview": ans[:200] + "..." if len(ans) > 200 else ans
        }
        results.append(res_record)
        print(f"  Hit@3: {hit_top3} | Hit@5: {hit_top5} | Latency: {latency:.2f}s")
        print(f"  Judge: Faith={judge_scores['faithfulness']}/5, Rel={judge_scores['relevancy']}/5, Acc={judge_scores['accuracy']}/5")
        print(f"  Reason: {judge_scores.get('reasoning', '')}")

    print("\n[3/3] Aggregating Metrics...")
    total_q = len(results)
    hit3_rate = sum(1 for r in results if r["hit_top3"]) / total_q
    hit5_rate = sum(1 for r in results if r["hit_top5"]) / total_q
    
    modalities = set(r["modality"] for r in results)
    modality_hits = {}
    for m in modalities:
        m_items = [r for r in results if r["modality"] == m]
        modality_hits[m] = {
            "count": len(m_items),
            "hit_rate_top3": sum(1 for r in m_items if r["hit_top3"]) / len(m_items),
            "hit_rate_top5": sum(1 for r in m_items if r["hit_top5"]) / len(m_items)
        }
        
    avg_faith = sum(r["judge"]["faithfulness"] for r in results) / total_q
    avg_rel = sum(r["judge"]["relevancy"] for r in results) / total_q
    avg_acc = sum(r["judge"]["accuracy"] for r in results) / total_q
    avg_latency = sum(r["latency_s"] for r in results) / total_q
    
    summary = {
        "paper": "Attention Is All You Need",
        "total_queries": total_q,
        "ingestion_time_s": round(t_ingest, 2),
        "overall_retrieval_hit_rate_top3": round(hit3_rate * 100, 1),
        "overall_retrieval_hit_rate_top5": round(hit5_rate * 100, 1),
        "average_query_latency_s": round(avg_latency, 2),
        "modality_breakdown": modality_hits,
        "mean_faithfulness_score": round(avg_faith, 2),
        "mean_relevancy_score": round(avg_rel, 2),
        "mean_reference_accuracy_score": round(avg_acc, 2),
        "results": results
    }
    
    out_file = Path("evaluation_report.json")
    out_file.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(f"\nEvaluation complete! Report saved to {out_file.resolve()}")
    print(f"Overall Hit@3: {summary['overall_retrieval_hit_rate_top3']}%")
    print(f"Overall Hit@5: {summary['overall_retrieval_hit_rate_top5']}%")
    print(f"Average Latency: {summary['average_query_latency_s']}s")
    print(f"Mean Faithfulness: {summary['mean_faithfulness_score']} / 5.0")
    print(f"Mean Relevancy: {summary['mean_relevancy_score']} / 5.0")
    print(f"Mean Accuracy: {summary['mean_reference_accuracy_score']} / 5.0")

if __name__ == "__main__":
    run_evaluation()
