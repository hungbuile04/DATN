"""Phân tích chi tiết: Reranker giúp/hại ở câu nào?"""
import json, os, sys, time
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))
from dotenv import load_dotenv
load_dotenv()
from config.settings import CFG
from src.retrieval.retriever import DualRetriever
from src.eval.eval_retrieval import precision_at_k, hit_rate_at_k, mrr, _is_relevant

google_key = os.environ["GOOGLE_API_KEY"]
openrouter_key = os.environ["OPENROUTER_API_KEY"]

retriever = DualRetriever(
    chroma_path=str(CFG["paths"]["vector_db"]),
    api_key=google_key, cfg=CFG,
    openrouter_key=openrouter_key, use_reranker=True,
)

with open("src/eval/questions.json", encoding="utf-8") as f:
    questions = json.load(f)

# Chỉ lấy 30 câu đầu để phân tích nhanh
questions = [q for q in questions[:30] if q.get("doc")]

helped = []   # reranker cải thiện
hurt = []     # reranker làm tệ hơn
same = []     # không thay đổi

total_k = 10  # text_top_k + image_top_k

for i, q in enumerate(questions):
    qid = q["id"]
    question = q["question"]
    doc = q.get("doc", "")
    requires = q.get("requires", [])

    print(f"\n[{i+1}/{len(questions)}] {qid}: {question[:60]}...")

    # Dense only
    retriever.use_reranker = False
    try:
        r1 = retriever.retrieve(question, text_top_k=6, image_top_k=4)
    except:
        time.sleep(10); continue

    # With reranker
    retriever.use_reranker = True
    try:
        r2 = retriever.retrieve(question, text_top_k=6, image_top_k=4)
    except:
        time.sleep(10); continue

    all1 = r1.text_chunks + r1.image_chunks
    all2 = r2.text_chunks + r2.image_chunks

    mrr1 = mrr(all1, doc, requires)
    mrr2 = mrr(all2, doc, requires)
    p1 = precision_at_k(all1, doc, requires, total_k)
    p2 = precision_at_k(all2, doc, requires, total_k)

    diff_mrr = mrr2 - mrr1
    diff_p = p2 - p1

    entry = {
        "qid": qid, "category": q.get("category"),
        "doc": doc, "question": question[:60],
        "mrr_dense": mrr1, "mrr_rerank": mrr2, "diff_mrr": diff_mrr,
        "p_dense": p1, "p_rerank": p2, "diff_p": diff_p,
    }

    if diff_mrr > 0.01 or diff_p > 0.01:
        helped.append(entry)
        print(f"  ✅ HELPED: MRR {mrr1:.3f}→{mrr2:.3f} P@k {p1:.3f}→{p2:.3f}")
    elif diff_mrr < -0.01 or diff_p < -0.01:
        hurt.append(entry)
        print(f"  ❌ HURT:   MRR {mrr1:.3f}→{mrr2:.3f} P@k {p1:.3f}→{p2:.3f}")
    else:
        same.append(entry)
        print(f"  ➖ SAME:   MRR {mrr1:.3f}→{mrr2:.3f} P@k {p1:.3f}→{p2:.3f}")

    time.sleep(1)

print(f"\n\n{'='*60}")
print(f"TỔNG KẾT (trên {len(questions)} câu)")
print(f"{'='*60}")
print(f"  ✅ Reranker giúp:  {len(helped)} câu")
print(f"  ❌ Reranker hại:   {len(hurt)} câu")
print(f"  ➖ Không thay đổi: {len(same)} câu")

if helped:
    print(f"\n  Câu được giúp:")
    for e in helped:
        print(f"    {e['qid']} [{e['category']}] MRR +{e['diff_mrr']:.3f} P@k +{e['diff_p']:.3f}")

if hurt:
    print(f"\n  Câu bị hại:")
    for e in hurt:
        print(f"    {e['qid']} [{e['category']}] MRR {e['diff_mrr']:.3f} P@k {e['diff_p']:.3f}")
        # Xem chunks bị thay đổi
