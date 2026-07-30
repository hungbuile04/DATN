import json
import random
import logging
import argparse
import sys
from pathlib import Path
from collections import defaultdict

# Thiết lập đường dẫn gốc (source_code_2) vào sys.path để import modules
ROOT_DIR = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT_DIR))

try:
    from config.settings import CFG
except ImportError:
    logging.error("Không thể import CFG từ config.settings. Hãy kiểm tra lại cấu trúc thư mục.")
    sys.exit(1)

import chromadb

# Thiết lập logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

def parse_page_range(page_str) -> set:
    """
    Phân tích chuỗi trang (ví dụ: '2', '2-3', '1,3,5') thành một tập hợp các số trang hợp lệ.
    
    Args:
        page_str (str|int): Chuỗi hoặc số đại diện cho trang.
        
    Returns:
        set: Tập hợp các số trang nguyên (int).
    """
    if not page_str:
        return set()
    
    if isinstance(page_str, int):
        return {page_str}
        
    pages = set()
    page_str = str(page_str)
    for part in page_str.split(','):
        part = part.strip()
        if not part:
            continue
        if '-' in part:
            try:
                start, end = map(int, part.split('-'))
                pages.update(range(start, end + 1))
            except ValueError:
                logger.warning(f"Không thể phân tích dải trang: {part}")
        else:
            try:
                pages.add(int(part))
            except ValueError:
                logger.warning(f"Không thể phân tích số trang: {part}")
    return pages

def load_questions(eval_dir: Path) -> list:
    """
    Tải toàn bộ câu hỏi từ các file JSON đánh giá.
    
    Args:
        eval_dir (Path): Thư mục chứa các file eval.
        
    Returns:
        list: Danh sách các câu hỏi.
    """
    questions = []
    for fname in ["questions.json", "questions_hard.json"]:
        filepath = eval_dir / fname
        if not filepath.exists():
            logger.warning(f"Không tìm thấy file: {filepath}")
            continue
        try:
            with open(filepath, "r", encoding="utf-8") as f:
                data = json.load(f)
                questions.extend(data)
            logger.info(f"Đã tải {len(data)} câu hỏi từ {fname}")
        except Exception as e:
            logger.error(f"Lỗi khi đọc file {filepath}: {e}")
    return questions

def load_chroma_chunks(db_path: str, collection_name: str) -> dict:
    """
    Tải và lập chỉ mục (index) tất cả các chunk từ ChromaDB mà không cần re-embedding.
    
    Args:
        db_path (str): Đường dẫn đến thư mục ChromaDB.
        collection_name (str): Tên collection cần lấy.
        
    Returns:
        dict: Dictionary chứa index các chunk theo cấu trúc {doc: {page: [(id, content)]}}
    """
    try:
        chroma_client = chromadb.PersistentClient(path=db_path)
        text_col = chroma_client.get_collection(collection_name)
        all_data = text_col.get(include=["documents", "metadatas"])
        
        chunk_index = defaultdict(lambda: defaultdict(list))
        
        # Handle cases where metadatas or documents might be empty
        ids = all_data.get("ids", [])
        documents = all_data.get("documents", []) or []
        metadatas = all_data.get("metadatas", []) or []
        
        for chunk_id, doc, meta in zip(ids, documents, metadatas):
            meta = meta or {}
            doc_name = meta.get("doc", "")
            page = meta.get("page", -1)
            
            chunk_index[doc_name][page].append((chunk_id, doc))
            
        logger.info(f"Đã lập chỉ mục {len(ids)} chunks từ {len(chunk_index)} tài liệu.")
        return chunk_index
    except Exception as e:
        logger.error(f"Lỗi khi tải dữ liệu từ ChromaDB: {e}")
        sys.exit(1)

def generate_training_pairs(questions: list, chunk_index: dict, negative_ratio: int = 3) -> tuple:
    """
    Tạo các cặp (query, passage, label) cho việc huấn luyện reranker.
    
    Args:
        questions (list): Danh sách các câu hỏi (query).
        chunk_index (dict): Index các passage từ ChromaDB.
        negative_ratio (int): Tỷ lệ số lượng negative mong muốn so với positive.
        
    Returns:
        tuple: (Danh sách các cặp training, Dictionary chứa thống kê).
    """
    pairs = []
    stats = defaultdict(lambda: {"pos": 0, "neg": 0})
    
    for q in questions:
        query = q.get("question", "")
        target_doc = q.get("doc", "")
        source_page = q.get("source_page", "")
        category = q.get("category", "unknown")
        question_id = q.get("id", "unknown")
        
        if not query:
            continue

        # Chuẩn hóa target_doc thành list (multi_doc_comparison có doc là list)
        if isinstance(target_doc, list):
            target_docs = target_doc
        elif target_doc:
            target_docs = [target_doc]
        else:
            continue
            
        valid_pages = parse_page_range(source_page)
        
        positives = []
        hard_negatives = []
        
        # Trích xuất positive và hard negative từ các tài liệu mục tiêu
        for td in target_docs:
            if td in chunk_index:
                for page, chunks in chunk_index[td].items():
                    for chunk_id, content in chunks:
                        if page in valid_pages:
                            positives.append(content)
                        else:
                            hard_negatives.append(content)
                        
        # Trích xuất easy negative từ các tài liệu khác
        easy_negatives = []
        other_docs = [d for d in chunk_index if d not in target_docs]
        
        # Lấy mẫu ngẫu nhiên từ các tài liệu khác để tránh mất cân bằng
        sampled_docs = random.sample(other_docs, min(5, len(other_docs)))
        for other_doc in sampled_docs:
            for page, chunks in chunk_index[other_doc].items():
                for chunk_id, content in chunks:
                    easy_negatives.append(content)
                    if len(easy_negatives) >= 20: # Lấy đủ số lượng để sample
                        break
                if len(easy_negatives) >= 20:
                    break
                    
        # Bỏ qua nếu không có positive sample
        if not positives:
            logger.debug(f"Không tìm thấy positive chunk cho câu hỏi '{question_id}'")
            continue
            
        # Thêm positive pairs
        doc_label = target_docs[0] if len(target_docs) == 1 else ",".join(target_docs)
        for p in positives:
            pairs.append({
                "query": query, 
                "passage": p, 
                "label": 1, 
                "doc": doc_label, 
                "question_id": question_id
            })
            stats[category]["pos"] += 1
            
        # Lấy mẫu negative pairs (Cân bằng số lượng, kết hợp hard và easy)
        n_neg_target = max(len(positives) * negative_ratio, 5)
        
        n_hard = min(len(hard_negatives), n_neg_target // 2)
        n_easy = min(len(easy_negatives), n_neg_target - n_hard)
        
        sampled_hard = random.sample(hard_negatives, n_hard)
        sampled_easy = random.sample(easy_negatives, n_easy)
        
        for n in sampled_hard + sampled_easy:
            pairs.append({
                "query": query, 
                "passage": n, 
                "label": 0, 
                "doc": doc_label, 
                "question_id": question_id
            })
            stats[category]["neg"] += 1
            
    # Trộn ngẫu nhiên danh sách (shuffle) để mô hình không học theo thứ tự
    random.shuffle(pairs)
    return pairs, stats

def main():
    parser = argparse.ArgumentParser(description="Tạo dữ liệu huấn luyện cho mô hình Reranker.")
    parser.add_argument("--seed", type=int, default=42, help="Random seed để tái lập kết quả.")
    parser.add_argument("--neg_ratio", type=int, default=3, help="Tỷ lệ negative/positive.")
    args = parser.parse_args()

    random.seed(args.seed)
    
    # 1. Định nghĩa đường dẫn
    eval_dir = ROOT_DIR / "src" / "eval"
    output_dir = ROOT_DIR / "training_data"
    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / "reranker_pairs.jsonl"
    
    # Lấy cấu hình từ CFG
    try:
        db_path = str(CFG["paths"]["vector_db"])
        collection_name = CFG["embedding"]["text_collection"]
    except KeyError as e:
        logger.error(f"Thiếu cấu hình trong settings: {e}")
        sys.exit(1)
        
    # 2. Tải câu hỏi
    logger.info("Đang tải các câu hỏi đánh giá...")
    questions = load_questions(eval_dir)
    if not questions:
        logger.error("Không có câu hỏi nào được tải. Dừng chương trình.")
        sys.exit(1)
        
    # 3. Tải chunks từ ChromaDB
    logger.info(f"Đang tải chunks từ ChromaDB ({db_path})...")
    chunk_index = load_chroma_chunks(db_path, collection_name)
    
    # 4. Tạo các cặp dữ liệu huấn luyện
    logger.info("Đang tạo các cặp dữ liệu huấn luyện (query, passage, label)...")
    pairs, stats = generate_training_pairs(questions, chunk_index, args.neg_ratio)
    
    # 5. Lưu vào file JSONL
    logger.info(f"Đang lưu {len(pairs)} cặp dữ liệu vào {output_path}...")
    try:
        with open(output_path, "w", encoding="utf-8") as f:
            for pair in pairs:
                f.write(json.dumps(pair, ensure_ascii=False) + "\n")
        logger.info("Đã lưu xong!")
    except Exception as e:
        logger.error(f"Lỗi khi ghi file {output_path}: {e}")
        sys.exit(1)
        
    # 6. In thống kê
    n_pos = sum(1 for p in pairs if p["label"] == 1)
    n_neg = sum(1 for p in pairs if p["label"] == 0)
    
    print("\n" + "="*50)
    print("THỐNG KÊ DỮ LIỆU HUẤN LUYỆN")
    print("="*50)
    print(f"Tổng số cặp: {len(pairs)}")
    print(f"Positives (label=1): {n_pos}")
    print(f"Negatives (label=0): {n_neg}")
    ratio = n_neg / max(n_pos, 1)
    print(f"Tỷ lệ Positive:Negative: 1:{ratio:.1f}")
    
    print("\nThống kê theo category:")
    print(f"{'Category':<20} | {'Positive':<10} | {'Negative':<10} | {'Total':<10}")
    print("-" * 57)
    for cat, stat in stats.items():
        print(f"{cat:<20} | {stat['pos']:<10} | {stat['neg']:<10} | {stat['pos'] + stat['neg']:<10}")
    print("="*50 + "\n")

if __name__ == "__main__":
    main()
