"""Xóa chunks của ed49d43d khỏi metadata.json và ChromaDB, chuẩn bị chạy lại."""
import json, sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from config.settings import CFG
import chromadb

OLD_DOC = "ed49d43d_3a4d_4987_35c2_08dec74ffd7b"

# 1. Xóa khỏi metadata.json
meta_path = CFG["paths"]["metadata"]
with open(meta_path, encoding="utf-8") as f:
    entries = json.load(f)

before = len(entries)
entries = [e for e in entries if e["doc"] != OLD_DOC]
after = len(entries)

with open(meta_path, "w", encoding="utf-8") as f:
    json.dump(entries, f, ensure_ascii=False, indent=2)
print(f"Metadata: {before} → {after} (xóa {before - after} chunks)")

# 2. Xóa khỏi ChromaDB
chroma_path = str(CFG["paths"]["vector_db"])
client = chromadb.PersistentClient(path=chroma_path)

for col_name in ["rag_text", "rag_image"]:
    try:
        col = client.get_collection(col_name)
        # Tìm IDs có chứa old doc name
        result = col.get(where={"doc": OLD_DOC})
        ids = result["ids"]
        if ids:
            col.delete(ids=ids)
            print(f"{col_name}: xóa {len(ids)} vectors")
        else:
            print(f"{col_name}: không có vectors của {OLD_DOC}")
    except Exception as e:
        print(f"{col_name}: {e}")

print("\n✓ Sẵn sàng chạy: python src/extraction/run_incremental.py")
