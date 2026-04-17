# src/extraction/run_extraction.py
import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent.parent.parent))  # để import config

import json
from config.settings import CFG
from extractor import extract_pdf


def run():
    pdf_dir = CFG["paths"]["pdf_raw"]
    out_dir = CFG["paths"]["pdf_processed"]

    # Kiểm tra thư mục tồn tại — báo lỗi rõ ràng thay vì crash im lặng
    if not pdf_dir.exists():
        raise FileNotFoundError(
            f"Không tìm thấy thư mục PDF: {pdf_dir}\n"
            f"Hãy kiểm tra lại config/config.yaml"
        )

    pdf_files = list(pdf_dir.glob("*.pdf"))
    if not pdf_files:
        raise ValueError(f"Không có file PDF nào trong: {pdf_dir}")

    print(f"Tìm thấy {len(pdf_files)} file PDF trong: {pdf_dir}")

    all_metadata = []

    for pdf_path in sorted(pdf_files):
        print(f"\nExtracting: {pdf_path.name}")

        chunks = extract_pdf(
            str(pdf_path),
            str(out_dir),
            images_dpi=CFG["extraction"].get("dpi", 150),
        )

        # FIX: append kết quả vào all_metadata
        all_metadata.extend(chunks)

        print(f"  → {len(chunks)} chunks")

    # Lưu metadata tổng hợp
    meta_path = CFG["paths"]["metadata"]
    meta_path.parent.mkdir(parents=True, exist_ok=True)
    meta_path.write_text(
        json.dumps(all_metadata, ensure_ascii=False, indent=2),
        encoding="utf-8"
    )

    print(f"\nDone. {len(all_metadata)} chunks → {meta_path}")


if __name__ == "__main__":
    run()