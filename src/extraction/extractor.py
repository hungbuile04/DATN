# src/extraction/extractor.py

"""
PDF Extractor cho Multimodal RAG (financial report).

Pipeline sử dụng Docling (IBM) để extract:
  - Tables  → markdown table chunks
  - Pictures/Charts → image files (embed trực tiếp bằng Gemini Embedding 2)
  - Text → narrative text chunks (sliding window)

Output: metadata list với chunk_type rõ ràng:
  - "text"  : narrative text → embed text
  - "table" : markdown table → embed text  
  - "image" : chart/figure   → embed image trực tiếp (không qua caption)

PDF → Docling DocumentConverter → DoclingDocument
  → doc.tables      → markdown table chunks
  → doc.pictures    → chart/figure image files
  → iterate_items() → text chunks (sliding window)
  → metadata.json
"""

import re
import sys
import fitz  # PyMuPDF – chỉ dùng cho full-page screenshot

from pathlib import Path

from docling.document_converter import DocumentConverter, PdfFormatOption
from docling.datamodel.pipeline_options import PdfPipelineOptions, TableFormerMode
from docling.datamodel.base_models import InputFormat
from docling_core.types.doc import DocItemLabel

# Đảm bảo project root trong sys.path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

# ======================================================
# Constants
# ======================================================

CHUNK_SIZE = 400        # tokens (xấp xỉ bằng từ)
CHUNK_OVERLAP = 80      # tokens overlap giữa các chunk
MIN_CHART_SIZE_PT = 80  # kích thước tối thiểu (pt) để coi là biểu đồ thật (loại logo)


# ======================================================
# Main Entry Point
# ======================================================

def extract_pdf(pdf_path: str, out_dir: str, images_dpi: int = 150) -> list[dict]:
    """
    Extract một PDF thành các chunks multimodal sử dụng Docling.

    Args:
        images_dpi: DPI dùng để render ảnh chart (mặc định 150).
                    Docling dùng `images_scale = images_dpi / 72` để tính tỷ lệ
                    so với độ phân giải gốc của PDF (1 pt = 1/72 inch).

    Output: list[metadata dict], mỗi entry có:
        - chunk_type: "text" | "table" | "image"
        - content_for_embedding: path tới file cần embed
        - metadata đầy đủ (doc, page, ticker, report_date, source, ...)
    """

    doc_name = Path(pdf_path).stem
    out_base = Path(out_dir)
    _dm = _parse_doc_name(doc_name)
    _ticker: str = str(_dm.get("ticker") or "")
    _report_date: str = str(_dm.get("report_date") or "")
    _source: str = str(_dm.get("source") or "")

    # Tạo thư mục output
    (out_base / "chunks").mkdir(parents=True, exist_ok=True)
    (out_base / "images").mkdir(parents=True, exist_ok=True)

    # ---------------------------------------------------
    # STEP 1 — Docling conversion
    # ---------------------------------------------------
    pipeline_options = PdfPipelineOptions()
    pipeline_options.do_table_structure = True
    pipeline_options.table_structure_options.mode = TableFormerMode.ACCURATE
    pipeline_options.generate_picture_images = True
    # Tăng độ phân giải ảnh chart: images_scale = DPI / 72 (base resolution của PDF)
    pipeline_options.images_scale = images_dpi / 72.0

    converter = DocumentConverter(
        format_options={
            InputFormat.PDF: PdfFormatOption(pipeline_options=pipeline_options)
        }
    )

    result = converter.convert(pdf_path)
    doc = result.document

    # Mở PDF bằng fitz để screenshot full page khi cần
    fitz_doc = fitz.open(pdf_path)

    # ── Fallback: trích ticker từ nội dung trang 1 nếu tên file không parse được ──
    if not _ticker or not re.match(r'^[A-Z]{2,5}$', _ticker):
        first_page_text = fitz_doc[0].get_text() if len(fitz_doc) > 0 else ""
        fallback_ticker = _extract_ticker_from_content(first_page_text)
        if fallback_ticker:
            print(f"  ⚡ Ticker từ nội dung: {fallback_ticker} (tên file không parse được)")
            _ticker = fallback_ticker

    metadata: list[dict] = []

    # ---------------------------------------------------
    # STEP 2 — Group items by page
    # ---------------------------------------------------
    page_texts: dict[int, list[str]] = {}      # page_no → list of text content
    page_tables: dict[int, list] = {}           # page_no → list of table items
    page_picture_captions: dict[int, list[tuple]] = {}  # page_no → [(pic, caption)]

    # Track captions per page for caption-picture linking
    _pending_captions: dict[int, list[str]] = {}  # page_no → ordered captions

    # Single pass: collect text, captions, and build caption→picture pairs
    for item, level in doc.iterate_items():
        if not hasattr(item, 'prov') or not item.prov:
            continue
        page_no = item.prov[0].page_no
        label = item.label if hasattr(item, 'label') else ''

        # Text content (bao gồm cả caption — vẫn thêm vào text)
        text_labels = {
            DocItemLabel.TEXT, DocItemLabel.SECTION_HEADER,
            DocItemLabel.LIST_ITEM, DocItemLabel.CAPTION,
            DocItemLabel.PARAGRAPH, DocItemLabel.TITLE,
        }
        if label in text_labels:
            text = item.text if hasattr(item, 'text') else ''
            if text and text.strip():
                if label == DocItemLabel.SECTION_HEADER:
                    text = f"## {text}"
                elif label == DocItemLabel.TITLE:
                    text = f"# {text}"

                # Bỏ qua nhãn trục biểu đồ bị rò rỉ vào text block
                if _is_axis_noise(text):
                    continue
                # Strip phần axis ticks đứng đầu khi item là text thật bị lẫn prefix nhiễu
                text = _strip_axis_prefix(text)

                page_texts.setdefault(page_no, []).append(_clean_ocr_text(text.strip()))

                # Track captions riêng để link với picture phía sau
                if label == DocItemLabel.CAPTION:
                    _pending_captions.setdefault(page_no, []).append(text.strip())

    # Group tables
    for table in doc.tables:
        if table.prov:
            page_no = table.prov[0].page_no
            page_tables.setdefault(page_no, []).append(table)

    # Group pictures + link with captions
    # Captions appear BEFORE pictures in Docling's item order
    _SOURCE_NOTE_RE = re.compile(
        r'^(?:ngu[oồ]n|ngu\s*[oồ]\s*n|source)\s*:', re.IGNORECASE
    )
    _CHART_TITLE_RE = re.compile(
        r'(?:bi\s*[eêế]\s*[uư]\s*[^\s]*\s*đ|'   # "biểu đồ" variants
        r'đ\s*[oồô]\s*\s*th|'                    # "đồ thị"
        r'h[iì]\s*nh\s+\d|'                      # "hình 1"
        r'chart|figure)',
        re.IGNORECASE
    )

    for pic in doc.pictures:
        if not pic.prov:
            continue
        page_no = pic.prov[0].page_no
        # Pop the first pending caption for this page (FIFO)
        captions = _pending_captions.get(page_no, [])
        caption = captions.pop(0) if captions else ""

        # Fallback: nếu caption chỉ là source note ("Nguồn: SSI Research"),
        # tìm tiêu đề biểu đồ thực sự
        if not caption or _SOURCE_NOTE_RE.match(caption.strip()):
            source_note = caption  # giữ lại để append sau
            caption = ""

            # Fallback A: lấy text block nằm ngay PHÍA TRÊN bbox ảnh trong PDF
            # (nhiều biểu đồ có title trên đầu nhưng không match regex)
            if pic.prov:
                above_title = _find_title_above_pic(pic, fitz_doc, page_no)
                if above_title:
                    caption = _clean_ocr_text(above_title)

            # Fallback B: tìm theo regex keyword trong page_texts
            if not caption:
                for t in page_texts.get(page_no, []):
                    clean_t = _clean_ocr_text(t.lstrip('#').strip())
                    if _CHART_TITLE_RE.search(clean_t):
                        caption = clean_t
                        break

            # Nếu tìm được title, append source note vào cuối
            if caption and source_note:
                caption = f"{caption} ({source_note})"
            elif not caption:
                caption = source_note or f"Biểu đồ trang {page_no}"

        page_picture_captions.setdefault(page_no, []).append((pic, caption))

    # ---------------------------------------------------
    # STEP 3 — Get all unique page numbers
    # ---------------------------------------------------
    all_pages = sorted(set(
        list(page_texts.keys()) +
        list(page_tables.keys()) +
        list(page_picture_captions.keys())
    ))

    # ---------------------------------------------------
    # STEP 4 — Process each page
    # ---------------------------------------------------
    for page_no in all_pages:
        texts = page_texts.get(page_no, [])
        tables = page_tables.get(page_no, [])
        pic_caps = page_picture_captions.get(page_no, [])  # [(pic, caption)]

        has_text = bool(texts)
        has_table = bool(tables)
        has_chart = bool(pic_caps)

        # Classify page type
        if has_table and has_chart:
            page_type = "mixed"
        elif has_table:
            page_type = "table"
        elif has_chart and not has_text:
            page_type = "chart"
        elif has_chart:
            page_type = "mixed"
        elif has_text:
            # Kiểm tra header_only: quá ít text
            total_text = " ".join(texts)
            if len(total_text.split()) < 20:
                page_type = "header_only"
            else:
                page_type = "text"
        else:
            page_type = "header_only"

        if page_type == "header_only":
            continue

        chunk_base_id = f"{doc_name}_p{page_no - 1:03d}"

        # ---- 4a. Text chunks ----
        text_chunks: list[dict] = []
        if texts:
            full_text = "\n".join(texts)
            # Topic summary: first sentence or heading
            topic = _extract_topic(texts)

            chunks = _sliding_window_chunk(full_text, CHUNK_SIZE, CHUNK_OVERLAP)
            for ci, chunk_text in enumerate(chunks):
                chunk_id = f"{chunk_base_id}_c{ci:02d}"
                # Thêm metadata header
                header = f"<!-- document: {doc_name} | page: {page_no} | topic: {topic} -->"
                chunk_content = f"{header}\n\n{chunk_text}"

                text_path = out_base / "chunks" / f"{chunk_id}_text.md"
                text_path.write_text(chunk_content, encoding="utf-8")

                text_chunks.append({
                    "chunk_id": chunk_id,
                    "chunk_type": "text",
                    "doc": doc_name,
                    "page": page_no,
                    "chunk_index": ci,
                    "page_type": page_type,
                    "text_path": str(text_path),
                    "img_path": None,
                    "content_for_embedding": str(text_path),
                    "has_text": True,
                    "has_table": has_table,
                    "has_chart": has_chart,
                    "ticker": _ticker,
                    "report_date": _report_date,
                    "source": _source,
                    "token_count": len(chunk_text.split()),
                })

        # ---- 4b. Table chunks (một chunk riêng cho mỗi bảng) ----
        table_captions = _find_table_captions(texts)
        _cap_iter = iter(table_captions)   # consume captions in order, one per table
        table_chunks: list[dict] = []
        for ti, table in enumerate(tables):
            table_caption: str = next(_cap_iter, "")
            table_md: str = ""
            try:
                df = table.export_to_dataframe(doc)
                table_md = df.to_markdown(index=False) or ""
            except Exception:
                if hasattr(table, 'text') and table.text:
                    table_md = table.text

            if not table_md:
                continue

            # Sửa lỗi Docling tách dấu tiếng Việt trong nội dung bảng
            table_md = _clean_ocr_text(table_md)

            # Normalize table_name: slug lowercase
            raw_name = table_caption or f"table_{ti}"
            table_name = re.sub(r'[^\w]+', '_', raw_name.lower()).strip('_')
            if len(table_name) > 50:
                m = re.match(r'[\w_]{0,50}', table_name)
                table_name = m.group(0).rstrip('_') if m else table_name

            topic = table_caption or (_extract_topic(texts) if texts else doc_name)
            header = f"<!-- document: {doc_name} | page: {page_no} | topic: {topic} -->"
            table_content = f"{header}\n\n{table_md}"

            table_path = out_base / "chunks" / f"{chunk_base_id}_table{ti:02d}.md"
            table_path.write_text(table_content, encoding="utf-8")

            table_chunks.append({
                "chunk_id": f"{chunk_base_id}_table{ti:02d}",
                "chunk_type": "table",
                "doc": doc_name,
                "page": page_no,
                "chunk_index": ti,
                "page_type": page_type,
                "text_path": str(table_path),
                "img_path": None,
                "content_for_embedding": str(table_path),
                "has_text": True,
                "has_table": True,
                "has_chart": has_chart,
                "table_name": table_name,
                "ticker": _ticker,
                "report_date": _report_date,
                "source": _source,
                "token_count": len(table_md.split()),
            })

        # ---- 4c. Image chunks (embed ảnh trực tiếp, không tạo text chunk) ----
        image_chunks: list[dict] = []
        if pic_caps:
            # Trang bìa SSI luôn tạo 3 ảnh: [0] logo, [1] header, [2] biểu đồ giá CP
            # → chỉ giữ ảnh cuối (index 2), bỏ 2 ảnh đầu
            if page_no == 1 and len(pic_caps) >= 3:
                pic_caps = pic_caps[2:]

            for pi, (pic, caption) in enumerate(pic_caps):
                # Bỏ qua ảnh quá nhỏ (icon trang trí)
                if pic.prov:
                    _pb = pic.prov[0].bbox
                    _w = _pb.r - _pb.l
                    _h = _pb.t - _pb.b
                    if _w < MIN_CHART_SIZE_PT or _h < MIN_CHART_SIZE_PT:
                        continue

                # === Detect 2-chart split: kiểm tra vùng ảnh có 2 biểu đồ riêng lẻ ===
                sub_rects = _split_chart_regions(pic, fitz_doc, page_no)

                if len(sub_rects) >= 2:
                    # Tách thành 2 ảnh + 2 JSON entries
                    fitz_page = fitz_doc[page_no - 1]
                    scale = images_dpi / 72.0
                    mat = fitz.Matrix(scale, scale)
                    for si, sub_rect in enumerate(sub_rects[:2]):
                        img_filename = f"{chunk_base_id}_chart{pi:02d}_{si:02d}.png"
                        img_save_path = out_base / "images" / img_filename
                        try:
                            pix = fitz_page.get_pixmap(matrix=mat, clip=sub_rect)
                            pix.save(str(img_save_path))
                        except Exception:
                            continue

                        sub_caption_raw = _extract_chart_title_in_region(fitz_page, sub_rect)
                        caption_clean = _clean_ocr_text(
                            sub_caption_raw or f"{caption} ({si + 1})"
                        ) if (sub_caption_raw or caption) else f"Biểu đồ trang {page_no} ({si + 1})"

                        chart_id = f"{chunk_base_id}_chart{pi:02d}_{si:02d}"
                        image_chunks.append({
                            "chunk_id": chart_id,
                            "chunk_type": "image",
                            "doc": doc_name,
                            "page": page_no,
                            "chunk_index": pi * 10 + si,
                            "page_type": "chart",
                            "text_path": None,
                            "img_path": str(img_save_path),
                            "content_for_embedding": str(img_save_path),
                            "has_text": False,
                            "has_table": False,
                            "has_chart": True,
                            "caption": caption_clean,
                            "ticker": _ticker,
                            "report_date": _report_date,
                            "source": _source,
                            "token_count": 0,
                        })
                else:
                    # Single chart — logic gốc
                    img_save_path = None
                    img_filename = f"{chunk_base_id}_chart{pi:02d}.png"
                    saved = _crop_chart_expanded(
                        pic, fitz_doc, page_no, images_dpi,
                        out_base / "images" / img_filename,
                    )
                    if saved:
                        img_save_path = out_base / "images" / img_filename
                    else:
                        # Fallback: Docling tight crop
                        try:
                            img = pic.get_image(doc)
                            if img:
                                img_save_path = out_base / "images" / img_filename
                                img.save(str(img_save_path))
                        except Exception:
                            pass

                    if not img_save_path:
                        continue

                    chart_id = f"{chunk_base_id}_chart{pi:02d}"
                    caption_clean = _clean_ocr_text(caption) if caption else f"Biểu đồ trang {page_no}"

                    image_chunks.append({
                        "chunk_id": chart_id,
                        "chunk_type": "image",
                        "doc": doc_name,
                        "page": page_no,
                        "chunk_index": pi,
                        "page_type": "chart",
                        "text_path": None,
                        "img_path": str(img_save_path),
                        "content_for_embedding": str(img_save_path),
                        "has_text": False,
                        "has_table": False,
                        "has_chart": True,
                        "caption": caption_clean,
                        "ticker": _ticker,
                        "report_date": _report_date,
                        "source": _source,
                        "token_count": 0,
                    })

        # ---- 4d. Finalize metadata ----
        if text_chunks:
            metadata.extend(text_chunks)
        if table_chunks:
            metadata.extend(table_chunks)
        if image_chunks:
            metadata.extend(image_chunks)

    fitz_doc.close()

    # Post-process: merge bảng cross-page
    metadata = _merge_cross_page_tables(metadata, out_base)

    return metadata


# ======================================================
# Helper Functions
# ======================================================

_RE_TRUNC = re.compile(r'[\s\S]{0,200}')

def _trunc(s: str) -> str:
    """Trả về tối đa 200 ký tự đầu của s (Pyright-safe, tránh str[:n] slice errors)."""
    m = _RE_TRUNC.match(s)
    return m.group(0) if m else s


def _clean_ocr_text(text: str) -> str:
    """
    Sửa lỗi Docling tách dấu tiếng Việt thành token riêng biệt.

    Docling hay split một âm tiết thành mảnh: 'Bi ểu' → 'Biểu', 'L ợ i' → 'Lợi',
    'ngh ị' → 'nghị', 'KH Ả' → 'KHẢ'.

    Thuật toán 2 pass (lặp 2 lần để xử lý chuỗi nhiều tầng):
      Pass A: [non-space] + space + [Vietnamese_precomposed tại cuối token] → merge
              Ví dụ: "v ề"  → "về",  "c ổ" → "cổ",  "KH Ả" → "KHẢ"
              Safe: chỉ fire khi Vietnamese char theo sau là space/punct (cuối token),
              tức là nó bị tách ra, không phải đầu từ tiếp theo.
      Pass B: [Vietnamese_precomposed] + space + [ASCII tại cuối token] → merge
              Ví dụ: "Lợ i" → "Lợi", "Khuyế n" → "Khuyến", "nhuậ n" → "nhuận"
              Safe: chỉ fire khi ASCII char theo sau là space/punct (isolated final consonant).
    """
    _VIET = r'[\u1EA0-\u1EF9\u00C0-\u024F]'
    _END  = r'(?=[\s,.:;!?()\[\]"\u2018\u2019]|$)'

    for _ in range(2):
        # Pass A: non-space + space + Vietnamese ở cuối token — "v ề"→"về", "KH Ả"→"KHẢ"
        text = re.sub(rf'(\S) ({_VIET}){_END}', r'\1\2', text)
        # Pass B: Vietnamese + space + phụ âm cuối âm tiết — "Lợ i"→"Lợi", "lượ ng"→"lượng"
        _FIN = r'(?:ch|ng|nh|[acimnotupy])'
        text = re.sub(rf'({_VIET}) ({_FIN}){_END}', r'\1\2', text)
        # Pass C: ASCII prefix đầu từ + space + Vietnamese vowel-diacritic — "Bi ểu"→"Biểu"
        _VIET_NOD = r'[\u1EA0-\u1EF9\u00C0-\u010F\u0112-\u024F]'
        text = re.sub(rf'\b([a-zA-Z]{{1,4}}) ({_VIET_NOD})', r'\1\2', text)

    text = re.sub(r'  +', ' ', text)
    return text.strip()


def _strip_axis_prefix(text: str) -> str:
    """
    Strip phần nhãn trục biểu đồ đứng đầu một text item hợp lệ.

    Docling đôi khi gộp axis ticks với đoạn văn thật thành một TEXT item:
      "HS 400 25% 30% 60,0% tính từ đầu năm. Ở chiều ngược lại..."
    Hàm này cắt bỏ phần prefix không có ký tự Việt dấu, giữ lại phần văn bản.

    Điều kiện strip (phải thoả cả 3):
      1. Prefix dài hơn 5 ký tự (tránh cắt nhầm ký tự đơn)
      2. Prefix chứa ít nhất 1 số hoặc %
      3. Tỷ lệ ký tự "nhiễu" (số, %, dấu câu, space) trong prefix > 60%
    """
    _VIET_RE = r'[\u1EA0-\u1EF9\u00C0-\u024F]'
    parts = re.split(rf'\b(?=\w*{_VIET_RE})', text, maxsplit=1)
    if len(parts) < 2:
        return text

    prefix, rest = parts[0], parts[1]
    if len(prefix) <= 5:
        return text

    # Kiểm tra prefix có chứa số/% không (axis ticks luôn có)
    if not re.search(r'[\d%]', prefix):
        return text

    # Tính tỷ lệ ký tự nhiễu trong prefix
    noise_chars = sum(1 for c in prefix if c in '0123456789%,. ()+-/\t')
    if noise_chars / len(prefix) > 0.60:
        return rest.strip()

    return text


def _crop_chart_expanded(
    pic,
    fitz_doc: fitz.Document,
    page_no: int,
    dpi: int,
    out_path: Path,
    pad_above: float = 40.0,
    pad_below: float = 20.0,
    pad_side: float = 10.0,
) -> bool:
    """
    Crop ảnh biểu đồ từ PDF bằng fitz với vùng mở rộng để bắt caption.

    Caption thường nằm PHÍA TRÊN biểu đồ (ví dụ: "Biểu đồ 5: Lợi suất tài sản..."),
    nguồn ("Nguồn: SSI Research") nằm phía dưới.

    Docling bbox dùng hệ toạ độ bottom-left (gốc góc dưới-trái của trang).
    fitz dùng top-left. Hàm này convert và mở rộng vùng crop:
      - pad_above: thêm phía trên biểu đồ (~70pt ≈ 1-2 dòng caption tiêu đề)
      - pad_below: thêm phía dưới (~35pt để bắt dòng "Nguồn:")
      - pad_side:  mở rộng hai bên (tránh cắt sát mép)

    Returns True nếu lưu thành công, False nếu thất bại.
    """
    try:
        if not pic.prov:
            return False
        bbox = pic.prov[0].bbox        # Docling BoundingBox, origin=BOTTOMLEFT
        fitz_page = fitz_doc[page_no - 1]   # fitz: 0-indexed
        pw = fitz_page.rect.width
        ph = fitz_page.rect.height

        # Chuyển Docling (bottom-left) → fitz (top-left)
        fitz_top    = ph - bbox.t
        fitz_bottom = ph - bbox.b

        x0 = max(0.0,  bbox.l  - pad_side)
        x1 = min(pw,   bbox.r  + pad_side)
        y0 = max(0.0,  fitz_top    - pad_above)
        y1 = min(ph,   fitz_bottom + pad_below)

        clip = fitz.Rect(x0, y0, x1, y1)
        scale = dpi / 72.0
        mat = fitz.Matrix(scale, scale)
        pix = fitz_page.get_pixmap(matrix=mat, clip=clip)
        pix.save(str(out_path))
        return True
    except Exception:
        return False


def _is_axis_noise(text: str) -> bool:
    """
    Phát hiện chuỗi là nhãn trục / ticks biểu đồ bị OCR rò rỉ vào text block.

    Hai trường hợp:
      1. Single-token: thuần số/phần trăm, ví dụ "25%", "80,0%", "400" — luôn là axis tick.
      2. Multi-token: không có ký tự Việt dấu + ≥3 token số/% chiếm >50%.
         Ví dụ: "HS 400 25% 30% 20% 10% 80,0% 70,0%".
    """
    stripped = text.strip()
    if not stripped:
        return False
    # Không có ký tự Việt → có thể là noise
    has_viet = bool(re.search(r'[\u1EA0-\u1EF9\u00C0-\u024F]', stripped))
    if has_viet:
        return False  # Có chữ Việt → không phải axis noise
    # Case 1: chuỗi rất ngắn (≤2 ký tự), toàn chữ hoa — axis label như "HS", "Q", "IV"
    if re.match(r'^[A-Z]{1,2}$', stripped):
        return True
    # Case 2: thuần số/phần trăm (single hoặc multi token, tất cả đều là số)
    tokens = stripped.split()
    num_tokens = sum(1 for t in tokens if re.match(r'^[\d.,]+%?$', t))
    if num_tokens == len(tokens) and num_tokens >= 1:
        return True   # tất cả token đều là số/phần trăm → axis ticks
    if len(stripped) < 5:
        return False
    # Case 3: đa số token là số (ngưỡng gốc)
    return num_tokens >= 3 and num_tokens / len(tokens) > 0.5


def _extract_topic(texts: list[str]) -> str:
    """
    Trích topic từ danh sách text items của một trang.

    Ưu tiên: SECTION_HEADER (##) → TITLE (#) → đoạn text dài đầu tiên.
    Bỏ qua các dòng chỉ là source note ("Nguồn:", "Source:") hoặc quá ngắn.
    """
    _SKIP_RE = re.compile(r'^(?:ngu[oồ]n|source)\s*:', re.IGNORECASE)

    # Ưu tiên 1: heading có dấu ## hoặc #
    for t in texts:
        if t.startswith("##") or t.startswith("#"):
            clean = t.lstrip("#").strip()
            if len(clean) > 10 and not _SKIP_RE.match(clean):
                return _clean_ocr_text(_trunc(clean))

    # Ưu tiên 2: đoạn text thực sự (dài, không phải source note)
    for t in texts:
        clean = t.lstrip("#").strip()
        if len(clean) > 30 and not _SKIP_RE.match(clean) and not _is_axis_noise(clean):
            return _clean_ocr_text(_trunc(clean))

    return _clean_ocr_text(_trunc(texts[0].lstrip("#").strip())) if texts else ""


def _sliding_window_chunk(text: str, chunk_size: int, overlap: int) -> list[str]:
    """Chia text thành chunks bằng sliding window trên tokens."""
    words = text.split()
    if len(words) <= chunk_size:
        return [text]

    chunks = []
    start = 0
    while start < len(words):
        end = start + chunk_size
        chunk = " ".join(words[start:end])
        chunks.append(chunk)
        if end >= len(words):
            break
        start += chunk_size - overlap

    return chunks


def _parse_doc_name(doc_name: str) -> dict:
    """
    Parse doc_name thành structured metadata.

    Ví dụ: "HDB_2025_11_21_SSIResearch" → {"ticker": "HDB", "report_date": "2025-11-21", "source": "SSIResearch"}
    Format: TICKER_YYYY_MM_DD_SOURCE (source có thể gồm nhiều phần nối bằng _)
    """
    parts = doc_name.split("_")
    result: dict = {"ticker": "", "report_date": "", "source": ""}
    if len(parts) >= 4:
        result["ticker"] = parts[0]
        # Kiểm tra 3 phần tiếp theo có phải là YYYY_MM_DD không
        if all(re.match(r'^\d+$', parts[i]) for i in range(1, 4)):
            result["report_date"] = f"{parts[1]}-{parts[2]}-{parts[3]}"
            suffix: list[str] = []
            i = 4
            while i < len(parts):
                suffix.append(parts[i])
                i += 1
            result["source"] = "_".join(suffix)
        else:
            tail: list[str] = []
            j = 1
            while j < len(parts):
                tail.append(parts[j])
                j += 1
            result["source"] = "_".join(tail)
    elif parts:
        result["ticker"] = parts[0]
    return result


def _extract_ticker_from_content(text: str) -> str:
    """
    Trích mã cổ phiếu từ nội dung trang 1 khi tên file không theo format chuẩn.

    Nhận diện các pattern phổ biến trong báo cáo tài chính Việt Nam:
      - "HSX: DCM", "HOSE: VNM", "HNX: PVS", "UPCOM: ACV"
      - "Mã CK: DCM", "Mã CP: VNM"
      - "(HSX: DCM)", "(HOSE: VNM)"
      - "Mã cổ phiếu: DCM"
    """
    patterns = [
        # (HSX: DCM), (HOSE: VNM), (HNX: PVS), (UPCOM: ACV)
        r'\(?(?:HSX|HOSE|HNX|UPCOM|UPCoM)\s*[:\-]\s*([A-Z]{2,5})\)?',
        # Mã CK: DCM, Mã CP: VNM, Mã cổ phiếu: DCM
        r'[Mm]ã\s+(?:CK|CP|cổ phiếu|chứng khoán)\s*[:\-]\s*([A-Z]{2,5})',
        # Ticker: DCM
        r'[Tt]icker\s*[:\-]\s*([A-Z]{2,5})',
    ]
    for pattern in patterns:
        match = re.search(pattern, text)
        if match:
            return match.group(1)
    return ""


def _split_chart_regions(
    pic,
    fitz_doc: fitz.Document,
    page_no: int,
    min_sub_height: int = 70,
    padding: float = 8.0,
) -> list:
    """
    Kiểm tra một Docling picture có chứa 2 biểu đồ riêng lẻ không.

    Dùng gap dọc giữa các text block bên trong vùng ảnh để xác định ranh giới.

    Returns:
        list[fitz.Rect] của từng sub-region nếu detect được ≥ 2 biểu đồ,
        ngược lại trả về [] (caller xử lý như ảnh đơn bình thường).
    """
    if not pic.prov:
        return []

    bbox = pic.prov[0].bbox
    fitz_page = fitz_doc[page_no - 1]
    ph = fitz_page.rect.height
    pw = fitz_page.rect.width

    # Docling (bottom-left) → fitz (top-left)
    x0 = max(0.0, bbox.l - 5.0)
    x1 = min(pw,  bbox.r + 5.0)
    y0 = max(0.0, ph - bbox.t - 15.0)   # mở rộng lên trên để bắt caption
    y1 = min(ph,  ph - bbox.b + 10.0)

    pic_rect = fitz.Rect(x0, y0, x1, y1)

    # Lấy text blocks bên trong vùng ảnh
    raw_blocks = fitz_page.get_text("blocks")
    inner_rects = sorted(
        [
            fitz.Rect(b[:4])
            for b in raw_blocks
            if b[6] == 0                                  # text block
            and len(b[4].strip()) > 5                     # có nội dung
            and fitz.Rect(b[:4]).x0 >= pic_rect.x0 - 5
            and fitz.Rect(b[:4]).x1 <= pic_rect.x1 + 5
            and fitz.Rect(b[:4]).y0 >= pic_rect.y0
            and fitz.Rect(b[:4]).y1 <= pic_rect.y1
        ],
        key=lambda r: r.y0,
    )

    if not inner_rects:
        return []

    # Gom các text block gần nhau thành text band
    bands: list = []
    current = inner_rects[0]
    for r in inner_rects[1:]:
        if r.y0 - current.y1 <= 6:
            current = current | r
        else:
            bands.append(current)
            current = r
    bands.append(current)

    # Tìm gap dọc đủ lớn → ranh giới giữa 2 biểu đồ
    sub_regions: list = []
    y_cursor = pic_rect.y0

    for band in bands:
        gap = band.y0 - y_cursor
        if gap >= min_sub_height:
            sub_regions.append(fitz.Rect(
                pic_rect.x0,
                max(pic_rect.y0, y_cursor - padding),
                pic_rect.x1,
                min(pic_rect.y1, band.y0 + padding),
            ))
        y_cursor = max(y_cursor, band.y1)

    if pic_rect.y1 - y_cursor >= min_sub_height:
        sub_regions.append(fitz.Rect(
            pic_rect.x0,
            max(pic_rect.y0, y_cursor - padding),
            pic_rect.x1,
            pic_rect.y1,
        ))

    return sub_regions if len(sub_regions) >= 2 else []


def _extract_chart_title_in_region(fitz_page: fitz.Page, region: fitz.Rect) -> str:
    """
    Tìm tiêu đề biểu đồ trong vùng region.

    Lấy text block đầu tiên (dài nhất) nằm ở trên cùng của region
    (trong khoảng 50pt từ đỉnh vùng), dùng làm caption cho sub-chart.
    """
    raw_blocks = fitz_page.get_text("blocks")
    candidates = sorted(
        [
            b[4].strip()
            for b in raw_blocks
            if b[6] == 0
            and len(b[4].strip()) > 5
            and fitz.Rect(b[:4]).x0 >= region.x0 - 10
            and fitz.Rect(b[:4]).x1 <= region.x1 + 10
            and fitz.Rect(b[:4]).y0 >= region.y0 - 5
            and fitz.Rect(b[:4]).y1 <= region.y0 + 50
        ],
        key=lambda t: len(t),
        reverse=True,
    )
    return candidates[0] if candidates else ""


def _find_title_above_pic(
    pic,
    fitz_doc: fitz.Document,
    page_no: int,
    search_height: float = 50.0,
    min_len: int = 10,
) -> str:
    """
    Tìm text block nằm ngay PHÍA TRÊN bbox ảnh trong PDF.

    Nhiều biểu đồ tài chính có title kiểu "Giá than nhập khẩu bình quân..."
    nằm ngay trên ảnh nhưng Docling không detect thành CAPTION.

    Dùng fitz.get_text("blocks") → lọc block nằm trong vùng
    [pic.y0 - search_height .. pic.y0] và overlap ngang.

    Returns:
        Title string hoặc "" nếu không tìm thấy.
    """
    bbox = pic.prov[0].bbox
    fitz_page = fitz_doc[page_no - 1]
    ph = fitz_page.rect.height

    # Docling (bottom-left) → fitz (top-left)
    pic_top    = ph - bbox.t          # y0 trong fitz
    pic_left   = bbox.l
    pic_right  = bbox.r

    # Vùng tìm kiếm: phía trên ảnh, rộng bằng ảnh
    search_y0 = max(0.0, pic_top - search_height)
    search_y1 = pic_top + 5.0   # +5 tolerance

    raw_blocks = fitz_page.get_text("blocks")
    candidates: list[tuple[float, str]] = []
    _SOURCE_RE = re.compile(r'^(?:ngu[oồ]n|source)\s*:', re.IGNORECASE)

    for b in raw_blocks:
        if b[6] != 0:  # chỉ text blocks
            continue
        bx0, by0, bx1, by1 = b[:4]
        text = b[4].strip()

        # Block phải nằm trong vùng phía trên ảnh
        if by1 < search_y0 or by0 > search_y1:
            continue
        # Block phải overlap ngang với ảnh (không nằm lệch bên)
        if bx1 < pic_left - 20 or bx0 > pic_right + 20:
            continue
        # Bỏ text quá ngắn hoặc source note
        if len(text) < min_len:
            continue
        if _SOURCE_RE.match(text):
            continue
        # Bỏ nhãn trục (toàn số)
        if _is_axis_noise(text):
            continue

        # Ưu tiên block gần ảnh nhất (by1 lớn nhất = gần pic_top nhất)
        candidates.append((by1, text))

    if not candidates:
        return ""

    # Lấy block gần ảnh nhất
    candidates.sort(key=lambda x: x[0], reverse=True)
    # Gộp dòng, lấy dòng đầu (thường là title chính)
    title = candidates[0][1].replace("\n", " ").strip()
    return title


def _find_table_captions(texts: list[str]) -> list[str]:
    """
    Tìm các caption bảng ("Bảng 1:", "Bảng 2:") từ danh sách text items của trang.

    Trả về list theo thứ tự xuất hiện để ghép 1-1 với tables.
    """
    _TABLE_RE = re.compile(r'b[aả]ng\s*\d+\s*:', re.IGNORECASE)
    captions = []
    for t in texts:
        clean = t.lstrip('#').strip()
        if _TABLE_RE.search(clean):
            captions.append(_clean_ocr_text(clean))
    return captions


def _merge_cross_page_tables(
    metadata: list[dict], out_base: Path
) -> list[dict]:
    """
    Post-process: merge bảng continuation cross-page.

    Khi một bảng trải qua 2 trang PDF, Docling tạo 2 table items riêng biệt.
    Bảng ở trang sau thường:
      - Không có caption → tên generic "table_0"
      - Không có header row riêng (dòng đầu là data)
      - Cùng số cột với bảng cuối trang trước

    Hàm này merge rows của bảng trang sau vào bảng trang trước,
    cập nhật file .md và metadata, rồi xoá entry trang sau.

    Returns:
        metadata list đã loại bỏ các continuation tables đã merge.
    """
    # Chỉ xử lý table chunks, group theo doc
    from collections import defaultdict
    doc_tables: dict[str, list[int]] = defaultdict(list)
    for idx, m in enumerate(metadata):
        if m["chunk_type"] == "table":
            doc_tables[m["doc"]].append(idx)

    merged_indices: set[int] = set()  # indices to remove

    for doc_name, indices in doc_tables.items():
        # Sort theo page rồi chunk_index
        indices.sort(key=lambda i: (metadata[i]["page"], metadata[i]["chunk_index"]))

        for pos in range(len(indices) - 1):
            idx_prev = indices[pos]
            idx_next = indices[pos + 1]
            prev = metadata[idx_prev]
            nxt = metadata[idx_next]

            # Chỉ merge nếu trang liên tiếp
            if nxt["page"] != prev["page"] + 1:
                continue

            # Chỉ merge nếu bảng trang sau có tên generic
            if not nxt.get("table_name", "").startswith("table_"):
                continue

            # Bảng trước phải là bảng cuối trên trang đó
            same_page_after = [
                i for i in indices
                if metadata[i]["page"] == prev["page"]
                and metadata[i]["chunk_index"] > prev["chunk_index"]
            ]
            if same_page_after:
                continue

            # Bảng sau phải là bảng đầu trên trang đó
            same_page_before = [
                i for i in indices
                if metadata[i]["page"] == nxt["page"]
                and metadata[i]["chunk_index"] < nxt["chunk_index"]
            ]
            if same_page_before:
                continue

            # Đọc nội dung 2 bảng
            try:
                prev_path = Path(prev["text_path"])
                next_path = Path(nxt["text_path"])
                prev_content = prev_path.read_text(encoding="utf-8")
                next_content = next_path.read_text(encoding="utf-8")
            except Exception:
                continue

            # Parse markdown table lines
            prev_lines = [l for l in prev_content.split("\n") if l.strip().startswith("|")]
            next_lines = [l for l in next_content.split("\n") if l.strip().startswith("|")]

            if not prev_lines or not next_lines:
                continue

            # Kiểm tra số cột bằng nhau
            prev_cols = prev_lines[0].count("|") - 1
            next_cols = next_lines[0].count("|") - 1
            if prev_cols != next_cols:
                continue

            # Kiểm tra bảng sau có separator row (---|---) không
            # Docling to_markdown() LUÔN tạo separator (dùng data row đầu làm header)
            # → cần phân biệt: bảng mới thật sự vs continuation bị format lại
            sep_idx = -1
            for si, line in enumerate(next_lines[:3]):
                if all(c in "-|: " for c in line):
                    sep_idx = si
                    break

            if sep_idx >= 0 and len(next_lines) >= 2:
                # Có separator → kiểm tra "header" bảng sau
                next_header = next_lines[0].strip() if sep_idx > 0 else ""
                prev_header = prev_lines[0].strip()

                if next_header == prev_header:
                    # Docling lặp lại header y hệt → bỏ header + sep
                    next_data_lines = [
                        l for l in next_lines[sep_idx + 1:]
                        if l.strip().startswith("|")
                    ]
                else:
                    # Header khác → có thể là data row bị format thành pseudo-header
                    # Vẫn merge: lấy tất cả non-separator rows (gồm cả pseudo-header)
                    next_data_lines = [
                        l for l in next_lines
                        if l.strip().startswith("|") and not all(c in "-|: " for c in l)
                    ]
            else:
                # Không có separator → toàn bộ là data rows
                next_data_lines = next_lines

            if not next_data_lines:
                continue

            # === Merge ===
            # Nối data rows vào cuối file bảng trước
            merged_content = prev_content.rstrip("\n") + "\n" + "\n".join(next_data_lines) + "\n"
            prev_path.write_text(merged_content, encoding="utf-8")

            # Cập nhật token_count
            all_text = "\n".join(prev_lines + next_data_lines)
            prev["token_count"] = len(all_text.split())

            # Xoá file bảng sau
            try:
                next_path.unlink(missing_ok=True)
            except Exception:
                pass

            # Đánh dấu entry bảng sau để xoá khỏi metadata
            merged_indices.add(idx_next)

            parent_name = prev.get("table_name", "")
            print(f"  [merge] {nxt['chunk_id']} → {prev['chunk_id']} "
                  f"(+{len(next_data_lines)} rows, table: {parent_name})")

    # Loại bỏ các entry đã merge
    if merged_indices:
        metadata = [m for i, m in enumerate(metadata) if i not in merged_indices]
        print(f"  [merge] Đã merge {len(merged_indices)} continuation tables")

    return metadata