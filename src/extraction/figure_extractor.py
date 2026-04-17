# src/extraction/figure_extractor.py
# pyright: ignore[reportMissingModuleSource, reportAttributeAccessIssue, reportOperatorIssue]

"""
Tách biệt vùng biểu đồ từ trang PDF.

Vấn đề: biểu đồ trong báo cáo tài chính thường là VECTOR GRAPHICS
(PDF drawing operations), không phải ảnh nhúng (PNG/JPEG).
→ get_images() trả về rỗng.
→ get_drawings() trả về hàng trăm path nhỏ, clustering không đáng tin.

Giải pháp: tìm vùng chart bằng cách LOẠI TRỪ TEXT BLOCKS.
    "Vùng biểu đồ = vùng lớn trên trang không có text"

Pipeline:
    1. get_text("blocks") → lấy tất cả text bounding box
    2. Gom các text block gần nhau thành "text band"
    3. Khoảng trống dọc giữa các text band ≥ min_height → vùng biểu đồ
    4. Crop và lưu từng vùng

Fallback (nếu không phát hiện được):
    - Ảnh raster nhúng (get_images) — cho PDF nhúng chart dạng PNG/JPEG
    - Toàn trang (screenshot_full_page)
"""

import fitz  # type: ignore  # PyMuPDF — no type stubs available
from pathlib import Path


def get_chart_rects(page: fitz.Page) -> list:
    """
    Trả về list fitz.Rect của các vùng biểu đồ trên trang.
    Dùng bởi extractor.py để lọc text trước khi chunking.
    Kết hợp kết quả từ tất cả các phương pháp detect.
    """
    rects = []
    rects += _find_chart_regions_by_text_exclusion(page)
    rects += _find_chart_regions_by_columns(page)
    rects += _find_chart_regions_by_drawings(page)
    return rects


def extract_figures_from_page(
    page: fitz.Page,
    doc_name: str,
    page_index: int,
    out_dir: Path,
    min_height: int = 120,   # chiều cao tối thiểu (pt) của vùng chart
    dpi: int = 200,
) -> list[str]:
    """
    Trích xuất vùng biểu đồ từ một trang PDF.

    Ưu tiên phương pháp loại trừ text (phù hợp với vector chart).
    Fallback sang ảnh raster nếu text exclusion không tìm được gì.

    Returns:
        list đường dẫn ảnh đã lưu (rỗng nếu không tìm được vùng chart)
    """
    out_dir = Path(out_dir)
    figures: list[str] = []

    # ─────────────────────────────────────────────────────────────
    # Method 1a: Khoảng trống DỌC giữa các text band
    # Phù hợp với layout: text trên/dưới, chart ở giữa
    # ─────────────────────────────────────────────────────────────
    vertical_regions = _find_chart_regions_by_text_exclusion(page, min_height=min_height)
    for ci, region in enumerate(vertical_regions):
        clip_pix = page.get_pixmap(clip=region, dpi=dpi)
        fig_path = out_dir / f"{doc_name}_p{page_index:03d}_chart{ci:02d}.png"
        clip_pix.save(str(fig_path))
        figures.append(str(fig_path))

    # ─────────────────────────────────────────────────────────────
    # Method 1b: Khoảng trống NGANG — layout 2 cột (text | chart)
    # Sau khi phát hiện cột chart, chia nhỏ thêm thành từng biểu đồ
    # dựa trên chart title trong cột đó (gap dọc trong cột).
    # ─────────────────────────────────────────────────────────────
    column_regions = _find_chart_regions_by_columns(page)
    for col_idx, col_region in enumerate(column_regions):
        # Chia cột chart thành từng biểu đồ riêng lẻ
        sub_regions = _subdivide_region(page, col_region, min_height=60, padding=15)
        for ci, region in enumerate(sub_regions):
            clip_pix = page.get_pixmap(clip=region, dpi=dpi)
            fig_path = out_dir / f"{doc_name}_p{page_index:03d}_col{col_idx}_{ci:02d}.png"
            clip_pix.save(str(fig_path))
            figures.append(str(fig_path))

    if figures:
        return figures

    # ─────────────────────────────────────────────────────────────
    # Method 1c: Drawing-based — nhận diện trực tiếp thanh cột /
    # đường biểu đồ qua màu fill/stroke, bỏ qua đường kẻ bảng
    # ─────────────────────────────────────────────────────────────
    drawing_regions = _find_chart_regions_by_drawings(page, padding=25)
    for ci, region in enumerate(drawing_regions):
        clip_pix = page.get_pixmap(clip=region, dpi=dpi)
        fig_path = out_dir / f"{doc_name}_p{page_index:03d}_draw{ci:02d}.png"
        clip_pix.save(str(fig_path))
        figures.append(str(fig_path))

    if figures:
        return figures

    # ─────────────────────────────────────────────────────────────
    # Method 2 (fallback): Ảnh raster nhúng trong PDF
    # Một số báo cáo nhúng chart dưới dạng PNG/JPEG
    # ─────────────────────────────────────────────────────────────
    img_list = page.get_images(full=True)
    for img_idx, img_info in enumerate(img_list):
        xref  = img_info[0]
        rects = page.get_image_rects(xref)
        if not rects:
            continue
        rect = rects[0]
        # Bỏ qua icon / logo / đường viền nhỏ
        if rect.width < 100 or rect.height < 100:
            continue
        clip_pix = page.get_pixmap(clip=rect, dpi=dpi)
        fig_path = out_dir / f"{doc_name}_p{page_index:03d}_img{img_idx:02d}.png"
        clip_pix.save(str(fig_path))
        figures.append(str(fig_path))

    return figures


def screenshot_full_page(
    page: fitz.Page,
    doc_name: str,
    page_index: int,
    out_dir: Path,
    dpi: int = 200,
) -> str:
    """
    Fallback cuối cùng: chụp toàn trang.
    Chỉ gọi khi extract_figures_from_page() trả về rỗng.
    """
    out_dir = Path(out_dir)
    img_path = out_dir / f"{doc_name}_p{page_index:03d}_full.png"
    pix = page.get_pixmap(dpi=dpi)
    pix.save(str(img_path))
    return str(img_path)


# ─────────────────────────────────────────────────────────────────────
# Core logic: tìm vùng chart bằng loại trừ text
# ─────────────────────────────────────────────────────────────────────

def _find_chart_regions_by_text_exclusion(
    page: fitz.Page,
    min_height: int = 60,
    padding: int = 60,
) -> list[fitz.Rect]:
    """
    Tìm vùng biểu đồ bằng cách loại trừ các text block.

    Thuật toán:
        1. Lấy tất cả text blocks (get_text "blocks")
        2. Lọc block có đủ nội dung (bỏ số lẻ / nhãn trục ngắn)
        3. Gom các block gần nhau thành text band (theo chiều dọc)
        4. Khoảng trống dọc giữa các text band ≥ min_height → vùng chart

    Args:
        min_height: chiều cao tối thiểu (pt) của khoảng trống để coi là chart.
                    Thấp (60) để không bỏ sót biểu đồ nhỏ.
        padding:    số pt mở rộng ra mỗi cạnh — bắt đủ nhãn trục / legend
                    nằm sát biểu đồ. Thừa vẫn được, thiếu mới hỏng.

    Returns:
        list[fitz.Rect] — các vùng chart đã xác định, đã có padding
    """
    page_rect = page.rect

    raw_blocks = page.get_text("blocks")

    # Chỉ lấy text block có nội dung thực (bỏ số lẻ / nhãn trục ngắn)
    text_rects = [
        fitz.Rect(b[:4])
        for b in raw_blocks
        if b[6] == 0 and len(b[4].strip()) > 8
    ]

    if not text_rects:
        return []

    text_rects.sort(key=lambda r: r.y0)
    bands = _merge_rects_vertical(text_rects, merge_gap=8)

    chart_regions: list[fitz.Rect] = []
    y_cursor = page_rect.y0

    for band in bands:
        gap = band.y0 - y_cursor
        if gap >= min_height:
            # Mở rộng padding mọi hướng, giới hạn trong trang
            region = fitz.Rect(
                page_rect.x0,
                max(page_rect.y0, y_cursor - padding),
                page_rect.x1,
                min(page_rect.y1, band.y0  + padding),
            )
            chart_regions.append(region)
        y_cursor = max(y_cursor, band.y1)

    # Khoảng trống cuối trang
    if page_rect.y1 - y_cursor >= min_height:
        region = fitz.Rect(
            page_rect.x0,
            max(page_rect.y0, y_cursor - padding),
            page_rect.x1,
            page_rect.y1,
        )
        chart_regions.append(region)

    return chart_regions


def _merge_rects_vertical(rects: list[fitz.Rect], merge_gap: int = 8) -> list[fitz.Rect]:
    """
    Gom các rect gần nhau theo chiều dọc thành một rect lớn.

    Hai rect được gom nếu khoảng cách dọc giữa chúng ≤ merge_gap.
    """
    if not rects:
        return []

    merged: list[fitz.Rect] = []
    current = rects[0]

    for r in rects[1:]:
        if r.y0 - current.y1 <= merge_gap:
            current = current | r   # union
        else:
            merged.append(current)
            current = r

    merged.append(current)
    return merged


def _find_chart_regions_by_drawings(
    page: fitz.Page,
    min_size: int = 60,
    padding: int = 35,
) -> list[fitz.Rect]:
    """
    Phát hiện vùng biểu đồ bằng cách nhận diện trực tiếp các thành phần
    đồ hoạ có màu (thanh cột, đường biểu đồ) qua get_drawings().

    Phân biệt chart vs table border:
        Thanh cột  → fill màu (fill ≠ None, ≠ trắng, ≠ đen)
        Đường chart→ stroke màu (color ≠ đen), width > 1pt
        Viền bảng  → stroke đen/xám, width ≈ 0.5pt  ← loại bỏ
        Lưới trục  → stroke xám nhạt, width mảnh     ← loại bỏ

    Sau khi lọc ra các element chart, gom cụm 2D và trả về bounding box
    của từng cụm (= từng biểu đồ riêng lẻ).

    Args:
        min_size: kích thước tối thiểu của cụm để coi là biểu đồ
        padding:  số pt mở rộng mỗi cạnh

    Returns:
        list[fitz.Rect] — vùng của từng biểu đồ phát hiện được
    """
    drawings = page.get_drawings()
    if not drawings:
        return []

    chart_elements: list[fitz.Rect] = []

    for d in drawings:
        raw_rect = d.get("rect")
        if not raw_rect:
            continue
        r = fitz.Rect(raw_rect)

        # Bỏ element quá nhỏ (điểm, tick nhỏ)
        if r.width < 3 or r.height < 3:
            continue

        fill  = d.get("fill")    # (r,g,b) hoặc None
        color = d.get("color")   # (r,g,b) hoặc None
        width = d.get("width") or 0

        # ── Thanh cột: fill màu (không phải trắng / đen / None) ──
        is_colored_fill = (
            fill is not None
            and fill != (1.0, 1.0, 1.0)   # không trắng
            and fill != (0.0, 0.0, 0.0)   # không đen
            and not (fill[0] > 0.85 and fill[1] > 0.85 and fill[2] > 0.85)  # không xám rất nhạt
        )

        # ── Đường biểu đồ: stroke màu, đủ dày, đủ dài ──
        is_colored_stroke = (
            color is not None
            and color != (0.0, 0.0, 0.0)   # không đen (loại viền bảng)
            and not (color[0] > 0.7 and color[1] > 0.7 and color[2] > 0.7)  # không xám nhạt
            and width >= 1.0               # đủ dày (loại grid line mảnh)
            and (r.width > 15 or r.height > 15)  # đủ dài
        )

        if is_colored_fill or is_colored_stroke:
            chart_elements.append(r)

    if not chart_elements:
        return []

    # Gom cụm 2D: hai element thuộc cùng cụm nếu khoảng cách ≤ gap
    clusters = _cluster_rects_2d(chart_elements, gap=40)

    page_rect = page.rect
    result: list[fitz.Rect] = []

    for cluster in clusters:
        if cluster.width < min_size or cluster.height < min_size:
            continue
        padded = fitz.Rect(
            max(page_rect.x0, cluster.x0 - padding),
            max(page_rect.y0, cluster.y0 - padding),
            min(page_rect.x1, cluster.x1 + padding),
            min(page_rect.y1, cluster.y1 + padding),
        )
        result.append(padded)

    return result


def _subdivide_region(
    page: fitz.Page,
    region: fitz.Rect,
    min_height: int = 60,
    padding: int = 25,
) -> list[fitz.Rect]:
    """
    Chia một vùng chart lớn (cả cột) thành từng biểu đồ nhỏ riêng lẻ.

    Dùng chart title bên trong cột làm ranh giới phân tách.
    Nếu không tìm được gap, trả về [region] nguyên vẹn.
    """
    raw_blocks = page.get_text("blocks")

    inner_rects = [
        fitz.Rect(b[:4])
        for b in raw_blocks
        if b[6] == 0
        and len(b[4].strip()) > 8
        and fitz.Rect(b[:4]).x0 >= region.x0 - 10
        and fitz.Rect(b[:4]).x1 <= region.x1 + 10
        and fitz.Rect(b[:4]).y0 >= region.y0
        and fitz.Rect(b[:4]).y1 <= region.y1
    ]

    if not inner_rects:
        return [region]

    inner_rects.sort(key=lambda r: r.y0)
    bands = _merge_rects_vertical(inner_rects, merge_gap=8)

    sub_regions: list[fitz.Rect] = []
    y_cursor = region.y0

    for band in bands:
        gap = band.y0 - y_cursor
        if gap >= min_height:
            sub_regions.append(fitz.Rect(
                region.x0,
                max(region.y0, y_cursor - padding),
                region.x1,
                min(region.y1, band.y0 + padding),
            ))
        y_cursor = max(y_cursor, band.y1)

    if region.y1 - y_cursor >= min_height:
        sub_regions.append(fitz.Rect(
            region.x0,
            max(region.y0, y_cursor - padding),
            region.x1,
            region.y1,
        ))

    return sub_regions if sub_regions else [region]


def _cluster_rects_2d(rects: list[fitz.Rect], gap: int = 40) -> list[fitz.Rect]:
    """
    Gom các rect gần nhau trong không gian 2D thành cụm.

    Hai rect thuộc cùng cụm nếu khoảng cách theo bất kỳ chiều nào ≤ gap.
    Khác với _merge_rects_vertical (chỉ theo chiều dọc), hàm này xử lý
    cả layout ngang (ví dụ: thanh cột nằm cạnh nhau).

    Dùng thuật toán union-find đơn giản:
        - Bắt đầu mỗi rect là 1 cụm riêng
        - Lặp: merge hai cụm nếu chúng đủ gần (expand mỗi cụm bởi gap,
          kiểm tra intersects)
        - Dừng khi không có merge nào xảy ra
    """
    if not rects:
        return []

    clusters = list(rects)   # bắt đầu: mỗi rect = 1 cụm

    changed = True
    while changed:
        changed = False
        merged: list[fitz.Rect] = []
        used = [False] * len(clusters)

        for i in range(len(clusters)):
            if used[i]:
                continue
            current = clusters[i]
            for j in range(i + 1, len(clusters)):
                if used[j]:
                    continue
                # Expand current bởi gap rồi kiểm tra overlap
                expanded = fitz.Rect(
                    current.x0 - gap, current.y0 - gap,
                    current.x1 + gap, current.y1 + gap,
                )
                if expanded.intersects(clusters[j]):
                    current = current | clusters[j]
                    used[j] = True
                    changed = True
            merged.append(current)
            used[i] = True

        clusters = merged

    return clusters


def _find_chart_regions_by_columns(
    page: fitz.Page,
    min_col_ratio: float = 0.25,
    text_side_threshold: float = 0.65,
    padding: int = 40,
) -> list[fitz.Rect]:
    """
    Phát hiện layout 2 cột: một cột text, một cột biểu đồ.

    Ví dụ: trang SSI có text bên trái (0~50%) và 2 biểu đồ bên phải (50~100%).
    Thuật toán dọc bỏ sót vì text trái bao phủ toàn chiều cao → không có gap dọc.

    Thuật toán:
        1. Tính x-center của từng text block
        2. Nếu ≥ text_side_threshold (65%) số block có x-center < 50% trang
           → text chủ yếu bên TRÁI → vùng phải là chart
        3. Ngược lại nếu ≥ 65% block bên phải → vùng trái là chart
        4. Mở rộng padding để bắt đủ nhãn

    Args:
        min_col_ratio:        cột chart phải chiếm ít nhất X% chiều ngang
        text_side_threshold:  tỷ lệ tối thiểu text blocks về 1 phía
        padding:              số pt mở rộng mỗi cạnh vùng chart
    """
    page_rect = page.rect
    raw_blocks = page.get_text("blocks")

    # Dùng "body text" block (đoạn văn dài) để phán đoán cột text vs cột chart.
    # Lọc >50 ký tự để loại chart title, legend, source note ngắn
    # — những thứ này nằm bên phải và làm sai tỷ lệ left/right.
    body_rects = [
        fitz.Rect(b[:4])
        for b in raw_blocks
        if b[6] == 0 and len(b[4].strip()) > 50
    ]

    # Fallback: nếu ít block thỏa >50, thử lại với >15
    text_rects = body_rects if len(body_rects) >= 2 else [
        fitz.Rect(b[:4])
        for b in raw_blocks
        if b[6] == 0 and len(b[4].strip()) > 15
    ]

    if len(text_rects) < 2:
        return []

    page_mid = page_rect.width / 2
    x_centers = [(r.x0 + r.x1) / 2 for r in text_rects]
    left_count  = sum(1 for x in x_centers if x < page_mid)
    right_count = len(x_centers) - left_count
    total       = len(x_centers)

    # y-extent của toàn bộ text (dùng để giới hạn chiều cao vùng chart)
    text_y0 = min(r.y0 for r in text_rects)
    text_y1 = max(r.y1 for r in text_rects)

    chart_regions: list[fitz.Rect] = []

    # Phần lớn text bên TRÁI → chart bên PHẢI
    if left_count / total >= text_side_threshold:
        left_blocks = [r for r in text_rects if (r.x0 + r.x1) / 2 < page_mid]
        split_x = max(r.x1 for r in left_blocks)
        chart_width = page_rect.x1 - split_x
        if chart_width >= min_col_ratio * page_rect.width:
            chart_regions.append(fitz.Rect(
                max(page_rect.x0, split_x    - padding),
                max(page_rect.y0, text_y0    - padding),
                page_rect.x1,
                min(page_rect.y1, text_y1    + padding),
            ))

    # Phần lớn text bên PHẢI → chart bên TRÁI
    elif right_count / total >= text_side_threshold:
        right_blocks = [r for r in text_rects if (r.x0 + r.x1) / 2 >= page_mid]
        split_x = min(r.x0 for r in right_blocks)
        chart_width = split_x - page_rect.x0
        if chart_width >= min_col_ratio * page_rect.width:
            chart_regions.append(fitz.Rect(
                page_rect.x0,
                max(page_rect.y0, text_y0    - padding),
                min(page_rect.x1, split_x    + padding),
                min(page_rect.y1, text_y1    + padding),
            ))

    return chart_regions
