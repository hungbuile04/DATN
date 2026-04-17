# config/settings.py
import yaml
from pathlib import Path

# Tìm thư mục gốc (source_code/) dù gọi từ bất kỳ đâu
# Lý do: nếu dùng Path("pdfs/raw") thì phụ thuộc vào
# thư mục hiện tại khi chạy script → dễ bị lỗi
ROOT = Path(__file__).parent.parent  # config/ -> source_code/

def load_config() -> dict:
    config_path = ROOT / "config" / "config.yaml"
    with open(config_path, encoding="utf-8") as f:
        cfg = yaml.safe_load(f)

    # Chuyển tất cả path thành Path tuyệt đối
    # Lý do: tránh lỗi "file not found" khi chạy từ thư mục khác
    for key, rel_path in cfg["paths"].items():
        cfg["paths"][key] = ROOT / rel_path

    return cfg

# Load một lần khi import — không load lại mỗi lần gọi
CFG = load_config()