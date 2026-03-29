import os
from pathlib import Path

# Project root
BASE_DIR = Path(__file__).parent.parent

# Model paths
model_dir_env = os.getenv("MODEL_DIR")
if model_dir_env:
    MODEL_DIR = Path(model_dir_env)
else:
    MODEL_DIR = next(
        (path for path in (BASE_DIR / "models", BASE_DIR / "Models") if path.exists()),
        BASE_DIR / "models"
    )

ACCIDENT_MODEL_PATH = MODEL_DIR / "accident_detection_model.pt"
HELMET_MODEL_PATH = MODEL_DIR / "helmet_detection_model.pt"
POTHOLE_MODEL_PATH = MODEL_DIR / "pothole_detection_model.pt"


def _resolve_model_file(*candidates: str) -> Path:
    for candidate in candidates:
        path = MODEL_DIR / candidate
        if path.exists():
            return path
    return MODEL_DIR / candidates[0]


LICENSE_PLATE_MODEL_PATH = _resolve_model_file(
    "license_plate_recognization_model.pt",
    "License_plate_recognization_model.pt"
)

# Upload settings
UPLOAD_DIR = BASE_DIR / "uploads"
UPLOAD_DIR.mkdir(exist_ok=True)
MAX_UPLOAD_SIZE = 50 * 1024 * 1024  # 50 MB for images
ALLOWED_IMAGE_EXTENSIONS = {'.jpg', '.jpeg', '.png', '.bmp', '.gif', '.webp'}

# Image processing
IMAGE_RESIZE = (640, 480)  # Resize images for processing

# Device
DEVICE = "cuda" if __import__('torch').cuda.is_available() else "cpu"

# API settings
API_TITLE = "Smart City Road Intelligence API"
API_VERSION = "1.0.0"
