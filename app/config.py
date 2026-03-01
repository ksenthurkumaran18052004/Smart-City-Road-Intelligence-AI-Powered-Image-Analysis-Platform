import os
from pathlib import Path

# Project root
BASE_DIR = Path(__file__).parent.parent

# Model paths
MODEL_DIR = BASE_DIR / "models"
ACCIDENT_MODEL_PATH = MODEL_DIR / "accident_detection_model.pt"
HELMET_MODEL_PATH = MODEL_DIR / "helmet_detection_model.pt"
LICENSE_PLATE_MODEL_PATH = MODEL_DIR / "license_plate_recognization_model.pt"
POTHOLE_MODEL_PATH = MODEL_DIR / "pothole_detection_model.pt"

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
