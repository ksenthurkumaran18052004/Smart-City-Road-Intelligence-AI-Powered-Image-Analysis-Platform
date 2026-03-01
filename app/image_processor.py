import cv2
import torch
import numpy as np
from pathlib import Path
from app.models import (
    ModelLoader, AccidentDetector, HelmetDetector, 
    LicensePlateDetector, PotholeDetector
)
from app.ocr import LicensePlateOCR
from app.config import IMAGE_RESIZE


class ImageAnalyzer:
    """
    Single image analysis pipeline
    Processes an image with all detection models
    """
    
    def __init__(self):
        """Initialize image analyzer with loaded models"""
        self.model_loader = ModelLoader()
        self.model_loader.load_all_models()
        
        # Initialize detector instances
        self.accident_detector = AccidentDetector(
            self.model_loader.get_model('accident')
        )
        self.helmet_detector = HelmetDetector(
            self.model_loader.get_model('helmet')
        )
        self.license_plate_detector = LicensePlateDetector(
            self.model_loader.get_model('license_plate')
        )
        self.pothole_detector = PotholeDetector(
            self.model_loader.get_model('pothole')
        )
        
        # OCR module
        self.ocr = LicensePlateOCR()
    
    def analyze_image(self, image_path):
        """
        Complete image analysis pipeline
        
        Args:
            image_path: Path to image file
            
        Returns:
            Dictionary with analysis results
        """
        image_path = Path(image_path)
        
        if not image_path.exists():
            raise FileNotFoundError(f"Image not found: {image_path}")
        
        # Read image
        image = cv2.imread(str(image_path))
        
        if image is None:
            raise ValueError(f"Cannot read image: {image_path}")
        
        # Resize for processing
        processed_image = cv2.resize(image, IMAGE_RESIZE)
        
        # Initialize results
        results = {
            'accident_detected': False,
            'accident_confidence': 0.0,
            'helmet_violations': {
                'with_helmet': 0,
                'without_helmet': 0
            },
            'detected_plates': [],
            'potholes_detected': 0
        }
        
        # Run all detections
        self._process_image(processed_image, results)
        
        return results
    
    def _process_image(self, image, results):
        """
        Process a single image with all detections
        
        Args:
            image: OpenCV image (BGR)
            results: Results dictionary to update
        """
        with torch.no_grad():
            # 1. Accident Detection
            is_accident, confidence = self.accident_detector.predict(image)
            results['accident_detected'] = bool(is_accident)
            results['accident_confidence'] = float(confidence)
            
            # 2. Helmet Detection
            with_helmet, without_helmet = self.helmet_detector.predict(image)
            results['helmet_violations']['with_helmet'] = with_helmet
            results['helmet_violations']['without_helmet'] = without_helmet
            
            # 3. License Plate Detection & OCR
            plates = self.license_plate_detector.detect_plates(image)
            detected_plates = []
            
            for plate_info in plates:
                plate_region = plate_info['region']
                plate_conf = plate_info['confidence']
                
                # Run OCR
                plate_text, ocr_conf = self.ocr.recognize_plate(plate_region)
                
                if plate_text:
                    # Create plate entry
                    plate_entry = {
                        'plate_text': plate_text,
                        'confidence': min(float(plate_conf), float(ocr_conf))
                    }
                    detected_plates.append(plate_entry)
            
            results['detected_plates'] = detected_plates
            
            # 4. Pothole Detection
            potholes = self.pothole_detector.detect_potholes(image)
            results['potholes_detected'] = int(potholes)


def validate_image_file(file_path):
    """
    Validate image file
    
    Args:
        file_path: Path to file
        
    Returns:
        True if valid, False otherwise
    """
    from app.config import ALLOWED_IMAGE_EXTENSIONS
    
    file_path = Path(file_path)
    
    # Check extension
    if file_path.suffix.lower() not in ALLOWED_IMAGE_EXTENSIONS:
        return False
    
    # Check if file exists and is readable
    if not file_path.exists() or not file_path.is_file():
        return False
    
    # Try to open with OpenCV
    img = cv2.imread(str(file_path))
    is_valid = img is not None
    
    return is_valid
