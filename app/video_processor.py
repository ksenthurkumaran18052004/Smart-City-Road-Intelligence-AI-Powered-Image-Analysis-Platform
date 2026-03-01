import cv2
import torch
from pathlib import Path
from app.models import (
    ModelLoader, AccidentDetector, HelmetDetector, 
    LicensePlateDetector, PotholeDetector
)
from app.ocr import LicensePlateOCR
from app.config import FRAME_SAMPLING_INTERVAL, ACCIDENT_THRESHOLD


class VideoAnalyzer:
    """
    Main video analysis pipeline
    Processes video frames and aggregates detection results
    """
    
    def __init__(self):
        """Initialize video analyzer with loaded models"""
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
    
    def analyze_video(self, video_path):
        """
        Complete video analysis pipeline
        
        Args:
            video_path: Path to video file
            
        Returns:
            Dictionary with analysis results
        """
        video_path = Path(video_path)
        
        if not video_path.exists():
            raise FileNotFoundError(f"Video not found: {video_path}")
        
        # Initialize results
        results = {
            'video_type': 'Normal',
            'accident_confidence': 0.0,
            'helmet_violations': {
                'with_helmet': 0,
                'without_helmet': 0
            },
            'detected_plates': [],
            'potholes_detected': 0
        }
        
        # Open video
        cap = cv2.VideoCapture(str(video_path))
        
        if not cap.isOpened():
            raise ValueError(f"Cannot open video: {video_path}")
        
        fps = cap.get(cv2.CAP_PROP_FPS)
        frame_interval = int(fps * FRAME_SAMPLING_INTERVAL) if fps > 0 else 1
        
        # Process frames
        frame_count = 0
        accident_predictions = []
        seen_plates = set()
        
        while True:
            ret, frame = cap.read()
            
            if not ret:
                break
            
            # Sample frames at specified interval
            if frame_count % frame_interval == 0:
                # Resize frame for processing (optional, for speed)
                processed_frame = cv2.resize(frame, (640, 480))
                
                # Run all detections
                self._process_frame(
                    processed_frame, results, 
                    accident_predictions, seen_plates
                )
            
            frame_count += 1
        
        cap.release()
        
        # Aggregate accident predictions
        if accident_predictions:
            accident_ratio = sum(accident_predictions) / len(accident_predictions)
            results['accident_confidence'] = float(accident_ratio)
            results['video_type'] = 'Accident' if accident_ratio > ACCIDENT_THRESHOLD else 'Normal'
        
        # Deduplicate plates
        results['detected_plates'] = list(seen_plates)
        
        return results
    
    def _process_frame(self, frame, results, accident_predictions, seen_plates):
        """
        Process a single frame with all detections
        
        Args:
            frame: OpenCV frame
            results: Results dictionary to update
            accident_predictions: List to append accident predictions
            seen_plates: Set to track detected plates
        """
        with torch.no_grad():
            # 1. Accident Detection
            is_accident, confidence = self.accident_detector.predict(frame)
            accident_predictions.append(1 if is_accident else 0)
            
            # 2. Helmet Detection
            with_helmet, without_helmet = self.helmet_detector.predict(frame)
            results['helmet_violations']['with_helmet'] += with_helmet
            results['helmet_violations']['without_helmet'] += without_helmet
            
            # 3. License Plate Detection & OCR
            plates = self.license_plate_detector.detect_plates(frame)
            for plate_info in plates:
                plate_region = plate_info['region']
                plate_conf = plate_info['confidence']
                
                # Run OCR
                plate_text, ocr_conf = self.ocr.recognize_plate(plate_region)
                
                if plate_text:
                    # Create plate entry
                    plate_entry = {
                        'plate_text': plate_text,
                        'confidence': min(plate_conf, ocr_conf)
                    }
                    
                    # Add to set (deduplicated)
                    plate_key = f"{plate_text}_{int(ocr_conf*100)}"
                    if plate_key not in seen_plates:
                        seen_plates.add(plate_key)
            
            # 4. Pothole Detection
            potholes = self.pothole_detector.detect_potholes(frame)
            results['potholes_detected'] += potholes


def validate_video_file(file_path):
    """
    Validate video file
    
    Args:
        file_path: Path to file
        
    Returns:
        True if valid, False otherwise
    """
    from app.config import ALLOWED_VIDEO_EXTENSIONS
    
    file_path = Path(file_path)
    
    # Check extension
    if file_path.suffix.lower() not in ALLOWED_VIDEO_EXTENSIONS:
        return False
    
    # Check if file exists and is readable
    if not file_path.exists() or not file_path.is_file():
        return False
    
    # Try to open with OpenCV
    cap = cv2.VideoCapture(str(file_path))
    is_valid = cap.isOpened()
    cap.release()
    
    return is_valid
