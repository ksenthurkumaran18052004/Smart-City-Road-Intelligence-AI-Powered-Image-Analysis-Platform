import torch
import torch.nn as nn
from pathlib import Path
from app.config import (
    ACCIDENT_MODEL_PATH, HELMET_MODEL_PATH, 
    LICENSE_PLATE_MODEL_PATH, POTHOLE_MODEL_PATH, DEVICE
)


class ModelLoader:
    """
    Singleton class to load and manage all detection models
    Models are loaded once at startup and reused for inference
    """
    
    _instance = None
    _models = {}
    
    def __new__(cls):
        if cls._instance is None:
            cls._instance = super(ModelLoader, cls).__new__(cls)
            cls._instance._loaded = False
        return cls._instance
    
    def load_all_models(self):
        """
        Load all models at startup
        """
        if self._loaded:
            return
        
        print(f"Loading models on device: {DEVICE}")
        
        try:
            # Try loading accident detection model as YOLO first
            self._models['accident'] = self._load_yolo_model(ACCIDENT_MODEL_PATH)
            if self._models['accident']:
                print(f"✓ Accident detection model loaded (YOLO)")
            else:
                self._models['accident'] = None
                print(f"✗ Accident detection model skipped (not YOLO compatible)")
        except Exception as e:
            print(f"✗ Failed to load accident model: {str(e)}")
            self._models['accident'] = None
        
        try:
            # Load helmet detection model
            self._models['helmet'] = self._load_yolo_model(HELMET_MODEL_PATH)
            if self._models['helmet']:
                print(f"✓ Helmet detection model loaded (YOLO)")
            else:
                self._models['helmet'] = None
                print(f"✗ Helmet detection model skipped (not YOLO compatible)")
        except Exception as e:
            print(f"✗ Failed to load helmet model: {str(e)}")
            self._models['helmet'] = None
        
        try:
            # Load license plate recognition model
            self._models['license_plate'] = self._load_yolo_model(LICENSE_PLATE_MODEL_PATH)
            if self._models['license_plate']:
                print(f"✓ License plate model loaded (YOLO)")
            else:
                self._models['license_plate'] = None
                print(f"✗ License plate model skipped (not YOLO compatible)")
        except Exception as e:
            print(f"✗ Failed to load license plate model: {str(e)}")
            self._models['license_plate'] = None
        
        try:
            # Load pothole detection model
            self._models['pothole'] = self._load_yolo_model(POTHOLE_MODEL_PATH)
            if self._models['pothole']:
                print(f"✓ Pothole detection model loaded (YOLO)")
            else:
                self._models['pothole'] = None
                print(f"✗ Pothole detection model skipped (not YOLO compatible)")
        except Exception as e:
            print(f"✗ Failed to load pothole model: {str(e)}")
            self._models['pothole'] = None
        
        self._loaded = True
    
    def _load_yolo_model(self, model_path):
        """
        Load a YOLO model from file
        
        Args:
            model_path: Path to .pt model file
            
        Returns:
            Loaded model or None if not available
        """
        model_path = Path(model_path)
        
        if not model_path.exists():
            print(f"  Model file not found: {model_path}")
            return None
        
        try:
            from ultralytics import YOLO
            model = YOLO(str(model_path))
            return model
        except Exception as e:
            print(f"  YOLO load error: {str(e)}")
            return None
    
    def get_model(self, model_name):
        """
        Get a loaded model
        
        Args:
            model_name: 'accident', 'helmet', 'license_plate', or 'pothole'
            
        Returns:
            Loaded model or None if not available
        """
        if not self._loaded:
            self.load_all_models()
        
        return self._models.get(model_name)
    
    def get_all_models(self):
        """
        Get all loaded models
        
        Returns:
            Dictionary of all models
        """
        if not self._loaded:
            self.load_all_models()
        
        return self._models


class AccidentDetector:
    """
    Accident detection inference
    """
    
    def __init__(self, model):
        self.model = model
    
    def predict(self, image):
        """
        Predict if image contains accident
        
        Args:
            image: OpenCV image (BGR)
            
        Returns:
            Tuple of (is_accident: bool, confidence: float)
        """
        if self.model is None:
            return False, 0.0
        
        try:
            with torch.no_grad():
                # Inference on image
                results = self.model(image)
                
                # Extract prediction (assuming YOLO output)
                if hasattr(results[0], 'probs'):  # YOLOv8 classification
                    probs = results[0].probs
                    is_accident = probs.top1 == 1  # Assuming class 1 is accident
                    confidence = float(probs.top1conf)
                    return bool(is_accident), float(confidence)
                
                return False, 0.0
        
        except Exception as e:
            print(f"Accident detection error: {str(e)}")
            return False, 0.0


class HelmetDetector:
    """
    Helmet detection inference
    """
    
    def __init__(self, model):
        self.model = model
    
    def predict(self, image):
        """
        Detect helmets in image
        
        Args:
            image: OpenCV image (BGR)
            
        Returns:
            Tuple of (with_helmet_count, without_helmet_count)
        """
        if self.model is None:
            return 0, 0
        
        try:
            with torch.no_grad():
                # Inference on image
                results = self.model(image)
                
                with_helmet = 0
                without_helmet = 0
                
                # Parse detections
                if hasattr(results[0], 'boxes'):
                    for det in results[0].boxes:
                        conf = float(det.conf)
                        cls = int(det.cls)
                        
                        # Assuming class mapping: 1=with helmet, 0=without helmet
                        if conf > 0.5:
                            if cls == 1:
                                with_helmet += 1
                            elif cls == 0:
                                without_helmet += 1
                
                return with_helmet, without_helmet
        
        except Exception as e:
            print(f"Helmet detection error: {str(e)}")
            return 0, 0


class LicensePlateDetector:
    """
    License plate detection inference
    """
    
    def __init__(self, model):
        self.model = model
    
    def detect_plates(self, image):
        """
        Detect license plates in image
        
        Args:
            image: OpenCV image (BGR)
            
        Returns:
            List of detected plate regions as numpy arrays
        """
        if self.model is None:
            return []
        
        try:
            with torch.no_grad():
                results = self.model(image)
                
                plates = []
                
                if hasattr(results[0], 'boxes'):
                    for box in results[0].boxes:
                        conf = float(box.conf)
                        
                        if conf > 0.5:
                            # Extract bounding box coordinates
                            x1, y1, x2, y2 = map(int, box.xyxy[0])
                            
                            # Crop plate region
                            plate_region = image[y1:y2, x1:x2]
                            
                            if plate_region.size > 0:
                                plates.append({
                                    'region': plate_region,
                                    'confidence': conf,
                                    'bbox': (x1, y1, x2, y2)
                                })
                
                return plates
        
        except Exception as e:
            print(f"License plate detection error: {str(e)}")
            return []


class PotholeDetector:
    """
    Pothole detection inference
    """
    
    def __init__(self, model):
        self.model = model
    
    def detect_potholes(self, image):
        """
        Detect potholes in image
        
        Args:
            image: OpenCV image (BGR)
            
        Returns:
            Number of potholes detected
        """
        if self.model is None:
            return 0
        
        try:
            with torch.no_grad():
                results = self.model(image)
                
                pothole_count = 0
                
                if hasattr(results[0], 'boxes'):
                    for det in results[0].boxes:
                        conf = float(det.conf)
                        
                        if conf > 0.5:
                            pothole_count += 1
                
                return pothole_count
        
        except Exception as e:
            print(f"Pothole detection error: {str(e)}")
            return 0
