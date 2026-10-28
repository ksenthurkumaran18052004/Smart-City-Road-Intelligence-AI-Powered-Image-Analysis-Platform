import torch
import torch.nn as nn
import torch.nn.functional as F
import cv2
from pathlib import Path
from app.config import (
    ACCIDENT_MODEL_PATH, HELMET_MODEL_PATH, 
    LICENSE_PLATE_MODEL_PATH, POTHOLE_MODEL_PATH, DEVICE, VEHICLE_MODEL_NAME
)


class PotholeModel(nn.Module):
    """Architecture used by Training python Code/pothole_detection.py."""

    def __init__(self):
        super().__init__()
        import timm
        self.net = timm.create_model('res2net50d.in1k', pretrained=False, num_classes=4)

    def forward(self, inputs):
        return self.net(inputs)


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
            self._models['vehicle'] = self._load_vehicle_model()
            if self._models['vehicle']:
                print("✓ Person/vehicle model loaded (COCO)")
            else:
                print("✗ Person/vehicle model unavailable")
        except Exception as e:
            print(f"✗ Failed to load person/vehicle model: {str(e)}")
            self._models['vehicle'] = None

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
            self._models['pothole'] = self._load_pothole_model(POTHOLE_MODEL_PATH)
            if self._models['pothole']:
                print(f"✓ Pothole detection model loaded (PyTorch)")
            else:
                self._models['pothole'] = None
                print(f"✗ Pothole detection model skipped (not YOLO compatible)")
        except Exception as e:
            print(f"✗ Failed to load pothole model: {str(e)}")
            self._models['pothole'] = None
        
        self._loaded = True

    def _load_vehicle_model(self):
        """Load a general COCO model for rider and two-wheeler association."""
        try:
            from ultralytics import YOLO
            return YOLO(VEHICLE_MODEL_NAME)
        except Exception as e:
            print(f"  Vehicle model load error: {str(e)}")
            return None

    def _load_pothole_model(self, model_path):
        """Load the box-regression checkpoint produced by the training script."""
        model_path = Path(model_path)
        if not model_path.exists():
            return None
        try:
            checkpoint = torch.load(str(model_path), map_location=DEVICE, weights_only=False)
            model = PotholeModel().to(DEVICE)
            model.load_state_dict(checkpoint, strict=True)
            model.eval()
            return model
        except Exception as e:
            print(f"  Pothole checkpoint load error: {str(e)}")
            return None
    
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
        detections = self.detect(image)
        with_helmet = sum(1 for detection in detections if detection['class_id'] == 1)
        without_helmet = sum(1 for detection in detections if detection['class_id'] == 0)
        return with_helmet, without_helmet

    def detect(self, image):
        """Return helmet detections with coordinates for video reporting."""
        if self.model is None:
            return []
        try:
            with torch.no_grad():
                results = self.model.predict(source=image, conf=0.25, imgsz=1280, verbose=False)
            detections = []
            if hasattr(results[0], 'boxes'):
                for det in results[0].boxes:
                    conf = float(det.conf)
                    cls = int(det.cls)
                    if conf >= 0.25 and cls in (0, 1):
                        x1, y1, x2, y2 = map(int, det.xyxy[0])
                        detections.append({
                            'label': 'with_helmet' if cls == 1 else 'without_helmet',
                            'class_id': cls,
                            'confidence': conf,
                            'bbox': [x1, y1, x2, y2],
                        })
            return detections
        except Exception as e:
            print(f"Helmet detection error: {str(e)}")
            return []


class VehicleDetector:
    """Detect people and two-wheelers for traffic-context filtering."""

    PERSON = 0
    BICYCLE = 1
    CAR = 2
    MOTORCYCLE = 3
    BUS = 5
    TRUCK = 7

    def __init__(self, model):
        self.model = model

    def detect(self, image):
        if self.model is None:
            return []
        try:
            results = self.model.predict(
                source=image,
                conf=0.35,
                imgsz=1280,
                classes=[
                    self.PERSON, self.BICYCLE, self.CAR,
                    self.MOTORCYCLE, self.BUS, self.TRUCK,
                ],
                verbose=False,
            )
            detections = []
            for box in results[0].boxes:
                class_id = int(box.cls)
                x1, y1, x2, y2 = map(int, box.xyxy[0])
                detections.append({
                    'class_id': class_id,
                    'label': self.model.names[class_id],
                    'confidence': float(box.conf),
                    'bbox': [x1, y1, x2, y2],
                })
            return detections
        except Exception as e:
            print(f"Vehicle detection error: {str(e)}")
            return []


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
            height, width = image.shape[:2]
            tiles = [(image, 0, 0)]
            if height > 900:
                tile_height = int(height * 0.62)
                step = int(height * 0.42)
                tiles = []
                for y_offset in range(0, height, step):
                    y_end = min(height, y_offset + tile_height)
                    tiles.append((image[y_offset:y_end, :], 0, y_offset))
                    if y_end == height:
                        break

            candidates = []
            with torch.no_grad():
                for tile, x_offset, y_offset in tiles:
                    results = self.model.predict(
                        source=tile,
                        conf=0.15,
                        imgsz=1280,
                        max_det=50,
                        verbose=False,
                    )
                    if not hasattr(results[0], 'boxes'):
                        continue
                    for box in results[0].boxes:
                        conf = float(box.conf)
                        x1, y1, x2, y2 = map(int, box.xyxy[0])
                        x1, x2 = x1 + x_offset, x2 + x_offset
                        y1, y2 = y1 + y_offset, y2 + y_offset
                        x1, x2 = max(0, x1), min(width, x2)
                        y1, y2 = max(0, y1), min(height, y2)
                        box_width = x2 - x1
                        box_height = y2 - y1
                        aspect_ratio = box_width / box_height if box_height else 0
                        if (
                            conf >= 0.30
                            and box_width >= 20
                            and box_height >= 6
                            and 1.5 <= aspect_ratio <= 8.0
                        ):
                            candidates.append((conf, x1, y1, x2, y2))

            # Keep the strongest candidate when overlapping tiles see the same plate.
            candidates.sort(reverse=True)
            plates = []
            for conf, x1, y1, x2, y2 in candidates:
                duplicate = False
                for existing in plates:
                    ex1, ey1, ex2, ey2 = existing['bbox']
                    intersection = max(0, min(x2, ex2) - max(x1, ex1)) * max(0, min(y2, ey2) - max(y1, ey1))
                    union = (x2 - x1) * (y2 - y1) + (ex2 - ex1) * (ey2 - ey1) - intersection
                    if union and intersection / union > 0.4:
                        duplicate = True
                        break
                if not duplicate:
                    plates.append({
                        'region': image[y1:y2, x1:x2],
                        'confidence': conf,
                        'bbox': (x1, y1, x2, y2),
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
        self.is_regression_model = isinstance(model, PotholeModel)
    
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

        # This training artifact predicts a box for every image but has no
        # objectness/classification output, so it cannot distinguish a pothole
        # from a frame without a pothole.
        if self.is_regression_model:
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
