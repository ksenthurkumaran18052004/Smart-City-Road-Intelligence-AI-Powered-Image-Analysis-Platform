import re
import cv2
import pytesseract
import numpy as np


class LicensePlateOCR:
    """
    OCR module for license plate text recognition
    """
    
    def __init__(self, tesseract_path=None):
        """
        Initialize OCR module
        
        Args:
            tesseract_path: Path to tesseract executable (if not in system PATH)
        """
        if tesseract_path:
            pytesseract.pytesseract.pytesseract_cmd = tesseract_path
    
    def preprocess_plate_image(self, image):
        """
        Preprocess license plate image for better OCR accuracy
        
        Args:
            image: OpenCV image of license plate
            
        Returns:
            Preprocessed image
        """
        # Convert to grayscale
        if len(image.shape) == 3:
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        else:
            gray = image
        
        # Apply bilateral filter to reduce noise while keeping edges
        blurred = cv2.bilateralFilter(gray, 11, 17, 17)
        
        # Apply thresholding
        _, thresh = cv2.threshold(blurred, 150, 255, cv2.THRESH_BINARY)
        
        return thresh
    
    def extract_text(self, image):
        """
        Extract text from license plate image using Tesseract
        
        Args:
            image: OpenCV image of license plate
            
        Returns:
            Tuple of (text, confidence)
        """
        try:
            # Preprocess image
            processed = self.preprocess_plate_image(image)
            
            # Use Tesseract to extract text
            data = pytesseract.image_to_data(processed, output_type=pytesseract.Output.DICT)
            
            # Aggregate text and confidence
            text = " ".join(data["text"])
            confidences = [int(conf) for conf in data["conf"] if int(conf) > 0]
            
            if confidences:
                avg_confidence = np.mean(confidences) / 100.0
            else:
                avg_confidence = 0.0
            
            return text.strip(), avg_confidence
        
        except Exception as e:
            print(f"OCR Error: {str(e)}")
            return "", 0.0
    
    def clean_plate_text(self, text):
        """
        Clean and normalize license plate text
        
        Args:
            text: Raw OCR text
            
        Returns:
            Cleaned plate text
        """
        # Remove extra spaces
        text = " ".join(text.split())
        
        # Remove special characters, keep alphanumeric
        text = re.sub(r'[^A-Z0-9\s]', '', text.upper())
        
        return text
    
    def recognize_plate(self, image):
        """
        Complete pipeline: Extract and clean license plate text
        
        Args:
            image: OpenCV image of license plate
            
        Returns:
            Cleaned plate text
        """
        raw_text, confidence = self.extract_text(image)
        cleaned_text = self.clean_plate_text(raw_text)
        return cleaned_text, confidence
