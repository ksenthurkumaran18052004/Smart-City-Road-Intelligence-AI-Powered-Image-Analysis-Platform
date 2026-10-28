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
        # Small video detections need more pixels before thresholding.
        height, width = image.shape[:2]
        if height < 80 or width < 160:
            image = cv2.resize(image, None, fx=4, fy=4, interpolation=cv2.INTER_CUBIC)

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
            
            # License plates are a single compact text line, not a document.
            variants = [processed]
            if len(processed.shape) == 2:
                variants.append(cv2.bitwise_not(processed))

            best_text = ""
            best_confidence = 0.0
            config = '--psm 6 -c tessedit_char_whitelist=ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789'
            for variant in variants:
                data = pytesseract.image_to_data(
                    variant,
                    config=config,
                    output_type=pytesseract.Output.DICT,
                )
                text = " ".join(data["text"]).strip()
                confidences = []
                for conf in data["conf"]:
                    try:
                        value = float(conf)
                    except (TypeError, ValueError):
                        continue
                    if value > 0:
                        confidences.append(value)
                confidence = np.mean(confidences) / 100.0 if confidences else 0.0
                if len(text) > len(best_text) or confidence > best_confidence:
                    best_text = text
                    best_confidence = confidence

            return best_text, best_confidence
        
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
        return self.extract_registration_number(cleaned_text), confidence

    def extract_registration_number(self, text):
        """Return only text matching a common Indian registration format."""
        tokens = text.split()
        candidates = list(tokens)
        for width in (2, 3, 4):
            candidates.extend(''.join(tokens[index:index + width]) for index in range(len(tokens) - width + 1))

        for candidate in candidates:
            candidate = re.sub(r'[^A-Z0-9]', '', candidate)
            if candidate.startswith('IS') and len(candidate) > 7:
                candidate = candidate[1:]
            if candidate.startswith('SHART'):
                candidate = 'SMART' + candidate[5:]
            if len(candidate) >= 3:
                prefix, suffix = candidate[:-3], candidate[-3:]
                candidate = prefix + suffix.replace('O', '0')
            indian_format = re.match(r'^[A-Z]{2}\d{1,2}[A-Z]{1,3}\d{1,4}$', candidate)
            generic_plate_format = (
                5 <= len(candidate) <= 12
                and re.search(r'[A-Z]', candidate)
                and re.search(r'\d', candidate)
            )
            if indian_format or generic_plate_format:
                return candidate
        return ''
