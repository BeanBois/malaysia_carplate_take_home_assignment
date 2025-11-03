import cv2
import numpy as np
import re
from typing import List, Dict, Optional, Tuple
import logging
import yaml
from pathlib import Path

# OCR engines
try:
    import easyocr
    EASYOCR_AVAILABLE = True
except ImportError:
    EASYOCR_AVAILABLE = False

try:
    from paddleocr import PaddleOCR
    PADDLEOCR_AVAILABLE = True
except ImportError:
    PADDLEOCR_AVAILABLE = False

try:
    import pytesseract
    TESSERACT_AVAILABLE = True
except ImportError:
    TESSERACT_AVAILABLE = False

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class LicensePlateRecognizer:
    """
    Recognizes text from license plate images
    Validates against Malaysian plate formats
    """
    
    def __init__(self, config_path: str = "config.yaml"):
        """
        Initialize the recognizer
        
        Args:
            config_path: Path to configuration file
        """
        with open(config_path, 'r') as f:
            self.config = yaml.safe_load(f)
        
        self.recognition_config = self.config['recognition']
        self.plate_format = self.config['plate_format']
        
        # Initialize OCR engine
        self.ocr_engine = self.recognition_config['ocr_engine']
        self._initialize_ocr()
        
        # Compile regex patterns for validation
        self.patterns = [re.compile(p) for p in self.plate_format['patterns']]
        
        # Log validation status
        if not self.plate_format.get('enforce_validation', True):
            logger.info("⚠️  Malaysian plate format validation is DISABLED (enforce_validation=false)")
            logger.info("   This is appropriate when using international/integrated datasets")
        else:
            logger.info("✓ Malaysian plate format validation is ENABLED")
    
    def _initialize_ocr(self):
        """Initialize the selected OCR engine"""
        
        if self.ocr_engine == 'easyocr':
            if not EASYOCR_AVAILABLE:
                raise ImportError("EasyOCR not installed. Install with: pip install easyocr")
            logger.info("Initializing EasyOCR...")
            self.reader = easyocr.Reader(
                self.recognition_config['languages'],
                gpu=True
            )
        
        elif self.ocr_engine == 'paddleocr':
            if not PADDLEOCR_AVAILABLE:
                raise ImportError("PaddleOCR not installed. Install with: pip install paddleocr")
            logger.info("Initializing PaddleOCR...")
            self.reader = PaddleOCR(
                use_angle_cls=True,
                lang='en',
                show_log=False
            )
        
        elif self.ocr_engine == 'tesseract':
            if not TESSERACT_AVAILABLE:
                raise ImportError("Tesseract not installed. Install with: pip install pytesseract")
            logger.info("Using Tesseract OCR...")
            self.reader = None  # Tesseract doesn't need initialization
        
        else:
            raise ValueError(f"Unknown OCR engine: {self.ocr_engine}")
    
    def preprocess_plate(self, plate_image: np.ndarray) -> np.ndarray:
        """
        Preprocess plate image for better OCR results
        
        Args:
            plate_image: Cropped plate image
        
        Returns:
            Preprocessed image
        """
        # Convert to grayscale
        if len(plate_image.shape) == 3:
            gray = cv2.cvtColor(plate_image, cv2.COLOR_BGR2GRAY)
        else:
            gray = plate_image
        
        # Resize to target dimensions for consistent processing
        target_h = self.recognition_config['target_height']
        target_w = self.recognition_config['target_width']
        resized = cv2.resize(gray, (target_w, target_h))
        
        # Apply CLAHE for better contrast
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        enhanced = clahe.apply(resized)
        
        # Denoise
        denoised = cv2.fastNlMeansDenoising(enhanced)
        
        # Adaptive thresholding
        binary = cv2.adaptiveThreshold(
            denoised,
            255,
            cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
            cv2.THRESH_BINARY,
            11,
            2
        )
        
        # Morphological operations to connect broken characters
        kernel = np.ones((2, 2), np.uint8)
        morph = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, kernel)
        
        return morph
    
    def recognize(self, plate_image: np.ndarray) -> Dict:
        """
        Recognize text from a license plate image
        
        Args:
            plate_image: Cropped license plate image
        
        Returns:
            Dictionary containing:
                - text: recognized text
                - confidence: recognition confidence
                - is_valid: whether text matches Malaysian format
        """
        # Preprocess
        processed = self.preprocess_plate(plate_image)
        
        # Run OCR
        if self.ocr_engine == 'easyocr':
            results = self.reader.readtext(processed)
            if results:
                # EasyOCR returns list of (bbox, text, confidence)
                text = " ".join([r[1] for r in results])
                confidence = np.mean([r[2] for r in results])
            else:
                text = ""
                confidence = 0.0
        
        elif self.ocr_engine == 'paddleocr':
            results = self.reader.ocr(processed, cls=True)
            if results and results[0]:
                # PaddleOCR returns nested list structure
                text = " ".join([line[1][0] for line in results[0]])
                confidence = np.mean([line[1][1] for line in results[0]])
            else:
                text = ""
                confidence = 0.0
        
        elif self.ocr_engine == 'tesseract':
            text = pytesseract.image_to_string(
                processed,
                config='--psm 7 --oem 3'  # Single line, LSTM mode
            )
            # Tesseract doesn't provide confidence easily, use default
            confidence = 0.8 if text.strip() else 0.0
        
        # Clean and normalize text
        text = self._clean_text(text)
        
        # Validate format (only if enforcement is enabled)
        is_valid = self._validate_format(text) if self.plate_format.get('enforce_validation', True) else True
        
        return {
            'text': text,
            'confidence': float(confidence),
            'is_valid': is_valid
        }
    
    def _clean_text(self, text: str) -> str:
        """
        Clean and normalize OCR output
        
        Args:
            text: Raw OCR text
        
        Returns:
            Cleaned text
        """
        # Remove extra whitespace
        text = " ".join(text.split())
        
        # Convert to uppercase
        text = text.upper()
        
        # Common OCR corrections
        corrections = {
            '0': 'O',  # In letter positions
            'O': '0',  # In number positions
            'I': '1',
            'S': '5',
            'B': '8',
            'Z': '2',
        }
        
        # Remove special characters
        text = re.sub(r'[^A-Z0-9\s]', '', text)
        
        return text.strip()
    
    def _validate_format(self, text: str) -> bool:
        """
        Validate if text matches Malaysian license plate format
        
        Args:
            text: Recognized text
        
        Returns:
            True if valid format, False otherwise
        """
        if not text:
            return False
        
        # Check length
        text_no_space = text.replace(" ", "")
        if len(text_no_space) < self.plate_format['min_length'] or \
           len(text_no_space) > self.plate_format['max_length']:
            return False
        
        # Check against patterns
        for pattern in self.patterns:
            if pattern.match(text):
                logger.debug(f"Valid plate format: {text}")
                return True
        
        logger.debug(f"Invalid plate format: {text}")
        return False
    
    def recognize_batch(self, plate_images: List[np.ndarray]) -> List[Dict]:
        """
        Recognize multiple plates in batch
        
        Args:
            plate_images: List of cropped plate images
        
        Returns:
            List of recognition results
        """
        results = []
        
        for img in plate_images:
            result = self.recognize(img)
            results.append(result)
        
        return results


class MultiEngineRecognizer(LicensePlateRecognizer):
    """
    Uses multiple OCR engines and combines results for higher accuracy
    """
    
    def __init__(self, config_path: str = "config.yaml"):
        super().__init__(config_path)
        
        # Initialize all available engines
        self.engines = []
        
        if EASYOCR_AVAILABLE:
            self.engines.append(('easyocr', easyocr.Reader(['en'], gpu=True)))
        
        if PADDLEOCR_AVAILABLE:
            self.engines.append(('paddleocr', PaddleOCR(use_angle_cls=True, lang='en')))
        
        if TESSERACT_AVAILABLE:
            self.engines.append(('tesseract', None))
        
        logger.info(f"Initialized {len(self.engines)} OCR engines")
    
    def recognize(self, plate_image: np.ndarray) -> Dict:
        """
        Run multiple OCR engines and combine results
        """
        processed = self.preprocess_plate(plate_image)
        
        results = []
        
        for engine_name, engine in self.engines:
            try:
                if engine_name == 'easyocr':
                    ocr_results = engine.readtext(processed)
                    if ocr_results:
                        text = " ".join([r[1] for r in ocr_results])
                        confidence = np.mean([r[2] for r in ocr_results])
                        results.append((text, confidence))
                
                elif engine_name == 'paddleocr':
                    ocr_results = engine.ocr(processed)
                    if ocr_results and ocr_results[0]:
                        text = " ".join([line[1][0] for line in ocr_results[0]])
                        confidence = np.mean([line[1][1] for line in ocr_results[0]])
                        results.append((text, confidence))
                
                elif engine_name == 'tesseract':
                    text = pytesseract.image_to_string(processed, config='--psm 7 --oem 3')
                    if text.strip():
                        results.append((text, 0.7))
            
            except Exception as e:
                logger.warning(f"Engine {engine_name} failed: {e}")
                continue
        
        # Select best result (highest confidence among valid formats)
        best_text = ""
        best_confidence = 0.0
        
        # Check if validation is enforced
        enforce_validation = self.plate_format.get('enforce_validation', True)
        
        for text, conf in results:
            cleaned = self._clean_text(text)
            # If validation is enforced, only accept valid formats
            # If not enforced, accept any result based on confidence
            if enforce_validation:
                if self._validate_format(cleaned) and conf > best_confidence:
                    best_text = cleaned
                    best_confidence = conf
            else:
                if conf > best_confidence:
                    best_text = cleaned
                    best_confidence = conf
        
        # If no valid format found (and validation is enforced), use highest confidence result
        if not best_text and results:
            text, conf = max(results, key=lambda x: x[1])
            best_text = self._clean_text(text)
            best_confidence = conf
        
        return {
            'text': best_text,
            'confidence': float(best_confidence),
            'is_valid': self._validate_format(best_text) if enforce_validation else True
        }


if __name__ == "__main__":
    # Example usage
    recognizer = LicensePlateRecognizer()
    
    test_plate_path = "data/raw/test_plate.jpg"
    if Path(test_plate_path).exists():
        plate_img = cv2.imread(test_plate_path)
        result = recognizer.recognize(plate_img)
        print(f"Recognized: {result['text']}")
        print(f"Confidence: {result['confidence']:.2f}")
        print(f"Valid format: {result['is_valid']}")