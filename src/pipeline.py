"""
End-to-End License Plate Recognition Pipeline
Combines detection and recognition for complete ALPR system
"""

import cv2
import numpy as np
from pathlib import Path
from typing import List, Dict, Optional, Union
import logging
import yaml
import time
from datetime import datetime

from detector import LicensePlateDetector
from recognizer import LicensePlateRecognizer, MultiEngineRecognizer

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class LicensePlateRecognitionPipeline:
    """
    Complete end-to-end ALPR pipeline
    """
    
    def __init__(
        self,
        config_path: str = "config.yaml",
        detection_model_path: Optional[str] = None,
        use_multi_engine: bool = False
    ):
        """
        Initialize the pipeline
        
        Args:
            config_path: Path to configuration file
            detection_model_path: Path to trained detection model
            use_multi_engine: Whether to use multiple OCR engines
        """
        logger.info("Initializing License Plate Recognition Pipeline...")
        
        with open(config_path, 'r') as f:
            self.config = yaml.safe_load(f)
        
        # Initialize detector
        logger.info("Loading detection module...")
        self.detector = LicensePlateDetector(
            config_path=config_path,
            model_path=detection_model_path
        )
        
        # Initialize recognizer
        logger.info("Loading recognition module...")
        if use_multi_engine:
            self.recognizer = MultiEngineRecognizer(config_path=config_path)
        else:
            self.recognizer = LicensePlateRecognizer(config_path=config_path)
        
        logger.info("Pipeline initialization complete!")
    
    def process_image(
        self,
        image: Union[str, np.ndarray],
        save_visualizations: bool = False,
        output_dir: Optional[str] = None
    ) -> List[Dict]:
        """
        Process a single image
        
        Args:
            image: Input image path or numpy array
            save_visualizations: Whether to save visualization images
            output_dir: Directory to save visualizations
        
        Returns:
            List of results, each containing:
                - bbox: bounding box coordinates
                - plate_text: recognized text
                - confidence: overall confidence
                - detection_confidence: detection confidence
                - recognition_confidence: recognition confidence
                - is_valid: whether format is valid
        """
        # Load image if path provided
        if isinstance(image, str):
            image_path = image
            image = cv2.imread(image_path)
            if image is None:
                raise ValueError(f"Could not load image: {image_path}")
        else:
            image_path = None
        
        start_time = time.time()
        
        # Step 1: Detect plates
        logger.info("Detecting license plates...")
        detections = self.detector.detect(image)
        
        if not detections:
            logger.warning("No license plates detected")
            return []
        
        logger.info(f"Found {len(detections)} potential plates")
        
        # Step 2: Recognize text
        results = []
        
        for idx, detection in enumerate(detections):
            plate_crop = detection['plate_crop']
            
            logger.info(f"Recognizing plate {idx + 1}/{len(detections)}...")
            recognition = self.recognizer.recognize(plate_crop)
            
            # Combine results
            result = {
                'bbox': detection['bbox'],
                'plate_text': recognition['text'],
                'confidence': (detection['confidence'] + recognition['confidence']) / 2,
                'detection_confidence': detection['confidence'],
                'recognition_confidence': recognition['confidence'],
                'is_valid': recognition['is_valid']
            }
            
            results.append(result)
            
            logger.info(f"  Text: {recognition['text']}")
            logger.info(f"  Confidence: {recognition['confidence']:.2f}")
            logger.info(f"  Valid format: {recognition['is_valid']}")
        
        processing_time = time.time() - start_time
        logger.info(f"Processing completed in {processing_time:.2f}s")
        
        # Save visualizations if requested
        if save_visualizations and output_dir:
            self._save_visualizations(image, results, output_dir, image_path)
        
        return results
    
    def _visualize_results(self, image: np.ndarray, results: List[Dict]) -> np.ndarray:
        """Draw bounding boxes and text on image"""
        
        for result in results:
            x1, y1, x2, y2 = result['bbox']
            text = result['plate_text']
            confidence = result['confidence']
            is_valid = result['is_valid']
            
            # Color: green if valid, yellow if invalid
            color = (0, 255, 0) if is_valid else (0, 255, 255)
            
            # Draw rectangle
            cv2.rectangle(image, (x1, y1), (x2, y2), color, 2)
            
            # Draw text background
            label = f"{text} ({confidence:.2f})"
            (label_w, label_h), _ = cv2.getTextSize(
                label, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2
            )
            
            cv2.rectangle(
                image,
                (x1, y1 - label_h - 10),
                (x1 + label_w, y1),
                color,
                -1
            )
            
            # Draw text
            cv2.putText(
                image,
                label,
                (x1, y1 - 5),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.6,
                (0, 0, 0),
                2
            )
        
        return image
    
    def _save_visualizations(
        self,
        image: np.ndarray,
        results: List[Dict],
        output_dir: str,
        image_path: Optional[str]
    ):
        """Save visualization images"""
        
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        
        # Generate filename
        if image_path:
            base_name = Path(image_path).stem
        else:
            base_name = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        # Save full image with annotations
        vis_image = self._visualize_results(image.copy(), results)
        output_path = output_dir / f"{base_name}_result.jpg"
        cv2.imwrite(str(output_path), vis_image)
        logger.info(f"Saved visualization: {output_path}")
        
        # Save individual plate crops
        for idx, result in enumerate(results):
            x1, y1, x2, y2 = result['bbox']
            crop = image[y1:y2, x1:x2]
            
            text = result['plate_text'].replace(' ', '_')
            crop_path = output_dir / f"{base_name}_plate_{idx}_{text}.jpg"
            cv2.imwrite(str(crop_path), crop)
    


