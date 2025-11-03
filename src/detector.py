import cv2
import numpy as np
import torch
from ultralytics import YOLO
from pathlib import Path
import yaml
from typing import List, Tuple, Dict, Optional
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class LicensePlateDetector:
    """
    Detects license plates in images using YOLOv8
    Includes false positive filtering for robust detection
    """
    
    def __init__(self, config_path: str = "config.yaml", model_path: Optional[str] = None):
        """
        Initialize the detector
        
        Args:
            config_path: Path to configuration file
            model_path: Path to trained model (optional, will use pretrained if None)
        """
        with open(config_path, 'r') as f:
            self.config = yaml.safe_load(f)
        
        self.detection_config = self.config['detection']
        self.fp_config = self.config['false_positive_filter']
        
        # Initialize model
        if model_path and Path(model_path).exists():
            logger.info(f"Loading trained model from {model_path}")
            self.model = YOLO(model_path)
        else:
            logger.info(f"Initializing {self.detection_config['model_size']} model")
            self.model = YOLO(f"{self.detection_config['model_size']}.pt")
        
        self.confidence_threshold = self.detection_config['confidence_threshold']
        self.iou_threshold = self.detection_config['iou_threshold']
        self.input_size = self.detection_config['input_size']
    
    def detect(self, image: np.ndarray) -> List[Dict]:
        """
        Detect license plates in an image
        
        Args:
            image: Input image as numpy array (BGR format)
        
        Returns:
            List of detections, each containing:
                - bbox: [x1, y1, x2, y2]
                - confidence: detection confidence
                - plate_crop: cropped plate image
        """
        # Run detection
        results = self.model.predict(
            image,
            conf=self.confidence_threshold,
            iou=self.iou_threshold,
            imgsz=self.input_size,
            verbose=False
        )
        
        detections = []
        
        for result in results:
            boxes = result.boxes
            
            for box in boxes:
                # Extract bounding box coordinates
                x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
                confidence = float(box.conf[0])
                
                # Convert to integers
                x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
                
                # Apply false positive filtering
                if self.fp_config['enabled']:
                    if not self._validate_detection(image, (x1, y1, x2, y2)):
                        logger.debug(f"Filtered false positive at ({x1}, {y1}, {x2}, {y2})")
                        continue
                
                # Crop plate region
                plate_crop = image[y1:y2, x1:x2]
                
                detection = {
                    'bbox': [x1, y1, x2, y2],
                    'confidence': confidence,
                    'plate_crop': plate_crop
                }
                
                detections.append(detection)
        
        logger.info(f"Detected {len(detections)} license plates")
        return detections
    
    def _validate_detection(self, image: np.ndarray, bbox: Tuple[int, int, int, int]) -> bool:
        """
        Validate detection to filter false positives
        
        Args:
            image: Full image
            bbox: Bounding box (x1, y1, x2, y2)
        
        Returns:
            True if valid plate, False if false positive
        """
        x1, y1, x2, y2 = bbox
        width = x2 - x1
        height = y2 - y1
        
        if width <= 0 or height <= 0:
            return False
        
        # Check aspect ratio
        aspect_ratio = width / height
        if aspect_ratio < self.fp_config['min_aspect_ratio'] or \
           aspect_ratio > self.fp_config['max_aspect_ratio']:
            logger.debug(f"Invalid aspect ratio: {aspect_ratio:.2f}")
            return False
        
        # Check size relative to image
        image_area = image.shape[0] * image.shape[1]
        bbox_area = width * height
        relative_area = bbox_area / image_area
        
        if relative_area < self.fp_config['min_plate_area'] or \
           relative_area > self.fp_config['max_plate_area']:
            logger.debug(f"Invalid relative area: {relative_area:.4f}")
            return False
        
        # Check if region has reasonable contrast (likely has text)
        plate_region = image[y1:y2, x1:x2]
        gray = cv2.cvtColor(plate_region, cv2.COLOR_BGR2GRAY)
        
        # Calculate standard deviation (low std = uniform, likely not a plate)
        std_dev = np.std(gray)
        if std_dev < 20:  # Threshold for minimum contrast
            logger.debug(f"Low contrast region: {std_dev:.2f}")
            return False
        
        return True
    
    def train(self, data_yaml: str, epochs: Optional[int] = None):
        """
        Train the detection model
        
        Args:
            data_yaml: Path to dataset YAML file
            epochs: Number of training epochs (uses config if None)
        """
        epochs = epochs or self.detection_config['epochs']
        
        logger.info(f"Training detector for {epochs} epochs...")
        
        results = self.model.train(
            data=data_yaml,
            epochs=epochs,
            imgsz=self.input_size,
            batch=self.detection_config['batch_size'],
            lr0=self.detection_config['learning_rate'],
            patience=self.detection_config['patience'],
            augment=self.detection_config['augment'],
            project="models/detection_training",
            name="plate_detector"
        )
        
        logger.info("Training completed!")
        return results
    
    def visualize_detections(self, image: np.ndarray, detections: List[Dict]) -> np.ndarray:
        """
        Draw bounding boxes on image
        
        Args:
            image: Input image
            detections: List of detections from detect()
        
        Returns:
            Image with drawn bounding boxes
        """
        vis_image = image.copy()
        
        for det in detections:
            x1, y1, x2, y2 = det['bbox']
            conf = det['confidence']
            
            # Draw rectangle
            cv2.rectangle(vis_image, (x1, y1), (x2, y2), (0, 255, 0), 2)
            
            # Add label
            label = f"Plate: {conf:.2f}"
            cv2.putText(vis_image, label, (x1, y1 - 10),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
        
        return vis_image


if __name__ == "__main__":
    # Example usage
    detector = LicensePlateDetector()
    
    # Test on sample image
    test_image_path = "data/raw/test_image.jpg"
    if Path(test_image_path).exists():
        image = cv2.imread(test_image_path)
        detections = detector.detect(image)
        
        # Visualize
        result_image = detector.visualize_detections(image, detections)
        cv2.imwrite("outputs/detection_result.jpg", result_image)
        print(f"Detected {len(detections)} plates")
