import xml.etree.ElementTree as ET
import cv2
import numpy as np
from pathlib import Path
import shutil
import logging
from typing import Tuple, List, Dict
import random
from tqdm import tqdm

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class DatasetIntegrator:
    """
    Integrates additional datasets into existing YOLO format structure
    """
    
    def __init__(
        self,
        additional_data_dir: str = "data/additional_data",
        existing_data_dir: str = "data/raw",
        output_dir: str = "data/integrated"
    ):
        """
        Initialize the integrator
        
        Args:
            additional_data_dir: Directory containing new data (images and annotations subfolders)
            existing_data_dir: Directory with existing YOLO data (train/val/test with images/ and labels/ subfolders)
            output_dir: Output directory for integrated dataset
        """
        self.additional_data_dir = Path(additional_data_dir)
        self.existing_data_dir = Path(existing_data_dir)
        self.output_dir = Path(output_dir)
        
        # Create output directory structure
        for split in ['train', 'val', 'test']:
            (self.output_dir / split / 'images').mkdir(parents=True, exist_ok=True)
            (self.output_dir / split / 'labels').mkdir(parents=True, exist_ok=True)
    
    def parse_voc_xml(self, xml_path: Path) -> List[Dict]:
        """
        Parse Pascal VOC XML annotation file
        
        Args:
            xml_path: Path to XML file
        
        Returns:
            List of bounding boxes with format:
                - class_name: object class
                - xmin, ymin, xmax, ymax: pixel coordinates
        """
        try:
            tree = ET.parse(xml_path)
            root = tree.getroot()
            
            # Get image dimensions
            size = root.find('size')
            width = int(size.find('width').text)
            height = int(size.find('height').text)
            
            # Get all objects
            objects = []
            for obj in root.findall('object'):
                class_name = obj.find('name').text.lower()
                
                # Get bounding box
                bbox = obj.find('bndbox')
                xmin = int(bbox.find('xmin').text)
                ymin = int(bbox.find('ymin').text)
                xmax = int(bbox.find('xmax').text)
                ymax = int(bbox.find('ymax').text)
                
                objects.append({
                    'class_name': class_name,
                    'xmin': xmin,
                    'ymin': ymin,
                    'xmax': xmax,
                    'ymax': ymax,
                    'width': width,
                    'height': height
                })
            
            return objects
        
        except Exception as e:
            logger.error(f"Error parsing {xml_path}: {e}")
            return []
    
    def voc_to_yolo(
        self,
        bbox: Dict,
        class_mapping: Dict[str, int] = None
    ) -> Tuple[int, float, float, float, float]:
        """
        Convert VOC bounding box to YOLO format
        
        Args:
            bbox: Bounding box dict from parse_voc_xml
            class_mapping: Mapping from class names to class IDs
        
        Returns:
            (class_id, x_center, y_center, width, height) in YOLO format (normalized)
        """
        if class_mapping is None:
            class_mapping = {
                'licence': 0,
                'license': 0,
                'plate': 0,
                'license_plate': 0,
                'licence_plate': 0,
                'car_plate': 0,
                'numberplate': 0,
                'number_plate': 0
            }
        
        # Get class ID
        class_name = bbox['class_name'].lower().replace('-', '_').replace(' ', '_')
        class_id = class_mapping.get(class_name, 0)  # Default to 0 (license_plate)
        
        # Convert to YOLO format (normalized)
        img_width = bbox['width']
        img_height = bbox['height']
        
        # Calculate center and dimensions
        x_center = ((bbox['xmin'] + bbox['xmax']) / 2) / img_width
        y_center = ((bbox['ymin'] + bbox['ymax']) / 2) / img_height
        width = (bbox['xmax'] - bbox['xmin']) / img_width
        height = (bbox['ymax'] - bbox['ymin']) / img_height
        
        # Ensure values are in valid range [0, 1]
        x_center = max(0, min(1, x_center))
        y_center = max(0, min(1, y_center))
        width = max(0, min(1, width))
        height = max(0, min(1, height))
        
        return class_id, x_center, y_center, width, height
    
    def convert_voc_dataset(
        self,
        images_dir: str = "images",
        annotations_dir: str = "annotations",
        output_images_dir: Path = None,
        output_labels_dir: Path = None
    ) -> Tuple[int, int]:
        """
        Convert Pascal VOC dataset to YOLO format
        
        Args:
            images_dir: Subdirectory name for images in additional_data_dir
            annotations_dir: Subdirectory name for annotations in additional_data_dir
            output_images_dir: Output directory for images
            output_labels_dir: Output directory for labels
        
        Returns:
            (successful_conversions, skipped_files)
        """
        logger.info("Converting Pascal VOC dataset to YOLO format...")
        
        images_path = self.additional_data_dir / images_dir
        annotations_path = self.additional_data_dir / annotations_dir
        
        if not images_path.exists():
            logger.error(f"Images directory not found: {images_path}")
            return 0, 0
        
        if not annotations_path.exists():
            logger.error(f"Annotations directory not found: {annotations_path}")
            return 0, 0
        
        # Get all XML files
        xml_files = list(annotations_path.glob("*.xml"))
        logger.info(f"Found {len(xml_files)} annotation files")
        
        successful = 0
        skipped = 0
        
        # Create temporary directories if not provided
        if output_images_dir is None:
            output_images_dir = self.additional_data_dir / "converted" / "images"
            output_labels_dir = self.additional_data_dir / "converted" / "labels"
        
        # Ensure output directories exist
        output_images_dir.mkdir(parents=True, exist_ok=True)
        output_labels_dir.mkdir(parents=True, exist_ok=True)
        
        for xml_file in tqdm(xml_files, desc="Converting annotations"):
            # Parse XML
            objects = self.parse_voc_xml(xml_file)
            
            if not objects:
                skipped += 1
                continue
            
            # Find corresponding image
            base_name = xml_file.stem
            image_path = None
            
            # Try different image extensions
            for ext in ['.png', '.jpg', '.jpeg', '.PNG', '.JPG', '.JPEG']:
                candidate = images_path / f"{base_name}{ext}"
                if candidate.exists():
                    image_path = candidate
                    break
            
            if image_path is None:
                logger.warning(f"No image found for {xml_file.name}")
                skipped += 1
                continue
            
            # Verify image can be read
            img = cv2.imread(str(image_path))
            if img is None:
                logger.warning(f"Cannot read image: {image_path}")
                skipped += 1
                continue
            
            # Convert annotations to YOLO format
            yolo_annotations = []
            for obj in objects:
                class_id, x_center, y_center, width, height = self.voc_to_yolo(obj)
                yolo_annotations.append(f"{class_id} {x_center:.6f} {y_center:.6f} {width:.6f} {height:.6f}")
            
            # Skip if no valid annotations
            if not yolo_annotations:
                logger.warning(f"No valid annotations for {xml_file.name}")
                skipped += 1
                continue
            
            # Ensure output directories exist (in case called with custom dirs)
            output_labels_dir.mkdir(parents=True, exist_ok=True)
            output_images_dir.mkdir(parents=True, exist_ok=True)
            
            # Save label file
            label_file = output_labels_dir / f"{base_name}.txt"
            with open(label_file, 'w') as f:
                f.write('\n'.join(yolo_annotations))
            
            # Copy image
            output_image = output_images_dir / f"{base_name}{image_path.suffix}"
            shutil.copy(image_path, output_image)
            
            successful += 1
        
        logger.info(f"Conversion complete: {successful} successful, {skipped} skipped")
        return successful, skipped
    
    def split_dataset(
        self,
        images_dir: Path,
        labels_dir: Path,
        train_ratio: float = 0.7,
        val_ratio: float = 0.2,
        test_ratio: float = 0.1,
        seed: int = 42
    ) -> Dict[str, List[str]]:
        """
        Split dataset into train/val/test
        
        Args:
            images_dir: Directory containing images
            labels_dir: Directory containing labels
            train_ratio: Proportion for training set
            val_ratio: Proportion for validation set
            test_ratio: Proportion for test set
            seed: Random seed for reproducibility
        
        Returns:
            Dictionary with lists of image names for each split
        """
        # Verify ratios sum to 1
        assert abs(train_ratio + val_ratio + test_ratio - 1.0) < 0.001, "Ratios must sum to 1"
        
        # Get all image files
        image_files = []
        for ext in ['*.png', '*.jpg', '*.jpeg', '*.PNG', '*.JPG', '*.JPEG']:
            image_files.extend(images_dir.glob(ext))
        
        # Filter to only keep images that have corresponding labels
        valid_images = []
        for img_path in image_files:
            label_path = labels_dir / f"{img_path.stem}.txt"
            if label_path.exists():
                valid_images.append(img_path.name)
        
        logger.info(f"Found {len(valid_images)} valid image-label pairs")
        
        # Shuffle
        random.seed(seed)
        random.shuffle(valid_images)
        
        # Calculate split indices
        n_total = len(valid_images)
        n_train = int(n_total * train_ratio)
        n_val = int(n_total * val_ratio)
        
        # Split
        train_files = valid_images[:n_train]
        val_files = valid_images[n_train:n_train + n_val]
        test_files = valid_images[n_train + n_val:]
        
        splits = {
            'train': train_files,
            'val': val_files,
            'test': test_files
        }
        
        logger.info(f"Split: Train={len(train_files)}, Val={len(val_files)}, Test={len(test_files)}")
        
        return splits
    
    def copy_existing_data(self):
        """Copy existing processed data to output directory"""
        logger.info("Copying existing dataset...")
        
        for split in ['train', 'val', 'test']:
            src_images = self.existing_data_dir / split / 'images'
            src_labels = self.existing_data_dir / split / 'labels'
            
            dst_images = self.output_dir / split / 'images'
            dst_labels = self.output_dir / split / 'labels'
            
            # Ensure destination directories exist
            dst_images.mkdir(parents=True, exist_ok=True)
            dst_labels.mkdir(parents=True, exist_ok=True)
            
            # Copy images
            if src_images.exists():
                for img_file in src_images.glob("*"):
                    if img_file.suffix.lower() in ['.jpg', '.png', '.jpeg']:
                        shutil.copy(img_file, dst_images / img_file.name)
            
            # Copy labels
            if src_labels.exists():
                for label_file in src_labels.glob("*.txt"):
                    shutil.copy(label_file, dst_labels / label_file.name)
        
        # Count files
        for split in ['train', 'val', 'test']:
            n_images = len(list((self.output_dir / split / 'images').glob("*")))
            n_labels = len(list((self.output_dir / split / 'labels').glob("*.txt")))
            logger.info(f"  {split}: {n_images} images, {n_labels} labels (existing data)")
    
    def integrate_datasets(
        self,
        train_ratio: float = 0.7,
        val_ratio: float = 0.2,
        test_ratio: float = 0.1
    ):
        """
        Complete pipeline to integrate additional dataset
        
        Args:
            train_ratio: Proportion for training set
            val_ratio: Proportion for validation set
            test_ratio: Proportion for test set
        """
        logger.info("="*60)
        logger.info("INTEGRATING ADDITIONAL DATASET")
        logger.info("="*60)
        
        # Ensure all output directories exist upfront
        logger.info("\nCreating output directory structure...")
        for split in ['train', 'val', 'test']:
            (self.output_dir / split / 'images').mkdir(parents=True, exist_ok=True)
            (self.output_dir / split / 'labels').mkdir(parents=True, exist_ok=True)
        logger.info(f"✓ Output directory created: {self.output_dir}")
        
        # Step 1: Convert VOC format to YOLO format
        converted_images = self.additional_data_dir / "converted" / "images"
        converted_labels = self.additional_data_dir / "converted" / "labels"
        
        # Ensure converted directories exist
        converted_images.mkdir(parents=True, exist_ok=True)
        converted_labels.mkdir(parents=True, exist_ok=True)
        
        successful, skipped = self.convert_voc_dataset(
            output_images_dir=converted_images,
            output_labels_dir=converted_labels
        )
        
        if successful == 0:
            logger.error("No files were successfully converted. Aborting.")
            return
        
        # Step 2: Split additional data
        logger.info("\nSplitting additional dataset...")
        splits = self.split_dataset(
            converted_images,
            converted_labels,
            train_ratio,
            val_ratio,
            test_ratio
        )
        
        # Step 3: Copy existing data
        logger.info("\nCopying existing dataset...")
        self.copy_existing_data()
        
        # Step 4: Copy new data to appropriate splits
        logger.info("\nIntegrating additional data into splits...")
        
        for split_name, file_list in splits.items():
            logger.info(f"  Adding {len(file_list)} files to {split_name}...")
            
            dst_images = self.output_dir / split_name / 'images'
            dst_labels = self.output_dir / split_name / 'labels'
            
            for filename in file_list:
                # Copy image
                img_path = converted_images / filename
                if img_path.exists():
                    # Rename to avoid conflicts
                    new_name = f"kaggle_{img_path.stem}{img_path.suffix}"
                    shutil.copy(img_path, dst_images / new_name)
                
                # Copy label
                label_path = converted_labels / f"{Path(filename).stem}.txt"
                if label_path.exists():
                    new_name = f"kaggle_{Path(filename).stem}.txt"
                    shutil.copy(label_path, dst_labels / new_name)
        
        # Step 5: Create data.yaml
        self.create_data_yaml()
        
        # Step 6: Print statistics
        logger.info("\n" + "="*60)
        logger.info("INTEGRATION COMPLETE")
        logger.info("="*60)
        
        for split in ['train', 'val', 'test']:
            n_images = len(list((self.output_dir / split / 'images').glob("*")))
            n_labels = len(list((self.output_dir / split / 'labels').glob("*.txt")))
            logger.info(f"{split.capitalize()}: {n_images} images, {n_labels} labels")
        
        logger.info(f"\nIntegrated dataset ready at: {self.output_dir}")
        logger.info(f"Use data.yaml at: {self.output_dir / 'data.yaml'}")
    
    def create_data_yaml(self):
        """Create data.yaml for the integrated dataset"""
        import yaml
        
        data_yaml = {
            'path': str(self.output_dir.absolute()),
            'train': 'train/images',
            'val': 'val/images',
            'test': 'test/images',
            'names': {0: 'license_plate'},
            'nc': 1
        }
        
        yaml_path = self.output_dir / 'data.yaml'
        with open(yaml_path, 'w') as f:
            yaml.dump(data_yaml, f, default_flow_style=False)
        
        logger.info(f"Created data.yaml at: {yaml_path}")
    
    def validate_integration(self):
        """Validate the integrated dataset"""
        logger.info("\n" + "="*60)
        logger.info("VALIDATING INTEGRATED DATASET")
        logger.info("="*60)
        
        issues = []
        
        for split in ['train', 'val', 'test']:
            images_dir = self.output_dir / split / 'images'
            labels_dir = self.output_dir / split / 'labels'
            
            images = list(images_dir.glob("*"))
            labels = list(labels_dir.glob("*.txt"))
            
            logger.info(f"\n{split.capitalize()}:")
            logger.info(f"  Images: {len(images)}")
            logger.info(f"  Labels: {len(labels)}")
            
            # Check for missing labels
            missing_labels = 0
            for img_path in images:
                label_path = labels_dir / f"{img_path.stem}.txt"
                if not label_path.exists():
                    missing_labels += 1
            
            if missing_labels > 0:
                issues.append(f"{split}: {missing_labels} images without labels")
                logger.warning(f"  Missing labels: {missing_labels}")
            
            # Sample check: validate some label files
            sample_size = min(10, len(labels))
            invalid_labels = 0
            
            for label_path in random.sample(labels, sample_size):
                with open(label_path, 'r') as f:
                    lines = f.readlines()
                    for line in lines:
                        parts = line.strip().split()
                        if len(parts) != 5:
                            invalid_labels += 1
                            break
            
            if invalid_labels > 0:
                issues.append(f"{split}: Found {invalid_labels}/{sample_size} invalid label files")
                logger.warning(f"  Invalid labels found: {invalid_labels}/{sample_size}")
        
        if issues:
            logger.warning("\n⚠️  Issues found:")
            for issue in issues:
                logger.warning(f"  - {issue}")
        else:
            logger.info("\n✓ No issues found. Dataset is ready for training!")


def main():
    """Main function"""
    integrator = DatasetIntegrator(
        additional_data_dir="data/additional_data",
        existing_data_dir="data/raw",
        output_dir="data/integrated"
    )
    
    # Integrate datasets
    integrator.integrate_datasets(
        train_ratio=0.7,
        val_ratio=0.2,
        test_ratio=0.1
    )
    
    # Validate
    integrator.validate_integration()


if __name__ == "__main__":
    main()