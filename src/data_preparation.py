import cv2
import numpy as np
import albumentations as A
from pathlib import Path
import json
import yaml
from typing import List, Tuple, Dict
import logging
from tqdm import tqdm
import shutil

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class DatasetPreparer:
    """
    Prepares and augments license plate dataset
    Works with pre-split YOLO format data (train/val/test)
    Supports both standard YOLO bbox and polygon/OBB formats
    """
    
    def __init__(self, config_path: str = "config.yaml"):
        """Initialize dataset preparer"""
        with open(config_path, 'r') as f:
            self.config = yaml.safe_load(f)
        
        self.augmentation_config = self.config['augmentation']
        self.data_config = self.config['data']
        
        # Setup augmentation pipeline
        self.transform = self._create_augmentation_pipeline()
    
    def _polygon_to_bbox(self, polygon_coords: List[float]) -> Tuple[float, float, float, float]:
        """
        Convert polygon coordinates to YOLO bounding box format
        
        Args:
            polygon_coords: List of coordinates [x1, y1, x2, y2, x3, y3, x4, y4]
            
        Returns:
            (x_center, y_center, width, height) in normalized YOLO format
        """
        # Reshape to pairs of (x, y)
        points = np.array(polygon_coords).reshape(-1, 2)
        
        # Get bounding box
        x_min = points[:, 0].min()
        x_max = points[:, 0].max()
        y_min = points[:, 1].min()
        y_max = points[:, 1].max()
        
        # Convert to YOLO format (center x, center y, width, height)
        x_center = (x_min + x_max) / 2
        y_center = (y_min + y_max) / 2
        width = x_max - x_min
        height = y_max - y_min
        
        return x_center, y_center, width, height
    
    def _parse_label_line(self, line: str) -> Tuple[int, List[float]]:
        """
        Parse a label line and detect format (standard bbox or polygon/OBB)
        
        Returns:
            (class_id, bbox_coords) where bbox_coords is [x_center, y_center, width, height]
        """
        parts = line.strip().split()
        
        if len(parts) < 5:
            raise ValueError(f"Invalid annotation format: {line}")
        
        class_id = int(parts[0])
        coords = [float(p) for p in parts[1:]]
        
        # Standard YOLO bbox: class_id x_center y_center width height
        if len(coords) == 4:
            return class_id, coords
        
        # Polygon/OBB format: class_id x1 y1 x2 y2 x3 y3 x4 y4 (or more points)
        elif len(coords) >= 8 and len(coords) % 2 == 0:
            logger.debug(f"Converting polygon with {len(coords)//2} points to bbox")
            bbox = self._polygon_to_bbox(coords)
            return class_id, list(bbox)
        
        else:
            raise ValueError(f"Unknown annotation format with {len(coords)} coordinates")
    
    def _create_augmentation_pipeline(self) -> A.Compose:
        """
        Create albumentations augmentation pipeline
        """
        transforms = []
        
        if not self.augmentation_config['enabled']:
            return A.Compose(transforms, bbox_params=A.BboxParams(
                format='yolo',
                label_fields=['class_labels']
            ))
        
        for technique in self.augmentation_config['techniques']:
            name = technique['name']
            prob = technique['prob']
            
            if name == 'horizontal_flip':
                transforms.append(A.HorizontalFlip(p=prob))
            
            elif name == 'rotation':
                limit = technique.get('limit', 15)
                transforms.append(A.Rotate(limit=limit, p=prob, border_mode=cv2.BORDER_CONSTANT))
            
            elif name == 'brightness_contrast':
                brightness_limit = technique.get('brightness_limit', 0.2)
                contrast_limit = technique.get('contrast_limit', 0.2)
                transforms.append(A.RandomBrightnessContrast(
                    brightness_limit=brightness_limit,
                    contrast_limit=contrast_limit,
                    p=prob
                ))
            
            elif name == 'blur':
                blur_limit = technique.get('blur_limit', 5)
                transforms.append(A.OneOf([
                    A.MotionBlur(blur_limit=blur_limit),
                    A.GaussianBlur(blur_limit=blur_limit),
                    A.MedianBlur(blur_limit=blur_limit),
                ], p=prob))
            
            elif name == 'noise':
                transforms.append(A.OneOf([
                    A.GaussNoise(),
                    A.ISONoise(),
                ], p=prob))
            
            elif name == 'weather':
                transforms.append(A.OneOf([
                    A.RandomRain(drop_length=20, blur_value=3),
                    A.RandomFog(fog_coef_lower=0.1, fog_coef_upper=0.3),
                    A.RandomSunFlare(
                        flare_roi=(0, 0, 1, 0.5),
                        angle_lower=0,
                        angle_upper=1
                    ),
                ], p=prob))
            
            elif name == 'perspective':
                scale = technique.get('scale', 0.1)
                transforms.append(A.Perspective(scale=scale, p=prob))
        
        # Add bbox parameters for YOLO format detection
        return A.Compose(
            transforms,
            bbox_params=A.BboxParams(
                format='yolo',
                label_fields=['class_labels'],
                min_visibility=0.3
            )
        )
    
    def augment_yolo_split(
        self,
        input_base_dir: str,
        output_base_dir: str,
        num_augmentations: int = 3,
        splits: List[str] = ['train']
    ):
        """
        Augment pre-split YOLO format dataset
        
        Args:
            input_base_dir: Base directory with train/val/test folders
            output_base_dir: Output directory for augmented data
            num_augmentations: Number of augmented versions per image
            splits: Which splits to augment (default: only 'train')
        """
        logger.info("="*60)
        logger.info("AUGMENTING PRE-SPLIT YOLO DATASET")
        logger.info("="*60)
        
        input_base = Path(input_base_dir)
        output_base = Path(output_base_dir)
        
        stats = {}
        conversion_stats = {'polygon_converted': 0, 'standard_bbox': 0}
        
        for split in splits:
            logger.info(f"\nProcessing {split} split...")
            
            input_images_dir = input_base / split / 'images'
            input_labels_dir = input_base / split / 'labels'
            
            output_images_dir = output_base / split / 'images'
            output_labels_dir = output_base / split / 'labels'
            
            output_images_dir.mkdir(parents=True, exist_ok=True)
            output_labels_dir.mkdir(parents=True, exist_ok=True)
            
            if not input_images_dir.exists():
                logger.warning(f"Images directory not found: {input_images_dir}")
                continue
            
            if not input_labels_dir.exists():
                logger.warning(f"Labels directory not found: {input_labels_dir}")
                continue
            
            # Get all images
            image_files = sorted(list(input_images_dir.glob("*.jpg")) + 
                               list(input_images_dir.glob("*.png")) +
                               list(input_images_dir.glob("*.jpeg")))
            
            logger.info(f"Found {len(image_files)} images in {split}")
            
            original_count = 0
            augmented_count = 0
            skipped_count = 0
            
            for img_path in tqdm(image_files, desc=f"Augmenting {split}"):
                # Read image
                image = cv2.imread(str(img_path))
                if image is None:
                    logger.warning(f"Could not read {img_path}")
                    skipped_count += 1
                    continue
                
                image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
                height, width = image.shape[:2]
                
                # Read corresponding label file
                label_path = input_labels_dir / f"{img_path.stem}.txt"
                
                if not label_path.exists():
                    logger.warning(f"No label file for {img_path.name}")
                    skipped_count += 1
                    continue
                
                # Parse YOLO labels (handles both standard and polygon formats)
                bboxes = []
                class_labels = []
                
                try:
                    with open(label_path, 'r') as f:
                        for line in f:
                            line = line.strip()
                            if not line:
                                continue
                            
                            try:
                                class_id, bbox = self._parse_label_line(line)
                                bboxes.append(bbox)
                                class_labels.append(class_id)
                                
                                # Track conversion statistics
                                if len(line.split()) > 5:
                                    conversion_stats['polygon_converted'] += 1
                                else:
                                    conversion_stats['standard_bbox'] += 1
                                    
                            except ValueError as e:
                                logger.warning(f"Skipping invalid annotation in {label_path}: {e}")
                                continue
                    
                    if not bboxes:
                        logger.warning(f"No valid bboxes in {label_path}")
                        skipped_count += 1
                        continue
                    
                    # Save original image and labels (in standard YOLO format)
                    orig_img_path = output_images_dir / img_path.name
                    orig_label_path = output_labels_dir / f"{img_path.stem}.txt"
                    
                    cv2.imwrite(str(orig_img_path), cv2.cvtColor(image, cv2.COLOR_RGB2BGR))
                    
                    # Write in standard YOLO bbox format
                    with open(orig_label_path, 'w') as f:
                        for bbox, label in zip(bboxes, class_labels):
                            x_center, y_center, bbox_width, bbox_height = bbox
                            f.write(f"{label} {x_center:.6f} {y_center:.6f} {bbox_width:.6f} {bbox_height:.6f}\n")
                    
                    original_count += 1
                    
                    # Generate augmented versions
                    for aug_idx in range(num_augmentations):
                        try:
                            # Apply augmentation
                            transformed = self.transform(
                                image=image,
                                bboxes=bboxes,
                                class_labels=class_labels
                            )
                            
                            aug_image = transformed['image']
                            aug_bboxes = transformed['bboxes']
                            aug_labels = transformed['class_labels']
                            
                            if not aug_bboxes:
                                continue
                            
                            # Save augmented image
                            aug_img_name = f"{img_path.stem}_aug{aug_idx}{img_path.suffix}"
                            aug_img_path = output_images_dir / aug_img_name
                            cv2.imwrite(str(aug_img_path), cv2.cvtColor(aug_image, cv2.COLOR_RGB2BGR))
                            
                            # Save augmented labels in YOLO format
                            aug_label_path = output_labels_dir / f"{img_path.stem}_aug{aug_idx}.txt"
                            with open(aug_label_path, 'w') as f:
                                for bbox, label in zip(aug_bboxes, aug_labels):
                                    x_center, y_center, bbox_width, bbox_height = bbox
                                    f.write(f"{label} {x_center:.6f} {y_center:.6f} {bbox_width:.6f} {bbox_height:.6f}\n")
                            
                            augmented_count += 1
                        
                        except Exception as e:
                            logger.warning(f"Augmentation failed for {img_path.name} (aug {aug_idx}): {e}")
                            continue
                
                except Exception as e:
                    logger.error(f"Error processing {img_path.name}: {e}")
                    skipped_count += 1
                    continue
            
            stats[split] = {
                'original': original_count,
                'augmented': augmented_count,
                'skipped': skipped_count,
                'total': original_count + augmented_count
            }
            
            logger.info(f"{split} - Original: {original_count}, Augmented: {augmented_count}, Skipped: {skipped_count}")
        
        logger.info("\n" + "="*60)
        logger.info("AUGMENTATION COMPLETE")
        logger.info("="*60)
        logger.info(f"Annotation Format Statistics:")
        logger.info(f"  - Polygon/OBB converted to bbox: {conversion_stats['polygon_converted']}")
        logger.info(f"  - Standard bounding boxes: {conversion_stats['standard_bbox']}")
        logger.info("")
        
        for split, stat in stats.items():
            logger.info(f"{split}: {stat['total']} images ({stat['original']} original + {stat['augmented']} augmented, {stat['skipped']} skipped)")
        
        return stats
    
    def copy_non_augmented_splits(
        self,
        input_base_dir: str,
        output_base_dir: str,
        splits: List[str] = ['val', 'test']
    ):
        """
        Copy validation and test splits without augmentation
        Also converts polygon/OBB annotations to standard bbox format
        
        Args:
            input_base_dir: Base directory with train/val/test folders
            output_base_dir: Output directory
            splits: Which splits to copy (default: 'val' and 'test')
        """
        logger.info(f"\nCopying non-augmented splits: {splits}")
        
        input_base = Path(input_base_dir)
        output_base = Path(output_base_dir)
        
        for split in splits:
            input_split_dir = input_base / split
            output_split_dir = output_base / split
            
            if not input_split_dir.exists():
                logger.warning(f"Split directory not found: {input_split_dir}")
                continue
            
            # Setup directories
            input_images = input_split_dir / 'images'
            input_labels = input_split_dir / 'labels'
            output_images = output_split_dir / 'images'
            output_labels = output_split_dir / 'labels'
            
            output_images.mkdir(parents=True, exist_ok=True)
            output_labels.mkdir(parents=True, exist_ok=True)
            
            # Copy and convert labels
            if input_labels.exists():
                for label_file in input_labels.iterdir():
                    if label_file.suffix == '.txt':
                        try:
                            # Read and convert annotations
                            bboxes = []
                            class_labels = []
                            
                            with open(label_file, 'r') as f:
                                for line in f:
                                    line = line.strip()
                                    if not line:
                                        continue
                                    try:
                                        class_id, bbox = self._parse_label_line(line)
                                        bboxes.append(bbox)
                                        class_labels.append(class_id)
                                    except ValueError:
                                        continue
                            
                            # Write in standard YOLO format
                            output_label_file = output_labels / label_file.name
                            with open(output_label_file, 'w') as f:
                                for bbox, label in zip(bboxes, class_labels):
                                    x_center, y_center, bbox_width, bbox_height = bbox
                                    f.write(f"{label} {x_center:.6f} {y_center:.6f} {bbox_width:.6f} {bbox_height:.6f}\n")
                        
                        except Exception as e:
                            logger.warning(f"Error converting {label_file.name}: {e}")
                            continue
            
            # Copy images
            if input_images.exists():
                for img_file in input_images.iterdir():
                    if img_file.suffix.lower() in ['.jpg', '.png', '.jpeg']:
                        shutil.copy(img_file, output_images / img_file.name)
            
            # Count files
            img_count = len(list(output_images.glob("*")))
            label_count = len(list(output_labels.glob("*.txt")))
            logger.info(f"{split}: Copied {img_count} images and {label_count} labels")
    
    def create_data_yaml(
        self,
        output_dir: str,
        class_names: List[str] = None
    ):
        """
        Create data.yaml file for YOLO training
        
        Args:
            output_dir: Directory containing train/val/test splits
            class_names: List of class names (default: ['license_plate'])
        """
        if class_names is None:
            class_names = ['license_plate']
        
        output_dir = Path(output_dir)
        
        data_yaml = {
            'path': str(output_dir.absolute()),
            'train': 'train/images',
            'val': 'val/images',
            'test': 'test/images',
            'names': {i: name for i, name in enumerate(class_names)},
            'nc': len(class_names)
        }
        
        yaml_path = output_dir / 'data.yaml'
        with open(yaml_path, 'w') as f:
            yaml.dump(data_yaml, f, default_flow_style=False)
        
        logger.info(f"Created data.yaml at: {yaml_path}")
        return str(yaml_path)
    
    def prepare_dataset_from_presplit(
        self,
        input_dir: str,
        output_dir: str,
        num_augmentations: int = 5,
        augment_splits: List[str] = ['train']
    ):
        """
        Complete pipeline for pre-split YOLO dataset
        
        Args:
            input_dir: Input directory with train/val/test structure
            output_dir: Output directory for processed dataset
            num_augmentations: Number of augmentations per training image
            augment_splits: Which splits to augment (default: only 'train')
        """
        logger.info("="*60)
        logger.info("PREPARING DATASET FROM PRE-SPLIT YOLO FORMAT")
        logger.info("="*60)
        logger.info(f"Input: {input_dir}")
        logger.info(f"Output: {output_dir}")
        logger.info(f"Augmentations per image: {num_augmentations}")
        logger.info(f"Augmenting splits: {augment_splits}")
        
        # Step 1: Augment training data
        if augment_splits:
            self.augment_yolo_split(
                input_base_dir=input_dir,
                output_base_dir=output_dir,
                num_augmentations=num_augmentations,
                splits=augment_splits
            )
        
        # Step 2: Copy non-augmented splits
        all_splits = ['train', 'val', 'test']
        non_augmented = [s for s in all_splits if s not in augment_splits]
        
        if non_augmented:
            self.copy_non_augmented_splits(
                input_base_dir=input_dir,
                output_base_dir=output_dir,
                splits=non_augmented
            )
        
        # Step 3: Create data.yaml
        yaml_path = self.create_data_yaml(output_dir)
        
        logger.info("\n" + "="*60)
        logger.info("DATASET PREPARATION COMPLETE!")
        logger.info("="*60)
        logger.info(f"Dataset ready at: {output_dir}")
        logger.info(f"Use this data.yaml for training: {yaml_path}")
        
        return yaml_path


if __name__ == "__main__":
    preparer = DatasetPreparer()
    
    # Prepare dataset from pre-split YOLO format
    yaml_path = preparer.prepare_dataset_from_presplit(
        input_dir="data/raw",
        output_dir="data/processed",
        num_augmentations=5,
        augment_splits=['train']  # Only augment training data
    )
    
    print(f"\nDataset ready! Use this command to train:")
    print(f"python train.py --data {yaml_path}")