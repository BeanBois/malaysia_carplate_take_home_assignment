import shutil
from pathlib import Path
from typing import List, Tuple
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class NegativeExampleManager:
    """Manages negative examples in YOLO format dataset"""
    def __init__(self):
        self.stats = {
            'added': 0,
            'skipped': 0,
            'errors': 0
        }
    
    # Process and copy image files with empty annotations
    def _process_files(
        self,
        image_files: List[Path],
        dest_images_dir: Path,
        dest_labels_dir: Path,
        prefix: str,
        split_name: str
    ) -> int:
        count = 0
        for img_path in image_files:
            try:
                # Generate new filename with prefix
                new_name = f"{prefix}_{img_path.name}"
                dest_img = dest_images_dir / new_name
                
                # Skip if already exists
                if dest_img.exists():
                    logger.warning(f"Skipping (already exists): {new_name}")
                    self.stats['skipped'] += 1
                    continue
                
                # Copy image
                shutil.copy2(img_path, dest_img)
                
                # Create empty annotation file
                label_file = dest_labels_dir / f"{dest_img.stem}.txt"
                label_file.touch()
                
                count += 1
                self.stats['added'] += 1
                
            except Exception as e:
                logger.error(f"Error processing {img_path.name}: {e}")
                self.stats['errors'] += 1
                continue
        
        logger.info(f"Added {count} images to {split_name}")
        return count
   
    def add_negative_examples(
        self,
        negative_images_dir: str,
        output_base_dir: str,
        prefix: str = "negative",
        splits: List[str] = ['train', 'val', 'test'],
        split_ratios: dict = None
    ) -> dict:
        """
        Add negative examples to multiple splits (train, val, test)
        
        Args:
            negative_images_dir: Directory containing negative example images
            output_base_dir: Base directory (e.g., data/raw)
            prefix: Prefix for negative image filenames
            splits: List of splits to add negatives to (e.g., ['train', 'val', 'test'])
            split_ratios: Dictionary of split ratios (default: {'train': 0.70, 'val': 0.15, 'test': 0.15})
            
        Returns:
            Dictionary with counts for each split: {'train': count, 'val': count, 'test': count}
        """
        logger.info("="*60)
        logger.info("ADDING NEGATIVE EXAMPLES TO DATASET")
        logger.info("="*60)
        
        neg_dir = Path(negative_images_dir)
        if not neg_dir.exists():
            logger.error(f"Negative images directory not found: {neg_dir}")
            return {}
        
        # Default split ratios
        if split_ratios is None:
            split_ratios = {
                'train': 0.70,
                'val': 0.15,
                'test': 0.15
            }
        
        # Normalize ratios to sum to 1.0
        total_ratio = sum(split_ratios.get(s, 0) for s in splits)
        if total_ratio > 0:
            split_ratios = {k: v / total_ratio for k, v in split_ratios.items()}
        
        base_dir = Path(output_base_dir)
        
        # Setup directories for all requested splits
        split_dirs = {}
        for split in splits:
            images_dir = base_dir / split / 'images'
            labels_dir = base_dir / split / 'labels'
            images_dir.mkdir(parents=True, exist_ok=True)
            labels_dir.mkdir(parents=True, exist_ok=True)
            split_dirs[split] = {'images': images_dir, 'labels': labels_dir}
        
        # Collect all image files
        image_files = []
        for ext in ['*.jpg', '*.jpeg', '*.png', '*.JPG', '*.JPEG', '*.PNG']:
            image_files.extend(list(neg_dir.glob(ext)))
        
        if not image_files:
            logger.warning(f"No image files found in {neg_dir}")
            return {}
        
        logger.info(f"Found {len(image_files)} negative images")
        logger.info(f"Distributing across splits: {', '.join(splits)}")
        
        # Shuffle and split files
        import random
        random.shuffle(image_files)
        
        split_files = {}
        start_idx = 0
        
        for i, split in enumerate(splits):
            ratio = split_ratios.get(split, 0)
            
            # For the last split, take all remaining files
            if i == len(splits) - 1:
                end_idx = len(image_files)
            else:
                count = int(len(image_files) * ratio)
                end_idx = start_idx + count
            
            split_files[split] = image_files[start_idx:end_idx]
            start_idx = end_idx
            
            logger.info(f"  {split}: {len(split_files[split])} images")
        
        # Process files for each split
        results = {}
        for split in splits:
            count = self._process_files(
                split_files[split],
                split_dirs[split]['images'],
                split_dirs[split]['labels'],
                prefix,
                split
            )
            results[split] = count
        
        logger.info("\n" + "="*60)
        logger.info("SUMMARY")
        logger.info("="*60)
        total_count = sum(results.values())
        logger.info(f"Total negative examples added: {total_count}")
        for split, count in results.items():
            logger.info(f"  {split}: {count}")
        logger.info(f"  Skipped: {self.stats['skipped']}")
        logger.info(f"  Errors: {self.stats['errors']}")
        
        return results
    
    def remove_negative_examples(
        self,
        dataset_dir: str,
        prefix: str = "negative",
        split: str = 'train',
        dry_run: bool = True
    ) -> int:
        """
        Remove negative examples (useful if you want to start over)
        
        Args:
            dataset_dir: Base directory
            prefix: Prefix used for negative examples
            split: Which split to clean
            dry_run: If True, only show what would be deleted
            
        Returns:
            Number of files removed/would be removed
        """
        base_dir = Path(dataset_dir)
        images_dir = base_dir / split / 'images'
        labels_dir = base_dir / split / 'labels'
        
        count = 0
        
        for img_file in images_dir.glob(f"{prefix}_*"):
            label_file = labels_dir / f"{img_file.stem}.txt"
            
            if dry_run:
                logger.info(f"Would remove: {img_file.name}")
            else:
                img_file.unlink()
                if label_file.exists():
                    label_file.unlink()
            
            count += 1
        
        if dry_run:
            logger.info(f"\nDry run: Would remove {count} negative examples")
            logger.info("Run with dry_run=False to actually delete")
        else:
            logger.info(f"Removed {count} negative examples from {split}")
        
        return count



if __name__ == "__main__":

    manager = NegativeExampleManager()
   
   # Add to all splits (train, val, test)
    results = manager.add_negative_examples(
        negative_images_dir="data/negative_examples",
        output_base_dir="data/raw",
        prefix="negative",
        splits=['train', 'val', 'test'],
        split_ratios={'train': 0.70, 'val': 0.15, 'test': 0.15}
    )

    # to reverse 
    # reverse = manager.remove_negative_examples(
    #     dataset_dir="data/raw",
    #     prefix='negative',
    #     splits=['train', 'val', 'test'],
    #     dry_run=False
    # )