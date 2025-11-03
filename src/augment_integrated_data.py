import logging
import sys
from pathlib import Path

sys.path.append('.')

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def augment_integrated_data():
    """
    Apply augmentation to the integrated dataset
    """
    from data_preparation import DatasetPreparer
    
    logger.info("="*70)
    logger.info("AUGMENTING INTEGRATED DATASET")
    logger.info("="*70)
    
    # Initialize preparer
    preparer = DatasetPreparer(config_path="config.yaml")
    
    # Augment the integrated dataset
    yaml_path = preparer.prepare_dataset_from_presplit(
        input_dir="data/integrated",
        output_dir="data/integrated_augmented",
        num_augmentations=3,   
        augment_splits=['train']   
    )
    
    logger.info("\n" + "="*70)
    logger.info("AUGMENTATION COMPLETE!")
    logger.info("="*70)
    
    # Count files in augmented dataset
    augmented_path = Path("data/integrated_augmented")
    train_images = len(list((augmented_path / "train" / "images").glob("*")))
    val_images = len(list((augmented_path / "val" / "images").glob("*")))
    test_images = len(list((augmented_path / "test" / "images").glob("*")))
    total_images = train_images + val_images + test_images
    
    logger.info(f"\nAugmented Dataset Statistics:")
    logger.info(f"  Total images: {total_images}")
    logger.info(f"  Training:     {train_images} (includes augmented data)")
    logger.info(f"  Validation:   {val_images} (no augmentation)")
    logger.info(f"  Test:         {test_images} (no augmentation)")
    
    logger.info(f"\n✓ Augmented dataset ready at: {augmented_path.absolute()}")
    logger.info(f"✓ Use this data.yaml for training: {yaml_path}")
    
    logger.info("\n" + "="*70)
    logger.info("START TRAINING")
    logger.info("="*70)
    logger.info(f"\nRun this command to start training:")
    logger.info(f"python detector.py --train --data {yaml_path} --epochs 100")
    logger.info("\nOr if you have a separate training script:")
    logger.info(f"python train.py --data {yaml_path}")
    logger.info("="*70)


if __name__ == "__main__":
    augment_integrated_data()