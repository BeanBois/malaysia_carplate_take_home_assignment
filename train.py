import argparse
import yaml
from pathlib import Path
import logging
from datetime import datetime
import sys

sys.path.append('src')
sys.path.append('.')

from detector import LicensePlateDetector
from data_preparation import DatasetPreparer
from integrate_additional_data import DatasetIntegrator

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def get_dataset_path(dataset_type: str, config: dict) -> str:
    """
    Get the path to data.yaml for the specified dataset type
    
    Args:
        dataset_type: Type of dataset ('raw', 'processed', 'integrated', 'integrated_augmented')
        config: Configuration dictionary
    
    Returns:
        Path to data.yaml file
    """
    dataset_dirs = {
        'raw': config['data']['raw_dir'],
        'processed': config['data']['processed_dir'],
        'integrated': config['data']['integrated_dir'],
        'integrated_augmented': config['data']['integrated_augmented_dir']
    }
    
    if dataset_type not in dataset_dirs:
        raise ValueError(f"Unknown dataset type: {dataset_type}. Choose from: {list(dataset_dirs.keys())}")
    
    data_yaml = Path(dataset_dirs[dataset_type]) / 'data.yaml'
    
    if not data_yaml.exists():
        raise FileNotFoundError(
            f"data.yaml not found at: {data_yaml}\n"
            f"Please ensure the dataset is prepared. Run:\n"
            f"  - For 'integrated': python data_inteegration_pipeline.py\n"
            f"  - For 'integrated_augmented': python augment_integrated_data.py"
        )
    
    return str(data_yaml)


def train_detector(
    data_yaml: str,
    config_path: str = "config.yaml",
    epochs: int = None,
    resume: bool = False
):
    """
    Train the detection model
    
    Args:
        data_yaml: Path to dataset YAML file
        config_path: Path to configuration file
        epochs: Number of epochs (uses config if None)
        resume: Whether to resume from checkpoint
    """
    logger.info("="*60)
    logger.info("TRAINING LICENSE PLATE DETECTOR")
    logger.info("="*60)
    
    # Initialize detector
    detector = LicensePlateDetector(config_path=config_path)
    
    # Load config
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    
    if epochs is None:
        epochs = config['detection']['epochs']
    
    logger.info(f"Dataset: {data_yaml}")
    logger.info(f"Epochs: {epochs}")
    logger.info(f"Batch size: {config['detection']['batch_size']}")
    logger.info(f"Learning rate: {config['detection']['learning_rate']}")
    logger.info(f"Image size: {config['detection']['input_size']}")
    logger.info(f"Patience: {config['detection']['patience']}")
    
    # Count dataset statistics
    dataset_dir = Path(data_yaml).parent
    train_images = len(list((dataset_dir / 'train' / 'images').glob('*')))
    val_images = len(list((dataset_dir / 'val' / 'images').glob('*')))
    test_images = len(list((dataset_dir / 'test' / 'images').glob('*'))) if (dataset_dir / 'test' / 'images').exists() else 0
    
    logger.info(f"\nDataset Statistics:")
    logger.info(f"  Training images:   {train_images}")
    logger.info(f"  Validation images: {val_images}")
    logger.info(f"  Test images:       {test_images}")
    logger.info(f"  Total:             {train_images + val_images + test_images}")
    
    # Start training
    start_time = datetime.now()
    logger.info(f"\nTraining started at: {start_time}")
    
    results = detector.train(data_yaml=data_yaml, epochs=epochs)
    
    end_time = datetime.now()
    duration = end_time - start_time
    
    logger.info("="*60)
    logger.info("TRAINING COMPLETE!")
    logger.info("="*60)
    logger.info(f"Duration: {duration}")
    logger.info(f"Best model saved to: models/detection_training/plate_detector/weights/best.pt")
    
    return results


def prepare_and_train(
    raw_data_dir: str,
    config_path: str = "config.yaml",
    num_augmentations: int = 5,
    epochs: int = None,
    output_dir: str = "data/processed"
):
    """
    Complete pipeline: prepare pre-split data and train
    
    Args:
        raw_data_dir: Directory with train/val/test folders (YOLO format)
        config_path: Configuration file path
        num_augmentations: Number of augmented versions per training image
        epochs: Training epochs
        output_dir: Output directory for processed data
    """
    logger.info("="*60)
    logger.info("FULL TRAINING PIPELINE (PRE-SPLIT DATA)")
    logger.info("="*60)
    
    # Step 1: Prepare dataset from pre-split YOLO format
    logger.info("\nStep 1: Data Preparation and Augmentation")
    logger.info("-"*60)
    
    preparer = DatasetPreparer(config_path=config_path)
    
    yaml_path = preparer.prepare_dataset_from_presplit(
        input_dir=raw_data_dir,
        output_dir=output_dir,
        num_augmentations=num_augmentations,
        augment_splits=['train']  # Only augment training data
    )
    
    # Step 2: Train detector
    logger.info("\nStep 2: Training Detector")
    logger.info("-"*60)
    
    results = train_detector(
        data_yaml=yaml_path,
        config_path=config_path,
        epochs=epochs
    )
    
    logger.info("\n" + "="*60)
    logger.info("PIPELINE COMPLETE!")
    logger.info("="*60)
    
    return results


def integrate_and_train(
    config_path: str = "config.yaml",
    apply_augmentation: bool = True,
    epochs: int = None
):
    """
    NEW: Integrate Kaggle dataset and train
    
    Args:
        config_path: Configuration file path
        apply_augmentation: Whether to apply augmentation to integrated dataset
        epochs: Training epochs
    """
    logger.info("="*60)
    logger.info("INTEGRATE KAGGLE DATASET AND TRAIN")
    logger.info("="*60)
    
    # Load config
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    
    
    # Step 1: Integrate datasets
    logger.info("\nStep 1: Integrating Datasets")
    logger.info("-"*60)
    
    integrator = DatasetIntegrator(
        additional_data_dir=config['data']['additional_data_dir'],
        existing_data_dir=config['data']['raw_dir'],
        output_dir=config['data']['integrated_dir']
    )
    
    integrator.integrate_datasets(
        train_ratio=config['integration']['train_ratio'],
        val_ratio=config['integration']['val_ratio'],
        test_ratio=config['integration']['test_ratio']
    )
    
    # Step 2: Apply augmentation if requested
    if apply_augmentation:
        logger.info("\nStep 2: Applying Augmentation")
        logger.info("-"*60)
        
        preparer = DatasetPreparer(config_path=config_path)
        
        yaml_path = preparer.prepare_dataset_from_presplit(
            input_dir=config['data']['integrated_dir'],
            output_dir=config['data']['integrated_augmented_dir'],
            num_augmentations=config['integration']['num_augmentations'],
            augment_splits=['train']
        )
    else:
        # Create data.yaml for non-augmented integrated dataset
        yaml_path = str(Path(config['data']['integrated_dir']) / 'data.yaml')
    
    # Step 3: Train detector
    logger.info("\nStep 3: Training Detector")
    logger.info("-"*60)
    
    results = train_detector(
        data_yaml=yaml_path,
        config_path=config_path,
        epochs=epochs
    )
    
    logger.info("\n" + "="*60)
    logger.info("INTEGRATED PIPELINE COMPLETE!")
    logger.info("="*60)
    
    return results

def main():
    parser = argparse.ArgumentParser(
        description="Train License Plate Detector with Support for Integrated Datasets",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  
  # Full pipeline: prepare pre-split data and train
  python train.py --mode full_pipeline --raw-data data/raw --augmentations 5
  
  # NEW: Integrate Kaggle dataset and train in one command
  python train.py --mode integrate --augment
        """
    )
    
    parser.add_argument(
        "--mode",
        type=str,
        choices=["train", "full_pipeline", "integrate", "verify", "auto"],
        default="auto",
        help="Training mode (auto uses default_dataset from config)"
    )
    
    parser.add_argument(
        "--dataset",
        type=str,
        choices=["raw", "processed", "integrated", "integrated_augmented"],
        help="Dataset type to use (overrides config default)"
    )
    
    parser.add_argument(
        "--data",
        type=str,
        help="Path to data.yaml (for train mode)"
    )
    
    parser.add_argument(
        "--raw-data",
        type=str,
        default="data/raw",
        help="Path to raw data directory with train/val/test folders"
    )
    
    parser.add_argument(
        "--output",
        type=str,
        default="data/processed",
        help="Output directory for processed data (full_pipeline mode)"
    )
    
    parser.add_argument(
        "--config",
        type=str,
        default="config.yaml",
        help="Path to config file"
    )
    
    parser.add_argument(
        "--epochs",
        type=int,
        default=None,
        help="Number of training epochs (uses config default if not specified)"
    )
    
    parser.add_argument(
        "--augmentations",
        type=int,
        default=5,
        help="Number of augmentations per training image (full_pipeline mode)"
    )
    
    parser.add_argument(
        "--augment",
        action="store_true",
        help="Apply augmentation when integrating datasets (integrate mode)"
    )
    
    parser.add_argument(
        "--no-augment",
        action="store_true",
        help="Skip augmentation when integrating datasets (integrate mode)"
    )
    
    args = parser.parse_args()
    
    # Load config
    with open(args.config, 'r') as f:
        config = yaml.safe_load(f)
    
    try:
        
        if args.mode == "full_pipeline":
            
            prepare_and_train(
                raw_data_dir=args.raw_data,
                config_path=args.config,
                num_augmentations=args.augmentations,
                epochs=args.epochs,
                output_dir=args.output
            )
        
        elif args.mode == "integrate":
            # Integrate Kaggle dataset and train
            apply_aug = not args.no_augment if args.no_augment else args.augment
            
            integrate_and_train(
                config_path=args.config,
                apply_augmentation=apply_aug,
                epochs=args.epochs
            )
        
        elif args.mode == "auto":
            # Use dataset type from args or config
            dataset_type = args.dataset or config['data']['default_dataset']
            
            logger.info(f"Using dataset type: {dataset_type}")
            
            # Get path to data.yaml
            data_yaml = get_dataset_path(dataset_type, config)
            
            # Train
            train_detector(
                data_yaml=data_yaml,
                config_path=args.config,
                epochs=args.epochs
            )
    
    except Exception as e:
        logger.error(f"Error: {e}", exc_info=True)
        return 1
    
    return 0


if __name__ == "__main__":
    exit(main())