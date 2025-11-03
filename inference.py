import argparse
import cv2
from pathlib import Path
import json
import logging
import sys
import re
from typing import List, Dict, Tuple

sys.path.append('.')
sys.path.append('src')

from pipeline import LicensePlateRecognitionPipeline

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def parse_filename_labels(filename: str) -> List[str]:
    """
    Extract ground truth plate text(s) from filename
    
    Args:
        filename: Image filename (e.g., "ABC1234.jpg" or "ABC1234-XYZ5678.jpg")
    
    Returns:
        List of plate texts found in filename
    """
    # Remove file extension
    name_without_ext = Path(filename).stem
    
    # Split by dash to handle multiple plates
    plates = name_without_ext.split('-')
    
    # Clean and normalize each plate
    cleaned_plates = []
    for plate in plates:
        # Remove any special characters except spaces
        cleaned = re.sub(r'[^A-Za-z0-9\s]', '', plate)
        cleaned = cleaned.strip()
        if cleaned:
            cleaned_plates.append(cleaned.upper())
    
    return cleaned_plates


def calculate_edit_distance(s1: str, s2: str) -> int:
    """Calculate Levenshtein distance between two strings"""
    if len(s1) < len(s2):
        return calculate_edit_distance(s2, s1)
    
    if len(s2) == 0:
        return len(s1)
    
    previous_row = range(len(s2) + 1)
    for i, c1 in enumerate(s1):
        current_row = [i + 1]
        for j, c2 in enumerate(s2):
            insertions = previous_row[j + 1] + 1
            deletions = current_row[j] + 1
            substitutions = previous_row[j] + (c1 != c2)
            current_row.append(min(insertions, deletions, substitutions))
        previous_row = current_row
    
    return previous_row[-1]


def normalize_text(text: str) -> str:
    """Normalize text for comparison"""
    return text.upper().replace(' ', '').replace('-', '')


def match_predictions_to_ground_truth(
    predictions: List[Dict],
    ground_truth: List[str]
) -> Tuple[List[Dict], Dict]:
    """
    Match predictions to ground truth labels and calculate metrics
    
    Args:
        predictions: List of prediction results from pipeline
        ground_truth: List of ground truth plate texts from filename
    
    Returns:
        Tuple of (matched_results, metrics)
    """
    matched_results = []
    
    # Normalize ground truth
    gt_normalized = [normalize_text(gt) for gt in ground_truth]
    
    # Track which ground truths have been matched
    gt_matched = [False] * len(ground_truth)
    
    # Match each prediction to closest ground truth
    for pred in predictions:
        pred_text_normalized = normalize_text(pred['plate_text'])
        
        best_match_idx = None
        best_match_score = float('inf')
        
        for idx, gt_norm in enumerate(gt_normalized):
            if gt_matched[idx]:
                continue
            
            # Calculate edit distance
            edit_dist = calculate_edit_distance(pred_text_normalized, gt_norm)
            
            if edit_dist < best_match_score:
                best_match_score = edit_dist
                best_match_idx = idx
        
        # Create matched result
        if best_match_idx is not None:
            gt_matched[best_match_idx] = True
            gt_text = ground_truth[best_match_idx]
            gt_norm = gt_normalized[best_match_idx]
            
            # Calculate character accuracy
            correct_chars = sum(
                p == g for p, g in zip(pred_text_normalized, gt_norm)
            )
            total_chars = max(len(pred_text_normalized), len(gt_norm))
            char_accuracy = correct_chars / total_chars if total_chars > 0 else 0.0
            
            # Check if exact match (after normalization)
            exact_match = pred_text_normalized == gt_norm
            
            matched_result = {
                'predicted_text': pred['plate_text'],
                'ground_truth': gt_text,
                'bbox': pred['bbox'],
                'confidence': pred['confidence'],
                'detection_confidence': pred['detection_confidence'],
                'recognition_confidence': pred['recognition_confidence'],
                'is_valid_format': pred['is_valid'],
                'exact_match': exact_match,
                'character_accuracy': char_accuracy,
                'edit_distance': best_match_score
            }
            
            matched_results.append(matched_result)
    
    # Calculate overall metrics
    num_predictions = len(predictions)
    num_ground_truth = len(ground_truth)
    num_matched = sum(gt_matched)
    
    # Detection metrics
    true_positives = num_matched
    false_positives = num_predictions - num_matched
    false_negatives = num_ground_truth - num_matched
    
    precision = true_positives / num_predictions if num_predictions > 0 else 0.0
    recall = true_positives / num_ground_truth if num_ground_truth > 0 else 0.0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0
    
    # Recognition metrics (only for matched plates)
    if matched_results:
        avg_char_accuracy = sum(r['character_accuracy'] for r in matched_results) / len(matched_results)
        avg_edit_distance = sum(r['edit_distance'] for r in matched_results) / len(matched_results)
        word_accuracy = sum(r['exact_match'] for r in matched_results) / len(matched_results)
    else:
        avg_char_accuracy = 0.0
        avg_edit_distance = 0.0
        word_accuracy = 0.0
    
    metrics = {
        'detection': {
            'precision': precision,
            'recall': recall,
            'f1': f1,
            'true_positives': true_positives,
            'false_positives': false_positives,
            'false_negatives': false_negatives
        },
        'recognition': {
            'character_accuracy': avg_char_accuracy,
            'word_accuracy': word_accuracy,
            'avg_edit_distance': avg_edit_distance
        },
        'counts': {
            'num_predictions': num_predictions,
            'num_ground_truth': num_ground_truth,
            'num_matched': num_matched
        }
    }
    
    return matched_results, metrics


def process_image_with_ground_truth(
    image_path: str,
    pipeline: LicensePlateRecognitionPipeline,
    output_dir: str = "outputs/results",
    save_viz: bool = True
) -> Dict:
    """
    Process a single image and compare with ground truth from filename
    
    Args:
        image_path: Path to input image
        pipeline: Initialized pipeline
        output_dir: Output directory for results
        save_viz: Whether to save visualizations
    
    Returns:
        Results dictionary with predictions, ground truth, and metrics
    """
    image_path = Path(image_path)
    logger.info(f"Processing: {image_path.name}")
    
    # Extract ground truth from filename
    ground_truth = parse_filename_labels(image_path.name)
    logger.info(f"Ground truth from filename: {ground_truth}")
    
    # Run inference
    predictions = pipeline.process_image(
        image=str(image_path),
        save_visualizations=save_viz,
        output_dir=output_dir
    )
    
    # Match predictions to ground truth
    matched_results, metrics = match_predictions_to_ground_truth(predictions, ground_truth)
    
    # Compile results
    result = {
        'filename': image_path.name,
        'ground_truth': ground_truth,
        'num_predictions': len(predictions),
        'matched_results': matched_results,
        'metrics': metrics
    }
    
    return result


def process_directory_with_evaluation(
    input_dir: str,
    model_path: str = None,
    output_dir: str = "outputs/results",
    save_viz: bool = True,
    use_multi_engine: bool = False
) -> Dict:
    """
    Process all images in a directory and evaluate against filename labels
    
    Args:
        input_dir: Directory containing images with labels in filenames
        model_path: Path to trained detection model
        output_dir: Output directory for results
        save_viz: Whether to save visualizations
        use_multi_engine: Use multiple OCR engines
    
    Returns:
        Aggregated results and metrics
    """
    input_dir = Path(input_dir)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Find all images
    image_files = []
    for ext in ['*.jpg', '*.jpeg', '*.png', '*.JPG', '*.JPEG', '*.PNG']:
        image_files.extend(list(input_dir.glob(ext)))
    
    if not image_files:
        logger.error(f"No images found in {input_dir}")
        return {}
    
    logger.info(f"Found {len(image_files)} images in {input_dir}")
    
    # Initialize pipeline once
    pipeline = LicensePlateRecognitionPipeline(
        detection_model_path=model_path,
        use_multi_engine=use_multi_engine
    )
    
    # Process all images
    all_results = []
    
    for img_path in image_files:
        try:
            result = process_image_with_ground_truth(
                image_path=str(img_path),
                pipeline=pipeline,
                output_dir=output_dir,
                save_viz=save_viz
            )
            all_results.append(result)
            
            # Print per-image results
            print(f"\n{'='*60}")
            print(f"Image: {result['filename']}")
            print(f"Ground Truth: {result['ground_truth']}")
            print(f"Predictions: {result['num_predictions']}")
            print(f"Detection F1: {result['metrics']['detection']['f1']:.3f}")
            print(f"Recognition Accuracy: {result['metrics']['recognition']['word_accuracy']:.3f}")
            
            for idx, match in enumerate(result['matched_results'], 1):
                print(f"\n  Match {idx}:")
                print(f"    Predicted: {match['predicted_text']}")
                print(f"    Ground Truth: {match['ground_truth']}")
                print(f"    Exact Match: {match['exact_match']}")
                print(f"    Char Accuracy: {match['character_accuracy']:.3f}")
                print(f"    Edit Distance: {match['edit_distance']}")
        
        except Exception as e:
            logger.error(f"Error processing {img_path.name}: {e}")
            all_results.append({
                'filename': img_path.name,
                'error': str(e)
            })
    
    # Calculate aggregate metrics
    logger.info("\n" + "="*60)
    logger.info("CALCULATING AGGREGATE METRICS")
    logger.info("="*60)
    
    valid_results = [r for r in all_results if 'error' not in r]
    
    if not valid_results:
        logger.error("No valid results to aggregate")
        return {'all_results': all_results}
    
    # Aggregate detection metrics
    total_tp = sum(r['metrics']['detection']['true_positives'] for r in valid_results)
    total_fp = sum(r['metrics']['detection']['false_positives'] for r in valid_results)
    total_fn = sum(r['metrics']['detection']['false_negatives'] for r in valid_results)
    
    overall_precision = total_tp / (total_tp + total_fp) if (total_tp + total_fp) > 0 else 0.0
    overall_recall = total_tp / (total_tp + total_fn) if (total_tp + total_fn) > 0 else 0.0
    overall_f1 = 2 * overall_precision * overall_recall / (overall_precision + overall_recall) \
                 if (overall_precision + overall_recall) > 0 else 0.0
    
    # Aggregate recognition metrics
    all_matched = []
    for r in valid_results:
        all_matched.extend(r['matched_results'])
    
    if all_matched:
        overall_char_accuracy = sum(m['character_accuracy'] for m in all_matched) / len(all_matched)
        overall_word_accuracy = sum(m['exact_match'] for m in all_matched) / len(all_matched)
        overall_edit_distance = sum(m['edit_distance'] for m in all_matched) / len(all_matched)
    else:
        overall_char_accuracy = 0.0
        overall_word_accuracy = 0.0
        overall_edit_distance = 0.0
    
    aggregate_metrics = {
        'dataset': {
            'total_images': len(image_files),
            'successfully_processed': len(valid_results),
            'failed': len(image_files) - len(valid_results)
        },
        'detection': {
            'precision': overall_precision,
            'recall': overall_recall,
            'f1': overall_f1,
            'true_positives': total_tp,
            'false_positives': total_fp,
            'false_negatives': total_fn
        },
        'recognition': {
            'character_accuracy': overall_char_accuracy,
            'word_accuracy': overall_word_accuracy,
            'avg_edit_distance': overall_edit_distance,
            'total_matched_plates': len(all_matched)
        }
    }
    
    # Save detailed results
    results_file = output_dir / 'detailed_results.json'
    with open(results_file, 'w') as f:
        # Convert numpy arrays to lists for JSON serialization
        json_results = []
        for r in all_results:
            json_r = r.copy()
            if 'matched_results' in json_r:
                for match in json_r['matched_results']:
                    match['bbox'] = [int(x) for x in match['bbox']]
            json_results.append(json_r)
        
        json.dump({
            'aggregate_metrics': aggregate_metrics,
            'per_image_results': json_results
        }, f, indent=2)
    
    logger.info(f"Detailed results saved to: {results_file}")
    
    # Save summary
    summary_file = output_dir / 'summary.json'
    with open(summary_file, 'w') as f:
        json.dump(aggregate_metrics, f, indent=2)
    
    logger.info(f"Summary saved to: {summary_file}")
    
    # Print summary
    print("\n" + "="*60)
    print("EVALUATION SUMMARY")
    print("="*60)
    print(f"\nDataset:")
    print(f"  Total images: {aggregate_metrics['dataset']['total_images']}")
    print(f"  Successfully processed: {aggregate_metrics['dataset']['successfully_processed']}")
    print(f"  Failed: {aggregate_metrics['dataset']['failed']}")
    
    print(f"\nDetection Metrics:")
    print(f"  Precision: {aggregate_metrics['detection']['precision']:.3f}")
    print(f"  Recall:    {aggregate_metrics['detection']['recall']:.3f}")
    print(f"  F1 Score:  {aggregate_metrics['detection']['f1']:.3f}")
    print(f"  TP: {aggregate_metrics['detection']['true_positives']}, "
          f"FP: {aggregate_metrics['detection']['false_positives']}, "
          f"FN: {aggregate_metrics['detection']['false_negatives']}")
    
    print(f"\nRecognition Metrics:")
    print(f"  Character Accuracy: {aggregate_metrics['recognition']['character_accuracy']:.3f}")
    print(f"  Word Accuracy:      {aggregate_metrics['recognition']['word_accuracy']:.3f}")
    print(f"  Avg Edit Distance:  {aggregate_metrics['recognition']['avg_edit_distance']:.2f}")
    print(f"  Total Matched:      {aggregate_metrics['recognition']['total_matched_plates']}")
    
    print("\n" + "="*60 + "\n")
    
    return {
        'aggregate_metrics': aggregate_metrics,
        'all_results': all_results
    }


def main():
    n = 2 
    parser = argparse.ArgumentParser(
        description="License Plate Recognition Inference with Filename-Based Ground Truth",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
    # To run inference: 
    python inference.py --directory testdata/images/ --model models/detection_training/plate_detector{n}/wieghts/best.pt --multi-engine
    
    where n depends on how many times you have trained the model. Basically locate the desired model under models/detection_training/*
    Filename Format:
    - Single plate: ABC1234.jpg
    - Multiple plates: ABC1234-XYZ5678.jpg (separated by dash)
        """
    )
    
    # Input options
    input_group = parser.add_mutually_exclusive_group(required=True)
    input_group.add_argument("--image", type=str, help="Path to input image")
    input_group.add_argument("--directory", type=str, help="Path to directory of images")
    
    # Model options
    parser.add_argument(
        "--model",
        type=str,
        default=None,
        help="Path to trained detection model (optional)"
    )
    
    parser.add_argument(
        "--multi-engine",
        action="store_true",
        help="Use multiple OCR engines for better accuracy"
    )
    
    # Output options
    parser.add_argument(
        "--output",
        type=str,
        default="outputs/results",
        help="Output directory"
    )
    
    parser.add_argument(
        "--no-viz",
        action="store_true",
        help="Don't save visualization images"
    )
    
    args = parser.parse_args()
    
    try:
        if args.image:
            # Single image processing
            pipeline = LicensePlateRecognitionPipeline(
                detection_model_path=args.model,
                use_multi_engine=args.multi_engine
            )
            
            result = process_image_with_ground_truth(
                image_path=args.image,
                pipeline=pipeline,
                output_dir=args.output,
                save_viz=not args.no_viz
            )
            
            # Print results
            print("\n" + "="*60)
            print("RESULTS")
            print("="*60)
            print(f"Image: {result['filename']}")
            print(f"Ground Truth: {result['ground_truth']}")
            print(f"Predictions: {result['num_predictions']}")
            print(f"\nMetrics:")
            print(f"  Detection F1: {result['metrics']['detection']['f1']:.3f}")
            print(f"  Recognition Accuracy: {result['metrics']['recognition']['word_accuracy']:.3f}")
            
            for idx, match in enumerate(result['matched_results'], 1):
                print(f"\n  Plate {idx}:")
                print(f"    Predicted: {match['predicted_text']}")
                print(f"    Ground Truth: {match['ground_truth']}")
                print(f"    Match: {'✓' if match['exact_match'] else '✗'}")
                print(f"    Confidence: {match['confidence']:.3f}")
        
        elif args.directory:
            # Directory processing with evaluation
            process_directory_with_evaluation(
                input_dir=args.directory,
                model_path=args.model,
                output_dir=args.output,
                save_viz=not args.no_viz,
                use_multi_engine=args.multi_engine
            )
    
    except Exception as e:
        logger.error(f"Error during processing: {e}", exc_info=True)
        return 1
    
    return 0


if __name__ == "__main__":
    exit(main())