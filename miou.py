import laspy
import numpy as np
from sklearn.metrics import confusion_matrix, accuracy_score
import argparse
import os

# DALES class mapping
BERGEN_DALES_NUM_CLASSES = 8  # Number of classes in DALES (0-7)
DALES_CLASS_NAMES = ['Ground', 'Vegetation', 'Cars', 'Trucks', 'Power lines', 'Fences', 'Poles', 'Buildings']

# Bergen to DALES class mapping
ID2TRAINID_BERGEN_DALES = np.asarray([
    BERGEN_DALES_NUM_CLASSES,  #  0 Not used          ->  8 Ignored
    0,                         #  1 Unclassified      ->  0 Ground   (rough fallback)
    0,                         #  2 Terreng           ->  0 Ground
    1,                         #  3 Low vegetation    ->  1 Vegetation
    1,                         #  4 Medium vegetation ->  1 Vegetation
    1,                         #  5 High vegetation   ->  1 Vegetation
    7,                         #  6 Bygning           ->  7 Buildings
    BERGEN_DALES_NUM_CLASSES,  #  7 Noise             ->  8 Ignored
    BERGEN_DALES_NUM_CLASSES,  #  8 Not used          ->  8 Ignored
    0,                         #  9 Vann              ->  0 Ground   (DALES has no water class)
    BERGEN_DALES_NUM_CLASSES,  # 10 Not used          ->  8 Ignored
    BERGEN_DALES_NUM_CLASSES,  # 11 Not used          ->  8 Ignored
    BERGEN_DALES_NUM_CLASSES,  # 12 Not used          ->  8 Ignored
    BERGEN_DALES_NUM_CLASSES,  # 13 Not used          ->  8 Ignored
    4,                         # 14 Powerlines        ->  4 Powerlines
    BERGEN_DALES_NUM_CLASSES,  # 15 Transmission tower->  8 Ignored
    BERGEN_DALES_NUM_CLASSES,  # 16 Not used          ->  8 Ignored
    7,                         # 17 Bro               ->  7 Buildings
    BERGEN_DALES_NUM_CLASSES,  # 18 Not used          ->  8 Ignored
    BERGEN_DALES_NUM_CLASSES,  # 19 Not used          ->  8 Ignored
    BERGEN_DALES_NUM_CLASSES,  # 20 Not used          ->  8 Ignored
    BERGEN_DALES_NUM_CLASSES,  # 21 Not used          ->  8 Ignored
    BERGEN_DALES_NUM_CLASSES,  # 22 Not used          ->  8 Ignored
    BERGEN_DALES_NUM_CLASSES,  # 23 Not used          ->  8 Ignored
    0                          # 24 Snø               ->  0 Ground
])

def load_point_cloud(file_path):
    """
    Load point cloud from LAS/LAZ file
    """
    las = laspy.read(file_path)
    return las

def extract_labels(las_file):
    """
    Extract classification labels from LAS file
    """
    return las_file.classification

def map_bergen_to_dales(labels):
    """
    Map Bergen class labels to DALES class labels
    """
    # Ensure labels don't exceed the mapping array size
    max_label = len(ID2TRAINID_BERGEN_DALES) - 1
    labels = np.clip(labels, 0, max_label)
    
    # Apply mapping
    mapped_labels = ID2TRAINID_BERGEN_DALES[labels]
    
    return mapped_labels

def filter_ignored_classes(y_true, y_pred):
    """
    Filter out ignored classes (class 8) from evaluation
    """
    valid_mask = (y_true != BERGEN_DALES_NUM_CLASSES) & (y_pred != BERGEN_DALES_NUM_CLASSES)
    return y_true[valid_mask], y_pred[valid_mask]

def calculate_overall_accuracy(y_true, y_pred):
    """
    Calculate Overall Accuracy (OA)
    """
    return accuracy_score(y_true, y_pred)

def calculate_miou_present_classes_only(y_true, y_pred):
    """
    Calculate Mean Intersection over Union (MIOU) only for classes present in ground truth
    """
    # Get unique classes present in ground truth only
    unique_gt_classes = np.unique(y_true)
    
    # Create masks for predictions to only include classes present in ground truth
    mask = np.isin(y_pred, unique_gt_classes)
    y_true_filtered = y_true[mask]
    y_pred_filtered = y_pred[mask]
    
    print(f"Classes present in ground truth: {unique_gt_classes}")
    print(f"Points with predictions matching ground truth classes: {len(y_true_filtered)}/{len(y_true)} ({len(y_true_filtered)/len(y_true)*100:.2f}%)")
    
    # Debug: Show distribution after filtering
    print(f"Ground truth distribution after filtering:")
    unique_true_filtered, counts_true_filtered = np.unique(y_true_filtered, return_counts=True)
    for cls, count in zip(unique_true_filtered, counts_true_filtered):
        class_name = DALES_CLASS_NAMES[cls] if cls < len(DALES_CLASS_NAMES) else f"Unknown({cls})"
        print(f"  Class {cls} ({class_name}): {count} points ({count/len(y_true_filtered)*100:.2f}%)")
    
    print(f"Predictions distribution after filtering:")
    unique_pred_filtered, counts_pred_filtered = np.unique(y_pred_filtered, return_counts=True)
    for cls, count in zip(unique_pred_filtered, counts_pred_filtered):
        class_name = DALES_CLASS_NAMES[cls] if cls < len(DALES_CLASS_NAMES) else f"Unknown({cls})"
        print(f"  Class {cls} ({class_name}): {count} points ({count/len(y_pred_filtered)*100:.2f}%)")
    
    if len(y_true_filtered) == 0:
        return 0.0, [], []
    
    # Calculate confusion matrix only for present classes
    cm = confusion_matrix(y_true_filtered, y_pred_filtered, labels=unique_gt_classes)
    
    print(f"\nConfusion matrix for present classes only:")
    print(f"Predicted classes: {unique_gt_classes}")
    print(cm)
    
    # Calculate IoU for each present class
    iou_per_class = []
    for i, class_label in enumerate(unique_gt_classes):
        # True Positives
        tp = cm[i, i]
        # False Positives
        fp = np.sum(cm[:, i]) - tp
        # False Negatives
        fn = np.sum(cm[i, :]) - tp
        
        # IoU = TP / (TP + FP + FN)
        if tp + fp + fn == 0:
            iou = 0.0  # Handle case where class is not present
        else:
            iou = tp / (tp + fp + fn)
        
        iou_per_class.append(iou)
        class_name = DALES_CLASS_NAMES[class_label] if class_label < len(DALES_CLASS_NAMES) else f"Unknown({class_label})"
        print(f"Class {class_label} ({class_name}): TP={tp}, FP={fp}, FN={fn}, IoU = {iou:.4f}")
    
    # Calculate mean IoU
    miou = np.mean(iou_per_class)
    return miou, iou_per_class, unique_gt_classes

def calculate_miou(y_true, y_pred, num_classes=None):
    """
    Calculate Mean Intersection over Union (MIOU)
    """
    if num_classes is None:
        # Get unique classes from both true and predicted labels
        unique_classes = np.unique(np.concatenate([y_true, y_pred]))
    else:
        unique_classes = np.arange(num_classes)
    
    # Calculate confusion matrix
    cm = confusion_matrix(y_true, y_pred, labels=unique_classes)
    
    # Calculate IoU for each class
    iou_per_class = []
    for i, class_label in enumerate(unique_classes):
        # True Positives
        tp = cm[i, i]
        # False Positives
        fp = np.sum(cm[:, i]) - tp
        # False Negatives
        fn = np.sum(cm[i, :]) - tp
        
        # IoU = TP / (TP + FP + FN)
        if tp + fp + fn == 0:
            iou = 0.0  # Handle case where class is not present
        else:
            iou = tp / (tp + fp + fn)
        
        iou_per_class.append(iou)
        print(f"Class {class_label}: IoU = {iou:.4f}")
    
    # Calculate mean IoU
    miou = np.mean(iou_per_class)
    return miou, iou_per_class, unique_classes

def print_confusion_matrix(y_true, y_pred, class_names=None):
    """
    Print confusion matrix with class statistics
    """
    unique_classes = np.unique(np.concatenate([y_true, y_pred]))
    cm = confusion_matrix(y_true, y_pred, labels=unique_classes)
    
    if class_names is None:
        class_names = [f"Class {i}" for i in unique_classes]
    
    print("\nConfusion Matrix:")
    print("Predicted ->")
    print("True |")
    
    # Print header with class names
    print("     ", end="")
    for cls in unique_classes:
        if cls < len(DALES_CLASS_NAMES):
            print(f"{DALES_CLASS_NAMES[cls][:8]:>8}", end="")
        else:
            print(f"{cls:>8}", end="")
    print()
    
    # Print matrix
    for i, true_cls in enumerate(unique_classes):
        if true_cls < len(DALES_CLASS_NAMES):
            print(f"{DALES_CLASS_NAMES[true_cls][:4]:>4} ", end="")
        else:
            print(f"{true_cls:>4} ", end="")
        for j, pred_cls in enumerate(unique_classes):
            print(f"{cm[i, j]:>8}", end="")
        print()

def main():
    # File paths
    ground_truth_path = ""
    predictions_path = ""
    output_file = ""

    
    print("Loading ground truth point cloud...")
    if not os.path.exists(ground_truth_path):
        raise FileNotFoundError(f"Ground truth file not found: {ground_truth_path}")
    
    print("Loading predictions point cloud...")
    if not os.path.exists(predictions_path):
        raise FileNotFoundError(f"Predictions file not found: {predictions_path}")
    
    # Load point clouds
    ground_truth_las = load_point_cloud(ground_truth_path)
    predictions_las = load_point_cloud(predictions_path)
    
    # Extract labels
    y_true_raw = extract_labels(ground_truth_las)
    y_pred_raw = extract_labels(predictions_las)
    
    print(f"\nOriginal Bergen ground truth classes: {np.unique(y_true_raw)}")
    print(f"Original prediction classes: {np.unique(y_pred_raw)}")
    
    # Print original Bergen class distribution
    print("\nOriginal Bergen Ground Truth Class Distribution:")
    BERGEN_CLASS_NAMES = {
        1: "Unclassified",
        2: "Terreng (Ground)",
        3: "Low vegetation",
        4: "Medium vegetation",
        5: "High vegetation",
        6: "Bygning (Building)",
        7: "Noise",
        9: "Vann (Water)",
        14: "Powerlines",
        15: "Transmission tower",
        17: "Bro (Bridge)",
        24: "Snø (Snow)"
    }
    unique_bergen, counts_bergen = np.unique(y_true_raw, return_counts=True)
    for cls, count in zip(unique_bergen, counts_bergen):
        class_name = BERGEN_CLASS_NAMES.get(cls, f"Unknown({cls})")
        print(f"Class {cls} ({class_name}): {count} points ({count/len(y_true_raw)*100:.2f}%)")
    
    # Map Bergen classes to DALES classes (only for ground truth, predictions are already in DALES format)
    print("\nMapping Bergen classes to DALES classes...")
    y_true = map_bergen_to_dales(y_true_raw)
    y_pred = y_pred_raw  # Predictions are already in DALES format (0-7)
    
    print(f"Mapped ground truth classes: {np.unique(y_true)}")
    print(f"Final prediction classes: {np.unique(y_pred)}")
    
    # Check if the number of points match
    if len(y_true) != len(y_pred):
        print(f"Warning: Number of points don't match!")
        print(f"Ground truth: {len(y_true)} points")
        print(f"Predictions: {len(y_pred)} points")
        
        # Take the minimum number of points if they don't match
        min_points = min(len(y_true), len(y_pred))
        y_true = y_true[:min_points]
        y_pred = y_pred[:min_points]
        print(f"Using first {min_points} points for comparison")
    
    # Print class distribution BEFORE filtering
    print("\nGround Truth Class Distribution (DALES classes) - ALL MAPPED CLASSES:")
    unique_true_all, counts_true_all = np.unique(y_true, return_counts=True)
    for cls, count in zip(unique_true_all, counts_true_all):
        if cls == BERGEN_DALES_NUM_CLASSES:
            class_name = "Ignored/Noise"
        elif cls < len(DALES_CLASS_NAMES):
            class_name = DALES_CLASS_NAMES[cls]
        else:
            class_name = f"Unknown({cls})"
        print(f"Class {cls} ({class_name}): {count} points ({count/len(y_true)*100:.2f}%)")
    
    print("\nPredicted Class Distribution (DALES classes) - ALL CLASSES:")
    unique_pred_all, counts_pred_all = np.unique(y_pred, return_counts=True)
    for cls, count in zip(unique_pred_all, counts_pred_all):
        class_name = DALES_CLASS_NAMES[cls] if cls < len(DALES_CLASS_NAMES) else f"Unknown({cls})"
        print(f"Class {cls} ({class_name}): {count} points ({count/len(y_pred)*100:.2f}%)")
    
    # Filter out ignored classes for evaluation
    print("\nFiltering ignored classes...")
    y_true_filtered, y_pred_filtered = filter_ignored_classes(y_true, y_pred)
    
    print(f"Evaluating {len(y_true_filtered)} points (after filtering ignored classes)...")
    print(f"Filtered out {len(y_true) - len(y_true_filtered)} ignored points")
    
    # Calculate metrics
    print("\n" + "="*50)
    print("EVALUATION METRICS")
    print("="*50)
    
    # Overall Accuracy
    oa = calculate_overall_accuracy(y_true_filtered, y_pred_filtered)
    print(f"Overall Accuracy (OA): {oa:.4f} ({oa*100:.2f}%)")
    
    # Mean IoU (all classes)
    miou, iou_per_class, unique_classes = calculate_miou(y_true_filtered, y_pred_filtered)
    print(f"\nMean Intersection over Union (MIOU) - All Classes: {miou:.4f}")
    
    # Mean IoU (only classes present in ground truth)
    print(f"\n{'-'*50}")
    print("EVALUATION FOR PRESENT CLASSES ONLY")
    print(f"{'-'*50}")
    miou_present, iou_present, gt_classes = calculate_miou_present_classes_only(y_true_filtered, y_pred_filtered)
    print(f"\nMean Intersection over Union (MIOU) - Present Classes Only: {miou_present:.4f}")
    
    # Overall accuracy for present classes only
    mask_present = np.isin(y_pred_filtered, gt_classes)
    y_true_present = y_true_filtered[mask_present]
    y_pred_present = y_pred_filtered[mask_present]
    oa_present = calculate_overall_accuracy(y_true_present, y_pred_present)
    print(f"Overall Accuracy (OA) - Present Classes Only: {oa_present:.4f} ({oa_present*100:.2f}%)")
    
    print(f"\n{'-'*50}")
    print("DETAILED RESULTS")
    print(f"{'-'*50}")
    
    # Print detailed confusion matrix
    print_confusion_matrix(y_true_filtered, y_pred_filtered)
    
    # Additional per-class metrics
    print("\nPer-class IoU Summary:")
    for cls, iou in zip(unique_classes, iou_per_class):
        class_name = DALES_CLASS_NAMES[cls] if cls < len(DALES_CLASS_NAMES) else f"Unknown({cls})"
        print(f"Class {cls} ({class_name}): {iou:.4f}")
    
    print("\n" + "="*50)
    print("FINAL SUMMARY")
    print("="*50)
    print(f"Overall Accuracy (All Classes): {oa:.4f}")
    print(f"Overall Accuracy (Present Classes Only): {oa_present:.4f}")
    print(f"Mean IoU (All Classes): {miou:.4f}")
    print(f"Mean IoU (Present Classes Only): {miou_present:.4f}")
    print(f"Number of classes in ground truth: {len(gt_classes)}")
    print(f"Number of classes in predictions: {len(unique_classes)}")
    print(f"Classes present in ground truth: {gt_classes}")
    print(f"Classes predicted by model: {unique_classes}")
    
    # Write results to file
    print(f"\nWriting results to {output_file}...")
    with open(output_file, 'w') as f:
        f.write("="*80 + "\n")
        f.write("SEMANTIC SEGMENTATION EVALUATION RESULTS\n")
        f.write("="*80 + "\n")
        f.write(f"Ground Truth: {ground_truth_path}\n")
        f.write(f"Predictions: {predictions_path}\n")
        f.write(f"Date: {__import__('datetime').datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
        f.write("="*80 + "\n\n")
        
        f.write("CLASS MAPPING\n")
        f.write("-"*80 + "\n")
        f.write(f"Original Bergen ground truth classes: {np.unique(y_true_raw)}\n")
        f.write(f"Original prediction classes: {np.unique(y_pred_raw)}\n")
        f.write(f"Mapped ground truth classes: {np.unique(y_true)}\n")
        f.write(f"Final prediction classes: {np.unique(y_pred)}\n\n")
        
        f.write("DATA STATISTICS\n")
        f.write("-"*80 + "\n")
        f.write(f"Total ground truth points: {len(y_true)}\n")
        f.write(f"Total prediction points: {len(y_pred)}\n")
        f.write(f"Points evaluated (after filtering): {len(y_true_filtered)}\n")
        f.write(f"Filtered out (ignored classes): {len(y_true) - len(y_true_filtered)}\n\n")
        
        f.write("ORIGINAL BERGEN GROUND TRUTH CLASS DISTRIBUTION\n")
        f.write("-"*80 + "\n")
        for cls, count in zip(unique_bergen, counts_bergen):
            class_name = BERGEN_CLASS_NAMES.get(cls, f"Unknown({cls})")
            f.write(f"Class {cls} ({class_name}): {count} points ({count/len(y_true_raw)*100:.2f}%)\n")
        f.write("\n")
        
        f.write("GROUND TRUTH CLASS DISTRIBUTION (DALES classes) - AFTER MAPPING\n")
        f.write("-"*80 + "\n")
        for cls, count in zip(unique_true_all, counts_true_all):
            if cls == BERGEN_DALES_NUM_CLASSES:
                class_name = "Ignored/Noise"
            elif cls < len(DALES_CLASS_NAMES):
                class_name = DALES_CLASS_NAMES[cls]
            else:
                class_name = f"Unknown({cls})"
            f.write(f"Class {cls} ({class_name}): {count} points ({count/len(y_true)*100:.2f}%)\n")
        f.write("\n")
        
        f.write("PREDICTED CLASS DISTRIBUTION (DALES classes) - ALL CLASSES\n")
        f.write("-"*80 + "\n")
        for cls, count in zip(unique_pred_all, counts_pred_all):
            class_name = DALES_CLASS_NAMES[cls] if cls < len(DALES_CLASS_NAMES) else f"Unknown({cls})"
            f.write(f"Class {cls} ({class_name}): {count} points ({count/len(y_pred)*100:.2f}%)\n")
        f.write("\n")
        
        f.write("="*80 + "\n")
        f.write("EVALUATION METRICS - ALL CLASSES\n")
        f.write("="*80 + "\n")
        f.write(f"Overall Accuracy (OA): {oa:.4f} ({oa*100:.2f}%)\n")
        f.write(f"Mean Intersection over Union (MIOU): {miou:.4f}\n\n")
        
        f.write("Per-class IoU (All Classes):\n")
        for cls, iou in zip(unique_classes, iou_per_class):
            class_name = DALES_CLASS_NAMES[cls] if cls < len(DALES_CLASS_NAMES) else f"Unknown({cls})"
            f.write(f"  Class {cls} ({class_name}): {iou:.4f}\n")
        f.write("\n")
        
        f.write("="*80 + "\n")
        f.write("EVALUATION METRICS - PRESENT CLASSES ONLY\n")
        f.write("="*80 + "\n")
        f.write(f"Classes present in ground truth: {gt_classes}\n")
        f.write(f"Points with predictions matching ground truth classes: {len(y_true_present)}/{len(y_true_filtered)} ({len(y_true_present)/len(y_true_filtered)*100:.2f}%)\n\n")
        f.write(f"Overall Accuracy (OA): {oa_present:.4f} ({oa_present*100:.2f}%)\n")
        f.write(f"Mean Intersection over Union (MIOU): {miou_present:.4f}\n\n")
        
        f.write("Per-class IoU (Present Classes Only):\n")
        for cls, iou in zip(gt_classes, iou_present):
            class_name = DALES_CLASS_NAMES[cls] if cls < len(DALES_CLASS_NAMES) else f"Unknown({cls})"
            f.write(f"  Class {cls} ({class_name}): {iou:.4f}\n")
        f.write("\n")
        
        f.write("="*80 + "\n")
        f.write("FINAL SUMMARY\n")
        f.write("="*80 + "\n")
        f.write(f"Overall Accuracy (All Classes):           {oa:.4f} ({oa*100:.2f}%)\n")
        f.write(f"Overall Accuracy (Present Classes Only):  {oa_present:.4f} ({oa_present*100:.2f}%)\n")
        f.write(f"Mean IoU (All Classes):                   {miou:.4f}\n")
        f.write(f"Mean IoU (Present Classes Only):          {miou_present:.4f}\n")
        f.write(f"Number of classes in ground truth:        {len(gt_classes)}\n")
        f.write(f"Number of classes in predictions:         {len(unique_classes)}\n")
        f.write("="*80 + "\n")
    
    print(f"Results successfully written to {output_file}")

if __name__ == "__main__":
    main()
