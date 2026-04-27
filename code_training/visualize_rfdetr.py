"""
RF-DETR Confusion Matrix Generator v2
Diubah untuk 4 kelas: ball, player, referee, goalkeeper
Fixed untuk Roboflow _annotations.coco.json format
"""
import os
import json
import torch
import cv2
import numpy as np
from collections import defaultdict
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix, classification_report
from tqdm import tqdm

# ========================================
# CONFIG
# ========================================
DATASET_DIR = "football-dataset2-rfdetr600"
OUTPUT_DIR = "runs/rfdetr_visualize_pretrained_true"
CHECKPOINT_PATH = "models/RFDETR Result Dataset With Augmentation Version 1/....."
CONFIDENCE_THRESHOLD = 0.01
IOU_THRESHOLD = 0.01

NUM_CLASSES = 4
CLASS_NAMES = ["ball", "player", "referee", "goalkeeper"]

os.makedirs(OUTPUT_DIR, exist_ok=True)

print("="*70)
print("RF-DETR CONFUSION MATRIX GENERATOR")
print("="*70)

# ========================================
# LOAD MODEL
# ========================================
print("\nLoading model...")
from rfdetr import RFDETRBase

model = RFDETRBase(num_classes=NUM_CLASSES, pretrained=True)
print(f"✅ Model initialized")

# ========================================
# LOAD VALIDATION DATA
# ========================================
print("\n" + "="*70)
print("LOADING VALIDATION DATA")
print("="*70)

# Handle both 'val' and 'valid' directories
val_dir = os.path.join(DATASET_DIR, "val")
if not os.path.exists(val_dir):
    val_dir = os.path.join(DATASET_DIR, "valid")
    print(f"Using 'valid' directory: {val_dir}")
else:
    print(f"Using 'val' directory: {val_dir}")

images_dir = os.path.join(val_dir, "images")
if not os.path.exists(images_dir):
    images_dir = val_dir
    print(f"Images are in root of {val_dir}")

# Find annotation file
annotation_paths = [
    os.path.join(val_dir, "_annotations.coco.json"),  # Roboflow format
    os.path.join(val_dir, "annotations.coco.json"),
    os.path.join(val_dir, "instances_val.json"),
]

annotations_file = None
for path in annotation_paths:
    if os.path.exists(path):
        annotations_file = path
        print(f"✅ Found annotations: {os.path.basename(path)}")
        break

if not annotations_file:
    print(f"❌ No annotation file found in {val_dir}")
    print(f"Files in directory:")
    for f in os.listdir(val_dir)[:10]:
        print(f"  - {f}")
    raise FileNotFoundError("Annotation file not found!")

# Load COCO data
print(f"Loading from: {annotations_file}")
with open(annotations_file, 'r') as f:
    coco_data = json.load(f)

print(f"✅ Loaded {len(coco_data['images'])} validation images")

# ========================================
# CREATE CATEGORY MAPPING
# ========================================
print("\nCategories found in annotations:")
id_to_class = {}
id_to_name = {}

# First pass: identify all categories
print("Available categories:")
for cat in coco_data['categories']:
    print(f"  ID {cat['id']}: {cat['name']}")

print("\nMapping to classes:")
for cat in coco_data['categories']:
    cat_id = cat['id']
    cat_name = cat['name'].lower()
    id_to_name[cat_id] = cat['name']

    # Custom mapping untuk 4 kelas
    if 'ball' in cat_name or 'football' in cat_name:
        id_to_class[cat_id] = 0  # ball
        print(f"  [{cat_id}] {cat['name']:20s} -> ball")
    elif 'player' in cat_name:
        id_to_class[cat_id] = 1  # player
        print(f"  [{cat_id}] {cat['name']:20s} -> player")
    elif 'referee' in cat_name:
        id_to_class[cat_id] = 2  # referee
        print(f"  [{cat_id}] {cat['name']:20s} -> referee")
    elif 'goalkeeper' in cat_name:
        id_to_class[cat_id] = 3  # goalkeeper
        print(f"  [{cat_id}] {cat['name']:20s} -> goalkeeper")
    else:
        # Unknown: assign to player by default
        id_to_class[cat_id] = 1
        print(f"  [{cat_id}] {cat['name']:20s} -> player (default)")

# ========================================
# PARSE GROUND TRUTH
# ========================================
print("\nParsing ground truth...")
gt_boxes_by_image = defaultdict(list)

for ann in coco_data['annotations']:
    img_id = ann['image_id']
    x, y, w, h = ann['bbox']
    bbox = np.array([x, y, x+w, y+h], dtype=np.float32)
    cat_id = ann['category_id']
    class_id = id_to_class.get(cat_id, 0)
    gt_boxes_by_image[img_id].append({
        'bbox': bbox,
        'class_id': class_id,
        'class_name': id_to_name.get(cat_id, 'unknown')
    })

# Count GT per class
print("Ground Truth Distribution:")
for idx, class_name in enumerate(CLASS_NAMES):
    count = sum(len([b for b in boxes if b['class_id'] == idx])
                for boxes in gt_boxes_by_image.values())
    print(f"  {class_name:15s}: {count:4d} instances")

# ========================================
# RUN INFERENCE
# ========================================
print("\n" + "="*70)
print("RUNNING INFERENCE")
print("="*70)

pred_boxes_by_image = defaultdict(list)
error_count = 0

for img_info in tqdm(coco_data['images'], desc="Inferencing"):
    img_id = img_info['id']
    img_filename = img_info['file_name']
    img_path = os.path.join(images_dir, img_filename)

    if not os.path.exists(img_path):
        error_count += 1
        continue

    try:
        with torch.no_grad():
            detections = model.predict(img_path, confidence=CONFIDENCE_THRESHOLD)

        if detections is not None and len(detections) > 0:
            for box, conf, cls_id in zip(
                detections.xyxy,
                detections.confidence,
                detections.class_id
            ):
                cls_id_int = int(cls_id)
                # Clamp to valid range [0, NUM_CLASSES-1]
                if cls_id_int >= NUM_CLASSES:
                    cls_id_int = cls_id_int % NUM_CLASSES

                pred_boxes_by_image[img_id].append({
                    'bbox': box.astype(np.float32),
                    'class_id': cls_id_int,
                    'class_name': CLASS_NAMES[cls_id_int],
                    'confidence': float(conf)
                })

    except Exception as e:
        error_count += 1
        continue

print(f"✅ Inference complete ({error_count} errors)")

# ========================================
# IoU CALCULATION
# ========================================
def box_iou(box1, box2):
    """Calculate IoU between two boxes [x1, y1, x2, y2]"""
    x1_min, y1_min, x1_max, y1_max = box1
    x2_min, y2_min, x2_max, y2_max = box2

    inter_x_min = max(x1_min, x2_min)
    inter_y_min = max(y1_min, y2_min)
    inter_x_max = min(x1_max, x2_max)
    inter_y_max = min(y1_max, y2_max)

    if inter_x_max <= inter_x_min or inter_y_max <= inter_y_min:
        return 0.0

    inter_area = (inter_x_max - inter_x_min) * (inter_y_max - inter_y_min)
    box1_area = (x1_max - x1_min) * (y1_max - y1_min)
    box2_area = (x2_max - x2_min) * (y2_max - y2_min)

    union_area = box1_area + box2_area - inter_area
    return inter_area / union_area if union_area > 0 else 0

# ========================================
# MATCH PREDICTIONS TO GT
# ========================================
print("\n" + "="*70)
print("MATCHING PREDICTIONS TO GROUND TRUTH (IoU >= {:.1%})".format(IOU_THRESHOLD))
print("="*70)

true_labels = []
pred_labels = []
tp_count = 0
misclass_count = 0
fp_count = 0
fn_count = 0

for img_id in gt_boxes_by_image.keys():
    gt_list = gt_boxes_by_image[img_id]
    pred_list = pred_boxes_by_image.get(img_id, [])

    matched_gt = set()

    # Match predictions to GT
    for pred in pred_list:
        best_iou = 0
        best_gt_idx = -1

        for gt_idx, gt in enumerate(gt_list):
            if gt_idx in matched_gt:
                continue

            iou = box_iou(pred['bbox'], gt['bbox'])
            if iou > best_iou:
                best_iou = iou
                best_gt_idx = gt_idx

        if best_iou >= IOU_THRESHOLD and best_gt_idx >= 0:
            gt = gt_list[best_gt_idx]
            true_labels.append(gt['class_id'])
            pred_labels.append(pred['class_id'])
            matched_gt.add(best_gt_idx)

            if gt['class_id'] == pred['class_id']:
                tp_count += 1
            else:
                misclass_count += 1
        else:
            fp_count += 1

    # Unmatched GT = FN
    for gt_idx, gt in enumerate(gt_list):
        if gt_idx not in matched_gt:
            fn_count += 1

print(f"\nMatching Statistics:")
print(f"  True Positives (correct):  {tp_count}")
print(f"  Misclassifications:        {misclass_count}")
print(f"  False Positives:           {fp_count}")
print(f"  False Negatives:           {fn_count}")
print(f"  Total matched detections:  {len(true_labels)}")

# ========================================
# BUILD CONFUSION MATRIX
# ========================================
print("\n" + "="*70)
print("CONFUSION MATRIX")
print("="*70)

cm = confusion_matrix(
    true_labels,
    pred_labels,
    labels=list(range(NUM_CLASSES))
)

print("\nRaw Confusion Matrix:")
print(cm)
print()

# ========================================
# VISUALIZATION
# ========================================
fig, axes = plt.subplots(1, 2, figsize=(16, 6))

# Raw counts
sns.heatmap(
    cm,
    annot=True,
    fmt='d',
    cmap='Blues',
    ax=axes[0],
    xticklabels=CLASS_NAMES,
    yticklabels=CLASS_NAMES,
    cbar_kws={'label': 'Count'},
    vmin=0
)
axes[0].set_title('RF-DETR Confusion Matrix (Counts)', fontsize=14, fontweight='bold')
axes[0].set_ylabel('True Label', fontsize=12)
axes[0].set_xlabel('Predicted Label', fontsize=12)

# Normalized
cm_norm = cm.astype('float') / (cm.sum(axis=1, keepdims=True) + 1e-8)
sns.heatmap(
    cm_norm,
    annot=True,
    fmt='.1%',
    cmap='RdYlGn',
    ax=axes[1],
    xticklabels=CLASS_NAMES,
    yticklabels=CLASS_NAMES,
    vmin=0,
    vmax=1,
    cbar_kws={'label': 'Percentage'}
)
axes[1].set_title('RF-DETR Confusion Matrix (Normalized)', fontsize=14, fontweight='bold')
axes[1].set_ylabel('True Label', fontsize=12)
axes[1].set_xlabel('Predicted Label', fontsize=12)

plt.tight_layout()
output_path = os.path.join(OUTPUT_DIR, 'confusion_matrix.png')
plt.savefig(output_path, dpi=300, bbox_inches='tight')
print(f"✅ Saved: confusion_matrix.png")
plt.show()

# ========================================
# CLASSIFICATION REPORT
# ========================================
print("\n" + "="*70)
print("CLASSIFICATION METRICS")
print("="*70 + "\n")

report = classification_report(
    true_labels,
    pred_labels,
    labels=list(range(NUM_CLASSES)),
    target_names=CLASS_NAMES,
    digits=4,
    zero_division=0
)
print(report)

# ========================================
# SAVE RESULTS
# ========================================
report_path = os.path.join(OUTPUT_DIR, 'classification_report.txt')
with open(report_path, 'w') as f:
    f.write("RF-DETR Object Detection - Classification Report\n")
    f.write("="*70 + "\n")
    f.write(f"Confidence Threshold: {CONFIDENCE_THRESHOLD}\n")
    f.write(f"IoU Threshold (for matching): {IOU_THRESHOLD}\n")
    f.write(f"Number of Classes: {NUM_CLASSES}\n")
    f.write(f"Validation Images: {len(coco_data['images'])}\n")
    f.write("="*70 + "\n\n")

    f.write("Detection Statistics:\n")
    f.write(f"  True Positives (correct):  {tp_count}\n")
    f.write(f"  Misclassifications:       {misclass_count}\n")
    f.write(f"  False Positives:          {fp_count}\n")
    f.write(f"  False Negatives:          {fn_count}\n")
    f.write("="*70 + "\n\n")

    f.write("Confusion Matrix:\n")
    f.write(str(cm) + "\n\n")

    f.write("Classification Report:\n")
    f.write(report)

print(f"✅ Saved: classification_report.txt")

# ========================================
# PER-CLASS SUMMARY
# ========================================
print("\n" + "="*70)
print("PER-CLASS SUMMARY")
print("="*70)

for idx, class_name in enumerate(CLASS_NAMES):
    row = cm[idx]
    total = row.sum()
    correct = row[idx]
    accuracy = correct / total if total > 0 else 0

    print(f"\n{class_name.upper()}:")
    print(f"  Total GT instances:  {total}")
    print(f"  Correct predictions: {correct} ({accuracy:.1%})")
    for other_idx in range(NUM_CLASSES):
        if other_idx != idx and row[other_idx] > 0:
            print(f"  Confused as {CLASS_NAMES[other_idx]}: {row[other_idx]}")

print("\n" + "="*70)
print("✅ EVALUATION COMPLETE!")
print("="*70)
print(f"\nResults saved to: {OUTPUT_DIR}")